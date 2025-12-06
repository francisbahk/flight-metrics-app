from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Any
import re

try:  # pragma: no cover - import guard
    from vllm import LLM, SamplingParams
except ImportError as _exc:  # pragma: no cover - fallback handled at runtime
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_IMPORT_ERROR = _exc
else:  # pragma: no cover - executed when vllm available
    _VLLM_IMPORT_ERROR = None


class LocalVLLMPreferenceClient:
    """
    Preference client backed by a local vLLM instance.

    This mirrors the minimal interface expected by the LISTEN experiments:
      - generate_response(prompt, ...)
      - call_oracle(prompt, sched_a, sched_b, ...)
    """

    _FINAL_TAG_RE = re.compile(
        r"^\s*(?:```[a-zA-Z]*\s*)?FINAL\s*[:=\-]?\s*([AB])\b",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        dtype: str | None = "auto",
        default_temperature: float = 0.2,
        default_top_p: float = 0.95,
        default_max_new_tokens: int = 512,
        default_seed: Optional[int] = None,
        default_stop_sequences: Optional[List[str]] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        sampling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if LLM is None or SamplingParams is None:  # pragma: no cover - guard for optional dep
            raise RuntimeError(
                "vllm is required for LocalVLLMPreferenceClient. Install it with `pip install vllm`."
            ) from _VLLM_IMPORT_ERROR

        self.model_id = model_id
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_max_new_tokens = default_max_new_tokens
        self.default_seed = default_seed
        self.default_stop_sequences = list(default_stop_sequences or [])

        self._sampling_kwargs = dict(sampling_kwargs or {})

        llm_extra_kwargs = dict(llm_kwargs or {})

        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            **llm_extra_kwargs,
        )

        # Pre-build and cache the default SamplingParams to avoid per-call construction
        # when the caller does not override any sampling arguments.
        self._cached_sampling_default = self._build_sampling_params(
            temperature=self.default_temperature,
            top_p=self.default_top_p,
            max_new_tokens=self.default_max_new_tokens,
            seed=None,
            stop=None,
        )

    def _build_sampling_params(
        self,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: Optional[int],
        stop: Optional[List[str]],
    ) -> SamplingParams:
        params: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_new_tokens),
        }

        if stop:
            params["stop"] = list(stop)
        elif self.default_stop_sequences:
            params["stop"] = list(self.default_stop_sequences)

        if seed is None:
            if self.default_seed is not None:
                params["seed"] = int(self.default_seed)
        else:
            params["seed"] = int(seed)

        params.update({k: v for k, v in self._sampling_kwargs.items() if v is not None})

        return SamplingParams(**params)

    @staticmethod
    def _apply_stop(text: str, stop_sequences: Optional[List[str]]) -> str:
        if not stop_sequences:
            return text
        result = text
        for token in stop_sequences:
            if not token:
                continue
            idx = result.find(token)
            if idx != -1:
                result = result[:idx]
        return result

    def _parse_final_choice(self, text: str) -> Optional[str]:
        matches = self._FINAL_TAG_RE.findall(text)
        if matches:
            return matches[-1].upper()

        kv = re.findall(r'"?final"?\s*[:=]\s*"?([AB])"?', text, flags=re.IGNORECASE)
        if kv:
            return kv[-1].upper()

        last_a = text.rfind("{A}")
        last_b = text.rfind("{B}")
        if last_a != -1 or last_b != -1:
            if last_a > last_b:
                return "A"
            if last_b > last_a:
                return "B"
            if last_a != -1:
                return "A"
            if last_b != -1:
                return "B"

        lone = re.findall(r'^\s*([AB])\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if lone:
            return lone[-1].upper()

        return None

    def _generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        use_cached_default = (
            temperature is None
            and top_p is None
            and max_new_tokens is None
            and stop is None
            and seed is None
        )

        if use_cached_default:
            sampling = self._cached_sampling_default
        else:
            sampling = self._build_sampling_params(
                temperature=self.default_temperature if temperature is None else temperature,
                top_p=self.default_top_p if top_p is None else top_p,
                max_new_tokens=self.default_max_new_tokens if max_new_tokens is None else max_new_tokens,
                seed=seed,
                stop=stop,
            )

        outputs = self.llm.generate([prompt], sampling)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no generations")

        text = outputs[0].outputs[0].text

        stop_sequences = stop if stop is not None else (
            sampling.stop if hasattr(sampling, "stop") else None
        )

        return self._apply_stop(text.strip(), stop_sequences)

    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        return self._generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop=stop,
            seed=seed,
        )

    def call_oracle(
        self,
        prompt: str,
        sched_a,
        sched_b,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, str]:
        text = self._generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop=stop,
            seed=seed,
        )

        choice = self._parse_final_choice(text)
        if choice is None:
            if text and text[0] in ("A", "B"):
                choice = text[0].upper()
            else:
                choice = "A"

        return choice, text


