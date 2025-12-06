from __future__ import annotations
from typing import Optional, Dict, Tuple, Any, List
import os
import time
import random
import re
import json

# pip install google-generativeai
import google.generativeai as genai


class FreeLLMPreferenceClient:
    """
    Minimal, robust Gemini client compatible with your previous interface:
      - call_oracle(prompt, sched_a, sched_b, ...)
      - generate_response(prompt, ...)

    Key choices:
    - No server-side stop sequences or JSON mode (we trim client-side).
    - Streaming first (captures partial text even if server stops early).
    - Simple fallback if response is empty or capped.
    - Optional random seed: sent when supported; auto-disabled if SDK rejects it.
    """

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
        simple: bool = False,
        rate_limit_delay: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 4,
        default_seed: Optional[int] = 12345,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable (or api_key) is required")

        self.model_name = model_name
        self.model_id = model_name
        self.simple = simple
        self.rate_limit_delay = rate_limit_delay
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.default_seed = default_seed

        # Defaults
        self.default_temperature = 0.0
        self.default_top_p = 1.0
        self.default_max_new_tokens = max_tokens

        # SDK init
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        # Parse helpers
        self._FINAL_TAG_RE = re.compile(
            r'^\s*(?:FINAL)\s*[:=\-]?\s*([AB])\b',
            re.IGNORECASE | re.MULTILINE
        )

        # Seed support flag (auto-detected)
        self._seed_supported = True

    # -------------------------
    # Utilities
    # -------------------------

    def _sleep_backoff(self, attempt: int):
        base_delay = min(2 ** (attempt - 1), 8)
        jitter = random.random() * 0.25
        time.sleep(base_delay + jitter + self.rate_limit_delay)

    def _finish_reason_str(self, cand) -> str:
        fr = getattr(cand, "finish_reason", None)
        name = getattr(fr, "name", None)
        if name:
            return name
        try:
            return str(fr)
        except Exception:
            return repr(fr)

    def _extract_text_from_response(self, response) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (text, meta). Extracts both text parts and inline_data (e.g., JSON bytes/strings).
        """
        meta: Dict[str, Any] = {}
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            pf = response.prompt_feedback
            meta["prompt_feedback"] = getattr(pf, "block_reason", None) or str(pf)

        text_chunks: List[str] = []
        candidates = getattr(response, "candidates", None) or []

        if candidates:
            cand0 = candidates[0]
            meta["finish_reason"] = self._finish_reason_str(cand0)
            if hasattr(cand0, "safety_ratings") and cand0.safety_ratings:
                try:
                    meta["safety_ratings"] = [
                        getattr(r, "category", None) or str(r) for r in cand0.safety_ratings
                    ]
                except Exception:
                    meta["safety_ratings"] = [str(cand0.safety_ratings)]

        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for p in parts:
                # text part
                t = getattr(p, "text", None)
                if t:
                    text_chunks.append(t)
                    continue
                # inline data (JSON/string/bytes)
                inline = getattr(p, "inline_data", None)
                if inline:
                    data = getattr(inline, "data", None)
                    if isinstance(data, (bytes, bytearray)):
                        try:
                            text_chunks.append(data.decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
                    elif isinstance(data, str):
                        text_chunks.append(data)

        return "".join(text_chunks).strip(), meta

    @staticmethod
    def _apply_stop(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        out = text
        for s in stop:
            if not s:
                continue
            i = out.find(s)
            if i != -1:
                out = out[:i]
        return out

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
            elif last_b > last_a:
                return "B"
            if last_a != -1:
                return "A"
            if last_b != -1:
                return "B"

        lone = re.findall(r'^\s*([AB])\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if lone:
            return lone[-1].upper()

        return None

    # -------------------------
    # Core request (simple & efficient)
    # -------------------------

    def _post_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
        stop: Optional[List[str]],
    ) -> str:
        """
        Simple flow:
          1) Build a single prompt.
          2) Streaming attempt (captures partial tokens).
          3) Non-streaming attempt.
          4) Short fallback (lower cap + stricter guard).
        Only client-side stop trimming. Optional seed is sent if supported.
        """

        # 1) Build prompt
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}\n")
            else:
                parts.append(f"{content}\n")
        base_prompt = "\n".join(parts).strip()

        wants_json = ("<END_JSON>" in base_prompt) or ("STRICT OUTPUT RULES" in base_prompt)
        guard = (
            "Reply with ONLY the JSON object. Keep it under 250 tokens. "
            "No code fences. End with <END_JSON>.\n\n" if wants_json
            else "Be concise. Keep output under 300 tokens.\n\n"
        )
        prompt = guard + base_prompt

        # 2) Generation config (no server stops, no JSON mode)
        gen_cfg: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_tokens),
        }

        # Helper to call API (streaming or not) with optional seed
        def _call(stream: bool, prompt_text: str, cap: int) -> Tuple[str, Dict[str, Any]]:
            cfg = dict(gen_cfg)
            cfg["max_output_tokens"] = int(cap)
            # Try seed if we believe it's supported
            try_seed = (seed is not None) and self._seed_supported
            try:
                if stream:
                    # streaming path
                    if try_seed:
                        try:
                            resp_stream = self.model.generate_content(
                                prompt_text, generation_config={**cfg, "seed": int(seed)}, stream=True, safety_settings=None
                            )
                        except Exception as e:
                            # If seed unsupported, remember & retry without it
                            if "unknown field" in str(e).lower() and "seed" in str(e).lower():
                                self._seed_supported = False
                                resp_stream = self.model.generate_content(
                                    prompt_text, generation_config=cfg, stream=True, safety_settings=None
                                )
                            else:
                                raise
                    else:
                        resp_stream = self.model.generate_content(
                            prompt_text, generation_config=cfg, stream=True, safety_settings=None
                        )
                    chunks: List[str] = []
                    for ev in resp_stream:
                        t = getattr(ev, "text", None)
                        if t:
                            chunks.append(t)
                    return "".join(chunks).strip(), {}
                else:
                    # non-streaming path
                    if try_seed:
                        try:
                            resp = self.model.generate_content(
                                prompt_text, generation_config={**cfg, "seed": int(seed)}, safety_settings=None
                            )
                        except Exception as e:
                            if "unknown field" in str(e).lower() and "seed" in str(e).lower():
                                self._seed_supported = False
                                resp = self.model.generate_content(
                                    prompt_text, generation_config=cfg, safety_settings=None
                                )
                            else:
                                raise
                    else:
                        resp = self.model.generate_content(
                            prompt_text, generation_config=cfg, safety_settings=None
                        )
                    text, meta = self._extract_text_from_response(resp)
                    return text, meta
            except Exception as e:
                # Surface meta via exception message to keep this simple
                raise e

        # Attempts (simple & bounded)
        # A) Streaming
        try:
            text, _ = _call(stream=True, prompt_text=prompt, cap=max_tokens)
            print("GEMINI TEXT , ", text)
            if text:
                return self._apply_stop(text, stop)
        except Exception:
            pass  # fall through

        # B) Non-streaming
        try:
            text, meta = _call(stream=False, prompt_text=prompt, cap=max_tokens)
            print("GEMINI TEXT , ", text)
            if text:
                return self._apply_stop(text, stop)
            last_meta = meta
        except Exception as e:
            last_meta = {"error": str(e)}

        # C) Short fallback (smaller cap + stricter guard)
        tight_prompt = (
            ("Reply with ONLY the JSON object. Keep it under 180 tokens. "
             "No code fences. End with <END_JSON>.\n\n") if wants_json
            else "Be concise. Keep output under 180 tokens.\n\n"
        ) + base_prompt

        small_cap = min(int(max_tokens), 512)

        try:
            # Prefer streaming even for fallback (captures partials)
            text, _ = _call(stream=True, prompt_text=tight_prompt, cap=small_cap)
            print("GEMINI TEXT , ", text)
            if text:
                return self._apply_stop(text, stop)
        except Exception:
            pass

        text2, meta2 = _call(stream=False, prompt_text=tight_prompt, cap=small_cap)
        print("GEMINI TEXT , ", text2)
        if text2:
            return self._apply_stop(text2, stop)

        raise RuntimeError(
            f"Gemini API request failed: no text returned. "
            f"meta1={json.dumps(last_meta, default=str)} meta2={json.dumps(meta2, default=str)}"
        )

    # -------------------------
    # Public API
    # -------------------------

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
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        seed = self.default_seed if seed is None else seed

        messages = [{"role": "user", "content": prompt}]
        text = self._post_chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
        )
        text = self._apply_stop(text.strip(), stop)

        choice = self._parse_final_choice(text)
        if choice is None:
            choice = text[0] if text and text[0] in ("A", "B") else "A"
        return choice, text

    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        seed = self.default_seed if seed is None else seed

        messages = [{"role": "user", "content": prompt}]
        text = self._post_chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
        )
        return self._apply_stop(text.strip(), stop)


# Global client instance for backwards compatibility
_CLIENT = None


def get_local_client(
    model_id: str = "gemini-1.5-pro",
    *,
    force_full_precision: bool = None,  # compatibility, unused
):
    """
    Backwards-compatible factory function.
    Returns a FreeLLMPreferenceClient configured for Gemini API.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required")

    model_name = os.getenv("GEMINI_MODEL_NAME", model_id)
    print(f"[gemini_client] Using Gemini API with model: {model_name}")

    _CLIENT = FreeLLMPreferenceClient(
        provider="gemini",
        api_key=api_key,
        model_name=model_name,
        simple=False,
        rate_limit_delay=0.1,
        max_tokens=4096,
        max_retries=4,
    )
    return _CLIENT


if __name__ == "__main__":
    client = get_local_client()

    # Test generate_response
    try:
        response = client.generate_response(
            "What is 2+2?",
            temperature=0.0,
            max_new_tokens=64,
            seed=42,  # seed will be used if supported by your SDK
        )
        print(f"Generate response test: {response}")
    except Exception as e:
        print("Generate response test failed:", e)

    # Test call_oracle
    try:
        choice, text = client.call_oracle(
            "Which option is better? A or B? Answer with FINAL: A or FINAL: B",
            sched_a=None,
            sched_b=None,
            temperature=0.0,
            max_new_tokens=64,
            seed=42,
        )
        print(f"Oracle test - Choice: {choice}, Text: {text}")
    except Exception as e:
        print("Oracle test failed:", e)
