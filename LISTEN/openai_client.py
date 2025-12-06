"""
OpenAI API client implementation replacing Groq client.
Uses the OpenAI cloud API with gpt-4o model.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Any, List
import os
import time
import random
import re
import json
from openai import OpenAI


class FreeLLMPreferenceClient:
    """
    Client for OpenAI API that mimics the interface of the original remoteOss client.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        simple: bool = False,
        rate_limit_delay: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 2,
        default_seed: Optional[int] = 12345,
    ):
        """
        Initialize the OpenAI API client.
        
        Args:
            provider: API provider (openai)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: Model to use (default: gpt-4o)
            simple: Whether to use simple mode
            rate_limit_delay: Delay between requests to avoid rate limits
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries for failed requests
            default_seed: Default seed for reproducibility
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")
        
        self.model_name = model_name
        self.model_id = model_name  # Alias for compatibility
        self.simple = simple
        self.rate_limit_delay = rate_limit_delay
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.default_seed = default_seed
        
        # Default generation parameters
        self.default_temperature = 0.0
        self.default_top_p = 1.0
        self.default_max_new_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Regex patterns for parsing responses
        self._FINAL_TAG_RE = re.compile(
            r'^\s*(?:```[a-zA-Z]*\s*)?FINAL\s*[:=\-]?\s*([AB])\b',
            re.IGNORECASE | re.MULTILINE
        )
    
    def _sleep_backoff(self, attempt: int):
        """Exponential backoff with jitter."""
        base_delay = min(2 ** (attempt - 1), 8)
        jitter = random.random() * 0.25
        total_delay = base_delay + jitter + self.rate_limit_delay
        time.sleep(1)
    

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
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                )

                # Extract text safely
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_err = e
                error_msg = str(e).lower()
                print('error ' , error_msg)
                if "rate" in error_msg or "429" in error_msg:
                    self._sleep_backoff(attempt)
                    continue
                if any(code in error_msg for code in ["500", "502", "503", "504"]):
                    self._sleep_backoff(attempt)
                    continue
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue

                raise

        raise RuntimeError(f"OpenAI API request failed after {self.max_retries} attempts: {last_err}")
    @staticmethod
    def _apply_stop(text: str, stop: Optional[List[str]]) -> str:
        """Apply stop sequences to truncate text."""
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
        """
        Parse the model's output and return 'A' or 'B' if found, else None.
        Priority:
          1) Explicit "FINAL: A/B" tag (last occurrence wins)
          2) JSON-like {"final": "A"} or "final: B"
          3) Legacy tokens {A}/{B}
          4) Lone 'A' or 'B' on its own line
        """
        # 1) Explicit FINAL: X (prefer the last match)
        matches = self._FINAL_TAG_RE.findall(text)
        if matches:
            return matches[-1].upper()
        
        # 2) JSON-like or key-value "final"
        kv = re.findall(r'"?final"?\s*[:=]\s*"?([AB])"?', text, flags=re.IGNORECASE)
        if kv:
            return kv[-1].upper()
        
        # 3) Legacy brace tokens
        last_a = text.rfind("{A}")
        last_b = text.rfind("{B}")
        if last_a != -1 or last_b != -1:
            if last_a > last_b:
                return "A"
            elif last_b > last_a:
                return "B"
            # if only one exists, return that
            if last_a != -1:
                return "A"
            if last_b != -1:
                return "B"
        
        # 4) Lone A/B on its own line (take the last occurrence)
        lone = re.findall(r'^\s*([AB])\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if lone:
            return lone[-1].upper()
        
        return None
    
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
        """
        Returns (choice, raw_text) where choice is 'A' or 'B' (best-effort parse).
        """
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
            if text and text[0] in ("A", "B"):
                choice = text[0]
            else:
                choice = "A"
        
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
        """
        Generate a response for a given prompt.
        This method is compatible with ScheduleBatchExp's expectations.
        """
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
        
        return text


# Global client instance for backwards compatibility
_CLIENT = None


def get_local_client(
    model_id: str = "gpt-4o",
    *,
    force_full_precision: bool = None,  # kept for compatibility, unused
):
    """
    Backwards-compatible factory function.
    Returns a FreeLLMPreferenceClient configured for OpenAI API.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    model_name = os.getenv("OPENAI_MODEL_NAME", model_id)
    
    print(f"[openai_client] Using OpenAI API with model: {model_name}")
    
    _CLIENT = FreeLLMPreferenceClient(
        provider="openai",
        api_key=api_key,
        model_name=model_name,
        simple=False,
        rate_limit_delay=0.1,
        max_tokens=8192,
    )
    
    return _CLIENT


if __name__ == "__main__":

    client = get_local_client()
    
    response = client.generate_response(
        "What is 2+2?",
        temperature=0.0,
        max_new_tokens=1000
    )
    print(f"Generate response test: {response}")
    
    choice, text = client.call_oracle(
        "Which option is better? A or B? Answer with FINAL: A or FINAL: B",
        sched_a=None,
        sched_b=None,
        temperature=0.0,
        max_new_tokens=1000
    )
    print(f"Oracle test - Choice: {choice}, Text: {text}")
