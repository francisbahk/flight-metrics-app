"""
Gemini API client for LISTEN-U preference comparisons.
Adapted from LISTEN's FreeLLMPreferenceClient to use Google Gemini instead of Groq.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Any, List
import os
import time
import random
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class GeminiLLMPreferenceClient:
    """
    Client for Google Gemini API that mimics LISTEN's LLM client interface.
    Used for preference comparisons in dueling bandit optimization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        simple: bool = False,
        rate_limit_delay: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 10,
        default_seed: Optional[int] = 12345,
    ):
        """
        Initialize the Gemini API client.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            model_name: Model to use (default: gemini-2.0-flash)
            simple: Whether to use simple mode
            rate_limit_delay: Delay between requests to avoid rate limits
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retries for failed requests
            default_seed: Default seed for reproducibility
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

        # Configure Gemini
        genai.configure(api_key=self.api_key)

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

        # Initialize Gemini model
        self.model = genai.GenerativeModel(model_name)

        # Regex patterns for parsing responses (same as Groq client)
        self._FINAL_TAG_RE = re.compile(
            r'^\s*(?:```[a-zA-Z]*\s*)?FINAL\s*[:=\-]?\s*([AB])\b',
            re.IGNORECASE | re.MULTILINE
        )

    def _sleep_backoff(self, attempt: int):
        """Exponential backoff with jitter."""
        base_delay = min(2 ** (attempt - 1), 8)
        jitter = random.random() * 0.25
        total_delay = base_delay + jitter + self.rate_limit_delay
        time.sleep(total_delay)

    def _generate_content(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        """
        Send a generation request to Gemini API.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text response
        """
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Configure generation parameters
                generation_config = genai.GenerationConfig(
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_output_tokens=int(max_tokens),
                )

                # Make the API call
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                # Extract and return the content
                return response.text

            except Exception as e:
                last_err = e
                error_msg = str(e).lower()

                # Check for rate limit errors
                if "rate" in error_msg or "429" in error_msg or "quota" in error_msg:
                    print(f"  Rate limit hit, backing off (attempt {attempt}/{self.max_retries})...")
                    self._sleep_backoff(attempt)
                    continue

                # Check for server errors
                if any(code in error_msg for code in ["500", "502", "503", "504"]):
                    print(f"  Server error, retrying (attempt {attempt}/{self.max_retries})...")
                    self._sleep_backoff(attempt)
                    continue

                # For other errors, retry with backoff
                if attempt < self.max_retries:
                    print(f"  Error: {str(e)[:100]}, retrying (attempt {attempt}/{self.max_retries})...")
                    self._sleep_backoff(attempt)
                    continue

                raise

        raise RuntimeError(f"Gemini API request failed after {self.max_retries} attempts: {last_err}")

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
        sched_a=None,
        sched_b=None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Returns (choice, raw_text) where choice is 'A' or 'B' (best-effort parse).

        Args:
            prompt: The prompt to send to the LLM
            sched_a: Schedule A (included for compatibility)
            sched_b: Schedule B (included for compatibility)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
            stop: Stop sequences
            seed: Random seed for reproducibility

        Returns:
            Tuple of (choice, raw_text) where choice is 'A' or 'B'
        """
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens

        text = self._generate_content(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        text = self._apply_stop(text.strip(), stop)

        choice = self._parse_final_choice(text)
        if choice is None:
            # Absolute last resort: keep the previous heuristic
            # (first char if it's A/B; otherwise default to 'A')
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
        This method is compatible with LISTEN's expectations.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
            stop: Stop sequences
            seed: Random seed for reproducibility

        Returns:
            The raw text response from the model
        """
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens

        text = self._generate_content(
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        # Apply stop sequences if provided
        text = self._apply_stop(text.strip(), stop)

        return text


# For testing
if __name__ == "__main__":
    # Test the client
    client = GeminiLLMPreferenceClient()

    # Test generate_response
    response = client.generate_response(
        "What is 2+2?",
        temperature=0.0,
        max_new_tokens=100
    )
    print(f"Generate response test: {response}")

    # Test call_oracle
    choice, text = client.call_oracle(
        "Which option is better? A or B? Answer with FINAL: A or FINAL: B",
        sched_a=None,
        sched_b=None,
        temperature=0.0,
        max_new_tokens=100
    )
    print(f"Oracle test - Choice: {choice}, Text: {text}")
