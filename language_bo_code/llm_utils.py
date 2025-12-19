import asyncio
import random
import time
from typing import Optional

MAX_RETRY = 5


class LLMClient:
    def __init__(
        self,
        api_key: str = "",
        model: str = "llama3.3-70b-instruct",
    ):
        self.api_key = api_key
        self.model = model

    async def get_llm_response(
        self, prompt: str, num_responses: int = 1, kwargs: Optional[dict] = None
    ) -> list:
        """
        Generates LLM responses to a signle prompt.

        Args:
            prompt: The prompt to send to the LLM.
            num_responses: The number of responses to return.
            kwargs: Additional generation parameters (temperature, top_p, etc.)
        """
        raise NotImplementedError  # "Replace with your own implementation to call an LLM client server"

    async def get_batch_llm_responses(
        self,
        prompts: list,
        num_responses: int = 1,
        kwargs: Optional[dict] = None,
    ) -> list:
        """
        Generates LLM responses to a batch of prompts using genie batch_inference

        Args:
            prompts: The list of prompts to send to the LLM.
            num_responses: The number of responses to return per prompt.
            kwargs: Additional generation parameters (temperature, top_p, etc.)
            max_retries: Number of times to retry on timeout.
            timeout_per_call: Timeout (in seconds) for each batch call.
        """
        raise NotImplementedError  # "Replace with your own implementation to call an LLM client server"
