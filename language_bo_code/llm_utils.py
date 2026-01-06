import asyncio
import os
import random
import time
from typing import Optional

from google import genai

MAX_RETRY = 5


class LLMClient:
    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-1.5-flash",  # Use stable model by default
    ):
        # Use environment variable if api_key not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        # Map common model names to Gemini models
        model_mapping = {
            "llama3.3-70b-instruct": "gemini-1.5-flash",  # Use stable model
            "llama3.1-405b-instruct": "gemini-1.5-pro",
            "gpt-4": "gemini-1.5-pro",
            "gpt-3.5-turbo": "gemini-1.5-flash",  # Use stable model
            # Removed redirect for gemini-1.5-flash - let it use the actual stable model
        }
        self.model = model_mapping.get(model, model)

        # Configure the new Gemini API client
        self.client = genai.Client(api_key=self.api_key)

    async def get_llm_response(
        self, prompt: str, num_responses: int = 1, kwargs: Optional[dict] = None
    ) -> list:
        """
        Generates LLM responses to a single prompt using Google Gemini API.

        Args:
            prompt: The prompt to send to the LLM.
            num_responses: The number of responses to return.
            kwargs: Additional generation parameters (temperature, top_p, max_tokens, etc.)
        """
        if kwargs is None:
            kwargs = {}

        # Build generation config
        config = {}
        if "max_tokens" in kwargs:
            config["max_output_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            config["top_p"] = kwargs["top_p"]

        responses = []
        for _ in range(num_responses):
            retry_count = 0
            while retry_count < MAX_RETRY:
                try:
                    # Use new google-genai API
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model,
                        contents=prompt,
                        config=config if config else None
                    )
                    responses.append(response.text)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= MAX_RETRY:
                        print(f"Error after {MAX_RETRY} retries: {e}")
                        responses.append("")
                    else:
                        wait_time = 2 ** retry_count + random.random()
                        print(f"Error: {e}. Retrying in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)

        return responses

    async def get_batch_llm_responses(
        self,
        prompts: list,
        num_responses: int = 1,
        kwargs: Optional[dict] = None,
    ) -> list:
        """
        Generates LLM responses to a batch of prompts using Google Gemini API.

        Args:
            prompts: The list of prompts to send to the LLM.
            num_responses: The number of responses to return per prompt.
            kwargs: Additional generation parameters (temperature, top_p, etc.)
        """
        # Process all prompts concurrently
        tasks = [
            self.get_llm_response(prompt, num_responses=num_responses, kwargs=kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
