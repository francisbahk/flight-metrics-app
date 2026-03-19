"""
Prompt quality validation using Groq LLM.
Kept separate from app.py so it can be imported and tested without Streamlit.
"""
import os
import json
from groq import Groq


def validate_prompt_with_groq(prompt: str) -> tuple[bool, str]:
    """Use Groq llama-70b to judge whether the prompt is a detailed flight preference description.
    Returns (is_detailed, feedback_message).
    If GROQ_API_KEY is not set, returns (True, '') — fail open.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return True, ""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a validator for a flight-ranking study. "
                    "Decide whether the participant's description of their flight preferences is sufficiently detailed. "
                    "A good description mentions at least some of: airline preference, departure/arrival time preference, "
                    "price vs speed priority, layover tolerance, or seat class. "
                    "Gibberish, random words, or descriptions that don't mention any actual flight preferences are NOT detailed. "
                    "Reply with ONLY a JSON object, no other text: "
                    "{\"detailed\": true or false, \"feedback\": \"one sentence of specific encouragement if not detailed, else empty string\"}"
                ),
            },
            {"role": "user", "content": f"Participant's description:\n\n{prompt}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    if data.get("detailed"):
        return True, ""
    return False, data.get("feedback", "")
