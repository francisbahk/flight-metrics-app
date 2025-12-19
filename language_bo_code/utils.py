import json
import re
from json import JSONDecodeError

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid function.

    Args:
        x: Input array.

    Returns:
        Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-x))


def escape_unescaped_backslashes(s: str) -> str:
    """
    This regex finds backslashes that are not already escaped
    and are not followed by a valid JSON escape character
    """
    s = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", s)
    return s


def extract_json_from_text(text: str) -> dict | None:
    """
    Extracts the first JSON object from a text code block labeled as 'json'.
    Returns the parsed Python dict, or None if not found or invalid.

    Args:
        text: The text to search for the JSON code block.

    Returns:
        The parsed Python dict, or None if not found or invalid.
    """
    # Regex to match ```json ... ```
    pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    for match in matches:
        try:
            # Remove leading/trailing whitespace
            json_str = match.strip()
            json_str = escape_unescaped_backslashes(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    return None


def extract_jsonl_from_text(text: str) -> list | None:
    """
    Extracts the first JSONL object from a text code block labeled as 'jsonl'.
    Returns the parsed Python list of dicts, or None if not found or invalid.

    Args:
        text: The text to search for the JSONL code block.

    Returns:
        The parsed Python list of dicts, or None if not found or invalid.
    """
    pattern = r"```jsonl\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        pattern = r"```jsonl\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            pattern = r"```\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                return None
    jsonl_str = matches[-1]
    jsonl_str = escape_unescaped_backslashes(jsonl_str)
    jsonl_str = jsonl_str.strip("[\n").strip("]")
    lines = []
    if "},\n" in jsonl_str:
        jsonl_str_ls = jsonl_str.split("},\n")
    else:
        jsonl_str_ls = jsonl_str.split("}\n")

    for json_str in jsonl_str_ls:
        try:
            json_obj = json.loads(json_str + "}")
            lines.append(json_obj)
        except JSONDecodeError:
            try:
                json_obj = json.loads(json_str)
                lines.append(json_obj)
            except JSONDecodeError:
                return None

    return lines
