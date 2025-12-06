from abc import  abstractmethod
from typing import Any, Dict, List

from promptTemplate import PromptTemplateInterface


def _excel_column_label(n: int) -> str:
    """
    Generate Excel-style column labels: A, B, ..., Z, AA, AB, ..., AZ, BA, ...
    
    Args:
        n: 0-based index
        
    Returns:
        Label string (e.g., 0 -> 'A', 25 -> 'Z', 26 -> 'AA')
    """
    label = ""
    n += 1  # Convert to 1-based for Excel-style
    while n > 0:
        n -= 1  # Adjust for 0-based modulo
        label = chr(65 + (n % 26)) + label
        n //= 26
    return label


class ComparisonPromptTemplate(PromptTemplateInterface):
    """
    Base class for prompt templates that compare multiple items.
    This version renders items in a single JSON block to prevent number-copy errors.
    Subclasses must still implement format() and get_base_prompt().
    """

    MAX_HISTORY = 5  # Default max history items #TODO: make this a parameter

    def __init__(self, reasoning: bool = True, reasoning_history: bool = False, **kwargs):
        """
        Initialize comparison prompt template.

        Args:
            reasoning: Include reasoning in output
            reasoning_history: Include history (previous decisions) in the prompt
            **kwargs: Additional parameters
        """
        super().__init__(reasoning=reasoning)
        self.reasoning_history = reasoning_history

    @abstractmethod
    def format(self, *args, **kwargs) -> str:
        """Format the main prompt. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_base_prompt(self) -> str:
        """Get the base prompt. Must be implemented by subclasses."""
        pass
        # ---------- NEW: JSON-based comparison formatting ----------
    def format_comparison(
            self,
            items: List[Dict[str, Any]],
            item_formatter: callable = None,  # kept for compatibility; ignored in JSON mode
            option_labels: List[str] = None,
            metric_list: List[str] = None,
        ) -> str:
        """
        Render the comparison as JSON so the model reads numbers reliably.

        Args:
            items: list of schedule/item dicts (each contains metric -> value)
            item_formatter: (ignored) kept for API compatibility
            option_labels: optional labels to use for each item; defaults to A, B, C, ..., Z, AA, AB, ...
            metric_list: optional explicit key order for metrics in each option
        """
        if not items:
            raise ValueError("At least one item is required for comparison.")

        # Labels: A, B, C, ..., Z, AA, AB, AC, ... (Excel-style)
        n = len(items)
        if option_labels is None:
            option_labels = [_excel_column_label(i) for i in range(n)]
        if len(option_labels) != n:
            raise ValueError("option_labels length must match items length.")

        # Prepare JSON payload
        options_json = []
        for label, item in zip(option_labels, items):
            # Only include metrics specified in metric_list
            filtered_item = {k: item[k] for k in metric_list if k in item} if metric_list else dict(item)

            # Wrap each option with a label and its metrics
            options_json.append({
                "label": label,
                "metrics": filtered_item
            })

        # Optional HISTORY block (JSON)
        history_block = ""
        if self.reasoning_history and getattr(self, "history", None):
            hist = []
            for entry in reversed(self.history[-self.MAX_HISTORY:]):
                hist.append({
                    "choice": entry.get("choice"),
                    # keep rationale short if present
                    "reasoning": entry.get("reasoning", None)
                })
            history_block = (
                "=== PREVIOUS DECISIONS (JSON) ===\n"
                "```json\n" + self._to_minified_json({"history": hist}) + "\n```\n\n"
                "=== CURRENT DECISION ===\n\n"
            )

        # Base prompt + JSON ITEMS block
        prompt_parts = []
        prompt_parts.append(self.get_base_prompt())

        if history_block:
            prompt_parts.append(history_block)

        prompt_parts.append("=== ITEMS TO COMPARE (JSON) ===\n")
        prompt_parts.append("Return EXACTLY these numbers when reasoning; do not infer or re-calc.\n")
        prompt_parts.append("```json\n")
        prompt_parts.append(self._to_minified_json({"options": options_json}))
        prompt_parts.append("\n```\n")

        # Clear, strict instructions & FINAL tag menu
        prompt_parts.append(self._format_instructions(n))

        return "".join(prompt_parts)

    # ---------- Helpers ----------
    @staticmethod
    def _to_minified_json(obj: Any) -> str:
        import json
        # stable keys for determinism; no extra spaces
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

    def _format_instructions(self, num_options: int) -> str:
        """Instructions that reference the JSON block above."""
        # Generate Excel-style labels for all options
        valid_choices = [_excel_column_label(i) for i in range(num_options)]

        if self.reasoning:
            parts = [
                "\nDecide using ONLY the JSON metrics above. "
                "Briefly explain your reasoning using those exact numbers. "
                "On the last line, output only one final tag in this exact format:\n"
            ]
        else:
            parts = [
                "\nDo not explain. Using ONLY the JSON metrics above, "
                "output one final line in this exact format:\n"
            ]

        for i, choice in enumerate(valid_choices):
            parts.append("FINAL: " + choice + ("\n" if i < len(valid_choices) - 1 else ""))

        return "".join(parts)

    # (Kept for compatibility; no longer used to render items)
    def _default_item_formatter(self, item: Dict[str, Any]) -> str:
        import json
        return json.dumps(item, ensure_ascii=False, indent=2)
        # ---------- Robust FINAL tag parsing ----------
    def _parse_final_tag(self, response: str, valid_choices: list) -> str:
        import re
        response = response.strip()
        print(f"Raw response:\n{response}\n")
        
        # Pattern to match multi-letter labels like AA, AB, etc. or single letters
        patterns = [
            r'FINAL\s*[:=]\s*([A-Z]+)\b',
            r'Final\s*[:=]\s*([A-Z]+)\b',
            r'final\s*[:=]\s*([A-Z]+)\b',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                choice = matches[-1].upper()
                if choice in valid_choices:
                    return choice

        loose_patterns = [
            r'FINAL\s+([A-Z]+)\b',
            r'Final\s+([A-Z]+)\b',
            r'final\s+([A-Z]+)\b',
        ]
        for pattern in loose_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                choice = matches[-1].upper()
                if choice in valid_choices:
                    return choice

        answer_patterns = [
            r'(?:answer|choice|select|choose|pick|best|option)\s+(?:is|would be|:)?\s*([A-Z]+)\b',
            r'Option\s+([A-Z]+)\s+is\s+(?:the\s+)?best',
        ]
        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                choice = matches[-1].upper()
                if choice in valid_choices:
                    return choice

        lines = response.split('\n')
        import re as _re
        for line in reversed(lines):
            s = line.strip()
            if s.upper() in valid_choices:
                return s.upper()
            for patt in [r'^(?:Option|Choice|Answer)?\s*([A-Z]+)$']:
                m = _re.match(patt, s, _re.IGNORECASE)
                if m:
                    c = m.group(1).upper()
                    if c in valid_choices:
                        return c

        # Search for any valid choice in the last part of the response
        last_chunk = response[-100:] if len(response) > 100 else response
        for c in valid_choices:
            patt = rf'(?<![A-Za-z]){re.escape(c)}(?![A-Za-z])'
            if _re.search(patt, last_chunk, _re.IGNORECASE):
                matches = _re.finditer(patt, last_chunk, _re.IGNORECASE)
                for m in matches:
                    start = max(0, m.start() - 10)
                    end = min(len(last_chunk), m.end() + 10)
                    context = last_chunk[start:end].lower()
                    if any(w in context for w in ['final', 'answer', 'choice', 'best', 'select', 'option']):
                        return c

        print("COULD NOT PARSE")
        return valid_choices[0] if valid_choices else "A"

    def parse_response(self, response: str, num_options: int = None) -> str:
        """
        Parse response to extract choice.
        
        Args:
            response: The LLM response text
            num_options: Number of options (if known, to validate choices)
        """
        # Generate valid choices based on num_options if provided
        if num_options:
            valid_choices = [_excel_column_label(i) for i in range(num_options)]
        else:
            # Default to extended range (A-Z, AA-AZ, BA-BZ, etc., up to ~700 options)
            valid_choices = [_excel_column_label(i) for i in range(702)]  # A-ZZ
        return self._parse_final_tag(response, valid_choices)


class ComparisonPromptAdapter(ComparisonPromptTemplate):
    """
    A generic prompt adapter for the comparison algorithm.
    It is configured with a fully-formed base prompt, making it adaptable to any scenario.
    """
    def __init__(self,
                 base_prompt: str,
                 reasoning: bool = True,
                 reasoning_history: bool = False,
                 metric_columns: list = None):
        super().__init__(reasoning=reasoning, reasoning_history=reasoning_history)
        self.base_prompt = base_prompt.strip()
        self.metric_columns = list(metric_columns or [])

    def get_base_prompt(self) -> str:
        """Returns the configured base prompt."""
        return self.base_prompt

    def format(self, items: list) -> str:
        """Formats the complete comparison prompt with a JSON block of items."""
        if not items:
            raise ValueError("At least one item must be provided for comparison.")
        return self.format_comparison(
            items,
            metric_list=self.metric_columns
        )