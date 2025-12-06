from __future__ import annotations
from typing import Dict, Any

class PromptManager:
    """
    Loads and assembles scenario-specific prompt templates from a configuration dictionary.
    """
    def __init__(self, config: Dict[str, Any]):
        self.prompts_config = config.get("prompts", {})
        if not self.prompts_config:
            raise ValueError("Scenario config must contain a 'prompts' section.")

        self.scenario_header = self.prompts_config.get("scenario_header")
        if not self.scenario_header:
            raise ValueError("Prompts config must define a 'scenario_header'.")

    def get_comparison_base(self, policy_guidance: str = "") -> str:
        """
        Gets the fully assembled base prompt for the comparison algorithm.
        """
        template = self.prompts_config.get("comparison_base")
        if not template:
            raise ValueError("'comparison_base' template not found in config.")

        base_prompt = template.replace("{scenario_header}", self.scenario_header)
        base_prompt = base_prompt.replace("{policy_guidance}", policy_guidance)
        if policy_guidance:
            return f"{base_prompt.strip()}\n\nPolicy guidance: {policy_guidance.strip()}\n"
        return base_prompt

    def get_utility_base(self) -> str:
        """
        Gets the fully assembled base/initial prompt for the utility algorithm.
        """
        template = self.prompts_config.get("utility_base")
        if not template:
            raise ValueError("'utility_base' template not found in config.")
        return template.replace("{scenario_header}", self.scenario_header)

    def get_utility_refinement(self) -> str:
        """
        Gets the refinement prompt template for the utility algorithm.
        This one does not need the header as it refers to a previous turn.
        """
        template = self.prompts_config.get("utility_refinement")
        if not template:
            raise ValueError("'utility_refinement' template not found in config.")
        return template
