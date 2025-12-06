from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import re


class PromptTemplateInterface(ABC):
    """
    Abstract base class defining the interface for all prompt templates.
    This ensures consistent API across different prompt template implementations.
    """
    
    @abstractmethod
    def __init__(self, reasoning: bool = True, **kwargs):
        """
        Initialize the prompt template.
        
        Args:
            reasoning: Whether to include reasoning in the prompt
            **kwargs: Additional implementation-specific parameters
        """
        self.reasoning = reasoning
        self.history = []
    
    @abstractmethod
    def format(self, *args, **kwargs) -> str:
        """
        Format the main prompt based on input data.
        
        Returns:
            Formatted prompt string ready for LLM consumption
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the final choice.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed choice (e.g., 'A', 'B')
        """
        pass
    
    def add_to_history(self, entry: Dict[str, Any], choice: str, reasoning: Optional[str] = None):
        """
        Add a decision to the history for few-shot learning.
        
        Args:
            entry: The data that was compared
            choice: The choice that was made
            reasoning: Optional reasoning for the choice
        """
        history_entry = {
            "data": entry,
            "choice": choice
        }
        if reasoning:
            history_entry["reasoning"] = reasoning
        self.history.append(history_entry)
    
    def clear_history(self):
        """Clear all history entries."""
        self.history = []
    
    def get_history_length(self) -> int:
        """Get the number of entries in history."""
        return len(self.history)
    
    def _parse_final_tag(self, response: str, valid_choices: List[str]) -> str:
        """
        Common parsing logic for FINAL: X pattern.
        
        Args:
            response: LLM response string
            valid_choices: List of valid choices (e.g., ['A', 'B'])
            
        Returns:
            Parsed choice
        """
        # Look for FINAL: X pattern
        pattern = r'FINAL:\s*([' + ''.join(valid_choices) + '])'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback: look for isolated choice at the end
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip().upper()
            if last_line in valid_choices:
                return last_line
        
        raise ValueError(f"Could not parse response: {response}")
    
    @abstractmethod
    def get_base_prompt(self) -> str:
        """
        Get the base/system prompt for this template.
        
        Returns:
            Base prompt string
        """
        pass