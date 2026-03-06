from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, prompts_dir: str = "prompts"):
        self.name = name
        self.prompts_dir = Path(prompts_dir)
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load the system prompt for this agent."""
        prompt_file = self.prompts_dir / f"{self.name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r') as f:
            return f.read()
    
    @abstractmethod
    def process(self, input_data: str, context: Dict[str, Any] = None) -> str:
        """Process input and return output."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"