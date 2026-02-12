import os
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, host: Optional[str] = None, timeout: int = 120):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", timeout))
        self.model_mappings = self._load_model_mappings()
        
    def _load_model_mappings(self) -> Dict[str, str]:
        """Load model aliases from environment variables."""
        mappings = {}
        for key, value in os.environ.items():
            if key.endswith("_MODEL"):
                alias = key.replace("_MODEL", "").lower()
                mappings[alias] = value
                mappings[key] = value  # Also store full key for direct access
        return mappings
    
    def resolve_model(self, model_alias: str) -> str:
        """Resolve a model alias to actual model name."""
        # Check if it's an alias like "REASONING_MODEL" or "reasoning"
        clean_alias = model_alias.replace("_MODEL", "").lower()
        
        if model_alias in self.model_mappings:
            return self.model_mappings[model_alias]
        if clean_alias in self.model_mappings:
            return self.model_mappings[clean_alias]
        
        # If not found in mappings, assume it's the actual model name
        return model_alias
    
    def generate(self, 
                 model: str, 
                 system_prompt: str, 
                 user_input: str,
                 stream: bool = False) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            model: Model alias (e.g., 'REASONING_MODEL') or actual model name
            system_prompt: System prompt to set behavior
            user_input: User message/prompt
            stream: Whether to stream response (not implemented for simplicity)
            
        Returns:
            Generated text response
        """
        actual_model = self.resolve_model(model)
        
        url = f"{self.host}/api/generate"
        
        payload = {
            "model": actual_model,
            "system": system_prompt,
            "prompt": user_input,
            "stream": stream
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. "
                "Is Ollama running?"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request timed out after {self.timeout} seconds. "
                "Consider increasing REQUEST_TIMEOUT in .env"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def list_models(self) -> list:
        """List available models from Ollama."""
        url = f"{self.host}/api/tags"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            raise ConnectionError(f"Could not fetch models: {e}")