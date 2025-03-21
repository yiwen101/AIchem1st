import json
import requests
from typing import Dict, Any, List, Optional
import time

from src.utils.config import Config
from src.utils.logging import LoggingManager

logger = LoggingManager.get_logger()


class LLMAdapter:
    """
    Adapter for interfacing with language models.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the LLM adapter.
        
        Args:
            config: Configuration for the LLM
        """
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.api_base_url = config.get("api_base_url")
        
        if not self.api_key:
            logger.warning("No API key provided for LLM. API calls will fail.")
    
    def generate(self, 
                 prompt: str, 
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 deep_thinking: bool = False) -> str:
        """
        Generate a completion using the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature parameter for the LLM (if None, use config value)
            max_tokens: Maximum number of tokens to generate (if None, use config value)
            deep_thinking: Whether to use deep thinking mode
            
        Returns:
            The completion text from the LLM
        """
        temperature = temperature if temperature is not None else self.config.get("temperature")
        max_tokens = max_tokens if max_tokens is not None else self.config.get("max_tokens")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if deep_thinking:
            payload["deep_thinking"] = True
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Sending request to LLM API: {payload}")
                response = requests.post(
                    self.api_base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.debug(f"Received response from LLM")
                return content
                
            except Exception as e:
                logger.error(f"Error in LLM API call (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All retry attempts failed")
                    raise RuntimeError(f"Failed to get response from LLM after {max_retries} attempts: {str(e)}")
    
    def generate_json(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            schema: The expected JSON schema
            **kwargs: Additional arguments to pass to generate()
            
        Returns:
            The parsed JSON response
        """
        # Add schema instruction to the prompt
        schema_text = json.dumps(schema, indent=2)
        full_prompt = f"{prompt}\n\nRespond with a valid JSON object that conforms to the following schema:\n{schema_text}\n\nJSON:"
        
        # Set a higher temperature for more creative responses
        temperature = kwargs.get("temperature", 0.2)
        kwargs["temperature"] = temperature
        
        # Get the response
        response_text = self.generate(full_prompt, **kwargs)
        
        # Extract JSON from the response
        try:
            # Try to find JSON-like content in the response
            json_text = self._extract_json(response_text)
            result = json.loads(json_text)
            logger.debug("Successfully parsed JSON from LLM response")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {str(e)}")
            logger.debug(f"Problematic response: {response_text}")
            raise ValueError(f"LLM did not return valid JSON: {str(e)}")
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that may contain extra content.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            The extracted JSON as a string
        """
        # If the text starts with a markdown code block, extract it
        if "```json" in text or "```" in text:
            start = text.find("```") + 3
            if text[start:start+4] == "json":
                start += 4
            end = text.find("```", start)
            return text[start:end].strip()
        
        # If the text appears to be plain JSON, return it as is
        elif text.strip().startswith(("{", "[")):
            return text.strip()
        
        # Otherwise, try to find JSON-like content
        else:
            # Look for the first { or [ and the last } or ]
            start = text.find("{") if "{" in text else text.find("[")
            if start == -1:
                raise ValueError("No JSON object found in the response")
            
            end = text.rfind("}") + 1 if "}" in text else text.rfind("]") + 1
            return text[start:end].strip()


class PromptTemplate:
    """
    Template for generating prompts for the LLM.
    """
    
    def __init__(self, template: str):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with placeholders in {variable} format
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            The formatted prompt
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Missing required variable for prompt template: {missing_key}") 