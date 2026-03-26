"""
Configuration management for the agent system.

Provides a unified interface for loading and accessing configuration
from various sources, including environment variables and config files.
"""

import os
import json
import copy
from typing import Any, Dict, Optional

class ConfigManager:
    """
    Manages configuration settings for the agent system.
    
    Provides a unified interface for accessing configuration values,
    allowing configuration to be set from various sources.
    """
    
    def __init__(
        self,
        # Provider configuration
        provider: str = 'deepseek',
        
        # DeepSeek configuration
        deepseek_api_key: Optional[str] = None,
        deepseek_base_url: str = 'https://api.deepseek.com',
        deepseek_model: str = 'deepseek-chat',
        deepseek_max_tokens: int = 4096,
        deepseek_temperature: float = 1.0,
        
        # OpenAI configuration
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: str = 'gpt-4',
        openai_tool_calling_model: str = 'gpt-3.5-turbo',
        openai_max_tokens: int = 4096,
        openai_temperature: float = 1.0,
        
        # OpenRouter configuration
        openrouter_api_key: Optional[str] = None,
        openrouter_base_url: str = 'https://openrouter.ai/api/v1',
        openrouter_model: str = 'openai/gpt-4o',
        openrouter_max_tokens: int = 4096,
        openrouter_temperature: float = 1.0,
        
        # Qwen configuration
        qwen_api_key: Optional[str] = None,
        qwen_base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        qwen_model: str = 'qwen-plus',
        qwen_max_tokens: int = 4096,
        qwen_temperature: float = 1.0,
        
        # Agent behavior configuration
        max_iterations: int = 10,
        custom_system_prompt: str = '',
        call_path: str = '',
        
        # Tool-calling configuration
        tool_calling_max_retries: int = 5,
        tool_calling_base_url: str = 'https://api.deepseek.com',
        tool_calling_model: str = 'deepseek-chat',
        tool_calling_version: str = 'stable',
        tool_calling_temperature: float = 0,
    ):
        """
        Initialize the config manager with configuration parameters.
        
        Args:
            provider: Model provider, one of 'deepseek', 'openai', or 'qwen'
            deepseek_api_key: DeepSeek API key, auto-loaded from DEEPSEEK_API_KEY when omitted
            deepseek_base_url: Base URL for the DeepSeek API
            deepseek_model: DeepSeek model name, recommended: 'deepseek-chat'
            deepseek_max_tokens: Maximum number of tokens for DeepSeek
            deepseek_temperature: DeepSeek temperature parameter controlling randomness
            openai_api_key: OpenAI API key, auto-loaded from COMPLETION_API_KEY when omitted
            openai_base_url: Base URL for the OpenAI API
            openai_model: OpenAI model name
            openai_tool_calling_model: OpenAI model dedicated to tool calling
            openai_max_tokens: Maximum number of tokens for OpenAI
            openai_temperature: OpenAI temperature parameter
            openrouter_api_key: OpenRouter API key, auto-loaded from OPENROUTER_API_KEY when omitted
            openrouter_base_url: Base URL for the OpenRouter API
            openrouter_model: OpenRouter model name, recommended: 'openai/gpt-4o'
            openrouter_max_tokens: Maximum number of tokens for OpenRouter
            openrouter_temperature: OpenRouter temperature parameter
            qwen_api_key: Qwen API key, auto-loaded from QWEN_API_KEY when omitted
            qwen_base_url: Base URL for the Qwen API
            qwen_model: Qwen model name
            qwen_max_tokens: Maximum number of tokens for Qwen
            qwen_temperature: Qwen temperature parameter
            max_iterations: Maximum number of agent iterations, which affects complex-task depth
            custom_system_prompt: Custom system prompt used to adjust agent behavior
            call_path: Call path used for logging hierarchy
            tool_calling_max_retries: Maximum number of tool-calling retries
            tool_calling_base_url: Base URL for the tool-calling API
            tool_calling_model: Model used for tool calling
            tool_calling_version: Tool-calling version; 'stable' is more reliable and 'turbo' is faster
            tool_calling_temperature: Temperature parameter for tool calling
        """
        # Automatically read API keys from environment variables
        if deepseek_api_key is None:
            deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        if openai_api_key is None:
            openai_api_key = os.getenv('COMPLETION_API_KEY')
        if openrouter_api_key is None:
            openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if qwen_api_key is None:
            qwen_api_key = os.getenv('QWEN_API_KEY')
        
        # Build the internal configuration dictionary
        self._config = {
            'openai': {
                'api_key': openai_api_key,
                'base_url': openai_base_url,
                'model': openai_model,
                'tool_calling_model': openai_tool_calling_model,
                'max_tokens': openai_max_tokens,
                'temperature': openai_temperature,
            },
            'deepseek': {
                'api_key': deepseek_api_key,
                'base_url': deepseek_base_url,
                'model': deepseek_model,
                'max_tokens': deepseek_max_tokens,
                'temperature': deepseek_temperature,
            },
            'openrouter': {
                'api_key': openrouter_api_key,
                'base_url': openrouter_base_url,
                'model': openrouter_model,
                'max_tokens': openrouter_max_tokens,
                'temperature': openrouter_temperature,
            },
            'qwen': {
                'api_key': qwen_api_key,
                'base_url': qwen_base_url,
                'model': qwen_model,
                'max_tokens': qwen_max_tokens,
                'temperature': qwen_temperature,
            },
            'agent': {
                'max_iterations': max_iterations,
                'custom_system_prompt': custom_system_prompt,
                'provider': provider,
                'call_path': call_path,
            },
            'tool_calling': {
                'max_retries': tool_calling_max_retries,
                'base_url': tool_calling_base_url,
                'model': tool_calling_model,
                'version': tool_calling_version,
                'temperature': tool_calling_temperature,
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            A deep copy of the entire configuration dictionary
        """
        # Return a deep copy to prevent external modification of the internal state
        return copy.deepcopy(self._config)
    
    def create_copy(self) -> 'ConfigManager':
        """
        Create a new ConfigManager instance with the same configuration.
        
        Returns:
            A new ConfigManager instance with a copy of the current configuration
        """
        new_config = ConfigManager()
        new_config.set_config(self.get_config())
        return new_config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set multiple configuration values at once.
        
        Args:
            config: Configuration dictionary to set
            
        Raises:
            KeyError: If any key does not exist in the default configuration structure
        """
        # Process each section in the config
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if value is not None:  # Skip None values
                        self.set(f"{section}.{key}", value)
            elif values is not None:  # Handle direct values like 'provider'
                self.set(section, values)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value (e.g. 'openai.api_key')
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value, or the default if not found
        """
        parts = key.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Dot-separated path to the configuration value (e.g. 'openai.api_key')
            value: The value to set
            
        Raises:
            KeyError: If the key does not exist in the default configuration structure
        """
        # Skip None values to prevent overriding defaults
        if value is None:
            return
            
        # Check if the key exists in the default configuration by creating a temporary default instance
        default_config = ConfigManager()._config
        parts = key.split('.')
        check_config = default_config
        
        for part in parts:
            if isinstance(check_config, dict) and part in check_config:
                check_config = check_config[part]
            else:
                raise KeyError(f"Config key '{key}' does not exist in the default configuration structure")
        
        # If we got here, the key exists in the default config, so we can set it
        config = self._config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]
            
        config[parts[-1]] = value
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
        """
        try:
            with open(file_path, 'r') as f:
                file_config = json.load(f)
                
            # Instead of deep merging manually, use the set_config method
            self.set_config(file_config)
        except Exception as e:
            print(f"Error loading configuration from {file_path}: {e}")

    def push_to_call_path(self, module_name: str) -> None:
        """
        Push a module name to the call path.
        
        Args:
            module_name: The module name to add to the call path
        """
        current_path = self.get('agent.call_path', '')
        if current_path:
            new_path = f"{current_path}->{module_name}"
        else:
            new_path = module_name
        self.set('agent.call_path', new_path)
    
    def get_call_path(self) -> str:
        """
        Get the current call path.
        
        Returns:
            The current call path as a string
        """
        return self.get('agent.call_path', '')
