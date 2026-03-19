"""
OpenRouter model implementation.

Provides implementation of the BaseModel interface for OpenRouter models.
"""

from typing import Dict, List, Any, Optional

from .orchestrator_model import OrchestratorModel
from ..infra.config import ConfigManager
from ..infra.logging_utils import get_logger
from ..conversation.provider_adapters.openrouter_adapter import OpenRouterHistoryAdapter

class OpenRouterModel(OrchestratorModel):
    """
    Implementation of OrchestratorModel for OpenRouter models.
    
    Handles user interaction, understands requirements, and generates
    high-quality tool calling instructions using OpenRouter's unified API
    that provides access to multiple AI models.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize the OpenRouter model with OpenRouter-specific configuration.
        
        Args:
            config: Configuration manager instance to use
        """
        if config is None:
            config = ConfigManager()
            
        # Push model type to call path
        config.push_to_call_path("openrouter")
        
        # Initialize logger
        self.logger = get_logger(config.get_call_path())
            
        history_adapter = OpenRouterHistoryAdapter()
        
        self.logger.debug("Creating OpenRouter model")
        
        super().__init__(
            base_url=config.get('openrouter.base_url', 'https://openrouter.ai/api/v1'),
            api_key=config.get('openrouter.api_key'),
            model_name=config.get('openrouter.model', 'openai/gpt-4o'),
            provider_name='openrouter',
            history_adapter=history_adapter,
            config=config
        )
        
        self.logger.debug("OpenRouter model created", {
            "model": config.get('openrouter.model', 'openai/gpt-4o'),
            "base_url": config.get('openrouter.base_url', 'https://openrouter.ai/api/v1')
        }) 