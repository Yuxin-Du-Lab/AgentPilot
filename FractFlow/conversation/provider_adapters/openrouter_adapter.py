"""
OpenRouter history adapter.

Formats conversation history according to OpenRouter's API requirements.
"""

from typing import List, Dict, Any, Optional
from .base_adapter import HistoryAdapter

class OpenRouterHistoryAdapter(HistoryAdapter):
    """
    History adapter for OpenRouter models.
    
    Formats conversation history according to OpenRouter's requirements.
    OpenRouter models use OpenAI-compatible API format, expecting alternating 
    user and assistant messages, with tools information embedded in the content 
    rather than as separate roles.
    """
    # All implementations moved to base class
    pass 