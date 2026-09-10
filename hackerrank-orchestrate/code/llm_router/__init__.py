"""LLM Router Package for Message Notification Router.

Contains:
- LLMRouter: Gemini 3.6 Flash batch caller with 4-key round-robin rotation.
- RoutingDecision: Final structured output per message.
"""

from .llm_router import LLMRouter, RoutingDecision

__all__ = ["LLMRouter", "RoutingDecision"]
