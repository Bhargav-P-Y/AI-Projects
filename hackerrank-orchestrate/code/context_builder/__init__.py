"""Context Builder Package for Message Notification Router.

Contains:
- HybridRetriever: BM25 + Dense Embedding + Recency + Engagement evidence retriever.
- ContextBuilder: Assembles rich XML-sandboxed context for LLM prompt reasoning.
"""

from .hybrid_retriever import HybridRetriever, EvidenceCandidate
from .context_builder import ContextBuilder, AssembledContext

__all__ = [
    "HybridRetriever",
    "EvidenceCandidate",
    "ContextBuilder",
    "AssembledContext",
]
