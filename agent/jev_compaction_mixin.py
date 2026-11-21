"""Mixin for integrating Jev tool-call compaction into context_compressor.

This mixin adds methods to run Jev-based compaction as an optional early phase
before summary-based compression. It can also be used standalone as an alternative
compaction strategy for scenarios where tool-call pruning is prioritized over
text summarization.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.jev_compaction import JevCompactor, JevOptions

logger = logging.getLogger(__name__)


class JevCompactionMixin:
    """Mixin for context_compressor to add Jev tool-call compaction support."""

    def _get_jev_compactor(self, options: Optional[JevOptions] = None) -> JevCompactor:
        """Lazily instantiate and cache a Jev compactor."""
        if not hasattr(self, "_jev_compactor_cache"):
            self._jev_compactor_cache = JevCompactor(options=options or JevOptions())
        return self._jev_compactor_cache

    def jev_compact(
        self,
        messages: List[Dict[str, Any]],
        options: Optional[JevOptions] = None,
    ) -> Dict[str, Any]:
        """Run Jev-based tool-call compaction.

        This is a standalone compaction method that uses Jev to score and drop
        stale tool calls/results without any text summarization.

        Args:
            messages: The conversation messages to compact
            options: Optional JevOptions to override defaults

        Returns:
            Dict with keys:
            - compacted: The compacted messages
            - stats: Compaction statistics (calls found, decisions made, usage)
        """
        try:
            compactor = self._get_jev_compactor(options)
            result = compactor.compact(messages)
            logger.info(f"Jev compaction stats: {result.get('stats', {})}")
            return result
        except Exception as e:
            logger.exception("Jev compaction failed; returning original messages")
            return {"compacted": messages, "stats": {"error": str(e)}}

    def _run_jev_preprocessing(
        self,
        messages: List[Dict[str, Any]],
        jev_options: Optional[JevOptions] = None,
    ) -> List[Dict[str, Any]]:
        """Run Jev compaction as a preprocessing step before summary-based compression.

        This can be called early in the compress() pipeline (e.g., in Phase 1)
        to clean up tool calls/results before the more expensive summary LLM is invoked.

        Returns the compacted messages, or original messages if Jev fails.
        """
        if not messages:
            return messages
        try:
            result = self.jev_compact(messages, jev_options)
            compacted = result.get("compacted", messages)
            stats = result.get("stats", {})
            if stats.get("calls_found", 0) > 0:
                logger.info(
                    f"Jev preprocessing removed {stats.get('decisions', {}).get('call_dropped', 0)} "
                    f"stale tool calls from {stats.get('calls_found')} total"
                )
            return compacted
        except Exception:
            logger.exception("Jev preprocessing failed; proceeding without it")
            return messages
