"""
LexGuard — Recursive Text Splitter
====================================
Splits long text into semantically coherent chunks using a hierarchy
of separators (paragraphs → sentences → words).  Each chunk respects
a maximum token budget with configurable overlap so that clauses
spanning boundaries are never lost.

Usage:
    from text_splitter import RecursiveTextSplitter
    splitter = RecursiveTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(long_contract_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RecursiveTextSplitter:
    """Split text recursively using a hierarchy of separators.

    Parameters
    ----------
    chunk_size : int
        Target maximum number of *words* per chunk (proxy for tokens).
    chunk_overlap : int
        Number of *words* to overlap between consecutive chunks so that
        clause boundaries are not lost.
    separators : list[str]
        Ordered list of separator patterns (most preferred first).
        Defaults to paragraphs → sentence-ending punctuation → spaces.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",       # double newline (paragraph break)
        "\n",         # single newline
        ". ",         # sentence boundary
        "; ",         # semicolon (common in legal text)
        ", ",         # clause boundary
        " ",          # word boundary (last resort)
    ])

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────
    def split_text(self, text: str) -> list[str]:
        """Split *text* into chunks respecting the configured limits.

        Returns a list of non-empty stripped strings.
        """
        if not text or not text.strip():
            return []
        chunks = self._recursive_split(text.strip(), self.separators)
        # Merge very small trailing chunks into the previous one
        return self._merge_small_chunks(chunks)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────
    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Try to split on the first separator; recurse with the next if
        any resulting piece is still too large."""

        if self._word_count(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No separators left — hard-split by word count
            return self._hard_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        # Split on the current separator, keeping the separator at the
        # end of each segment (e.g., "sentence. " stays as "sentence.")
        parts = self._split_keeping_sep(text, sep)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + part) if current else part
            if self._word_count(candidate) <= self.chunk_size:
                current = candidate
            else:
                # Flush what we have
                if current:
                    chunks.append(current.strip())
                # If this single part is still too long, recurse deeper
                if self._word_count(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part.strip(), remaining_seps))
                    current = ""
                else:
                    current = part

        if current and current.strip():
            chunks.append(current.strip())

        # Add overlap between consecutive chunks
        return self._add_overlap(chunks)

    def _split_keeping_sep(self, text: str, sep: str) -> list[str]:
        """Split text and keep the separator attached to the preceding segment."""
        if sep in (".", ". ", "; "):
            # Use regex to split after sentence-ending punctuation
            pattern = re.escape(sep)
            parts = re.split(f"({pattern})", text)
            # Re-join separator with the preceding segment
            merged = []
            i = 0
            while i < len(parts):
                segment = parts[i]
                if i + 1 < len(parts) and parts[i + 1] == sep:
                    segment += parts[i + 1]
                    i += 2
                else:
                    i += 1
                if segment:
                    merged.append(segment)
            return merged
        else:
            return text.split(sep)

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split by word count."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the last N words of the previous chunk to the next one."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].split()
            overlap_words = prev_words[-self.chunk_overlap:]
            overlap_text = " ".join(overlap_words)
            merged = overlap_text + " " + chunks[i]
            # Don't let overlap push us over the limit too aggressively
            if self._word_count(merged) <= self.chunk_size + self.chunk_overlap:
                result.append(merged.strip())
            else:
                result.append(chunks[i])
        return result

    def _merge_small_chunks(self, chunks: list[str], min_words: int = 20) -> list[str]:
        """Merge chunks that are too small into the previous one."""
        if not chunks:
            return []
        result = [chunks[0]]
        for chunk in chunks[1:]:
            if self._word_count(chunk) < min_words and result:
                result[-1] = result[-1] + " " + chunk
            else:
                result.append(chunk)
        return [c.strip() for c in result if c.strip()]
