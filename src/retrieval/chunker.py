"""Structure-aware chunker for nutrition documents.

Splits parsed sections into token-bounded chunks with:
- Sentence boundary splitting (never cut mid-sentence)
- Table handling (atomic if ≤ max_tokens, else split by row groups)
- Minimum chunk merging (< min_tokens merged with adjacent)
- Overlap at sentence boundaries
"""

import json
import re
from typing import Any, Dict, List

from transformers import AutoTokenizer


class StructureAwareChunker:
    def __init__(self, max_tokens: int = 256, overlap_tokens: int = 48, min_tokens: int = 30):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_document(self, parsed_doc: dict) -> List[Dict[str, Any]]:
        """Chunk all sections in a parsed document.

        Args:
            parsed_doc: Output from a parser (has source_id, document, sections, url).

        Returns:
            List of chunk dicts with id, content, metadata.
        """
        source_id = parsed_doc.get("source_id", "unknown")
        document = parsed_doc.get("document", "unknown")
        url = parsed_doc.get("url", "")
        source_type = self._infer_source_type(source_id)

        all_chunks = []

        for section in parsed_doc.get("sections", []):
            heading = section.get("heading", "")
            heading_hierarchy = section.get("heading_hierarchy", [document, heading])
            page = section.get("page")

            # Chunk text content
            content = section.get("content", "")
            if content.strip():
                text_chunks = self._chunk_text(content)
                for chunk_text in text_chunks:
                    all_chunks.append(self._build_chunk(
                        content=chunk_text,
                        source_id=source_id,
                        document=document,
                        section=heading,
                        heading_hierarchy=heading_hierarchy,
                        url=url,
                        page=page,
                        source_type=source_type,
                        is_table=False,
                    ))

            # Chunk tables
            for table in section.get("tables", []):
                table_chunks = self._chunk_table(table)
                for chunk_text in table_chunks:
                    all_chunks.append(self._build_chunk(
                        content=chunk_text,
                        source_id=source_id,
                        document=document,
                        section=heading,
                        heading_hierarchy=heading_hierarchy,
                        url=url,
                        page=page,
                        source_type=source_type,
                        is_table=True,
                    ))

        # Merge small chunks with adjacent, then discard any still below min
        all_chunks = self._merge_small_chunks(all_chunks)
        all_chunks = [c for c in all_chunks if self.token_count(c["content"]) >= self.min_tokens]

        # Assign sequential IDs
        section_counters: Dict[str, int] = {}
        for chunk in all_chunks:
            section_key = f"{source_id}__{self._slug(chunk['metadata']['document'])}__{self._slug(chunk['metadata']['section'])}"
            section_counters[section_key] = section_counters.get(section_key, 0) + 1
            chunk["id"] = f"{section_key}__{section_counters[section_key]:03d}"

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries with overlap."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.token_count(sentence)

            # Single sentence exceeds max? Split it further
            if sent_tokens > self.max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                # Force-split oversized sentence into smaller pieces
                chunks.extend(self._force_split_sentence(sentence))
                current_sentences = []
                current_tokens = 0
                continue

            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))

                # Overlap: keep trailing sentences that fit in overlap budget
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    s_tokens = self.token_count(s)
                    if overlap_count + s_tokens > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_count

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _chunk_table(self, table: dict) -> List[str]:
        """Chunk a table. Prefer atomic; split by row groups if too large."""
        table_text = self._table_to_text(table)
        if self.token_count(table_text) <= self.max_tokens:
            return [table_text]

        # Split by row groups, each retaining caption + header row
        caption = table.get("caption", "Table")
        rows = table.get("rows", [])
        if len(rows) < 2:
            return [table_text]

        header_row = rows[0]
        data_rows = rows[1:]
        header_text = f"{caption}\n{' | '.join(header_row)}"
        header_tokens = self.token_count(header_text)

        chunks = []
        current_rows: List[List[str]] = []
        current_len = header_tokens

        for row in data_rows:
            row_text = " | ".join(str(cell) for cell in row)
            row_tokens = self.token_count(row_text)

            if current_len + row_tokens > self.max_tokens and current_rows:
                chunk_text = self._format_table_chunk(caption, header_row, current_rows)
                chunks.append(chunk_text)
                current_rows = []
                current_len = header_tokens

            current_rows.append(row)
            current_len += row_tokens

        if current_rows:
            chunk_text = self._format_table_chunk(caption, header_row, current_rows)
            chunks.append(chunk_text)

        return chunks

    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge chunks below min_tokens with the next chunk from the same section.

        Only merges if the result stays within max_tokens.
        """
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            tc = self.token_count(chunk["content"])

            if tc < self.min_tokens and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                # Only merge within same section and if result stays within max_tokens
                if chunk["metadata"]["section"] == next_chunk["metadata"]["section"]:
                    combined_tokens = tc + self.token_count(next_chunk["content"])
                    if combined_tokens <= self.max_tokens:
                        next_chunk["content"] = chunk["content"] + " " + next_chunk["content"]
                        next_chunk["metadata"]["token_count"] = combined_tokens
                        i += 1
                        continue

            merged.append(chunk)
            i += 1

        return merged

    def _build_chunk(self, content: str, source_id: str, document: str,
                     section: str, heading_hierarchy: list, url: str,
                     page: int | None, source_type: str, is_table: bool) -> Dict[str, Any]:
        return {
            "id": "",  # assigned later
            "content": content,
            "metadata": {
                "source_id": source_id,
                "document": document,
                "section": section,
                "heading_hierarchy": heading_hierarchy,
                "url": url,
                "page": page,
                "source_type": source_type,
                "is_table": is_table,
                "token_count": self.token_count(content),
            },
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences. Handles abbreviations and list items."""
        # Split on sentence-ending punctuation followed by space+capital or directly by capital
        # (handles cases like "important?Vitamins" without space)
        parts = re.split(r'(?<=[.!?])\s*(?=[A-Z])', text)
        sentences = []
        for part in parts:
            # Further split on double newlines (paragraph breaks)
            for sub in part.split("\n\n"):
                sub = sub.strip()
                if sub:
                    sentences.append(sub)
        return sentences

    def _force_split_sentence(self, sentence: str) -> List[str]:
        """Force-split an oversized sentence into chunks ≤ max_tokens.

        Tries to split on:
        1. List items (- or * at start of line)
        2. Newlines
        3. Word boundaries (last resort)
        """
        # Try splitting on list items first
        list_items = re.split(r'\n\s*[-*]\s*', sentence)
        if len(list_items) > 1:
            # Reconstruct with bullet prefix
            items = [list_items[0]] if list_items[0].strip() else []
            items.extend(f"- {item}" for item in list_items[1:] if item.strip())

            # Recursively chunk each item if still too large
            result = []
            for item in items:
                if self.token_count(item) <= self.max_tokens:
                    result.append(item)
                else:
                    result.extend(self._force_split_sentence(item))
            return result if result else [sentence]

        # Try splitting on single newlines
        lines = sentence.split('\n')
        if len(lines) > 1:
            result = []
            current = []
            current_tokens = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line_tokens = self.token_count(line)
                if current_tokens + line_tokens > self.max_tokens and current:
                    result.append('\n'.join(current))
                    current = []
                    current_tokens = 0
                current.append(line)
                current_tokens += line_tokens
            if current:
                result.append('\n'.join(current))
            return result if result else [sentence]

        # Last resort: split on word boundaries
        words = sentence.split()
        result = []
        current_words = []
        current_tokens = 0
        for word in words:
            word_tokens = self.token_count(word)
            if current_tokens + word_tokens > self.max_tokens and current_words:
                result.append(' '.join(current_words))
                current_words = []
                current_tokens = 0
            current_words.append(word)
            current_tokens += word_tokens
        if current_words:
            result.append(' '.join(current_words))
        return result

    def _table_to_text(self, table: dict) -> str:
        caption = table.get("caption", "Table")
        rows = table.get("rows", [])
        lines = [caption]
        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    def _format_table_chunk(self, caption: str, header_row: list, data_rows: list) -> str:
        lines = [caption, " | ".join(str(cell) for cell in header_row)]
        for row in data_rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)

    def _slug(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

    def _infer_source_type(self, source_id: str) -> str:
        if source_id.startswith("nih"):
            return "government"
        if source_id.startswith("who"):
            return "government"
        if source_id.startswith("dga"):
            return "government"
        if source_id.startswith("myplate"):
            return "government"
        if source_id.startswith("acog"):
            return "professional_org"
        if source_id.startswith("issn"):
            return "professional_org"
        return "other"
