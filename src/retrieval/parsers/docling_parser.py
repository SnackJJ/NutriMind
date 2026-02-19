"""PDF parser using docling for layout-aware extraction."""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DoclingPDFParser:
    """Parse PDF documents using docling.

    Advantages over pdfplumber:
    - Layout-aware heading detection (font size, bold)
    - Structured table extraction
    - Multi-column support
    """

    def __init__(self):
        from docling.document_converter import DocumentConverter

        self.converter = DocumentConverter()

    def parse(self, pdf_path: Path, manifest: dict) -> Dict[str, Any]:
        source_id = manifest.get("source_id", "pdf_source")
        doc_title = manifest.get("document_title", pdf_path.stem)

        try:
            result = self.converter.convert(str(pdf_path))
        except Exception as e:
            logger.error(f"Docling failed to convert {pdf_path}: {e}")
            return {"source_id": source_id, "document": doc_title, "sections": []}

        doc = result.document
        sections: List[Dict[str, Any]] = []

        current_heading = "Introduction"
        current_content: List[str] = []
        current_tables: List[Dict] = []
        current_page: int = 1

        for item, _level in doc.iterate_items():
            label = item.label if hasattr(item, "label") else ""

            # Track page number from provenance
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no"):
                        current_page = prov.page_no

            if label == "section_header":
                if current_content or current_tables:
                    sections.append(
                        self._build_section(doc_title, current_heading, current_content, current_tables, current_page)
                    )
                current_heading = item.text.strip() if hasattr(item, "text") else str(item)
                current_content = []
                current_tables = []

            elif label == "table":
                table_data = self._extract_table(item)
                if table_data:
                    current_tables.append(table_data)

            elif label in ("text", "paragraph", "list_item", "caption"):
                text = item.text.strip() if hasattr(item, "text") else ""
                if text:
                    current_content.append(text)

        # Final section
        if current_content or current_tables:
            sections.append(
                self._build_section(doc_title, current_heading, current_content, current_tables, current_page)
            )

        return {
            "source_id": source_id,
            "document": doc_title,
            "sections": sections,
        }

    def _build_section(
        self,
        doc_title: str,
        heading: str,
        content: List[str],
        tables: List[Dict],
        page: int,
    ) -> Dict[str, Any]:
        return {
            "heading": heading,
            "heading_hierarchy": [doc_title, heading],
            "content": "\n\n".join(content),
            "tables": tables,
            "page": page,
        }

    def _extract_table(self, table_item) -> Dict[str, Any] | None:
        """Extract table rows from a docling table item."""
        try:
            df = table_item.export_to_dataframe()
            rows = [df.columns.tolist()] + df.values.tolist()
            # Convert all cells to strings
            rows = [[str(cell) for cell in row] for row in rows]
            caption = ""
            if hasattr(table_item, "caption") and table_item.caption:
                caption = table_item.caption
            elif hasattr(table_item, "text") and table_item.text:
                caption = table_item.text[:80]
            return {"caption": caption or "Table", "rows": rows}
        except Exception as e:
            logger.warning(f"Failed to extract table: {e}")
            return None
