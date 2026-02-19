"""Generic HTML parser for non-NIH sources (ACOG, MyPlate, etc.)."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class GenericHTMLParser:
    """Parse generic HTML pages into structured sections.

    Works with any HTML page that has heading tags (h1-h4) and
    paragraph/list content. Less specialized than NIHFactSheetParser.
    """

    def parse(self, html_path: Path, manifest: dict) -> Dict[str, Any]:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")

        source_id = manifest.get("source_id", "html_source")
        doc_title = manifest.get("document_title", "")

        if not doc_title:
            title_el = soup.find("title") or soup.find("h1")
            doc_title = title_el.get_text(strip=True) if title_el else html_path.stem

        # Try to find main content area
        content_container = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="content")
            or soup.find("div", {"role": "main"})
            or soup.body
        )

        if not content_container:
            return {"source_id": source_id, "document": doc_title, "sections": []}

        sections: List[Dict[str, Any]] = []
        current_heading = "Introduction"
        current_content: List[str] = []
        current_tables: List[Dict] = []

        heading_tags = {"h1", "h2", "h3", "h4"}

        for element in content_container.find_all(heading_tags | {"p", "ul", "ol", "table"}):
            if element.name in heading_tags:
                text = element.get_text(strip=True)
                if not text:
                    continue
                if current_content or current_tables:
                    sections.append(self._build_section(doc_title, current_heading, current_content, current_tables))
                current_heading = text
                current_content = []
                current_tables = []
            elif element.name == "table":
                rows = []
                for tr in element.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if any(cells):
                        rows.append(cells)
                if rows:
                    caption = element.find("caption")
                    cap_text = caption.get_text(strip=True) if caption else "Table"
                    current_tables.append({"caption": cap_text, "rows": rows})
            elif element.name in ("ul", "ol"):
                items = [li.get_text(strip=True) for li in element.find_all("li", recursive=False)]
                bullet = "- " if element.name == "ul" else "1. "
                list_text = "\n".join([f"{bullet}{item}" for item in items if item])
                if list_text:
                    current_content.append(list_text)
            elif element.name == "p":
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # Skip tiny nav/footer text
                    current_content.append(text)

        if current_content or current_tables:
            sections.append(self._build_section(doc_title, current_heading, current_content, current_tables))

        # Filter out empty or very short sections
        sections = [s for s in sections if len(s["content"]) > 50]

        return {
            "source_id": source_id,
            "document": doc_title,
            "sections": sections,
        }

    def _build_section(
        self, title: str, heading: str, content: List[str], tables: List[Dict]
    ) -> Dict[str, Any]:
        return {
            "heading": heading,
            "heading_hierarchy": [title, heading],
            "content": "\n\n".join(content),
            "tables": tables,
            "page": None,
        }
