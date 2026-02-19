"""NIH Office of Dietary Supplements HTML parser."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NIHFactSheetParser:
    """Parse NIH ODS Health Professional Fact Sheets (HTML).

    These have a consistent structure:
    - <h1> title
    - <h2>/<h3> section headings
    - Tables with RDA/AI values
    - Bulleted lists of food sources
    """

    def parse(self, html_path: Path, manifest: dict) -> Dict[str, Any]:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")

        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else html_path.stem
        title = (
            title.replace(" — Health Professional Fact Sheet", "")
            .replace(" \u2014 Health Professional Fact Sheet", "")
            .replace("\u2014 Health Professional Fact Sheet", "")
            .strip()
        )

        sections: List[Dict[str, Any]] = []

        content_container = (
            soup.find("div", {"id": "omni-col2"})
            or soup.find("main")
            or soup.find("div", class_="article")
        )
        if not content_container:
            content_container = soup.body
        if not content_container:
            return {"source_id": manifest.get("source_id", "nih_ods"), "document": title, "sections": []}

        current_heading = "Introduction"
        current_content: List[str] = []
        current_tables: List[Dict] = []

        for element in content_container.find_all(["h2", "h3", "p", "ul", "ol", "table"]):
            if element.name in ["h2", "h3"]:
                if current_content or current_tables:
                    sections.append(self._build_section(title, current_heading, current_content, current_tables))
                current_heading = element.get_text(strip=True)
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
            elif element.name in ["ul", "ol"]:
                items = [li.get_text(strip=True) for li in element.find_all("li")]
                bullet = "- " if element.name == "ul" else "1. "
                list_text = "\n".join([f"{bullet}{item}" for item in items if item])
                if list_text:
                    current_content.append(list_text)
            elif element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    current_content.append(text)

        if current_content or current_tables:
            sections.append(self._build_section(title, current_heading, current_content, current_tables))

        return {
            "source_id": manifest.get("source_id", "nih_ods"),
            "document": title,
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
