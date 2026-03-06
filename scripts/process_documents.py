"""Parse raw documents into structured JSON.

Uses NIHFactSheetParser for HTML and DoclingPDFParser for PDFs.
Output: data/parsed/{source_id}_{filename}.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path for src.* imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_source_meta(source_id: str) -> dict:
    """Return document title metadata per source."""
    TITLES = {
        "dga_2020": "Dietary Guidelines for Americans 2020-2025",
        "who_sugars": "WHO Sugars Intake Guideline",
        "who_sodium": "WHO Sodium Intake Guideline",
        "issn_protein": "ISSN Position Stand: Protein and Exercise",
        "issn_nutrient_timing": "ISSN Position Stand: Nutrient Timing",
        "myplate": "USDA MyPlate Dietary Guidelines",
        "acog_pregnancy": "ACOG Nutrition During Pregnancy",
        "niddk_gallstones": "NIH NIDDK: Eating, Diet & Nutrition for Gallstones",
        "niddk_gerd": "NIH NIDDK: Eating, Diet & Nutrition for Acid Reflux (GERD)",
        "niddk_ibs": "NIH NIDDK: Eating, Diet & Nutrition for Irritable Bowel Syndrome",
        "niddk_celiac": "NIH NIDDK: Eating, Diet & Nutrition for Celiac Disease",
        "niaid_food_allergy": "NIH NIAID: Food Allergy Overview",
        "medlineplus_food_allergy": "MedlinePlus: Food Allergy Overview",
    }
    return {"document_title": TITLES.get(source_id, source_id)}


def process_documents(raw_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error("Manifest not found. Run collect_sources.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Lazy-load parsers
    nih_parser = None
    pdf_parser = None
    html_parser = None

    for url, info in manifest.items():
        rel_path = info["filename"]
        source_id = info["source_id"]
        filepath = raw_dir / rel_path

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}, skipping")
            continue

        logger.info(f"Processing {filepath}")

        parsed_data = None

        if info["type"] == "nih_ods_html":
            if nih_parser is None:
                from src.retrieval.parsers.nih_parser import NIHFactSheetParser
                nih_parser = NIHFactSheetParser()
            parsed_data = nih_parser.parse(filepath, info)
            parsed_data["url"] = url

        elif info["type"] == "html":
            if html_parser is None:
                from src.retrieval.parsers.generic_html_parser import GenericHTMLParser
                html_parser = GenericHTMLParser()
            meta = {**info, **get_source_meta(source_id)}
            parsed_data = html_parser.parse(filepath, meta)
            parsed_data["url"] = url

        elif info["type"] == "pdf":
            if pdf_parser is None:
                from src.retrieval.parsers.docling_parser import DoclingPDFParser
                pdf_parser = DoclingPDFParser()
            meta = {**info, **get_source_meta(source_id)}
            parsed_data = pdf_parser.parse(filepath, meta)
            parsed_data["url"] = url

        if parsed_data and parsed_data.get("sections"):
            out_name = f"{source_id}_{filepath.stem}.json"
            out_path = output_dir / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  -> {out_path} ({len(parsed_data['sections'])} sections)")
        else:
            logger.warning(f"  -> No sections extracted from {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw documents into structured JSON.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw documents directory")
    parser.add_argument("--output-dir", type=str, default="data/parsed", help="Output parsed directory")
    args = parser.parse_args()

    process_documents(Path(args.raw_dir), Path(args.output_dir))
