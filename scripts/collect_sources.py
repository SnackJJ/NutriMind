import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import httpx
from httpx import HTTPStatusError, RequestError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NIH_PAGES = [
    "VitaminA", "VitaminB6", "VitaminB12", "VitaminC", "VitaminD", "VitaminE", "VitaminK",
    "Folate", "Biotin", "Choline", "PantothenicAcid", "Riboflavin", "Thiamin", "Niacin",
    "Calcium", "Iron", "Magnesium", "Zinc", "Iodine", "Selenium", "Potassium",
    "Omega3FattyAcids", "Probiotics"
]

SOURCE_MANIFEST = [
    {
        "id": "nih_ods",
        "description": "NIH Office of Dietary Supplements Fact Sheets",
        "base_url": "https://ods.od.nih.gov/factsheets/",
        "pages": [f"{page}-HealthProfessional/" for page in NIH_PAGES],
        "format": "html",
        "domain": "micronutrients",
        "license": "public_domain"
    },
    {
        "id": "dga_2020",
        "description": "Dietary Guidelines for Americans 2020-2025",
        "url": "https://www.dietaryguidelines.gov/sites/default/files/2021-03/Dietary_Guidelines_for_Americans-2020-2025.pdf",
        "filename": "dga_2020_2025.pdf",
        "format": "pdf",
        "domain": "dietary_guidelines",
        "license": "public_domain"
    },
    {
        "id": "who_sugars",
        "description": "WHO Sugars Intake Guideline",
        "url": "https://iris.who.int/bitstream/handle/10665/149782/9789241549028_eng.pdf",
        "filename": "who_sugars_intake.pdf",
        "format": "pdf",
        "domain": "dietary_guidelines",
        "license": "public_domain"
    },
    {
        "id": "who_sodium",
        "description": "WHO Sodium Intake Guideline",
        "url": "https://iris.who.int/bitstream/handle/10665/77985/9789241504836_eng.pdf",
        "filename": "who_sodium_intake.pdf",
        "format": "pdf",
        "domain": "dietary_guidelines",
        "license": "public_domain"
    },
    {
        "id": "issn_protein",
        "description": "ISSN Position Stand: protein and exercise",
        "url": "https://jissn.biomedcentral.com/counter/pdf/10.1186/s12970-017-0177-8.pdf",
        "filename": "issn_protein.pdf",
        "format": "pdf",
        "domain": "sports_nutrition",
        "license": "open_access"
    },
    {
        "id": "issn_nutrient_timing",
        "description": "ISSN Position Stand: nutrient timing",
        "url": "https://jissn.biomedcentral.com/counter/pdf/10.1186/s12970-017-0189-4.pdf",
        "filename": "issn_nutrient_timing.pdf",
        "format": "pdf",
        "domain": "sports_nutrition",
        "license": "open_access"
    },
    {
        "id": "myplate",
        "description": "USDA MyPlate Guidelines",
        "url": "https://fns-prod.azureedge.us/sites/default/files/resource-files/MyPlate-Dietary-Guidelines-Publication.pdf",
        "filename": "myplate_guidelines.pdf",
        "format": "pdf",
        "domain": "meal_planning",
        "license": "public_domain"
    },
    {
        "id": "acog_pregnancy",
        "description": "ACOG Nutrition During Pregnancy FAQ",
        "url": "https://www.acog.org/-/media/project/acog/acogorg/patients/files/faqs/nutrition-during-pregnancy.pdf",
        "filename": "acog_pregnancy.pdf",
        "format": "pdf",
        "domain": "life_stage",
        "license": "public_domain"
    },
    {
        "id": "niddk_gallstones",
        "description": "NIH NIDDK: Eating, Diet & Nutrition for Gallstones",
        "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/gallstones/eating-diet-nutrition",
        "filename": "gallstones_diet.html",
        "format": "html",
        "domain": "medical_nutrition",
        "license": "public_domain"
    },
    {
        "id": "niddk_gerd",
        "description": "NIH NIDDK: Eating, Diet & Nutrition for Acid Reflux (GERD)",
        "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/acid-reflux-ger-gerd-adults/eating-diet-nutrition",
        "filename": "gerd_diet.html",
        "format": "html",
        "domain": "medical_nutrition",
        "license": "public_domain"
    },
    {
        "id": "niddk_ibs",
        "description": "NIH NIDDK: Eating, Diet & Nutrition for Irritable Bowel Syndrome",
        "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/irritable-bowel-syndrome/eating-diet-nutrition",
        "filename": "ibs_diet.html",
        "format": "html",
        "domain": "medical_nutrition",
        "license": "public_domain"
    },
    {
        "id": "niddk_celiac",
        "description": "NIH NIDDK: Eating, Diet & Nutrition for Celiac Disease",
        "url": "https://www.niddk.nih.gov/health-information/digestive-diseases/celiac-disease/eating-diet-nutrition",
        "filename": "celiac_diet.html",
        "format": "html",
        "domain": "medical_nutrition",
        "license": "public_domain"
    },
    {
        "id": "medlineplus_food_allergy",
        "description": "MedlinePlus (NIH NLM): Food Allergy Overview",
        "url": "https://medlineplus.gov/foodallergy.html",
        "filename": "food_allergy.html",
        "format": "html",
        "domain": "food_safety",
        "license": "public_domain"
    }
]


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download_file(client: httpx.Client, url: str) -> bytes:
    try:
        logger.info(f"Downloading {url}...")
        resp = client.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code != 200:
            logger.error(f"Failed to download {url}: Status code {resp.status_code}")
            return b""
        time.sleep(1.0)  # rate limit
        return resp.content
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return b""


def collect_sources(output_dir: Path):
    manifest_info = {}
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_info = json.load(f)

    # Use a custom user agent to avoid basic blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NutriAgent/1.0; +http://example.com/bot)"
    }

    with httpx.Client(headers=headers) as client:
        # We also need a dummy HTML source for safety templates, so we don't try to download it
        for source in SOURCE_MANIFEST:
            source_id = source["id"]
            save_dir = output_dir / source_id
            save_dir.mkdir(parents=True, exist_ok=True)

            if "pages" in source:
                base_url = source["base_url"]
                for page in source["pages"]:
                    url = f"{base_url}{page}"
                    # filename like VitaminA-HealthProfessional.html
                    filename = page.strip("/").split("/")[-1] + ".html"
                    filepath = save_dir / filename
                    
                    if filepath.exists():
                        # File exists — skip download but still register in manifest
                        logger.info(f"Skipping download {url}, file exists.")
                        if url not in manifest_info:
                            content = filepath.read_bytes()
                            manifest_info[url] = {
                                "source_id": source_id,
                                "filename": str(filepath.relative_to(output_dir)),
                                "hash": file_hash(content),
                                "timestamp": time.time(),
                                "type": "nih_ods_html",
                                "domain": source.get("domain", "general")
                            }
                        continue
                    
                    content = download_file(client, url)
                    if content:
                        filepath.write_bytes(content)
                        manifest_info[url] = {
                            "source_id": source_id,
                            "filename": str(filepath.relative_to(output_dir)),
                            "hash": file_hash(content),
                            "timestamp": time.time(),
                            "type": "nih_ods_html",
                            "domain": source.get("domain", "general")
                        }
            elif "url" in source:
                url = source["url"]
                filename = source["filename"]
                filepath = save_dir / filename
                
                if filepath.exists():
                    # File exists — skip download but still register in manifest
                    logger.info(f"Skipping download {url}, file exists.")
                    if url not in manifest_info:
                        content = filepath.read_bytes()
                        manifest_info[url] = {
                            "source_id": source_id,
                            "filename": str(filepath.relative_to(output_dir)),
                            "hash": file_hash(content),
                            "timestamp": time.time(),
                            "type": source["format"],
                            "domain": source.get("domain", "general")
                        }
                    continue
                
                content = download_file(client, url)
                if content:
                    # Basic PDF validation (only for PDF format sources)
                    if source["format"] == "pdf" and content[:4] != b"%PDF":
                        logger.error(f"Downloaded content for {url} is not a valid PDF")
                        continue

                    filepath.write_bytes(content)
                    manifest_info[url] = {
                        "source_id": source_id,
                        "filename": str(filepath.relative_to(output_dir)),
                        "hash": file_hash(content),
                        "timestamp": time.time(),
                        "type": source["format"],
                        "domain": source.get("domain", "general")
                    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    logger.info("Source collection completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download external primary sources.")
    parser.add_argument("--output-dir", type=str, default="data/raw/", help="Output raw directory")
    args = parser.parse_args()

    collect_sources(Path(args.output_dir))
