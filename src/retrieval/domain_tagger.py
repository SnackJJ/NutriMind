"""Two-level domain tagging for RAG chunks.

Level 1 (heading): Single keyword match in section heading — high signal.
Level 2 (content): Requires min_content_hits different keywords — reduces false positives.
"""

from typing import Dict, List

DOMAIN_RULES: Dict[str, Dict] = {
    "sports_nutrition": {
        "heading_keywords": ["exercise", "athlete", "training", "sport", "performance"],
        "content_keywords": [
            "exercise", "athlete", "training", "ergogenic", "recovery",
            "muscle protein", "post-workout", "pre-workout", "endurance",
        ],
        "min_content_hits": 2,
    },
    "life_stage": {
        "heading_keywords": ["pregnancy", "lactation", "infant", "elderly", "pediatric", "aging"],
        "content_keywords": [
            "pregnant", "pregnancy", "lactation", "breastfeeding", "infant",
            "elderly", "older adult", "children", "prenatal", "postnatal",
        ],
        "min_content_hits": 2,
    },
    "food_safety": {
        "heading_keywords": ["safety", "toxicity", "interaction", "adverse", "allergen",
                             "allergy", "intolerance", "sensitivity", "reaction"],
        "content_keywords": [
            "toxicity", "overdose", "upper limit", "drug interaction",
            "contraindicated", "adverse effect", "allergen", "intolerance",
            "food allergy", "food intolerance", "allergic reaction", "anaphylaxis",
            "elimination diet", "food sensitivity",
        ],
        "min_content_hits": 2,
    },
    "medical_nutrition": {
        "heading_keywords": ["diabetes", "cardiovascular", "kidney", "disease", "clinical",
                             "gallbladder", "gallstone", "digestive", "gerd", "celiac",
                             "reflux", "liver", "ibs"],
        "content_keywords": [
            "diabetes", "cardiovascular", "hypertension", "kidney disease",
            "renal", "heart disease", "cholesterol", "blood pressure",
            "gallbladder", "gallstones", "biliary", "gerd", "acid reflux",
            "celiac disease", "irritable bowel", "crohn", "colitis",
            "fatty liver", "liver disease",
        ],
        "min_content_hits": 2,
    },
    "meal_planning": {
        "heading_keywords": ["food source", "dietary source", "intake"],
        "content_keywords": [
            "food source", "rich in", "good source", "dietary intake",
            "servings", "portion", "meal", "daily intake",
        ],
        "min_content_hits": 2,
    },
    "supplements": {
        "heading_keywords": ["supplement", "supplementation", "fortification"],
        "content_keywords": [
            "supplement", "supplementation", "fortified",
            "capsule", "tablet", "dosage", "milligrams",
            "micrograms", "iu", "dietary supplement",
        ],
        "min_content_hits": 2,
    },
    "weight_management": {
        "heading_keywords": ["weight", "obesity", "calorie", "energy balance", "bmi"],
        "content_keywords": [
            "weight loss", "weight gain", "obesity", "overweight",
            "caloric deficit", "energy balance", "bmi",
            "body weight", "calorie restriction",
        ],
        "min_content_hits": 2,
    },
}


def assign_domains(chunk: dict, source_primary_domain: str) -> List[str]:
    """Assign domains to a chunk using two-level matching.

    Args:
        chunk: Must have metadata.section and content fields.
        source_primary_domain: The source document's primary domain (always included).

    Returns:
        List of domain strings assigned to this chunk.
    """
    domains = {source_primary_domain}

    heading = chunk["metadata"]["section"].lower()
    content = chunk["content"].lower()

    for domain, rules in DOMAIN_RULES.items():
        # Level 1: Heading match (high signal, single keyword sufficient)
        if any(kw in heading for kw in rules["heading_keywords"]):
            domains.add(domain)
            continue

        # Level 2: Content match (require multiple keywords to reduce false positives)
        content_hits = [kw for kw in rules["content_keywords"] if kw in content]
        unique_hits = len(set(content_hits))
        if unique_hits >= rules["min_content_hits"]:
            domains.add(domain)

    return list(domains)
