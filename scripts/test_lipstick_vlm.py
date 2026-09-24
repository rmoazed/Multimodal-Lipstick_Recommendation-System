from getpass import getpass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from src.database import LipstickDatabase
from src.lipstick_enrichment import (
    build_search_query,
    search_images,
    evaluate_identity_evidence,
)

QWEN_MODEL_NAME = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
LIPSTICK_ID = "L0139"

DB_PATH = "data/lipstick_recommender.db"
CANDIDATE_DIR = Path("data/enrichment_candidates")


def download_candidate(candidate):
    """Download one SearchAPI candidate and return its local path."""

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    candidate_id = candidate["candidate_id"]

    # Prefer the original image, but fall back to the thumbnail.
    urls_to_try = [
        candidate.get("original"),
        candidate.get("thumbnail"),
    ]

    for url in urls_to_try:
        if not url:
            continue

        try:
            response = requests.get(
                url,
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")

            image_path = (
                CANDIDATE_DIR / f"candidate_{candidate_id}.jpg"
            )
            image.save(image_path, quality=95)

            return str(image_path)

        except Exception as exc:
            print(f"Download attempt failed: {exc}")

    return None


def main():

    # ---------------------------------------------------------
    # 1. Load target lipstick
    # ---------------------------------------------------------

    db = LipstickDatabase(DB_PATH)
    lipstick = db.get_lipstick(LIPSTICK_ID)

    if lipstick is None:
        raise ValueError(
            f"{LIPSTICK_ID} was not found in the database."
        )

    print("\nTARGET LIPSTICK")
    print("----------------")
    print(lipstick)

    # ---------------------------------------------------------
    # 2. Search for image candidates
    # ---------------------------------------------------------

    query = build_search_query(lipstick)

    print("\nSEARCH QUERY")
    print("------------")
    print(query)

    searchapi_key = getpass("\nSearchAPI key: ")

    candidates = search_images(
        query,
        searchapi_key,
        num_candidates=25,
    )

    print(f"\nFound {len(candidates)} candidates.")

    if not candidates:
        raise RuntimeError(
            "SearchAPI returned no image candidates."
        )

    # ---------------------------------------------------------
    # 3. Load Qwen ONCE
    # ---------------------------------------------------------

    print("\nLoading Qwen...")

    model, processor = load(QWEN_MODEL_NAME)
    config = load_config(QWEN_MODEL_NAME)

    print("Qwen loaded!")

    # ---------------------------------------------------------
    # 4. Target metadata
    # ---------------------------------------------------------

    brand = lipstick.get("brand")
    product_name = lipstick.get("product_name")
    shade_name = lipstick.get("shade_name")

    # ---------------------------------------------------------
    # 5. Evaluate first 5 candidates independently
    # ---------------------------------------------------------

    results = []

    for candidate in candidates[:5]:

        print("\n" + "=" * 60)
        print(f"CANDIDATE {candidate['candidate_id']}")
        print("=" * 60)

        print("Title:", candidate.get("title"))
        print("Source:", candidate.get("source"))
        print("Page:", candidate.get("link"))
        print("Original image:", candidate.get("original"))

        identity = evaluate_identity_evidence(
            lipstick,
            candidate,
        )

        print(
            "Deterministic identity:",
            identity["identity_status"],
        )

        for item in identity["evidence"]:
            print("  -", item)

        if identity["identity_status"] == "contradicted":
            print(
                "Explicit identity contradiction — "
                "rejecting before Qwen."
            )
            continue

        # -----------------------------------------------------
        # Download this candidate
        # -----------------------------------------------------

        image_path = download_candidate(candidate)

        if image_path is None:
            print("Could not download candidate — skipping.")
            continue

        print("Downloaded:", image_path)

        # -----------------------------------------------------
        # Build verification prompt for THIS candidate
        # -----------------------------------------------------

        prompt = f"""
        You are evaluating an image for a lipstick catalog enrichment system.

        TARGET PRODUCT:
        Brand: {brand}
        Product: {product_name}
        Shade: {shade_name}

        The product identity is being evaluated separately using deterministic
        retrieval evidence.

        Your job is ONLY to evaluate whether the visible image is useful for
        estimating lipstick color.

        Do NOT decide whether the image belongs to the target product or shade.
        Do NOT infer product identity from visual color similarity.

    Classify the COLOR EVIDENCE as:

    - "good": a clear lipstick bullet, clear swatch, or clear close-up of
      lips wearing lipstick, with relatively useful lighting and enough
      visible color area for estimation

    - "usable": lipstick color is visible, but lighting, image size,
      composition, occlusion, or other conditions make color estimation
      less reliable

    - "poor": the image does not provide a sufficiently clear or isolated
      representation of lipstick color

    Also identify the primary visual evidence type as one of:

    - "lipstick_bullet"
    - "swatch"
    - "lips"
    - "multiple_products"
    - "packaging"
    - "other"

    Return ONLY valid JSON in exactly this structure:

    {{
        "color_evidence": "good|usable|poor",
        "visual_type": "lipstick_bullet|swatch|lips|multiple_products|packaging|other",
        "reason": "brief explanation"
    }}
    """

        # -----------------------------------------------------
        # Run Qwen on THIS candidate
        # -----------------------------------------------------

        print("Formatting prompt...")

        formatted_prompt = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=1,
        )

        print("Running inference...")

        response = generate(
            model,
            processor,
            formatted_prompt,
            [image_path],
            max_tokens=300,
            temperature=0.0,
        )

        print("\n--- QWEN VERIFICATION ---")
        print(response.text)

        # Save raw response for later analysis.
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "title": candidate.get("title"),
                "source": candidate.get("source"),
                "image_path": image_path,
                "qwen_response": response.text,
            }
        )

    # ---------------------------------------------------------
    # 6. Finish
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINISHED SCREENING FIRST 5 CANDIDATES")
    print("=" * 60)
    print(f"Successfully evaluated {len(results)} candidates.")


if __name__ == "__main__":
    main()