from getpass import getpass
from pathlib import Path
import shutil

from src.database import LipstickDatabase
from src.lipstick_enrichment import (
    build_search_query,
    search_images,
    download_candidate,
)


LIPSTICK_ID = "L0139"
DB_PATH = "data/lipstick_recommender.db"
TARGET_VARIANT_ID = "C338700008"

OUTPUT_DIR = Path("data/color_sources")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD TARGET
# ------------------------------------------------------------

db = LipstickDatabase(DB_PATH)
lipstick = db.get_lipstick(LIPSTICK_ID)

query = build_search_query(lipstick)

print("Target:", LIPSTICK_ID)
print("Query:", query)


# ------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------

searchapi_key = getpass("SearchAPI key: ")

candidates = search_images(
    query,
    searchapi_key,
    num_candidates=25,
)


# ------------------------------------------------------------
# FIND VERIFIED VARIANT CANDIDATES
# ------------------------------------------------------------

verified_candidates = []

for candidate in candidates:

    searchable_text = " ".join([
        str(candidate.get("title") or ""),
        str(candidate.get("link") or ""),
        str(candidate.get("original") or ""),
    ])

    if TARGET_VARIANT_ID.lower() in searchable_text.lower():
        verified_candidates.append(candidate)


print(
    f"\nFound {len(verified_candidates)} candidate(s) "
    f"containing {TARGET_VARIANT_ID}."
)


# ------------------------------------------------------------
# DOWNLOAD EACH VERIFIED CANDIDATE
# ------------------------------------------------------------

for i, candidate in enumerate(verified_candidates, start=1):

    print("\n" + "=" * 60)
    print(f"VERIFIED IMAGE {i}")
    print("=" * 60)

    print("Search candidate:", candidate.get("candidate_id"))
    print("Title:", candidate.get("title"))
    print("Source:", candidate.get("source"))
    print("Original:", candidate.get("original"))

    downloaded_path = download_candidate(
        candidate,
        candidate_dir=OUTPUT_DIR,
    )

    if downloaded_path is None:
        print("Download failed.")
        continue

    # Give the file a stable experiment name so candidate ranking
    # changes don't matter later.
    stable_path = OUTPUT_DIR / f"rouge_premier_8_{i}.jpg"

    shutil.copy2(
        downloaded_path,
        stable_path,
    )

    print("Saved as:", stable_path)