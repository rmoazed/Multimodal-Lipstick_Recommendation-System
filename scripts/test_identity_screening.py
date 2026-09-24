from getpass import getpass

from src.database import LipstickDatabase
from src.lipstick_enrichment import (
    build_search_query,
    search_images,
    evaluate_identity_evidence,
)


LIPSTICK_ID = "L0139"
DB_PATH = "data/lipstick_recommender.db"


def main():

    # Load our user-added lipstick
    db = LipstickDatabase(DB_PATH)
    lipstick = db.get_lipstick(LIPSTICK_ID)

    if lipstick is None:
        raise ValueError(
            f"{LIPSTICK_ID} was not found in the database."
        )

    print("\nTARGET LIPSTICK")
    print("----------------")
    print("Brand:", lipstick.get("brand"))
    print("Product:", lipstick.get("product_name"))
    print("Shade:", lipstick.get("shade_name"))

    # Build the same SearchAPI query used by our enrichment pipeline
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

    # Keep counts so we can inspect the overall distribution.
    counts = {
        "verified": 0,
        "contradicted": 0,
        "unresolved": 0,
    }

    # Evaluate every search result using deterministic evidence only.
    for candidate in candidates:

        result = evaluate_identity_evidence(
            lipstick,
            candidate,
        )

        status = result["identity_status"]
        counts[status] += 1

        print("\n" + "=" * 60)
        print(
            f"CANDIDATE {candidate['candidate_id']}: "
            f"{status.upper()}"
        )
        print("=" * 60)

        print("Title:", candidate.get("title"))
        print("Source:", candidate.get("source"))
        print("Page:", candidate.get("link"))
        print("Original image:", candidate.get("original"))

        print("\nEvidence:")

        for item in result["evidence"]:
            print("  -", item)

    # Final distribution
    print("\n" + "=" * 60)
    print("IDENTITY SCREENING SUMMARY")
    print("=" * 60)

    print("Verified:    ", counts["verified"])
    print("Contradicted:", counts["contradicted"])
    print("Unresolved:  ", counts["unresolved"])
    print("Total:       ", len(candidates))


if __name__ == "__main__":
    main()