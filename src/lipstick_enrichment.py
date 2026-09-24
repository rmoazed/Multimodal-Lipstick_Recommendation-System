import requests
import re
from urllib.parse import urlparse


def build_search_query(lipstick):

    parts = [
        lipstick.get("brand"),
        lipstick.get("product_name"),
        lipstick.get("shade_name"),
        "lipstick swatch",
    ]

    return " ".join(
        str(part).strip()
        for part in parts
        if part is not None
        and str(part).strip()
    )


def search_images(
    query,
    searchapi_key,
    num_candidates=25,
):

    url = "https://www.searchapi.io/api/v1/search"

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": searchapi_key,
        "safe": "active",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    results = response.json()

    image_results = results.get(
        "images",
        [],
    )

    candidates = []

    for index, image in enumerate(
        image_results[:num_candidates],
        start=1,
    ):

        original = image.get(
            "original",
            {},
        )

        source = image.get(
            "source",
            {},
        )

        candidates.append(
            {
                "candidate_id": index,
                "title": image.get("title"),
                "original": original.get("link"),
                "thumbnail": image.get("thumbnail"),
                "source": source.get("name"),
                "link": source.get("link"),
            }
        )

    return candidates



def evaluate_identity_evidence(lipstick, candidate):
    """
    Evaluate deterministic evidence that a search-result image belongs
    to the target lipstick shade.

    Returns:
        dict with:
            identity_status:
                "verified", "contradicted", or "unresolved"
            evidence:
                list of human-readable evidence strings
    """

    brand = str(lipstick.get("brand") or "").strip().lower()
    product_name = str(
        lipstick.get("product_name") or ""
    ).strip().lower()
    shade_name = str(
        lipstick.get("shade_name") or ""
    ).strip().lower()

    title = str(candidate.get("title") or "").strip().lower()
    source = str(candidate.get("source") or "").strip().lower()
    page_url = str(candidate.get("link") or "").strip().lower()
    image_url = str(candidate.get("original") or "").strip().lower()

    evidence = []

    # ---------------------------------------------------------
    # 1. Extract target shade number
    # ---------------------------------------------------------

    target_numbers = re.findall(r"\d+", shade_name)

    target_shade_number = (
        target_numbers[-1] if target_numbers else None
    )

    # ---------------------------------------------------------
    # 2. Look for explicit "shade<number>" evidence in URLs
    # ---------------------------------------------------------

    url_text = f"{page_url} {image_url}"

    url_shade_matches = re.findall(
        r"shade[\s_\-]*([0-9]+)",
        url_text,
        flags=re.IGNORECASE,
    )

    if target_shade_number and url_shade_matches:

        if target_shade_number in url_shade_matches:
            evidence.append(
                f"URL explicitly identifies target shade "
                f"{target_shade_number}."
            )

            return {
                "identity_status": "verified",
                "evidence": evidence,
            }

        evidence.append(
            f"URL identifies shade(s) "
            f"{', '.join(url_shade_matches)}, "
            f"not target shade {target_shade_number}."
        )

        return {
            "identity_status": "contradicted",
            "evidence": evidence,
        }

    # ---------------------------------------------------------
    # 3. Record weaker brand/product evidence
    # ---------------------------------------------------------

    if brand and brand in source:
        evidence.append(
            f"Search source matches target brand: {brand}."
        )

    if brand and brand in title:
        evidence.append(
            f"Search title contains target brand: {brand}."
        )

    if product_name and product_name in title:
        evidence.append(
            f"Search title contains target product: "
            f"{product_name}."
        )

    # Check whether page/image comes from brand's domain.
    # This is supporting evidence, NOT exact-shade verification.
    try:
        page_domain = urlparse(page_url).netloc.lower()
        image_domain = urlparse(image_url).netloc.lower()

        if brand and (
            brand in page_domain or brand in image_domain
        ):
            evidence.append(
                f"URL appears to come from the target "
                f"brand's domain."
            )

    except ValueError:
        pass

    # ---------------------------------------------------------
    # 4. No exact shade evidence
    # ---------------------------------------------------------

    if not evidence:
        evidence.append(
            "No deterministic identity evidence found."
        )
    else:
        evidence.append(
            "Brand/product evidence exists, but exact shade "
            "identity is not established."
        )

    return {
        "identity_status": "unresolved",
        "evidence": evidence,
    }