import requests
import re
from urllib.parse import urlparse
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

import numpy as np

from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab


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
    
def download_candidate(
    candidate,
    candidate_dir="data/enrichment_candidates",
):
    """Download one SearchAPI candidate and return its local path."""

    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

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

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            image_path = (
                candidate_dir / f"candidate_{candidate_id}.jpg"
            )

            image.save(image_path, quality=95)

            return str(image_path)

        except Exception as exc:
            print(f"Download attempt failed: {exc}")

    return None


def evaluate_identity_evidence(
    lipstick,
    candidate,
    variant_ids=None,
):
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

    variant_ids = variant_ids or []

    for variant_id in variant_ids:
        variant_id = str(variant_id).strip().lower()

        if variant_id and variant_id in url_text:
            evidence.append(
                f"URL contains verified variant identifier "
                f"{variant_id}."
            )
            return {
                "identity_status": "verified",
                "evidence": evidence,
            }

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

def rgb_to_lab(rgb):
    """Convert one sRGB color from 0-255 RGB to CIELAB."""
    rgb = np.asarray(rgb, dtype=float) / 255.0
    lab = rgb2lab(rgb.reshape(1, 1, 3))
    return lab[0, 0]


def extract_color_from_roi(
    image,
    roi,
    n_clusters=3,
    random_state=42,
):
    """
    Extract color-cluster information from an image ROI.

    Parameters
    ----------
    image : PIL.Image.Image
        Source RGB image.

    roi : tuple
        (left, top, right, bottom) crop coordinates.

    n_clusters : int
        Number of K-means color clusters.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ROI, pixel statistics, cluster centers in RGB and LAB,
        cluster proportions, brightness, labels, and cluster map.

    Notes
    -----
    This function intentionally does NOT decide which cluster
    represents the lipstick. Cluster selection is handled
    separately because the appropriate rule may depend on
    visual type and spatial evidence.
    """

    # Ensure consistent RGB input.
    image = image.convert("RGB")

    # -----------------------------
    # Crop ROI
    # -----------------------------

    roi_image = image.crop(roi)
    roi_array = np.asarray(roi_image)

    height, width, channels = roi_array.shape

    if channels != 3:
        raise ValueError(
            "Expected a 3-channel RGB image."
        )

    pixels = roi_array.reshape(-1, 3)

    if len(pixels) < n_clusters:
        raise ValueError(
            "ROI contains fewer pixels than requested clusters."
        )

    # -----------------------------
    # Basic ROI statistics
    # -----------------------------

    mean_rgb = pixels.mean(axis=0)
    median_rgb = np.median(pixels, axis=0)

    # -----------------------------
    # K-means
    # -----------------------------

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = kmeans.fit_predict(pixels)
    centers_rgb = kmeans.cluster_centers_

    counts = np.bincount(
        labels,
        minlength=n_clusters,
    )

    percentages = (
        counts / len(pixels) * 100.0
    )

    # Simple diagnostic brightness measure.
    brightness = centers_rgb.mean(axis=1)

    # -----------------------------
    # RGB -> LAB
    # -----------------------------

    centers_lab = np.array([
        rgb_to_lab(center)
        for center in centers_rgb
    ])

    # Restore labels to ROI geometry.
    cluster_map = labels.reshape(
        height,
        width,
    )

    return {
        "roi": roi,
        "roi_image": roi_image,
        "roi_size": roi_image.size,
        "n_pixels": len(pixels),

        "mean_rgb": mean_rgb,
        "median_rgb": median_rgb,
        "mean_lab": rgb_to_lab(mean_rgb),
        "median_lab": rgb_to_lab(median_rgb),

        "cluster_centers_rgb": centers_rgb,
        "cluster_centers_lab": centers_lab,
        "cluster_counts": counts,
        "cluster_percentages": percentages,
        "cluster_brightness": brightness,

        "cluster_labels": labels,
        "cluster_map": cluster_map,
    }