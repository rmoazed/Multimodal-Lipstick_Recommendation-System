from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

SOURCE_PATH = Path("data/color_diagnostics/verified_source.png")
OUTPUT_DIR = Path("data/color_diagnostics")

# Manually validated ROI for this specific 430 x 465 image
ROI = (240, 90, 300, 170)


# ------------------------------------------------------------
# LOAD VERIFIED SOURCE IMAGE
# ------------------------------------------------------------

image = Image.open(SOURCE_PATH).convert("RGB")

print("Image size:", image.size)

if image.size != (430, 465):
    raise ValueError(
        f"Expected source image size (430, 465), "
        f"but got {image.size}. "
        "Do not reuse this ROI on a different image."
    )


# ------------------------------------------------------------
# CROP ROI
# ------------------------------------------------------------

lipstick_roi = image.crop(ROI)

roi_path = OUTPUT_DIR / "lipstick_roi.png"
lipstick_roi.save(roi_path)

print("\nROI coordinates:", ROI)
print("ROI size:", lipstick_roi.size)
print("Saved ROI:", roi_path)


# ------------------------------------------------------------
# PIXEL STATISTICS
# ------------------------------------------------------------

roi_array = np.array(lipstick_roi)

# (height, width, RGB) -> one row per pixel
pixels = roi_array.reshape(-1, 3)

mean_rgb = pixels.mean(axis=0)
median_rgb = np.median(pixels, axis=0)

print("\nNumber of pixels:", len(pixels))

print("\nMean RGB:")
print(np.round(mean_rgb, 1))

print("\nMedian RGB:")
print(np.round(median_rgb, 1))

print("\nChannel ranges:")
print("R:", pixels[:, 0].min(), "to", pixels[:, 0].max())
print("G:", pixels[:, 1].min(), "to", pixels[:, 1].max())
print("B:", pixels[:, 2].min(), "to", pixels[:, 2].max())

from sklearn.cluster import KMeans


# ------------------------------------------------------------
# K-MEANS PIXEL CLUSTERING
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

cluster_labels = kmeans.fit_predict(pixels)

cluster_centers = kmeans.cluster_centers_

cluster_counts = np.bincount(
    cluster_labels,
    minlength=3,
)

cluster_percentages = (
    cluster_counts / len(pixels) * 100
)

# Sort clusters from darkest to brightest for easier inspection.
brightness = cluster_centers.mean(axis=1)
order = np.argsort(brightness)

print("\nK-MEANS CLUSTERS")
print("----------------")

for rank, cluster_id in enumerate(order, start=1):

    center = cluster_centers[cluster_id]
    count = cluster_counts[cluster_id]
    percentage = cluster_percentages[cluster_id]

    print(
        f"\nCluster {rank} "
        f"(original label {cluster_id})"
    )

    print(
        "Center RGB:",
        np.round(center, 1),
    )

    print(
        f"Pixels: {count} "
        f"({percentage:.1f}%)"
    )

    print(
        f"Mean channel brightness: "
        f"{brightness[cluster_id]:.1f}"
    )

# ------------------------------------------------------------
# SAVE CLUSTER MAP
# ------------------------------------------------------------

height, width, _ = roi_array.shape

cluster_map = cluster_labels.reshape(height, width)

# Re-label clusters by brightness:
# 0 = darkest
# 1 = middle
# 2 = brightest
sorted_cluster_map = np.zeros_like(cluster_map)

for new_label, original_label in enumerate(order):
    sorted_cluster_map[cluster_map == original_label] = new_label

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(lipstick_roi)
axes[0].set_title("Original ROI")
axes[0].axis("off")

axes[1].imshow(
    sorted_cluster_map,
    cmap="viridis",
    vmin=0,
    vmax=2,
)
axes[1].set_title("K-Means Cluster Map")
axes[1].axis("off")

fig.tight_layout()

cluster_map_path = OUTPUT_DIR / "cluster_map.png"
fig.savefig(cluster_map_path, dpi=150)
plt.close(fig)

print("\nSaved cluster map:", cluster_map_path)


# ------------------------------------------------------------
# SAVE CLUSTER COLOR PATCHES
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, cluster_id in zip(axes, order):

    center = cluster_centers[cluster_id]
    percentage = cluster_percentages[cluster_id]

    display_color = np.clip(
        center / 255.0,
        0,
        1,
    )

    ax.imshow([[display_color]])

    ax.set_title(
        f"RGB {np.round(center).astype(int)}\n"
        f"{percentage:.1f}% of ROI"
    )

    ax.axis("off")

fig.tight_layout()

patch_path = OUTPUT_DIR / "cluster_colors.png"
fig.savefig(patch_path, dpi=150)
plt.close(fig)

print("Saved cluster colors:", patch_path)

from skimage.color import rgb2lab, deltaE_ciede2000


# ------------------------------------------------------------
# RGB -> CIELAB
# ------------------------------------------------------------

# For this manually inspected experiment, the darkest K-means
# cluster corresponds spatially to the lipstick material.
dominant_cluster_id = order[0]
dominant_rgb = cluster_centers[dominant_cluster_id]

rgb_estimates = {
    "Mean ROI": mean_rgb,
    "Median ROI": median_rgb,
    "Lipstick cluster": dominant_rgb,
}


def rgb_to_lab(rgb):
    """Convert one sRGB color in 0-255 range to CIELAB."""
    
    rgb = np.asarray(rgb, dtype=float) / 255.0

    # skimage expects image-shaped input.
    rgb_image = rgb.reshape(1, 1, 3)

    lab_image = rgb2lab(rgb_image)

    return lab_image[0, 0]


lab_estimates = {
    name: rgb_to_lab(rgb)
    for name, rgb in rgb_estimates.items()
}


print("\nCIELAB ESTIMATES")
print("----------------")

for name, lab in lab_estimates.items():
    print(
        f"{name:18s}: "
        f"L*={lab[0]:6.2f}, "
        f"a*={lab[1]:6.2f}, "
        f"b*={lab[2]:6.2f}"
    )


# ------------------------------------------------------------
# CHROMA
# ------------------------------------------------------------

print("\nCHROMA")
print("------")

for name, lab in lab_estimates.items():
    chroma = np.sqrt(
        lab[1] ** 2 +
        lab[2] ** 2
    )

    print(
        f"{name:18s}: "
        f"C*={chroma:6.2f}"
    )


# ------------------------------------------------------------
# DELTA E 2000
# ------------------------------------------------------------

lipstick_lab = lab_estimates["Lipstick cluster"]

print("\nDELTA E 2000 VS LIPSTICK CLUSTER")
print("--------------------------------")

for name in ["Mean ROI", "Median ROI"]:

    comparison_lab = lab_estimates[name]

    delta_e = deltaE_ciede2000(
        lipstick_lab.reshape(1, 1, 3),
        comparison_lab.reshape(1, 1, 3),
    )[0, 0]

    print(
        f"{name:18s}: "
        f"Delta E 00 = {delta_e:.2f}"
    )

# ------------------------------------------------------------
# SECOND SOURCE: LIPS
# ------------------------------------------------------------

LIPS_SOURCE_PATH = Path(
    "data/color_sources/rouge_premier_8_2.jpg"
)

lips_image = Image.open(
    LIPS_SOURCE_PATH
).convert("RGB")

print("\nLips image size:", lips_image.size)

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(lips_image)
ax.set_title("Rouge Premier 8 — Lips Source")

# Keep axes visible so we can select ROI coordinates.
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.tight_layout()

lips_reference_path = (
    OUTPUT_DIR / "lips_coordinate_reference.png"
)

fig.savefig(
    lips_reference_path,
    dpi=150,
)

plt.close(fig)

print(
    "Saved lips coordinate reference:",
    lips_reference_path,
)

# ------------------------------------------------------------
# LIPS ROI
# ------------------------------------------------------------

LIPS_ROI = (70, 250, 280, 310)

lips_roi = lips_image.crop(LIPS_ROI)

lips_roi_path = OUTPUT_DIR / "lips_roi.png"
lips_roi.save(lips_roi_path)

print("\nLips ROI coordinates:", LIPS_ROI)
print("Lips ROI size:", lips_roi.size)
print("Saved lips ROI:", lips_roi_path)

# ------------------------------------------------------------
# LIPS PIXEL CLUSTERING
# ------------------------------------------------------------

lips_array = np.array(lips_roi)
lips_pixels = lips_array.reshape(-1, 3)

lips_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

lips_labels = lips_kmeans.fit_predict(lips_pixels)
lips_centers = lips_kmeans.cluster_centers_

lips_counts = np.bincount(
    lips_labels,
    minlength=3,
)

lips_percentages = (
    lips_counts / len(lips_pixels) * 100
)

lips_brightness = lips_centers.mean(axis=1)
lips_order = np.argsort(lips_brightness)

print("\nLIPS K-MEANS CLUSTERS")
print("---------------------")

for rank, cluster_id in enumerate(lips_order, start=1):

    center = lips_centers[cluster_id]

    print(f"\nCluster {rank}")
    print(
        "Center RGB:",
        np.round(center, 1),
    )
    print(
        f"Pixels: {lips_counts[cluster_id]} "
        f"({lips_percentages[cluster_id]:.1f}%)"
    )


# ------------------------------------------------------------
# LIPS CLUSTER MAP
# ------------------------------------------------------------

height, width, _ = lips_array.shape
lips_cluster_map = lips_labels.reshape(height, width)

sorted_lips_map = np.zeros_like(lips_cluster_map)

for new_label, original_label in enumerate(lips_order):
    sorted_lips_map[
        lips_cluster_map == original_label
    ] = new_label


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(lips_roi)
axes[0].set_title("Original Lips ROI")
axes[0].axis("off")

axes[1].imshow(
    sorted_lips_map,
    cmap="viridis",
    vmin=0,
    vmax=2,
)
axes[1].set_title("Lips K-Means Cluster Map")
axes[1].axis("off")

fig.tight_layout()

lips_map_path = (
    OUTPUT_DIR / "lips_cluster_map.png"
)

fig.savefig(
    lips_map_path,
    dpi=150,
)

plt.close(fig)

print(
    "\nSaved lips cluster map:",
    lips_map_path,
)

# ------------------------------------------------------------
# CROSS-IMAGE COMPARISON
# ------------------------------------------------------------

bullet_rgb = dominant_rgb

# From spatial inspection of the lips cluster map,
# the brightest cluster corresponds to the better-lit
# lipstick-covered central lip surface.
lips_lipstick_cluster_id = lips_order[-1]
lips_rgb = lips_centers[lips_lipstick_cluster_id]

bullet_lab = rgb_to_lab(bullet_rgb)
lips_lab = rgb_to_lab(lips_rgb)

cross_image_delta_e = deltaE_ciede2000(
    bullet_lab.reshape(1, 1, 3),
    lips_lab.reshape(1, 1, 3),
)[0, 0]


print("\nCROSS-IMAGE COMPARISON")
print("----------------------")

print(
    "Bullet RGB:",
    np.round(bullet_rgb, 1),
)

print(
    "Lips RGB:  ",
    np.round(lips_rgb, 1),
)

print(
    "\nBullet LAB:",
    np.round(bullet_lab, 2),
)

print(
    "Lips LAB:  ",
    np.round(lips_lab, 2),
)

print(
    f"\nCross-image Delta E 00: "
    f"{cross_image_delta_e:.2f}"
)