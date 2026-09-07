import cv2
import numpy as np
from skimage.color import deltaE_ciede2000


def rgb_to_lab(rgb):
    """
    Convert an RGB triplet in [0, 255]
    to CIELAB using OpenCV conventions.
    """
    rgb = (
        np.asarray(
            rgb,
            dtype=np.uint8
        )
        .reshape(1, 1, 3)
    )

    lab_cv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )[0, 0]

    L = lab_cv[0] * 100.0 / 255.0
    a = float(lab_cv[1]) - 128.0
    b = float(lab_cv[2]) - 128.0

    return np.array(
        [L, a, b],
        dtype=float
    )


def lab_chroma(L, a, b):
    return float(
        np.sqrt(
            a**2 + b**2
        )
    )


def lab_hue_degrees(a, b):
    hue = np.degrees(
        np.arctan2(
            b,
            a
        )
    )

    return float(
        hue % 360.0
    )


def circular_hue_distance(
    h1,
    h2
):
    diff = abs(
        float(h1)
        - float(h2)
    )

    return float(
        min(
            diff,
            360.0 - diff
        )
    )


def delta_e_2000(
    lab1,
    lab2
):
    lab1 = (
        np.asarray(
            lab1,
            dtype=float
        )
        .reshape(1, 1, 3)
    )

    lab2 = (
        np.asarray(
            lab2,
            dtype=float
        )
        .reshape(1, 1, 3)
    )

    return float(
        deltaE_ciede2000(
            lab1,
            lab2
        )[0, 0]
    )


def palette_entropy(
    proportions
):
    p = np.asarray(
        proportions,
        dtype=float
    )

    p = p[
        p > 0
    ]

    return float(
        -np.sum(
            p * np.log(p)
        )
    )


def palette_diversity(
    labs,
    proportions
):
    labs = np.asarray(
        labs,
        dtype=float
    )

    proportions = np.asarray(
        proportions,
        dtype=float
    )

    total = 0.0

    for i in range(
        len(labs)
    ):
        for j in range(
            i + 1,
            len(labs)
        ):
            total += (
                proportions[i]
                * proportions[j]
                * np.linalg.norm(
                    labs[i]
                    - labs[j]
                )
            )

    return float(total)