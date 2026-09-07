import numpy as np
import torch

from PIL import Image
from sklearn.cluster import KMeans
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from src.color_utils import (
    rgb_to_lab,
    lab_chroma,
    lab_hue_degrees,
    palette_diversity,
    palette_entropy,
)


SEGMENTATION_MODEL_NAME = (
    "mattmdjaga/segformer_b2_clothes"
)

CLOTHING_CLASSES = {
    "upper-clothes",
    "dress",
    "pants",
    "skirt",
    "coat",
    "jumpsuits",
}


class OutfitProcessor:
    def __init__(
        self,
        model_name=SEGMENTATION_MODEL_NAME,
    ):
        self.processor = (
            SegformerImageProcessor
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                model_name
            )
        )

        self.model.eval()

        self.id2label = {
            int(k): v
            for k, v
            in self.model.config.id2label.items()
        }

        label2id = {
            label.lower(): idx
            for idx, label
            in self.id2label.items()
        }

        self.clothing_class_ids = [
            label2id[label]
            for label
            in CLOTHING_CLASSES
            if label in label2id
        ]


    def segment_clothing(
        self,
        image
    ):
        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.open(
                image
            ).convert("RGB")

        else:
            image = image.convert(
                "RGB"
            )

        inputs = (
            self.processor(
                images=image,
                return_tensors="pt",
            )
        )

        with torch.no_grad():
            outputs = self.model(
                **inputs
            )

        logits = outputs.logits

        upsampled_logits = (
            torch.nn.functional.interpolate(
                logits,
                size=(
                    image.height,
                    image.width,
                ),
                mode="bilinear",
                align_corners=False,
            )
        )

        segmentation = (
            upsampled_logits
            .argmax(dim=1)[0]
            .cpu()
            .numpy()
        )

        clothing_mask = np.isin(
            segmentation,
            self.clothing_class_ids,
        )

        return (
            clothing_mask,
            segmentation,
        )


    def extract_clothing_palette(
        self,
        image,
        clothing_mask,
        n_colors=5,
        random_state=42,
    ):
        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.open(
                image
            ).convert("RGB")

        image_np = np.asarray(
            image
        )

        pixels = image_np[
            clothing_mask
        ]

        if len(pixels) == 0:
            raise ValueError(
                "No clothing pixels detected."
            )

        pixels = pixels.astype(
            np.float32
        )

        kmeans = KMeans(
            n_clusters=n_colors,
            random_state=random_state,
            n_init=10,
        )

        labels = (
            kmeans.fit_predict(
                pixels
            )
        )

        centers = (
            kmeans.cluster_centers_
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )

        counts = np.bincount(
            labels,
            minlength=n_colors,
        )

        proportions = (
            counts
            / counts.sum()
        )

        order = np.argsort(
            proportions
        )[::-1]

        centers = centers[
            order
        ]

        proportions = proportions[
            order
        ]

        return (
            centers,
            proportions,
        )


    def summarize_palette(
        self,
        palette_rgb,
        proportions,
    ):
        palette_rgb = np.asarray(
            palette_rgb,
            dtype=float
        )

        proportions = np.asarray(
            proportions,
            dtype=float
        )

        proportions = (
            proportions
            / proportions.sum()
        )

        palette_lab = np.array([
            rgb_to_lab(rgb)
            for rgb
            in palette_rgb
        ])

        L = palette_lab[:, 0]
        a = palette_lab[:, 1]
        b = palette_lab[:, 2]

        chroma = np.sqrt(
            a**2 + b**2
        )

        hue = (
            np.degrees(
                np.arctan2(
                    b,
                    a
                )
            )
            % 360.0
        )

        representative_lab = np.average(
            palette_lab,
            axis=0,
            weights=proportions,
        )

        rep_L = representative_lab[0]
        rep_a = representative_lab[1]
        rep_b = representative_lab[2]

        dominant_idx = int(
            np.argmax(
                proportions
            )
        )

        return {
            "palette_rgb":
                palette_rgb,

            "palette_lab":
                palette_lab,

            "palette_proportions":
                proportions,

            "representative_L":
                float(rep_L),

            "representative_a":
                float(rep_a),

            "representative_b":
                float(rep_b),

            "representative_chroma":
                lab_chroma(
                    rep_L,
                    rep_a,
                    rep_b,
                ),

            "representative_hue":
                lab_hue_degrees(
                    rep_a,
                    rep_b,
                ),

            "weighted_palette_L":
                float(
                    np.average(
                        L,
                        weights=proportions,
                    )
                ),

            "weighted_palette_chroma":
                float(
                    np.average(
                        chroma,
                        weights=proportions,
                    )
                ),

            "L_range":
                float(
                    L.max()
                    - L.min()
                ),

            "chroma_range":
                float(
                    chroma.max()
                    - chroma.min()
                ),

            "dominant_proportion":
                float(
                    proportions[
                        dominant_idx
                    ]
                ),

            "palette_entropy":
                palette_entropy(
                    proportions
                ),

            "palette_diversity":
                palette_diversity(
                    palette_lab,
                    proportions,
                ),

            "dominant_L":
                float(
                    L[
                        dominant_idx
                    ]
                ),

            "dominant_a":
                float(
                    a[
                        dominant_idx
                    ]
                ),

            "dominant_b":
                float(
                    b[
                        dominant_idx
                    ]
                ),

            "dominant_chroma":
                float(
                    chroma[
                        dominant_idx
                    ]
                ),

            "dominant_hue":
                float(
                    hue[
                        dominant_idx
                    ]
                ),
        }


    def preprocess(
        self,
        image_input,
    ):
        if isinstance(
            image_input,
            Image.Image
        ):
            image = image_input.convert(
                "RGB"
            )

        else:
            image = Image.open(
                image_input
            ).convert(
                "RGB"
            )

        (
            clothing_mask,
            segmentation,
        ) = self.segment_clothing(
            image
        )

        if clothing_mask.sum() == 0:
            raise ValueError(
                "No clothing region detected."
            )

        (
            palette_rgb,
            palette_proportions,
        ) = self.extract_clothing_palette(
            image,
            clothing_mask,
        )

        palette_summary = (
            self.summarize_palette(
                palette_rgb,
                palette_proportions,
            )
        )

        return {
            "image":
                image,

            "clothing_mask":
                clothing_mask,

            "segmentation":
                segmentation,

            "palette_rgb":
                palette_rgb,

            "palette_proportions":
                palette_proportions,

            "palette_summary":
                palette_summary,
        }