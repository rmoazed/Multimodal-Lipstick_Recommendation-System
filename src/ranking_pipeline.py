from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.color_utils import (
    circular_hue_distance,
    delta_e_2000,
    lab_hue_degrees,
)

from src.outfit_processing import (
    OutfitProcessor,
)


class LipstickRecommender:
    def __init__(
        self,
        catalog_path,
        artifact_dir,
    ):
        self.catalog_path = Path(
            catalog_path
        )

        self.artifact_dir = Path(
            artifact_dir
        )

        self.catalog = pd.read_csv(
            self.catalog_path
        )

        self.catalog = (
            self.catalog.rename(
                columns={
                    "L_star": "L",
                    "a_star": "a",
                    "b_star": "b",
                }
            )
        )

        self.scaler = joblib.load(
            self.artifact_dir
            / "v8_scaler.joblib"
        )

        self.model = joblib.load(
            self.artifact_dir
            / "v8_ridge_model.joblib"
        )

        self.features = (
            pd.read_csv(
                self.artifact_dir
                / "v8_feature_spec.csv"
            )["feature"]
            .tolist()
        )

        if len(
            self.features
        ) != 23:
            raise ValueError(
                "Expected 23 frozen V8 features."
            )

        self.outfit_processor = (
            OutfitProcessor()
        )


    def _candidate_features(
        self,
        lipstick,
        palette_summary,
    ):
        lip_L = float(
            lipstick["L"]
        )

        lip_a = float(
            lipstick["a"]
        )

        lip_b = float(
            lipstick["b"]
        )

        lip_chroma = float(
            lipstick["chroma"]
        )

        lip_hue = (
            lab_hue_degrees(
                lip_a,
                lip_b
            )
        )

        lip_lab = np.array(
            [
                lip_L,
                lip_a,
                lip_b,
            ],
            dtype=float,
        )

        rep_L = float(
            palette_summary[
                "representative_L"
            ]
        )

        rep_a = float(
            palette_summary[
                "representative_a"
            ]
        )

        rep_b = float(
            palette_summary[
                "representative_b"
            ]
        )

        rep_chroma = float(
            palette_summary[
                "representative_chroma"
            ]
        )

        rep_hue = float(
            palette_summary[
                "representative_hue"
            ]
        )

        rep_lab = np.array(
            [
                rep_L,
                rep_a,
                rep_b,
            ]
        )

        dom_L = float(
            palette_summary[
                "dominant_L"
            ]
        )

        dom_a = float(
            palette_summary[
                "dominant_a"
            ]
        )

        dom_b = float(
            palette_summary[
                "dominant_b"
            ]
        )

        dom_chroma = float(
            palette_summary[
                "dominant_chroma"
            ]
        )

        dom_hue = float(
            palette_summary[
                "dominant_hue"
            ]
        )

        dom_lab = np.array(
            [
                dom_L,
                dom_a,
                dom_b,
            ]
        )

        palette_lab = np.asarray(
            palette_summary[
                "palette_lab"
            ],
            dtype=float,
        )

        proportions = np.asarray(
            palette_summary[
                "palette_proportions"
            ],
            dtype=float,
        )

        palette_L = (
            palette_lab[:, 0]
        )

        palette_a = (
            palette_lab[:, 1]
        )

        palette_b = (
            palette_lab[:, 2]
        )

        palette_chroma = np.sqrt(
            palette_a**2
            + palette_b**2
        )

        palette_hue = (
            np.degrees(
                np.arctan2(
                    palette_b,
                    palette_a,
                )
            )
            % 360.0
        )

        palette_deltaE = np.array([
            delta_e_2000(
                lip_lab,
                color_lab,
            )
            for color_lab
            in palette_lab
        ])

        closest_idx = int(
            np.argmin(
                palette_deltaE
            )
        )

        weighted_mean_deltaE = float(
            np.average(
                palette_deltaE,
                weights=proportions,
            )
        )

        weighted_std_deltaE = float(
            np.sqrt(
                np.average(
                    (
                        palette_deltaE
                        - weighted_mean_deltaE
                    ) ** 2,
                    weights=proportions,
                )
            )
        )

        hue_distances = np.array([
            circular_hue_distance(
                lip_hue,
                h
            )
            for h
            in palette_hue
        ])

        features = {
            "representative_lightness_contrast":
                abs(
                    lip_L
                    - rep_L
                ),

            "representative_chroma_contrast":
                abs(
                    lip_chroma
                    - rep_chroma
                ),

            "representative_hue_distance":
                circular_hue_distance(
                    lip_hue,
                    rep_hue,
                ),

            "representative_perceptual_distance":
                delta_e_2000(
                    lip_lab,
                    rep_lab,
                ),

            "dominant_lightness_contrast":
                abs(
                    lip_L
                    - dom_L
                ),

            "dominant_chroma_contrast":
                abs(
                    lip_chroma
                    - dom_chroma
                ),

            "dominant_hue_distance":
                circular_hue_distance(
                    lip_hue,
                    dom_hue,
                ),

            "dominant_perceptual_distance":
                delta_e_2000(
                    lip_lab,
                    dom_lab,
                ),

            "min_palette_deltaE":
                float(
                    palette_deltaE.min()
                ),

            "closest_palette_lightness_contrast":
                abs(
                    lip_L
                    - palette_L[
                        closest_idx
                    ]
                ),

            "closest_palette_chroma_contrast":
                abs(
                    lip_chroma
                    - palette_chroma[
                        closest_idx
                    ]
                ),

            "closest_palette_hue_distance":
                circular_hue_distance(
                    lip_hue,
                    palette_hue[
                        closest_idx
                    ],
                ),

            "closest_palette_proportion":
                float(
                    proportions[
                        closest_idx
                    ]
                ),

            "max_palette_deltaE":
                float(
                    palette_deltaE.max()
                ),

            "weighted_mean_deltaE":
                weighted_mean_deltaE,

            "weighted_std_deltaE":
                weighted_std_deltaE,

            "weighted_lightness_contrast":
                float(
                    np.average(
                        np.abs(
                            palette_L
                            - lip_L
                        ),
                        weights=proportions,
                    )
                ),

            "weighted_chroma_contrast":
                float(
                    np.average(
                        np.abs(
                            palette_chroma
                            - lip_chroma
                        ),
                        weights=proportions,
                    )
                ),

            "weighted_hue_distance":
                float(
                    np.average(
                        hue_distances,
                        weights=proportions,
                    )
                ),

            "lipstick_L":
                lip_L,

            "lipstick_a":
                lip_a,

            "lipstick_b":
                lip_b,

            "lipstick_chroma":
                lip_chroma,
        }

        if (
            set(features)
            != set(self.features)
        ):
            raise ValueError(
                "Generated feature set does "
                "not match frozen V8 specification."
            )

        return features


    def _build_feature_table(
        self,
        palette_summary,
    ):
        rows = []

        for _, lipstick in (
            self.catalog.iterrows()
        ):
            rows.append({
                "lipstick_id":
                    lipstick[
                        "lipstick_id"
                    ],
                **self._candidate_features(
                    lipstick,
                    palette_summary,
                ),
            })

        return pd.DataFrame(
            rows
        )


    def _score(
        self,
        feature_table,
    ):
        X = (
            feature_table[
                self.features
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            X
        ).all():
            raise ValueError(
                "Non-finite ranking features detected."
            )

        X_scaled = (
            self.scaler.transform(
                X
            )
        )

        predicted_utility = (
            self.model.predict(
                X_scaled
            )
        )

        scored = (
            feature_table[
                ["lipstick_id"]
            ]
            .copy()
        )

        scored[
            "predicted_utility"
        ] = predicted_utility

        scored = (
            scored.sort_values(
                "predicted_utility",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        scored[
            "predicted_rank"
        ] = np.arange(
            1,
            len(scored) + 1,
        )

        return scored


    def _attach_metadata(
        self,
        scored,
    ):
        return scored.merge(
            self.catalog,
            on="lipstick_id",
            how="left",
        )


    @staticmethod
    def _lipstick_lab(
        row
    ):
        return np.array(
            [
                float(row["L"]),
                float(row["a"]),
                float(row["b"]),
            ],
            dtype=float,
        )


    def _select_diverse(
        self,
        rankings,
        top_k=5,
        delta_e_threshold=12.0,
        search_depth=30,
    ):
        selected = []

        candidates = (
            rankings
            .head(
                search_depth
            )
            .copy()
        )

        for _, candidate in (
            candidates.iterrows()
        ):
            if len(selected) == 0:
                selected.append(
                    candidate
                )
                continue

            candidate_lab = (
                self._lipstick_lab(
                    candidate
                )
            )

            distances = [
                delta_e_2000(
                    candidate_lab,
                    self._lipstick_lab(
                        chosen
                    ),
                )
                for chosen
                in selected
            ]

            if (
                min(distances)
                >= delta_e_threshold
            ):
                selected.append(
                    candidate
                )

            if len(selected) == top_k:
                break

        if len(selected) < top_k:
            selected_ids = {
                row[
                    "lipstick_id"
                ]
                for row
                in selected
            }

            for _, candidate in (
                rankings.iterrows()
            ):
                lipstick_id = (
                    candidate[
                        "lipstick_id"
                    ]
                )

                if (
                    lipstick_id
                    not in selected_ids
                ):
                    selected.append(
                        candidate
                    )

                    selected_ids.add(
                        lipstick_id
                    )

                if len(selected) == top_k:
                    break

        result = pd.DataFrame(
            selected
        ).reset_index(
            drop=True
        )

        result[
            "recommendation_order"
        ] = np.arange(
            1,
            len(result) + 1,
        )

        selected_df = pd.DataFrame(selected)

        selected_df = (
            selected_df
            .sort_values("predicted_rank")
            .reset_index(drop=True)
        )

        return selected_df


    def recommend(
        self,
        image_input,
        top_k=5,
        delta_e_threshold=12.0,
        search_depth=30,
    ):
        processed = (
            self.outfit_processor
            .preprocess(
                image_input
            )
        )

        feature_table = (
            self._build_feature_table(
                processed[
                    "palette_summary"
                ]
            )
        )

        scored = self._score(
            feature_table
        )

        full_rankings = (
            self._attach_metadata(
                scored
            )
        )

        recommendations = (
            self._select_diverse(
                full_rankings,
                top_k=top_k,
                delta_e_threshold=(
                    delta_e_threshold
                ),
                search_depth=(
                    search_depth
                ),
            )
        )

        return {
            "recommendations":
                recommendations,

            "full_rankings":
                full_rankings,

            "processed_outfit":
                processed,

            "candidate_features":
                feature_table,
        }