from pathlib import Path
from io import BytesIO
import json
import os

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from src.database import LipstickDatabase
from src.ranking_pipeline import LipstickRecommender


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lipstick Recommender",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>

/* ----------------------------------------------------------
   REMOVE STREAMLIT TOP BAR / HEADER
---------------------------------------------------------- */

[data-testid="stHeader"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stDecoration"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
    height: 0;
}

.block-container {
    max-width: 1320px;
    padding-top: 2rem !important;
    padding-bottom: 4rem;
}


/* ----------------------------------------------------------
   GLOBAL
---------------------------------------------------------- */

.stApp {
    background-color: #FBF8F6;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

p {
    line-height: 1.65;
    color: #54464A;
}


/* ----------------------------------------------------------
   HEADINGS
---------------------------------------------------------- */

h1, h2, h3 {
    font-family: Georgia, "Times New Roman", serif;
    color: #40252D;
}

h1 {
    font-size: 3.1rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.035em;
    line-height: 1.05 !important;
}

h2,
h3 {
    font-weight: 500 !important;
}


/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

[data-testid="stSidebar"] {
    background-color: #F2E9E8;
    border-right: 1px solid #E5D7D8;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #5A2938;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.stButton > button {
    border-radius: 999px !important;
    padding: 0.62rem 1.3rem !important;
    font-weight: 600 !important;
    transition: 0.2s ease !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background-color: #7C3048 !important;
    border: 1px solid #7C3048 !important;
    color: #FFFFFF !important;
}

/* Force all nested primary-button text to white */
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #FFFFFF !important;
}

/* Hover */
.stButton > button[kind="primary"]:hover {
    background-color: #67263B !important;
    border-color: #67263B !important;
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover *,
.stButton > button[kind="primary"]:hover p,
.stButton > button[kind="primary"]:hover span {
    color: #FFFFFF !important;
}

/* Focus / active */
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active {
    background-color: #67263B !important;
    border-color: #67263B !important;
    color: #FFFFFF !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #6C3344 !important;
    border: 1px solid #DCC7CD !important;
}

.stButton > button[kind="secondary"] * {
    color: #6C3344 !important;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #F8EFF1 !important;
    color: #5C283A !important;
    border-color: #CDAFB8 !important;
}


/* ----------------------------------------------------------
   FILE UPLOADER
---------------------------------------------------------- */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    padding: 1rem;
    border-radius: 18px;
    border: 1px solid #E7DADC;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #FCF8F7;
    border-radius: 14px;
    border: 1px dashed #D8BCC5;
}


/* ----------------------------------------------------------
   BORDERED CONTAINERS / CARDS
---------------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255, 255, 255, 0.96);
    border: 1px solid #E8DCDD !important;
    border-radius: 20px !important;
    box-shadow: 0 5px 22px rgba(91, 50, 62, 0.06);
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 8px 28px rgba(91, 50, 62, 0.10);
}


/* ----------------------------------------------------------
   IMAGES
---------------------------------------------------------- */

[data-testid="stImage"] img {
    border-radius: 14px;
}


/* ----------------------------------------------------------
   METRICS
---------------------------------------------------------- */

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E8DDDC;
    padding: 1rem 1.2rem;
    border-radius: 16px;
}


/* ----------------------------------------------------------
   EXPANDERS
---------------------------------------------------------- */

[data-testid="stExpander"] {
    background-color: white;
    border-radius: 16px;
    border: 1px solid #E8DDDC;
    overflow: hidden;
}


/* ----------------------------------------------------------
   DATAFRAME
---------------------------------------------------------- */

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #E8DDDC;
}


/* ----------------------------------------------------------
   CUSTOM TYPOGRAPHY
---------------------------------------------------------- */

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.74rem;
    font-weight: 700;
    color: #A05A6F;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    max-width: 720px;
    font-size: 1.08rem;
    line-height: 1.7;
    color: #6E5B61;
    margin-top: -0.3rem;
    margin-bottom: 1.6rem;
}

.section-intro {
    font-size: 1.02rem;
    color: #6F5D62;
    max-width: 850px;
    margin-bottom: 1.25rem;
}

.rank-badge {
    display: inline-block;
    background-color: #F3E4E8;
    color: #7C3048;
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.product-name {
    font-family: Georgia, "Times New Roman", serif;
    color: #40252D;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.28;
    margin-top: 0.6rem;
    margin-bottom: 0.2rem;
}

.shade-name {
    color: #8A4A5D;
    font-size: 0.93rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}

.descriptor {
    color: #927C82;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.85rem;
}

.why-label {
    color: #7C3048;
    font-weight: 700;
    font-size: 0.82rem;
    margin-top: 0.7rem;
    margin-bottom: 0.15rem;
}

.why-text {
    color: #5B4A4F;
    font-size: 0.92rem;
    line-height: 1.55;
}

.model-meta {
    border-top: 1px solid #EEE4E5;
    padding-top: 0.65rem;
    margin-top: 0.85rem;
    font-size: 0.75rem;
    color: #9A858A;
}

</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CATALOG_PATH = (
    PROJECT_ROOT
    / "lipstick_catalog_with_colors.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "model_artifacts"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "lipstick_recommender.db"
)

SWATCH_DIR = (
    PROJECT_ROOT
    / "lipstick_images"
    / "ranking_swatches"
)


# ============================================================
# BACKEND
# ============================================================

@st.cache_resource
def load_recommender():

    return LipstickRecommender(
        catalog_path=CATALOG_PATH,
        artifact_dir=ARTIFACT_DIR,
    )


@st.cache_resource
def load_database():

    db = LipstickDatabase(
        DATABASE_PATH
    )

    db.seed_from_csv(
        CATALOG_PATH
    )

    return db


@st.cache_resource
def load_openai_client():

    api_key = None

    # Streamlit Cloud
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    # Local .env / environment variable fallback
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key
    )


recommender = load_recommender()
db = load_database()
openai_client = load_openai_client()


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_text(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return None

    return text


def get_swatch_path(
    lipstick_id,
):

    return (
        SWATCH_DIR
        / f"{lipstick_id}.png"
    )


def lipstick_display_name(row):

    brand = safe_text(
        row.get("brand")
    )

    product = safe_text(
        row.get("product_name")
    )

    shade = safe_text(
        row.get("shade_name")
    )

    parts = [
        value
        for value in [
            brand,
            product,
        ]
        if value
    ]

    if parts:

        name = " ".join(
            parts
        )

    else:

        name = str(
            row["lipstick_id"]
        )

    if shade:

        name += (
            f" — {shade}"
        )

    return name


def lipstick_product_title(row):

    brand = safe_text(
        row.get("brand")
    )

    product = safe_text(
        row.get("product_name")
    )

    parts = [
        value
        for value in [
            brand,
            product,
        ]
        if value
    ]

    if parts:

        return " ".join(
            parts
        )

    return str(
        row["lipstick_id"]
    )


# ============================================================
# LLM EXPLANATIONS
# ============================================================

def clean_for_json(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(
        value,
        "item",
    ):

        try:
            return value.item()
        except Exception:
            pass

    return value


def make_outfit_summary(result):

    processed = result.get(
        "processed_outfit"
    )

    summary = {}

    if isinstance(
        processed,
        dict,
    ):

        useful_keys = [
            "representative_L",
            "representative_a",
            "representative_b",
            "representative_chroma",
            "representative_hue",
            "weighted_palette_L",
            "weighted_palette_chroma",
            "lightness_range",
            "chroma_range",
            "dominant_proportion",
            "palette_entropy",
            "palette_diversity",
        ]

        for key in useful_keys:

            if key in processed:

                summary[key] = (
                    clean_for_json(
                        processed[key]
                    )
                )

        for possible_key in [
            "palette_summary",
            "summary",
            "outfit_summary",
        ]:

            nested = processed.get(
                possible_key
            )

            if isinstance(
                nested,
                dict,
            ):

                for key, value in (
                    nested.items()
                ):

                    if (
                        isinstance(
                            value,
                            (
                                str,
                                int,
                                float,
                                bool,
                            ),
                        )
                        or value is None
                        or hasattr(
                            value,
                            "item",
                        )
                    ):

                        summary[key] = (
                            clean_for_json(
                                value
                            )
                        )

    return summary


def generate_llm_explanations(
    recommendations: pd.DataFrame,
    outfit_summary: dict,
) -> dict:

    if openai_client is None:

        return {}

    lipstick_descriptions = []

    for _, row in (
        recommendations.iterrows()
    ):

        lipstick_descriptions.append(
            {
                "lipstick_id": str(
                    row[
                        "lipstick_id"
                    ]
                ),

                "brand": safe_text(
                    row.get(
                        "brand"
                    )
                ),

                "product_name": safe_text(
                    row.get(
                        "product_name"
                    )
                ),

                "shade_name": safe_text(
                    row.get(
                        "shade_name"
                    )
                ),

                "finish": safe_text(
                    row.get(
                        "finish"
                    )
                ),

                "color_family": safe_text(
                    row.get(
                        "color_family"
                    )
                ),

                "model_rank": int(
                    row[
                        "predicted_rank"
                    ]
                ),

                "lightness_contrast": (
                    clean_for_json(
                        row.get(
                            "representative_lightness_contrast"
                        )
                    )
                ),

                "chroma_contrast": (
                    clean_for_json(
                        row.get(
                            "representative_chroma_contrast"
                        )
                    )
                ),

                "hue_distance": (
                    clean_for_json(
                        row.get(
                            "representative_hue_distance"
                        )
                    )
                ),

                "perceptual_distance": (
                    clean_for_json(
                        row.get(
                            "representative_perceptual_distance"
                        )
                    )
                ),
            }
        )

    prompt = f"""
You are the explanation layer for a lipstick recommendation app.

A separate machine-learning ranking system has ALREADY selected these
lipsticks for the uploaded outfit.

You are NOT choosing or reranking the lipsticks.
You are only writing appealing explanations for recommendations that
already exist.

OUTFIT COLOR INFORMATION:

{json.dumps(outfit_summary, indent=2, default=str)}

RECOMMENDED LIPSTICKS:

{json.dumps(lipstick_descriptions, indent=2, default=str)}

Write one short explanation for each lipstick.

Requirements:

- Use 1 or 2 concise sentences per lipstick.
- Sound like a knowledgeable, friendly beauty stylist.
- Explain how the lipstick visually relates to the outfit.
- Make all five explanations meaningfully different.
- Discuss harmony, contrast, depth, softness, brightness,
  richness, boldness, subtlety, or balance when appropriate.
- Use the lipstick's color family and finish when helpful.
- Do not mention numerical feature values.
- Do not mention machine learning, model scores, rankings,
  predicted utility, or algorithms.
- Do not invent the user's skin tone.
- Do not invent the occasion.
- Do not invent product properties that were not provided.
- Do not say a specific feature caused the recommendation.
- Avoid repeating the same sentence structure.

Return ONLY valid JSON.

The JSON must map each lipstick_id to its explanation.

Example:

{{
    "L0001": "This rich red creates a vivid contrast...",
    "L0002": "This softer shade keeps the palette..."
}}
"""

    try:

        response = (
            openai_client.responses.create(
                model="gpt-5-mini",
                input=prompt,
            )
        )

        text = (
            response.output_text
            .strip()
        )

        if text.startswith(
            "```json"
        ):

            text = text[
                len("```json"):
            ]

        elif text.startswith(
            "```"
        ):

            text = text[
                len("```"):
            ]

        if text.endswith(
            "```"
        ):

            text = text[:-3]

        text = text.strip()

        explanations = (
            json.loads(
                text
            )
        )

        if not isinstance(
            explanations,
            dict,
        ):

            return {}

        return explanations

    except Exception as error:

        st.error(
            f"Could not generate AI explanations: {error}"
        )

        return {}


# ============================================================
# RECOMMENDATION CARD
# ============================================================

def display_lipstick_card(
    row,
    position,
):

    lipstick_id = str(
        row[
            "lipstick_id"
        ]
    )

    product_title = (
        lipstick_product_title(
            row
        )
    )

    shade = safe_text(
        row.get(
            "shade_name"
        )
    )

    finish = safe_text(
        row.get(
            "finish"
        )
    )

    color_family = safe_text(
        row.get(
            "color_family"
        )
    )

    explanation = safe_text(
        row.get(
            "llm_explanation"
        )
    )

    swatch_path = (
        get_swatch_path(
            lipstick_id
        )
    )

    with st.container(
        border=True
    ):

        st.markdown(
            f'<div class="rank-badge">Recommendation #{position}</div>',
            unsafe_allow_html=True,
        )

        if swatch_path.exists():

            st.image(
                str(
                    swatch_path
                ),
                use_container_width=True,
            )

        st.markdown(
            f'<div class="product-name">{product_title}</div>',
            unsafe_allow_html=True,
        )

        if shade:

            st.markdown(
                f'<div class="shade-name">{shade}</div>',
                unsafe_allow_html=True,
            )

        descriptors = [
            value.title()
            for value in [
                finish,
                color_family,
            ]
            if value
        ]

        if descriptors:

            descriptor_text = (
                " · ".join(
                    descriptors
                )
            )

            st.markdown(
                f'<div class="descriptor">{descriptor_text}</div>',
                unsafe_allow_html=True,
            )

        if explanation:

            st.markdown(
                '<div class="why-label">Why it works</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="why-text">{explanation}</div>',
                unsafe_allow_html=True,
            )

        model_rank = int(
            row[
                "predicted_rank"
            ]
        )

        st.markdown(
            (
                '<div class="model-meta">'
                f'Model rank #{model_rank}'
                '&nbsp; · &nbsp;'
                f'{lipstick_id}'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
## 💄 Lipstick Match

Find shades from your own collection that complement what you're wearing.
"""
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Find a Match",
        "My Collection",
        "History",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Multimodal outfit analysis + "
    "color-aware lipstick ranking."
)


# ============================================================
# PAGE 1 — FIND A MATCH
# ============================================================

if page == "Find a Match":

    st.markdown(
        '<div class="eyebrow">Outfit-aware recommendations</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "Find your shade."
    )

    st.markdown(
        (
            '<div class="hero-subtitle">'
            'Upload an outfit and discover which lipsticks '
            'in your collection pair best with its color palette. '
            'The recommender balances model preference with variety, '
            'so you get several genuinely different directions '
            'to choose from.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if openai_client is None:

        st.info(
            "AI-written explanations are unavailable "
            "because OPENAI_API_KEY is not set. "
            "The recommender itself will still work."
        )

    # --------------------------------------------------------
    # UPLOAD IMAGE
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload an outfit photo",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="outfit_uploader",
    )

    if uploaded_file is not None:

        new_image_bytes = (
            uploaded_file.getvalue()
        )

        old_image_bytes = (
            st.session_state.get(
                "outfit_image_bytes"
            )
        )

        if (
            old_image_bytes
            != new_image_bytes
        ):

            st.session_state[
                "outfit_image_bytes"
            ] = new_image_bytes

            st.session_state.pop(
                "recommendations",
                None,
            )

            st.session_state.pop(
                "session_id",
                None,
            )

    # --------------------------------------------------------
    # RESTORE PERSISTENT IMAGE
    # --------------------------------------------------------

    outfit_image = None

    if (
        "outfit_image_bytes"
        in st.session_state
    ):

        outfit_image = (
            Image.open(
                BytesIO(
                    st.session_state[
                        "outfit_image_bytes"
                    ]
                )
            )
            .convert(
                "RGB"
            )
        )

    # --------------------------------------------------------
    # IMAGE + ACTION AREA
    # --------------------------------------------------------

    if outfit_image is not None:

        image_col, action_col = (
            st.columns(
                [
                    1.15,
                    1,
                ],
                gap="large",
            )
        )

        with image_col:

            st.image(
                outfit_image,
                caption="Your outfit",
                use_container_width=True,
            )

        with action_col:

            # Native Streamlit container instead of
            # multiline HTML. This prevents Markdown
            # from rendering our HTML as a code block.
            with st.container(
                border=True
            ):

                st.markdown(
                    '<div class="eyebrow">Ready when you are</div>',
                    unsafe_allow_html=True,
                )

                st.subheader(
                    "Let's find the best matches."
                )

                st.write(
                    "The app identifies the clothing region, "
                    "extracts its color palette, scores your "
                    "lipstick collection, and selects a varied "
                    "set of high-ranking options."
                )

            recommend_button = (
                st.button(
                    "Find My Lipsticks",
                    type="primary",
                    use_container_width=True,
                )
            )

            clear_button = (
                st.button(
                    "Clear outfit & start over",
                    use_container_width=True,
                )
            )

            if clear_button:

                st.session_state.pop(
                    "outfit_image_bytes",
                    None,
                )

                st.session_state.pop(
                    "recommendations",
                    None,
                )

                st.session_state.pop(
                    "session_id",
                    None,
                )

                st.rerun()

        # ----------------------------------------------------
        # RUN RECOMMENDER
        # ----------------------------------------------------

        if recommend_button:

            with st.spinner(
                "Analyzing your outfit..."
            ):

                result = (
                    recommender.recommend(
                        outfit_image,
                        top_k=5,
                    )
                )

                recommendations = (
                    result[
                        "recommendations"
                    ]
                    .sort_values(
                        "predicted_rank"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                # --------------------------------------------
                # PRODUCT METADATA
                # --------------------------------------------

                catalog_metadata = (
                    db.list_lipsticks(
                        active_only=True
                    )[
                        [
                            "lipstick_id",
                            "brand",
                            "product_name",
                            "shade_name",
                            "finish",
                            "color_family",
                        ]
                    ]
                )

                metadata_columns = [
                    "brand",
                    "product_name",
                    "shade_name",
                    "finish",
                    "color_family",
                ]

                recommendations = (
                    recommendations.drop(
                        columns=[
                            column
                            for column
                            in metadata_columns
                            if column
                            in recommendations.columns
                        ],
                        errors="ignore",
                    )
                )

                recommendations = (
                    recommendations.merge(
                        catalog_metadata,
                        on="lipstick_id",
                        how="left",
                    )
                )

                # --------------------------------------------
                # LLM EXPLANATIONS
                # --------------------------------------------

                outfit_summary = (
                    make_outfit_summary(
                        result
                    )
                )

                explanations = (
                    generate_llm_explanations(
                        recommendations,
                        outfit_summary,
                    )
                )

                recommendations[
                    "llm_explanation"
                ] = (
                    recommendations[
                        "lipstick_id"
                    ]
                    .map(
                        explanations
                    )
                )

                # --------------------------------------------
                # SAVE SESSION
                # --------------------------------------------

                session_id = (
                    db.save_recommendation_session(
                        recommendations
                    )
                )

                st.session_state[
                    "recommendations"
                ] = recommendations

                st.session_state[
                    "session_id"
                ] = session_id

            st.success(
                "Your matches are ready ✨"
            )

    else:

        with st.container(
            border=True
        ):

            st.markdown(
                '<div class="eyebrow">Start here</div>',
                unsafe_allow_html=True,
            )

            st.subheader(
                "Upload an outfit photo."
            )

            st.write(
                "Once you upload one, it will stay here "
                "while you browse the rest of the app."
            )

    # ========================================================
    # DISPLAY RECOMMENDATIONS
    # ========================================================

    if (
        "recommendations"
        in st.session_state
    ):

        recommendations = (
            st.session_state[
                "recommendations"
            ]
        )

        session_id = (
            st.session_state[
                "session_id"
            ]
        )

        st.divider()

        st.markdown(
            '<div class="eyebrow">Your matches</div>',
            unsafe_allow_html=True,
        )

        st.header(
            "Five ways to finish the look"
        )

        st.markdown(
            (
                '<div class="section-intro">'
                "These aren't simply the five highest raw scores. "
                "The final set also preserves color diversity, "
                "giving you several distinct directions rather "
                "than a row of nearly identical shades."
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        recommendation_columns = (
            st.columns(
                len(
                    recommendations
                ),
                gap="medium",
            )
        )

        for position, (
            column,
            (_, row),
        ) in enumerate(
            zip(
                recommendation_columns,
                recommendations.iterrows(),
            ),
            start=1,
        ):

            with column:

                display_lipstick_card(
                    row,
                    position,
                )

        # ----------------------------------------------------
        # FEEDBACK
        # ----------------------------------------------------

        st.divider()

        feedback_left, feedback_right = (
            st.columns(
                [
                    1.2,
                    1,
                ],
                gap="large",
            )
        )

        with feedback_left:

            st.subheader(
                "Which one did you choose?"
            )

            st.write(
                "Save your choice and reaction so "
                "the recommendation history can "
                "remember what you actually wore."
            )

        with feedback_right:

            option_lookup = {}

            for _, row in (
                recommendations.iterrows()
            ):

                label = (
                    lipstick_display_name(
                        row
                    )
                )

                option_lookup[
                    label
                ] = row[
                    "lipstick_id"
                ]

            selected_label = (
                st.selectbox(
                    "Selected lipstick",
                    options=list(
                        option_lookup.keys()
                    ),
                    index=None,
                    placeholder=(
                        "Choose a lipstick..."
                    ),
                )
            )

            if (
                selected_label
                is not None
            ):

                selected_lipstick = (
                    option_lookup[
                        selected_label
                    ]
                )

                feedback_label = (
                    st.radio(
                        "How did you like it?",
                        [
                            "Liked it",
                            "Neutral",
                            "Didn't like it",
                        ],
                        horizontal=True,
                    )
                )

                feedback_map = {
                    "Liked it": 1,
                    "Neutral": 0,
                    "Didn't like it": -1,
                }

                if st.button(
                    "Save Choice",
                    type="primary",
                    use_container_width=True,
                ):

                    db.mark_selected(
                        session_id,
                        selected_lipstick,
                    )

                    db.save_feedback(
                        session_id,
                        selected_lipstick,
                        feedback_map[
                            feedback_label
                        ],
                    )

                    st.success(
                        "Saved!"
                    )


# ============================================================
# PAGE 2 — MY COLLECTION
# ============================================================

elif page == "My Collection":

    st.markdown(
        '<div class="eyebrow">Persistent lipstick library</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "My Collection"
    )

    st.markdown(
        (
            '<div class="hero-subtitle">'
            "This collection is the source of truth for the "
            "recommender. Every match is selected from the "
            "lipsticks stored here."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    lipsticks = (
        db.list_lipsticks(
            active_only=True
        )
    )

    metric_left, metric_right = (
        st.columns(
            [
                1,
                3,
            ]
        )
    )

    with metric_left:

        st.metric(
            "Lipsticks",
            len(
                lipsticks
            ),
        )

    if lipsticks.empty:

        st.info(
            "There are no active lipsticks "
            "in your collection."
        )

    else:

        collection = (
            lipsticks[
                [
                    "brand",
                    "product_name",
                    "shade_name",
                    "finish",
                    "color_family",
                    "lipstick_id",
                ]
            ]
            .copy()
        )

        collection.columns = [
            "Brand",
            "Product",
            "Shade",
            "Finish",
            "Color Family",
            "Catalog ID",
        ]

        st.dataframe(
            collection,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 3 — HISTORY
# ============================================================

elif page == "History":

    st.markdown(
        '<div class="eyebrow">Recommendation memory</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "History"
    )

    st.markdown(
        (
            '<div class="hero-subtitle">'
            "Look back at past recommendation sessions, "
            "including the shade you selected and how you "
            "felt about it."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    history = (
        db.get_history()
    )

    if history.empty:

        st.info(
            "No recommendation sessions "
            "have been saved yet."
        )

    else:

        session_ids = (
            history[
                "session_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        st.metric(
            "Saved recommendation sessions",
            len(
                session_ids
            ),
        )

        for session_id in (
            session_ids
        ):

            session = (
                history[
                    history[
                        "session_id"
                    ]
                    == session_id
                ]
                .copy()
            )

            created_at = (
                session[
                    "created_at"
                ].iloc[0]
            )

            with st.expander(
                f"Session {session_id}  ·  {created_at}"
            ):

                for _, row in (
                    session.iterrows()
                ):

                    selected = (
                        row[
                            "selected"
                        ]
                        == 1
                    )

                    name = (
                        lipstick_display_name(
                            row
                        )
                    )

                    if selected:

                        st.markdown(
                            f"### 💄 {name}"
                        )

                        st.caption(
                            "Selected"
                        )

                    else:

                        st.markdown(
                            f"**{name}**"
                        )

                    details = []

                    finish = safe_text(
                        row.get(
                            "finish"
                        )
                    )

                    family = safe_text(
                        row.get(
                            "color_family"
                        )
                    )

                    if finish:

                        details.append(
                            finish.title()
                        )

                    if family:

                        details.append(
                            family.title()
                        )

                    details.append(
                        f"Model rank "
                        f"#{int(row['predicted_rank'])}"
                    )

                    st.caption(
                        " · ".join(
                            details
                        )
                    )

                    if pd.notna(
                        row[
                            "feedback"
                        ]
                    ):

                        feedback_text = {
                            1: "Liked",
                            0: "Neutral",
                            -1: "Disliked",
                        }[
                            int(
                                row[
                                    "feedback"
                                ]
                            )
                        ]

                        st.caption(
                            f"Feedback: "
                            f"{feedback_text}"
                        )

                    st.markdown("---")