import os
import joblib
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered"
)


# =========================================================
# MODEL PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logistic Regression / TF-IDF files are stored in GitHub
MODEL_FOLDER = os.path.join(BASE_DIR, "fake_news_hybrid")

# BERT model is stored on Hugging Face
BERT_MODEL = "Reemalz/fake-news"

MAX_LENGTH = 256


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    # Load tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # Load trained BERT model from Hugging Face
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL
    )

    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    bert_model.to(device)
    bert_model.eval()

    # Load TF-IDF vectorizer
    tfidf_vectorizer = joblib.load(
        os.path.join(
            MODEL_FOLDER,
            "tfidf_vectorizer.joblib"
        )
    )

    # Load Logistic Regression model
    lr_model = joblib.load(
        os.path.join(
            MODEL_FOLDER,
            "logistic_regression.joblib"
        )
    )

    # Load ensemble configuration
    config = joblib.load(
        os.path.join(
            MODEL_FOLDER,
            "ensemble_config.joblib"
        )
    )

    return (
        tokenizer,
        bert_model,
        tfidf_vectorizer,
        lr_model,
        config,
        device
    )


(
    tokenizer,
    bert_model,
    tfidf_vectorizer,
    lr_model,
    config,
    device
) = load_models()


# Ensemble weights
BERT_WEIGHT = config.get("bert_weight", 0.5)
LR_WEIGHT = config.get("lr_weight", 0.5)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_news(title, article):

    # -----------------------------
    # BERT prediction
    # -----------------------------

    inputs = tokenizer(
        title,
        article,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = bert_model(**inputs)

        bert_probs = torch.softmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()[0]


    # -----------------------------
    # Logistic Regression prediction
    # -----------------------------

    combined_text = title + " " + article

    tfidf_features = tfidf_vectorizer.transform(
        [combined_text]
    )

    lr_probs = lr_model.predict_proba(
        tfidf_features
    )[0]


    # -----------------------------
    # Hybrid Ensemble
    # -----------------------------

    final_probs = (
        BERT_WEIGHT * bert_probs
        +
        LR_WEIGHT * lr_probs
    )

    prediction = int(
        np.argmax(final_probs)
    )

    confidence = float(
        final_probs[prediction]
    ) * 100


    # IMPORTANT:
    # This is the mapping that worked
    # in your previous testing.
    if prediction == 0:
        result = "REAL"
    else:
        result = "FAKE"

    return result, confidence


# =========================================================
# WEBSITE DESIGN
# =========================================================

st.markdown(
    """
<style>

/* Background */

.stApp {
    background:
        radial-gradient(
            circle at 50% 0%,
            #142143 0%,
            #0a1122 45%,
            #050914 100%
        );
}


/* Main container */

.block-container {
    max-width: 760px;
    padding-top: 5rem;
    padding-bottom: 5rem;
}


/* Hide Streamlit elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none;
}


/* Labels */

label {
    color: #d7deea !important;
    font-weight: 500 !important;
}


/* Title input */

div[data-testid="stTextInput"] input {

    background-color: #0d1525;
    color: white;

    border: 1px solid #27344c;
    border-radius: 7px;
}


/* Article text */

div[data-testid="stTextArea"] textarea {

    background-color: #0d1525;
    color: white;

    border: 1px solid #27344c;
    border-radius: 7px;

    min-height: 190px;
}


/* Placeholder */

input::placeholder,
textarea::placeholder {

    color: #68758a !important;
}


/* Analyze button */

div.stButton > button {

    background:
        linear-gradient(
            90deg,
            #6557e8,
            #3788ff
        );

    color: white;

    border: none;
    border-radius: 7px;

    padding: 9px 22px;

    font-size: 14px;
    font-weight: 600;
}


div.stButton > button:hover {

    color: white;
    border: none;

    box-shadow:
        0px 5px 20px
        rgba(70,100,255,0.35);
}


div.stButton > button:focus {

    color: white;
    border: none;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

header_html = (

    '<div style="'
    'display:flex; '
    'align-items:center; '
    'gap:12px; '
    'margin-bottom:28px;">'

    '<div style="'
    'width:42px; '
    'height:42px; '
    'min-width:42px; '
    'border-radius:9px; '
    'background:linear-gradient(135deg,#3487ff,#7257f5); '
    'display:flex; '
    'align-items:center; '
    'justify-content:center; '
    'font-size:20px; '
    'color:white;">'

    '✦'

    '</div>'

    '<div>'

    '<div style="'
    'color:#f0f4ff; '
    'font-size:20px; '
    'font-weight:650;">'

    'Fake News Detector'

    '</div>'

    '<div style="'
    'color:#8491a7; '
    'font-size:12px; '
    'margin-top:3px;">'

    'AI-powered news credibility prediction'

    '</div>'

    '</div>'

    '</div>'
)


st.markdown(
    header_html,
    unsafe_allow_html=True
)


# =========================================================
# USER INPUT
# =========================================================

title = st.text_input(
    "Article Title (optional)",
    placeholder="Enter the news headline"
)


article = st.text_area(
    "Article Text",
    placeholder="Paste the full news article here..."
)


analyze = st.button(
    "Analyze Article",
    use_container_width=True
)


# =========================================================
# PREDICTION RESULT
# =========================================================

if analyze:

    if not article.strip():

        st.warning(
            "Please enter the article text."
        )

    else:

        with st.spinner(
            "Analyzing article..."
        ):

            result, confidence = predict_news(
                title,
                article
            )


        # =================================================
        # REAL NEWS
        # =================================================

        if result == "REAL":

            real_html = (

                '<div style="'
                'margin-top:20px; '
                'background:#0e1922; '
                'border:1px solid #1d3b3a; '
                'border-radius:10px; '
                'padding:20px;">'

                '<div style="'
                'display:flex; '
                'justify-content:space-between; '
                'align-items:center;">'

                '<span style="'
                'background:#143a38; '
                'color:#6ee7d0; '
                'padding:6px 14px; '
                'border-radius:20px; '
                'font-size:13px; '
                'font-weight:700;">'

                'REAL NEWS'

                '</span>'

                '<span style="'
                'color:#dce5ef; '
                'font-size:14px;">'

                f'Confidence: <b>{confidence:.2f}%</b>'

                '</span>'

                '</div>'

                '<div style="'
                'background:#172530; '
                'height:7px; '
                'border-radius:10px; '
                'margin-top:18px; '
                'overflow:hidden;">'

                f'<div style="'
                f'width:{confidence:.2f}%; '
                'background:#45d6c3; '
                'height:7px; '
                'border-radius:10px;">'
                '</div>'

                '</div>'

                '</div>'
            )


            st.markdown(
                real_html,
                unsafe_allow_html=True
            )


        # =================================================
        # FAKE NEWS
        # =================================================

        else:

            fake_html = (

                '<div style="'
                'margin-top:20px; '
                'background:#1a111a; '
                'border:1px solid #442336; '
                'border-radius:10px; '
                'padding:20px;">'

                '<div style="'
                'display:flex; '
                'justify-content:space-between; '
                'align-items:center;">'

                '<span style="'
                'background:#401d2b; '
                'color:#ff7b9c; '
                'padding:6px 14px; '
                'border-radius:20px; '
                'font-size:13px; '
                'font-weight:700;">'

                'FAKE NEWS'

                '</span>'

                '<span style="'
                'color:#dce5ef; '
                'font-size:14px;">'

                f'Confidence: <b>{confidence:.2f}%</b>'

                '</span>'

                '</div>'

                '<div style="'
                'background:#2a1921; '
                'height:7px; '
                'border-radius:10px; '
                'margin-top:18px; '
                'overflow:hidden;">'

                f'<div style="'
                f'width:{confidence:.2f}%; '
                'background:#f05c83; '
                'height:7px; '
                'border-radius:10px;">'
                '</div>'

                '</div>'

                '</div>'
            )


            st.markdown(
                fake_html,
                unsafe_allow_html=True
            )
