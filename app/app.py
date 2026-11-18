"""Streamlit interface for the Cats vs Dogs final project.

Prediction logic is imported from predict.py — the same code that was
verified from the command line in Session 1.
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import load_class_names, load_model, predict_pil_image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# Below this confidence we show a caution message.
LOW_CONFIDENCE = 0.60


@st.cache_resource
def get_model_and_labels():
    """Load the model and labels once and share them across reruns."""
    return load_model(), load_class_names()


model, class_names = get_model_and_labels()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a JPG or PNG photo of a cat or a dog.")
    st.write("2. The model predicts the class and shows its confidence.")
    st.write("3. Try several photos and compare the scores.")

    st.divider()
    st.header("About the model")
    st.write(
        "MobileNetV2 pretrained on ImageNet, with a new two-class "
        "head, trained on 2,000 cat and dog photographs."
    )
    st.write(f"Classes: {', '.join(class_names)}")

    st.divider()
    st.header("Limitations")
    st.write("- Only two classes: it cannot answer 'neither'.")
    st.write("- Small training set; unusual photos may be wrong.")
    st.write("- Scores are not calibrated probabilities.")

# -------------------------------------------------------------- main page
st.title("🐾 Cat vs Dog Classifier")
st.write(
    "Upload a JPG or PNG photograph and the model will predict "
    "whether it shows a cat or a dog."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("No image yet. Upload a JPG or PNG file to get a prediction.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
width, height = image.size

left, right = st.columns([1, 1])

with left:
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.caption(f"{uploaded_file.name} · {width}×{height} px")

with right:
    with st.spinner("Classifying…"):
        label, confidence, scores = predict_pil_image(model, class_names, image)

    st.success(f"Prediction: **{label}**")
    st.metric("Confidence", f"{confidence:.1%}")

    if confidence < LOW_CONFIDENCE:
        st.warning(
            "Confidence is low. The photo may be unclear, unusual, "
            "or show something other than a cat or a dog."
        )

    st.write("Scores for each class:")
    for name, score in zip(class_names, scores):
        st.write(f"**{name}** — {score:.1%}")
        st.progress(float(score))

st.divider()
st.caption(
    "This model was trained on a small classroom dataset. "
    "It may be wrong on unusual photos, and its scores are not "
    "calibrated probabilities."
)