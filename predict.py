"""Resuable inference helper for the Cats vs Dogs project

The saved model already contains MobileNetV2 processing, so images
must be passed as raw RGB pixels in the 0-255 range.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMG_SIZE = 160
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "cats_dogs_transfer.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "model" / "class_names.json"

def load_class_names(path=CLASS_NAMES_PATH):
    with open(path, encoding="UTF-8") as fhand:
        return json.load(fhand)

def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path)


def preprocess_pil_image(image, image_size=IMG_SIZE):
    """Turn a PIL image into a (1, size, size, 3) float32 batch.

    Pixels stay in the 0-255 range: the model scales them internally.
    """
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32)      # (H, W, 3), 0-255
    return np.expand_dims(array, axis=0)             # (1, H, W, 3)


def preprocess_image(image_path, image_size=IMG_SIZE):
    """Same as preprocess_pil_image, but starting from a file path."""
    with Image.open(image_path) as image:
        return preprocess_pil_image(image, image_size)


def _predict_batch(model, class_names, batch):
    """Shared core: run the model and pick the top class."""
    scores = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(scores))
    return class_names[index], float(scores[index]), scores


def predict_image(model, class_names, image_path):
    """Predict from a file path (used by the command-line script)."""
    return _predict_batch(model, class_names, preprocess_image(image_path))


def predict_pil_image(model, class_names, image):
    """Predict from a PIL image (used by the Streamlit app)."""
    return _predict_batch(model, class_names, preprocess_pil_image(image))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/image")

    model = load_model()
    class_names = load_class_names()

    label, confidence, scores = predict_image(model, class_names, sys.argv[1])

    print(f"Predictoin : {label}")
    print(f"Confidence : {confidence}")
    print(f"Scores : {scores}")
    
    

