"""
SignLens - Inference Example
Demonstrates how to load and run inference on the SignLens ISL & ASL models.
"""

import pickle
import numpy as np


def load_isl_model(model_path="isl_landmark_model.pkl"):
    """Load the 156-D ISL Dual-Hand Landmark Classifier."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded ISL Model: {len(data['classes'])} classes, {data['num_features']}-D input")
    return data["model"], data["classes"]


def load_asl_model(model_path="asl_landmark_model.pkl"):
    """Load the 78-D ASL Single-Hand Landmark Classifier."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded ASL Model: {len(data['classes'])} classes, {data['num_features']}-D input")
    return data["model"], data["classes"]


def predict_sign(model, classes, features):
    """
    Predict sign class and confidence score.
    :param model: Trained sklearn MLPClassifier
    :param classes: List of class labels
    :param features: 1D array of 78 (ASL) or 156 (ISL) normalized landmark features
    :return: (predicted_label, confidence_score, top3_predictions)
    """
    probs = model.predict_proba([features])[0]
    top_idx = int(np.argmax(probs))
    top_label = classes[top_idx]
    top_conf = float(probs[top_idx])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(classes[i], float(probs[i])) for i in top3_idx]

    return top_label, top_conf, top3


if __name__ == "__main__":
    # Test ISL Model
    isl_model, isl_classes = load_isl_model()
    dummy_isl_features = np.zeros(156, dtype=np.float32)
    label, conf, top3 = predict_sign(isl_model, isl_classes, dummy_isl_features)
    print(f"[ISL Test] Dummy Input Prediction: {label} ({conf:.2%})")

    # Test ASL Model
    asl_model, asl_classes = load_asl_model()
    dummy_asl_features = np.zeros(78, dtype=np.float32)
    label, conf, top3 = predict_sign(asl_model, asl_classes, dummy_asl_features)
    print(f"[ASL Test] Dummy Input Prediction: {label} ({conf:.2%})")
