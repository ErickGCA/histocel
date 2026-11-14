"""Analyse whole slides with a trained model: tile, predict, merge, count."""

from histocel.inference.slide import (
    InferenceError,
    ModelCard,
    load_model_card,
    open_predictor,
    predict_slide,
    render,
    to_annotation,
)

__all__ = [
    "InferenceError",
    "ModelCard",
    "load_model_card",
    "open_predictor",
    "predict_slide",
    "render",
    "to_annotation",
]
