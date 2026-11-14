"""Grade a model on the test slides nothing in training ever saw."""

from histocel.evaluation.evaluate import EvaluationError, evaluate_model

__all__ = ["EvaluationError", "evaluate_model"]
