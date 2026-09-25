import torch
import numpy as np
from typing import Union, Dict, Any

from ml.metrics.accuracy import calculate_accuracy
from ml.metrics.precision import calculate_precision
from ml.metrics.recall import calculate_recall
from ml.metrics.f1 import calculate_f1
from ml.metrics.confusion_matrix import generate_confusion_matrix


def evaluate_all_metrics(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray],
    average: str = "binary"
) -> Dict[str, Any]:
    """
    Master evaluator returning a complete dictionary of classification metrics.
    Useful for logging in FL server evaluation rounds.
    """
    return {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred, average=average),
        "recall": calculate_recall(y_true, y_pred, average=average),
        "f1_score": calculate_f1(y_true, y_pred, average=average),
        "confusion_matrix": generate_confusion_matrix(y_true, y_pred)
    }