import pytest
import torch
import numpy as np

from ml.metrics import (
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    generate_confusion_matrix,
    evaluate_all_metrics
)


@pytest.fixture
def dummy_predictions():
    # Ground Truth: [0, 0, 1, 1, 1]
    # Predictions : [0, 1, 1, 1, 0]
    # TP=2 (index 2,3), TN=1 (index 0), FP=1 (index 1), FN=1 (index 4)
    y_true = torch.tensor([0, 0, 1, 1, 1])
    y_pred = torch.tensor([0, 1, 1, 1, 0])
    return y_true, y_pred


def test_accuracy(dummy_predictions):
    y_true, y_pred = dummy_predictions
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == pytest.approx(0.60, 0.01)


def test_precision(dummy_predictions):
    y_true, y_pred = dummy_predictions
    precision = calculate_precision(y_true, y_pred, average="binary")
    # TP / (TP + FP) = 2 / 3 = 0.6667
    assert precision == pytest.approx(0.6667, 0.01)


def test_recall(dummy_predictions):
    y_true, y_pred = dummy_predictions
    recall = calculate_recall(y_true, y_pred, average="binary")
    # TP / (TP + FN) = 2 / 3 = 0.6667
    assert recall == pytest.approx(0.6667, 0.01)


def test_f1(dummy_predictions):
    y_true, y_pred = dummy_predictions
    f1 = calculate_f1(y_true, y_pred, average="binary")
    # 2 * (P * R) / (P + R) = 2 * (2/3 * 2/3) / (4/3) = 0.6667
    assert f1 == pytest.approx(0.6667, 0.01)


def test_confusion_matrix(dummy_predictions):
    y_true, y_pred = dummy_predictions
    cm = generate_confusion_matrix(y_true, y_pred)
    # [[TN, FP], [FN, TP]] -> [[1, 1], [1, 2]]
    assert cm == [[1, 1], [1, 2]]


def test_evaluate_all_metrics(dummy_predictions):
    y_true, y_pred = dummy_predictions
    results = evaluate_all_metrics(y_true, y_pred, average="binary")
    
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "confusion_matrix" in results