import json
import os
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

from src.evaluate import evaluate_model
from src.train import train_model


def test_dataset_integrity():
    """Verify the built-in breast cancer dataset shape, target classes, and absence of NaNs."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    assert X.ndim == 2
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert not np.isnan(X).any(), "Feature matrix contains NaN values"
    assert not np.isnan(y).any(), "Target vector contains NaN values"
    assert set(np.unique(y)) == {0, 1}


def test_training_and_evaluation_pipeline(tmp_path, monkeypatch):
    """Verify that training generates valid model.pkl & metrics.json and passes evaluation."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    artifacts_dir = str(tmp_path / "artifacts")
    payload = train_model(max_depth=4, n_estimators=30, artifacts_dir=artifacts_dir)

    model_path = os.path.join(artifacts_dir, "model.pkl")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    assert os.path.isfile(model_path), "model.pkl was not created"
    assert os.path.isfile(metrics_path), "metrics.json was not created"

    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, "predict"), "Loaded artifact is not a valid estimator"

    with open(metrics_path, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    assert "metrics" in saved_metrics
    assert "hyperparameters" in saved_metrics
    assert 0.85 <= saved_metrics["metrics"]["accuracy"] <= 1.0
    assert 0.85 <= saved_metrics["metrics"]["f1_score"] <= 1.0
    assert saved_metrics["hyperparameters"]["max_depth"] == 4
    assert saved_metrics["hyperparameters"]["n_estimators"] == 30
    assert payload == saved_metrics

    assert evaluate_model(min_accuracy=0.90, min_f1=0.90, metrics_path=metrics_path) is True
    assert evaluate_model(min_accuracy=0.9999, min_f1=0.9999, metrics_path=metrics_path) is False
