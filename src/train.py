import argparse
import json
import os
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def train_model(max_depth: int = 5, n_estimators: int = 50, artifacts_dir: str = "artifacts"):
    print("[INFO] Loading dataset (scikit-learn breast_cancer)...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")

    print(
        f"[INFO] Training RandomForestClassifier "
        f"(max_depth={max_depth}, n_estimators={n_estimators})..."
    )
    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    print(f"[METRICS] Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")

    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.pkl")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    joblib.dump(model, model_path)
    print(f"[INFO] Saved trained model to {model_path}")

    payload = {
        "metrics": {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
        },
        "hyperparameters": {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "random_state": 42,
            "test_size": 0.2,
        },
        "dataset": {
            "name": "breast_cancer",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Saved metrics and metadata to {metrics_path}")

    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RandomForest model on Breast Cancer data")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth of the trees (default: 5)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Number of trees in the forest (default: 50)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to save model and metrics artifacts (default: artifacts)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        artifacts_dir=args.artifacts_dir,
    )
