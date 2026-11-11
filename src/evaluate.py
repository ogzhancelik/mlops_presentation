import argparse
import json
import os
import sys


def build_markdown_report(payload: dict, min_accuracy: float, min_f1: float, passed: bool) -> str:
    metrics = payload["metrics"]
    params = payload["hyperparameters"]
    dataset = payload.get("dataset", {})

    acc = metrics["accuracy"]
    f1 = metrics["f1_score"]

    acc_pass = acc >= min_accuracy
    f1_pass = f1 >= min_f1

    banner = "## ✅ Quality Gate Passed" if passed else "## ❌ Quality Gate Failed"
    acc_status = "✅ PASS" if acc_pass else "❌ FAIL"
    f1_status = "✅ PASS" if f1_pass else "❌ FAIL"

    lines = [
        "# 🤖 ML CI/CD Evaluation Report",
        "",
        banner,
        "",
        "### 📊 Model Performance vs. Quality Thresholds",
        "",
        "| Metric | Value | Threshold | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Accuracy** | `{acc:.4f}` | `>= {min_accuracy:.4f}` | {acc_status} |",
        f"| **F1-Score** | `{f1:.4f}` | `>= {min_f1:.4f}` | {f1_status} |",
        "",
        "### ⚙️ Hyperparameters & Dataset Metadata",
        "",
        "| Parameter | Value |",
        "| :--- | :--- |",
        f"| `max_depth` | `{params.get('max_depth')}` |",
        f"| `n_estimators` | `{params.get('n_estimators')}` |",
        f"| `random_state` | `{params.get('random_state')}` |",
        f"| `dataset` | `{dataset.get('name', 'breast_cancer')}` "
        f"({dataset.get('n_samples', 569)} samples, {dataset.get('n_features', 30)} features) |",
        "",
    ]
    return "\n".join(lines)


def evaluate_model(
    min_accuracy: float = 0.92,
    min_f1: float = 0.92,
    metrics_path: str = os.path.join("artifacts", "metrics.json"),
) -> bool:
    if not os.path.exists(metrics_path):
        print(f"[ERROR] Metrics file not found at: {metrics_path}")
        return False

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload["metrics"]
    params = payload["hyperparameters"]
    acc = metrics["accuracy"]
    f1 = metrics["f1_score"]

    acc_ok = acc >= min_accuracy
    f1_ok = f1 >= min_f1
    passed = acc_ok and f1_ok

    print("=" * 60)
    print("               ML QUALITY GATE EVALUATION               ")
    print("=" * 60)
    print(f"[INFO] Hyperparameters : max_depth={params['max_depth']}, "
          f"n_estimators={params['n_estimators']}")
    print(f"[EVAL] Accuracy        : {acc:.4f} (Threshold: >= {min_accuracy:.4f}) -> "
          f"{'PASS' if acc_ok else 'FAIL'}")
    print(f"[EVAL] F1-Score        : {f1:.4f} (Threshold: >= {min_f1:.4f}) -> "
          f"{'PASS' if f1_ok else 'FAIL'}")
    print("-" * 60)

    report_md = build_markdown_report(payload, min_accuracy, min_f1, passed)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report_md + "\n")
        print(f"[INFO] Wrote Markdown evaluation summary to $GITHUB_STEP_SUMMARY ({summary_path})")

    if passed:
        print("[SUCCESS] ✅ Quality Gate Passed! Model meets all production criteria.")
    else:
        print("[FAILURE] ❌ Quality Gate Failed! Model metrics are below required thresholds.")

    return passed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model metrics against thresholds"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.92,
        help="Minimum required accuracy (default: 0.92)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.92,
        help="Minimum required F1-score (default: 0.92)",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=os.path.join("artifacts", "metrics.json"),
        help="Path to the metrics JSON file (default: artifacts/metrics.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = evaluate_model(
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
        metrics_path=args.metrics_path,
    )
    sys.exit(0 if success else 1)
