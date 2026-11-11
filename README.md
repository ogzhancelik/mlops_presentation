# 🚀 GitHub Actions – CI/CD for Machine Learning (Live Demo)

A fast, self-contained **MLOps Continuous Integration / Continuous Delivery (CI/CD)** pipeline built with **GitHub Actions**, **scikit-learn**, and **Pytest**. Designed to execute in **under 25 seconds** for a live 4-minute engineering demo.

---

## 🏗️ What This Pipeline Does

Every `git push` or Pull Request to `main` triggers an automated workflow (`.github/workflows/ml-ci.yml`) that executes 5 core MLOps stages:

1. **Code Quality & Linting (`flake8`)**: Enforces PEP8 formatting across `src/` and `tests/`.
2. **Data & Unit Validation (`pytest`)**: Validates dataset integrity (shape, zero NaNs, binary classes) and verifies the training/evaluation lifecycle in an isolated temp directory.
3. **Automated Model Training (`src/train.py`)**: Trains a `RandomForestClassifier` on the built-in Breast Cancer Wisconsin dataset and exports `artifacts/model.pkl` and `artifacts/metrics.json`.
4. **Automated Quality Gate (`src/evaluate.py`)**: Compares model **Accuracy** and **F1-Score** against strict production thresholds (`--min-accuracy 0.92`, `--min-f1 0.92`). Writes a rich Markdown report directly to `$GITHUB_STEP_SUMMARY` and blocks the pipeline (`exit 1`) if metrics degrade.
5. **Model Registry / Artifact Upload (`actions/upload-artifact@v4`)**: Packages and uploads `model.pkl` and `metrics.json` as a downloadable `trained-ml-model` release artifact if the Quality Gate passes.

---

## 📂 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ml-ci.yml          # GitHub Actions CI/CD workflow
├── src/
│   ├── train.py               # Model training & metric logging script
│   └── evaluate.py            # Quality gate & GitHub Step Summary generator
├── tests/
│   └── test_pipeline.py       # Data integrity & pipeline unit tests
├── requirements.txt           # Lightweight pinned Python dependencies
├── .gitignore                 # Ignores artifacts/, caches, and virtualenvs
└── README.md                  # Documentation & live demo guide
```

---

## 💻 Local Setup & Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Lint & Run Unit Tests

```bash
flake8 src/ tests/ --max-line-length=100
pytest tests/ -v
```

### 3. Train the Model

```bash
python src/train.py --max-depth 5 --n-estimators 50
```

Outputs:
- `artifacts/model.pkl` – Serialized `RandomForestClassifier` model
- `artifacts/metrics.json` – Accuracy, F1-Score, hyperparameters, and dataset metadata

### 4. Run the Quality Gate Evaluation

```bash
python src/evaluate.py --min-accuracy 0.92 --min-f1 0.92
```

---

## 🎬 4-Minute Live Demo Walkthrough

### Step 1: Show the Pipeline Code (1 min)
- Open `.github/workflows/ml-ci.yml` to highlight how **Code Lint → Data Tests → Training → Quality Gate → Artifact Upload** are chained together.
- Open `src/train.py` and `src/evaluate.py` to show how `$GITHUB_STEP_SUMMARY` automatically publishes Markdown tables to the GitHub UI.

### Step 2: Live Hyperparameter Update & Push (1 min)
Edit `.github/workflows/ml-ci.yml` (or `src/train.py` defaults) to tweak hyperparameters (e.g., change `--max-depth 5 --n-estimators 50` to `--max-depth 8 --n-estimators 100`), then commit and push live:

```bash
git add .
git commit -m "feat: tune RandomForest hyperparameters (max_depth=8, n_estimators=100)"
git push origin main
```

### Step 3: Inspect GitHub Actions UI Live (2 min)
1. Navigate to your GitHub repository → click the **Actions** tab.
2. Click on the newly triggered **ML CI/CD Pipeline** workflow run.
3. Watch the live logs execute in `<25s` thanks to `cache: 'pip'`.
4. Click **Summary** (top-left of the run page) and scroll down to view:
   - **🤖 ML CI/CD Evaluation Report** (`✅ Quality Gate Passed` banner, metrics vs. thresholds table, and hyperparameter table).
   - **Artifacts** section at the bottom containing the downloadable **`trained-ml-model`** (`model.pkl` + `metrics.json`).

