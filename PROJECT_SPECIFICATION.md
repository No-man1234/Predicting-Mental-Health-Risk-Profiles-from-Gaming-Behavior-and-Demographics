# Project Specification: Predicting Mental Health Risk Profiles from Gaming Behavior

Version: 1.0
Date: 2026-05-16

## Overview

This repository implements a research-grade machine-learning pipeline that predicts mental health risk profiles (Low / Moderate / High) from self-reported gaming behavior and demographic survey data. The project is implemented in Python and organized to support reproducible experiments, exploratory analysis via Jupyter notebooks, and modular model training and evaluation.

## Objective and Motivation

- Objective: Build and evaluate classification models that identify mental health risk categories using gaming behavior and demographic features derived from a 13k+ respondent survey.
- Motivation: Understand whether patterns in gaming behavior can provide signal for early screening of anxiety, social phobia, and life-satisfaction risk factors. This can inform public-health research and non-clinical monitoring tools.

## High-level Architecture

- Data: CSV survey file stored under `data/raw/GamingStudy_data.csv`.
- Ingest & Preprocess: `src/data_preprocessing.py` loads raw CSV, removes duplicates, imputes missing data, removes irrelevant columns, and performs basic outlier handling.
- Feature Engineering: `src/feature_engineering.py` creates risk targets (from GAD, SWL, SPIN), engineered gaming/demographic/behavioral features, and encodes categorical variables.
- Modeling: `src/modeling.py` trains multiple classifiers (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, CatBoost) and computes evaluation metrics.
- Evaluation: `src/evaluation.py` contains helpers for classification reports, confusion matrices, and feature importance plots.
- Visualization: `src/visualization.py` generates EDA charts and saves figures under `reports/figures`.
- Utilities & Config: `src/utils.py` provides helpers (I/O, logging, missing-value analysis) and `src/config.py` centralizes paths and model parameters.
- Notebook: `notebooks/main.ipynb` supports interactive exploration and re-running of the pipeline.

## Data Flow (End‑to‑End)

1. `main.py` (or the notebook) instantiates `DataPreprocessor` and runs `preprocess()`.
2. Cleaned data is saved to `data/processed/data_cleaned.csv`.
3. `FeatureEngineer` ingests the cleaned data and executes `engineer_features()` producing `X` and `y` and saving `data_engineered.csv`.
4. `ModelTrainer` loads `X` and `y`, splits into train/test, trains models via `train_all_models()`, stores results, and writes `model_comparison.csv`.
5. `ModelEvaluator` and `Visualizer` generate per-model evaluation artifacts and EDA visuals written to `reports/`.

## Folder-by-Folder Breakdown

- `data/raw/` — Original dataset; keep this directory tracked for reproducibility. Do not commit derived artifacts here.
- `data/processed/` — Generated cleaned and engineered CSV files; ignore in VCS and produce from pipeline.
- `src/` — Core Python packages with well-defined responsibilities:
  - `config.py`: central constants, data paths, model hyperparameters, and plotting defaults.
  - `utils.py`: project utilities (I/O, logging, missing value analysis, feature helpers).
  - `data_preprocessing.py`: Data cleaning, missing-value strategies, outlier treatment, quality reporting.
  - `feature_engineering.py`: Target creation, engineered features, encoding, and saving engineered dataset.
  - `modeling.py`: Model training orchestration, per-model training functions, evaluation aggregation, and saving results.
  - `evaluation.py`: Detailed evaluation helpers and plots for confusion matrices and feature importances.
  - `visualization.py`: EDA plotting and image exports.
- `notebooks/` — Exploratory analyses and prototyping; keep as narrative documentation and reproducible experiments (trusted but not authoritative for production runs).
- `models/` — Binary or serialized trained model artifacts (ignored in VCS).
- `reports/` — Generated research artifacts: `figures/`, `tables/`, and final `final_report.md`.

## Key Files and Responsibilities

- `main.py`: Lightweight orchestrator that runs preprocessing → feature engineering → training → evaluation → visualization. Use for reproducible CLI runs.
- `README.md` / `GETTING_STARTED.md`: Project setup, environment, and run instructions.
- `requirements.txt`: Pin lists of required Python packages for the research environment.

## Core Implementation Details

### Data Preprocessing (`src/data_preprocessing.py`)

- Loads data using `pandas.read_csv` with `latin-1` encoding and `index_col=0`.
- Dedupe with `drop_duplicates()`.
- Drops columns with >70% missing values to reduce noise and avoid imputation bias.
- Removes rows missing critical target columns (`GAD_T`, `SWL_T`, `SPIN_T`).
- Imputes remaining missing values using median for numeric columns and mode (or 'Unknown') for categoricals through `fillna_strategy()`.
- Caps `Hours` at 24 and filters `Age` to [13, 100] to remove unrealistic survey entries.
- Produces a small data quality report and writes `data_cleaned.csv`.

Rationale: Conservative cleaning choices prioritize training stability and avoid propagating sparsely populated features.

### Feature Engineering (`src/feature_engineering.py`)

- Creates individual risk categories `GAD_risk`, `SWL_risk`, `SPIN_risk` using quantile-based binning (33%, 66%) to mitigate class imbalance.
- Constructs a composite `mental_health_risk` by averaging mapped numeric risk scores and re-binning into three labels.
- Engineering of gaming behavior (e.g., `gaming_intensity`, `is_multiplayer`, `is_competitive`, `streams_watcher`, `has_rank`) from columns like `Hours`, `Playstyle`, `streams`, and `League`.
- Demographic transformations (age groups, employment groups, education level simplification).
- Behavioral aggregates (e.g., `anxiety_frequency`, `social_fear_avg`) derived from psychometric scale item columns.
- Categorical encoding: one-hot encoding with `pd.get_dummies(..., drop_first=True)` applied only to categorical columns with <=50 unique values; features with >50 unique categories are dropped to reduce dimensionality.

Engineering rationale: Use interpretable, low-dimensional encodings suitable for both linear and tree-based models; aggressive trimming of high-cardinality categories prevents overfitting in small-sample features.

### Modeling (`src/modeling.py`)

- Models supported: Logistic Regression (pipeline with StandardScaler), Random Forest, SVM (pipeline with StandardScaler), XGBoost, LightGBM, CatBoost.
- For scikit-learn models, training uses raw categorical/dummy-encoded matrices; for tree boosters, target labels are label-encoded before training.
- `train_all_models()` invokes each model-specific train function. For scikit-learn models, `_evaluate_model()` computes Accuracy, Macro Precision/Recall/F1 and stores results in `self.results`.
- XGBoost/LightGBM/CatBoost currently compute and print accuracy but store their trained estimators in `self.models` as tuples (estimator, label_encoder). Their metrics are not appended to `self.results` by default — this is a small technical inconsistency to address in later improvements.

Training rationale: Empirical ensemble selection to compare linear vs. tree-based approaches. Pipelines standardize inputs for algorithms sensitive to scaling.

### Evaluation (`src/evaluation.py`)

- Provides `ModelEvaluator` with methods to predict, compute classification report, confusion matrix, and plot feature importance when the model exposes `feature_importances_`.
- Uses scikit-learn metrics and seaborn/matplotlib for visual outputs.

Evaluation rationale: Separate evaluation responsibilities to keep modeling code focused and enable reusable per-model analysis.

### Visualization (`src/visualization.py`)

- Produces bar charts, histograms, boxplots, violin plots, and a correlation heatmap for numerical features. Exports images to `reports/figures`.

Visualization rationale: Provide publication-ready figures with seaborn themes and high DPI outputs.

## Configuration and Dependencies

- `src/config.py` centralizes paths, hyperparameters, plotting defaults, and dataset locations.
- `requirements.txt` lists the minimal packages necessary for reproducing experiments (pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, shap, matplotlib, seaborn, plotly, jupyter, etc.).

Why these libraries:
- `pandas`, `numpy`, `scipy`: Core data handling.
- `scikit-learn`: Baseline models and metrics.
- `xgboost`, `lightgbm`, `catboost`: Tree-boosting libraries that typically yield top performance on tabular data.
- `shap`: Model explainability if expanded.

## Execution Flow (Commands)

Run the full pipeline from CLI:

```bash
python main.py
```

Or run the notebook for interactive analysis:

1. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Open `notebooks/main.ipynb` in VS Code and select the `venv` kernel.

## Limitations and Known Issues

- Target binning uses quantiles rather than validated clinical thresholds; this simplifies class balance but reduces clinical interpretability.
- XGBoost/LightGBM/CatBoost metrics are not fully aggregated into the unified `results` table — a bug to fix.
- High-cardinality categorical columns are dropped without engineered alternatives (e.g., target encoding), potentially losing signal.
- The code relies on `pandas.read_csv(index_col=0)`, which requires a clean index column in the CSV.

## Suggested Improvements and Scalability Roadmap

Short-term fixes (low effort):
- Ensure all trained models append metrics to `self.results` consistently.
- Add unit tests for critical transformations (e.g., risk binning, `fillna_strategy`).
- Replace coarse `pd.get_dummies` for high-cardinality categoricals with frequency/target encoding.

Medium-term (engineering):
- Add a lightweight CLI with `argparse` to run specific pipeline stages (preprocess, engineer, train, evaluate).
- Introduce a small ML experiment tracking (e.g., `Mlflow` or `Weights & Biases`) to capture metric history and artifacts.
- Add a reproducible Dockerfile that installs system deps for `catboost` and `lightgbm` and pins Python version.

Long-term / scale: 
- Modularize project into a `src` package and add proper unit/integration tests and CI pipeline.
- Add data validation and schema checking (e.g., `pandera`) and type checking (`mypy`).
- Move heavy training to a distributed/accelerated environment and use DVC for large dataset/version control.

## File-to-Responsibility Map (Concise)

- `main.py`: Orchestration entry point.
- `src/config.py`: Global constants and defaults.
- `src/utils.py`: I/O, logging, missing-value analysis helpers.
- `src/data_preprocessing.py`: Cleaning + quality reporting.
- `src/feature_engineering.py`: Feature creation and encoding.
- `src/modeling.py`: Model training and comparison orchestration.
- `src/evaluation.py`: Metrics and plots for models.
- `src/visualization.py`: EDA and figure outputs.
- `notebooks/main.ipynb`: Interactive analysis and exploratory experiments.

## Governance and Ethical Considerations

- The repository includes an explicit disclaimer in `README.md` about ethical use: models are research instruments and not clinical diagnostics.
- Recommendations: maintain documented consent and IRB approvals for dataset usage; maintain privacy-preserving practices and data minimization.

---

Prepared by: Project Automation
