"""
Configuration module for the Mental Health Risk Prediction project.

This module contains all configuration parameters, paths, and constants
used throughout the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = DATA_RAW_PATH / "GamingStudy_data.csv"

# Model paths
MODELS_PATH = PROJECT_ROOT / "models"

# Reports and outputs
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
TABLES_PATH = REPORTS_PATH / "tables"

# Create directories if they don't exist
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, FIGURES_PATH, TABLES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Data preprocessing configuration
DATA_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "val_size": 0.15,
    "stratify_column": "risk_category",
}

# GAD-7 (Generalized Anxiety Disorder) scale configuration
GAD_CONFIG = {
    "scale_items": ["GAD1", "GAD2", "GAD3", "GAD4", "GAD5", "GAD6", "GAD7"],
    "total_score_column": "GAD_T",
    "thresholds": {
        "minimal": (0, 4),
        "mild": (5, 9),
        "moderate": (10, 14),
        "severe": (15, 21),
    },
}

# SWL (Satisfaction with Life Scale) configuration
SWL_CONFIG = {
    "scale_items": ["SWL1", "SWL2", "SWL3", "SWL4", "SWL5"],
    "total_score_column": "SWL_T",
    "thresholds": {
        "low_satisfaction": (5, 9),
        "moderate_satisfaction": (10, 24),
        "high_satisfaction": (25, 35),
    },
}

# SPIN (Social Phobia Inventory) configuration
SPIN_CONFIG = {
    "scale_items": ["SPIN" + str(i) for i in range(1, 18)],
    "total_score_column": "SPIN_T",
    "thresholds": {
        "minimal": (0, 19),
        "mild": (20, 39),
        "moderate": (40, 59),
        "severe": (60, 80),
    },
}

# Risk categories
RISK_CATEGORIES = ["Low Risk", "Moderate Risk", "High Risk"]

# Demographic features
DEMOGRAPHIC_FEATURES = [
    "Gender",
    "Age",
    "Work",
    "Degree",
    "Residence",
    "Birthplace",
]

# Gaming behavior features
GAMING_FEATURES = [
    "Game",
    "Platform",
    "Hours",
    "whyplay",
    "League",
    "streams",
    "Playstyle",
]

# Model configuration
MODEL_CONFIG = {
    "logistic_regression": {
        "max_iter": 500,
        "random_state": 42,
        "n_jobs": -1,
    },
    "random_forest": {
        "n_estimators": 50,
        "max_depth": 8,
        "random_state": 42,
        "n_jobs": -1,
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 50,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 50,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "catboost": {
        "iterations": 50,
        "depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbose": False,
    },
}

# Hyperparameter tuning search space
HYPERPARAMETER_SPACE = {
    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "penalty": ["l2"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "poly"],
    },
}

# Visualization configuration
PLOT_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "figure_size": (12, 6),
    "dpi": 300,
    "color_palette": "husl",
}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro f1",
    "roc_auc",
]
