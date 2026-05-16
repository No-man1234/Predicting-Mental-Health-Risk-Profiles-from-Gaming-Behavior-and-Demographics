# Predicting Mental Health Risk Profiles from Gaming Behavior and Demographics

## Project Overview

This is a comprehensive machine learning project that develops predictive models to assess mental health risk levels based on gaming behavior patterns and demographic characteristics. The project combines gaming activity data with validated psychometric scales (GAD-7, SWL, SPIN) to predict mental health risk indicators.

**Status**: Production-ready | **Type**: Academic + Portfolio Project

---

## Quick Start

### 🚀 Run Locally in 3 Steps

```bash
# 1. Clone and navigate
git clone https://github.com/neyamulhasan/Predicting-Mental-Health-Risk-Profiles-from-Gaming-Behavior-and-Demographics.git
cd Predicting-Mental-Health-Risk-Profiles-from-Gaming-Behavior-and-Demographics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

**For detailed setup instructions, see [GETTING_STARTED.md](GETTING_STARTED.md)**

---

## Requirements

### Tested Environment
- Python 3.12
- pip
- Git
- Jupyter / VS Code notebook support

### Python Packages
Install everything with:
```bash
pip install -r requirements.txt
```

### Notes
- On Linux, install `python3-venv` if virtual environment creation fails.
- In VS Code, select the same Python environment for both the terminal and the notebook kernel.
- The notebook and the script use the same project modules under `src/`.

---

## Project Structure

```
project/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned & engineered data
├── notebooks/
│   └── main.ipynb            # Interactive analysis
├── src/
│   ├── config.py             # Configuration
│   ├── data_preprocessing.py # Cleaning pipeline
│   ├── feature_engineering.py # Feature creation
│   ├── modeling.py           # Model training
│   ├── evaluation.py         # Evaluation metrics
│   └── visualization.py      # EDA plots
├── models/                   # Trained models
├── reports/                  # Results & visualizations
└── requirements.txt          # Dependencies
```

---

## Dataset

- **Size**: 13,464 responses (12,814 after cleaning)
- **Features**: Gaming behavior, demographics, psychometric scales
- **Targets**: GAD-7, SWL, SPIN mental health scores

---

## Methodology

1. **Data Preprocessing**: Handle missing values, standardize categories, remove outliers
2. **Feature Engineering**: Create gaming intensity, demographic groups, behavioral indicators
3. **Target Engineering**: Convert continuous scores to risk categories (Low/Moderate/High)
4. **Model Training**: Train 6 models (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, CatBoost)
5. **Evaluation**: Compare models using Accuracy, Precision, Recall, F1-Score
6. **Interpretation**: Feature importance and model insights

---

## Models Trained

- Logistic Regression (Baseline)
- Random Forest
- Support Vector Machine
- XGBoost
- LightGBM
- CatBoost

---

## Important Ethical Disclaimer

⚠️ **This model predicts mental health RISK INDICATORS and is NOT intended to:**
- Diagnose mental illness
- Replace professional mental health evaluation
- Provide medical advice
- Be used as sole basis for clinical decisions

Use only as supplementary assessment tool with professional guidance.

---

## Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.12)
- pip (Python package manager)
- ~2GB disk space

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/neyamulhasan/Predicting-Mental-Health-Risk-Profiles-from-Gaming-Behavior-and-Demographics.git
cd Predicting-Mental-Health-Risk-Profiles-from-Gaming-Behavior-and-Demographics

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, xgboost; print('✓ Ready to run')"
```

### Running the Project

**Option 1: Complete Pipeline (Recommended)**
```bash
python main.py
```
Runs all steps: data cleaning → feature engineering → model training → evaluation

**Option 2: Interactive Jupyter Notebook**
```bash
jupyter notebook notebooks/main.ipynb
```
Explore data and results step-by-step in your browser

**Option 3: Use Individual Modules**
```python
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer

preprocessor = DataPreprocessor()
df_cleaned = preprocessor.preprocess()
# ... continue with feature engineering and modeling
```

**For detailed setup guide, see [GETTING_STARTED.md](GETTING_STARTED.md)**

---

## Usage

### Run Complete Pipeline
```bash
python main.py
```

### Use Individual Modules
```python
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer

# Preprocess
preprocessor = DataPreprocessor()
df_cleaned = preprocessor.preprocess()

# Engineer features
engineer = FeatureEngineer(df_cleaned)
X, y, features = engineer.engineer_features()

# Train models
trainer = ModelTrainer(X, y)
trainer.split_data()
trainer.train_all_models()
results = trainer.evaluate_all_models()
```

### Launch Jupyter Notebook
```bash
jupyter notebook notebooks/main.ipynb
```

---

## Output Files

- `data/processed/data_cleaned.csv` - Preprocessed data
- `data/processed/data_engineered.csv` - Features + target
- `data/processed/model_comparison.csv` - Model metrics
- `reports/figures/` - Visualizations
- `models/` - Trained model files

---

## Key Findings

After running the pipeline, results show:
- Model performance metrics
- Feature importance analysis
- Gaming behavior patterns
- Demographic insights

---

## Configuration

Edit `src/config.py` to customize:
- Random state (reproducibility)
- Train/test split ratio
- Risk category thresholds
- Model hyperparameters
- Feature engineering settings

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- Survey participants
- Open-source ML community
- Mental health researchers

---

**Version**: 1.0.0 | **Last Updated**: 2026

