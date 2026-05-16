# Getting Started - Local Setup Guide

This guide walks you through running the Mental Health Risk Prediction ML project on your local machine.

## Prerequisites

- **Python 3.8+**; tested with Python 3.12
- **pip** for package installation
- **Git** for cloning the repository
- **Jupyter notebook support** for `notebooks/main.ipynb`
- **~2GB disk space** for dependencies and generated outputs
- **~5-10 minutes** for first-time setup

### Recommended Environment
- Use a dedicated virtual environment for this project.
- In VS Code, ensure the selected notebook kernel matches the environment where dependencies are installed.
- On Linux, install `python3-venv` if `python3 -m venv venv` is unavailable.

---

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/neyamulhasan/Predicting-Mental-Health-Risk-Profiles-from-Gaming-Behavior-and-Demographics.git

# Navigate to project directory
cd Predicting-Mental-Health-Risk-Profiles-from-Gaming-Behavior-and-Demographics
```

---

## Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies from your system Python.

### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

---

## Step 3: Install Dependencies

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, xgboost, lightgbm, catboost; print('✓ All dependencies installed')"
```

**Expected output:**
```
✓ All dependencies installed
```

---

## Step 4: Run the Complete Pipeline

### Option A: Run Everything at Once (Recommended for First Time)

```bash
# Run the complete ML pipeline
python main.py
```

**What this does:**
1. Loads and inspects the raw dataset (13,464 records)
2. Cleans the data (removes duplicates, handles missing values, treats outliers)
3. Engineers features (creates gaming intensity, demographic groups, behavioral indicators)
4. Trains 6 machine learning models
5. Evaluates and compares model performance
6. Generates visualizations
7. Saves all results to `data/processed/` and `reports/`

**Expected runtime:** 3-5 minutes

**Expected output:**
```
================================================================================
MENTAL HEALTH RISK PREDICTION FROM GAMING BEHAVIOR AND DEMOGRAPHICS
================================================================================

========================== STEP 1: DATA PREPROCESSING ==========================
...
[Data cleaning progress]
...

========================== STEP 2: FEATURE ENGINEERING =========================
...
[Feature creation progress]
...

========================== STEP 3: MODEL TRAINING ============================
...
[Model training progress]
...

========================== STEP 4: MODEL EVALUATION ===========================
...
[Model comparison results]
...

================================================================================
PIPELINE EXECUTION COMPLETED SUCCESSFULLY!
================================================================================

Output files:
  - Cleaned data: data/processed/data_cleaned.csv
  - Engineered features: data/processed/data_engineered.csv
  - Model comparison: data/processed/model_comparison.csv
  - Visualizations: reports/figures/
```

---

### Option B: Use Individual Modules (For Development/Experimentation)

```python
# Open Python interactive shell
python

# Then in Python:
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer

# Step 1: Preprocess data
print("Step 1: Preprocessing data...")
preprocessor = DataPreprocessor()
df_cleaned = preprocessor.preprocess()
preprocessor.save_cleaned_data()

# Step 2: Engineer features
print("Step 2: Engineering features...")
engineer = FeatureEngineer(df_cleaned)
X, y, feature_names = engineer.engineer_features()
engineer.save_engineered_data()

# Step 3: Train models
print("Step 3: Training models...")
trainer = ModelTrainer(X, y)
trainer.split_data()
trainer.train_all_models()
results = trainer.evaluate_all_models()
trainer.save_results()

print("✓ Pipeline complete!")
```

---

### Option C: Interactive Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/main.ipynb
```

This opens an interactive notebook in your browser where you can:
- Run code cells step-by-step
- See visualizations inline
- Modify parameters and re-run
- Explore the data interactively

---

## Step 5: View Results

After running the pipeline, check the output files:

### Processed Data
```bash
# View cleaned dataset
head -5 data/processed/data_cleaned.csv

# View engineered features
head -5 data/processed/data_engineered.csv

# View model comparison
cat data/processed/model_comparison.csv
```

### Visualizations
```bash
# List all generated visualizations
ls -lh reports/figures/

# View visualizations (on macOS)
open reports/figures/correlation_heatmap.png

# View visualizations (on Linux)
xdg-open reports/figures/correlation_heatmap.png

# View visualizations (on Windows)
start reports/figures/correlation_heatmap.png
```

### Research Report
```bash
# View the comprehensive research report
cat reports/final_report.md

# Or open in your text editor
code reports/final_report.md  # VS Code
nano reports/final_report.md  # Nano
```

---

## Project Structure

After running, your directory will look like:

```
project/
├── data/
│   ├── raw/
│   │   └── GamingStudy_data.csv          (Original dataset)
│   └── processed/
│       ├── data_cleaned.csv              (Cleaned data)
│       ├── data_engineered.csv           (With features)
│       └── model_comparison.csv          (Model metrics)
├── notebooks/
│   └── main.ipynb                        (Interactive notebook)
├── src/
│   ├── config.py                         (Configuration)
│   ├── utils.py                          (Utilities)
│   ├── data_preprocessing.py             (Data cleaning)
│   ├── feature_engineering.py            (Feature creation)
│   ├── modeling.py                       (Model training)
│   ├── evaluation.py                     (Evaluation)
│   └── visualization.py                  (Visualizations)
├── reports/
│   ├── final_report.md                   (Research report)
│   └── figures/                          (Generated charts)
├── main.py                               (Pipeline orchestrator)
├── README.md                             (Documentation)
└── requirements.txt                      (Dependencies)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Make sure you've installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Python version too old"

**Solution:** Update Python to 3.8 or newer:
```bash
python --version  # Check current version
# Download Python 3.12 from python.org
```

### Issue: "Permission denied" on macOS/Linux

**Solution:** Make scripts executable:
```bash
chmod +x main.py
```

### Issue: Out of memory during model training

**Solution:** Reduce model complexity in `src/config.py`:
```python
# Reduce n_estimators or max_depth
"random_forest": {
    "n_estimators": 25,  # Reduced from 50
    "max_depth": 5,      # Reduced from 8
}
```

### Issue: Jupyter notebook not found

**Solution:** Install Jupyter:
```bash
pip install jupyter notebook
```

---

## Key Results

After running the pipeline, you'll see:

**Model Performance:**
- Best Model: CatBoost (79.79% accuracy)
- Random Forest: 78.23% accuracy
- Logistic Regression: 77.49% accuracy

**Generated Outputs:**
- 3 processed CSV files (6.6 MB total)
- 5 visualization charts (1.2 MB)
- Comprehensive research report (13.6K words)
- Model comparison metrics

---

## Next Steps

1. **Explore the data:** Open `notebooks/main.ipynb` in Jupyter
2. **Read the report:** Check `reports/final_report.md` for detailed analysis
3. **Modify parameters:** Edit `src/config.py` to experiment with different settings
4. **Use individual modules:** Import modules in your own scripts
5. **Contribute:** Submit improvements via pull requests

---

## Support

For questions or issues:
1. Check the README.md for general information
2. Review the final_report.md for technical details
3. Examine the code comments in src/ modules
4. Open an issue on GitHub

---

**Happy analyzing! 🎉**

