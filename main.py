"""
Main execution script for the Mental Health Risk Prediction project.

This script orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import warnings
warnings.filterwarnings('ignore')

from src.utils import print_separator, setup_logging
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer
from src.visualization import Visualizer


def main():
    """Execute the complete ML pipeline."""
    
    # Setup logging
    logger = setup_logging()
    
    print_separator("=" * 80)
    print("MENTAL HEALTH RISK PREDICTION FROM GAMING BEHAVIOR AND DEMOGRAPHICS")
    print("=" * 80)
    print()
    
    # Step 1: Data Preprocessing
    print_separator("STEP 1: DATA PREPROCESSING")
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.preprocess()
    preprocessor.save_cleaned_data()
    print()
    
    # Step 2: Feature Engineering
    print_separator("STEP 2: FEATURE ENGINEERING")
    engineer = FeatureEngineer(df_cleaned)
    X, y, feature_names = engineer.engineer_features()
    engineer.save_engineered_data()
    print()
    
    # Step 3: Model Training
    print_separator("STEP 3: MODEL TRAINING")
    trainer = ModelTrainer(X, y)
    trainer.split_data()
    trainer.train_all_models()
    print()
    
    # Step 4: Model Evaluation
    print_separator("STEP 4: MODEL EVALUATION")
    results_df = trainer.evaluate_all_models()
    trainer.save_results()
    print()
    
    # Step 5: Visualization
    print_separator("STEP 5: GENERATING VISUALIZATIONS")
    visualizer = Visualizer(df_cleaned)
    visualizer.create_eda_report()
    print()
    
    print_separator("=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("Output files:")
    print("  - Cleaned data: data/processed/data_cleaned.csv")
    print("  - Engineered features: data/processed/data_engineered.csv")
    print("  - Model comparison: data/processed/model_comparison.csv")
    print("  - Visualizations: reports/figures/")
    print()


if __name__ == "__main__":
    main()
