"""
Data preprocessing module for the Mental Health Risk Prediction project.

This module handles data loading, cleaning, and initial transformation.
"""

import pandas as pd
import numpy as np
from src.config import DATASET_PATH, DATA_PROCESSED_PATH
from src.utils import (
    print_separator, print_data_info, analyze_missing_values,
    get_categorical_columns, get_numerical_columns, fillna_strategy,
    save_dataframe_to_csv, remove_columns
)


class DataPreprocessor:
    """Handle all data preprocessing tasks."""
    
    def __init__(self, raw_data_path: str = DATASET_PATH):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path (str): Path to raw data file.
        """
        self.raw_data_path = raw_data_path
        self.df = None
        self.df_cleaned = None
        self.cleaning_report = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw dataset.
        
        Returns:
            pd.DataFrame: Raw dataset.
        """
        print_separator("Loading Raw Data")
        self.df = pd.read_csv(self.raw_data_path, index_col=0, encoding='latin-1')
        print_data_info(self.df, "Raw Dataset")
        return self.df
    
    def inspect_data(self) -> dict:
        """
        Perform initial data inspection.
        
        Returns:
            dict: Inspection report.
        """
        print_separator("Data Inspection")
        
        report = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
        }
        
        print(f"Dataset Shape: {report['shape']}")
        print(f"\nDuplicate Rows: {report['duplicates']}")
        
        missing_data = analyze_missing_values(self.df)
        print(f"\nTop Missing Value Columns:\n{missing_data.head(10)}")
        
        return report
    
    def handle_duplicates(self) -> None:
        """Remove duplicate rows."""
        print_separator("Handling Duplicates")
        
        before = len(self.df)
        self.df_cleaned = self.df.drop_duplicates()
        after = len(self.df_cleaned)
        
        removed = before - after
        self.cleaning_report['duplicates_removed'] = removed
        print(f"Removed {removed} duplicate rows ({(removed/before*100):.2f}%)")
    
    def handle_missing_values(self) -> None:
        """Handle missing values strategically."""
        print_separator("Handling Missing Values")
        
        # Drop columns with >70% missing values
        missing_pct = (self.df_cleaned.isnull().sum() / len(self.df_cleaned)) * 100
        high_missing = missing_pct[missing_pct > 70].index.tolist()
        
        print(f"Dropping {len(high_missing)} columns with >70% missing:")
        print(high_missing)
        self.df_cleaned = remove_columns(self.df_cleaned, high_missing)
        
        # For critical target columns, drop rows with missing values
        target_cols = ['GAD_T', 'SWL_T', 'SPIN_T']
        before = len(self.df_cleaned)
        for col in target_cols:
            if col in self.df_cleaned.columns:
                self.df_cleaned = self.df_cleaned[self.df_cleaned[col].notna()]
        
        after = len(self.df_cleaned)
        rows_removed = before - after
        self.cleaning_report['rows_missing_targets'] = rows_removed
        print(f"Removed {rows_removed} rows with missing target values")
        
        # Fill remaining missing values
        numerical_cols = get_numerical_columns(self.df_cleaned)
        categorical_cols = get_categorical_columns(self.df_cleaned)
        
        self.df_cleaned = fillna_strategy(
            self.df_cleaned, 
            strategy='median',
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols
        )
        
        print(f"Filled remaining missing values using median/mode strategy")
    
    def standardize_categorical_values(self) -> None:
        """Standardize and normalize categorical values."""
        print_separator("Standardizing Categorical Values")
        
        categorical_cols = get_categorical_columns(self.df_cleaned)
        
        for col in categorical_cols:
            # Convert to string and strip whitespace
            self.df_cleaned[col] = self.df_cleaned[col].astype(str).str.strip()
            
            # Replace common placeholder values
            placeholders = ['N/A', 'NA', 'n/a', 'none', '-', 'None', 'NONE']
            for placeholder in placeholders:
                self.df_cleaned[col] = self.df_cleaned[col].replace(placeholder, 'Unknown')
            
            # Convert to lowercase for consistency
            if self.df_cleaned[col].dtype == 'object':
                self.df_cleaned[col] = self.df_cleaned[col].str.lower()
        
        print(f"Standardized {len(categorical_cols)} categorical columns")
    
    def remove_irrelevant_columns(self) -> None:
        """Remove irrelevant or administrative columns."""
        print_separator("Removing Irrelevant Columns")
        
        # Columns that are administrative or not useful for modeling
        irrelevant = [
            'Zeitstempel',  # Timestamp
            'GADE',  # Anxiety difficulty scale
            'Narcissism',  # Not part of main analysis
            'Reference',  # Data source
            'accept',  # Survey acceptance (all likely Accept)
            'Residence_ISO3',  # Redundant with Residence
            'Birthplace_ISO3',  # Redundant with Birthplace
            'highestleague',  # Too many missing values
            'earnings',  # Too sparse/unclear
        ]
        
        available_irrelevant = [col for col in irrelevant if col in self.df_cleaned.columns]
        self.df_cleaned = remove_columns(self.df_cleaned, available_irrelevant)
        
        self.cleaning_report['columns_removed'] = len(available_irrelevant)
        print(f"Removed {len(available_irrelevant)} irrelevant columns")
    
    def handle_outliers(self) -> None:
        """Handle extreme outliers in numerical features."""
        print_separator("Handling Outliers")
        
        numerical_cols = get_numerical_columns(self.df_cleaned)
        
        # For Hours played, cap at reasonable maximum (e.g., 24 hours/day)
        if 'Hours' in self.df_cleaned.columns:
            original_max = self.df_cleaned['Hours'].max()
            self.df_cleaned['Hours'] = self.df_cleaned['Hours'].clip(upper=24)
            if self.df_cleaned['Hours'].max() < original_max:
                print(f"Capped 'Hours' column at 24 (was {original_max})")
        
        # For Age, remove unrealistic values
        if 'Age' in self.df_cleaned.columns:
            before_age = len(self.df_cleaned)
            self.df_cleaned = self.df_cleaned[(self.df_cleaned['Age'] >= 13) & (self.df_cleaned['Age'] <= 100)]
            after_age = len(self.df_cleaned)
            if before_age > after_age:
                print(f"Removed {before_age - after_age} rows with unrealistic age values")
        
        print("Outliers handled")
    
    def create_quality_report(self) -> pd.DataFrame:
        """
        Create a comprehensive data quality report.
        
        Returns:
            pd.DataFrame: Quality report.
        """
        print_separator("Data Quality Report")
        
        report = pd.DataFrame({
            'Metric': [
                'Initial Rows',
                'Final Rows',
                'Rows Removed',
                'Initial Columns',
                'Final Columns',
                'Columns Removed',
                'Duplicates Removed',
                'Missing Value Rows Removed',
            ],
            'Value': [
                len(self.df),
                len(self.df_cleaned),
                len(self.df) - len(self.df_cleaned),
                len(self.df.columns),
                len(self.df_cleaned.columns),
                self.cleaning_report.get('columns_removed', 0),
                self.cleaning_report.get('duplicates_removed', 0),
                self.cleaning_report.get('rows_missing_targets', 0),
            ]
        })
        
        print(report.to_string(index=False))
        return report
    
    def preprocess(self) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline.
        
        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        print_separator("STARTING DATA PREPROCESSING")
        
        self.load_data()
        self.inspect_data()
        self.handle_duplicates()
        self.handle_missing_values()
        self.standardize_categorical_values()
        self.remove_irrelevant_columns()
        self.handle_outliers()
        self.create_quality_report()
        
        print_separator("Preprocessing Complete")
        print_data_info(self.df_cleaned, "Cleaned Dataset")
        
        return self.df_cleaned
    
    def save_cleaned_data(self) -> None:
        """Save cleaned dataset."""
        output_path = DATA_PROCESSED_PATH / "data_cleaned.csv"
        save_dataframe_to_csv(self.df_cleaned, str(output_path))


def main():
    """Main execution function."""
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.preprocess()
    preprocessor.save_cleaned_data()


if __name__ == "__main__":
    main()
