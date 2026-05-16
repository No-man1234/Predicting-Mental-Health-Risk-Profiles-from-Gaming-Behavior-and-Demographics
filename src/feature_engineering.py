"""
Feature engineering module for the Mental Health Risk Prediction project.

This module handles feature creation, target encoding, and feature transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import DATA_PROCESSED_PATH, RISK_CATEGORIES
from src.utils import (
    print_separator, print_data_info, get_categorical_columns, 
    get_numerical_columns, create_risk_categories, save_dataframe_to_csv
)


class FeatureEngineer:
    """Handle all feature engineering tasks."""
    
    def __init__(self, df_cleaned: pd.DataFrame):
        """
        Initialize feature engineer.
        
        Args:
            df_cleaned (pd.DataFrame): Cleaned dataset.
        """
        self.df = df_cleaned.copy()
        self.df_engineered = None
        self.X = None
        self.y = None
        self.categorical_features = None
        self.numerical_features = None
    
    def create_target_variables(self) -> None:
        """Create risk category target variables from continuous scores."""
        print_separator("Creating Target Variables")
        
        # Create risk categories for GAD
        if 'GAD_T' in self.df.columns:
            self.df['GAD_risk'] = create_risk_categories(self.df['GAD_T'], None)
            print(f"GAD Risk Distribution:\n{self.df['GAD_risk'].value_counts()}\n")
        
        # Create risk categories for SWL
        if 'SWL_T' in self.df.columns:
            # For SWL, higher scores are better (reverse coding needed)
            swl_inverted = 36 - self.df['SWL_T']
            self.df['SWL_risk'] = create_risk_categories(swl_inverted, None)
            print(f"SWL Risk Distribution:\n{self.df['SWL_risk'].value_counts()}\n")
        
        # Create risk categories for SPIN
        if 'SPIN_T' in self.df.columns:
            self.df['SPIN_risk'] = create_risk_categories(self.df['SPIN_T'], None)
            print(f"SPIN Risk Distribution:\n{self.df['SPIN_risk'].value_counts()}\n")
        
        # Create composite risk category (multi-target problem)
        if all(col in self.df.columns for col in ['GAD_risk', 'SWL_risk', 'SPIN_risk']):
            # Map risk categories to numeric scores
            risk_map = {"Low Risk": 1, "Moderate Risk": 2, "High Risk": 3}
            gad_score = self.df['GAD_risk'].astype(str).map(risk_map).astype(float)
            swl_score = self.df['SWL_risk'].astype(str).map(risk_map).astype(float)
            spin_score = self.df['SPIN_risk'].astype(str).map(risk_map).astype(float)
            
            # Composite: average of the three
            composite_score = (gad_score + swl_score + spin_score) / 3
            self.df['mental_health_risk'] = pd.cut(
                composite_score,
                bins=[0.5, 1.5, 2.5, 3.5],
                labels=['Low Risk', 'Moderate Risk', 'High Risk'],
                include_lowest=True
            )
            print(f"Composite Mental Health Risk Distribution:\n{self.df['mental_health_risk'].value_counts()}\n")
    
    def create_gaming_features(self) -> None:
        """Create gaming-related features."""
        print_separator("Creating Gaming Features")
        
        # Gaming intensity (based on hours and engagement)
        if 'Hours' in self.df.columns:
            self.df['gaming_intensity'] = pd.cut(
                self.df['Hours'],
                bins=[0, 5, 10, 15, 24],
                labels=['Low', 'Moderate', 'High', 'Very High'],
                include_lowest=True
            )
        
        # Multiplayer vs Single player engagement
        if 'Playstyle' in self.df.columns:
            self.df['is_multiplayer'] = self.df['Playstyle'].str.contains(
                'multiplayer', case=False, na=False
            ).astype(int)
            
            self.df['is_competitive'] = self.df['Playstyle'].str.contains(
                'competitive|online.*strangers', case=False, na=False
            ).astype(int)
        
        # Streams viewership (engagement indicator)
        if 'streams' in self.df.columns:
            self.df['streams_watcher'] = (self.df['streams'] > 0).astype(int)
        
        # League rank indicator (competitive engagement)
        if 'League' in self.df.columns:
            self.df['has_rank'] = (self.df['League'] != 'unknown').astype(int)
        
        print("✓ Created gaming intensity features")
    
    def create_demographic_features(self) -> None:
        """Create demographic-related features."""
        print_separator("Creating Demographic Features")
        
        # Age groups
        if 'Age' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['Age'],
                bins=[13, 18, 25, 35, 50, 100],
                labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior'],
                include_lowest=True
            )
        
        # Employment status groups
        if 'Work' in self.df.columns:
            self.df['employment_group'] = self.df['Work'].apply(
                lambda x: 'Student' if 'student' in str(x).lower()
                else 'Employed' if 'employed' in str(x).lower() and 'un' not in str(x).lower()
                else 'Unemployed' if 'unemployed' in str(x).lower()
                else 'Other'
            )
        
        # Education level
        if 'Degree' in self.df.columns:
            self.df['education_level'] = self.df['Degree'].apply(
                lambda x: 'High School' if 'high school' in str(x).lower()
                else 'Bachelor' if 'bachelor' in str(x).lower()
                else 'Master+' if 'master' in str(x).lower() or 'phd' in str(x).lower() or 'md' in str(x).lower()
                else 'Unknown'
            )
        
        print("✓ Created demographic features")
    
    def create_behavioral_features(self) -> None:
        """Create behavioral/psychometric features."""
        print_separator("Creating Behavioral Features")
        
        # Anxiety indicators (from GAD items)
        gad_items = [f'GAD{i}' for i in range(1, 8)]
        if all(col in self.df.columns for col in gad_items):
            self.df['anxiety_frequency'] = self.df[gad_items].mean(axis=1)
        
        # Social phobia indicators (from SPIN items)
        spin_items = [f'SPIN{i}' for i in range(1, 18)]
        spin_items = [col for col in spin_items if col in self.df.columns]
        if len(spin_items) > 0:
            self.df['social_fear_avg'] = self.df[spin_items].mean(axis=1)
        
        # Gaming purpose indicators
        if 'whyplay' in self.df.columns:
            self.df['plays_for_social'] = self.df['whyplay'].str.contains(
                'social|friend', case=False, na=False
            ).astype(int)
            
            self.df['plays_for_competition'] = self.df['whyplay'].str.contains(
                'compet|improv|win', case=False, na=False
            ).astype(int)
        
        print("✓ Created behavioral features")
    
    def select_and_encode_features(self) -> tuple:
        """
        Select features and handle encoding.
        
        Returns:
            tuple: (X, y, feature_names)
        """
        print_separator("Feature Selection and Encoding")
        
        # Drop original target columns and other non-features
        drop_cols = [
            'GAD_T', 'SWL_T', 'SPIN_T',  # Original scores
            'GAD_risk', 'SWL_risk', 'SPIN_risk',  # Risk categories
            'mental_health_risk',  # This will be our main target
        ]
        
        # Also drop individual GAD and SPIN items
        gad_items = [f'GAD{i}' for i in range(1, 8)]
        spin_items = [f'SPIN{i}' for i in range(1, 18)]
        swl_items = [f'SWL{i}' for i in range(1, 6)]
        
        drop_cols.extend([col for col in gad_items + spin_items + swl_items if col in self.df.columns])
        
        # Select features
        feature_cols = [col for col in self.df.columns if col not in drop_cols]
        
        X = self.df[feature_cols].copy()
        y = self.df['mental_health_risk'].copy()
        
        # Identify categorical and numerical features
        self.categorical_features = get_categorical_columns(X)
        self.numerical_features = get_numerical_features(X)
        
        print(f"Features selected: {len(feature_cols)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        print(f"Numerical features: {len(self.numerical_features)}")
        
        # Filter categorical features to only those with <= 50 unique values
        categorical_to_encode = []
        for col in self.categorical_features:
            n_unique = X[col].nunique()
            if n_unique <= 50:
                categorical_to_encode.append(col)
            else:
                X.drop(columns=[col], inplace=True)
        
        print(f"Categorical features to encode: {len(categorical_to_encode)}")
        
        # Handle categorical encoding (one-hot encoding)
        if categorical_to_encode:
            print("Encoding categorical features...")
            X_encoded = pd.get_dummies(
                X,
                columns=categorical_to_encode,
                drop_first=True
            )
        else:
            X_encoded = X
        
        # Clean feature names (replace all special characters for LightGBM compatibility)
        import re
        X_encoded.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_encoded.columns]
        
        print(f"Features after encoding: {X_encoded.shape[1]}")
        
        return X_encoded, y, X_encoded.columns.tolist()
    
    def engineer_features(self) -> tuple:
        """
        Execute full feature engineering pipeline.
        
        Returns:
            tuple: (X_engineered, y, feature_names)
        """
        print_separator("STARTING FEATURE ENGINEERING")
        
        self.create_target_variables()
        self.create_gaming_features()
        self.create_demographic_features()
        self.create_behavioral_features()
        
        X, y, feature_names = self.select_and_encode_features()
        
        print_separator("Feature Engineering Complete")
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}\n")
        
        self.X = X
        self.y = y
        
        return X, y, feature_names
    
    def save_engineered_data(self) -> None:
        """Save engineered features and target."""
        if self.X is not None and self.y is not None:
            # Create combined dataframe for reference
            combined = self.X.copy()
            combined['mental_health_risk'] = self.y
            
            output_path = DATA_PROCESSED_PATH / "data_engineered.csv"
            save_dataframe_to_csv(combined, str(output_path))


def get_numerical_features(df: pd.DataFrame) -> list:
    """Get numerical column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def main():
    """Main execution function."""
    from src.data_preprocessing import DataPreprocessor
    
    # Preprocess data first
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.preprocess()
    
    # Engineer features
    engineer = FeatureEngineer(df_cleaned)
    X, y, feature_names = engineer.engineer_features()
    engineer.save_engineered_data()


if __name__ == "__main__":
    main()
