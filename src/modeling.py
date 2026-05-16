"""
Modeling module for the Mental Health Risk Prediction project.

This module handles model training, evaluation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.config import MODEL_CONFIG, HYPERPARAMETER_SPACE, DATA_PROCESSED_PATH
from src.utils import print_separator, save_dataframe_to_csv
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            random_state (int): Random state for reproducibility.
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def split_data(self, test_size: float = 0.2) -> None:
        """
        Split data into train and test sets (stratified).
        
        Args:
            test_size (float): Test set size. Defaults to 0.2.
        """
        print_separator("Data Splitting")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"\nTarget distribution in training set:")
        print(self.y_train.value_counts())
    
    def train_logistic_regression(self) -> None:
        """Train Logistic Regression model."""
        print_separator("Training Logistic Regression")
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**MODEL_CONFIG['logistic_regression']))
        ])
        
        pipe.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = pipe
        
        self._evaluate_model('Logistic Regression', pipe)
    
    def train_random_forest(self) -> None:
        """Train Random Forest model."""
        print_separator("Training Random Forest")
        
        clf = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
        clf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = clf
        
        self._evaluate_model('Random Forest', clf)
    
    def train_svm(self) -> None:
        """Train Support Vector Machine model."""
        print_separator("Training Support Vector Machine")
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(**MODEL_CONFIG['svm'], probability=True))
        ])
        
        pipe.fit(self.X_train, self.y_train)
        self.models['SVM'] = pipe
        
        self._evaluate_model('SVM', pipe)
    
    def train_xgboost(self) -> None:
        """Train XGBoost model."""
        print_separator("Training XGBoost")
        
        # Create label encoding for target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(self.y_train)
        y_test_encoded = le.transform(self.y_test)
        
        clf = xgb.XGBClassifier(**MODEL_CONFIG['xgboost'])
        clf.fit(self.X_train, y_train_encoded)
        self.models['XGBoost'] = (clf, le)
        
        # For evaluation, use encoded labels
        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(y_test_encoded, y_pred)
        print(f"XGBoost - Accuracy: {acc:.4f}")
    
    def train_lightgbm(self) -> None:
        """Train LightGBM model."""
        print_separator("Training LightGBM")
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(self.y_train)
        y_test_encoded = le.transform(self.y_test)
        
        clf = lgb.LGBMClassifier(**MODEL_CONFIG['lightgbm'])
        clf.fit(self.X_train, y_train_encoded)
        self.models['LightGBM'] = (clf, le)
        
        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(y_test_encoded, y_pred)
        print(f"LightGBM - Accuracy: {acc:.4f}")
    
    def train_catboost(self) -> None:
        """Train CatBoost model."""
        print_separator("Training CatBoost")
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(self.y_train)
        y_test_encoded = le.transform(self.y_test)
        
        clf = CatBoostClassifier(**MODEL_CONFIG['catboost'])
        clf.fit(self.X_train, y_train_encoded, verbose=False)
        self.models['CatBoost'] = (clf, le)
        
        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(y_test_encoded, y_pred)
        print(f"CatBoost - Accuracy: {acc:.4f}")
    
    def _evaluate_model(self, model_name: str, model) -> dict:
        """
        Evaluate a model on test set.
        
        Args:
            model_name (str): Name of the model.
            model: Trained model.
            
        Returns:
            dict: Evaluation metrics.
        """
        y_pred = model.predict(self.X_test)
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(self.y_test, y_pred, average='macro', zero_division=0),
        }
        
        self.results[model_name] = metrics
        
        print(f"\nMetrics:")
        for key, value in metrics.items():
            if key != 'Model':
                print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def train_all_models(self) -> None:
        """Train all models."""
        print_separator("TRAINING ALL MODELS")
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        self.train_lightgbm()
        self.train_catboost()
        
        print_separator("All Models Trained")
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate and compare all models.
        
        Returns:
            pd.DataFrame: Comparison table.
        """
        print_separator("MODEL COMPARISON")
        
        results_df = pd.DataFrame(list(self.results.values()))
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(results_df.to_string(index=False))
        
        # Find best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models.get(self.best_model_name)
        
        print(f"\n✓ Best Model: {self.best_model_name}")
        
        return results_df
    
    def save_results(self) -> None:
        """Save model comparison results."""
        results_df = pd.DataFrame(list(self.results.values()))
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        output_path = DATA_PROCESSED_PATH / "model_comparison.csv"
        save_dataframe_to_csv(results_df, str(output_path))


def main():
    """Main execution function."""
    from src.feature_engineering import FeatureEngineer
    from src.data_preprocessing import DataPreprocessor
    import warnings
    warnings.filterwarnings('ignore')
    
    # Preprocess and engineer features
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.preprocess()
    
    engineer = FeatureEngineer(df_cleaned)
    X, y, feature_names = engineer.engineer_features()
    
    # Train models
    trainer = ModelTrainer(X, y)
    trainer.split_data()
    trainer.train_all_models()
    results = trainer.evaluate_all_models()
    trainer.save_results()


if __name__ == "__main__":
    main()
