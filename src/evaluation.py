"""
Evaluation module for the Mental Health Risk Prediction project.

This module provides detailed model evaluation and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, classification_report,
    roc_curve, auc
)
from src.config import PLOT_CONFIG, DATA_PROCESSED_PATH
from src.utils import print_separator
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Detailed model evaluation."""
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
            model_name (str): Model name.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = None
    
    def get_predictions(self):
        """Get model predictions."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred
    
    def generate_classification_report(self) -> str:
        """Generate classification report."""
        if self.y_pred is None:
            self.get_predictions()
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=['Low Risk', 'Moderate Risk', 'High Risk'],
            digits=4
        )
        
        return report
    
    def generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix."""
        if self.y_pred is None:
            self.get_predictions()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        return cm
    
    def plot_confusion_matrix(self, save_path: str = None) -> None:
        """Plot confusion matrix."""
        cm = self.generate_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 20, save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names (list): List of feature names.
            top_n (int): Top N features to display.
            save_path (str): Path to save plot.
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.model_name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    print_separator("Model Evaluation Complete")


if __name__ == "__main__":
    main()
