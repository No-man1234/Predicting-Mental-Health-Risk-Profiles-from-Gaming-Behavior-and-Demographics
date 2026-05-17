"""
Visualization module for the Mental Health Risk Prediction project.

This module creates professional visualizations for EDA and results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOT_CONFIG, FIGURES_PATH
from src.utils import print_separator
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Create professional visualizations."""
    
    def __init__(self, df: pd.DataFrame, figsize: tuple = (12, 6), dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            df (pd.DataFrame): Dataset.
            figsize (tuple): Figure size.
            dpi (int): DPI for saving.
        """
        self.df = df
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")
    
    def plot_target_distribution(self, target_col: str, save_path: str = None) -> None:
        """
        Plot target variable distribution.
        
        Args:
            target_col (str): Target column name.
            save_path (str): Path to save plot.
        """
        plt.figure(figsize=self.figsize)
        self.df[target_col].value_counts().plot(kind='bar', color='steelblue')
        plt.title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
        plt.ylabel('Count')
        plt.xlabel(target_col)
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_numerical_distributions(self, columns: list = None, save_path: str = None) -> None:
        """
        Plot distributions of numerical columns.
        
        Args:
            columns (list): Columns to plot.
            save_path (str): Path to save plot.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                axes[idx].hist(self.df[col].dropna(), bins=30, color='steelblue', edgecolor='black')
                axes[idx].set_title(f'{col}', fontweight='bold')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, save_path: str = None) -> None:
        """
        Plot correlation heatmap.
        
        Args:
            save_path (str): Path to save plot.
        """
        numerical_df = self.df.select_dtypes(include=[np.number])
        
        # Select top correlations
        if len(numerical_df.columns) > 15:
            numerical_df = numerical_df.iloc[:, :15]
        
        corr = numerical_df.corr()
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_gaming_hours_by_risk(self, risk_col: str, save_path: str = None) -> None:
        """
        Plot gaming hours by mental health risk.
        
        Args:
            risk_col (str): Risk column name.
            save_path (str): Path to save plot.
        """
        if 'Hours' not in self.df.columns or risk_col not in self.df.columns:
            return
        
        plt.figure(figsize=self.figsize)
        sns.boxplot(data=self.df, x=risk_col, y='Hours', palette='Set2')
        plt.title('Gaming Hours by Mental Health Risk Level', fontsize=14, fontweight='bold')
        plt.ylabel('Hours per Week')
        plt.xlabel('Risk Level')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_age_distribution_by_risk(self, risk_col: str, save_path: str = None) -> None:
        """
        Plot age distribution by risk level.
        
        Args:
            risk_col (str): Risk column name.
            save_path (str): Path to save plot.
        """
        if 'Age' not in self.df.columns or risk_col not in self.df.columns:
            return
        
        plt.figure(figsize=self.figsize)
        sns.violinplot(data=self.df, x=risk_col, y='Age', palette='Set2')
        plt.title('Age Distribution by Mental Health Risk Level', fontsize=14, fontweight='bold')
        plt.ylabel('Age')
        plt.xlabel('Risk Level')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_gender_distribution(self, save_path: str = None) -> None:
        """
        Plot gender distribution.
        
        Args:
            save_path (str): Path to save plot.
        """
        if 'Gender' not in self.df.columns:
            return
        
        plt.figure(figsize=(8, 6))
        self.df['Gender'].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
        plt.title('Distribution of Gender', fontsize=14, fontweight='bold')
        plt.ylabel('Count')
        plt.xlabel('Gender')
        plt.xticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_eda_report(self) -> None:
        """Create comprehensive EDA visualizations."""
        print_separator("Generating EDA Visualizations")
        
        # Target distributions
        for col in ['GAD_T', 'SWL_T', 'SPIN_T']:
            if col in self.df.columns:
                self.plot_target_distribution(col, str(FIGURES_PATH / f'{col}_distribution.png'))
        
        # Numerical distributions
        self.plot_numerical_distributions(
            save_path=str(FIGURES_PATH / 'numerical_distributions.png')
        )
        
        # Correlation heatmap
        self.plot_correlation_heatmap(
            save_path=str(FIGURES_PATH / 'correlation_heatmap.png')
        )
        
        print("✓ EDA visualizations created")


def main():
    """Main execution function."""
    print_separator("Visualization Module Ready")


if __name__ == "__main__":
    main()
