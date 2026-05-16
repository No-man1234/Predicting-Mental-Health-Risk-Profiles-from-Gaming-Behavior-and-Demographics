"""
Utility functions for the Mental Health Risk Prediction project.

This module provides utility functions for logging, data handling,
and common operations used throughout the project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_file (str, optional): Path to log file. Defaults to None.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def print_separator(title: str = None, width: int = 80) -> None:
    """
    Print a formatted separator line.
    
    Args:
        title (str, optional): Title to display. Defaults to None.
        width (int): Width of separator. Defaults to 80.
    """
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def print_data_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print comprehensive information about a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        name (str): Name of the dataset. Defaults to "Dataset".
    """
    print_separator(f"{name} Information")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        
    Returns:
        pd.DataFrame: DataFrame with missing value statistics.
    """
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0]
    missing_data = missing_data.sort_values('Missing_Percentage', ascending=False)
    
    return missing_data


def get_categorical_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Identify categorical columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        exclude (list, optional): Columns to exclude. Defaults to None.
        
    Returns:
        list: List of categorical column names.
    """
    if exclude is None:
        exclude = []
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return [col for col in categorical_cols if col not in exclude]


def get_numerical_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Identify numerical columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        exclude (list, optional): Columns to exclude. Defaults to None.
        
    Returns:
        list: List of numerical column names.
    """
    if exclude is None:
        exclude = []
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numerical_cols if col not in exclude]


def save_dataframe_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save a DataFrame to CSV with consistent formatting.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Path to save to.
        index (bool): Whether to save index. Defaults to False.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"✓ Saved: {path}")


def load_csv_file(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    
    Args:
        path (str): Path to CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(path, index_col=0)
    print(f"✓ Loaded: {path}")
    return df


def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove specified columns from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to modify.
        columns (list): Columns to remove.
        
    Returns:
        pd.DataFrame: DataFrame with columns removed.
    """
    available_cols = [col for col in columns if col in df.columns]
    return df.drop(columns=available_cols)


def fillna_strategy(df: pd.DataFrame, strategy: str = 'mean', 
                   numerical_cols: list = None, categorical_cols: list = None) -> pd.DataFrame:
    """
    Fill missing values using specified strategy.
    
    Args:
        df (pd.DataFrame): DataFrame to fill.
        strategy (str): Strategy to use ('mean', 'median', 'mode', 'forward_fill'). Defaults to 'mean'.
        numerical_cols (list, optional): Numerical columns to fill. Defaults to None.
        categorical_cols (list, optional): Categorical columns to fill. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with filled values.
    """
    df_filled = df.copy()
    
    if numerical_cols is None:
        numerical_cols = get_numerical_columns(df)
    if categorical_cols is None:
        categorical_cols = get_categorical_columns(df)
    
    if strategy == 'mean':
        df_filled[numerical_cols] = df_filled[numerical_cols].fillna(df_filled[numerical_cols].mean())
    elif strategy == 'median':
        df_filled[numerical_cols] = df_filled[numerical_cols].fillna(df_filled[numerical_cols].median())
    elif strategy == 'mode':
        for col in numerical_cols:
            df_filled[col].fillna(df_filled[col].mode()[0] if not df_filled[col].mode().empty else df_filled[col].mean(), inplace=True)
    elif strategy == 'forward_fill':
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
    
    # Fill categorical with mode
    for col in categorical_cols:
        if df_filled[col].isnull().sum() > 0:
            mode_val = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
            df_filled[col].fillna(mode_val, inplace=True)
    
    return df_filled


def create_risk_categories(scores: pd.Series, thresholds: dict) -> pd.Series:
    """
    Convert numerical scores to risk categories.
    
    Args:
        scores (pd.Series): Numerical scores.
        thresholds (dict): Dictionary of category thresholds.
        
    Returns:
        pd.Series: Risk categories.
    """
    categories = []
    
    # For this specific use case, we'll use quantile-based approach
    # to handle class imbalance better
    q33 = scores.quantile(0.33)
    q66 = scores.quantile(0.66)
    
    categories = pd.cut(scores, 
                       bins=[scores.min() - 1, q33, q66, scores.max()],
                       labels=["Low Risk", "Moderate Risk", "High Risk"],
                       include_lowest=True)
    
    return categories


def generate_timestamp() -> str:
    """Generate a timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
