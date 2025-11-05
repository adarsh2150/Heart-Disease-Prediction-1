"""
Heart Disease Prediction System - Data Exploration
Based on: Heart Disease Prediction Using Distinct Artificial Intelligence Techniques
Dataset: Z-Alizadeh Sani Heart Disease Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    """
    Load the Z-Alizadeh Sani Heart Disease Dataset
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """
    Perform exploratory data analysis
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*80)
    print("DATASET EXPLORATION")
    print("="*80)
    
    # Basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1] - 1}")  # Excluding target
    print(f"Number of Samples: {df.shape[0]}")
    
    # Data types
    print("\n" + "-"*80)
    print("Data Types:")
    print("-"*80)
    print(df.dtypes)
    
    # Missing values
    print("\n" + "-"*80)
    print("Missing Values:")
    print("-"*80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    # Statistical summary
    print("\n" + "-"*80)
    print("Statistical Summary:")
    print("-"*80)
    print(df.describe())
    
    # Target variable distribution
    if 'Cath' in df.columns:
        target_col = 'Cath'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        # Find the likely target column (usually last column or contains 'disease', 'diagnosis', etc.)
        target_col = df.columns[-1]
    
    print("\n" + "-"*80)
    print(f"Target Variable Distribution ({target_col}):")
    print("-"*80)
    print(df[target_col].value_counts())
    print(f"\nClass Distribution (%):")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    return target_col

def visualize_class_distribution(df, target_col, save_path='outputs/class_distribution.png'):
    """
    Visualize the distribution of the target variable
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        save_path: Path to save the plot
    """
    import os
    os.makedirs('outputs', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    counts = df[target_col].value_counts()
    
    plt.subplot(1, 2, 1)
    counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], startangle=90)
    plt.title('Class Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nClass distribution plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Load dataset
    data_path = "data/Z-Alizadeh sani dataset.xlsx"
    df = load_dataset(data_path)
    
    if df is not None:
        # Explore dataset
        target_col = explore_dataset(df)
        
        # Visualize class distribution
        visualize_class_distribution(df, target_col)
        
        # Save dataset info to file
        with open('outputs/dataset_info.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("Z-ALIZADEH SANI HEART DISEASE DATASET INFORMATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset Shape: {df.shape}\n")
            f.write(f"Number of Features: {df.shape[1] - 1}\n")
            f.write(f"Number of Samples: {df.shape[0]}\n\n")
            f.write("Columns:\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i}. {col}\n")
        
        print("\nDataset information saved to: outputs/dataset_info.txt")
