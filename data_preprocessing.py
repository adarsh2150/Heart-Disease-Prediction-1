"""
Heart Disease Prediction System - Data Preprocessing
Handles missing values, encoding, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Missing value imputation
    - Categorical encoding
    - Min-Max normalization
    - Train-test splitting
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, file_path):
        """
        Load the heart disease dataset
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame containing the dataset
        """
        try:
            df = pd.read_excel(file_path)
            print(f"✓ Dataset loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None
    
    def identify_target_column(self, df):
        """
        Identify the target column in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of target column
        """
        # Common target column names
        possible_targets = ['Cath', 'target', 'diagnosis', 'disease', 'class', 'label']
        
        for col in possible_targets:
            if col in df.columns:
                return col
        
        # If not found, assume last column is target
        return df.columns[-1]
    
    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate strategies
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)
        
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            print("✓ No missing values found!")
            return df
        
        print(f"Missing values found: {missing_count}")
        
        df_clean = df.copy()
        
        # Handle numerical columns: fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  → {col}: Filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")
        
        # Handle categorical columns: fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"  → {col}: Filled {df[col].isnull().sum()} missing values with mode ({mode_val})")
        
        print(f"\n✓ Missing values handled successfully!")
        return df_clean
    
    def encode_categorical_features(self, df, target_col):
        """
        Encode categorical features using Label Encoding
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            DataFrame with encoded features
        """
        print("\n" + "="*80)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*80)
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Encode all categorical columns including target
        if len(categorical_cols) == 0:
            print("✓ No categorical features to encode (all numerical)")
            return df_encoded
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le
            if col == target_col:
                print(f"  → {col} (target): Encoded {len(le.classes_)} unique categories")
            else:
                print(f"  → {col}: Encoded {len(le.classes_)} unique categories")
        
        print(f"\n✓ Encoded {len(categorical_cols)} categorical features")
        return df_encoded
    
    def normalize_features(self, X_train, X_test):
        """
        Normalize features using Min-Max normalization
        
        Args:
            X_train: Training features
            X_test: Testing features
            
        Returns:
            Normalized training and testing features
        """
        print("\n" + "="*80)
        print("NORMALIZING FEATURES (MIN-MAX SCALING)")
        print("="*80)
        
        # Fit scaler on training data only
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_test_normalized = self.scaler.transform(X_test)
        
        print(f"✓ Features normalized to range [0, 1]")
        print(f"  → Training samples: {X_train_normalized.shape[0]}")
        print(f"  → Testing samples: {X_test_normalized.shape[0]}")
        print(f"  → Number of features: {X_train_normalized.shape[1]}")
        
        # Convert back to DataFrame
        X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns, index=X_train.index)
        X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns, index=X_test.index)
        
        return X_train_normalized, X_test_normalized
    
    def prepare_dataset(self, file_path, test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: Path to dataset
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test (normalized)
        """
        print("\n" + "="*80)
        print("DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Identify target column
        target_col = self.identify_target_column(df)
        self.target_name = target_col
        print(f"\n✓ Target column identified: {target_col}")
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, target_col)
        
        # Separate features and target
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]
        
        self.feature_names = X.columns.tolist()
        
        print(f"\n✓ Features and target separated")
        print(f"  → Features: {X.shape[1]}")
        print(f"  → Target: {target_col}")
        print(f"  → Total samples: {len(y)}")
        print(f"  → Class distribution: {dict(y.value_counts())}")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Data split into train and test sets")
        print(f"  → Training set: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
        print(f"  → Testing set: {len(X_test)} samples ({test_size*100:.0f}%)")
        
        # Normalize features
        X_train_normalized, X_test_normalized = self.normalize_features(X_train, X_test)
        
        print("\n" + "="*80)
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return X_train_normalized, X_test_normalized, y_train, y_test
    
    def save_preprocessor(self, save_dir='models'):
        """
        Save the preprocessor components for later use
        
        Args:
            save_dir: Directory to save the preprocessor
        """
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.scaler, f'{save_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{save_dir}/label_encoders.pkl')
        joblib.dump(self.feature_names, f'{save_dir}/feature_names.pkl')
        joblib.dump(self.target_name, f'{save_dir}/target_name.pkl')
        
        print(f"\n✓ Preprocessor saved to {save_dir}/")
    
    @staticmethod
    def load_preprocessor(save_dir='models'):
        """
        Load a saved preprocessor
        
        Args:
            save_dir: Directory containing the saved preprocessor
            
        Returns:
            DataPreprocessor instance
        """
        preprocessor = DataPreprocessor()
        preprocessor.scaler = joblib.load(f'{save_dir}/scaler.pkl')
        preprocessor.label_encoders = joblib.load(f'{save_dir}/label_encoders.pkl')
        preprocessor.feature_names = joblib.load(f'{save_dir}/feature_names.pkl')
        preprocessor.target_name = joblib.load(f'{save_dir}/target_name.pkl')
        
        return preprocessor

if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = DataPreprocessor()
    
    data_path = "data/Z-Alizadeh sani dataset.xlsx"
    result = preprocessor.prepare_dataset(data_path)
    
    if result is not None:
        X_train, X_test, y_train, y_test = result
        
        print("\n" + "="*80)
        print("FINAL DATASET STATISTICS")
        print("="*80)
        print(f"\nTraining set:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_train distribution: {dict(y_train.value_counts())}")
        
        print(f"\nTesting set:")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  y_test distribution: {dict(y_test.value_counts())}")
        
        # Save preprocessor
        preprocessor.save_preprocessor()
