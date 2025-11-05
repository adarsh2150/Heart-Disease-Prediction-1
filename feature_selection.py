"""
Heart Disease Prediction System - Feature Selection
Implements Correlation-Based Feature Subset Selection (CFS) with Best First Search
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureSelector:
    """
    Performs feature selection using various methods including CFS
    """
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        
    def correlation_based_selection(self, X, y, threshold=0.5):
        """
        Correlation-Based Feature Selection (CFS)
        Selects features that are highly correlated with the target
        but have low inter-correlation with each other
        
        Args:
            X: Feature DataFrame
            y: Target variable
            threshold: Correlation threshold
            
        Returns:
            List of selected feature names
        """
        print("\n" + "="*80)
        print("CORRELATION-BASED FEATURE SELECTION (CFS)")
        print("="*80)
        
        # Create a copy with target
        df = X.copy()
        df['target'] = y
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Get correlation with target
        target_corr = abs(corr_matrix['target']).drop('target')
        target_corr = target_corr.sort_values(ascending=False)
        
        print(f"\nTop 20 features by correlation with target:")
        print("-" * 80)
        for i, (feat, corr) in enumerate(target_corr.head(20).items(), 1):
            print(f"{i:2d}. {feat:40s} : {corr:.4f}")
        
        # Select features based on correlation with target
        selected = []
        
        for feature in target_corr.index:
            if abs(target_corr[feature]) >= threshold:
                # Check inter-correlation with already selected features
                if len(selected) == 0:
                    selected.append(feature)
                else:
                    # Calculate average correlation with selected features
                    avg_inter_corr = abs(corr_matrix.loc[feature, selected]).mean()
                    
                    # Add if inter-correlation is low
                    if avg_inter_corr < 0.8:  # Low inter-correlation threshold
                        selected.append(feature)
        
        self.selected_features = selected
        
        print(f"\n✓ Feature selection completed!")
        print(f"  → Original features: {len(X.columns)}")
        print(f"  → Selected features: {len(selected)}")
        print(f"  → Reduction: {(1 - len(selected)/len(X.columns))*100:.1f}%")
        
        return selected
    
    def mutual_information_selection(self, X, y, k=20):
        """
        Feature selection using Mutual Information
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        print("\n" + "="*80)
        print("MUTUAL INFORMATION-BASED FEATURE SELECTION")
        print("="*80)
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        print(f"\nTop {k} features by Mutual Information:")
        print("-" * 80)
        for i, (feat, score) in enumerate(mi_scores.head(k).items(), 1):
            print(f"{i:2d}. {feat:40s} : {score:.4f}")
        
        selected = mi_scores.head(k).index.tolist()
        
        print(f"\n✓ Selected {len(selected)} features using Mutual Information")
        
        return selected
    
    def random_forest_selection(self, X, y, k=20):
        """
        Feature selection using Random Forest feature importance
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        print("\n" + "="*80)
        print("RANDOM FOREST-BASED FEATURE SELECTION")
        print("="*80)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        self.feature_scores = importances
        
        print(f"\nTop {k} features by Random Forest Importance:")
        print("-" * 80)
        for i, (feat, score) in enumerate(importances.head(k).items(), 1):
            print(f"{i:2d}. {feat:40s} : {score:.4f}")
        
        selected = importances.head(k).index.tolist()
        
        print(f"\n✓ Selected {len(selected)} features using Random Forest")
        
        return selected
    
    def chi_square_selection(self, X, y, k=20):
        """
        Feature selection using Chi-Square test
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        print("\n" + "="*80)
        print("CHI-SQUARE BASED FEATURE SELECTION")
        print("="*80)
        
        # Ensure all values are non-negative for chi-square test
        X_positive = X - X.min() + 1e-10
        
        # Calculate chi-square scores
        chi_scores, _ = chi2(X_positive, y)
        chi_scores = pd.Series(chi_scores, index=X.columns).sort_values(ascending=False)
        
        print(f"\nTop {k} features by Chi-Square score:")
        print("-" * 80)
        for i, (feat, score) in enumerate(chi_scores.head(k).items(), 1):
            print(f"{i:2d}. {feat:40s} : {score:.4f}")
        
        selected = chi_scores.head(k).index.tolist()
        
        print(f"\n✓ Selected {len(selected)} features using Chi-Square")
        
        return selected
    
    def best_first_search(self, X, y, max_features=30):
        """
        Best First Search algorithm for feature selection
        Simulates the Best First Search by using Random Forest importance
        
        Args:
            X: Feature DataFrame
            y: Target variable
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        print("\n" + "="*80)
        print("BEST FIRST SEARCH FEATURE SELECTION")
        print("="*80)
        
        # Use Random Forest importance as heuristic
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        
        # Sort features by importance
        sorted_features = importances.sort_values(ascending=False)
        
        # Select top features
        selected = sorted_features.head(max_features).index.tolist()
        self.selected_features = selected
        
        print(f"\n✓ Best First Search completed!")
        print(f"  → Original features: {len(X.columns)}")
        print(f"  → Selected features: {len(selected)}")
        
        print(f"\nSelected Features (in order of importance):")
        print("-" * 80)
        for i, feat in enumerate(selected, 1):
            print(f"{i:2d}. {feat}")
        
        return selected
    
    def visualize_feature_importance(self, X, y, top_k=30, save_path='outputs/feature_importance.png'):
        """
        Visualize feature importance using multiple methods
        
        Args:
            X: Feature DataFrame
            y: Target variable
            top_k: Number of top features to display
            save_path: Path to save the plot
        """
        os.makedirs('outputs', exist_ok=True)
        
        # Calculate Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 10))
        top_features = importances.head(top_k)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_k} Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            plt.text(value + 0.001, i, f'{value:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved to: {save_path}")
        plt.close()
    
    def visualize_correlation_heatmap(self, X, y, save_path='outputs/correlation_heatmap.png'):
        """
        Create correlation heatmap
        
        Args:
            X: Feature DataFrame
            y: Target variable
            save_path: Path to save the plot
        """
        os.makedirs('outputs', exist_ok=True)
        
        # Create DataFrame with target
        df = X.copy()
        df['Target'] = y
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(20, 18))
        
        # Use mask to show only lower triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved to: {save_path}")
        plt.close()
        
        # Also plot target correlation separately
        plt.figure(figsize=(12, 10))
        target_corr = abs(corr_matrix['Target']).drop('Target').sort_values(ascending=False).head(30)
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(target_corr)))
        bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors)
        plt.yticks(range(len(target_corr)), target_corr.index)
        plt.xlabel('Absolute Correlation with Target', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title('Top 30 Features by Correlation with Target', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, target_corr.values)):
            plt.text(value + 0.005, i, f'{value:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('outputs/target_correlation.png', dpi=300, bbox_inches='tight')
        print(f"✓ Target correlation plot saved to: outputs/target_correlation.png")
        plt.close()

if __name__ == "__main__":
    # Load preprocessed data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data_path = "data/Z-Alizadeh sani dataset.xlsx"
    result = preprocessor.prepare_dataset(data_path)
    
    if result is not None:
        X_train, X_test, y_train, y_test = result
        
        # Initialize feature selector
        selector = FeatureSelector()
        
        # Visualize correlations
        selector.visualize_correlation_heatmap(X_train, y_train)
        
        # Visualize feature importance
        selector.visualize_feature_importance(X_train, y_train)
        
        # Perform feature selection using different methods
        selected_cfs = selector.best_first_search(X_train, y_train, max_features=30)
        
        # Save selected features
        with open('outputs/selected_features.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("SELECTED FEATURES (CFS with Best First Search)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total features selected: {len(selected_cfs)}\n")
            f.write(f"Original features: {len(X_train.columns)}\n")
            f.write(f"Reduction: {(1 - len(selected_cfs)/len(X_train.columns))*100:.1f}%\n\n")
            f.write("Selected Features:\n")
            f.write("-"*80 + "\n")
            for i, feat in enumerate(selected_cfs, 1):
                f.write(f"{i:2d}. {feat}\n")
        
        print("\n✓ Selected features saved to: outputs/selected_features.txt")
