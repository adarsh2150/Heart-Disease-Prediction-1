"""
Heart Disease Prediction System - Model Training and Evaluation
Implements 7 machine learning models with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class ModelTrainer:
    """
    Trains and evaluates multiple machine learning models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.predictions = {}
        self.roc_data = {}
        
    def initialize_models(self):
        """
        Initialize all 7 machine learning models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='lbfgs'
            ),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbor': KNeighborsClassifier(
                n_neighbors=5,
                metric='minkowski'
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=15,
                min_samples_split=5,
                n_jobs=-1
            ),
            'Multilayer Perceptron': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                random_state=42,
                max_iter=500,
                early_stopping=True
            )
        }
        
        print("✓ Initialized 7 machine learning models:")
        for i, name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {name}")
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a single model
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n  Training {model_name}...", end=' ')
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        print("✓")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name: Name of the model
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Prediction probabilities for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Store predictions and ROC data
        self.predictions[model_name] = y_pred
        
        # Calculate ROC curve data
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            self.roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        return metrics
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test, dataset_type="All Features"):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            dataset_type: Type of dataset (All Features or Selected Features)
            
        Returns:
            DataFrame with results
        """
        print("\n" + "="*80)
        print(f"TRAINING AND EVALUATING MODELS - {dataset_type}")
        print("="*80)
        
        results_list = []
        
        for model_name in self.models.keys():
            # Train model
            self.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate_model(model_name, X_test, y_test)
            
            # Store results
            result = {
                'Model': model_name,
                'Dataset': dataset_type,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score'],
                'ROC-AUC': metrics['ROC-AUC']
            }
            results_list.append(result)
            
            # Store in instance variable
            self.results[f"{model_name}_{dataset_type}"] = metrics
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Display results
        print("\n" + "-"*80)
        print("RESULTS SUMMARY")
        print("-"*80)
        print(results_df.to_string(index=False))
        print("-"*80)
        
        return results_df
    
    def plot_confusion_matrices(self, y_test, dataset_type="All Features", save_path=None):
        """
        Plot confusion matrices for all models
        
        Args:
            y_test: True labels
            dataset_type: Type of dataset
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, model_name in enumerate(self.models.keys()):
            key = f"{model_name}_{dataset_type}"
            
            if key in self.results:
                cm = self.results[key]['Confusion Matrix']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[idx], cbar=False,
                           xticklabels=['No Disease', 'Disease'],
                           yticklabels=['No Disease', 'Disease'])
                
                axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
                axes[idx].set_xlabel('Predicted', fontsize=10)
                axes[idx].set_ylabel('Actual', fontsize=10)
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle(f'Confusion Matrices - {dataset_type}', 
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrices saved to: {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self, dataset_type="All Features", save_path=None):
        """
        Plot ROC curves for all models
        
        Args:
            dataset_type: Type of dataset
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        
        for idx, model_name in enumerate(self.models.keys()):
            if model_name in self.roc_data:
                fpr = self.roc_data[model_name]['fpr']
                tpr = self.roc_data[model_name]['tpr']
                roc_auc = self.roc_data[model_name]['auc']
                
                plt.plot(fpr, tpr, color=colors[idx], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves - {dataset_type}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curves saved to: {save_path}")
        
        plt.close()
    
    def plot_performance_comparison(self, results_df, save_path=None):
        """
        Plot performance comparison bar chart
        
        Args:
            results_df: DataFrame with results
            save_path: Path to save the plot
        """
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            data = results_df.pivot(index='Model', columns='Dataset', values=metric)
            
            data.plot(kind='bar', ax=ax, width=0.8, color=['#3498db', '#e74c3c'])
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.legend(title='Dataset', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=7)
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Performance comparison saved to: {save_path}")
        
        plt.close()
    
    def save_models(self, save_dir='models'):
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_')
            joblib.dump(model, f'{save_dir}/{filename}.pkl')
        
        print(f"\n✓ All models saved to {save_dir}/")
    
    def save_results(self, results_df, save_path='outputs/model_results.csv'):
        """
        Save results to CSV
        
        Args:
            results_df: DataFrame with results
            save_path: Path to save the CSV
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path, index=False)
        print(f"✓ Results saved to: {save_path}")
    
    def get_best_model(self, results_df, metric='Accuracy'):
        """
        Identify the best performing model
        
        Args:
            results_df: DataFrame with results
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model
        """
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        best_dataset = results_df.loc[best_idx, 'Dataset']
        
        print("\n" + "="*80)
        print("BEST PERFORMING MODEL")
        print("="*80)
        print(f"Model: {best_model}")
        print(f"Dataset: {best_dataset}")
        print(f"{metric}: {best_score:.4f}")
        print("="*80)
        
        return best_model

if __name__ == "__main__":
    # This will be called from main script
    pass
