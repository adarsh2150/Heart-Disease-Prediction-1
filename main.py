"""
Heart Disease Prediction System - Main Script
Complete pipeline from data loading to model evaluation

Based on: Heart Disease Prediction Using Distinct Artificial Intelligence Techniques
Dataset: Z-Alizadeh Sani Heart Disease Dataset
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from model_training import ModelTrainer

def create_output_directories():
    """Create necessary output directories"""
    directories = ['outputs', 'models', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Output directories created")

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)

def main():
    """
    Main execution pipeline
    """
    print_header("HEART DISEASE PREDICTION SYSTEM")
    print("Based on: Heart Disease Prediction Using Distinct Artificial Intelligence Techniques")
    print("Dataset: Z-Alizadeh Sani Heart Disease Dataset")
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    create_output_directories()
    
    # Configuration
    DATA_PATH = "data/Z-Alizadeh sani dataset.xlsx"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    NUM_SELECTED_FEATURES = 30
    
    # ========================================================================
    # STEP 1: DATA PREPROCESSING
    # ========================================================================
    print_header("STEP 1: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor()
    result = preprocessor.prepare_dataset(
        file_path=DATA_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    if result is None:
        print("✗ Error in data preprocessing. Exiting...")
        return
    
    X_train_full, X_test_full, y_train, y_test = result
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print(f"\n✓ Preprocessing completed successfully!")
    print(f"  → Training samples: {len(X_train_full)}")
    print(f"  → Testing samples: {len(X_test_full)}")
    print(f"  → Total features: {X_train_full.shape[1]}")
    
    # ========================================================================
    # STEP 2: FEATURE SELECTION
    # ========================================================================
    print_header("STEP 2: FEATURE SELECTION")
    
    selector = FeatureSelector()
    
    # Visualize correlations
    print("\nGenerating correlation visualizations...")
    selector.visualize_correlation_heatmap(X_train_full, y_train)
    selector.visualize_feature_importance(X_train_full, y_train, top_k=30)
    
    # Perform feature selection using Best First Search (simulated with RF importance)
    selected_features = selector.best_first_search(
        X_train_full, 
        y_train, 
        max_features=NUM_SELECTED_FEATURES
    )
    
    # Create reduced datasets
    X_train_selected = X_train_full[selected_features]
    X_test_selected = X_test_full[selected_features]
    
    print(f"\n✓ Feature selection completed!")
    print(f"  → Original features: {X_train_full.shape[1]}")
    print(f"  → Selected features: {len(selected_features)}")
    print(f"  → Reduction: {(1 - len(selected_features)/X_train_full.shape[1])*100:.1f}%")
    
    # Save selected features
    with open('outputs/selected_features.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SELECTED FEATURES (CFS with Best First Search)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total features selected: {len(selected_features)}\n")
        f.write(f"Original features: {X_train_full.shape[1]}\n")
        f.write(f"Reduction: {(1 - len(selected_features)/X_train_full.shape[1])*100:.1f}%\n\n")
        f.write("Selected Features:\n")
        f.write("-"*80 + "\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i:2d}. {feat}\n")
    
    # ========================================================================
    # STEP 3: MODEL TRAINING AND EVALUATION - ALL FEATURES
    # ========================================================================
    print_header("STEP 3: MODEL TRAINING - ALL FEATURES")
    
    trainer_full = ModelTrainer()
    trainer_full.initialize_models()
    
    results_full = trainer_full.train_and_evaluate_all(
        X_train_full, X_test_full, y_train, y_test,
        dataset_type="All Features"
    )
    
    # Generate visualizations for all features
    trainer_full.plot_confusion_matrices(
        y_test, 
        dataset_type="All Features",
        save_path='outputs/confusion_matrices_all_features.png'
    )
    
    trainer_full.plot_roc_curves(
        dataset_type="All Features",
        save_path='outputs/roc_curves_all_features.png'
    )
    
    # ========================================================================
    # STEP 4: MODEL TRAINING AND EVALUATION - SELECTED FEATURES
    # ========================================================================
    print_header("STEP 4: MODEL TRAINING - SELECTED FEATURES")
    
    trainer_selected = ModelTrainer()
    trainer_selected.initialize_models()
    
    results_selected = trainer_selected.train_and_evaluate_all(
        X_train_selected, X_test_selected, y_train, y_test,
        dataset_type="Selected Features"
    )
    
    # Generate visualizations for selected features
    trainer_selected.plot_confusion_matrices(
        y_test,
        dataset_type="Selected Features",
        save_path='outputs/confusion_matrices_selected_features.png'
    )
    
    trainer_selected.plot_roc_curves(
        dataset_type="Selected Features",
        save_path='outputs/roc_curves_selected_features.png'
    )
    
    # ========================================================================
    # STEP 5: COMPREHENSIVE COMPARISON
    # ========================================================================
    print_header("STEP 5: COMPREHENSIVE COMPARISON")
    
    # Combine results
    results_combined = pd.concat([results_full, results_selected], ignore_index=True)
    
    # Plot performance comparison
    trainer_full.plot_performance_comparison(
        results_combined,
        save_path='outputs/performance_comparison.png'
    )
    
    # Save combined results
    results_combined.to_csv('outputs/model_results_complete.csv', index=False)
    print("✓ Combined results saved to: outputs/model_results_complete.csv")
    
    # Display final comparison table
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    print(results_combined.to_string(index=False))
    print("="*80)
    
    # ========================================================================
    # STEP 6: IDENTIFY BEST MODEL
    # ========================================================================
    print_header("STEP 6: BEST MODEL IDENTIFICATION")
    
    best_model_name = trainer_full.get_best_model(results_combined, metric='Accuracy')
    
    # Get best model performance for all metrics
    best_model_data = results_combined[results_combined['Model'] == best_model_name]
    
    print("\nDetailed Performance:")
    print("-"*80)
    for _, row in best_model_data.iterrows():
        print(f"\nDataset: {row['Dataset']}")
        print(f"  Accuracy:  {row['Accuracy']:.4f}")
        print(f"  Precision: {row['Precision']:.4f}")
        print(f"  Recall:    {row['Recall']:.4f}")
        print(f"  F1-Score:  {row['F1-Score']:.4f}")
        print(f"  ROC-AUC:   {row['ROC-AUC']:.4f}")
    
    # ========================================================================
    # STEP 7: SAVE MODELS
    # ========================================================================
    print_header("STEP 7: SAVING MODELS")
    
    # Save models from selected features (preferred for deployment)
    trainer_selected.save_models(save_dir='models')
    
    # Save the best model separately for easy access
    best_model_filename = best_model_name.lower().replace(' ', '_')
    import joblib
    joblib.dump(
        trainer_selected.models[best_model_name],
        f'models/best_model_{best_model_filename}.pkl'
    )
    print(f"✓ Best model saved separately: models/best_model_{best_model_filename}.pkl")
    
    # ========================================================================
    # STEP 8: GENERATE FINAL REPORT
    # ========================================================================
    print_header("STEP 8: GENERATING FINAL REPORT")
    
    report_path = 'reports/final_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HEART DISEASE PREDICTION SYSTEM - FINAL REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Z-Alizadeh Sani Heart Disease Dataset\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Samples: {len(X_train_full) + len(X_test_full)}\n")
        f.write(f"Training Samples: {len(X_train_full)} ({(1-TEST_SIZE)*100:.0f}%)\n")
        f.write(f"Testing Samples: {len(X_test_full)} ({TEST_SIZE*100:.0f}%)\n")
        f.write(f"Total Features: {X_train_full.shape[1]}\n")
        f.write(f"Selected Features: {len(selected_features)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("MODEL PERFORMANCE RESULTS\n")
        f.write("-"*80 + "\n\n")
        f.write(results_combined.to_string(index=False))
        f.write("\n\n")
        
        f.write("-"*80 + "\n")
        f.write("BEST PERFORMING MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Model: {best_model_name}\n")
        for _, row in best_model_data.iterrows():
            f.write(f"\nDataset: {row['Dataset']}\n")
            f.write(f"  Accuracy:  {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {row['Precision']:.4f} ({row['Precision']*100:.2f}%)\n")
            f.write(f"  Recall:    {row['Recall']:.4f} ({row['Recall']*100:.2f}%)\n")
            f.write(f"  F1-Score:  {row['F1-Score']:.4f} ({row['F1-Score']*100:.2f}%)\n")
            f.write(f"  ROC-AUC:   {row['ROC-AUC']:.4f} ({row['ROC-AUC']*100:.2f}%)\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("SELECTED FEATURES\n")
        f.write("-"*80 + "\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"{i:2d}. {feat}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Final report saved to: {report_path}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print_header("EXECUTION COMPLETED SUCCESSFULLY!")
    
    print("\n📊 Generated Files:")
    print("  → outputs/correlation_heatmap.png")
    print("  → outputs/target_correlation.png")
    print("  → outputs/feature_importance.png")
    print("  → outputs/selected_features.txt")
    print("  → outputs/confusion_matrices_all_features.png")
    print("  → outputs/confusion_matrices_selected_features.png")
    print("  → outputs/roc_curves_all_features.png")
    print("  → outputs/roc_curves_selected_features.png")
    print("  → outputs/performance_comparison.png")
    print("  → outputs/model_results_complete.csv")
    print("  → reports/final_report.txt")
    print("  → models/ (all trained models)")
    
    print(f"\n⏱️  Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ Ready for web application deployment!")
    
    return results_combined, best_model_name

if __name__ == "__main__":
    results, best_model = main()
