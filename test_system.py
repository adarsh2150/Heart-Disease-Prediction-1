"""
Heart Disease Prediction System - Testing Script
Tests model predictions and validates functionality
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def test_model_loading():
    """Test if all models load correctly"""
    print("="*80)
    print("TESTING MODEL LOADING")
    print("="*80)
    
    models_to_test = [
        'logistic_regression',
        'naive_bayes',
        'k-nearest_neighbor',
        'support_vector_machine',
        'decision_tree',
        'random_forest',
        'multilayer_perceptron'
    ]
    
    success = True
    for model_name in models_to_test:
        try:
            model = joblib.load(f'models/{model_name}.pkl')
            print(f"✓ {model_name.replace('_', ' ').title()} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            success = False
    
    return success

def test_preprocessor_loading():
    """Test if preprocessor components load correctly"""
    print("\n" + "="*80)
    print("TESTING PREPROCESSOR LOADING")
    print("="*80)
    
    components = {
        'scaler': 'models/scaler.pkl',
        'label_encoders': 'models/label_encoders.pkl',
        'feature_names': 'models/feature_names.pkl',
        'target_name': 'models/target_name.pkl'
    }
    
    success = True
    for name, path in components.items():
        try:
            component = joblib.load(path)
            print(f"✓ {name} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
            success = False
    
    return success

def test_prediction():
    """Test making a prediction"""
    print("\n" + "="*80)
    print("TESTING PREDICTION FUNCTIONALITY")
    print("="*80)
    
    try:
        # Load model and preprocessor
        model = joblib.load('models/naive_bayes.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        # Load selected features
        with open('outputs/selected_features.txt', 'r') as f:
            lines = f.readlines()
            selected_features = []
            for line in lines:
                if '. ' in line and line[0].isdigit():
                    feature = line.split('. ', 1)[1].strip()
                    selected_features.append(feature)
        
        print(f"\n✓ Loaded model: Naive Bayes")
        print(f"✓ Loaded preprocessor components")
        print(f"✓ Number of features: {len(feature_names)}")
        print(f"✓ Selected features: {len(selected_features)}")
        
        # Create sample input (using zeros - not realistic but for testing)
        sample_input = {feature: 0.0 for feature in feature_names}
        
        # Add some realistic values
        if 'Age' in sample_input:
            sample_input['Age'] = 55
        if 'BP' in sample_input:
            sample_input['BP'] = 120
        if 'FBS' in sample_input:
            sample_input['FBS'] = 100
        
        # Convert to DataFrame
        input_df = pd.DataFrame([sample_input])
        
        # Scale
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Select features
        input_final = input_scaled_df[selected_features]
        
        # Predict
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]
        
        print(f"\n✓ Prediction made successfully")
        print(f"  → Prediction: {'Disease' if prediction == 1 else 'No Disease'}")
        print(f"  → Probability (No Disease): {probability[0]*100:.2f}%")
        print(f"  → Probability (Disease): {probability[1]*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_results_files():
    """Test if all result files exist"""
    print("\n" + "="*80)
    print("TESTING OUTPUT FILES")
    print("="*80)
    
    import os
    
    required_files = [
        'outputs/correlation_heatmap.png',
        'outputs/target_correlation.png',
        'outputs/feature_importance.png',
        'outputs/selected_features.txt',
        'outputs/confusion_matrices_all_features.png',
        'outputs/confusion_matrices_selected_features.png',
        'outputs/roc_curves_all_features.png',
        'outputs/roc_curves_selected_features.png',
        'outputs/performance_comparison.png',
        'outputs/model_results_complete.csv',
        'reports/final_report.txt'
    ]
    
    success = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ Missing: {file}")
            success = False
    
    return success

def test_results_accuracy():
    """Test results and display accuracy"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    try:
        results = pd.read_csv('outputs/model_results_complete.csv')
        
        print("\nBest Performance on Selected Features:")
        print("-"*80)
        
        selected_results = results[results['Dataset'] == 'Selected Features']
        selected_sorted = selected_results.sort_values('Accuracy', ascending=False)
        
        print(selected_sorted[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))
        
        best_model = selected_sorted.iloc[0]
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Model: {best_model['Model']}")
        print(f"Accuracy: {best_model['Accuracy']*100:.2f}%")
        print(f"Precision: {best_model['Precision']*100:.2f}%")
        print(f"Recall: {best_model['Recall']*100:.2f}%")
        print(f"F1-Score: {best_model['F1-Score']*100:.2f}%")
        print(f"ROC-AUC: {best_model['ROC-AUC']*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load results: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION SYSTEM - COMPREHENSIVE TESTING")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Preprocessor Loading", test_preprocessor_loading),
        ("Prediction Functionality", test_prediction),
        ("Output Files", test_results_files),
        ("Results Accuracy", test_results_accuracy)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "-"*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%")
    print("-"*80)
    
    if all(results.values()):
        print("\n✓ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT!")
    else:
        print("\n⚠ SOME TESTS FAILED - PLEASE REVIEW ERRORS ABOVE")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    run_all_tests()
