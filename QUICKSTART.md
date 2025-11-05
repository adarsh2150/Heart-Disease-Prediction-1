"""
HEART DISEASE PREDICTION SYSTEM - QUICK START GUIDE
====================================================

This guide will help you get started with the Heart Disease Prediction System quickly.

TABLE OF CONTENTS
-----------------
1. System Requirements
2. Installation
3. Quick Start
4. Running Components
5. Understanding Results
6. Using the Web Application
7. Troubleshooting

================================================================================
1. SYSTEM REQUIREMENTS
================================================================================

- Python 3.8 or higher
- 4GB RAM minimum
- Windows/macOS/Linux
- Internet connection (for initial package installation)

================================================================================
2. INSTALLATION
================================================================================

Step 1: Open Terminal/Command Prompt in the project directory
Step 2: Create a virtual environment (recommended):
   
   python -m venv .venv
   
Step 3: Activate the virtual environment:
   
   Windows: .venv\Scripts\activate
   macOS/Linux: source .venv/bin/activate
   
Step 4: Install dependencies:
   
   pip install -r requirements.txt

================================================================================
3. QUICK START
================================================================================

Option A: Run Everything at Once
---------------------------------
   python main.py

This will:
   ✓ Load and preprocess the dataset
   ✓ Perform feature selection
   ✓ Train all 7 models
   ✓ Generate visualizations
   ✓ Save trained models
   ✓ Create comprehensive reports

Duration: ~2-5 minutes

Option B: Run Components Individually
--------------------------------------
   1. python data_exploration.py       # Explore the dataset
   2. python data_preprocessing.py     # Preprocess data
   3. python feature_selection.py      # Select features
   4. python main.py                   # Train all models

================================================================================
4. RUNNING COMPONENTS
================================================================================

Data Exploration:
-----------------
   python data_exploration.py
   
   Outputs:
   - outputs/class_distribution.png
   - outputs/dataset_info.txt

Data Preprocessing:
-------------------
   python data_preprocessing.py
   
   Creates:
   - Normalized datasets
   - Saved preprocessor in models/

Feature Selection:
------------------
   python feature_selection.py
   
   Generates:
   - outputs/correlation_heatmap.png
   - outputs/feature_importance.png
   - outputs/selected_features.txt

Complete Pipeline:
------------------
   python main.py
   
   Creates:
   - All visualizations in outputs/
   - Trained models in models/
   - Final report in reports/

Web Application:
----------------
   streamlit run app.py
   
   Opens browser at: http://localhost:8501

================================================================================
5. UNDERSTANDING RESULTS
================================================================================

Key Output Files:
-----------------

1. outputs/model_results_complete.csv
   - Performance metrics for all models
   - Compare accuracy, precision, recall, F1-score, ROC-AUC

2. outputs/selected_features.txt
   - List of 30 selected features
   - Reduced from original 55 features

3. reports/final_report.txt
   - Comprehensive summary
   - Best model identification
   - Detailed metrics

4. Visualizations:
   - correlation_heatmap.png: Feature correlations
   - roc_curves_*.png: Model ROC curves
   - confusion_matrices_*.png: Confusion matrices
   - performance_comparison.png: Model comparison

Performance Metrics Explained:
-------------------------------

Accuracy: Overall correctness (correct predictions / total predictions)
Precision: Of predicted positives, how many are correct
Recall: Of actual positives, how many did we catch
F1-Score: Balance between precision and recall
ROC-AUC: Model's ability to distinguish between classes (higher is better)

Best Model: Naive Bayes (Selected Features)
--------------------------------------------
- Accuracy: 86.89%
- Precision: 81.25%
- Recall: 72.22%
- F1-Score: 76.47%
- ROC-AUC: 92.89%

================================================================================
6. USING THE WEB APPLICATION
================================================================================

Starting the App:
-----------------
   streamlit run app.py

The app will open in your browser at http://localhost:8501

Using the Interface:
--------------------

1. Main Tab - Prediction:
   - Enter patient information in the sidebar
   - Fill in all medical parameters
   - Click "Predict Heart Disease"
   - View results and probability scores

2. Model Performance Tab:
   - View all model metrics
   - See visualizations
   - Compare model performances

3. Documentation Tab:
   - Read about the methodology
   - Understand the system
   - View references

Interpreting Results:
---------------------

✅ GREEN (No Heart Disease):
   - Low risk prediction
   - Continue regular check-ups
   - Maintain healthy lifestyle

⚠️ RED (Heart Disease Detected):
   - High risk prediction
   - Consult healthcare professional immediately
   - This is NOT a diagnosis, only a prediction

================================================================================
7. TROUBLESHOOTING
================================================================================

Problem: Module not found errors
Solution:
   pip install --upgrade -r requirements.txt

Problem: Dataset not found
Solution:
   Ensure "data/Z-Alizadeh sani dataset.xlsx" exists
   Check file path and name

Problem: Models not loading in web app
Solution:
   Run "python main.py" first to train and save models

Problem: Port 8501 already in use
Solution:
   streamlit run app.py --server.port 8502

Problem: Permission denied errors
Solution:
   Run as administrator or check file permissions

Problem: Out of memory
Solution:
   - Close other applications
   - Reduce number of estimators in Random Forest
   - Use a machine with more RAM

Problem: Slow execution
Solution:
   - Models are training - this is normal
   - First run takes longer
   - Subsequent predictions are fast

================================================================================
8. FILE STRUCTURE OVERVIEW
================================================================================

RAA/
├── data/
│   └── Z-Alizadeh sani dataset.xlsx    [Dataset file]
│
├── models/                              [Trained models - auto-generated]
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── [model files].pkl
│
├── outputs/                             [Results - auto-generated]
│   ├── [visualizations].png
│   └── model_results_complete.csv
│
├── reports/                             [Reports - auto-generated]
│   └── final_report.txt
│
├── main.py                              [Main pipeline script]
├── data_exploration.py                  [Data exploration]
├── data_preprocessing.py                [Preprocessing module]
├── feature_selection.py                 [Feature selection]
├── model_training.py                    [Model training]
├── app.py                               [Web application]
├── requirements.txt                     [Dependencies]
└── README.md                            [Documentation]

================================================================================
9. NEXT STEPS
================================================================================

After running the system:

1. ✓ Review the final report in reports/final_report.txt
2. ✓ Examine visualizations in outputs/
3. ✓ Test the web application
4. ✓ Experiment with different features
5. ✓ Try adjusting model parameters
6. ✓ Deploy to cloud (optional)

================================================================================
10. SUPPORT & RESOURCES
================================================================================

Documentation:
   - README.md: Comprehensive documentation
   - Code comments: Detailed explanations
   - reports/final_report.txt: Results summary

Research Paper:
   "Heart Disease Prediction Using Distinct Artificial Intelligence Techniques"
   Iran Journal of Computer Science, 2023

Dataset Source:
   UCI Machine Learning Repository
   Z-Alizadeh Sani Heart Disease Dataset

Python Libraries:
   - scikit-learn: Machine learning
   - pandas: Data manipulation
   - streamlit: Web application
   - matplotlib/seaborn: Visualization

================================================================================
11. TIPS FOR BEST RESULTS
================================================================================

1. Data Quality:
   - Ensure accurate patient data entry
   - Complete all required fields
   - Use consistent units of measurement

2. Model Selection:
   - Naive Bayes: Best accuracy (86.89%)
   - Random Forest: Good balance (85.25%)
   - Choose based on your specific needs

3. Feature Importance:
   - Focus on top features: Typical Chest Pain, Age, Atypical
   - These have highest predictive power
   - Ensure accurate measurement of these

4. Regular Updates:
   - Retrain models with new data
   - Update feature selection periodically
   - Monitor model performance

5. Clinical Integration:
   - Use as decision support, not sole diagnosis
   - Combine with clinical expertise
   - Follow up with proper testing

================================================================================

✨ You're all set! Run "python main.py" to get started!

For detailed information, see README.md
For results explanation, see reports/final_report.txt
For web interface, run "streamlit run app.py"

================================================================================
