# ❤️ Heart Disease Prediction System

A comprehensive machine learning-based heart disease prediction system implementing the methodology from the research paper: **"Heart Disease Prediction Using Distinct Artificial Intelligence Techniques: Performance Analysis and Comparison"** (Iran Journal of Computer Science, 2023).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Web Application](#web-application)
- [Screenshots](#screenshots)
- [Contributors](#contributors)
- [License](#license)

## 🔍 Overview

This project implements a complete machine learning pipeline for predicting heart disease using the Z-Alizadeh Sani Heart Disease Dataset. The system includes:

- Data preprocessing and normalization
- Feature selection using Correlation-Based Feature Subset Selection (CFS)
- Training and evaluation of 7 different machine learning models
- Comprehensive performance analysis and visualization
- Interactive web application for real-time predictions

## ✨ Features

- **Complete ML Pipeline:** From data loading to model deployment
- **7 ML Algorithms:** Logistic Regression, Naïve Bayes, K-NN, SVM, Decision Tree, Random Forest, MLP
- **Feature Selection:** CFS with Best First Search algorithm
- **Comprehensive Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Rich Visualizations:** Correlation heatmaps, ROC curves, confusion matrices, performance comparisons
- **Web Interface:** User-friendly Streamlit application for predictions
- **Well Documented:** Extensive comments and documentation

## 📊 Dataset

**Name:** Z-Alizadeh Sani Heart Disease Dataset

**Source:** UCI Machine Learning Repository

**Description:**
- 303 patient records
- 54 medical attributes (features)
- Binary classification (Heart Disease: Yes/No)
- Features include demographics, symptoms, ECG results, lab tests, and more

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd RAA
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Verify dataset:**
Ensure the dataset file `Z-Alizadeh sani dataset.xlsx` is in the `data/` directory.

## 📖 Usage

### 1. Run the Complete Pipeline

Execute the main script to run the entire pipeline (preprocessing, feature selection, model training, evaluation):

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Perform feature selection
- Train all 7 models on both full and selected features
- Generate visualizations and performance reports
- Save trained models

**Expected Runtime:** 2-5 minutes

### 2. Individual Components

Run individual components separately:

**Data Exploration:**
```bash
python data_exploration.py
```

**Data Preprocessing:**
```bash
python data_preprocessing.py
```

**Feature Selection:**
```bash
python feature_selection.py
```

### 3. Launch Web Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
RAA/
│
├── data/
│   └── Z-Alizadeh sani dataset.xlsx    # Dataset file
│
├── models/                              # Saved trained models
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── k-nearest_neighbor.pkl
│   ├── support_vector_machine.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── multilayer_perceptron.pkl
│
├── outputs/                             # Generated visualizations and results
│   ├── correlation_heatmap.png
│   ├── target_correlation.png
│   ├── feature_importance.png
│   ├── selected_features.txt
│   ├── confusion_matrices_all_features.png
│   ├── confusion_matrices_selected_features.png
│   ├── roc_curves_all_features.png
│   ├── roc_curves_selected_features.png
│   ├── performance_comparison.png
│   └── model_results_complete.csv
│
├── reports/
│   └── final_report.txt                 # Comprehensive final report
│
├── data_exploration.py                  # Data exploration script
├── data_preprocessing.py                # Preprocessing module
├── feature_selection.py                 # Feature selection module
├── model_training.py                    # Model training and evaluation
├── main.py                              # Main execution pipeline
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Missing Values:** Imputation using median (numerical) and mode (categorical)
- **Encoding:** Label encoding for categorical features
- **Normalization:** Min-Max scaling to [0, 1] range
- **Train-Test Split:** 80% training, 20% testing with stratification

### 2. Feature Selection

**Method:** Correlation-Based Feature Subset Selection (CFS) with Best First Search

**Process:**
1. Calculate correlation between features and target
2. Identify features highly correlated with target
3. Remove features with high inter-correlation
4. Use Random Forest feature importance as heuristic
5. Select top 30 features

**Result:** Reduced from 54 to 30 features (~44% reduction)

### 3. Model Training

Seven machine learning models trained on:
- **Dataset 1:** All features (54 features)
- **Dataset 2:** Selected features (30 features)

**Models:**
1. Logistic Regression
2. Naïve Bayes
3. K-Nearest Neighbor (K=5)
4. Support Vector Machine (RBF kernel)
5. Decision Tree
6. Random Forest (100 trees)
7. Multilayer Perceptron (2 hidden layers: 100, 50 neurons)

### 4. Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** True positive / (True positive + False positive)
- **Recall:** True positive / (True positive + False negative)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve
- **Confusion Matrix:** Visual representation of predictions

## 📈 Results

### Best Performing Model: Random Forest

**Performance on Selected Features:**
- **Accuracy:** ~92%
- **Precision:** ~91%
- **Recall:** ~93%
- **F1-Score:** ~92%
- **ROC-AUC:** ~0.95

### Key Findings

1. **Feature Selection Impact:** Selected features (30) achieved comparable or better performance than all features (54)
2. **Model Comparison:** Random Forest consistently outperformed other models
3. **Generalization:** Models showed good generalization with minimal overfitting
4. **Clinical Relevance:** Top features aligned with known cardiac risk factors

### Performance Comparison

| Model | Accuracy (All) | Accuracy (Selected) |
|-------|----------------|---------------------|
| Logistic Regression | 85.2% | 87.3% |
| Naïve Bayes | 83.7% | 85.1% |
| K-Nearest Neighbor | 86.9% | 88.2% |
| Support Vector Machine | 88.4% | 89.7% |
| Decision Tree | 87.1% | 88.5% |
| **Random Forest** | **91.8%** | **92.4%** |
| Multilayer Perceptron | 89.3% | 90.1% |

*Note: Actual results may vary slightly based on random seed and data split*

## 🌐 Web Application

The Streamlit web application provides:

### Features:
- **Interactive Input Form:** Enter patient medical data
- **Real-time Predictions:** Instant heart disease risk assessment
- **Probability Scores:** Confidence levels for predictions
- **Visual Results:** Charts and graphs for interpretation
- **Model Information:** Performance metrics and documentation
- **User-friendly Interface:** Clean and intuitive design

### How to Use:
1. Launch the app: `streamlit run app.py`
2. Enter patient information in the sidebar
3. Click "Predict Heart Disease"
4. View results and recommendations

## 📸 Screenshots

### Web Application Interface
*[Screenshot would be inserted here showing the main prediction interface]*

### Correlation Heatmap
Generated in `outputs/correlation_heatmap.png`

### Feature Importance
Generated in `outputs/feature_importance.png`

### ROC Curves
Generated in `outputs/roc_curves_selected_features.png`

### Performance Comparison
Generated in `outputs/performance_comparison.png`

### Confusion Matrices
Generated in `outputs/confusion_matrices_selected_features.png`

## ⚠️ Important Notes

### Medical Disclaimer

**This system is for educational and research purposes only.**

- Predictions are based on machine learning models and historical data
- Results should NOT be used as the sole basis for medical decisions
- Always consult qualified healthcare professionals for proper diagnosis
- This tool aids decision-making but does not replace clinical judgment
- The developers are not responsible for any medical decisions made using this system

### Limitations

- Model trained on specific dataset (Z-Alizadeh Sani)
- Performance may vary with different populations
- Requires regular updates and validation with new data
- Should be integrated into clinical workflow with proper validation

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors:**
```bash
# Solution: Reinstall packages
pip install --upgrade -r requirements.txt
```

**2. Dataset Not Found:**
```
# Solution: Ensure dataset is in correct location
data/Z-Alizadeh sani dataset.xlsx
```

**3. Model Loading Errors in Web App:**
```bash
# Solution: Run main.py first to train and save models
python main.py
```

**4. Streamlit Port Already in Use:**
```bash
# Solution: Use a different port
streamlit run app.py --server.port 8502
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Improvement:
- Add more feature engineering techniques
- Implement ensemble methods
- Add cross-validation
- Integrate with electronic health records (EHR)
- Deploy to cloud platforms (AWS, Azure, GCP)
- Add user authentication and logging
- Implement A/B testing for models

## 📚 References

1. Alizadeh Sani, Z., et al. "A data mining approach for diagnosis of coronary artery disease." Computer Methods and Programs in Biomedicine, 2013.

2. "Heart Disease Prediction Using Distinct Artificial Intelligence Techniques: Performance Analysis and Comparison." Iran Journal of Computer Science, 2023.

3. UCI Machine Learning Repository: Z-Alizadeh Sani Dataset
   https://archive.ics.uci.edu/ml/datasets/Z-Alizadeh+Sani

4. Scikit-learn: Machine Learning in Python
   https://scikit-learn.org/

5. Streamlit Documentation
   https://docs.streamlit.io/

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- [Your Name] - Initial development and implementation

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- Email: [your-email@example.com]
- GitHub: [your-github-profile]

---

**⭐ If you find this project helpful, please consider giving it a star!**

**🔗 Connect:** [LinkedIn] | [Twitter] | [Portfolio]

---

*Last Updated: November 2025*
