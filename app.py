"""
Heart Disease Prediction System - Streamlit Web Application
Interactive web interface for heart disease prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .positive {
        background-color: #e74c3c;
        color: white;
    }
    .negative {
        background-color: #2ecc71;
        color: white;
    }
    .info-box {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_preprocessor():
    """
    Load trained models and preprocessor
    """
    try:
        # Load preprocessor components
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
        
        # Load best model (Random Forest)
        model = joblib.load('models/random_forest.pkl')
        
        return model, scaler, feature_names, selected_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def create_input_form(feature_names):
    """
    Create input form for user to enter patient data
    """
    st.sidebar.header("📋 Patient Information")
    
    input_data = {}
    
    # For demonstration, create inputs for common medical features
    # In a real application, you would have all features from the dataset
    
    st.sidebar.subheader("Demographics")
    input_data['Age'] = st.sidebar.number_input(
        "Age", min_value=18, max_value=100, value=50, step=1
    )
    
    st.sidebar.subheader("Clinical Measurements")
    
    # Create inputs for numerical features (using default values)
    # Note: In production, you'd want to create specific inputs for each feature
    # For now, we'll use a simplified approach
    
    for feature in feature_names:
        if feature.lower() == 'age':
            continue  # Already added
        
        # Generate reasonable default values based on feature name
        if 'BP' in feature.upper() or 'pressure' in feature.lower():
            default_val = 120.0
            min_val = 80.0
            max_val = 200.0
        elif 'HR' in feature.upper() or 'heart rate' in feature.lower():
            default_val = 75.0
            min_val = 40.0
            max_val = 150.0
        elif 'sugar' in feature.lower() or 'glucose' in feature.lower():
            default_val = 100.0
            min_val = 50.0
            max_val = 300.0
        elif 'cholesterol' in feature.lower():
            default_val = 200.0
            min_val = 100.0
            max_val = 400.0
        else:
            default_val = 0.0
            min_val = 0.0
            max_val = 100.0
        
        # Simplified input - in production, you'd have more specific inputs
        input_data[feature] = st.sidebar.number_input(
            feature[:40],  # Truncate long feature names
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1,
            key=feature
        )
    
    return input_data

def predict(model, scaler, input_data, feature_names, selected_features):
    """
    Make prediction using the trained model
    """
    try:
        # Create DataFrame with all features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0.0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Select only the features used by the model
        input_final = input_scaled_df[selected_features]
        
        # Make prediction
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def display_results(prediction, probability):
    """
    Display prediction results
    """
    st.header("🔍 Prediction Results")
    
    if prediction == 1:
        st.markdown("""
            <div class="prediction-box positive">
                ⚠️ HEART DISEASE DETECTED
            </div>
        """, unsafe_allow_html=True)
        
        st.error("The model indicates a high risk of heart disease.")
        st.warning("⚕️ Please consult a healthcare professional immediately for proper diagnosis and treatment.")
    else:
        st.markdown("""
            <div class="prediction-box negative">
                ✅ NO HEART DISEASE DETECTED
            </div>
        """, unsafe_allow_html=True)
        
        st.success("The model indicates a low risk of heart disease.")
        st.info("💚 Continue maintaining a healthy lifestyle and regular check-ups.")
    
    # Display probability
    st.subheader("📊 Prediction Confidence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "No Disease Probability",
            f"{probability[0]*100:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Disease Probability",
            f"{probability[1]*100:.2f}%",
            delta=None
        )
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ['No Disease', 'Heart Disease']
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(categories, probability, color=colors, alpha=0.7)
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, prob in zip(bars, probability):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    plt.close()

def display_model_info():
    """
    Display information about the model
    """
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    
    st.sidebar.info("""
    **Heart Disease Prediction System**
    
    This application uses machine learning to predict the likelihood of heart disease based on patient medical data.
    
    **Model:** Random Forest Classifier
    
    **Dataset:** Z-Alizadeh Sani Heart Disease Dataset
    
    **Features:** 30 selected features using Correlation-Based Feature Selection
    
    **Accuracy:** ~90%+
    
    ⚠️ **Disclaimer:** This is a predictive tool and should not replace professional medical advice.
    """)

def main():
    """
    Main application
    """
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Diagnosis Assistant</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        model, scaler, feature_names, selected_features = load_models_and_preprocessor()
    
    if model is None:
        st.error("❌ Failed to load models. Please ensure you have run the training script first.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # Display model info in sidebar
    display_model_info()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏥 Prediction", "📊 Model Performance", "📖 Documentation"])
    
    with tab1:
        st.header("Patient Data Input")
        
        st.info("""
        👨‍⚕️ **Instructions:**
        1. Enter patient information in the sidebar
        2. Adjust all relevant medical parameters
        3. Click 'Predict' to get the diagnosis
        """)
        
        # Create input form
        input_data = create_input_form(feature_names)
        
        # Predict button
        if st.button("🔮 Predict Heart Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                prediction, probability = predict(
                    model, scaler, input_data, feature_names, selected_features
                )
            
            if prediction is not None:
                display_results(prediction, probability)
                
                # Save prediction log
                log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction': 'Disease' if prediction == 1 else 'No Disease',
                    'disease_probability': f"{probability[1]*100:.2f}%"
                }
                st.markdown("---")
                st.caption(f"Prediction made at: {log_entry['timestamp']}")
    
    with tab2:
        st.header("📊 Model Performance Metrics")
        
        try:
            # Load and display results
            results_df = pd.read_csv('outputs/model_results_complete.csv')
            
            st.subheader("Performance Comparison")
            st.dataframe(results_df, use_container_width=True)
            
            # Display visualizations
            st.subheader("Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists('outputs/roc_curves_selected_features.png'):
                    st.image('outputs/roc_curves_selected_features.png', 
                            caption='ROC Curves - Selected Features')
            
            with col2:
                if os.path.exists('outputs/confusion_matrices_selected_features.png'):
                    st.image('outputs/confusion_matrices_selected_features.png',
                            caption='Confusion Matrices - Selected Features')
            
            if os.path.exists('outputs/performance_comparison.png'):
                st.image('outputs/performance_comparison.png',
                        caption='Model Performance Comparison',
                        use_container_width=True)
            
            if os.path.exists('outputs/feature_importance.png'):
                st.image('outputs/feature_importance.png',
                        caption='Feature Importance',
                        use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not load performance metrics: {e}")
    
    with tab3:
        st.header("📖 Documentation")
        
        st.markdown("""
        ## About the System
        
        This Heart Disease Prediction System is based on the research paper:
        **"Heart Disease Prediction Using Distinct Artificial Intelligence Techniques: Performance Analysis and Comparison"**
        published in the Iran Journal of Computer Science, 2023.
        
        ### Dataset
        - **Name:** Z-Alizadeh Sani Heart Disease Dataset
        - **Source:** UCI Machine Learning Repository
        - **Samples:** 303 patients
        - **Features:** 54 medical attributes
        
        ### Methodology
        
        1. **Data Preprocessing:**
           - Missing value imputation
           - Categorical encoding
           - Min-Max normalization (0-1 range)
        
        2. **Feature Selection:**
           - Correlation-Based Feature Subset Selection (CFS)
           - Best First Search algorithm
           - Reduced from 54 to 30 features
        
        3. **Machine Learning Models:**
           - Logistic Regression
           - Naïve Bayes
           - K-Nearest Neighbor (K-NN)
           - Support Vector Machine (SVM)
           - Decision Tree
           - Random Forest (Best Performer)
           - Multilayer Perceptron (MLP)
        
        4. **Evaluation Metrics:**
           - Accuracy
           - Precision
           - Recall
           - F1-Score
           - ROC-AUC Score
           - Confusion Matrix
        
        ### Best Model: Random Forest
        
        The Random Forest classifier achieved the highest performance with:
        - **High Accuracy:** ~90%+
        - **Good Generalization:** Consistent performance on test data
        - **Robust:** Handles complex feature interactions
        
        ### Important Notes
        
        ⚠️ **Medical Disclaimer:**
        - This system is for educational and research purposes
        - Predictions should not replace professional medical diagnosis
        - Always consult qualified healthcare professionals
        - This tool aids decision-making but doesn't provide definitive diagnosis
        
        ### References
        
        1. Heart Disease Prediction Using Distinct Artificial Intelligence Techniques (Iran Journal of Computer Science, 2023)
        2. Z-Alizadeh Sani Dataset - UCI Machine Learning Repository
        3. Scikit-learn Documentation
        4. Streamlit Documentation
        
        ### Credits
        
        Developed as an implementation of the research methodology for heart disease prediction using AI/ML techniques.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #7f8c8d;'>
            <p>Heart Disease Prediction System v1.0</p>
            <p>Powered by Machine Learning & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
