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

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_preprocessor():
    """
    Load trained model(s) and preprocessor(s).

    Priority:
    1. Try to load a full pipeline (pipeline_*.pkl or pipeline.pkl) that includes preprocessing + model.
    2. If not found, load scaler.pkl + random_forest.pkl + feature_names.pkl (legacy behavior).
    """
    pipeline = None
    model = None
    scaler = None
    feature_names = None
    selected_features = None

    # helper to try many pipeline filenames
    pipeline_paths = [
        "models/pipeline_rf.pkl",
        "models/pipeline.pkl",
        "models/pipeline_random_forest.pkl",
        "models/pipeline_rf.joblib",
    ]

    try:
        # 1) attempt to load a pipeline (preferred)
        for p in pipeline_paths:
            if os.path.exists(p):
                pipeline = joblib.load(p)
                break

        # 2) load fallback components if pipeline not found
        if pipeline is None:
            if os.path.exists("models/random_forest.pkl"):
                model = joblib.load("models/random_forest.pkl")
            if os.path.exists("models/scaler.pkl"):
                scaler = joblib.load("models/scaler.pkl")
            if os.path.exists("models/feature_names.pkl"):
                feature_names = joblib.load("models/feature_names.pkl")
            # read selected features (if available)
            if os.path.exists("outputs/selected_features.txt"):
                with open("outputs/selected_features.txt", "r") as f:
                    lines = f.readlines()
                    selected_features = []
                    for line in lines:
                        # preserve how selected_features.txt was formatted in the repo
                        # if lines look like "1. Age" we extract after ". "
                        if ". " in line and line.strip()[0].isdigit():
                            feature = line.split(". ", 1)[1].strip()
                            selected_features.append(feature)
                        else:
                            # fallback: append stripped line if not numbered
                            if line.strip():
                                selected_features.append(line.strip())
            # final fallback: if feature_names was saved but selected_features missing,
            # use first N features or all features
            if feature_names is not None and selected_features is None:
                # if repo documented 30 selected features, try using first 30 else all
                if len(feature_names) >= 30:
                    selected_features = feature_names[:30]
                else:
                    selected_features = list(feature_names)

        return pipeline, model, scaler, feature_names, selected_features
    except Exception as e:
        st.error(f"Error loading models/preprocessor: {e}")
        return None, None, None, None, None


def create_input_form(feature_names):
    """
    Create input form for user to enter patient data.
    Uses feature_names list (raw column names used during training).
    """
    st.sidebar.header("📋 Patient Information")

    input_data = {}

    # Basic Age input (ensure Age present)
    if "Age" in feature_names:
        st.sidebar.subheader("Demographics")
        input_data["Age"] = st.sidebar.number_input(
            "Age", min_value=18, max_value=100, value=50, step=1, key="Age"
        )

    st.sidebar.subheader("Clinical Measurements")

    # For every feature in feature_names (skip Age if already included)
    for feature in feature_names:
        if feature == "Age":
            continue

        # Generate reasonable default values based on feature name heuristics
        ft_lower = feature.lower()
        if "bp" in feature.upper() or "pressure" in ft_lower:
            default_val = 120.0
            min_val = 40.0
            max_val = 300.0
            step = 0.1
            input_data[feature] = st.sidebar.number_input(
                feature, min_value=float(min_val), max_value=float(max_val), value=float(default_val), step=step, key=feature
            )
        elif "hr" in feature.upper() or "heart rate" in ft_lower or "maxhr" in ft_lower:
            default_val = 75.0
            min_val = 20.0
            max_val = 220.0
            step = 0.1
            input_data[feature] = st.sidebar.number_input(
                feature, min_value=float(min_val), max_value=float(max_val), value=float(default_val), step=step, key=feature
            )
        elif "chol" in ft_lower or "ldl" in ft_lower or "hdl" in ft_lower or "tg" in ft_lower:
            default_val = 180.0
            min_val = 10.0
            max_val = 1000.0
            step = 0.1
            input_data[feature] = st.sidebar.number_input(
                feature, min_value=float(min_val), max_value=float(max_val), value=float(default_val), step=step, key=feature
            )
        elif "bmi" in ft_lower or "weight" in ft_lower or "length" in ft_lower:
            default_val = 25.0
            min_val = 0.0
            max_val = 500.0
            step = 0.1
            input_data[feature] = st.sidebar.number_input(
                feature, min_value=float(min_val), max_value=float(max_val), value=float(default_val), step=step, key=feature
            )
        elif any(tok in ft_lower for tok in ["sex", "gender", "smoker", "smoking", "typical", "atypical", "yes", "no"]):
            # small heuristic: present binary / categorical options
            options = ["No", "Yes"] if any(tok in ft_lower for tok in ["smoker", "smoking", "yes", "no"]) else ["M", "F"]
            input_data[feature] = st.sidebar.selectbox(feature, options, index=0, key=feature)
        else:
            # Generic numeric input
            default_val = 0.0
            input_data[feature] = st.sidebar.number_input(
                feature, min_value=-1e6, max_value=1e6, value=float(default_val), step=0.1, key=feature
            )

    return input_data


def predict(pipeline, model, scaler, feature_names, selected_features, input_data):
    """
    Make prediction using either pipeline (preferred) or fallback model+scaler.
    Returns: (prediction_label_int, probability_array)
    """
    try:
        # Build DataFrame from raw input using the same raw feature names
        X_raw = pd.DataFrame([input_data])

        # If a full pipeline is available, use it directly (it handles encoding & scaling)
        if pipeline is not None:
            # For safety, ensure all expected raw features exist in X_raw:
            # If pipeline expects feature_names_in_ (sklearn >= 1.0), check and warn.
            try:
                # Some pipelines store feature names in pipeline.feature_names_in_
                if hasattr(pipeline, "feature_names_in_"):
                    expected_raw = list(pipeline.feature_names_in_)
                    # Add any missing raw features with default 0/No
                    for col in expected_raw:
                        if col not in X_raw.columns:
                            X_raw[col] = 0
                    # Reorder columns in X_raw to expected order
                    X_raw = X_raw[expected_raw]
            except Exception:
                pass

            pred = pipeline.predict(X_raw)
            prob = pipeline.predict_proba(X_raw) if hasattr(pipeline, "predict_proba") else None
            return int(pred[0]), (prob[0] if prob is not None else None)

        # Fallback path: legacy model + scaler + feature_names (numeric matrix expected)
        if model is None or scaler is None or feature_names is None:
            raise ValueError("Required model/scaler/feature_names not available for fallback prediction.")

        # Ensure all feature columns exist in the DataFrame (pad missing with 0)
        for col in feature_names:
            if col not in X_raw.columns:
                # if some features are categorical encoded as strings in the UI (e.g., "M"/"F" or "Yes"/"No")
                # leave them as-is; padding numeric columns with 0.0
                X_raw[col] = 0.0

        # Reorder columns exactly as training
        X_raw = X_raw[feature_names]

        # Scale numeric data
        X_scaled = scaler.transform(X_raw)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        # Select only features used by the model
        if selected_features is not None:
            for sf in selected_features:
                if sf not in X_scaled_df.columns:
                    # If selected feature missing despite padding, add zero column
                    X_scaled_df[sf] = 0.0
            X_final = X_scaled_df[selected_features]
        else:
            X_final = X_scaled_df

        # Predict
        pred = model.predict(X_final)
        prob = model.predict_proba(X_final) if hasattr(model, "predict_proba") else None

        return int(pred[0]), (prob[0] if prob is not None else None)

    except Exception as e:
        # bubble up exception to be displayed by Streamlit
        raise


def display_results(prediction, probability):
    """
    Display prediction results
    """
    st.header("🔍 Prediction Results")

    if prediction == 1:
        st.markdown(
            """
            <div class="prediction-box positive">
                ⚠️ HEART DISEASE DETECTED
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.error("The model indicates a high risk of heart disease.")
        st.warning("⚕️ Please consult a healthcare professional immediately for proper diagnosis and treatment.")
    else:
        st.markdown(
            """
            <div class="prediction-box negative">
                ✅ NO HEART DISEASE DETECTED
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.success("The model indicates a low risk of heart disease.")
        st.info("💚 Continue maintaining a healthy lifestyle and regular check-ups.")

    # Display probability if available
    if probability is not None:
        st.subheader("📊 Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("No Disease Probability", f"{probability[0]*100:.2f}%")

        with col2:
            st.metric("Disease Probability", f"{probability[1]*100:.2f}%")

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ["No Disease", "Heart Disease"]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(categories, probability, color=colors, alpha=0.7)
        ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
        ax.set_title("Prediction Probabilities", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])

        for bar, prob_val in zip(bars, probability):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{prob_val*100:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        st.pyplot(fig)
        plt.close()


def display_model_info():
    """
    Display information about the model
    """
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")

    st.sidebar.info(
        """
    **Heart Disease Prediction System**

    This application uses machine learning to predict the likelihood of heart disease based on patient medical data.

    **Model:** Random Forest Classifier (or pipeline if provided)

    **Dataset:** Z-Alizadeh Sani Heart Disease Dataset

    **Features:** Selected features saved in models/feature_names.pkl

    **Disclaimer:** This is a predictive tool and should not replace professional medical advice.
    """
    )


def main():
    """
    Main application
    """
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Diagnosis Assistant</p>', unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models and preprocessors..."):
        pipeline, model, scaler, feature_names, selected_features = load_models_and_preprocessor()

    if pipeline is None and (model is None or scaler is None or feature_names is None):
        st.error("❌ Failed to load a usable model/pipeline. Please ensure models/pipeline or models/random_forest + models/scaler + models/feature_names exist.")
        st.stop()

    st.success("✅ Models loaded successfully!")

    # If feature_names were not loaded by pipeline, but present as a separate file, use them
    if feature_names is None and pipeline is not None:
        # try to extract training raw feature names from pipeline if possible
        if hasattr(pipeline, "feature_names_in_"):
            feature_names = list(pipeline.feature_names_in_)
        else:
            # fallback: if you saved a features file, try loading it
            if os.path.exists("models/feature_names.pkl"):
                feature_names = joblib.load("models/feature_names.pkl")

    # If still missing, create a minimal list (avoid crash)
    if feature_names is None:
        feature_names = []

    # Sidebar info
    display_model_info()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏥 Prediction", "📊 Model Performance", "📖 Documentation"])

    with tab1:
        st.header("Patient Data Input")
        st.info(
            """
        👨‍⚕️ **Instructions:**
        1. Enter patient information in the sidebar
        2. Adjust all relevant medical parameters
        3. Click 'Predict' to get the diagnosis
        """
        )

        # Create input form using the raw training feature names
        input_data = create_input_form(feature_names if feature_names else ["Age"])

        # Debug option to display mismatches
        show_debug = st.checkbox("Show feature debug info (expected vs provided)")

        if st.button("🔮 Predict Heart Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Run prediction (pipeline preferred)
                    prediction, probability = predict(pipeline, model, scaler, feature_names, selected_features, input_data)

                    # If debug requested, show mismatches
                    if show_debug:
                        # expected final features (if available)
                        expected_after_encoding = None
                        # prefer saved final feature names if present
                        if os.path.exists("models/feature_names.pkl"):
                            expected_after_encoding = joblib.load("models/feature_names.pkl")

                        st.write("Provided input keys:", list(input_data.keys()))
                        if expected_after_encoding is not None:
                            st.write("Expected (trained) feature names count:", len(expected_after_encoding))
                            # show sample missing features
                            missing = [c for c in expected_after_encoding if c not in input_data.keys()]
                            st.write("Sample missing features (first 20):", missing[:20])
                        else:
                            st.write("No saved final feature list available for detailed comparison.")

                    # Display results
                    display_results(prediction, probability)

                    # Save a short log/caption
                    log_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "prediction": "Disease" if prediction == 1 else "No Disease",
                        "disease_probability": f"{probability[1]*100:.2f}%" if probability is not None else "N/A",
                    }
                    st.markdown("---")
                    st.caption(f"Prediction made at: {log_entry['timestamp']}")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    if show_debug:
                        st.write("Debug: Input keys:", list(input_data.keys()))
                        if os.path.exists("models/feature_names.pkl"):
                            st.write("Saved feature_names (sample):", joblib.load("models/feature_names.pkl")[:30])

    with tab2:
        st.header("📊 Model Performance Metrics")

        try:
            results_path = "outputs/model_results_complete.csv"
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                st.subheader("Performance Comparison")
                st.dataframe(results_df, use_container_width=True)

            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("outputs/roc_curves_selected_features.png"):
                    st.image("outputs/roc_curves_selected_features.png", caption="ROC Curves - Selected Features")
                if os.path.exists("outputs/feature_importance.png"):
                    st.image("outputs/feature_importance.png", caption="Feature Importance")

            with col2:
                if os.path.exists("outputs/confusion_matrices_selected_features.png"):
                    st.image("outputs/confusion_matrices_selected_features.png", caption="Confusion Matrices - Selected Features")
                if os.path.exists("outputs/performance_comparison.png"):
                    st.image("outputs/performance_comparison.png", caption="Model Performance Comparison", use_container_width=True)

        except Exception as e:
            st.warning(f"Could not load performance metrics: {e}")

    with tab3:
        st.header("📖 Documentation")
        st.markdown(
            """
        ## About the System

        This Heart Disease Prediction System is based on the research paper:
        **"Heart Disease Prediction Using Distinct Artificial Intelligence Techniques: Performance Analysis and Comparison"**

        ### Dataset
        - **Name:** Z-Alizadeh Sani Heart Disease Dataset
        - **Samples:** 303 patients
        - **Features:** 54+ medical attributes

        ### Methodology
        1. Data Preprocessing: Missing value imputation, encoding, normalization.
        2. Feature Selection: Correlation-Based Feature Subset Selection (CFS).
        3. Models: Logistic Regression, Naive Bayes, K-NN, SVM, Decision Tree, Random Forest, MLP.
        4. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC.

        ### Important Notes
        ⚠️ **Medical Disclaimer:** This is a research/educational tool and not a medical device.
        """
        )
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #7f8c8d;'><p>Heart Disease Prediction System v1.0</p><p>Powered by Machine Learning & Streamlit</p></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
