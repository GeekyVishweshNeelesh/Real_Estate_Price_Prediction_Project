"""
================================================================================
STREAMLIT APP - REAL ESTATE INVESTMENT ADVISOR
Complete Production-Ready Web Application
================================================================================

Models Required (from PDF):
✓ Model 1: Logistic Regression (Classification)
✓ Model 2: Random Forest Classifier (Classification)
✓ Model 3: XGBoost Classifier (Classification) ⭐ BEST
✓ Model 4: Linear Regression (Regression)
✓ Model 5: Random Forest Regressor (Regression)
✓ Model 6: XGBoost Regressor (Regression) ⭐ BEST

Supporting Files:
✓ Scalers for each model
✓ Feature columns
✓ Categorical encodings
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND SCALERS (CACHED FOR PERFORMANCE)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    models = {}

    model_files = {
        'Model_1_LR': 'saved_models/model_1_logistic_regression.pkl',
        'Model_2_RF': 'saved_models/model_2_random_forest_classifier.pkl',
        'Model_3_XGB': 'saved_models/model_3_xgboost_classifier.pkl',
        'Model_4_Linear': 'saved_models/model_4_linear_regression.pkl',
        'Model_5_RFR': 'saved_models/model_5_random_forest_regressor.pkl',
        'Model_6_XGBR': 'saved_models/model_6_xgboost_regressor.pkl',
        'scaler': 'saved_models/scaler.pkl',
        'features': 'saved_models/feature_columns.pkl'
    }

    try:
        for name, filepath in model_files.items():
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)

        # Load categorical encodings
        with open('categorical_encodings.json', 'r') as f:
            models['encodings'] = json.load(f)

        return models, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return models, False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_categorical(df, encodings):
    """Encode categorical columns"""
    df_encoded = df.copy()

    for col, mapping in encodings.get('categorical_manual_mappings', {}).items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)

    return df_encoded

def prepare_data(input_dict, scaler, features):
    """Prepare input data for prediction"""
    df = pd.DataFrame([input_dict])

    # Ensure all features present
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    df = df[features]

    # Scale features
    X_scaled = scaler.transform(df)

    return X_scaled

def get_model_comparison():
    """Generate model comparison metrics"""
    comparison_data = {
        'Model': ['Logistic\nRegression', 'Random Forest\nClassifier', 'XGBoost\nClassifier'],
        'Accuracy': [0.9770, 0.9898, 0.9912],
        'Precision': [0.9695, 0.9850, 0.9858],
        'Recall': [0.9730, 0.9895, 0.9898],
        'ROC-AUC': [0.9982, 0.9997, 0.9997]
    }
    return pd.DataFrame(comparison_data)

def get_regression_comparison():
    """Generate regression model comparison"""
    comparison_data = {
        'Model': ['Linear\nRegression', 'Random Forest\nRegressor', 'XGBoost\nRegressor'],
        'R² Score': [0.4901, 0.9889, 0.9960],
        'RMSE (Lakhs)': [148.13, 21.85, 13.19],
        'MAE (Lakhs)': [119.22, 16.996, 10.49],
        'Within ±10%': [17.03, 80.77, 87.95]
    }
    return pd.DataFrame(comparison_data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("<h1 class='main-header'>🏠 Real Estate Investment Advisor</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: gray;'>"
                "AI-Powered Property Investment Analysis & Price Forecasting</p>",
                unsafe_allow_html=True)

    # Load models
    models, models_loaded = load_models()

    if not models_loaded:
        st.error("❌ Failed to load models. Please check saved_models/ directory.")
        st.info("Make sure you have run: python Save_All_6_Models_Complete.py")
        return

    st.success("✅ All models loaded successfully!")

    # ────────────────────────────────────────────────────────────────────
    # SIDEBAR - NAVIGATION
    # ────────────────────────────────────────────────────────────────────

    with st.sidebar:
        st.header("📊 Navigation")
        page = st.radio("Select Page:", [
            "🔮 Property Prediction",
            "📈 Model Comparison",
            "ℹ️ About Models",
            "📋 Feature Guide"
        ])

    # ────────────────────────────────────────────────────────────────────
    # PAGE 1: PROPERTY PREDICTION
    # ────────────────────────────────────────────────────────────────────

    if page == "🔮 Property Prediction":
        st.header("Property Investment Analysis")
        st.markdown("---")

        # Create columns for input
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📍 Property Details")
            size = st.number_input(
                "Size (Sq Ft)",
                min_value=500,
                max_value=10000,
                value=2500,
                step=100,
                help="Total area of property in square feet"
            )

            bhk = st.selectbox(
                "BHK",
                options=[1, 2, 3, 4, 5],
                index=2,
                help="Number of bedrooms"
            )

            price_per_sqft = st.number_input(
                "Price per Sq Ft (₹)",
                min_value=1000,
                max_value=20000,
                value=6000,
                step=500,
                help="Current market price per square foot"
            )

        with col2:
            st.subheader("🏘️ Location & Amenities")
            schools = st.slider(
                "Nearby Schools",
                min_value=0,
                max_value=10,
                value=4,
                help="Number of schools within 5km"
            )

            hospitals = st.slider(
                "Nearby Hospitals",
                min_value=0,
                max_value=10,
                value=3,
                help="Number of hospitals within 5km"
            )

            parking = st.selectbox(
                "Parking Space",
                options=[0, 1, 2, 3],
                index=1,
                help="Number of parking spaces"
            )

        # Transport accessibility
        col3, col4 = st.columns(2)
        with col3:
            transport = st.slider(
                "Public Transport Accessibility (1-5)",
                min_value=1,
                max_value=5,
                value=4,
                help="1=Poor, 5=Excellent"
            )

        with col4:
            model_choice = st.selectbox(
                "Model Selection",
                options=["Model 3 - XGBoost (Best)", "All Models", "Model 1", "Model 2"],
                help="Select which model to use for prediction"
            )

        # ────────────────────────────────────────────────────────────────
        # MAKE PREDICTIONS
        # ────────────────────────────────────────────────────────────────

        if st.button("🎯 Analyze Property", use_container_width=True):

            # Prepare input data
            input_data = {
                'Size_in_SqFt': size,
                'BHK': bhk,
                'Price_per_SqFt': price_per_sqft,
                'Nearby_Schools': schools,
                'Nearby_Hospitals': hospitals,
                'Parking_Space': parking,
                'Public_Transport_Accessibility': transport
            }

            scaler = models['scaler']
            features = models['features']

            X_scaled = prepare_data(input_data, scaler, features)

            # ────────────────────────────────────────────────────────────
            # CLASSIFICATION PREDICTIONS
            # ────────────────────────────────────────────────────────────

            st.markdown("### 📊 Classification Results (Investment Quality)")
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            # Model 1 - Logistic Regression
            with col1:
                model_1 = models['Model_1_LR']
                pred_1 = model_1.predict(X_scaled)[0]
                prob_1 = model_1.predict_proba(X_scaled)[0, 1]

                st.metric(
                    "Model 1: Logistic Regression",
                    "✅ Good" if pred_1 == 1 else "❌ Bad",
                    f"Confidence: {prob_1*100:.1f}%"
                )
                st.write(f"📊 Accuracy: 97.70%")

            # Model 2 - Random Forest
            with col2:
                model_2 = models['Model_2_RF']
                pred_2 = model_2.predict(X_scaled)[0]
                prob_2 = model_2.predict_proba(X_scaled)[0, 1]

                st.metric(
                    "Model 2: Random Forest",
                    "✅ Good" if pred_2 == 1 else "❌ Bad",
                    f"Confidence: {prob_2*100:.1f}%"
                )
                st.write(f"📊 Accuracy: 98.98%")

            # Model 3 - XGBoost (BEST)
            with col3:
                model_3 = models['Model_3_XGB']
                pred_3 = model_3.predict(X_scaled)[0]
                prob_3 = model_3.predict_proba(X_scaled)[0, 1]

                st.metric(
                    "🌟 Model 3: XGBoost",
                    "✅ Good" if pred_3 == 1 else "❌ Bad",
                    f"Confidence: {prob_3*100:.1f}%"
                )
                st.write(f"📊 Accuracy: 99.12% ⭐")

            # ────────────────────────────────────────────────────────────
            # REGRESSION PREDICTIONS (IF GOOD INVESTMENT)
            # ────────────────────────────────────────────────────────────

            if pred_3 == 1:  # If good investment
                st.markdown("### 💰 5-Year Price Forecast (Regression Results)")
                st.markdown("---")

                col1, col2, col3 = st.columns(3)

                # Model 4 - Linear Regression
                with col1:
                    model_4 = models['Model_4_Linear']
                    pred_4 = model_4.predict(X_scaled)[0]

                    st.metric(
                        "Model 4: Linear Regression",
                        f"₹{pred_4:.2f}L",
                        "R²: 49.01% ⚠️"
                    )

                # Model 5 - Random Forest
                with col2:
                    model_5 = models['Model_5_RFR']
                    pred_5 = model_5.predict(X_scaled)[0]

                    st.metric(
                        "Model 5: Random Forest",
                        f"₹{pred_5:.2f}L",
                        "R²: 98.89%"
                    )

                # Model 6 - XGBoost (BEST)
                with col3:
                    model_6 = models['Model_6_XGBR']
                    pred_6 = model_6.predict(X_scaled)[0]

                    st.metric(
                        "🌟 Model 6: XGBoost",
                        f"₹{pred_6:.2f}L",
                        "R²: 99.60% ⭐"
                    )

                # Calculate investment metrics
                current_price = (size * price_per_sqft) / 100000
                appreciation = pred_6 - current_price
                appreciation_pct = (appreciation / current_price) * 100

                st.markdown("### 📈 Investment Analysis")
                st.markdown("---")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Price", f"₹{current_price:.2f}L")

                with col2:
                    st.metric("Predicted 5-Year", f"₹{pred_6:.2f}L")

                with col3:
                    st.metric("Appreciation", f"₹{appreciation:.2f}L")

                with col4:
                    st.metric("Growth Rate", f"{appreciation_pct:.1f}%")

                # Success message
                st.markdown(f"""
                <div class='success-box'>
                    <h4>✅ Good Investment Opportunity!</h4>
                    <p>
                        • Expected 5-year appreciation: <strong>₹{appreciation:.2f} Lakhs</strong><br>
                        • Growth rate: <strong>{appreciation_pct:.1f}%</strong><br>
                        • Prediction confidence: <strong>{prob_3*100:.1f}%</strong><br>
                        • Model reliability: <strong>99.60% R²</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class='danger-box'>
                    <h4>⚠️ Not Recommended</h4>
                    <p>
                        This property does not meet good investment criteria.<br>
                        Confidence: <strong>{(1-prob_3)*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # PAGE 2: MODEL COMPARISON
    # ────────────────────────────────────────────────────────────────────

    elif page == "📈 Model Comparison":
        st.header("Model Performance Comparison")
        st.markdown("---")

        # Classification Models
        st.subheader("Classification Models (Investment Quality)")

        clf_data = get_model_comparison()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(clf_data, use_container_width=True)

        with col2:
            fig = px.bar(
                clf_data.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title='Classification Model Performance',
                height=400
            )
            fig.update_yaxes(range=[0.95, 1.0])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Regression Models
        st.subheader("Regression Models (5-Year Price Prediction)")

        reg_data = get_regression_comparison()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(reg_data, use_container_width=True)

        with col2:
            fig = px.bar(
                reg_data[['Model', 'R² Score']],
                x='Model',
                y='R² Score',
                color='R² Score',
                title='Regression Model R² Score Comparison',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Recommendations
        st.subheader("🎯 Model Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class='success-box'>
                <h4>✅ Best Classification Model</h4>
                <p>
                    <strong>Model 3: XGBoost Classifier</strong><br>
                    • Accuracy: 99.12%<br>
                    • Precision: 98.58%<br>
                    • ROC-AUC: 0.9997<br>
                    → Use for investment decisions
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='success-box'>
                <h4>✅ Best Regression Model</h4>
                <p>
                    <strong>Model 6: XGBoost Regressor</strong><br>
                    • R² Score: 99.60%<br>
                    • RMSE: 13.19 Lakhs<br>
                    • Within ±10%: 87.95%<br>
                    → Use for price forecasting
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # PAGE 3: ABOUT MODELS
    # ────────────────────────────────────────────────────────────────────

    elif page == "ℹ️ About Models":
        st.header("About the ML Models")
        st.markdown("---")

        st.markdown("""
        ### 📊 Project Overview

        This Real Estate Investment Advisor system uses 6 machine learning models
        to help investors make data-driven decisions about property investments.

        **Two-Stage Prediction Pipeline:**
        1. **Classification Stage**: Determine if property is good investment
        2. **Regression Stage**: Forecast 5-year property price

        ### 🏆 Classification Models (Investment Quality)

        #### Model 1: Logistic Regression
        - Simple, interpretable linear model
        - Accuracy: **97.70%**
        - Use case: Baseline predictions

        #### Model 2: Random Forest Classifier
        - Ensemble of decision trees
        - Accuracy: **98.98%**
        - Use case: Feature importance analysis

        #### Model 3: XGBoost Classifier ⭐ BEST
        - Gradient boosted trees
        - Accuracy: **99.12%**
        - Use case: Production deployment
        - Precision: 98.58% | Recall: 98.98%

        ### 📈 Regression Models (Price Forecasting)

        #### Model 4: Linear Regression
        - Simple linear relationship
        - R² Score: **49.01%** ⚠️ Not recommended

        #### Model 5: Random Forest Regressor
        - Multiple decision trees
        - R² Score: **98.89%**
        - RMSE: 21.85 Lakhs

        #### Model 6: XGBoost Regressor ⭐ BEST
        - Gradient boosted trees
        - R² Score: **99.60%**
        - RMSE: 13.19 Lakhs (9% on 150L property)
        - 87.95% predictions within ±10%

        ### 🔧 Hyperparameter Optimization

        Model 6 was optimized using GridSearchCV:
        - **102,060 model fits** evaluated
        - **9 parameters** fine-tuned
        - **3.30% improvement** in accuracy

        ### 📊 Training Data

        - **Total samples**: 250,000 properties
        - **Features**: 7 input variables
        - **Training samples**: 200,000 (80%)
        - **Test samples**: 50,000 (20%)
        - **Cross-validation**: 5-fold
        """)

    # ────────────────────────────────────────────────────────────────────
    # PAGE 4: FEATURE GUIDE
    # ────────────────────────────────────────────────────────────────────

    elif page == "📋 Feature Guide":
        st.header("Feature Definitions & Impact")
        st.markdown("---")

        st.markdown("""
        ### 📍 Input Features (7 Total)

        #### 1. Size in Square Feet
        - **Range**: 800 - 5,000 sq ft
        - **Impact**: Secondary importance (17.59%)
        - **Interpretation**: Larger properties may have different appreciation

        #### 2. Number of Bedrooms (BHK)
        - **Range**: 1 - 5 BHK
        - **Impact**: Affects property category
        - **Interpretation**: Market segment indicator

        #### 3. Price per Square Foot (₹)
        - **Range**: ₹2,000 - ₹10,000 per sq ft
        - **Impact**: PRIMARY (82.01%) ⭐
        - **Interpretation**: Current market valuation, strongest predictor

        #### 4. Nearby Schools
        - **Range**: 0 - 10 schools
        - **Impact**: Location amenity indicator
        - **Interpretation**: Family-friendly area marker

        #### 5. Nearby Hospitals
        - **Range**: 0 - 10 hospitals
        - **Impact**: Healthcare proximity indicator
        - **Interpretation**: Essential service accessibility

        #### 6. Parking Space
        - **Range**: 0 - 3 spaces
        - **Impact**: Convenience factor
        - **Interpretation**: Urban area indicator

        #### 7. Public Transport Accessibility
        - **Range**: 1 - 5 (scale)
        - **Impact**: Connectivity indicator
        - **Interpretation**: Commute convenience

        ### 🎯 Feature Importance (SHAP Analysis)

        | Feature | Importance | Impact |
        |---------|-----------|--------|
        | **Price_per_SqFt** | **82.01%** | ⭐ Dominant |
        | **Size_in_SqFt** | **17.59%** | Secondary |
        | Other Features | **<1%** | Minimal |

        ### 💡 Investment Insights

        **High Importance Factors:**
        1. **Price per Square Foot** (82%)
           - Higher price/sqft = Better appreciation
           - Reflects market perception and quality

        2. **Property Size** (17%)
           - Larger properties differ in appreciation
           - Affects market segment

        **Lower Importance Factors:**
        - Amenities already priced in
        - Marginal direct model impact
        - Indirect effects through price/sqft

        ### 📊 Prediction Reliability

        - Classification: **99.12% accurate**
        - Regression: **99.60% R² (explains 99.60% variation)**
        - Confidence: **87.95% within ±10% error**
        """)

    # ────────────────────────────────────────────────────────────────────
    # FOOTER
    # ────────────────────────────────────────────────────────────────────

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("✅ **Status**: Production Ready")
    with col2:
        st.info("📊 **Models**: 6 Trained")
    with col3:
        st.info("🚀 **Accuracy**: 99.12% / 99.60%")

    st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 20px;'>
        <p>Real Estate Investment Advisor | ML Project | © 2024</p>
        <p>Powered by XGBoost, Random Forest, and Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
