import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD MODELS AND SUPPORT FILES ====================

@st.cache_resource
def load_models_and_files():
    """Load all models and support files with proper error handling"""
    try:
        # Load support files
        with open('saved_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('saved_models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)

        with open('saved_models/categorical_encodings.json', 'r') as f:
            categorical_encodings = json.load(f)

        with open('saved_models/metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load all 6 models
        models = {}
        model_names = [
            'model_1_logistic_regression.pkl',
            'model_2_random_forest_classifier.pkl',
            'model_3_xgboost_classifier.pkl',
            'model_4_linear_regression.pkl',
            'model_5_random_forest_regressor.pkl',
            'model_6_xgboost_regressor.pkl'
        ]

        for model_name in model_names:
            model_path = f'saved_models/{model_name}'
            with open(model_path, 'rb') as f:
                models[model_name.replace('.pkl', '')] = pickle.load(f)

        return scaler, feature_columns, categorical_encodings, metadata, models

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error("Make sure all files are in the 'saved_models/' directory")
        st.stop()

# Load models
scaler, feature_columns, categorical_encodings, metadata, models = load_models_and_files()

# ==================== PAGE HEADER ====================

st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
    <h1>🏠 Real Estate Investment Advisor</h1>
    <p style="font-size: 18px; color: #666;">
    AI-Powered Property Investment Analysis & 5-Year Price Forecasting
    </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🔮 Property Prediction", "📈 Model Comparison", "ℹ️ About Models", "📋 Feature Guide"]
)

# ==================== PAGE 1: PROPERTY PREDICTION ====================

if page == "🔮 Property Prediction":
    st.subheader("Analyze Property Investment Potential")

    col1, col2 = st.columns(2)

    with col1:
        size_sqft = st.number_input(
            "Size (sq ft)",
            min_value=800,
            max_value=5000,
            value=2500,
            step=100,
            help="Property size in square feet"
        )

        bhk = st.slider(
            "BHK (Bedrooms)",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of bedrooms, halls, kitchens"
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
        nearby_schools = st.slider(
            "Nearby Schools (0-10)",
            min_value=0,
            max_value=10,
            value=4,
            help="Number of schools within 5km radius"
        )

        nearby_hospitals = st.slider(
            "Nearby Hospitals (0-10)",
            min_value=0,
            max_value=10,
            value=3,
            help="Number of hospitals within 5km radius"
        )

        parking_space = st.slider(
            "Parking Spaces (0-3)",
            min_value=0,
            max_value=3,
            value=2,
            help="Number of parking spaces available"
        )

    public_transport = st.slider(
        "Public Transport Accessibility (1-5)",
        min_value=1,
        max_value=5,
        value=4,
        help="1=Poor, 5=Excellent"
    )

    # Calculate current price
    current_price = (size_sqft * price_per_sqft) / 100000  # Convert to Lakhs

    st.info(f"📍 **Current Property Price: ₹{current_price:.2f} Lakhs**")

    # Prepare input data
    input_data = pd.DataFrame({
        'Size_in_SqFt': [size_sqft],
        'BHK': [bhk],
        'Price_per_SqFt': [price_per_sqft],
        'Nearby_Schools': [nearby_schools],
        'Nearby_Hospitals': [nearby_hospitals],
        'Parking_Space': [parking_space],
        'Public_Transport_Accessibility': [public_transport]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make predictions button
    if st.button("🔍 Analyze Property", use_container_width=True, type="primary"):

        # Classification predictions (Good/Bad Investment)
        st.subheader("📊 Investment Quality Predictions")

        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            pred_1 = models['model_1_logistic_regression'].predict_proba(input_scaled)[0][1]
            st.metric(
                "Model 1: Logistic",
                f"{pred_1*100:.1f}%",
                f"{'✅ Good' if pred_1 > 0.5 else '❌ Bad'}"
            )

        with col_c2:
            pred_2 = models['model_2_random_forest_classifier'].predict_proba(input_scaled)[0][1]
            st.metric(
                "Model 2: Random Forest",
                f"{pred_2*100:.1f}%",
                f"{'✅ Good' if pred_2 > 0.5 else '❌ Bad'}"
            )

        with col_c3:
            pred_3 = models['model_3_xgboost_classifier'].predict_proba(input_scaled)[0][1]
            st.metric(
                "Model 3: XGBoost ⭐",
                f"{pred_3*100:.1f}%",
                f"{'✅ Good' if pred_3 > 0.5 else '❌ Bad'}"
            )

        st.divider()

        # Regression predictions (5-Year Price)
        st.subheader("💰 5-Year Price Forecast (in Lakhs)")

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            pred_4 = models['model_4_linear_regression'].predict(input_scaled)[0]
            st.metric(
                "Model 4: Linear",
                f"₹{pred_4:.2f}L",
                f"{((pred_4-current_price)/current_price)*100:.1f}% growth"
            )

        with col_r2:
            pred_5 = models['model_5_random_forest_regressor'].predict(input_scaled)[0]
            st.metric(
                "Model 5: Random Forest",
                f"₹{pred_5:.2f}L",
                f"{((pred_5-current_price)/current_price)*100:.1f}% growth"
            )

        with col_r3:
            pred_6 = models['model_6_xgboost_regressor'].predict(input_scaled)[0]
            st.metric(
                "Model 6: XGBoost ⭐",
                f"₹{pred_6:.2f}L",
                f"{((pred_6-current_price)/current_price)*100:.1f}% growth"
            )

        st.divider()

        # Investment analysis
        st.subheader("🎯 Investment Analysis")

        avg_classification = np.mean([pred_1, pred_2, pred_3])
        avg_regression = np.mean([pred_4, pred_5, pred_6])

        col_a1, col_a2, col_a3, col_a4 = st.columns(4)

        with col_a1:
            st.metric("Current Price", f"₹{current_price:.2f}L")

        with col_a2:
            st.metric("Avg Predicted Price", f"₹{avg_regression:.2f}L")

        with col_a3:
            appreciation = avg_regression - current_price
            st.metric("Expected Appreciation", f"₹{appreciation:.2f}L")

        with col_a4:
            growth_rate = ((avg_regression - current_price) / current_price) * 100
            st.metric("Growth Rate", f"{growth_rate:.1f}%")

        # Recommendation
        if avg_classification > 0.7:
            st.success(f"✅ **RECOMMENDED**: {avg_classification*100:.1f}% confidence this is a good investment")
        elif avg_classification > 0.5:
            st.warning(f"⚠️ **MODERATE**: {avg_classification*100:.1f}% confidence - Consider carefully")
        else:
            st.error(f"❌ **NOT RECOMMENDED**: {avg_classification*100:.1f}% confidence - High risk")

# ==================== PAGE 2: MODEL COMPARISON ====================

elif page == "📈 Model Comparison":
    st.subheader("Model Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Classification Models (Investment Quality)")
        classification_data = {
            'Model': ['Logistic', 'Random Forest', 'XGBoost'],
            'Accuracy': [97.70, 98.98, 99.12],
            'Precision': [96.95, 98.50, 98.58],
            'Recall': [97.30, 98.95, 98.98]
        }
        st.dataframe(pd.DataFrame(classification_data), use_container_width=True)

    with col2:
        st.write("### Regression Models (5-Year Price Forecast)")
        regression_data = {
            'Model': ['Linear', 'Random Forest', 'XGBoost'],
            'R² Score': [49.01, 98.89, 99.60],
            'RMSE': [148.13, 21.85, 13.19],
            'Within ±10%': [17.03, 80.77, 87.95]
        }
        st.dataframe(pd.DataFrame(regression_data), use_container_width=True)

    st.divider()

    st.info("""
    ⭐ **Recommended Models:**
    - **Classification**: XGBoost (99.12% accuracy)
    - **Regression**: XGBoost (99.60% R² score)
    """)

# ==================== PAGE 3: ABOUT MODELS ====================

elif page == "ℹ️ About Models":
    st.subheader("Model Information")

    st.write("""
    ### Classification Models (Investment Quality)

    **Model 1 - Logistic Regression**
    - Simple baseline model
    - Good for interpretability
    - 97.70% accuracy

    **Model 2 - Random Forest Classifier**
    - Ensemble method with multiple trees
    - Good feature importance
    - 98.98% accuracy

    **Model 3 - XGBoost Classifier** ⭐
    - Gradient boosting algorithm
    - Best performance (99.12% accuracy)
    - **Recommended for production**

    ---

    ### Regression Models (Price Forecast)

    **Model 4 - Linear Regression**
    - Simple baseline model
    - Fast predictions
    - Lower accuracy (49.01% R²)

    **Model 5 - Random Forest Regressor**
    - Ensemble method
    - Good performance (98.89% R²)
    - 80.77% within ±10% error

    **Model 6 - XGBoost Regressor** ⭐
    - Gradient boosting for regression
    - Best performance (99.60% R²)
    - 87.95% within ±10% error
    - **Recommended for production**
    """)

    st.divider()

    st.write("""
    ### Training Details
    - **Training Samples**: 8,000
    - **Test Samples**: 2,000
    - **Features**: 7 input variables
    - **Targets**: 2 (Investment Quality + Price)
    - **Hyperparameter Optimization**: GridSearchCV
    - **Cross-Validation**: 5-fold
    """)

# ==================== PAGE 4: FEATURE GUIDE ====================

elif page == "📋 Feature Guide":
    st.subheader("Feature Descriptions & Importance")

    features_info = {
        'Feature': [
            'Size_in_SqFt',
            'BHK',
            'Price_per_SqFt',
            'Nearby_Schools',
            'Nearby_Hospitals',
            'Parking_Space',
            'Public_Transport_Accessibility'
        ],
        'Range': [
            '800-5000',
            '1-5',
            '₹1,000-20,000',
            '0-10',
            '0-10',
            '0-3',
            '1-5 (scale)'
        ],
        'Importance': [
            '17.59%',
            'Secondary',
            '82.01% ⭐',
            '<1%',
            '<1%',
            '<1%',
            '<1%'
        ],
        'Description': [
            'Total property area',
            'Number of bedrooms',
            'Market price per sq ft (PRIMARY DRIVER)',
            'Schools within 5km',
            'Hospitals within 5km',
            'Available parking spaces',
            'Public transport connectivity'
        ]
    }

    st.dataframe(pd.DataFrame(features_info), use_container_width=True)

    st.info("""
    💡 **Key Insight**: Price per square foot (82.01% importance) is the strongest
    predictor of future property appreciation. This makes sense as it reflects
    the location desirability and market demand.
    """)

# ==================== FOOTER ====================

st.divider()

st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px; margin-top: 2rem;">
    <p>🏠 Real Estate Investment Advisor | ML-Powered Property Analysis System</p>
    <p>Version 1.0 | All 6 Models Available | Production Ready</p>
    </div>
    """, unsafe_allow_html=True)
