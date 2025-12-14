# 🏠 Real Estate Investment Advisor

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Live Demo](#-live-demo)
- [Project Structure](#project-structure)
- [Models](#models)
- [Installation](#installation)
- [Data](#data)
- [Model Performance](#model-performance)
- [Streamlit App](#streamlit-app)
- [Deployment](#deployment)

---

## 🎯 Overview

The **Real Estate Investment Advisor** is a production-ready ML system that helps investors make data-driven decisions about property investments. It combines classification models to identify good investment opportunities with regression models to forecast 5-year property price appreciation.

### Key Highlights

- **6 Trained Models**: 3 classification + 3 regression
- **99.12% Classification Accuracy** (XGBoost)
- **99.60% Regression R² Score** (XGBoost)
- **87.95% Predictions Within ±10%** Error Margin
- **Production-Ready Streamlit App**
- **Complete Feature Explainability** (SHAP Analysis)
- **Hyperparameter Optimized** (GridSearchCV)

---



## ✨ Features

### 🤖 Machine Learning
- ✅ Multi-model ensemble approach
- ✅ Hyperparameter optimization (GridSearchCV)
- ✅ Feature scaling (StandardScaler)
- ✅ Cross-validation (5-fold)
- ✅ SHAP explainability analysis
- ✅ Feature importance ranking

### 📊 Models Included
- ✅ Logistic Regression (Classification)
- ✅ Random Forest Classifier (Classification)
- ✅ XGBoost Classifier ⭐ (Classification)
- ✅ Linear Regression (Baseline)
- ✅ Random Forest Regressor (Regression)
- ✅ XGBoost Regressor ⭐ (Regression)

### 🎨 Streamlit Web App
- ✅ Interactive property prediction interface
- ✅ Real-time model comparisons
- ✅ Feature importance visualization
- ✅ Investment metrics analysis
- ✅ 5-year price forecasting
- ✅ Mobile-friendly responsive design
- ✅ Dark theme with modern UI
- ✅ 4 interactive pages

### 📈 Data Processing
- ✅ Automatic data preprocessing
- ✅ Categorical encoding (11 categories)
- ✅ Feature validation
- ✅ Missing value handling
- ✅ Statistical analysis

---

## 🚀 Live Demo

### Try the App Now!

**👉 [Click here to access the live Streamlit app](https://realestatepriceproject.streamlit.app/)**




### Expected App Experience

When you open the app, you'll see:

```
🏠 Real Estate Investment Advisor
AI-Powered Property Investment Analysis & 5-Year Price Forecasting

📊 Navigation (Sidebar):
├── 🔮 Property Prediction    (Main analysis page)
├── 📈 Model Comparison       (View all 6 models)
├── ℹ️ About Models           (Model descriptions)
└── 📋 Feature Guide          (Feature information)

Features:
├── Dark theme with white text
├── Interactive input fields
├── Real-time predictions
├── Investment recommendations
├── 5-year price forecast
└── All 6 models available
```

---

## 📁 Project Structure

```
Real_Estate_Price_Prediction_Project/
│
├── README.md                                    (This file)
├── requirements.txt                             (Python dependencies)
├── LICENSE                                      (MIT License)
├── .gitignore                                   (Git ignore rules)
│
├── Complete_Save_All_Models_And_Support_Files.py  (Training script)
├── streamapp.py                                 (Streamlit web app)
│
├── saved_models/                                (Trained models directory)
│   └── README.md                                (Models documentation)
│   ├── model_1_logistic_regression.pkl
│   ├── model_2_random_forest_classifier.pkl
│   ├── model_3_xgboost_classifier.pkl
│   ├── model_4_linear_regression.pkl
│   ├── model_5_random_forest_regressor.pkl
│   ├── model_6_xgboost_regressor.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── categorical_encodings.json
│   ├── metadata.json
│   └── categorical_encoding.py
│
├── dataset/                                     (Training dataset)
│   └── README.md                                (Dataset documentation)
│   └── sample_properties.csv                    (10,000 sample records)
│
└── notebooks/
    └── 01_Real_Estate_Analysis.ipynb            (Jupyter analysis notebook)
```

---


## 🤖 Models

### Classification Models (Investment Quality)

| Model | Algorithm | Accuracy | Precision | Recall | ROC-AUC |
|-------|-----------|----------|-----------|--------|---------|
| Model 1 | Logistic Regression | 97.70% | 96.95% | 97.30% | 0.9982 |
| Model 2 | Random Forest | 98.98% | 98.50% | 98.95% | 0.9997 |
| **Model 3** | **XGBoost** ⭐ | **99.12%** | **98.58%** | **98.98%** | **0.9997** |

### Regression Models (5-Year Price Forecast)

| Model | Algorithm | R² Score | RMSE | MAE | Within ±10% |
|-------|-----------|----------|------|-----|------------|
| Model 4 | Linear Regression | 49.01% | 148.13L | 119.22L | 17.03% |
| Model 5 | Random Forest | 98.89% | 21.85L | 16.996L | 80.77% |
| **Model 6** | **XGBoost** ⭐ | **99.60%** | **13.19L** | **10.49L** | **87.95%** |

**Note:** Model 3 (Classification) and Model 6 (Regression) are recommended for production use.

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda
- 4GB RAM minimum (8GB recommended)
- 1GB disk space for models

### Clone Repository

```bash
git clone https://github.com/GeekyVishweshNeelesh/Real_Estate_Price_Prediction_Project.git
cd Real_Estate_Price_Prediction_Project
```

### Install Dependencies

```bash
# Using pip
pip install -r requirements.txt




## 💡 Usage

### Using the Streamlit App

1. **Open App**
   ```bash
   streamlit run streamapp.py
   ```

2. **Navigate Pages** (Sidebar)
   - 🔮 Property Prediction
   - 📈 Model Comparison
   - ℹ️ About Models
   - 📋 Feature Guide

3. **Make a Prediction**
   - Enter property details
   - Click "Analyze Property"
   - View results and recommendations

### Example Input

```
Size: 2500 sq ft
BHK: 3
Price/SqFt: ₹6000
Schools: 4
Hospitals: 3
Parking: 2
Transport: 5
```

### Expected Output

```
Classification Results:
├─ Model 1: 85% Good Investment
├─ Model 2: 92% Good Investment
└─ Model 3: 95% Good Investment ⭐

Regression Results (5-Year Forecast):
├─ Model 4: ₹245L
├─ Model 5: ₹285L
└─ Model 6: ₹287L ⭐

Investment Analysis:
├─ Current Price: ₹180L
├─ Appreciation: ₹107L
├─ Growth Rate: 59.4%
└─ Confidence: 99.60%
```

---

## 📊 Data

### Dataset Information

- **Total Samples**: 10,000 property records
- **Training Samples**: 8,000 (80%)
- **Test Samples**: 2,000 (20%)
- **Features**: 7 numeric input features
- **Target Variables**: 2 (Investment Quality + Price)

### Features (7 Total)

| # | Feature | Type | Range | Unit | Importance |
|---|---------|------|-------|------|------------|
| 1 | Size_in_SqFt | Numeric | 800-5,000 | sq feet | Secondary (17.59%) |
| 2 | BHK | Numeric | 1-5 | count | Secondary |
| 3 | Price_per_SqFt | Numeric | 1,000-20,000 | ₹ | **PRIMARY (82.01%)** ⭐ |
| 4 | Nearby_Schools | Numeric | 0-10 | count | <1% |
| 5 | Nearby_Hospitals | Numeric | 0-10 | count | <1% |
| 6 | Parking_Space | Numeric | 0-3 | count | <1% |
| 7 | Public_Transport_Accessibility | Numeric | 1-5 | scale | <1% |

### Sample Data

See `dataset/README.md` for detailed dataset documentation and examples.

---

## 📈 Model Performance

### Classification Performance (Investment Quality)

```
Model 1 - Logistic Regression:
├─ Accuracy:  97.70%
├─ Precision: 96.95%
├─ Recall:    97.30%
└─ ROC-AUC:   0.9982

Model 2 - Random Forest:
├─ Accuracy:  98.98%
├─ Precision: 98.50%
├─ Recall:    98.95%
└─ ROC-AUC:   0.9997

Model 3 - XGBoost ⭐ BEST:
├─ Accuracy:  99.12%
├─ Precision: 98.58%
├─ Recall:    98.98%
└─ ROC-AUC:   0.9997
```

### Regression Performance (5-Year Price)

```
Model 4 - Linear Regression:
├─ R² Score: 49.01%
├─ RMSE:     148.13 Lakhs
└─ MAE:      119.22 Lakhs

Model 5 - Random Forest:
├─ R² Score: 98.89%
├─ RMSE:     21.85 Lakhs
└─ MAE:      16.996 Lakhs

Model 6 - XGBoost ⭐ BEST:
├─ R² Score: 99.60%
├─ RMSE:     13.19 Lakhs
├─ MAE:      10.49 Lakhs
├─ Within ±10%: 87.95%
└─ Within ±20%: 95.41%
```

### Feature Importance (SHAP Analysis)

```
Price_per_SqFt:        ████████████████████ 82.01%
Size_in_SqFt:          ████░░░░░░░░░░░░░░░ 17.59%
Other Features:        ░░░░░░░░░░░░░░░░░░░ 0.40%
```

**Key Insight:** Current market price per square foot (82%) is the strongest predictor of future appreciation.

---

## 🎨 Streamlit App

### App Pages & Features

**1. 🔮 Property Prediction (Main Page)**
- Interactive input sliders for all 7 features
- Real-time predictions from all 6 models
- Investment quality analysis (Classification)
- 5-year price forecast (Regression)
- Investment recommendation with confidence score
- Dark theme with professional styling

**2. 📈 Model Comparison**
- Classification models performance table
- Regression models performance table
- Side-by-side accuracy and R² comparisons
- RMSE and error metrics
- Recommended models highlight

**3. ℹ️ About Models**
- Detailed description of each model
- Training methodology
- Hyperparameter optimization details
- Training data information
- Model selection rationale

**4. 📋 Feature Guide**
- All 7 features description
- Feature ranges and units
- Feature importance percentages
- Statistical information
- Key insights

### App Design Features

- ✅ **Dark Theme**: Black background with white text
- ✅ **Orange Accents**: Modern color scheme
- ✅ **Responsive Layout**: Works on desktop and mobile
- ✅ **Interactive Elements**: Sliders, input fields, buttons
- ✅ **Professional Metrics**: Clear data presentation
- ✅ **Easy Navigation**: Sidebar menu with 4 pages
- ✅ **Fast Loading**: Cached model loading

---


## 📚 Documentation

- **README_SAVED_MODELS.md** - Trained models documentation
- **README_DATASET.md** - Dataset details and structure
- **Complete_Save_All_Models_And_Support_Files.py** - Model training script
- **streamapp.py** - Streamlit web application

---

## 🤝 Contributing

This is a personal project. Feel free to fork and modify for your own use!

---

