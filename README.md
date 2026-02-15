# 🌟 Stellar Classification using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machinelearningassignment-6fd5mbn7vmoartn5uvyqya.streamlit.app/)

M.Tech Machine Learning Assignment 2 | BITS Pilani (WILP)

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Model Comparison Table](#model-comparison-table)
- [Model Observations](#model-observations)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [References](#references)

---

## Problem Statement

In astronomy, **stellar classification** is the classification of stars based on their spectral characteristics. This project aims to classify celestial objects into three categories:

1. **GALAXY** - A gravitationally bound system of stars, stellar remnants, interstellar gas, dust, and dark matter
2. **STAR** - An astronomical object comprising a luminous spheroid of plasma held together by its gravity
3. **QSO (Quasar)** - Quasi-Stellar Object, an extremely luminous active galactic nucleus powered by a supermassive black hole

The objective is to implement 6 different machine learning classification models, evaluate their performance using multiple metrics, and deploy an interactive web application for demonstration.

---

## Dataset Description

**Dataset:** Stellar Classification Dataset - SDSS17  
**Source:** [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)  
**Original Source:** Sloan Digital Sky Survey Data Release 17 (SDSS DR17)

### Dataset Overview
| Property | Value |
|----------|-------|
| Total Samples | 100,000 |
| Number of Features | 17 (12 used for classification) |
| Number of Classes | 3 (GALAXY, STAR, QSO) |
| Class Distribution | GALAXY: 59.45%, STAR: 21.59%, QSO: 18.96% |

### Features Used for Classification (12 Features)

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | `alpha` | Right Ascension angle (J2000 epoch) | Continuous (0-360°) |
| 2 | `delta` | Declination angle (J2000 epoch) | Continuous (-90 to +90°) |
| 3 | `u` | Ultraviolet filter magnitude | Continuous |
| 4 | `g` | Green filter magnitude | Continuous |
| 5 | `r` | Red filter magnitude | Continuous |
| 6 | `i` | Near-Infrared filter magnitude | Continuous |
| 7 | `z` | Infrared filter magnitude | Continuous |
| 8 | `redshift` | Redshift value based on wavelength increase | Continuous |
| 9 | `plate` | Plate ID in SDSS spectroscopic survey | Integer |
| 10 | `MJD` | Modified Julian Date of observation | Integer |
| 11 | `field_ID` | Field number to identify scan | Integer |
| 12 | `cam_col` | Camera column to identify scanline | Integer (1-6) |

### Features Dropped (Identifiers Only)
- `obj_ID`, `run_ID`, `rerun_ID`, `spec_obj_ID`, `fiber_ID` - Pure database identifiers that would cause data leakage

### Key Insight
**Redshift** is the most important feature for stellar classification:
- Stars: Very low redshift (~0) - located in our galaxy
- Galaxies: Moderate redshift (0.01 - 1.0)
- Quasars: High redshift (often > 1.0) - extremely distant objects

---

## Models Used

Six classification models were implemented as required:

1. **Logistic Regression** - Linear classifier using logistic/softmax function
2. **Decision Tree Classifier** - Rule-based classifier with threshold splits
3. **K-Nearest Neighbors (KNN)** - Distance-based instance learning
4. **Naive Bayes (Gaussian)** - Probabilistic classifier assuming feature independence
5. **Random Forest (Ensemble)** - Bagging ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9837 | 0.9989 | 0.9839 | 0.9837 | 0.9837 | 0.9713 |
| Decision Tree | 0.9963 | 0.9996 | 0.9964 | 0.9963 | 0.9963 | 0.9935 |
| KNN | 0.9567 | 0.9934 | 0.9584 | 0.9567 | 0.9563 | 0.9233 |
| Naive Bayes | 0.9895 | 0.9991 | 0.9899 | 0.9895 | 0.9896 | 0.9816 |
| Random Forest (Ensemble) | 0.9964 | 0.9997 | 0.9965 | 0.9964 | 0.9965 | 0.9937 |
| XGBoost (Ensemble) | 0.9962 | 0.9999 | 0.9963 | 0.9962 | 0.9963 | 0.9934 |

**Best Performing Model:** Random Forest with 99.64% accuracy and 0.9937 MCC

---

## Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved 98.37% accuracy with linear decision boundaries. Performs well because redshift creates nearly linear class separations. Limited by inability to capture non-linear patterns at class boundaries. The multinomial approach with balanced class weights effectively handles the 3-class imbalanced problem. Excellent interpretability through feature coefficients. |
| Decision Tree | Exceptional 99.63% accuracy with just 25 leaf nodes. Assigned 99.8% importance to redshift, confirming threshold-based nature of stellar classification. Simple rules like "redshift < 0.001 → STAR" capture most patterns. Controlled depth prevents overfitting while maintaining interpretability. Nearly matches ensemble methods with fraction of complexity. |
| KNN | Lowest performer at 95.67% accuracy. Distance-based classification treats all features equally, but redshift is far more important than photometric features. High k value creates overly smooth boundaries, causing majority class (GALAXY) to dominate predictions in ambiguous regions. Not ideal for threshold-based classification problems. |
| Naive Bayes | Strong 98.95% accuracy despite "naive" independence assumption. Gaussian assumption reasonably valid for photometric magnitudes. Extremely fast training (0.013s). Provides well-calibrated probability estimates. Robust to feature correlation violations because redshift alone provides strong separation. Excellent baseline model. |
| Random Forest (Ensemble) | Best overall performance at 99.64% accuracy. Ensemble of 200 trees reduces variance and improves stability. Distributes feature importance more evenly (redshift 62%, photometric bands 38%). OOB score matches test accuracy, confirming good generalization. Achieves perfect STAR classification. Recommended for production deployment. |
| XGBoost (Ensemble) | Near-best 99.62% accuracy with highest AUC (0.9999). Sequential boosting focuses on correcting errors. Built-in regularization prevents overfitting. Best probability calibration among all models. Efficient training despite complex algorithm. Slightly more GALAXY-QSO confusion than Random Forest but superior ranking ability. |

---

## Installation & Usage

### Prerequisites
- Python 3.11+
- pip package manager

### Local Installation

```bash
# Clone the repository
git clone https://github.com/prashant-alml4344/machine_learning_assignment.git
cd machine_learning_assignment

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Using the Application

1. **Upload Data**: Upload your own CSV file or use the default dataset
2. **Download Test Data**: Click the download button to get sample test data
3. **Select Model**: Choose from 6 available classification models
4. **Train**: Click "Train Model" to train the selected model
5. **View Results**: Check accuracy, metrics, and confusion matrix in the Results tab

---

## Project Structure

```
machine_learning_assignment/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .python-version             # Python version (3.11)
├── test_data.csv               # Sample test data for quick testing
│
└── model/                      # Model training scripts
    ├── 01_dataset_understanding.py    # EDA and preprocessing
    ├── 02_model_implementation.py     # All 6 models training
    └── 03_model_observations.py       # Model analysis
```

---

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub repository
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App"
5. Select repository and `app.py`
6. Click "Deploy"

**Live App Link:** [https://machinelearningassignment-6fd5mbn7vmoartn5uvyqya.streamlit.app/](https://machinelearningassignment-6fd5mbn7vmoartn5uvyqya.streamlit.app/)

---

## Streamlit App Features

- ✅ **Dataset upload option (CSV)** - Upload custom test data
- ✅ **Download test data** - Quick download button for sample data
- ✅ **Model selection dropdown** - Choose from 6 classification models
- ✅ **Display of evaluation metrics** - All 6 metrics shown
- ✅ **Confusion matrix visualization** - Interactive heatmap display

---

## References

1. **Dataset:** fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. Retrieved from [Kaggle](https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17)

2. **Data Source:** Abdurro'uf et al., The Seventeenth data release of the Sloan Digital Sky Surveys: Complete Release of MaNGA, MaStar and APOGEE-2 DATA



---
