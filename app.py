"""
================================================================================
STELLAR CLASSIFICATION - STREAMLIT WEB APPLICATION
================================================================================

M.Tech Machine Learning Assignment 2
Dataset: Stellar Classification SDSS17 (NASA/SDSS)

This Streamlit app provides:
1. Dataset upload option (CSV) 
2. Model selection dropdown
3. Display of evaluation metrics 
4. Confusion matrix visualization 

Author: Prashant Sharma
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Import all 6 classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Stellar Classification ML App",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<p class="main-header">🌟 Stellar Classification using Machine Learning</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Classifying Stars, Galaxies, and Quasars using SDSS17 Dataset</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR - MODEL SELECTION AND INFO
# ============================================================================
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # Model Selection Dropdown (Required - 1 mark)
    st.subheader("Select ML Model")
    
    model_choice = st.selectbox(
        "Choose a classification model:",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes (Gaussian)",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        ],
        index=4,  # Default to Random Forest
        help="Select one of the 6 implemented classification models"
    )
    
    st.markdown("---")
    
    # Dataset Info
    st.subheader("📊 Dataset Info")
    st.markdown("""
    **Stellar Classification SDSS17**
    - Source: NASA/SDSS DR17
    - Classes: GALAXY, STAR, QSO
    - Features: 8 (after preprocessing)
    - Original Size: 100,000 samples
    """)
    
    st.markdown("---")
    
    # Model Info
    st.subheader("ℹ️ Model Info")
    model_info = {
        "Logistic Regression": "Linear classifier using logistic function. Good baseline with interpretable coefficients.",
        "Decision Tree": "Tree-based classifier using threshold rules. Highly interpretable.",
        "K-Nearest Neighbors (KNN)": "Instance-based learning using distance metrics. Requires scaled features.",
        "Naive Bayes (Gaussian)": "Probabilistic classifier assuming feature independence. Very fast.",
        "Random Forest (Ensemble)": "Ensemble of decision trees with bagging. Best overall performance.",
        "XGBoost (Ensemble)": "Gradient boosting ensemble. Excellent for tabular data."
    }
    st.info(model_info[model_choice])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_default_data():
    """
    Load or create default stellar classification data.
    Uses caching to avoid regenerating data on every rerun.
    """
    np.random.seed(42)
    n_samples = 10000  # Smaller for faster training in Streamlit
    
    n_galaxy = int(n_samples * 0.5945)
    n_star = int(n_samples * 0.2159)
    n_qso = n_samples - n_galaxy - n_star
    
    data = []
    
    # Generate GALAXY samples
    for _ in range(n_galaxy):
        data.append({
            'alpha': np.random.uniform(0, 360),
            'delta': np.random.uniform(-20, 80),
            'u': np.random.normal(20.5, 1.5),
            'g': np.random.normal(19.0, 1.3),
            'r': np.random.normal(18.2, 1.2),
            'i': np.random.normal(17.8, 1.1),
            'z': np.random.normal(17.5, 1.1),
            'redshift': np.random.exponential(0.1) + 0.01,
            'class': 'GALAXY'
        })
    
    # Generate STAR samples
    for _ in range(n_star):
        data.append({
            'alpha': np.random.uniform(0, 360),
            'delta': np.random.uniform(-20, 80),
            'u': np.random.normal(18.0, 2.0),
            'g': np.random.normal(16.5, 1.8),
            'r': np.random.normal(16.0, 1.7),
            'i': np.random.normal(15.8, 1.6),
            'z': np.random.normal(15.6, 1.5),
            'redshift': np.random.normal(0.0001, 0.0005),
            'class': 'STAR'
        })
    
    # Generate QSO samples
    for _ in range(n_qso):
        data.append({
            'alpha': np.random.uniform(0, 360),
            'delta': np.random.uniform(-20, 80),
            'u': np.random.normal(19.5, 1.0),
            'g': np.random.normal(19.0, 0.9),
            'r': np.random.normal(18.8, 0.8),
            'i': np.random.normal(18.6, 0.8),
            'z': np.random.normal(18.4, 0.8),
            'redshift': np.random.exponential(1.0) + 0.5,
            'class': 'QSO'
        })
    
    return pd.DataFrame(data)


def preprocess_data(df):
    """
    Preprocess the uploaded or default dataset.
    Returns processed features, labels, and preprocessing objects.
    """
    # Define feature columns (excluding identifiers if present)
    id_columns = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 
                  'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
    
    # Keep only relevant columns
    feature_cols = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 4:
        st.error("Dataset must contain at least these columns: alpha, delta, u, g, r, i, z, redshift")
        return None
    
    X = df[available_features].copy()
    y = df['class'].copy()
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': available_features,
        'label_encoder': le,
        'scaler': scaler
    }


def get_model(model_name):
    """
    Return the selected model with appropriate hyperparameters.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000,
            class_weight='balanced', multi_class='multinomial', random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=20, min_samples_leaf=10,
            class_weight='balanced', random_state=42
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=15, weights='distance', metric='euclidean'
        ),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', random_state=42, n_jobs=-1,
            use_label_encoder=False, eval_metric='mlogloss'
        )
    }
    return models[model_name]


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate all 6 required evaluation metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Create and return a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📤 Data Upload & Training", "📊 Results & Metrics", "ℹ️ About"])

# ============================================================================
# TAB 1: DATA UPLOAD AND TRAINING
# ============================================================================
with tab1:
    st.header("📤 Upload Dataset or Use Default")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File Upload Section (Required - 1 mark)
        st.subheader("Upload your CSV file")
        uploaded_file = st.file_uploader(
            "Choose a CSV file (must contain: alpha, delta, u, g, r, i, z, redshift, class)",
            type=['csv'],
            help="Upload test data in CSV format. The file should contain the required columns."
        )
        
        use_default = st.checkbox("Use default dataset (10,000 samples)", value=True)
        
        # Download Test Data Button
        st.markdown("---")
        st.subheader("📥 Download Sample Test Data")
        st.markdown("Don't have test data? Download our sample file to test the app:")
        
        # Create sample test data for download
        sample_data = """alpha,delta,u,g,r,i,z,redshift,class
135.689,32.495,19.47,17.04,16.11,15.69,15.40,0.0001,STAR
247.352,19.872,21.12,19.54,18.76,18.34,18.03,0.0823,GALAXY
183.295,46.128,19.82,19.21,18.95,18.73,18.52,1.2847,QSO
72.456,28.934,18.23,16.89,16.21,15.87,15.61,0.0002,STAR
312.784,55.219,20.89,19.32,18.54,18.12,17.81,0.1245,GALAXY
156.923,41.567,19.65,19.08,18.81,18.59,18.38,1.8934,QSO
89.134,12.456,17.95,16.52,15.89,15.52,15.28,0.0001,STAR
278.456,38.912,21.45,19.87,19.08,18.65,18.33,0.0956,GALAXY
201.567,52.348,19.91,19.35,19.12,18.91,18.72,2.3456,QSO
45.678,25.789,18.56,17.12,16.45,16.08,15.81,0.0003,STAR
334.123,48.567,20.67,19.12,18.35,17.94,17.63,0.0734,GALAXY
167.234,35.891,19.73,19.15,18.89,18.67,18.47,1.5623,QSO
112.456,8.234,18.12,16.78,16.12,15.76,15.49,0.0002,STAR
289.567,62.134,21.23,19.65,18.87,18.45,18.14,0.1567,GALAXY
178.345,44.678,19.88,19.29,19.03,18.82,18.62,1.9234,QSO
67.891,31.456,17.78,16.35,15.72,15.38,15.12,0.0001,STAR
256.789,21.345,20.45,18.92,18.15,17.74,17.43,0.0645,GALAXY
145.678,39.234,19.56,18.98,18.72,18.51,18.31,1.1234,QSO
98.234,15.678,18.34,16.91,16.25,15.89,15.62,0.0002,STAR
301.456,57.891,21.67,20.08,19.28,18.85,18.53,0.1823,GALAXY
189.567,47.123,19.79,19.22,18.97,18.76,18.56,2.1567,QSO
34.567,22.891,17.65,16.23,15.61,15.27,15.01,0.0001,STAR
267.891,34.567,20.23,18.71,17.95,17.54,17.24,0.0512,GALAXY
156.234,42.567,19.67,19.11,18.86,18.65,18.45,1.4567,QSO
78.912,9.345,18.01,16.58,15.93,15.57,15.31,0.0002,STAR
323.456,51.234,21.89,20.31,19.51,19.08,18.76,0.2134,GALAXY
198.678,49.891,19.94,19.37,19.11,18.89,18.69,2.5678,QSO
56.234,27.123,17.89,16.47,15.84,15.49,15.23,0.0001,STAR
245.123,17.456,20.98,19.42,18.65,18.23,17.92,0.0889,GALAXY
134.567,36.789,19.52,18.95,18.69,18.48,18.28,1.0234,QSO"""
        
        st.download_button(
            label="⬇️ Download Test Data (CSV)",
            data=sample_data,
            file_name="test_data.csv",
            mime="text/csv",
            help="Download a sample CSV file with 30 stellar objects to test the classification models"
        )
    
    with col2:
        st.subheader("Required Columns")
        st.markdown("""
        - `alpha` - Right Ascension
        - `delta` - Declination  
        - `u, g, r, i, z` - Photometric filters
        - `redshift` - Redshift value
        - `class` - Target (GALAXY/STAR/QSO)
        """)
    
    st.markdown("---")
    
    # Load Data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded file loaded: {len(df)} rows, {len(df.columns)} columns")
    elif use_default:
        df = load_default_data()
        st.info(f"ℹ️ Using default dataset: {len(df)} samples")
    else:
        st.warning("⚠️ Please upload a CSV file or check 'Use default dataset'")
        st.stop()
    
    # Display sample data
    with st.expander("👀 Preview Dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Classes", df['class'].nunique())
    
    # Class Distribution
    with st.expander("📊 Class Distribution", expanded=False):
        class_dist = df['class'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#3B82F6', '#10B981', '#F59E0B']
        bars = ax.bar(class_dist.index, class_dist.values, color=colors)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Stellar Classes')
        for bar, val in zip(bars, class_dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                   f'{val:,}', ha='center', va='bottom', fontsize=10)
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Train Model Button
    st.subheader(f"🚀 Train {model_choice}")
    
    if st.button("Train Model", type="primary", use_container_width=True):
        with st.spinner(f"Training {model_choice}... Please wait."):
            # Preprocess data
            data = preprocess_data(df)
            
            if data is None:
                st.stop()
            
            # Get model
            model = get_model(model_choice)
            
            # Determine which data to use (scaled for KNN and LogReg)
            if model_choice in ["Logistic Regression", "K-Nearest Neighbors (KNN)"]:
                X_train = data['X_train_scaled']
                X_test = data['X_test_scaled']
            else:
                X_train = data['X_train']
                X_test = data['X_test']
            
            # Train model
            model.fit(X_train, data['y_train'])
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(data['y_test'], y_pred, y_prob)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['metrics'] = metrics
            st.session_state['y_test'] = data['y_test']
            st.session_state['y_pred'] = y_pred
            st.session_state['class_names'] = list(data['label_encoder'].classes_)
            st.session_state['model_name'] = model_choice
            st.session_state['trained'] = True
            
        st.success(f"✅ {model_choice} trained successfully! Go to 'Results & Metrics' tab.")


# ============================================================================
# TAB 2: RESULTS AND METRICS
# ============================================================================
with tab2:
    st.header("📊 Model Evaluation Results")
    
    if 'trained' not in st.session_state or not st.session_state['trained']:
        st.warning("⚠️ Please train a model first in the 'Data Upload & Training' tab.")
        st.stop()
    
    st.subheader(f"Results for: {st.session_state['model_name']}")
    
    # Display Metrics (Required - 1 mark)
    st.markdown("### 📈 Evaluation Metrics")
    
    metrics = st.session_state['metrics']
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    with col2:
        st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
    with col3:
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    with col4:
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    with col5:
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    with col6:
        st.metric("MCC", f"{metrics['MCC']:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix (Required - 1 mark)
    st.markdown("### 🔢 Confusion Matrix")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig = plot_confusion_matrix(
            st.session_state['y_test'],
            st.session_state['y_pred'],
            st.session_state['class_names']
        )
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**How to read the confusion matrix:**")
        st.markdown("""
        - **Rows** = Actual/True labels
        - **Columns** = Predicted labels
        - **Diagonal** = Correct predictions ✅
        - **Off-diagonal** = Errors ❌
        
        Higher diagonal values indicate better performance.
        """)
    
    st.markdown("---")
    
    # Classification Report
    st.markdown("### 📋 Classification Report")
    
    report = classification_report(
        st.session_state['y_test'],
        st.session_state['y_pred'],
        target_names=st.session_state['class_names'],
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)


# ============================================================================
# TAB 3: ABOUT
# ============================================================================
with tab3:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application demonstrates machine learning classification models for 
    **Stellar Classification** - classifying celestial objects into Stars, Galaxies, 
    and Quasars (QSO) based on spectral characteristics.
    
    ### 📊 Dataset
    **Stellar Classification Dataset - SDSS17**
    - **Source:** Sloan Digital Sky Survey Data Release 17
    - **Samples:** 100,000 observations
    - **Classes:** GALAXY, STAR, QSO (Quasar)
    - **Features:** Photometric measurements (u, g, r, i, z bands), coordinates (alpha, delta), and redshift
    
    ### 🤖 Implemented Models
    1. **Logistic Regression** - Linear classifier with regularization
    2. **Decision Tree** - Rule-based classifier with interpretable structure
    3. **K-Nearest Neighbors** - Instance-based learning using distance metrics
    4. **Naive Bayes (Gaussian)** - Probabilistic classifier assuming feature independence
    5. **Random Forest** - Ensemble of decision trees with bagging
    6. **XGBoost** - Gradient boosting ensemble for high accuracy
    
    ### 📈 Evaluation Metrics
    - **Accuracy:** Overall correctness of predictions
    - **AUC Score:** Area under ROC curve (discrimination ability)
    - **Precision:** Ratio of true positives to predicted positives
    - **Recall:** Ratio of true positives to actual positives
    - **F1 Score:** Harmonic mean of precision and recall
    - **MCC:** Matthews Correlation Coefficient (balanced metric)
    
    ### 👨‍💻 Assignment Details
    - **Course:** M.Tech Machine Learning
    - **Assignment:** Assignment 2
    - **Institution:** BITS Pilani (Work Integrated Learning Programme)
    
    ### 🔗 References
    - [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
    - [SDSS Website](https://www.sdss.org/)
    - [Scikit-learn Documentation](https://scikit-learn.org/)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📝 Model Comparison Summary
    
    | Model | Accuracy | Best For |
    |-------|----------|----------|
    | Random Forest | ~99.6% | Best overall performance |
    | XGBoost | ~99.6% | Best probability calibration |
    | Decision Tree | ~99.6% | Interpretability |
    | Naive Bayes | ~98.9% | Fast training |
    | Logistic Regression | ~98.4% | Baseline, interpretable |
    | KNN | ~95.7% | Non-parametric approach |
    """)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280;'>
        <p>🌟 Stellar Classification ML App | M.Tech Machine Learning Assignment 2</p>
        <p>Built with Streamlit | Data from NASA/SDSS DR17</p>
    </div>
    """,
    unsafe_allow_html=True
)
