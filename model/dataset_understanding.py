"""
================================================================================
STEP 1: DATASET UNDERSTANDING - Stellar Classification SDSS17
================================================================================

LEARNING OBJECTIVE:
Before jumping into modeling, a good ML engineer ALWAYS understands their data first.
This is what separates a good ML practitioner from someone who just runs code blindly.

DATASET OVERVIEW:
-----------------
Source: Sloan Digital Sky Survey Data Release 17 (SDSS DR17)
Origin: NASA's astronomical survey
Link: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

WHY THIS DATASET?
1. Unique domain (astronomy) - not overused like Titanic/Iris
2. 100,000 instances (well above 500 minimum)
3. 17 features (well above 12 minimum)
4. 3-class classification problem (GALAXY, STAR, QSO)
5. Published benchmarks: All 6 required algorithms achieve 95%+ accuracy
6. Real scientific data from NASA - adds credibility to your assignment

THE CLASSIFICATION PROBLEM:
---------------------------
We need to classify celestial objects into 3 categories:
    1. GALAXY - A gravitationally bound system of stars, stellar remnants, gas, dust
    2. STAR   - An astronomical object comprising a luminous spheroid of plasma
    3. QSO    - Quasar (Quasi-Stellar Object), extremely luminous active galactic nucleus

Class Distribution (approximately):
    - GALAXY: 59,445 (~59.4%) - MAJORITY CLASS
    - STAR:   21,594 (~21.6%)
    - QSO:    18,961 (~19.0%) - Quasars

NOTE: This is an imbalanced dataset! GALAXY dominates. This affects model evaluation.

FEATURE DESCRIPTIONS:
--------------------
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UNDERSTANDING EACH FEATURE
# ============================================================================
# 
# The features fall into TWO categories:
#
# CATEGORY 1: SPECTRAL/PHOTOMETRIC FEATURES (USEFUL FOR CLASSIFICATION)
# These contain actual astronomical measurements that help distinguish objects:
#
# | Feature  | Description                                    | Range/Type        |
# |----------|------------------------------------------------|-------------------|
# | alpha    | Right Ascension angle (J2000 epoch)            | 0-360 degrees     |
# | delta    | Declination angle (J2000 epoch)                | -90 to +90 degrees|
# | u        | Ultraviolet filter magnitude                   | ~12-30 mag        |
# | g        | Green filter magnitude                         | ~12-30 mag        |
# | r        | Red filter magnitude                           | ~10-30 mag        |
# | i        | Near-Infrared filter magnitude                 | ~10-30 mag        |
# | z        | Infrared filter magnitude                      | ~10-30 mag        |
# | redshift | Redshift value (wavelength increase)           | -0.01 to 7+       |
#
# CATEGORY 2: IDENTIFIER/METADATA FEATURES (NOT USEFUL - DROP THESE)
# These are just IDs used for database management, not scientific measurements:
#
# | Feature     | Description                                 | Why Drop?         |
# |-------------|---------------------------------------------|-------------------|
# | obj_ID      | Object Identifier                           | Just an ID number |
# | run_ID      | Run Number for scan identification          | Technical metadata|
# | rerun_ID    | Rerun Number for image processing           | Technical metadata|
# | cam_col     | Camera column identifier                    | Technical metadata|
# | field_ID    | Field number identifier                     | Technical metadata|
# | spec_obj_ID | Spectroscopic object ID                     | Just an ID number |
# | plate       | Plate ID in SDSS                            | Technical metadata|
# | MJD         | Modified Julian Date                        | Timestamp         |
# | fiber_ID    | Fiber identifier                            | Technical metadata|
#
# ============================================================================

# KEY INSIGHT: THE MOST IMPORTANT FEATURE
# ============================================================================
# REDSHIFT is the MOST IMPORTANT feature for stellar classification!
# 
# Why? Because:
# - Stars have very low redshift (~0) - they're in our galaxy
# - Galaxies have moderate redshift (0.01 - 1.0)  
# - Quasars have HIGH redshift (often > 1.0) - they're extremely distant
#
# This makes physical sense: redshift indicates how fast an object is moving
# away from us, which correlates with distance (Hubble's Law).
# ============================================================================


def create_simulated_stellar_data(n_samples=100000, random_state=42):
    """
    Creates a simulated dataset with the EXACT same structure as SDSS17.
    
    WHY SIMULATE?
    Due to network restrictions, we can't download from Kaggle directly.
    This simulation maintains:
    - Same column names
    - Same data types
    - Similar statistical distributions
    - Same class imbalance
    
    For your actual submission, you'll download the real dataset from:
    https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default: 100,000 like real dataset)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : Simulated stellar classification dataset
    """
    np.random.seed(random_state)
    
    # Class distribution matching real dataset
    n_galaxy = int(n_samples * 0.5945)  # 59.45%
    n_star = int(n_samples * 0.2159)    # 21.59%
    n_qso = n_samples - n_galaxy - n_star  # Remaining ~19%
    
    print(f"Creating simulated dataset with {n_samples:,} samples...")
    print(f"  - GALAXY: {n_galaxy:,} ({n_galaxy/n_samples*100:.1f}%)")
    print(f"  - STAR:   {n_star:,} ({n_star/n_samples*100:.1f}%)")
    print(f"  - QSO:    {n_qso:,} ({n_qso/n_samples*100:.1f}%)")
    
    data = []
    
    # Generate GALAXY samples
    for i in range(n_galaxy):
        data.append({
            'obj_ID': np.random.randint(int(1e15), int(1e16)),
            'alpha': np.random.uniform(0, 360),
            'delta': np.random.uniform(-20, 80),
            'u': np.random.normal(20.5, 1.5),
            'g': np.random.normal(19.0, 1.3),
            'r': np.random.normal(18.2, 1.2),
            'i': np.random.normal(17.8, 1.1),
            'z': np.random.normal(17.5, 1.1),
            'run_ID': np.random.randint(100, 9000),
            'rerun_ID': 301,
            'cam_col': np.random.randint(1, 7),
            'field_ID': np.random.randint(10, 900),
            'spec_obj_ID': np.random.randint(int(1e15), int(1e16)),
            'class': 'GALAXY',
            'redshift': np.random.exponential(0.1) + 0.01,  # Galaxies: low-moderate redshift
            'plate': np.random.randint(200, 15000),
            'MJD': np.random.randint(51000, 59000),
            'fiber_ID': np.random.randint(1, 1000)
        })
    
    # Generate STAR samples
    for i in range(n_star):
        data.append({
            'obj_ID': np.random.randint(int(1e15), int(1e16)),
            'alpha': np.random.uniform(0, 360),
            'delta': np.random.uniform(-20, 80),
            'u': np.random.normal(18.0, 2.0),
            'g': np.random.normal(16.5, 1.8),
            'r': np.random.normal(16.0, 1.7),
            'i': np.random.normal(15.8, 1.6),
            'z': np.random.normal(15.6, 1.5),
            'run_ID': np.random.randint(100, 9000),
            'rerun_ID': 301,
            'cam_col': np.random.randint(1, 7),
            'field_ID': np.random.randint(10, 900),
            'spec_obj_ID': np.random.randint(int(1e15), int(1e16)),
            'class': 'STAR',
            'redshift': np.random.normal(0.0001, 0.0005),  # Stars: very low redshift (in our galaxy)
            'plate': np.random.randint(200, 15000),
            'MJD': np.random.randint(51000, 59000),
            'fiber_ID': np.random.randint(1, 1000)
        })
    
    # Generate QSO (Quasar) samples
    for i in range(n_qso):
        data.append({
            'obj_ID': np.random.randint(int(1e15), int(1e16)),
            'alpha': np.random.uniform(0, 360),
            'delta': np.random.uniform(-20, 80),
            'u': np.random.normal(19.5, 1.0),
            'g': np.random.normal(19.0, 0.9),
            'r': np.random.normal(18.8, 0.8),
            'i': np.random.normal(18.6, 0.8),
            'z': np.random.normal(18.4, 0.8),
            'run_ID': np.random.randint(100, 9000),
            'rerun_ID': 301,
            'cam_col': np.random.randint(1, 7),
            'field_ID': np.random.randint(10, 900),
            'spec_obj_ID': np.random.randint(int(1e15), int(1e16)),
            'class': 'QSO',
            'redshift': np.random.exponential(1.0) + 0.5,  # Quasars: HIGH redshift
            'plate': np.random.randint(200, 15000),
            'MJD': np.random.randint(51000, 59000),
            'fiber_ID': np.random.randint(1, 1000)
        })
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def explore_dataset(df):
    """
    Comprehensive Exploratory Data Analysis (EDA) for the stellar dataset.
    
    This function demonstrates proper EDA workflow:
    1. Basic statistics
    2. Data types
    3. Missing values
    4. Class distribution
    5. Feature correlations
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # 1. Basic Info
    print("\n📊 1. BASIC DATASET INFO")
    print("-" * 40)
    print(f"Total samples: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. Data Types
    print("\n📋 2. DATA TYPES")
    print("-" * 40)
    print(df.dtypes.to_string())
    
    # 3. Missing Values
    print("\n❓ 3. MISSING VALUES")
    print("-" * 40)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    # 4. Class Distribution
    print("\n🎯 4. TARGET CLASS DISTRIBUTION")
    print("-" * 40)
    class_counts = df['class'].value_counts()
    class_pct = df['class'].value_counts(normalize=True) * 100
    for cls in class_counts.index:
        print(f"  {cls}: {class_counts[cls]:,} samples ({class_pct[cls]:.2f}%)")
    
    # Calculate imbalance ratio
    max_class = class_counts.max()
    min_class = class_counts.min()
    print(f"\n  Imbalance Ratio: {max_class/min_class:.2f}:1")
    print("  ⚠️ Note: Dataset is imbalanced. Consider using class_weight='balanced'")
    
    # 5. Feature Statistics (numerical only)
    print("\n📈 5. FEATURE STATISTICS")
    print("-" * 40)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'class' in numerical_cols:
        numerical_cols.remove('class')
    
    stats_df = df[numerical_cols].describe().round(3)
    print(stats_df.to_string())
    
    # 6. Key Insight: Redshift by Class
    print("\n🔑 6. KEY INSIGHT: REDSHIFT BY CLASS")
    print("-" * 40)
    print("Redshift is the MOST important feature for classification!")
    print("\nMean redshift by class:")
    redshift_stats = df.groupby('class')['redshift'].agg(['mean', 'std', 'min', 'max'])
    print(redshift_stats.round(4).to_string())
    print("\n💡 Notice: Stars have ~0 redshift, Galaxies have moderate, Quasars have HIGH!")
    
    return df


def preprocess_data(df):
    """
    Preprocess the stellar classification dataset.
    
    PREPROCESSING STEPS (and WHY):
    
    1. DROP IDENTIFIER COLUMNS
       - These are database IDs, not scientific measurements
       - Including them would be DATA LEAKAGE and scientifically wrong
    
    2. ENCODE TARGET VARIABLE
       - Convert 'GALAXY', 'STAR', 'QSO' to 0, 1, 2
       - Required for sklearn models
    
    3. FEATURE SCALING (for certain models)
       - KNN and Logistic Regression require scaled features
       - Tree-based models (Decision Tree, Random Forest, XGBoost) do NOT
       - We'll return both scaled and unscaled versions
    
    Returns:
    --------
    dict with X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Step 1: Identify columns to drop
    # These are identifiers/metadata, not scientific features
    columns_to_drop = [
        'obj_ID',      # Object ID - just a database identifier
        'run_ID',      # Run number - technical metadata
        'rerun_ID',    # Rerun number - technical metadata
        'cam_col',     # Camera column - technical metadata
        'field_ID',    # Field number - technical metadata
        'spec_obj_ID', # Spectroscopic object ID - database identifier
        'plate',       # Plate ID - technical metadata
        'MJD',         # Modified Julian Date - timestamp, not feature
        'fiber_ID'     # Fiber ID - technical metadata
    ]
    
    print(f"\n🗑️ Step 1: Dropping {len(columns_to_drop)} identifier/metadata columns")
    print(f"   Columns dropped: {columns_to_drop}")
    
    # Create feature matrix
    feature_cols = [col for col in df.columns if col not in columns_to_drop + ['class']]
    X = df[feature_cols].copy()
    y = df['class'].copy()
    
    print(f"\n📊 Features retained: {list(X.columns)}")
    print(f"   Number of features: {len(X.columns)}")
    
    # Step 2: Encode target variable
    print(f"\n🏷️ Step 2: Encoding target variable")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"   Classes: {list(le.classes_)}")
    print(f"   Encoded as: {list(range(len(le.classes_)))}")
    
    # Step 3: Train-Test Split
    # Using stratified split to maintain class proportions
    print(f"\n✂️ Step 3: Train-Test Split (80-20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded  # IMPORTANT: Maintain class proportions
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Verify stratification worked
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print("\n   Class distribution verification:")
    print(f"   Train: {dict(zip(le.classes_, train_dist.round(3)))}")
    print(f"   Test:  {dict(zip(le.classes_, test_dist.round(3)))}")
    
    # Step 4: Feature Scaling
    print(f"\n⚖️ Step 4: Feature Scaling (StandardScaler)")
    print("   Note: Required for KNN, Logistic Regression")
    print("   Note: Optional for tree-based models")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print("   ✅ Scaling complete!")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols,
        'label_encoder': le,
        'scaler': scaler
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("STELLAR CLASSIFICATION DATASET - UNDERSTANDING & PREPROCESSING")
    print("M.Tech Machine Learning Assignment 2")
    print("="*70)
    
    # Create simulated dataset (replace with real data when available)
    df = create_simulated_stellar_data(n_samples=100000)
    
    # Save the simulated dataset
    df.to_csv('star_classification.csv', index=False)
    print("\n✅ Dataset saved to 'star_classification.csv'")
    
    # Explore the dataset
    df = explore_dataset(df)
    
    # Preprocess the data
    data_dict = preprocess_data(df)
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE - READY FOR MODELING")
    print("="*70)
    print(f"""
Next Steps:
1. Train 6 classification models on this data
2. Calculate 6 evaluation metrics for each model
3. Create comparison table
4. Build Streamlit app
5. Deploy to Streamlit Community Cloud

Feature Summary:
- Total features: {len(data_dict['feature_names'])}
- Features: {data_dict['feature_names']}
- Target classes: {list(data_dict['label_encoder'].classes_)}
- Training samples: {len(data_dict['X_train']):,}
- Test samples: {len(data_dict['X_test']):,}
""")
    
    # Save preprocessed data for next step
    import joblib
    joblib.dump(data_dict, 'preprocessed_data.joblib')
    print("✅ Preprocessed data saved to 'preprocessed_data.joblib'")
