"""
================================================================================
STEP 2: MODEL IMPLEMENTATION - All 6 Classification Models
================================================================================

ASSIGNMENT REQUIREMENT:
Implement 6 classification models and calculate 6 evaluation metrics for each.

MODELS TO IMPLEMENT:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

METRICS TO CALCULATE:
1. Accuracy
2. AUC Score (ROC-AUC)
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

LEARNING OBJECTIVES:
- Understand when to use scaled vs unscaled features
- Learn proper multi-class metric calculation
- Understand class imbalance handling
- Learn hyperparameter selection reasoning

Author: Prashant Sharma
Assignment: M.Tech ML Assignment 2
Dataset: Stellar Classification SDSS17
================================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from time import time

# Sklearn imports for models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Sklearn imports for metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# ============================================================================
# UNDERSTANDING EVALUATION METRICS
# ============================================================================
"""
Before implementing, let's understand WHAT each metric measures and WHEN to use it:

1. ACCURACY = (TP + TN) / Total
   - Simple percentage of correct predictions
   - PROBLEM: Misleading with imbalanced data!
   - If 60% are GALAXY, predicting all GALAXY gives 60% accuracy but useless model
   
2. AUC (Area Under ROC Curve)
   - Measures model's ability to distinguish between classes
   - Range: 0.5 (random) to 1.0 (perfect)
   - ADVANTAGE: Works well with imbalanced data
   - For multi-class: We use One-vs-Rest (OvR) strategy
   
3. PRECISION = TP / (TP + FP)
   - "Of all predicted positives, how many were actually positive?"
   - High precision = Few false positives
   - USE WHEN: False positives are costly (e.g., spam detection)
   
4. RECALL (Sensitivity) = TP / (TP + FN)
   - "Of all actual positives, how many did we find?"
   - High recall = Few false negatives
   - USE WHEN: Missing positives is costly (e.g., disease detection)
   
5. F1 SCORE = 2 * (Precision * Recall) / (Precision + Recall)
   - Harmonic mean of Precision and Recall
   - Balances both metrics
   - USE WHEN: You need balance between precision and recall
   
6. MCC (Matthews Correlation Coefficient)
   - Range: -1 (total disagreement) to +1 (perfect prediction)
   - BEST METRIC for imbalanced datasets!
   - Uses all four confusion matrix values (TP, TN, FP, FN)
   - A classifier that only predicts majority class gets MCC ≈ 0

FOR MULTI-CLASS PROBLEMS (like ours with 3 classes):
- We use 'weighted' averaging for Precision, Recall, F1
- 'weighted' accounts for class imbalance by weighting by support (class frequency)
- Alternative: 'macro' (simple average) or 'micro' (global calculation)
"""


def calculate_all_metrics(y_true, y_pred, y_prob, model_name):
    """
    Calculate all 6 required evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (ground truth)
    y_pred : array-like
        Predicted labels from model.predict()
    y_prob : array-like
        Predicted probabilities from model.predict_proba()
        Shape: (n_samples, n_classes)
    model_name : str
        Name of the model (for display purposes)
    
    Returns:
    --------
    dict : Dictionary containing all 6 metrics
    
    IMPORTANT NOTES:
    ----------------
    - For AUC: We need probabilities, not predictions!
    - For multi-class AUC: Use 'ovr' (One-vs-Rest) strategy
    - For Precision/Recall/F1: Use 'weighted' average for imbalanced data
    - MCC handles multi-class automatically
    """
    
    # 1. ACCURACY
    # Simple and interpretable, but be cautious with imbalanced data
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. AUC (ROC-AUC)
    # For multi-class, we use One-vs-Rest (ovr) strategy
    # multi_class='ovr': Computes AUC for each class vs rest, then averages
    # average='weighted': Weights by class frequency (handles imbalance)
    try:
        auc = roc_auc_score(
            y_true, 
            y_prob, 
            multi_class='ovr',      # One-vs-Rest for multi-class
            average='weighted'       # Weight by class frequency
        )
    except Exception as e:
        print(f"    ⚠️ AUC calculation issue: {e}")
        auc = 0.0
    
    # 3. PRECISION
    # 'weighted': Calculate for each class, weight by support (frequency)
    # This handles class imbalance appropriately
    precision = precision_score(
        y_true, 
        y_pred, 
        average='weighted',  # Weighted by class frequency
        zero_division=0      # Return 0 if no predictions for a class
    )
    
    # 4. RECALL
    # Same averaging strategy as precision
    recall = recall_score(
        y_true, 
        y_pred, 
        average='weighted',
        zero_division=0
    )
    
    # 5. F1 SCORE
    # Harmonic mean of precision and recall
    f1 = f1_score(
        y_true, 
        y_pred, 
        average='weighted',
        zero_division=0
    )
    
    # 6. MATTHEWS CORRELATION COEFFICIENT (MCC)
    # Best single metric for imbalanced classification!
    # Range: -1 to +1, where 0 means no better than random
    # Automatically handles multi-class
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }


def print_detailed_report(y_true, y_pred, class_names):
    """
    Print detailed classification report and confusion matrix.
    Useful for understanding per-class performance.
    """
    print("\n  📋 Classification Report:")
    print("  " + "-"*60)
    report = classification_report(y_true, y_pred, target_names=class_names)
    for line in report.split('\n'):
        print(f"  {line}")
    
    print("\n  📊 Confusion Matrix:")
    print("  " + "-"*60)
    cm = confusion_matrix(y_true, y_pred)
    # Create a nice formatted confusion matrix
    print(f"  Predicted →  {class_names[0]:>10} {class_names[1]:>10} {class_names[2]:>10}")
    print("  Actual ↓")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>10}  {row[0]:>10} {row[1]:>10} {row[2]:>10}")


# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================
def train_logistic_regression(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate Logistic Regression classifier.
    
    WHY LOGISTIC REGRESSION?
    - Despite its name, it's a CLASSIFICATION algorithm (not regression!)
    - Works by fitting a logistic (sigmoid) curve to the data
    - For multi-class: Uses One-vs-Rest (OvR) or Multinomial approach
    - Outputs probabilities, not just class labels
    
    PREPROCESSING REQUIREMENTS:
    - REQUIRES feature scaling (StandardScaler)
    - Features should have similar scales for gradient descent to converge properly
    
    KEY HYPERPARAMETERS:
    - C: Inverse regularization strength (smaller C = more regularization)
    - solver: Algorithm to use ('lbfgs' works well for most cases)
    - max_iter: Maximum iterations for solver to converge
    - class_weight: 'balanced' adjusts weights inversely to class frequencies
    
    WHEN TO USE:
    - When you need interpretable results (coefficients show feature importance)
    - When you want probability outputs
    - As a baseline model
    """
    print("\n" + "="*70)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*70)
    
    print("\n  📖 Theory:")
    print("  - Linear classifier that uses logistic function for probability estimation")
    print("  - For multi-class: Uses 'multinomial' with softmax function")
    print("  - Requires scaled features for optimal convergence")
    
    print("\n  ⚙️ Hyperparameters chosen:")
    print("  - C=1.0: Default regularization (balanced bias-variance)")
    print("  - solver='lbfgs': Good for multi-class, handles L2 regularization")
    print("  - max_iter=1000: Enough iterations for convergence")
    print("  - class_weight='balanced': Handles our imbalanced dataset")
    print("  - multi_class='multinomial': Proper multi-class handling")
    
    # Initialize the model
    model = LogisticRegression(
        C=1.0,                        # Regularization parameter
        solver='lbfgs',               # Algorithm for optimization
        max_iter=1000,                # Max iterations to converge
        class_weight='balanced',      # Handle class imbalance!
        multi_class='multinomial',    # Proper multi-class approach
        random_state=42,              # Reproducibility
        n_jobs=-1                     # Use all CPU cores
    )
    
    # Train the model
    print("\n  🏋️ Training...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"  ✅ Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_prob, "Logistic Regression")
    
    print("\n  📊 Evaluation Metrics:")
    print("  " + "-"*50)
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric:>12}: {value:.4f}")
    
    # Detailed report
    print_detailed_report(y_test, y_pred, class_names)
    
    return model, metrics


# ============================================================================
# MODEL 2: DECISION TREE CLASSIFIER
# ============================================================================
def train_decision_tree(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate Decision Tree classifier.
    
    WHY DECISION TREE?
    - Highly interpretable (can visualize the tree!)
    - Handles non-linear relationships
    - No feature scaling required
    - Can capture feature interactions
    
    PREPROCESSING REQUIREMENTS:
    - NO scaling required (splits are based on thresholds, not distances)
    - Can handle mixed feature types
    
    KEY HYPERPARAMETERS:
    - max_depth: Maximum depth of tree (controls overfitting)
    - min_samples_split: Minimum samples to split a node
    - min_samples_leaf: Minimum samples in a leaf node
    - criterion: 'gini' or 'entropy' for measuring split quality
    
    COMMON PITFALL:
    - Decision trees EASILY OVERFIT if not pruned!
    - Default settings create a tree that memorizes training data
    - We MUST limit depth or use pruning
    
    WHEN TO USE:
    - When you need interpretable rules
    - When feature interactions matter
    - As a baseline before trying ensembles
    """
    print("\n" + "="*70)
    print("MODEL 2: DECISION TREE CLASSIFIER")
    print("="*70)
    
    print("\n  📖 Theory:")
    print("  - Recursively splits data based on feature thresholds")
    print("  - Creates a tree of if-then rules")
    print("  - Prone to overfitting without proper constraints")
    
    print("\n  ⚙️ Hyperparameters chosen:")
    print("  - max_depth=10: Prevents overly deep trees (overfitting)")
    print("  - min_samples_split=20: Need 20+ samples to create a split")
    print("  - min_samples_leaf=10: Each leaf must have 10+ samples")
    print("  - criterion='gini': Gini impurity (faster than entropy)")
    print("  - class_weight='balanced': Handles class imbalance")
    
    # Initialize the model with constraints to prevent overfitting
    model = DecisionTreeClassifier(
        max_depth=10,                 # Limit tree depth
        min_samples_split=20,         # Min samples to split
        min_samples_leaf=10,          # Min samples in leaf
        criterion='gini',             # Split quality measure
        class_weight='balanced',      # Handle imbalance
        random_state=42               # Reproducibility
    )
    
    # Train the model
    print("\n  🏋️ Training...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"  ✅ Training completed in {train_time:.2f} seconds")
    print(f"  📏 Actual tree depth: {model.get_depth()}")
    print(f"  🍃 Number of leaves: {model.get_n_leaves()}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_prob, "Decision Tree")
    
    print("\n  📊 Evaluation Metrics:")
    print("  " + "-"*50)
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric:>12}: {value:.4f}")
    
    # Feature importance (unique to tree-based models!)
    print("\n  🔑 Top 5 Feature Importances:")
    print("  " + "-"*50)
    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Detailed report
    print_detailed_report(y_test, y_pred, class_names)
    
    return model, metrics


# ============================================================================
# MODEL 3: K-NEAREST NEIGHBORS (KNN)
# ============================================================================
def train_knn(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate K-Nearest Neighbors classifier.
    
    WHY KNN?
    - Simple, intuitive algorithm ("you are what your neighbors are")
    - No training phase (lazy learner) - stores all data
    - Can capture complex decision boundaries
    - Good for smaller datasets
    
    PREPROCESSING REQUIREMENTS:
    - CRITICAL: REQUIRES feature scaling!
    - KNN uses distance metrics (Euclidean by default)
    - Without scaling, features with larger ranges dominate
    
    KEY HYPERPARAMETERS:
    - n_neighbors (k): Number of neighbors to consider
      - Small k: More sensitive to noise, risk of overfitting
      - Large k: Smoother boundaries, risk of underfitting
      - Rule of thumb: Start with sqrt(n_samples), use odd number
    - weights: 'uniform' or 'distance'
      - 'distance': Closer neighbors have more influence (usually better)
    - metric: Distance metric ('euclidean', 'manhattan', etc.)
    
    WHEN TO USE:
    - Smaller datasets (slow with large data)
    - When decision boundary is complex/non-linear
    - When interpretability is not crucial
    """
    print("\n" + "="*70)
    print("MODEL 3: K-NEAREST NEIGHBORS (KNN)")
    print("="*70)
    
    print("\n  📖 Theory:")
    print("  - Instance-based learning (memorizes training data)")
    print("  - Classifies based on majority vote of k nearest neighbors")
    print("  - Distance-based, so FEATURE SCALING IS CRITICAL!")
    
    # Calculate optimal k using sqrt rule
    k = int(np.sqrt(len(X_train)))
    if k % 2 == 0:  # Make it odd to avoid ties
        k += 1
    
    print(f"\n  ⚙️ Hyperparameters chosen:")
    print(f"  - n_neighbors={k}: sqrt({len(X_train)}) ≈ {k}, made odd")
    print("  - weights='distance': Closer neighbors matter more")
    print("  - metric='euclidean': Standard distance metric")
    print("  - algorithm='auto': Let sklearn choose best algorithm")
    
    # Initialize the model
    model = KNeighborsClassifier(
        n_neighbors=k,                # Number of neighbors
        weights='distance',           # Weight by distance
        metric='euclidean',           # Distance metric
        algorithm='auto',             # Auto-select algorithm
        n_jobs=-1                     # Use all CPU cores
    )
    
    # Train the model (for KNN, this just stores the data)
    print("\n  🏋️ Training (storing data)...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"  ✅ Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    print("  🔮 Predicting (this is where KNN does the work)...")
    start_time = time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    pred_time = time() - start_time
    print(f"  ✅ Prediction completed in {pred_time:.2f} seconds")
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_prob, "KNN")
    
    print("\n  📊 Evaluation Metrics:")
    print("  " + "-"*50)
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric:>12}: {value:.4f}")
    
    # Detailed report
    print_detailed_report(y_test, y_pred, class_names)
    
    return model, metrics


# ============================================================================
# MODEL 4: NAIVE BAYES (GAUSSIAN)
# ============================================================================
def train_naive_bayes(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate Gaussian Naive Bayes classifier.
    
    WHY NAIVE BAYES?
    - Extremely fast training and prediction
    - Works well with high-dimensional data
    - Based on Bayes' theorem with "naive" independence assumption
    - Good baseline for text classification
    
    THE "NAIVE" ASSUMPTION:
    - Assumes all features are INDEPENDENT given the class
    - This is rarely true in practice, but works surprisingly well!
    - For stellar data: Assumes u, g, r, i, z magnitudes are independent
      (they're not, but Naive Bayes is robust to this violation)
    
    WHY GAUSSIAN?
    - GaussianNB assumes features follow normal (Gaussian) distribution
    - Good for continuous numerical features (like our photometric data)
    - Alternatives: MultinomialNB (for counts), BernoulliNB (for binary)
    
    PREPROCESSING REQUIREMENTS:
    - No scaling strictly required
    - Features should be approximately normally distributed (GaussianNB)
    
    KEY HYPERPARAMETERS:
    - var_smoothing: Smoothing parameter to prevent zero probabilities
    
    WHEN TO USE:
    - Fast baseline model
    - High-dimensional data
    - When training data is limited
    """
    print("\n" + "="*70)
    print("MODEL 4: GAUSSIAN NAIVE BAYES")
    print("="*70)
    
    print("\n  📖 Theory:")
    print("  - Based on Bayes' theorem: P(y|X) ∝ P(X|y) * P(y)")
    print("  - 'Naive' because it assumes feature independence")
    print("  - Gaussian: Assumes features follow normal distribution")
    print("  - Despite naive assumption, works well in practice!")
    
    print("\n  ⚙️ Hyperparameters chosen:")
    print("  - var_smoothing=1e-9: Default smoothing for numerical stability")
    print("  - (GaussianNB has very few hyperparameters!)")
    
    # Initialize the model
    model = GaussianNB(
        var_smoothing=1e-9  # Smoothing parameter
    )
    
    # Train the model
    print("\n  🏋️ Training...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"  ✅ Training completed in {train_time:.4f} seconds (FAST!)")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_prob, "Naive Bayes")
    
    print("\n  📊 Evaluation Metrics:")
    print("  " + "-"*50)
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric:>12}: {value:.4f}")
    
    # Class priors (what NB learned about class distribution)
    print("\n  📈 Learned Class Priors:")
    print("  " + "-"*50)
    for i, (cls, prior) in enumerate(zip(class_names, model.class_prior_)):
        print(f"  P({cls}) = {prior:.4f}")
    
    # Detailed report
    print_detailed_report(y_test, y_pred, class_names)
    
    return model, metrics


# ============================================================================
# MODEL 5: RANDOM FOREST (ENSEMBLE)
# ============================================================================
def train_random_forest(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate Random Forest classifier.
    
    WHY RANDOM FOREST?
    - Ensemble of Decision Trees (bagging + feature randomization)
    - Much better than single Decision Tree (reduces overfitting)
    - Handles non-linear relationships
    - Provides feature importance
    - Robust to outliers and noise
    
    HOW IT WORKS:
    1. Creates multiple decision trees (n_estimators)
    2. Each tree trained on bootstrap sample (random subset with replacement)
    3. Each split considers random subset of features (max_features)
    4. Final prediction = majority vote of all trees
    
    PREPROCESSING REQUIREMENTS:
    - NO scaling required (tree-based)
    - Handles missing values (in some implementations)
    
    KEY HYPERPARAMETERS:
    - n_estimators: Number of trees (more = better, but slower)
    - max_depth: Maximum depth of each tree
    - min_samples_split: Minimum samples to split
    - max_features: Features to consider for each split
    - oob_score: Use out-of-bag samples for validation (free!)
    
    WHEN TO USE:
    - When you need high accuracy
    - When interpretability is not critical (but feature importance is)
    - As a strong baseline before trying boosting methods
    """
    print("\n" + "="*70)
    print("MODEL 5: RANDOM FOREST (ENSEMBLE)")
    print("="*70)
    
    print("\n  📖 Theory:")
    print("  - Ensemble of Decision Trees with bagging and feature randomization")
    print("  - Each tree votes, majority wins (reduces variance)")
    print("  - 'Random' refers to: random samples + random features at each split")
    print("  - Typically outperforms single Decision Tree significantly")
    
    print("\n  ⚙️ Hyperparameters chosen:")
    print("  - n_estimators=200: 200 trees in the forest")
    print("  - max_depth=15: Deeper than single tree (ensemble handles overfitting)")
    print("  - min_samples_split=10: Minimum samples to split")
    print("  - max_features='sqrt': sqrt(n_features) at each split (default)")
    print("  - class_weight='balanced': Handle class imbalance")
    print("  - oob_score=True: Get out-of-bag score (free validation!)")
    
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=200,             # Number of trees
        max_depth=15,                 # Max depth per tree
        min_samples_split=10,         # Min samples to split
        min_samples_leaf=5,           # Min samples in leaf
        max_features='sqrt',          # Features per split
        class_weight='balanced',      # Handle imbalance
        oob_score=True,               # Out-of-bag score
        random_state=42,              # Reproducibility
        n_jobs=-1                     # Use all CPU cores
    )
    
    # Train the model
    print("\n  🏋️ Training (building 200 trees)...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"  ✅ Training completed in {train_time:.2f} seconds")
    print(f"  📊 Out-of-Bag Score: {model.oob_score_:.4f} (free validation!)")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_prob, "Random Forest")
    
    print("\n  📊 Evaluation Metrics:")
    print("  " + "-"*50)
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric:>12}: {value:.4f}")
    
    # Feature importance
    print("\n  🔑 Top 5 Feature Importances:")
    print("  " + "-"*50)
    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Detailed report
    print_detailed_report(y_test, y_pred, class_names)
    
    return model, metrics


# ============================================================================
# MODEL 6: XGBOOST (ENSEMBLE)
# ============================================================================
def train_xgboost(X_train, X_test, y_train, y_test, class_names):
    """
    Train and evaluate XGBoost classifier.
    
    WHY XGBOOST?
    - eXtreme Gradient Boosting - often wins ML competitions!
    - Sequential ensemble: Each tree corrects previous tree's errors
    - Regularization built-in (prevents overfitting)
    - Handles missing values natively
    - Very fast (optimized C++ implementation)
    
    BOOSTING vs BAGGING (Random Forest):
    - Bagging (RF): Trees built independently, parallel
    - Boosting (XGB): Trees built sequentially, each corrects previous errors
    - Boosting usually gives better accuracy but can overfit
    
    KEY HYPERPARAMETERS:
    - n_estimators: Number of boosting rounds
    - max_depth: Depth of each tree (usually shallow: 3-10)
    - learning_rate: How much each tree contributes (smaller = more trees needed)
    - subsample: Fraction of samples for each tree
    - colsample_bytree: Fraction of features for each tree
    
    PREPROCESSING REQUIREMENTS:
    - NO scaling required (tree-based)
    - Handles missing values automatically
    
    WHEN TO USE:
    - When you need maximum accuracy
    - Structured/tabular data
    - When you have time to tune hyperparameters
    """
    print("\n" + "="*70)
    print("MODEL 6: XGBOOST (ENSEMBLE)")
    print("="*70)
    
    print("\n  📖 Theory:")
    print("  - Gradient Boosting: Builds trees sequentially")
    print("  - Each tree focuses on errors of previous trees")
    print("  - Uses gradient descent to minimize loss function")
    print("  - Regularization (L1, L2) prevents overfitting")
    print("  - Often the top performer on tabular data!")
    
    print("\n  ⚙️ Hyperparameters chosen:")
    print("  - n_estimators=200: 200 boosting rounds")
    print("  - max_depth=6: Shallow trees (boosting works best with weak learners)")
    print("  - learning_rate=0.1: Step size for each tree's contribution")
    print("  - subsample=0.8: Use 80% of samples per tree (reduces overfitting)")
    print("  - colsample_bytree=0.8: Use 80% of features per tree")
    print("  - objective='multi:softprob': Multi-class with probabilities")
    
    # Initialize the model
    model = XGBClassifier(
        n_estimators=200,             # Number of boosting rounds
        max_depth=6,                  # Shallow trees
        learning_rate=0.1,            # Step size
        subsample=0.8,                # Sample fraction
        colsample_bytree=0.8,         # Feature fraction
        objective='multi:softprob',   # Multi-class objective
        num_class=3,                  # Number of classes
        use_label_encoder=False,      # Suppress warning
        eval_metric='mlogloss',       # Evaluation metric
        random_state=42,              # Reproducibility
        n_jobs=-1                     # Use all CPU cores
    )
    
    # Train the model
    print("\n  🏋️ Training (sequential boosting)...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    print(f"  ✅ Training completed in {train_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_prob, "XGBoost")
    
    print("\n  📊 Evaluation Metrics:")
    print("  " + "-"*50)
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"  {metric:>12}: {value:.4f}")
    
    # Feature importance
    print("\n  🔑 Top 5 Feature Importances:")
    print("  " + "-"*50)
    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Detailed report
    print_detailed_report(y_test, y_pred, class_names)
    
    return model, metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main function to train all 6 models and generate comparison table.
    """
    print("="*70)
    print("STELLAR CLASSIFICATION - MODEL TRAINING & EVALUATION")
    print("M.Tech Machine Learning Assignment 2")
    print("="*70)
    
    # Load preprocessed data
    print("\n📂 Loading preprocessed data...")
    data = joblib.load('preprocessed_data.joblib')
    
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    class_names = list(data['label_encoder'].classes_)
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {list(X_train.columns)}")
    print(f"  Classes: {class_names}")
    
    # Store all results
    all_metrics = []
    all_models = {}
    
    # =========================================================================
    # Train all 6 models
    # Note: Some models use scaled data, others use unscaled
    # =========================================================================
    
    # Model 1: Logistic Regression (NEEDS SCALED DATA)
    model, metrics = train_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, class_names
    )
    all_metrics.append(metrics)
    all_models['logistic_regression'] = model
    
    # Model 2: Decision Tree (NO SCALING NEEDED)
    model, metrics = train_decision_tree(
        X_train, X_test, y_train, y_test, class_names
    )
    all_metrics.append(metrics)
    all_models['decision_tree'] = model
    
    # Model 3: KNN (NEEDS SCALED DATA)
    model, metrics = train_knn(
        X_train_scaled, X_test_scaled, y_train, y_test, class_names
    )
    all_metrics.append(metrics)
    all_models['knn'] = model
    
    # Model 4: Naive Bayes (NO SCALING NEEDED)
    model, metrics = train_naive_bayes(
        X_train, X_test, y_train, y_test, class_names
    )
    all_metrics.append(metrics)
    all_models['naive_bayes'] = model
    
    # Model 5: Random Forest (NO SCALING NEEDED)
    model, metrics = train_random_forest(
        X_train, X_test, y_train, y_test, class_names
    )
    all_metrics.append(metrics)
    all_models['random_forest'] = model
    
    # Model 6: XGBoost (NO SCALING NEEDED)
    model, metrics = train_xgboost(
        X_train, X_test, y_train, y_test, class_names
    )
    all_metrics.append(metrics)
    all_models['xgboost'] = model
    
    # =========================================================================
    # Generate Comparison Table (Required for Assignment)
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON TABLE")
    print("(This table goes in your README.md)")
    print("="*70)
    
    results_df = pd.DataFrame(all_metrics)
    results_df = results_df.set_index('Model')
    
    print("\n")
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv('model_comparison_results.csv')
    print("\n✅ Results saved to 'model_comparison_results.csv'")
    
    # Print markdown table for README
    print("\n" + "-"*70)
    print("MARKDOWN TABLE (Copy this to your README.md):")
    print("-"*70)
    print("\n| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
    print("|---------------|----------|-----|-----------|--------|----|----|")
    for idx, row in results_df.iterrows():
        print(f"| {idx} | {row['Accuracy']:.4f} | {row['AUC']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['MCC']:.4f} |")
    
    # =========================================================================
    # Save all models for Streamlit app
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING MODELS FOR STREAMLIT APP")
    print("="*70)
    
    # Save each model
    import os
    os.makedirs('models', exist_ok=True)
    
    for name, model in all_models.items():
        filepath = f'models/{name}.joblib'
        joblib.dump(model, filepath)
        print(f"  ✅ Saved: {filepath}")
    
    # Save the scaler (needed for Logistic Regression and KNN)
    joblib.dump(data['scaler'], 'models/scaler.joblib')
    print(f"  ✅ Saved: models/scaler.joblib")
    
    # Save label encoder
    joblib.dump(data['label_encoder'], 'models/label_encoder.joblib')
    print(f"  ✅ Saved: models/label_encoder.joblib")
    
    # Save feature names
    joblib.dump(data['feature_names'], 'models/feature_names.joblib')
    print(f"  ✅ Saved: models/feature_names.joblib")
    
    # Save results dataframe
    joblib.dump(results_df, 'models/results_df.joblib')
    print(f"  ✅ Saved: models/results_df.joblib")
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE!")
    print("="*70)
    print("""
    Next Steps:
    1. Review the comparison table above
    2. Write observations for each model (Step 3)
    3. Build Streamlit app (Step 4)
    4. Deploy to Streamlit Cloud (Step 5)
    
    Files created:
    - models/*.joblib (all trained models)
    - model_comparison_results.csv (metrics table)
    """)
    
    return results_df, all_models


if __name__ == "__main__":
    results_df, all_models = main()
