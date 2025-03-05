#!/usr/bin/env python
"""
Main script to run multiple pipelines:
1. Classic CSP + LDA
2. CORAL adaptation
3. TA-CSPNN
4. Deep CORAL (custom training loop)
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Local imports
from utils.preprocessing import load_and_preprocess_data, print_data_info
from models.csp import cov, spatialFilter, apply_CSP_filter, log_norm_band_power
from models.evaluate import evaluate_model
from models.coral import coral_transform
from scipy.signal import butter, lfilter
from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score, confusion_matrix

# TA-CSPNN
from models.train import train_tacnn_pipeline

# Deep CORAL
from models.deep_coral import build_deep_coral_model
from models.train import train_deep_coral_model

# If you need multi_band_filter/csp_transform for test:
from models.tacnn import multi_band_filter

def main():
    # ----------------------------------------------------
    # 1. Load and Preprocess Data
    # ----------------------------------------------------
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(
        train_cnt_file="data/raw/train/Competition_train_cnt.txt",
        train_lab_file="data/raw/train/Competition_train_lab.txt",
        test_cnt_file="data/raw/test/test.txt",
        test_lab_file="data/raw/test/test_label.txt",
        n_trials_train=278,
        n_trials_test=100,
        n_channels=64,
        n_samples=3000,
        test_size=0.2,
        random_state=42,
        lowcut=8,
        highcut=30,
        fs=1000,
        order=5
    )


    # ====================================================
    # =           2. Baseline CSP + LDA                =
    # ====================================================
    print("\n=== Baseline CSP + LDA ===")

    X_train_class1 = X_train[y_train == -1]
    X_train_class2 = X_train[y_train == 1]
    Ra = np.mean([cov(trial) for trial in X_train_class1], axis=0)
    Rb = np.mean([cov(trial) for trial in X_train_class2], axis=0)
    n_components = 3
    csp_filters = spatialFilter(Ra, Rb)[:n_components]
    
    # Transform train/val/test via CSP
    X_train_filtered = apply_CSP_filter(X_train, csp_filters)
    X_val_filtered = apply_CSP_filter(X_val, csp_filters)
    X_test_filtered = apply_CSP_filter(X_test, csp_filters)

    # Extract features
    X_train_features = log_norm_band_power(X_train_filtered)
    X_val_features = log_norm_band_power(X_val_filtered)
    X_test_features = log_norm_band_power(X_test_filtered)

    # Train LDA on baseline CSP features
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_train_features, y_train)

    # Evaluate on Validation + Test
    evaluate_model(clf, X_val_features, y_val, label="Validation")
    evaluate_model(clf, X_test_features, y_test, label="Test")
    

    # ====================================================
    # =           3. CORAL Adaptation on CSP           =
    # ====================================================
    print("\n=== CORAL-Adaptive CSP ===")
    # Align val and test sets to training distribution
    X_val_csp_coral  = coral_transform(X_train_features, X_val_features,  reg=1e-2)
    X_test_csp_coral = X_test_features
    X_train_csp_coral = coral_transform(X_train_features, X_train_features, reg=1e-2)

    clf_coral = LinearDiscriminantAnalysis()
    clf_coral.fit(X_train_csp_coral, y_train)

    # Evaluate CORAL on validation + test
    evaluate_model(clf_coral, X_val_csp_coral,  y_val,  label="Validation CORAL")
    evaluate_model(clf_coral, X_test_csp_coral, y_test, label="Test CORAL")

    print("Train Cov:", np.cov(X_train_features, rowvar=False))
    print("Test Cov:", np.cov(X_test_features,  rowvar=False))
    
    print(clf_coral.predict(X_test_csp_coral))
    print(y_test)

    # ====================================================
    # =           4. TA-CSPNN (Multi-Band)             =
    # ====================================================
    print("\n=== TA-CSPNN ===")
    freq_bands = [(4,8), (8,12), (12,16), (16,20), (20,24), (24,28), (28,32), (32,36), (36,40)]

    model_tacnn, filters_tacnn, freq_bands_used = train_tacnn_pipeline(
        X_train, y_train,
        X_val,   y_val,
        frequency_bands=freq_bands,
        n_components=6,
        epochs=100,
        batch_size=16,
        patience=5
    )

    X_test_filt = multi_band_filter(X_test, freq_bands_used)

    X_test_csp_list = []

    for band_idx, filters in enumerate(filters_tacnn):
        X_test_band = X_test_filt[..., band_idx]
        X_test_csp_band = apply_CSP_filter(X_test_band, filters)
        X_test_csp_features = log_norm_band_power(X_test_csp_band)
        X_test_csp_list.append(X_test_csp_features)

    # Merge features from all bands
    X_test_csp = np.concatenate(X_test_csp_list, axis=1)

    # Reshape for CNN (samples, features, 1)
    X_test_csp = X_test_csp.reshape((X_test_csp.shape[0], X_test_csp.shape[1], 1))

    # Get model predictions
    probs = model_tacnn.predict(X_test_csp)
    preds = np.where(probs >= 0.5, 1, -1)

    # Compute accuracy
    test_acc = accuracy_score(y_test, preds)
    print("Test Accuracy:", test_acc)
    print(probs)
    print(y_test)

    # ====================================================
    # =           5. Deep CORAL Training               =
    # ====================================================
    print("\n=== Deep CORAL ===")
    # We can reuse X_train_csp, X_test_csp as "source" and "target" features
    # if we want to do domain adaptation from train->test.
    # Compute covariance matrices
    C_S = np.cov(X_train_features, rowvar=False) + np.eye(X_train_features.shape[1])
    C_T = np.cov(X_test_features, rowvar=False) + np.eye(X_test_features.shape[1])

    C_S_sqrt = sqrtm(C_S)
    C_T_sqrt = sqrtm(C_T)

    A_coral = np.linalg.inv(C_S_sqrt) @ C_T_sqrt

    # Transform source features
    X_train_coral = X_train_features @ A_coral
    X_test_coral  = X_test_features
    # For CNN input, reshape so shape = (samples, features, 1)
    X_train_csp_deep = X_train_coral.reshape(X_train_coral.shape[0], X_train_coral.shape[1], 1)
    X_test_csp_deep  = X_test_coral.reshape( X_test_coral.shape[0],  X_test_coral.shape[1], 1)

    # Build the Deep CORAL model
    input_shape = (X_train_csp_deep.shape[1], 1)
    model_deep_coral = build_deep_coral_model(input_shape)

    # Use a custom training function that includes CORAL alignment
    # e.g. train_deep_coral_model(
    #   model, X_train_csp_deep, y_train, X_test_csp_deep, y_test, ...
    # )
    alpha = 0.01  # weighting for CORAL loss
    epochs = 50
    batch_size = 16
    y_train_new = np.where(y_train == -1, 0, 1)

    final_acc, model_deep_coral, feature_extractor = train_deep_coral_model(
        model_deep_coral,
        X_train_csp_deep, y_train_new,       # Source data
        X_test_csp_deep,  y_test,        # Target data
        epochs=epochs,
        batch_size=batch_size,
        alpha=alpha,
        verbose=1
    )

    print(model_deep_coral.predict(X_test_csp_deep))
    print(y_test)

    # Optionally do PCA or other analysis on feature_extractor outputs.

if __name__ == "__main__":
    main()
