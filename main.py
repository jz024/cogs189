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
from models.csp import csp_fit, csp_transform
from models.evaluate import evaluate_model
from models.coral import coral_transform

# TA-CSPNN
from models.train import train_tacnn_pipeline

# Deep CORAL
from models.deep_coral import build_deep_coral_model
from models.train import train_deep_coral_model

# If you need multi_band_filter/csp_transform for test:
from models.tacnn import multi_band_filter, csp_transform as tacnn_csp_transform

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

    # Print dataset shapes
    print_data_info(X_train, y_train, X_val, y_val, X_test, y_test)

    # ====================================================
    # =           2. Baseline CSP + LDA                =
    # ====================================================
    print("\n=== Baseline CSP + LDA ===")
    n_components = 3
    csp_filters = csp_fit(X_train, y_train, n_components=n_components)

    # Transform train/val/test via CSP
    X_train_csp = csp_transform(X_train, csp_filters)
    X_val_csp   = csp_transform(X_val,   csp_filters)
    X_test_csp  = csp_transform(X_test,  csp_filters)

    # Train LDA on baseline CSP features
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train_csp, y_train)

    # Evaluate on Validation + Test
    evaluate_model(clf, X_val_csp, y_val, label="Validation")
    evaluate_model(clf, X_test_csp, y_test, label="Test")

    # ====================================================
    # =           3. CORAL Adaptation on CSP           =
    # ====================================================
    print("\n=== CORAL-Adaptive CSP ===")
    # Align val and test sets to training distribution
    X_val_csp_coral  = coral_transform(X_train_csp, X_val_csp,  reg=1e-5)
    X_test_csp_coral = coral_transform(X_train_csp, X_test_csp, reg=1e-5)

    # Evaluate CORAL on validation + test
    evaluate_model(clf, X_val_csp_coral,  y_val,  label="Validation CORAL")
    evaluate_model(clf, X_test_csp_coral, y_test, label="Test CORAL")

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

    # Evaluate on test. Must do same multi-band filter + CSP transform
    X_test_filt = multi_band_filter(X_test, freq_bands_used)
    X_test_csp_tacnn = tacnn_csp_transform(X_test_filt, filters_tacnn)
    X_test_csp_tacnn = X_test_csp_tacnn.reshape((X_test_csp_tacnn.shape[0], X_test_csp_tacnn.shape[1], 1))

    evaluate_model(model_tacnn, X_test_csp_tacnn, y_test, label="Test TACSPNN")

    # ====================================================
    # =           5. Deep CORAL Training               =
    # ====================================================
    print("\n=== Deep CORAL ===")
    # We can reuse X_train_csp, X_test_csp as "source" and "target" features
    # if we want to do domain adaptation from train->test.

    # For CNN input, reshape so shape = (samples, features, 1)
    X_train_csp_deep = X_train_csp.reshape(X_train_csp.shape[0], X_train_csp.shape[1], 1)
    X_test_csp_deep  = X_test_csp.reshape( X_test_csp.shape[0],  X_test_csp.shape[1], 1)

    # Build the Deep CORAL model
    from models.deep_coral import build_deep_coral_model
    input_shape = (X_train_csp_deep.shape[1], 1)
    model_deep_coral = build_deep_coral_model(input_shape)

    # Use a custom training function that includes CORAL alignment
    # e.g. train_deep_coral_model(
    #   model, X_train_csp_deep, y_train, X_test_csp_deep, y_test, ...
    # )
    alpha = 0.01  # weighting for CORAL loss
    epochs = 50
    batch_size = 16

    final_acc, model_deep_coral, feature_extractor = train_deep_coral_model(
        model_deep_coral,
        X_train_csp_deep, y_train,       # Source data
        X_test_csp_deep,  y_test,        # Target data
        epochs=epochs,
        batch_size=batch_size,
        alpha=alpha,
        verbose=1
    )

    print(f"Deep CORAL final test accuracy: {final_acc:.4f}")

    # Optionally do PCA or other analysis on feature_extractor outputs.

if __name__ == "__main__":
    main()
