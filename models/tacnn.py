# models/tacnn.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, 
                                     AveragePooling1D, Dropout, 
                                     Flatten, Dense, Lambda)
from scipy.linalg import eigh

from utils.preprocessing import bandpass_filter
from models.csp import cov, spatialFilter, apply_CSP_filter, log_norm_band_power

def multi_band_filter(X, frequency_bands, fs=1000):
    """
    Apply bandpass filtering for multiple frequency bands to EEG data.
    Returns shape: (trials, channels, samples, n_bands).
    """
    band_filtered = []
    for (low, high) in frequency_bands:
        filtered = bandpass_filter(X, low, high, fs=fs)
        band_filtered.append(filtered)
    X_filt = np.stack(band_filtered, axis=-1)
    return X_filt

def _regularized_cov(trial_data, epsilon=1e-6):
    """
    Compute covariance for a single trial with slight regularization.
    Ensures shape (channels, samples) and adds epsilon on diagonal.
    """
    if trial_data.shape[0] > trial_data.shape[1]:
        trial_data = trial_data.T
    cov_matrix = np.cov(trial_data)
    cov_matrix += epsilon * np.eye(cov_matrix.shape[0])
    return cov_matrix

def apply_csp_per_band(X_train, y_train, X_val, frequency_bands, n_components=6):
    """Applies CSP transformation separately for each frequency band."""
    n_bands = len(frequency_bands)
    
    X_train_csp_list, X_val_csp_list = [], []
    filters_list = []
    
    for band_idx in range(n_bands):
        X_train_band = X_train[..., band_idx]  # Extract one frequency band
        X_val_band = X_val[..., band_idx]
        
        # Separate trials by class
        X_train_class1 = X_train_band[y_train == -1]
        X_train_class2 = X_train_band[y_train == 1]

        if len(X_train_class1) == 0 or len(X_train_class2) == 0:
            raise ValueError("One of the classes has no samples!")

        # Compute CSP filters
        Ra = np.mean([cov(trial) for trial in X_train_class1], axis=0)
        Rb = np.mean([cov(trial) for trial in X_train_class2], axis=0)
        filters = spatialFilter(Ra, Rb)[:n_components]
        filters_list.append(filters)

        # Apply CSP and extract features
        X_train_csp = apply_CSP_filter(X_train_band, filters)
        X_val_csp = apply_CSP_filter(X_val_band, filters)

        X_train_csp_list.append(log_norm_band_power(X_train_csp))
        X_val_csp_list.append(log_norm_band_power(X_val_csp))

    # Merge features from all bands
    X_train_csp = np.concatenate(X_train_csp_list, axis=1)
    X_val_csp = np.concatenate(X_val_csp_list, axis=1)

    return X_train_csp, X_val_csp, filters_list

def build_tacnn_model(input_shape):
    """
    Build the TA-CSPNN model with 1D convolution, square activation,
    pooling, dropout, and dense output.
    """
    model = Sequential([
        Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Lambda(lambda x: tf.math.square(x)),  # Square activation
        AveragePooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(name="feature_layer"),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
