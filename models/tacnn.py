# models/tacnn.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, 
                                     AveragePooling1D, Dropout, 
                                     Flatten, Dense, Lambda)
from scipy.linalg import eigh

from utils.preprocessing import bandpass_filter

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

def csp_fit(X, y, n_components=6, epsilon=1e-6):
    """
    Fit CSP filters for two-class data in {0,1}.
    If X has shape (trials, channels, samples, freq_bands),
    we average across freq bands dimension before computing covariance.
    """
    # If multi-band, average over freq band => (trials, channels, samples)
    if X.ndim == 4:
        X = np.mean(X, axis=-1)

    # Separate trials by class
    X_class0 = X[y == -1]
    X_class1 = X[y == 1]

    if len(X_class0) == 0 or len(X_class1) == 0:
        raise ValueError("One of the classes (0 or 1) has no samples!")

    cov_class0 = np.mean([_regularized_cov(trial_data, epsilon) for trial_data in X_class0], axis=0)
    cov_class1 = np.mean([_regularized_cov(trial_data, epsilon) for trial_data in X_class1], axis=0)

    w, v = eigh(cov_class0, cov_class1)
    idx = np.argsort(np.abs(w))[::-1]
    v = v[:, idx]

    # Top & bottom n_components
    filters = np.hstack([v[:, :n_components], v[:, -n_components:]])
    return filters

def csp_transform(X, filters):
    """
    Transform EEG data X with the given CSP filters to obtain log-variance features.
    If multi-band => average freq dimension first.
    Returns shape: (n_trials, 2*n_components).
    """
    if X.ndim == 4:
        X = np.mean(X, axis=-1)

    n_trials, _, _ = X.shape
    n_filters = filters.shape[1]
    features = np.zeros((n_trials, n_filters))

    for i in range(n_trials):
        projected = filters.T @ X[i]
        var = np.var(projected, axis=1)
        features[i, :] = np.log(var + 1e-10)

    return features

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
