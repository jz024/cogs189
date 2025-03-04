# utils/preprocessing.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=8, highcut=30, fs=1000, order=5):
    """
    Apply a zero-phase bandpass filter to the input data using filtfilt.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def load_and_preprocess_data(
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
):
    """
    Load, reshape, split (train/validation), filter, and save the data.
    Returns the filtered arrays (X_train_filt, y_train, X_val_filt, y_val, X_test_filt, y_test).
    """

    # --------------------------
    # 1) Load raw data
    # --------------------------
    train_cnt = np.loadtxt(train_cnt_file)
    y_train = np.loadtxt(train_lab_file)
    test_cnt = np.loadtxt(test_cnt_file)
    y_test = np.loadtxt(test_lab_file)

    # --------------------------
    # 2) Reshape
    # --------------------------
    X_train = train_cnt.reshape(n_trials_train, n_channels, n_samples)
    X_test  = test_cnt.reshape(n_trials_test,  n_channels, n_samples)

    # --------------------------
    # 3) Split off validation set from train
    # --------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state
    )

    # --------------------------
    # 4) Bandpass filter
    # --------------------------
    # X_train_filt = bandpass_filter(X_train, lowcut, highcut, fs, order)
    # X_val_filt   = bandpass_filter(X_val,   lowcut, highcut, fs, order)
    # X_test_filt  = bandpass_filter(X_test,  lowcut, highcut, fs, order)

    # --------------------------
    # 5) Save preprocessed data
    # --------------------------
    # Create directories (if not exist)
    os.makedirs("data/preprocessed/train", exist_ok=True)
    os.makedirs("data/preprocessed/validation", exist_ok=True)
    os.makedirs("data/preprocessed/test", exist_ok=True)

    # Train
    np.save("data/preprocessed/train/X_train_filt.npy", X_train)
    np.save("data/preprocessed/train/y_train.npy",      y_train)

    # Validation
    np.save("data/preprocessed/validation/X_val_filt.npy", X_val)
    np.save("data/preprocessed/validation/y_val.npy",      y_val)

    # Test
    np.save("data/preprocessed/test/X_test_filt.npy", X_test)
    np.save("data/preprocessed/test/y_test.npy",      y_test)

    # Return the filtered data (useful if you want to proceed in-memory)
    return X_train, y_train, X_val, y_val, X_test, y_test

def print_data_info(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Helper function to print dataset shapes.
    """
    print(f"Train Set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val Set:   X={X_val.shape},   y={y_val.shape}")
    print(f"Test Set:  X={X_test.shape},  y={y_test.shape}")
