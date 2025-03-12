# utils/preprocessing.py

import os.path as path
import numpy as np
from wyrm.types import Data

def load_dataset(train_cnt_file, train_lab_file, test_cnt_file, test_lab_file, val_ratio=0.2, random_state=42):

    train_data  = np.loadtxt(train_cnt_file,  dtype='float64')
    train_label = np.loadtxt(train_lab_file, dtype='int').ravel()
    test_data   = np.loadtxt(test_cnt_file,   dtype='float64')
    test_label  = np.loadtxt(test_lab_file,  dtype='int').ravel()

    train_data = train_data.reshape(278, 3000, 64)
    test_data = test_data.reshape(100, 3000, 64)

    # ----------------------------------------------------------------
    # 3) Convert -1 => 0 in labels if your dataset uses -1/+1
    # ----------------------------------------------------------------
    train_label[train_label == -1] = 0
    test_label[test_label == -1]   = 0

    # ----------------------------------------------------------------
    # 4) Split the training data into train/val sets
    # ----------------------------------------------------------------
    n_epochs = train_data.shape[0]
    idx = np.arange(n_epochs)
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(idx)
    boundary = int(n_epochs * (1 - val_ratio))
    idx_train = idx[:boundary]
    idx_val   = idx[boundary:]

    data_train  = train_data[idx_train]
    label_train = train_label[idx_train]
    data_val    = train_data[idx_val]
    label_val   = train_label[idx_val]

    # ----------------------------------------------------------------
    # 5) Create epoched Wyrm Data objects
    #    - "Class" axis => labels
    #    - "Time" axis  => time samples
    #    - "Channel"    => channel names (if you have them)
    # ----------------------------------------------------------------
    # TRAIN
    axes_train = [
        label_train,                             # axis 0: labels
        np.arange(data_train.shape[1]),          # axis 1: time
        [str(ch) for ch in range(data_train.shape[2])]  # axis 2: channels
    ]
    dat_train = Data(
        data=data_train,
        axes=axes_train,
        names=['Class', 'Time', 'Channel'],
        units=['#', 'ms', '#'],
    )
    dat_train.fs = 1000                     # sampling rate, if known
    dat_train.class_names = ['pinky', 'tongue']  # or ['class0', 'class1']

    # VAL
    axes_val = [
        label_val,
        np.arange(data_val.shape[1]),
        [str(ch) for ch in range(data_val.shape[2])]
    ]
    dat_val = Data(
        data=data_val,
        axes=axes_val,
        names=['Class', 'Time', 'Channel'],
        units=['#', 'ms', '#'],
    )
    dat_val.fs = 1000
    dat_val.class_names = ['pinky', 'tongue']

    # TEST
    # For test data, the axis 0 is often "Epoch" (unknown labels),
    # but if you *do* have real labels, put them as well.
    axes_test = [
        test_label,                               # or np.arange(num_test_epochs)
        np.arange(test_data.shape[1]),
        [str(ch) for ch in range(test_data.shape[2])]
    ]
    dat_test = Data(
        data=test_data,
        axes=axes_test,
        names=['Class', 'Time', 'Channel'],
        units=['#', 'ms', '#'],
    )
    dat_test.fs = 1000

    # ----------------------------------------------------------------
    # 6) Map any remaining -1 => 0 (just a precaution)
    # ----------------------------------------------------------------
    dat_train.axes[0][dat_train.axes[0] == -1] = 0
    dat_val.axes[0][dat_val.axes[0] == -1]     = 0
    dat_test.axes[0][dat_test.axes[0] == -1]   = 0

    # ----------------------------------------------------------------
    # 7) Return three Data objects
    # ----------------------------------------------------------------
    return dat_train, dat_val, dat_test


    # # --------------------------
    # # 5) Save preprocessed data
    # # --------------------------
    # # Create directories (if not exist)
    # os.makedirs("data/preprocessed/train", exist_ok=True)
    # os.makedirs("data/preprocessed/validation", exist_ok=True)
    # os.makedirs("data/preprocessed/test", exist_ok=True)

    # # Train
    # np.save("data/preprocessed/train/X_train_filt.npy", X_train)
    # np.save("data/preprocessed/train/y_train.npy",      y_train)

    # # Validation
    # np.save("data/preprocessed/validation/X_val_filt.npy", X_val)
    # np.save("data/preprocessed/validation/y_val.npy",      y_val)

    # # Test
    # np.save("data/preprocessed/test/X_test_filt.npy", X_test)
    # np.save("data/preprocessed/test/y_test.npy",      y_test)

    # # Return the filtered data (useful if you want to proceed in-memory)
    # return X_train, y_train, X_val, y_val, X_test, y_test

def print_data_info(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Helper function to print dataset shapes.
    """
    print(f"Train Set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val Set:   X={X_val.shape},   y={y_val.shape}")
    print(f"Test Set:  X={X_test.shape},  y={y_test.shape}")
