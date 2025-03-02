# models/coral.py

import numpy as np

def coral_transform(X_source, X_target, reg=1e-5):
    """
    Apply CORAL adaptation to align the target (test) feature distribution
    to the source (train) feature distribution.

    Parameters:
    -----------
    X_source : ndarray (n_samples_source, n_features)
        Source domain features (e.g., training CSP features)
    X_target : ndarray (n_samples_target, n_features)
        Target domain features (e.g., test CSP features)
    reg : float, optional
        Regularization term to avoid division by zero in eigenvalues

    Returns:
    --------
    X_target_aligned : ndarray (n_samples_target, n_features)
        Adapted target features with distribution aligned to the source
    """
    # 1. Compute covariance matrices
    cov_source = np.cov(X_source, rowvar=False)
    cov_target = np.cov(X_target, rowvar=False)

    # 2. Eigen-decomposition of each covariance
    d_s, V_s = np.linalg.eigh(cov_source)
    d_t, V_t = np.linalg.eigh(cov_target)

    # 3. Regularize small eigenvalues
    d_s[d_s < reg] = reg
    d_t[d_t < reg] = reg

    # 4. Compute "whitening" for source and "re-coloring" for target
    #    Ws:  (n_features x n_features)
    #    Wt:  (n_features x n_features)
    Ws = V_s @ np.diag(1.0 / np.sqrt(d_s)) @ V_s.T
    Wt = V_t @ np.diag(np.sqrt(d_t))       @ V_t.T

    # 5. Adapt X_target by re-coloring with Ws
    X_target_aligned = (X_target @ Wt) @ Ws

    return X_target_aligned
