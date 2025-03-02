import numpy as np
from scipy.linalg import eigh

def compute_covariance(trial_data):
    """
    Compute the covariance matrix for a given single-trial EEG segment.

    Parameters:
        trial_data (ndarray): EEG data of shape (channels, samples)

    Returns:
        ndarray: Covariance matrix of shape (channels, channels)
    """
    return np.cov(trial_data)

def csp_fit(X, y, n_components=3):
    """
    Fit CSP spatial filters given EEG data from two classes.

    Parameters:
        X (ndarray): EEG data, shape (trials, channels, samples)
        y (ndarray): Corresponding labels, shape (trials,). 
                     Assumed to be -1 or +1.
        n_components (int): Number of CSP components to retain per class

    Returns:
        ndarray: CSP spatial filters, shape (channels, 2*n_components)
    """
    # Separate data by class
    X_class1 = X[y == -1]
    X_class2 = X[y == 1]

    # Compute mean covariance for each class
    cov_class1 = np.mean([compute_covariance(trial) for trial in X_class1], axis=0)
    cov_class2 = np.mean([compute_covariance(trial) for trial in X_class2], axis=0)

    # Composite covariance
    cov_sum = cov_class1 + cov_class2

    # Solve generalized eigenvalue problem
    w, v = eigh(cov_class1, cov_sum)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(w)[::-1]
    v = v[:, sorted_indices]

    # Take the first and last n_components
    filters = np.hstack((v[:, :n_components], v[:, -n_components:]))

    return filters

def csp_transform(X, filters):
    """
    Apply CSP filters to extract log-variance features.

    Parameters:
        X (ndarray): EEG data, shape (trials, channels, samples)
        filters (ndarray): CSP filters, shape (channels, 2*n_components)

    Returns:
        ndarray: Log-variance features, shape (trials, 2*n_components)
    """
    n_trials, _, _ = X.shape
    n_filters = filters.shape[1]
    features = np.zeros((n_trials, n_filters))

    for i in range(n_trials):
        # Project data through CSP filters
        projected = filters.T @ X[i, :, :]
        var = np.var(projected, axis=1)
        # Add small constant to avoid log(0)
        features[i, :] = np.log(var + 1e-10)

    return features
