# wsrc.py
"""
Weighted Sparse Representation-based Classification (WSRC).

SRC theory: a test sample x can be approximately represented as a linear
combination of the training samples. The class whose dictionary columns
best reconstruct x (minimum residual) wins. WSRC extends this by
weighting training samples before solving the sparse representation,
allowing the model to emphasize more relevant training examples.

Reference:
    Wright et al. (2009) - Robust Face Recognition via Sparse Representation
    (original SRC paper; WSRC is the weighted extension)
"""

import numpy as np
from collections import Counter
from sklearn.linear_model import Lasso


# Weight computation methods
def weights_by_similarity(X_train, X_test_sample):
    """
    Compute weights as inverse Euclidean distance between the test sample
    and each training sample. Closer training samples receive higher weights,
    making the representation more locally adaptive.

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
        X_test_sample (np.ndarray): Single test sample (n_features,).

    Returns:
        np.ndarray: Normalized weights in [0, 1] for each training sample.
    """
    distances = np.linalg.norm(X_train - X_test_sample, axis=1)
    distances = np.where(distances == 0, 1e-6, distances)  # avoid division by zero
    weights = 1.0 / distances
    return weights / weights.max()


def weights_by_class_frequency(y_train, top_k=1):
    """
    Compute weights based on class frequency. Samples belonging to the
    top-k most frequent classes receive weight 1.0; all others get 0.5.
    This biases representation towards dominant classes.

    Args:
        y_train (np.ndarray): Training labels (n_samples,).
        top_k (int): Number of top classes to prioritize.

    Returns:
        np.ndarray: Normalized weights for each training sample.
    """
    counts = Counter(y_train)
    top_classes = {c for c, _ in counts.most_common(top_k)}
    weights = np.array([1.0 if label in top_classes else 0.5 for label in y_train])
    return weights / weights.max()


def compute_weights(X_train, y_train, X_test_sample, method="similarity", top_k=1):
    """
    Wrapper to select the weight computation method.

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
        y_train (np.ndarray): Training labels (n_samples,).
        X_test_sample (np.ndarray): Single test sample (n_features,).
        method (str): One of:
            - "similarity"  : inverse Euclidean distance (recommended)
            - "class"       : class-frequency-based weights
            - "uniform"     : all weights = 1 (equivalent to standard SRC)
        top_k (int): Only for method="class": number of top classes to weight up.

    Returns:
        np.ndarray: Weights for each training sample.
    """
    if method == "similarity":
        return weights_by_similarity(X_train, X_test_sample)
    elif method == "class":
        return weights_by_class_frequency(y_train, top_k=top_k)
    elif method == "uniform":
        return np.ones(len(y_train))
    else:
        raise ValueError(
            f"Unknown weight method: '{method}'. "
            "Choose from 'similarity', 'class', 'uniform'."
        )


# WSRC predictor
def wsrc_predict(X_train, y_train, X_test, weights=None, alpha=0.01):
    """
    Weighted Sparse Representation-based Classification (WSRC).

    For each test sample x:
        1. Scale the training dictionary by per-sample weights.
        2. Solve a Lasso problem to find sparse coefficients.
        3. For each class c, reconstruct x using only the coefficients
           of class c's training samples.
        4. Predict the class with the smallest reconstruction error.

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
        y_train (np.ndarray): Training labels (n_samples,).
        X_test  (np.ndarray): Test features   (n_test, n_features).
        weights (np.ndarray or None): Per-sample weights (n_samples,).
                                      If None, uniform weights are used (= SRC).
        alpha   (float): Lasso regularization strength. Smaller = denser solution.

    Returns:
        np.ndarray: Predicted labels for each test sample.
    """
    if weights is None:
        weights = np.ones(len(y_train))

    classes = np.unique(y_train)
    # Transpose: shape (n_features, n_samples) — each column is a training sample
    X_train_T = X_train.T
    predictions = []

    for x in X_test:
        # Weight the training dictionary columns
        # Broadcasting: each column j of X_train_T is scaled by weights[j]
        X_weighted = X_train_T * weights  # (n_features, n_samples)

        # Lasso: find sparse alpha s.t. X_weighted @ alpha ≈ x
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
        lasso.fit(X_weighted, x)
        coef = lasso.coef_  # (n_samples,)

        # Reconstruction error per class
        errors = []
        for c in classes:
            mask = (y_train == c)
            coef_c = np.zeros_like(coef)
            coef_c[mask] = coef[mask]
            x_hat  = X_train_T @ coef_c  # reconstruct using class-c atoms only
            errors.append(np.linalg.norm(x - x_hat))

        predictions.append(classes[np.argmin(errors)])

    return np.array(predictions)