# src.py
"""
Sparse Representation-based Classification (SRC).

The unweighted baseline of WSRC: each test sample is represented as a
sparse linear combination of ALL training dictionary atoms (via Lasso),
and the class with smallest reconstruction residual wins.

Reference:
    Wright et al. (2009) - Robust Face Recognition via Sparse Representation
"""

import numpy as np
from sklearn.linear_model import Lasso


def src_predict(X_train, y_train, X_test, alpha=0.01):
    """
    Sparse Representation-based Classification (SRC).

    For each test sample x:
        1. Solve Lasso: find sparse coef s.t. X_train.T @ coef ≈ x
        2. For each class c, zero out all coefficients not belonging to c
        3. Reconstruct x using only class-c atoms
        4. Predict the class with the smallest reconstruction error

    Args:
        X_train (np.ndarray): Training features (n_samples, n_features).
        y_train (np.ndarray): Training labels   (n_samples,).
        X_test  (np.ndarray): Test features     (n_test, n_features).
        alpha   (float): Lasso regularization. Smaller = denser representation.

    Returns:
        np.ndarray: Predicted labels for each test sample.
    """
    predictions = []
    classes = np.unique(y_train)
    X_train_T = X_train.T   # shape: (n_features, n_samples)

    for x in X_test:
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
        lasso.fit(X_train_T, x)
        coef = lasso.coef_
        errors = []
        for c in classes:
            mask = (y_train == c)
            coef_c = np.zeros_like(coef)
            coef_c[mask] = coef[mask]
            x_hat = X_train_T @ coef_c
            errors.append(np.linalg.norm(x - x_hat))

        predictions.append(classes[np.argmin(errors)])

    return np.array(predictions)