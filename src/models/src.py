import numpy as np
from sklearn.linear_model import Lasso


def src_predict(X_train, y_train, X_test, alpha=0.01):
    """
    Perform classification using Sparse Representation-based Classification (SRC).

    Args:
        X_train (np.array): Training features (n_samples, n_features)
        y_train (np.array): Training labels (n_samples,)
        X_test (np.array): Test features (n_samples, n_features)
        alpha (float): Regularization parameter for Lasso

    Returns:
        np.array: Predicted labels for X_test
    """

    predictions = []
    classes = np.unique(y_train)

    # Transpose once to avoid repeating
    X_train_T = X_train.T

    for x in X_test:
        # Fit Lasso to represent x as combination of training samples
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
        lasso.fit(X_train_T, x)

        coef = lasso.coef_
        errors = []

        # Compute reconstruction error per class
        for c in classes:
            mask = (y_train == c)

            coef_c = np.zeros_like(coef)
            coef_c[mask] = coef[mask]

            x_reconstructed = X_train_T @ coef_c
            error = np.linalg.norm(x - x_reconstructed)

            errors.append(error)

        pred = classes[np.argmin(errors)]
        predictions.append(pred)

    return np.array(predictions)