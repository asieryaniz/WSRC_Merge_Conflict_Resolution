# knn.py
"""
K-Nearest Neighbors classifier wrapper.

KNN is the natural baseline for comparing against WSRC because both are
distance-based methods. WSRC uses inverse-distance weights to build a sparse
representation; KNN directly votes from the k nearest neighbors. If WSRC
does not outperform KNN, it suggests that the sparse-representation mechanism
is not adding value beyond simple proximity-based classification.

Notes:
    - Requires feature normalization (StandardScaler) before use.
      KNN is sensitive to feature scale; the same normalization pipeline
      used for SRC/WSRC should be applied here.
    - weights="distance" makes KNN closest in spirit to WSRC: both
      give more influence to nearby training samples.
    - n_jobs=-1 uses all available CPU cores for neighbor search.
"""

from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train, n_neighbors=11, metric="euclidean",
              weights="distance", **kwargs):
    """
    Train a KNN classifier.

    Args:
        X_train (np.ndarray or pd.DataFrame): Normalized training features.
        y_train (np.ndarray): Training labels.
        n_neighbors (int): Number of neighbors k. Odd values avoid ties.
        metric (str): "euclidean" (consistent with WSRC) or "manhattan".
        weights (str): "distance" (inverse-distance votes) or "uniform".
        **kwargs: Additional sklearn KNeighborsClassifier parameters.

    Returns:
        KNeighborsClassifier: Fitted model.
    """
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        algorithm="auto",
        n_jobs=-1,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def predict_knn(model, X_test):
    """
    Generate predictions from a trained KNN model.

    Args:
        model (KNeighborsClassifier): Fitted model.
        X_test (np.ndarray or pd.DataFrame): Normalized test features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    return model.predict(X_test)