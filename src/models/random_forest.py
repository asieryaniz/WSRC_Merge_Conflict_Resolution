# random_forest.py
from sklearn.ensemble import RandomForestClassifier
 
 
# Hyperparameters from the paper (Section III-E)
RF_HYPERPARAMS = {
    "n_estimators": 400,
    "random_state": 42,
    "n_jobs": -1,
}
 
 
def train_rf(X_train, y_train, **kwargs):
    """
    Train a Random Forest classifier.
 
    Uses the hyperparameters from Elias et al. by default, which are the same
    ones adopted by the paper (Section III-E) to ensure comparability.
 
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        **kwargs: Optional overrides for hyperparameters.
 
    Returns:
        RandomForestClassifier: Fitted model.
    """
    params = {**RF_HYPERPARAMS, **kwargs}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model
 
 
def predict_rf(model, X_test):
    """
    Generate predictions from a trained Random Forest model.
 
    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame or np.ndarray): Test features.
 
    Returns:
        np.ndarray: Predicted class labels.
    """
    return model.predict(X_test)