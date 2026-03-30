from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train, n_estimators=400, random_state=42):
    """
    Trains a Random Forest.

    Args:
        X_train (pd.DataFrame or np.array): Training features.
        y_train (np.array): Training labels.
        n_estimators (int): Number of trees.
        random_state (int): Seed for reproducibility.

    Returns:
        RandomForestClassifier: Trained model.
    """    
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def predict_rf(model, X_test):
    """
    Makes predictions using the Random Forest model.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame or np.array): Test features.

    Returns:
        np.array: Predictions.
    """
    return model.predict(X_test)