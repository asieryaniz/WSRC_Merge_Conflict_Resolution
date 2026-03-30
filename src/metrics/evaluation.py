from sklearn.metrics import accuracy_score, f1_score

def compute_accuracy(y_true, y_pred):
    """Calculate the accuracy."""
    return accuracy_score(y_true, y_pred)

def compute_f1(y_true, y_pred, average="weighted"):
    """Calculate the F1-score."""
    return f1_score(y_true, y_pred, average=average)