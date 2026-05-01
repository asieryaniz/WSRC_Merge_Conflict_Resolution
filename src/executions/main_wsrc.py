# main_wsrc.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

from src.data.preprocess_dataset import load_dataset, reduce_dataset_by_merges
from src.data.feature_builder import build_features
from src.models.wsrc import wsrc_predict, compute_weights
from src.metrics.evaluation import compute_accuracy, compute_f1

# Ignore convergence warnings from Lasso
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    # Absolute paths (to avoid path errors)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data_path = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    results_path = os.path.join(BASE_DIR, "results", "wsrc_results.csv")

    # Parameters (configurable)
    weight_method = "inverse_distance"  # "similarity", "inverse_distance", "class"
    top_k = 2                # Only relevant for class-based weights
    alpha = 0.01             # Lasso regularization
    max_merges = 20          # Reduce dataset size for faster testing
    n_splits = 5             # GroupKFold splits

    # Load dataset
    df = load_dataset(data_path)
    
    ###########################################################################
    # Reduce dataset size for faster execution
    df = reduce_dataset_by_merges(df, max_merges=20)
    ###########################################################################
    
    X, y, merge_ids, _ = build_features(df)

    # GroupKFold cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []
    f1_scores = []

    # weights = None  # np.ones(len(y))        

    for train_idx, test_idx in gkf.split(X, y, groups=merge_ids):
        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Normalization (important for WSRC)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_pred = []
        for i, x_test_sample in enumerate(X_test):        
            # Compute dynamic weights for each test sample
            weights = compute_weights(X_train, y_train, x_test_sample, method=weight_method, top_k=top_k)
            # WSRC prediction for this sample
            pred = wsrc_predict(X_train, y_train, x_test_sample.reshape(1, -1), weights=weights, alpha=alpha)
            y_pred.append(pred[0])

        # Evaluation
        accuracies.append(compute_accuracy(y_test, np.array(y_pred)))
        f1_scores.append(compute_f1(y_test, np.array(y_pred)))

    # Prepare results including parameters
    params = {
        "model": "WSRC",
        "weight_method": weight_method,
        "top_k": top_k,
        "alpha": alpha,
        "max_merges": max_merges,
        "n_splits": n_splits,
        "accuracy": sum(accuracies) / len(accuracies),
        "f1_score": sum(f1_scores) / len(f1_scores)
    }

    # Append results to CSV
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        results = pd.concat([existing, pd.DataFrame([params])], ignore_index=True)
    else:
        results = pd.DataFrame([params])

    results.to_csv(results_path, index=False)
    print(f"Results saved in {results_path}")



if __name__ == "__main__":
    main()