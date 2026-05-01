#main_src.py
import os
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

from src.data.feature_builder import build_features
from src.data.preprocess_dataset import load_dataset, reduce_dataset_by_merges
from src.models.src import src_predict
from src.metrics.evaluation import compute_accuracy, compute_f1

# Ignore convergence warnings from Lasso
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    # Absolute paths (to avoid path errors)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data_path = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    results_path = os.path.join(BASE_DIR, "results", "src_results.csv")

    # Load dataset
    df = load_dataset(data_path)
    
    ##################################################################################################################################################3
    # Reduce dataset size for faster execution
    df = reduce_dataset_by_merges(df, max_merges=20)
    ###################################################################################################################################################3
    
    X, y, merge_ids, _ = build_features(df)

    gkf = GroupKFold(n_splits=5)
    accuracies = []
    f1_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups=merge_ids):
        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Normalization (SRC needs it)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # SRC prediction
        y_pred = src_predict(X_train, y_train, X_test, alpha=0.01)

        # Evaluation
        accuracies.append(compute_accuracy(y_test, y_pred))
        f1_scores.append(compute_f1(y_test, y_pred))

    # Save results
    results = pd.DataFrame([{
        "model": "SRC",
        "accuracy": sum(accuracies) / len(accuracies),
        "f1_score": sum(f1_scores) / len(f1_scores)
    }])

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)

    print(f"Results saved in {results_path}")


if __name__ == "__main__":
    main()