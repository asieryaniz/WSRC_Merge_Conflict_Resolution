import pandas as pd
import os
from src.data.preprocess_dataset import load_dataset
from src.data.feature_builder import build_features
from src.models.random_forest import train_rf, predict_rf
from src.metrics.evaluation import compute_accuracy, compute_f1
from sklearn.model_selection import GroupKFold

def main():
    
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Build correct path to dataset
    data_path = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")

    # Load the preprocess dataset
    df = load_dataset(data_path)
    X, y, merge_ids, _ = build_features(df)

    gkf = GroupKFold(n_splits=5)
    accuracies, f1_scores = [], []

    for train_idx, test_idx in gkf.split(X, y, groups=merge_ids):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = train_rf(X_train, y_train)
        y_pred = predict_rf(model, X_test)

        # Evaluate
        accuracies.append(compute_accuracy(y_test, y_pred))
        f1_scores.append(compute_f1(y_test, y_pred))
        
    # Save results
    results = pd.DataFrame([{
        "model": "RandomForest",
        "accuracy": sum(accuracies)/len(accuracies),
        "f1_score": sum(f1_scores)/len(f1_scores)
    }])
    
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "rf_results.csv")
    results.to_csv(results_path, index=False)

    print(f"Results saved in {results_path}")


if __name__ == "__main__":
    main()