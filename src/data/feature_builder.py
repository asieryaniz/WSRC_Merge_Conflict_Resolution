# feature_builder.py
import pandas as pd


# Columns that should not be used as features for model training
COLS_TO_REMOVE = [
    'project_id', 'project_name', 'remote_url',
    'merge_id', 'merge_time',
    'file_report_id', 'file_path',
    'chunk_id', 'developersIntersection',
    'conflictResolutionResult',
    'label_encoded'
]


def build_features(df):
    """
    Build the feature matrix, target vector, and grouping variables
    from the preprocessed dataset.

    Args:
        df (pd.DataFrame): Preprocessed dataset containing features, metadata, and encoded labels.

    Returns:
        tuple:
            - pd.DataFrame: Feature matrix (X) containing only numeric features
            - np.ndarray: Target vector (y) with encoded labels
            - pd.Series: Merge IDs used for GroupKFold (to avoid data leakage)
            - pd.Series: Project IDs (useful for project-level analysis)
    """
    
    # Extract encoded labels
    y = df["label_encoded"].values
    
    # Remove non-feature columns and keep only numeric data
    X = df.drop(columns=COLS_TO_REMOVE, errors='ignore')
    X = X.select_dtypes(include=['number'])

    # Grouping variable to prevent data leakage in cross-validation
    merge_ids = df["merge_id"]

    # Additional grouping (useful for analysis per project)
    project_ids = df["project_name"]

    return X, y, merge_ids, project_ids