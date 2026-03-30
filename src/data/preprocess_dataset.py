import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np


def load_dataset(input_path):
    """
    Load the dataset from a CSV file.

    Args:
        input_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    return df

def reduce_dataset_by_merges(df, max_merges=100, random_state=42):
    """
    Reduce the dataset by selecting a subset of merge_ids (groups), ensuring that all samples belonging to the same merge_id are kept.
    This is useful for debugging slow models like SRC.

    Args:
        df (pd.DataFrame): Original dataset
        max_merges (int): Number of merge_id groups to keep
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Reduced dataset
    """

    print(f"\n[INFO] Reducing dataset to {max_merges} merge groups...")

    # Get unique merge_ids
    unique_merges = df["merge_id"].unique()

    # Sample merge_ids
    np.random.seed(random_state)
    selected_merges = np.random.choice(
        unique_merges,
        size=min(max_merges, len(unique_merges)),
        replace=False
    )

    # Filter dataset
    df_reduced = df[df["merge_id"].isin(selected_merges)]

    print(f"[INFO] Original samples: {len(df)}")
    print(f"[INFO] Reduced samples: {len(df_reduced)}")
    print(f"[INFO] Unique merges: {len(selected_merges)}")

    return df_reduced


def merge_labels(df):
    """
    Merge semicanonical conflict resolution labels into a single category.
    This reduces the number of classes by grouping specific variants into 'CHUNK_SEMICANONICAL_OTHERS'.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with merged labels.
    """
    
    print("Merging semicanonical labels...")

    # Define the set of values to be merged
    merge_set = [
        "CHUNK_SEMICANONICAL_OURSBASETHEIRS",
        "CHUNK_SEMICANONICAL_BASETHEIRS",
        "CHUNK_SEMICANONICAL_EMPTY",
        "CHUNK_SEMICANONICAL_OURSBASE"
    ]

    before = df["conflictResolutionResult"].nunique()

    # Replace the values in the "conflictResolutionResult" column with "CHUNK_SEMICANONICAL_OTHERS"
    df["conflictResolutionResult"] = df["conflictResolutionResult"].replace(
        merge_set, "CHUNK_SEMICANONICAL_OTHERS"
    )

    after = df["conflictResolutionResult"].nunique()

    print(f"Labels before: {before}, after: {after}")
    return df


def filter_projects(df, min_samples=1000):
    """
    Filter out projects with fewer than a minimum number of samples.

    Args:
        df (pd.DataFrame): Input dataset.
        min_samples (int, optional): Minimum number of samples required per project. Default is 1000.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    
    print(f"Filtering projects with less than {min_samples} samples...")

    before = df["project_name"].nunique()

    counts = df["project_name"].value_counts()
    valid_projects = counts[counts >= min_samples].index

    df = df[df["project_name"].isin(valid_projects)]

    after = df["project_name"].nunique()

    print(f"Projects before: {before}, after: {after}")
    print(f"New shape: {df.shape}")

    return df


def encode_labels(df):
    """
    Encode the target labels into numeric values. Adds a new column 'label_encoded' to the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        tuple:
            - pd.DataFrame: Dataset with encoded labels.
            - LabelEncoder: Fitted label encoder.
    """
    
    print("Encoding labels...")

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["conflictResolutionResult"])

    print("Classes:", list(le.classes_))

    return df, le


def save_dataset(df, output_path):
    """
    Save the preprocessed dataset to a CSV file.

    Args:
        df (pd.DataFrame): Dataset to save.
        output_path (str): Output file path.

    Returns:
        None
    """
    
    print(f"Saving preprocessed dataset to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Dataset saved successfully.")


def main():
    """
    Main preprocessing pipeline.

    Executes the full preprocessing workflow:
    - Load dataset
    - Merge labels
    - Filter projects
    - Encode labels
    - Save processed dataset
    """
    
    input_path = "../../data/dataset_chunks_RQ1(in).csv"
    output_path = "../../data/dataset_preprocessed.csv"

    df = load_dataset(input_path)

    df = merge_labels(df)
    df = filter_projects(df)
    df, le = encode_labels(df)

    save_dataset(df, output_path)


if __name__ == "__main__":
    main()