# Predicting Merge Conflict Resolutions: WSRC vs Random Forest

## Overview

Merge conflicts are a common challenge in Git-based software development. Resolving them manually can slow down development and reduce productivity.

This project explores the use of **machine learning techniques** to automatically predict merge conflict resolutions. Specifically, it compares:

* Random Forest (baseline model)
* Sparse Representation Classifier (SRC)
* Weighted Sparse Representation Classifier (WSRC)

The task is formulated as a **multi-class classification problem**, where each class represents a resolution pattern.

---

## Objectives

* Preprocess and analyze a real-world dataset of merge conflicts
* Implement and evaluate different classification models
* Compare traditional ML methods with sparse representation approaches
* Determine whether WSRC can outperform Random Forest

---

##  Project Structure

```
project/
│
├── data/                   # Dataset files
│
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # ML models (RF, SRC, WSRC)
│   ├── evaluation/        # Metrics and evaluation
│   ├── utils/             # Utilities (cross-validation, etc.)
│   └── main.py            # Main pipeline
│
├── experiments/             # Experiments and prototyping
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset contains merge conflict chunks extracted from open-source projects.

### Key preprocessing steps:

* Merge semicanonical labels into a single class
* Filter projects with fewer than 1000 samples
* Remove irrelevant or leakage-prone features
* Encode labels using numerical values

### Data Leakage Prevention

To ensure a fair evaluation:

* Samples sharing the same `merge_id` are kept in the same fold
* Implemented using **GroupKFold**

---

## Models

### 1. Random Forest

* Baseline model
* Implemented using `scikit-learn`
* Robust and efficient

### 2. Sparse Representation Classifier (SRC)

* Based on L1 minimization (Lasso)
* Classifies samples via reconstruction error
* Requires feature normalization

### 3. Weighted SRC (WSRC)

* Extension of SRC with weighted contributions
* Expected to improve classification performance *(work in progress)*

---

## Evaluation Strategy

* Cross-validation: **GroupKFold (k=5)**
* Grouping variable: `merge_id`
* Metrics:

  * Accuracy
  * F1-score (weighted)

---

## Running the Project

Run the main pipeline:

```bash
python src/main.py
```

---

## Results

Results will include:

* Average accuracy across folds
* Average F1-score
* Comparison between RF, SRC, and WSRC


---

## Key Concepts

* Merge conflict resolution patterns
* Classification in high-dimensional data
* Sparse representation and L1 optimization
* Data leakage and grouped cross-validation

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn

---

## References

* ACM Paper on merge conflict resolution prediction
* Empirical studies on software merge conflicts

---

## Author

Asier Yániz
Seminar of Software Engineering

---
