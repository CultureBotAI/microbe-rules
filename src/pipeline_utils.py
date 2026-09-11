"""
Shared utilities for the microbe-rules pipeline.

This module contains common functions used across the pipeline scripts
(01_prepare_data_binary.py, 02_compute_compare_models.py, 03_compute_feature_importance_agreement.py).
"""

import os
from typing import Tuple, Dict, List, Optional

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Constants
DEFAULT_RANDOM_SEED = 12
DEFAULT_CB_SEED = 9759
DEFAULT_TEST_SIZE = 0.3
DEFAULT_VAL_RATIO = 0.33  # of the test set


def load_preprocessed_data(
    mediumid: int,
    data_path: str = "data"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load preprocessed data from the 01_prepare_data_binary.py output.

    Parameters:
    - mediumid: The medium ID (e.g., 65 or 514)
    - data_path: Path to the data directory

    Returns:
    - X: Feature matrix (DataFrame)
    - y: Target labels (Series)
    """
    modellabel = "binary_permute_" + str(mediumid)
    file_path = os.path.join(
        data_path,
        f'taxa_to_media__{modellabel}_data_df_clean.tsv.gz'
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Preprocessed data file not found: {file_path}\n"
            f"Please run 01_prepare_data_binary.py first for medium {mediumid}"
        )

    data_df_clean = pd.read_csv(file_path, sep='\t')
    print(f"Dataset has {len(data_df_clean.index)} rows and {len(data_df_clean.columns)} columns")

    # Splitting the data into features and target labels
    X = data_df_clean.drop('medium', axis=1)
    X = X.drop('subject', axis=1)
    y = data_df_clean['medium']

    return X, y


def create_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    mediumid: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_ratio: float = DEFAULT_VAL_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create train/val/test splits with proper stratification.

    Parameters:
    - X: Feature matrix
    - y: Target labels
    - mediumid: Medium ID to convert labels
    - random_seed: Random seed for reproducibility
    - test_size: Proportion of data for test+val set
    - val_ratio: Proportion of test+val set to use for validation

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Convert labels to binary (medium:X vs other)
    y_binary = y.replace(['other', f'medium:{mediumid}'], [0, 1])

    # First split: separate training from (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_binary,
        test_size=test_size,
        stratify=y_binary,
        random_state=random_seed
    )

    # Second split: separate validation from test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def convert_labels_to_catboost_format(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> Tuple[List[int], List[int], List[int], Dict[int, int]]:
    """
    Convert labels to CatBoost category indexes.

    Parameters:
    - y_train, y_val, y_test: Label series

    Returns:
    - y_train_cat, y_val_cat, y_test_cat: Converted labels as lists
    - label_dict: Mapping from original labels to category indexes
    """
    # Create mapping from labels to category indexes
    label_dict = {x: i for i, x in enumerate(sorted(set(y_train) | set(y_val) | set(y_test)))}

    y_train_cat = [label_dict[x] for x in y_train]
    y_val_cat = [label_dict[x] for x in y_val]
    y_test_cat = [label_dict[x] for x in y_test]

    return y_train_cat, y_val_cat, y_test_cat, label_dict


def train_catboost_model(
    X_train: pd.DataFrame,
    y_train: List[int],
    X_val: pd.DataFrame,
    y_val: List[int],
    iterations: int = 100,
    random_seed: int = DEFAULT_CB_SEED,
    verbose: int = 100
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier.

    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - iterations: Number of boosting iterations
    - random_seed: Random seed for CatBoost
    - verbose: Verbosity level

    Returns:
    - Trained CatBoostClassifier model
    """
    train_data = Pool(data=X_train, label=y_train, cat_features=[0])
    val_data = Pool(data=X_val, label=y_val, cat_features=[0])

    model = CatBoostClassifier(
        random_seed=random_seed,
        iterations=iterations,
        loss_function="MultiClass",
        verbose=verbose
    )

    model.fit(train_data, eval_set=val_data)

    return model


def evaluate_model(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: List[int],
    model_name: str = "Model"
) -> Dict[str, any]:
    """
    Evaluate a trained model on test data.

    Parameters:
    - model: Trained CatBoost model
    - X_test: Test features
    - y_test: Test labels
    - model_name: Name for display purposes

    Returns:
    - Dictionary with predictions, probabilities, accuracy, and classification report
    """
    test_data = Pool(data=X_test, label=y_test, cat_features=[0])

    y_pred = model.predict(test_data)
    y_pred_proba = model.predict_proba(test_data)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{class_report}")

    return {
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'classification_report': class_report
    }


def evaluate_model_on_train(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: List[int],
    model_name: str = "Model"
) -> float:
    """
    Evaluate a trained model on training data (to check overfitting).

    Parameters:
    - model: Trained CatBoost model
    - X_train: Training features
    - y_train: Training labels
    - model_name: Name for display purposes

    Returns:
    - Training accuracy
    """
    train_data = Pool(data=X_train, label=y_train, cat_features=[0])
    y_pred_train = model.predict(train_data)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    print(f"{model_name} Training Accuracy: {accuracy_train:.4f}")

    return accuracy_train


def save_confusion_matrix(
    y_test: List[int],
    y_pred: np.ndarray,
    output_path: str,
    figsize: Tuple[int, int] = (20, 20)
) -> None:
    """
    Create and save a confusion matrix heatmap.

    Parameters:
    - y_test: True labels
    - y_pred: Predicted labels
    - output_path: Path to save the PDF
    - figsize: Figure size (width, height)
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(output_path, format='pdf')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
