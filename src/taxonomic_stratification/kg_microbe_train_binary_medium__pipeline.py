#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import random
import joblib
import json  # for loading config files

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv, EFstrType

from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                       cross_val_score)
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                             balanced_accuracy_score, make_scorer, precision_score, 
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import (CalibratedClassifierCV, calibration_curve, 
                                 IsotonicRegression)
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.utils.class_weight import compute_class_weight

import shap
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from kg_microbe_pipeline_utils import (
    save_random_seeds,
    load_data,
    process_data_pairs,
    add_closure,
    taxa_media_groups,
    EC_RHEA_annotations,    
    remove_singleton_row_and_col,
    remove_identical_pattern,
    remove_one_edit_distance_features,
    name_columns,
    check_and_split_data,
    perform_permutation_knockoff_analysis,
    perform_cross_validation,
    prob_calibration,
    final_shap_analysis,
    remove_low_variance_features,
    train_final_model,
    process_predictions,
    taxonomy_annotations,
    eggnog_annotations,
    traithop_annotations,
    add_closure_to_table
)

def run_model_training(
    mediumid,
    cv_folds=0,
    n_samples=0,
    gpu=False,
    data_choice="traits",
    config_path=None,
    closure=True,
    remove_one_edit=False,
    taxonomic_stratify=False,
    taxonomic_level="family"
):
    """
    Main function to run model training.

    :param mediumid: The medium ID for model training.
    :param cv_folds: Number of cross-validation folds (0 means no CV).
    :param n_samples: Number of random seeds for permutation knockoff (0 means no permutation).
    :param gpu: Boolean, whether to use GPU-accelerated calls.
    :param data_choice: Comma-separated string specifying which data sets to include
                        (possible tokens: 'traits', 'ec_rhea', 'taxonomy').
    :param config_path: Path to a JSON config file for data paths.
    :param closure: Boolean, whether to perform semantic closure operation on features.
    :param remove_one_edit: Boolean, whether to remove features that are one binary edit away,
                           keeping the one with most non-zero values.
    :param taxonomic_stratify: Boolean, whether to use taxonomic stratification in data splitting.
    :param taxonomic_level: String, taxonomic level to use for stratification (e.g., 'family', 'genus', 'species').
    """
    # Check config is provided
    if not config_path:
        raise ValueError("A configuration file must be provided via --config.")

    # Load config
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    edges_path = config["data_paths"]["edges"]
    nodes_path = config["data_paths"]["nodes"]
    ec_path = config["annotation_paths"]["EC"]
    rhea_path = config["annotation_paths"]["RHEA"]
    taxonomy_path = config["annotation_paths"]["taxonomy"]
    eggnog_path = config["annotation_paths"]["eggnog"]
    traithop_path = config["annotation_paths"]["traithop"]

    # Determine if we actually want CV or Perm
    do_cv = (cv_folds > 0)
    do_perm = (n_samples > 0)

    # Parse data choice
    data_list = [item.strip().lower() for item in data_choice.split(",")]

    print("----------- Command-Line Settings -----------")
    print(f"MEDIUMID:           {mediumid}")
    print(f"DATA_CHOICE (raw):  {data_choice}")
    print(f"DATA_LIST (parsed): {data_list}")
    print(f"CV_FOLDS:           {cv_folds}  (CV enabled: {do_cv})")
    print(f"N_SAMPLES:          {n_samples} (Permutation enabled: {do_perm})")
    print(f"GPU:                {gpu}")
    print(f"CLOSURE:            {closure}")
    print(f"REMOVE_ONE_EDIT:    {remove_one_edit}")
    print(f"TAXONOMIC_STRATIFY: {taxonomic_stratify}")
    print(f"TAXONOMIC_LEVEL:    {taxonomic_level}")
    print(f"CONFIG PATH:        {config_path}")
    print("---------------------------------------------")

    current_datetime = datetime.datetime.now()
    formatted_date = current_datetime.strftime("%Y-%m-%d_%H_%M_%S")
    print("Current date/time:", formatted_date)

    # Hard-coded parameters
    RANDOM_SEED = 12
    print("RANDOM_SEED:", RANDOM_SEED)

    # Set random seeds for all libraries that might be used
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set CatBoost random seed via environment variable
    import os
    os.environ['CATBOOST_RANDOM_SEED'] = str(RANDOM_SEED)
    
    # Set other parameters
    cutoff = 0.4
    ITERATIONS = 100000
    ITERATIONS_FINAL = 100000
    SHAP_PERM_PERCENTILE_THRESHOLD = 0.5
    SIGNIFICANCE_LEVEL = 0.05
    TOP_N_FEAT_SHAP = 40

    # Create random seeds for permutations
    RANDOM_SEEDS = [random.randint(0, 10000) for _ in range(n_samples)]
    print("\nRANDOM_SEEDS:", RANDOM_SEEDS)

    # Construct modellabel
    data_label = "_".join(data_list)
    cv_label = f"cv{cv_folds}"
    perm_label = f"perm{n_samples}"
    closure_label = "closure" if closure else ""
    one_edit_label = "oneedit" if remove_one_edit else ""
    taxonomic_label = f"taxstrat_{taxonomic_level}" if taxonomic_stratify else ""
    gpu_label = "gpu" if gpu else ""
    
    modellabel = f"{mediumid}__{data_label}__{cv_label}__{perm_label}__{closure_label}__{one_edit_label}__{taxonomic_label}__{gpu_label}"

    save_random_seeds(RANDOM_SEEDS, modellabel)

    # ------------------------------------------------------
    # Load base data from config
    # ------------------------------------------------------
    data_edges, data_nodes = load_data(edges_path, nodes_path)

    data_pairs_clean, data_pairs_rest = process_data_pairs(
        data=data_edges,
        modellabel=modellabel,
        output_dir="."
    )

    data_pairs_features = data_pairs_rest
    if closure:
        data_pairs_features = add_closure(
            data=data_edges,
            data_pairs_rest=data_pairs_rest,
            modellabel=modellabel,
            output_dir="."
        )

    data_df = data_pairs_features.pivot_table(
        index='subject',
        columns='object',
        values='Value',
        aggfunc='sum',
        fill_value=0
    )

    # Taxa-media grouping
    data_pairs_clean, data_df = taxa_media_groups(
        data_pairs_clean=data_pairs_clean,
        data_df=data_df,
        mediumid=mediumid,
        modellabel=modellabel,
        output_dir="."
    )

    data_df_orig = data_df.copy(deep=True)
    data_df = data_df[data_df['medium'].notna()]

    index_series = pd.Series(data_df.index.values)
    index_series.to_csv(
        f"data_df__taxa_to_media__NCBITaxon__{modellabel}.tsv",
        sep='\t', index=False, header=False
    )

    total_sum_numeric = data_df.select_dtypes(include=['number']).sum().sum()
    print("\nSummation of numeric columns:", total_sum_numeric)
    print("Data shape (before cleaning):", data_df.shape)

    data_df_clean = data_df.copy()

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Make a deep copy of the clean data to use as the original/base data 
    # before adding any annotations
    # ----------------------------------------------------------------------
    data_df_clean_orig = data_df_clean.copy(deep=True)
    
    # Annotation dataframes to track what was added and for the final prediction processing
    EC_annot_df = None
    RHEA_annot_df = None
    tax_annot_df = None
    egg_annot_df = None
    traithop_annot_df = None
    
    # ----------------------------------------------------------------------
    # EC/RHEA annotations if "ec_rhea" in data_list
    # ----------------------------------------------------------------------
    if "ec_rhea" in data_list:
        print("\nAdding EC/RHEA annotations to the data ...")
        data_df_clean, _, EC_annot_df, RHEA_annot_df = EC_RHEA_annotations(
            data_df_clean,
            ec_file_path=ec_path,
            rhea_file_path=rhea_path,
            target_column='medium',
            remove_original=False  # Keep original features
        )
        prefixes = ["EC:", "RHEA:"]
        data_df_clean = add_closure_to_table(data_df_clean, data_edges, prefixes)
    
    # ----------------------------------------------------------------------
    # Taxonomy annotations if "taxonomy" in data_list
    # ----------------------------------------------------------------------
    if "taxonomy" in data_list:
        print("\nAdding Taxonomy annotations to the data ...")
        data_df_clean, _, tax_annot_df = taxonomy_annotations(
            data_df_clean=data_df_clean,
            taxonomy_file_path=taxonomy_path,
            target_column="medium",
            remove_original=False  # Keep original features
        )

    # ----------------------------------------------------------------------
    # EggNOG annotations if "eggnog" in data_list
    # ----------------------------------------------------------------------
    if "eggnog" in data_list:
        print("\nAdding eggnog annotations to the data ...")
        data_df_clean, _, egg_annot_df = eggnog_annotations(
            data_df_clean=data_df_clean,
            eggnog_file_path=eggnog_path,
            target_column="medium",
            remove_original=False,  # Keep original features
            chunk_size=1000,  # Process 1000 columns at a time
            variance_threshold=0.0,  # No variance filtering - keep all features if possible
            max_features=20000  # Limit to top 20k features by variance if memory issues persist
        )
    
    # ----------------------------------------------------------------------
    # TraitHop annotations if "traithop" in data_list
    # ----------------------------------------------------------------------
    if "traithop" in data_list:
        print("\nAdding traithop annotations to the data ...")
        data_df_clean, _, traithop_annot_df = traithop_annotations(
            data_df_clean=data_df_clean,
            traithop_file_path=traithop_path,
            target_column="medium",
            remove_original=False  # Keep original features
        )
    
    # ------------------------------------------------------
    # Remove singletons and identical patterns
    # ------------------------------------------------------
    data_df_clean = remove_singleton_row_and_col(
        data_df_clean=data_df_clean,
        exclude_cols=['medium'],
        verbose=True
    )

    data_df_clean = remove_identical_pattern(
        data_df_clean=data_df_clean,
        modellabel=modellabel,
        output_dir=".",
        verbose=True
    )

    # ------------------------------------------------------
    # Remove one-edit distance features (if enabled)
    # ------------------------------------------------------
    if remove_one_edit:
        data_df_clean = remove_one_edit_distance_features(
            data_df_clean=data_df_clean,
            modellabel=modellabel,
            output_dir=".",
            verbose=True
        )

    file_path = f"taxa_to_media__{modellabel}_data_df_clean.tsv.gz"
    data_df_clean.to_csv(file_path, sep='\t', index=True, header=True, compression='gzip')

    # ------------------------------------------------------
    # Split into features (X) and target (y)
    # ------------------------------------------------------
    X = data_df_clean.drop('medium', axis=1)
    y = data_df_clean['medium']

    X_train_full, y_train_full_binary, X_test, y_test, y_test_binary, \
        X_val, y_val, y_val_binary = check_and_split_data(
            X=X,
            y=y,
            mediumid=mediumid,
            modellabel=modellabel,
            output_dir=".",
            random_seed=12,
            verbose=False,
            taxonomic_stratify=taxonomic_stratify,
            data_nodes=data_nodes,
            taxonomic_level=taxonomic_level,
            data_edges=data_edges
        )

    # ------------------------------------------------------
    # Permutation Knockoff Analysis
    # ------------------------------------------------------
    if do_perm:
        print("\nRunning permutation knockoff analysis ...")
        (selected_feature_names,
         metrics_list,
         p_values_df) = perform_permutation_knockoff_analysis(
            X_train_full=X_train_full,
            y_train_full_binary=y_train_full_binary,
            X_val=X_val, 
            y_val_binary=y_val_binary,
            RANDOM_SEEDS=RANDOM_SEEDS,
            CV_FOLDS=cv_folds,
            ITERATIONS=ITERATIONS,
            modellabel=modellabel,
            SHAP_PERM_PERCENTILE_THRESHOLD=0.5,
            SIGNIFICANCE_LEVEL=1,
            output_dir=".",
            GPU=gpu
        )
    else:
        print("\nSkipping permutation knockoff analysis ...")
        selected_feature_names = list(X_train_full.columns.values)

    # ------------------------------------------------------
    # Final Model Training
    # ------------------------------------------------------
    print("\nTraining final model ...")
    #test_metrics, cv_metrics, model, X_train_full_selected, val_data_final 
    test_metrics, cv_metrics, model, X_train_full_selected = train_final_model(
        X_train_full = X_train_full,
        y_train_full_binary = y_train_full_binary,
        X_val = X_val,
        y_val = y_val,
        y_val_binary = y_val_binary,
        X_test = X_test,
        y_test_binary = y_test_binary,
        selected_feature_names = selected_feature_names,
        modellabel = modellabel,
        formatted_date=current_datetime.strftime("%Y-%m-%d_%H_%M_%S"),
        iterations_final= ITERATIONS_FINAL,
        random_seed = 12,
        cv_folds = cv_folds,
        output_dir=".",
        GPU=gpu
    )

    # ------------------------------------------------------
    # Probability Calibration
    # ------------------------------------------------------
    print("\nPerforming probability calibration ...")
    results, iso_reg = prob_calibration(
        model=model,
        X_val_final=X_val,
        y_val_binary=y_val_binary,
        X_test=X_test,
        y_test=y_test,
        y_test_binary=y_test_binary,
        selected_feature_names=selected_feature_names,
        mediumid=mediumid,
        modellabel=modellabel,
        RANDOM_SEED=12,
        output_dir=".",
        verbose=True
    )

    calibrated_metrics = results['calibrated_metrics']
    uncalibrated_metrics = results['uncalibrated_metrics']
    calibrated_test_pred_proba = results['calibrated_test_pred_proba']
    uncalibrated_test_pred_proba = results['uncalibrated_test_pred_proba']

    print("\nCalibrated Test Probabilities:")
    print(calibrated_test_pred_proba)
    print("\nUncalibrated Test Probabilities:")
    print(uncalibrated_test_pred_proba)

    # ------------------------------------------------------
    # Final SHAP Analysis
    # ------------------------------------------------------
    print("\nPerforming SHAP analysis ...")
    shap_summary_df, top_features_df = final_shap_analysis(
        model=model,
        X_train_final=X_train_full_selected,
        y_train_final=y_train_full_binary,
        selected_feature_names=selected_feature_names,
        mediumid=mediumid,
        modellabel=modellabel,
        output_dir=".",
        topn=40,
        verbose=True
    )

    # ------------------------------------------------------
    # Process Final Predictions
    # ------------------------------------------------------
    print("\nProcessing final predictions ...")
    if "ec_rhea" not in data_list:
        EC_annot_df = None
        RHEA_annot_df = None
    if "taxonomy" not in data_list:
        tax_annot_df = None
    if "eggnog" not in data_list:
        egg_annot_df = None
    if "traithop" not in data_list:
        traithop_annot_df = None

    # Check if process_predictions accepts traithop_annot_df parameter
    import inspect
    process_pred_params = inspect.signature(process_predictions).parameters
    
    if 'traithop_annot_df' in process_pred_params:
        # If it does accept the parameter, include it
        merged_df = process_predictions(
            data_df_orig=data_df_orig,
            data_df_clean=data_df_clean,
            data_nodes=data_nodes,
            model=model,
            iso_reg=iso_reg,
            mediumid=mediumid,
            modellabel=modellabel,
            cutoff=0.4,
            EC_annot_df=EC_annot_df,
            RHEA_annot_df=RHEA_annot_df,
            tax_annot_df=tax_annot_df,
            egg_annot_df=egg_annot_df,
            traithop_annot_df=traithop_annot_df,
            formatted_date=current_datetime.strftime("%Y-%m-%d_%H_%M_%S"),
            y=y
        )
    else:
        # If it doesn't accept the parameter, exclude it
        merged_df = process_predictions(
            data_df_orig=data_df_orig,
            data_df_clean=data_df_clean,
            data_nodes=data_nodes,
            model=model,
            iso_reg=iso_reg,
            mediumid=mediumid,
            modellabel=modellabel,
            cutoff=0.4,
            EC_annot_df=EC_annot_df,
            RHEA_annot_df=RHEA_annot_df,
            tax_annot_df=tax_annot_df,
            egg_annot_df=egg_annot_df,
            formatted_date=current_datetime.strftime("%Y-%m-%d_%H_%M_%S"),
            y=y
        )


def str2bool(v: str) -> bool:
    """
    Convert a string to a boolean.
    Interprets 'false', '0', 'no' as False. Everything else -> True.
    """
    return str(v).lower() not in ['false', '0', 'no']

def parse_arguments():
    """
    Parse command-line arguments.

    Examples:
      # No CV or permutation (cv_folds=0, n_samples=0)
      python script.py HPC_Something --cv_folds=0 --n_samples=0 --config=config.json

      # CV with 5 folds, no permutation
      python script.py HPC_Something --cv_folds=5 --n_samples=0 --config=config.json

      # Permutation with 10 seeds, no CV
      python script.py HPC_Something --cv_folds=0 --n_samples=10 --config=config.json

      # Both CV and permutation
      python script.py HPC_Something --cv_folds=3 --n_samples=20 --config=config.json

      # Semantic closure disabled
      python script.py HPC_Something --closure=false --config=config.json
    """
    parser = argparse.ArgumentParser(description='Run model training with a specified medium ID.')
    parser.add_argument('mediumid', type=str, help='The medium ID for model training.')

    # CV is inferred by cv_folds>0
    parser.add_argument('--cv_folds', type=int, default=0,
                        help='Number of cross-validation folds (0 means no CV).')

    # Perm analysis is inferred by n_samples>0
    parser.add_argument('--n_samples', type=int, default=0,
                        help='Number of samples (random seeds) for permutation knockoff (0 means no permutation).')

    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training/permutation knockoff (default: CPU).')

    parser.add_argument('--data', type=str, default='traits',
                        help=("Comma-separated data sources to include, e.g. "
                              "'traits', 'ec_rhea', 'taxonomy', 'eggnog', 'traithop', or any combination: "
                              "'traits,ec_rhea,taxonomy'."))

    parser.add_argument('--config', type=str, required=True,
                        help="Path to a JSON config file for data paths.")

    # New --closure argument, defaults to True
    parser.add_argument('--closure', default=True, type=str2bool,
                        help=("Whether to perform a semantic closure operation on the features. "
                              "Defaults to 'true'. Accepts 'false', '0', or 'no' to disable."))

    # New --remove_one_edit argument, defaults to False
    parser.add_argument('--remove_one_edit', action='store_true',
                        help=("Remove features that are one binary edit away from other features, "
                              "keeping the one with the most non-zero values. This reduces data size "
                              "by filtering out very similar binary features."))

    # New --taxonomic-stratify argument, defaults to False
    parser.add_argument('--taxonomic-stratify', action='store_true',
                        help=("Use taxonomic stratification in data splitting to prevent data leakage "
                              "by ensuring that closely related taxa are not split across training "
                              "and test sets."))

    # New --taxonomic-level argument, defaults to 'family'
    parser.add_argument('--taxonomic-level', type=str, default='family',
                        choices=['species', 'genus', 'family', 'order', 'class', 'phylum'],
                        help=("Taxonomic level to use for stratification. Higher levels (e.g., family) "
                              "provide broader groupings, while lower levels (e.g., species) provide "
                              "more specific groupings. Default: 'family'."))

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    run_model_training(
        mediumid=args.mediumid,
        cv_folds=args.cv_folds,
        n_samples=args.n_samples,
        gpu=args.gpu,
        data_choice=args.data,
        config_path=args.config,
        closure=args.closure,
        remove_one_edit=args.remove_one_edit,
        taxonomic_stratify=args.taxonomic_stratify,
        taxonomic_level=args.taxonomic_level
    )
