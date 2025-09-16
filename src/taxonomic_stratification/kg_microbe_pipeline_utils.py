#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import random
import joblib
import os

from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from sklearn.utils.class_weight import compute_class_weight

import shap
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt



def save_random_seeds(
    RANDOM_SEEDS,
    modellabel,
    output_dir=".",
):
    """
    Save random seeds to a TSV file.

    Parameters:
    - RANDOM_SEEDS (list of int): List of random seeds to save.
    - modellabel (str): Label to include in the random seeds file name.
    - output_dir (str): Directory to save the random seeds file. Default is the current directory.

    Returns:
    """

    print("STEP save_random_seeds")

    # Define the path for the random seeds file
    seeds_file_path = os.path.join(output_dir, f'random_seeds_{modellabel}.tsv')
    
    # Write random seeds to the TSV file
    with open(seeds_file_path, 'w') as file:
        for seed in RANDOM_SEEDS:
            file.write(f"{seed}\n")
    print(f"Random seeds saved to {seeds_file_path}")



def load_data(
    edges_path="../../master/kg-microbe/data/merged/20241029/merged-kg_edges.tsv",
    nodes_path="../../master/kg-microbe/data/merged/20241029/merged-kg_nodes.tsv",
    output_dir=".",
):
    """
    Save random seeds to a TSV file and load edges and nodes data from TSV files.

    Parameters:
    - modellabel (str): Label to include in the random seeds file name.
    - edges_path (str): Path to the 'merged-kg_edges.tsv' file.
    - nodes_path (str): Path to the 'merged-kg_nodes.tsv' file.
    - output_dir (str): Directory to save the random seeds file. Default is the current directory.

    Returns:
    - data_edges (pd.DataFrame): DataFrame containing edges data.
    - data_nodes (pd.DataFrame): DataFrame containing nodes data.
    """

    print("STEP load_data")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Read the edges data from the TSV file
    try:
        data_edges = pd.read_csv(edges_path, header=0, sep="\t")
        print(f"Edges data loaded from {edges_path} with shape {data_edges.shape}")
    except FileNotFoundError:
        print(f"Error: The file {edges_path} was not found.")
        data_edges = pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading {edges_path}: {e}")
        data_edges = pd.DataFrame()
    
    # Read the nodes data from the TSV file
    try:
        data_nodes = pd.read_csv(nodes_path, header=0, sep="\t")
        print(f"Nodes data loaded from {nodes_path} with shape {data_nodes.shape}")
    except FileNotFoundError:
        print(f"Error: The file {nodes_path} was not found.")
        data_nodes = pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading {nodes_path}: {e}")
        data_nodes = pd.DataFrame()
    
    return data_edges, data_nodes


def process_data_pairs(
    data,
    modellabel,
    output_dir=".",
):
    """
    Process data pairs by filtering based on specific patterns in 'subject' and 'object' columns,
    save the cleaned data to TSV files, and return the filtered DataFrames.
    
    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'subject', 'predicate', and 'object' columns.
    - modellabel (str): Label to include in the output filenames.
    - output_dir (str, default="."): Directory to save the output TSV files.
    
    Returns:
    - data_pairs_clean (pd.DataFrame): Filtered DataFrame where 'subject' contains 'NCBITaxon:' or 'strain:'
      and 'object' contains 'medium:'.
    - data_pairs_rest (pd.DataFrame): Filtered DataFrame containing various other patterns in 'object'.
    """
    
    print("STEP process_data_pairs")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data_pairs by selecting relevant columns and dropping duplicates
    data_pairs = data[['subject', 'predicate', 'object']].drop_duplicates()
    
    # Filter based on 'subject' containing 'NCBITaxon:' or 'strain:'
    data_pairs_clean = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:', regex=True)]
    # Further filter 'object' containing 'medium:'
    data_pairs_clean = data_pairs_clean[data_pairs_clean['object'].str.contains('medium:', regex=True)]
    # Save to TSV
    clean_file_path = os.path.join(output_dir, f"NCBITaxon_to_medium_{modellabel}.tsv")
    data_pairs_clean.to_csv(clean_file_path, sep="\t", header=True, index=False)
    print(f"Cleaned data saved to {clean_file_path}")
    
    # Filter for chemical-related objects
    data_pairs_chem = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:', regex=True)]
    data_pairs_chem = data_pairs_chem[data_pairs_chem['object'].str.contains('CHEBI:', regex=True)]
    print(f"Chemical pairs shape: {data_pairs_chem.shape}")
    
    # Filter for GO-related objects
    data_pairs_go = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:', regex=True)]
    data_pairs_go = data_pairs_go[data_pairs_go['object'].str.contains('GO:', regex=True)]
    print(f"GO pairs shape: {data_pairs_go.shape}")
    
    # Define patterns for 'object' to include in 'data_pairs_rest'
    object_patterns_rest = [
        'carbon_substrates:', 'pathways:', 'trophic_type:', 'production:', 'CAS-RN:',
        'CHEBI:', 'EC:', 'GO:', 'cell_shape:', 'cell_length:', 'cell_width:',
        'motility:', 'sporulation:', 'pigment:', 'gram_stain:', 'gc:',
        'pH_.*:', 'temp_.*:', 'temperature:', 'salinity:', 'NaCl_.*:',
        'oxygen:', 'pathogen:', 'isolation_source:', 'ENVO:', 'UBERON:', 'PO:'
    ]
    
    # Initialize data_pairs_rest by filtering 'subject'
    data_pairs_rest_all = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:', regex=True)]
    data_pairs_rest = pd.DataFrame()
    
    # Iterate over each pattern and concatenate the filtered DataFrame
    for pattern in object_patterns_rest:
        filtered = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains(pattern, regex=True)]
        data_pairs_rest = pd.concat([data_pairs_rest, filtered], ignore_index=True)
        print(f"process_data_pairs Added {filtered.shape[0]} rows for pattern '{pattern}'")

    # Handle additional patterns after swapping 'subject' and 'object'
    data_pairs_rest_all2 = data_pairs[data_pairs['object'].str.contains('NCBITaxon:|strain:', regex=True)]
    # Swap 'subject' and 'object'
    data_pairs_rest_all2_swapped = data_pairs_rest_all2.copy()
    data_pairs_rest_all2_swapped['subject'], data_pairs_rest_all2_swapped['object'] = \
        data_pairs_rest_all2_swapped['object'], data_pairs_rest_all2_swapped['subject']

    # Open a text file to save the output
    with open(f'process_data_pairs_output_{modellabel}.txt', 'w') as f:
        # Initialize data_pairs_rest by filtering 'subject'
        data_pairs_rest_all = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:', regex=True)]
        data_pairs_rest = pd.DataFrame()
        
        # Iterate over each pattern and concatenate the filtered DataFrame
        for pattern in object_patterns_rest:
            filtered = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains(pattern, regex=True)]
            data_pairs_rest = pd.concat([data_pairs_rest, filtered], ignore_index=True)
            message = f"process_data_pairs Added {filtered.shape[0]} rows for pattern '{pattern}'\n"
            print(message.strip())  # Optional: print to console
            f.write(message)
        
        # Handle additional patterns after swapping 'subject' and 'object'
        data_pairs_rest_all2 = data_pairs[data_pairs['object'].str.contains('NCBITaxon:|strain:', regex=True)]
        # Swap 'subject' and 'object'
        data_pairs_rest_all2_swapped = data_pairs_rest_all2.copy()
        data_pairs_rest_all2_swapped['subject'], data_pairs_rest_all2_swapped['object'] = \
            data_pairs_rest_all2_swapped['object'], data_pairs_rest_all2_swapped['subject']
        
        # Iterate over each pattern for the swapped data
        for pattern in object_patterns_rest:
            filtered_swapped = data_pairs_rest_all2_swapped[
                data_pairs_rest_all2_swapped['object'].str.contains(pattern, regex=True)
            ]
            data_pairs_rest = pd.concat([data_pairs_rest, filtered_swapped], ignore_index=True)
            message = f"process_data_pairs Added {filtered_swapped.shape[0]} rows for swapped pattern '{pattern}'\n"
            print(message.strip())
            f.write(message)


    
    
    
    # Define additional patterns for swapped DataFrame
    object_patterns_rest_swap = [
        'PATO:', 'UBERON:', 'FOODON:', 'CHEBI:', 'ENVO:', 'PO:', 'assay:', 'isolation_source:'
    ]
    
    for pattern in object_patterns_rest_swap:
        filtered = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains(pattern, regex=True)]
        data_pairs_rest = pd.concat([data_pairs_rest, filtered], ignore_index=True)
        print(f"process_data_pairs Added {filtered.shape[0]} rows for swapped pattern '{pattern}'")
    
    # Add a 'Value' column with 1
    data_pairs_rest['Value'] = 1
    print(f"Total number of rows in data_pairs_rest: {len(data_pairs_rest)}")
    print(data_pairs_rest.head())

    return data_pairs_clean, data_pairs_rest



def add_closure(
    data,
    data_pairs_rest,
    modellabel,
    output_dir=".",
):
    """
    Add closure edges to the data_pairs_rest DataFrame based on subclass relationships.
    
    This function performs the following steps:
    1. Filters data_pairs to create data_pairs_clean where 'subject' contains 'NCBITaxon:' or 'strain:'
       and 'object' contains 'medium:'.
    2. Saves data_pairs_clean to a TSV file.
    3. Filters data_pairs_rest based on various patterns in the 'object' column.
    4. Adds closure edges by traversing subclass relationships.
    5. Returns the filtered DataFrames: data_pairs_clean and data_pairs_rest.
    
    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'subject', 'predicate', and 'object' columns.
    - modellabel (str): Label to include in the output filenames.
    - output_dir (str, default="."): Directory to save the output TSV files.
    
    Returns:
    - data_pairs_clean (pd.DataFrame): Filtered DataFrame where 'subject' contains 'NCBITaxon:' or 'strain:'
      and 'object' contains 'medium:'.
    - data_pairs_rest (pd.DataFrame): Filtered DataFrame containing various other patterns in 'object',
      including closure edges.
    """
    
    print("STEP add_closure")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Subject prefixes to filter for closure
    subject_prefixes = ['CHEBI:', 'GO:', 'ENVO:', 'UBERON:', 'EC:', 'PO:', 'PATO:', 'FOODON:']#, 'isolation_source:']
    
    # Extract relevant object terms from data_pairs_rest
    relevant_objects = set()
    for obj in data_pairs_rest['object']:
        if any(obj.startswith(prefix) for prefix in subject_prefixes):
            relevant_objects.add(obj)
    
    # Filter data for biolink:subclass_of predicate and relevant objects
    filtered_data = data[(data['predicate'] == 'biolink:subclass_of') & (data['subject'].isin(relevant_objects))]
    
    # Build a dictionary of child to parent relationships
    subclass_dict = {}
    for _, row in filtered_data.iterrows():
        child = row['subject']
        parent = row['object']
        if child not in subclass_dict:
            subclass_dict[child] = []
        subclass_dict[child].append(parent)
    
    print(f"subclass_dict contains {len(subclass_dict)} child-parent relationships")
    
    # Function to get all parent terms following subclass relationships
    def get_parents(term, subclass_dict):
        parents = []
        current_term = term
        while current_term in subclass_dict:
            parent_terms = subclass_dict[current_term]
            if not parent_terms:
                break
            # Assume there is only one parent per term for simplicity
            parent = parent_terms[0]
            parents.append(parent)
            current_term = parent
        return parents
    
    # Create a new DataFrame for the closure
    data_pairs_rest_closure = data_pairs_rest.copy(deep=True)
    
    # Extend data_pairs_rest_closure with parent subclass edges
    new_edges = []
    for _, row in data_pairs_rest.iterrows():
        subject = row['subject']
        obj = row['object']
        if any(obj.startswith(prefix) for prefix in subject_prefixes):
            if obj in subclass_dict:
                parents = get_parents(obj, subclass_dict)
                for parent in parents:
                    new_edges.append({'subject': subject, 'object': parent})
    
    # Convert new_edges to DataFrame
    new_edges_df = pd.DataFrame(new_edges)
    print(f"Number of new closure edges: {new_edges_df.shape[0]}")
    
    # Concatenate the original and new closure edges DataFrames
    data_pairs_rest_closure = pd.concat([data_pairs_rest_closure, new_edges_df], ignore_index=True)
    
    return data_pairs_rest_closure


import pandas as pd

def add_closure_to_table(
    data_df_clean: pd.DataFrame,
    edges_df: pd.DataFrame,
    prefixes: list,
    subclass_predicate: str = "biolink:subclass_of"
) -> pd.DataFrame:
    """
    Perform a closure-like expansion on data_df_clean by adding new feature columns
    whenever there is a subclass_of chain leading from an existing column to its parents.
    
    The edges_df must have columns: ['subject', 'predicate', 'object'] for building
    child->parent relationships. For each existing column in data_df_clean whose name
    starts with any of the given prefixes, we find all parents (via subclass_of edges),
    then add new columns to data_df_clean indicating presence of those parent terms.
    
    :param data_df_clean: A DataFrame where rows are subjects (index) and columns
                          are features. Values > 0 indicate a relationship/presence.
    :param edges_df: A DataFrame of edges, containing 'subject', 'predicate', 'object'.
    :param prefixes: A list of string prefixes (e.g. ['EC:', 'RHEA:']) to look for
                     among the columns of data_df_clean.
    :param subclass_predicate: The predicate denoting subclass_of relationships in edges_df.
    :return: A modified copy of data_df_clean, with new columns added (if any).
    """
    # --- 1) Build a dictionary of child -> list of parent terms based on subclass_of edges
    # Filter edges_df by the desired subclass predicate
    sub_df = edges_df[edges_df['predicate'] == subclass_predicate].copy()
    
    # Build a dictionary: child -> [parents...]
    subclass_dict = {}
    for _, row in sub_df.iterrows():
        child = row['subject']
        parent = row['object']
        if child not in subclass_dict:
            subclass_dict[child] = []
        subclass_dict[child].append(parent)
    
    # A helper function to recursively get *all* parents
    # or walk up the chain until no more parents
    def get_all_parents(term):
        parents = set()
        stack = [term]
        visited = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in subclass_dict:
                for p in subclass_dict[current]:
                    if p not in visited:
                        parents.add(p)
                        stack.append(p)
        return parents
    
    # --- 2) Identify columns in data_df_clean whose names start with any prefix
    #     For each of these 'child' columns, find all 'parent' terms and add columns
    columns_of_interest = [col for col in data_df_clean.columns
                           if any(col.startswith(pref) for pref in prefixes)]
    
    # We'll collect new columns to fill, mapping parent -> column_name
    # so we can create them as needed
    all_parents_needed = set()
    for col in columns_of_interest:
        # get all parent terms for this column name
        parent_terms = get_all_parents(col)
        all_parents_needed.update(parent_terms)
    
    # Remove any that might already exist in data_df_clean
    all_parents_needed = [pt for pt in all_parents_needed if pt not in data_df_clean.columns]
    
    # Add these new columns to data_df_clean, initialize to 0
    new_cols = {col: 0 for col in all_parents_needed}
    data_df_clean = data_df_clean.assign(**new_cols)

    
    # --- 3) For each row, if data_df_clean[row, child_col] > 0, set row, parent_col = 1
    for child_col in columns_of_interest:
        parent_terms = get_all_parents(child_col)
        # For efficiency: get indices where child_col is > 0
        child_positive_index = data_df_clean.index[data_df_clean[child_col] > 0]
        for parent_col in parent_terms:
            # Mark these rows as 1 in the parent_col
            data_df_clean.loc[child_positive_index, parent_col] = 1
    
    # Optionally, remove columns that ended up all 0 if you want
    # data_df_clean = data_df_clean.loc[:, (data_df_clean != 0).any(axis=0)]
    
    return data_df_clean


def taxa_media_groups(
    data_pairs_clean,
    data_df,
    mediumid,
    modellabel,
    output_dir=".",
):
    """
    Classify taxa based on their association with a specific medium and merge the classification into data_df.
    
    This function performs the following steps:
    1. Copies the input DataFrame `data_pairs_clean`.
    2. Groups the data by 'subject' and aggregates 'object' values into lists.
    3. Classifies each taxon as associated with the specified medium or as 'other'.
    4. Saves the classification results to a TSV file.
    5. Sets 'NCBITaxon' as the index and drops the column for merging.
    6. Merges the classification into the provided `data_df` DataFrame.
    
    Parameters:
    - data_pairs_clean (pd.DataFrame): DataFrame containing 'subject', 'predicate', and 'object' columns.
                                      Should be pre-filtered to include relevant subjects (e.g., containing 'NCBITaxon:' or 'strain:').
    - data_df (pd.DataFrame): DataFrame to merge the classification results into.
                               Must have 'NCBITaxon' as its index for proper merging.
    - mediumid (str or int): Identifier for the medium to classify taxa (e.g., 'Glucose').
    - modellabel (str): Label to include in the output filename for the classification TSV.
    - output_dir (str, default="."): Directory to save the classification TSV file.
    
    Returns:
    - data_pairs_clean (pd.DataFrame): The original `data_pairs_clean` DataFrame (unchanged).
    - data_df (pd.DataFrame): The `data_df` DataFrame merged with the taxa media classification.
    """
    
    print("STEP taxa_media_groups")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Copy the original DataFrame
    data_pairs_clean_filtered = data_pairs_clean.copy(deep=True)

    # STEP 2: Ensure that all NCBITaxon: and strain: subject values are considered
    all_subjects = data_pairs_clean_filtered['subject'].unique()

    # STEP 3: Group by 'subject' and list all 'object' (medium)
    taxon_media_groups = (
        data_pairs_clean_filtered
        .groupby('subject')['object']
        .agg(list)
        .reindex(all_subjects, fill_value=[])
    )

    # STEP 4: Classify taxa based on association with medium:X
    def classify_taxa(media_list):
        """
        Classify a taxon based on its association with a specific medium.

        Parameters:
        - media_list (list): List of media associated with the taxon.

        Returns:
        - str: The medium string if associated, else 'other'.
        """
        medstr = 'medium:' + str(mediumid)
        if medstr in media_list:
            return medstr
        else:
            return 'other'

    # Apply the classification function
    classified_taxa = taxon_media_groups.apply(classify_taxa)

    # STEP 5: Prepare the final DataFrame
    taxmed_final_df = classified_taxa.reset_index()
    taxmed_final_df.columns = ['NCBITaxon', 'medium']

    # STEP 6: Export the classification to a TSV file
    classification_file = f'taxa_media_classification_{modellabel}.tsv'
    classification_file_path = os.path.join(output_dir, classification_file)
    taxmed_final_df.to_csv(classification_file_path, sep="\t", header=True, index=False)
    print(f"Taxa media classification saved to {classification_file_path}")

    # STEP 7: Set 'NCBITaxon' as index and drop the column
    print("taxmed_final_df "+taxmed_final_df.columns)
    taxmed_final_df.set_index('NCBITaxon', inplace=True)
    print("taxmed_final_df "+taxmed_final_df.columns)
    #taxmed_final_df.drop(columns=['NCBITaxon'], inplace=True)

    # STEP 8: Print DataFrame information for verification
    print("Columns in data_df:", data_df.columns)
    print("Columns in taxmed_final_df:", taxmed_final_df.columns)
    print("Rows in data_df:", data_df.index)
    print("Rows in taxmed_final_df:", taxmed_final_df.index)

    # STEP 9: Merge the classification into data_df
    data_df = data_df.merge(
        taxmed_final_df,
        left_index=True,
        right_index=True,
        how='left'
    )

    return data_pairs_clean, data_df


def EC_RHEA_annotations(data_df_clean, ec_file_path, rhea_file_path, target_column, remove_original=False):
    """
    Extend data_df_clean with EC and RHEA annotations.

    Parameters:
        data_df_clean (pd.DataFrame): The original DataFrame to be extended.
        ec_file_path (str): Path to the EC annotations TSV file.
        rhea_file_path (str): Path to the RHEA annotations TSV file.
        target_column (str): The column in data_df_clean that we want to keep for sure.
        remove_original (bool): If True, remove all original columns except target_column 
                                after merging with annotation columns. 
                                If False, keep all original columns plus the new ones.

    Returns:
        data_df_clean (pd.DataFrame): The extended (and possibly pruned) DataFrame.
        data_df_clean_orig (pd.DataFrame): A copy of the original DataFrame, before modifications.
        EC_annot_df (pd.DataFrame): The crosstab DataFrame of EC annotations.
        RHEA_annot_df (pd.DataFrame): The crosstab DataFrame of RHEA annotations.
    """
    
    print("STEP EC_RHEA_annotations")

    # Make a copy of the original DataFrame for safe-keeping
    data_df_clean_orig = data_df_clean.copy(deep=True)

    # 1) Load EC annotations
    EC_annot = pd.read_csv(ec_file_path, sep='\t', index_col=0)
    EC_annot_df = pd.crosstab(EC_annot.index, EC_annot['EC'])
    
    # Handle duplicate column names by summing duplicate columns
    EC_annot_df = EC_annot_df.groupby(EC_annot_df.columns, axis=1).sum()

    # 2) Subset only target_column from data_df_clean
    data_df_clean_sub = data_df_clean[[target_column]]

    # 3) If you need to filter only columns starting with 'EC:', you can do so here
    EC_columns = [col for col in EC_annot_df.columns if col.startswith('EC:')]
    EC_annot_df_sub = EC_annot_df[EC_columns]

    # 4) Join the subset DataFrame with the entire (or filtered) EC crosstab
    #    resulting in (target_column + EC columns)
    data_df_clean_join_EC = data_df_clean_sub.join(EC_annot_df_sub, how='left', lsuffix='_trait', rsuffix='_annot')

    # 5) Load RHEA annotations
    RHEA_annot = pd.read_csv(rhea_file_path, sep='\t', index_col=0)
    RHEA_annot_df = pd.crosstab(RHEA_annot.index, RHEA_annot['RHEA'])
    
    # Handle duplicate column names by summing duplicate columns
    RHEA_annot_df = RHEA_annot_df.groupby(RHEA_annot_df.columns, axis=1).sum()

    # 6) Join RHEA crosstab columns
    #    now data_df_clean_join_EC = target_column + EC + RHEA columns
    data_df_clean_join_EC = data_df_clean_join_EC.join(RHEA_annot_df, how='left', lsuffix='_trait', rsuffix='_annot')

    # 7) Decide how to handle original columns
    if remove_original:
        # Keep only target_column plus annotation columns
        data_df_clean = data_df_clean_join_EC.copy(deep=True)
    else:
        # Merge the new annotation columns back into the original DataFrame
        # while preserving all original columns.
        # Drop target_column from data_df_clean_join_EC to avoid duplication,
        # then join everything else (annotation columns) to the original.
        data_df_clean = data_df_clean_orig.join(
            data_df_clean_join_EC.drop(columns=[target_column], errors='ignore'),
            how='left', lsuffix='_trait', rsuffix='_annot'
        )

    return data_df_clean, data_df_clean_orig, EC_annot_df, RHEA_annot_df


def remove_low_variance_features(X_train_full_selected, feature_variances, threshold=1e-5):
    """
    Remove features with zero or near-zero variance from the dataset.

    Parameters:
    - X_train_full_selected: pd.DataFrame, the input DataFrame from which to remove low variance features.
    - feature_variances: pd.Series, the variances of the features.
    - threshold: float, the variance threshold below which features will be removed. Default is 1e-5.

    Returns:
    - X_train_full_selected: pd.DataFrame, the DataFrame with low variance features removed.
    - selected_feature_names: list, the list of remaining feature names.
    """

    print("STEP remove_low_variance_features")

    # Identify low variance features
    low_variance_features = feature_variances[feature_variances < threshold].index.tolist()

    # Remove low variance features if any
    if low_variance_features:
        print(f"Removing low variance features: {low_variance_features}")
        X_train_full_selected = X_train_full_selected.drop(columns=low_variance_features)
        selected_feature_names = [f for f in X_train_full_selected.columns if f not in low_variance_features]
    else:
        selected_feature_names = X_train_full_selected.columns.tolist()

    return X_train_full_selected, selected_feature_names


def taxonomy_annotations(data_df_clean, taxonomy_file_path, target_column, remove_original=False):
    """
    Extend data_df_clean with binary taxonomy annotations.

    Parameters:
        data_df_clean (pd.DataFrame): The original DataFrame to be extended.
        taxonomy_file_path (str): Path to the taxonomy annotations TSV file.
        target_column (str): Name of the column in data_df_clean that you want to keep
                            (e.g., a target or key column).
        remove_original (bool): If True, remove the original columns from data_df_clean
                                after the join, keeping only the target_column and
                                newly added taxonomy columns.

    Returns:
        data_df_clean (pd.DataFrame): The extended (and possibly pruned) DataFrame.
        data_df_clean_orig (pd.DataFrame): A copy of the original DataFrame, before modifications.
        tax_annot_df (pd.DataFrame): The crosstab/binary annotation DataFrame generated from taxonomy_file_path.
    """

    print("STEP taxonomy_annotations")

    # Copy the original DataFrame for safe-keeping
    data_df_clean_orig = data_df_clean.copy(deep=True)

    # Load taxonomy annotations
    tax_annot_df = pd.read_csv(taxonomy_file_path, sep='\t', index_col=0)
    
    # Focus on only the target_column from data_df_clean for the initial join
    data_df_clean_sub = data_df_clean[[target_column]]

    # Perform the join to add taxonomy binary columns
    data_df_clean_join = data_df_clean_sub.join(tax_annot_df, how='left')

    # Decide how to handle original columns
    if remove_original:
        # Keep only target_column and newly added taxonomy columns
        data_df_clean = data_df_clean_join.copy(deep=True)
    else:
        # Otherwise, merge the new columns into the original DataFrame
        # and retain all original columns
        # (drops the target_column to avoid duplication before merging it back)
        data_df_clean = data_df_clean_orig.join(data_df_clean_join.drop(columns=[target_column]), how='left')

    return data_df_clean, data_df_clean_orig, tax_annot_df


def eggnog_annotations(data_df_clean, eggnog_file_path, target_column, remove_original=False, chunk_size=1000, variance_threshold=0.0, max_features=10000):
    """
    Extend data_df_clean with binary eggnog annotations.
    Uses chunking to prevent memory issues with large eggnog data.

    Parameters:
        data_df_clean (pd.DataFrame): The original DataFrame to be extended.
        eggnog_file_path (str): Path to the eggnog annotations TSV file.
        target_column (str): Name of the column in data_df_clean that you want to keep
                            (e.g., a target or key column).
        remove_original (bool): If True, remove the original columns from data_df_clean
                                after the join, keeping only the target_column and
                                newly added eggnog columns.
        chunk_size (int): Number of columns to process at a time to avoid memory issues.
        variance_threshold (float): Minimum variance required to keep a feature column.
                                   Set to 0 to keep all features.
        max_features (int): Maximum number of features to include. If more features
                           are available, keep those with highest variance. Set to 0
                           for no limit.

    Returns:
        data_df_clean (pd.DataFrame): The extended (and possibly pruned) DataFrame.
        data_df_clean_orig (pd.DataFrame): A copy of the original DataFrame, before modifications.
        egg_annot_df (pd.DataFrame): The filtered annotation DataFrame generated from eggnog_file_path.
    """

    print("STEP eggnog_annotations")

    # Copy the original DataFrame for safe-keeping
    data_df_clean_orig = data_df_clean.copy(deep=True)

    # Focus on only the target_column from data_df_clean for the initial join
    data_df_clean_sub = data_df_clean[[target_column]]
    
    print(f"Loading eggnog data from {eggnog_file_path}")
    
    # Determine number of columns in file without loading it entirely
    # Just read the header
    try:
        reader = pd.read_csv(eggnog_file_path, sep='\t', index_col=0, nrows=0)
        total_columns = len(reader.columns)
        print(f"Found {total_columns} total eggnog features in file")
    except Exception as e:
        print(f"Error reading header: {str(e)}")
        print("Will attempt to process without knowing total column count")
        total_columns = -1  # Unknown column count
    
    # List to collect all chunks
    all_chunks = []
    # Dictionary to track feature variances if we need to limit features
    feature_variances = {}
    
    # Process the file in chunks to reduce memory usage
    if total_columns > 0:
        # If we know total column count, use standard chunking
        chunk_ranges = [(i, min(i + chunk_size, total_columns)) 
                        for i in range(0, total_columns, chunk_size)]
    else:
        # If total column count is unknown, use an alternative approach
        # First, get all column names from the file
        print("Attempting to read column names from file")
        try:
            # Read just the header row to get column names
            reader = pd.read_csv(eggnog_file_path, sep='\t', nrows=0)
            col_names = reader.columns.tolist()
            # First column is the index
            index_col = col_names[0]
            # Remaining columns are the features
            feature_cols = col_names[1:]
            total_columns = len(feature_cols)
            print(f"Successfully identified {total_columns} features")
            
            # Create chunk ranges based on feature columns
            chunk_ranges = []
            for i in range(0, len(feature_cols), chunk_size):
                chunk_cols = feature_cols[i:i+chunk_size]
                chunk_ranges.append((i, i+len(chunk_cols), [index_col] + chunk_cols))
        except Exception as e:
            print(f"Failed to read column names: {str(e)}")
            print("Cannot proceed without column information")
            return data_df_clean_orig, data_df_clean_orig, pd.DataFrame()
    
    # Process each chunk
    for chunk_idx, chunk_info in enumerate(chunk_ranges):
        if len(chunk_info) == 2:
            # Standard approach with known ranges
            chunk_start, chunk_end = chunk_info
            usecols = None  # Will be calculated below
            print(f"Processing eggnog features {chunk_start} to {chunk_end} of {total_columns}")
        else:
            # Approach with pre-determined columns
            chunk_start, chunk_end, usecols = chunk_info
            print(f"Processing eggnog feature chunk {chunk_idx+1} with {len(usecols)-1} features")
        
        # Calculate column indices to use if not already provided
        if usecols is None:
            usecols = [0] + list(range(chunk_start + 1, chunk_end + 1))
        
        # Read just these columns
        try:
            if isinstance(usecols[0], int):
                # Using position-based indexing
                chunk_df = pd.read_csv(
                    eggnog_file_path, 
                    sep='\t', 
                    index_col=0,
                    usecols=usecols,
                    low_memory=False
                )
            else:
                # Using name-based indexing
                chunk_df = pd.read_csv(
                    eggnog_file_path, 
                    sep='\t', 
                    index_col=usecols[0],  # First item is index column name
                    usecols=usecols,       # All columns including index
                    low_memory=False
                )
            
            # Apply basic filtering if threshold > 0
            if variance_threshold > 0:
                variances = chunk_df.var()
                keep_columns = variances[variances >= variance_threshold].index
                
                if len(keep_columns) > 0:
                    chunk_df = chunk_df[keep_columns]
                    # Store variances for potential feature limitation
                    for col in keep_columns:
                        feature_variances[col] = variances[col]
                    print(f"  Kept {len(keep_columns)} features with variance >= {variance_threshold}")
                else:
                    print(f"  No features in this chunk exceeded variance threshold {variance_threshold}")
                    continue  # Skip this chunk if no columns pass the filter
            else:
                # If we're not filtering by variance but might need to limit features later
                if max_features > 0:
                    variances = chunk_df.var()
                    for col in chunk_df.columns:
                        feature_variances[col] = variances[col]
            
            # Collect the chunk
            all_chunks.append(chunk_df)
            
            # Print memory usage periodically
            if (chunk_start // chunk_size) % 5 == 0:
                import psutil
                memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
                print(f"  Current memory usage: {memory_usage:.2f} MB")
                
        except Exception as e:
            print(f"Error processing chunk {chunk_start}-{chunk_end}: {str(e)}")
            # Continue with next chunk even if this one fails
    
    # Apply max_features limit if needed
    if max_features > 0 and feature_variances and sum(len(chunk.columns) for chunk in all_chunks) > max_features:
        print(f"Limiting to top {max_features} features by variance")
        
        # Sort features by variance (highest first)
        sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
        top_features = [feat[0] for feat in sorted_features[:max_features]]
        
        # Filter each chunk to only include top features
        filtered_chunks = []
        for chunk in all_chunks:
            # Get intersection of chunk columns and top features
            keep_cols = list(set(chunk.columns) & set(top_features))
            if keep_cols:
                filtered_chunks.append(chunk[keep_cols])
        
        all_chunks = filtered_chunks
    
    # Combine all the chunks
    if all_chunks:
        try:
            egg_annot_df = pd.concat(all_chunks, axis=1)
            print(f"Final eggnog feature set has {egg_annot_df.shape[1]} columns")
            
            # Perform the join to add eggnog binary columns
            print("Joining eggnog features to data")
            data_df_clean_join = data_df_clean_sub.join(egg_annot_df, how='left')
            
            # Decide how to handle original columns
            if remove_original:
                # Keep only target_column and newly added eggnog columns
                data_df_clean = data_df_clean_join.copy(deep=True)
            else:
                # Otherwise, merge the new columns into the original DataFrame
                # and retain all original columns
                # (drops the target_column to avoid duplication before merging it back)
                data_df_clean = data_df_clean_orig.join(data_df_clean_join.drop(columns=[target_column]), how='left')
        except Exception as e:
            print(f"Error during final dataframe construction: {str(e)}")
            print("Returning original dataframe due to joining error")
            egg_annot_df = pd.DataFrame()
            data_df_clean = data_df_clean_orig.copy(deep=True)
    else:
        print("No eggnog features available, returning original dataframe")
        egg_annot_df = pd.DataFrame()  # Empty DataFrame as no features available
        data_df_clean = data_df_clean_orig.copy(deep=True)

    return data_df_clean, data_df_clean_orig, egg_annot_df


def traithop_annotations(data_df_clean, traithop_file_path, target_column, remove_original=False):
    """
    Extend data_df_clean with binary traithop annotations.

    Parameters:
        data_df_clean (pd.DataFrame): The original DataFrame to be extended.
        traithop_file_path (str): Path to the traithop annotations TSV file.
        target_column (str): Name of the column in data_df_clean that you want to keep
                            (e.g., a target or key column).
        remove_original (bool): If True, remove the original columns from data_df_clean
                                after the join, keeping only the target_column and
                                newly added traithop columns.

    Returns:
        data_df_clean (pd.DataFrame): The extended (and possibly pruned) DataFrame.
        data_df_clean_orig (pd.DataFrame): A copy of the original DataFrame, before modifications.
        traithop_annot_df (pd.DataFrame): The crosstab/binary annotation DataFrame generated from eggnog_file_path.
    """

    print("STEP traithop_annotations")

    # Copy the original DataFrame for safe-keeping
    data_df_clean_orig = data_df_clean.copy(deep=True)

    # Load traithop annotations
    traithop_annot_df = pd.read_csv(traithop_file_path, sep='\t', index_col=0)
    
    # Focus on only the target_column from data_df_clean for the initial join
    data_df_clean_sub = data_df_clean[[target_column]]

    # Perform the join to add traithop binary columns
    data_df_clean_join = data_df_clean_sub.join(traithop_annot_df, how='left')

    # Decide how to handle original columns
    if remove_original:
        # Keep only target_column and newly added traithop columns
        data_df_clean = data_df_clean_join.copy(deep=True)
    else:
        # Otherwise, merge the new columns into the original DataFrame
        # and retain all original columns
        # (drops the target_column to avoid duplication before merging it back)
        data_df_clean = data_df_clean_orig.join(data_df_clean_join.drop(columns=[target_column]), how='left')

    return data_df_clean, data_df_clean_orig, traithop_annot_df


def remove_singleton_row_and_col(
    data_df_clean,
    exclude_cols=['medium'],
    verbose=True
):
    """
    Remove singleton rows and columns from a DataFrame based on specified criteria.
    
    This function performs the following steps:
    1. Removes numeric columns with a sum less than or equal to 1, excluding specified columns.
    2. Removes numeric columns where all values are 1.
    3. Removes rows with a sum of numeric values less than or equal to 1.
    4. Removes rows where all numeric values are 1.
    
    Parameters:
    - data_df_clean (pd.DataFrame): The input DataFrame to be cleaned.
    - exclude_cols (list of str, default=['medium']): Columns to exclude from being considered for column removal.
    - verbose (bool, default=True): If True, prints the shape of the DataFrame before and after each cleaning step.
    
    Returns:
    - data_df_clean (pd.DataFrame): The cleaned DataFrame after removing specified singleton rows and columns.
    """

    print("STEP remove_singleton_row_and_col")

    # STEP 1: Remove columns with sum <= 1, excluding specified columns
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns.difference(exclude_cols)
    
    if verbose:
        print(f"Initial shape: {data_df_clean.shape}")
    
    sum_less_eq_1 = data_df_clean[numeric_cols].sum(axis=0)
    cols_to_drop = sum_less_eq_1[sum_less_eq_1 <= 1].index.tolist()
    
    if verbose:
        print(f"Columns to drop (sum <= 1): {cols_to_drop}")
    
    data_df_clean = data_df_clean.drop(columns=cols_to_drop)
    
    if verbose:
        print(f"Shape after dropping columns with sum <= 1: {data_df_clean.shape}")
    
    # STEP 2: Remove columns that are all 1's
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    sum_all_ones = data_df_clean[numeric_cols].sum(axis=0)
    cols_all_ones = sum_all_ones[sum_all_ones == data_df_clean.shape[0]].index.tolist()
    
    if verbose:
        print(f"Columns to drop (all ones): {cols_all_ones}")
    
    data_df_clean = data_df_clean.drop(columns=cols_all_ones)
    
    if verbose:
        print(f"Shape after dropping columns that are all ones: {data_df_clean.shape}")
    
    # STEP 3: Remove rows with sum <= 1
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    sum_rows_less_eq_1 = data_df_clean[numeric_cols].sum(axis=1)
    rows_to_drop_le_1 = sum_rows_less_eq_1[sum_rows_less_eq_1 <= 1].index.tolist()
    
    if verbose:
        print(f"Rows to drop (sum <= 1): {rows_to_drop_le_1}")
    
    data_df_clean = data_df_clean.drop(index=rows_to_drop_le_1)
    
    if verbose:
        print(f"Shape after dropping rows with sum <= 1: {data_df_clean.shape}")
    
    # STEP 4: Remove rows that are all 1's
    sum_rows_all_ones = data_df_clean[numeric_cols].sum(axis=1)
    rows_all_ones = sum_rows_all_ones[sum_rows_all_ones == data_df_clean.shape[1]].index.tolist()
    
    if verbose:
        print(f"Rows to drop (all ones): {rows_all_ones}")
    
    data_df_clean = data_df_clean.drop(index=rows_all_ones)
    
    if verbose:
        print(f"Shape after dropping rows that are all ones: {data_df_clean.shape}")
    
    return data_df_clean



def remove_identical_pattern(
    data_df_clean,
    modellabel,
    output_dir=".",
    verbose=True
):
    """
    Remove duplicate columns in a DataFrame based on identical patterns and log the retention mapping.
    
    This function performs the following steps:
    1. Identifies columns with identical patterns (i.e., columns where all values are the same).
    2. Drops the duplicate columns, retaining only one column for each unique pattern.
    3. Creates a retention mapping that records which columns were retained and which were dropped.
    4. Saves the retention mapping to a CSV file named 'retention_df__{modellabel}.csv' in the specified output directory.
    
    Parameters:
    - data_df_clean (pd.DataFrame): The input DataFrame from which duplicate columns will be removed.
    - modellabel (str): A label to include in the retention mapping filename for identification.
    - output_dir (str, default="."): The directory where the retention mapping CSV will be saved.
    - verbose (bool, default=True): If True, prints the shape of the DataFrame before and after each cleaning step,
                                     and logs information about dropped columns.
    
    Returns:
    - data_df_clean (pd.DataFrame): The cleaned DataFrame with duplicate columns removed.
    """

    print("STEP remove_identical_pattern")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    patterns = {}
    columns_to_drop = []
    retention_map = {}

    if verbose:
        print("Initial DataFrame shape:", data_df_clean.shape)
    
    # Iterate over columns to identify duplicates
    for col in data_df_clean.columns:
        # Convert the column values to a tuple to use as a hashable pattern
        pattern = tuple(data_df_clean[col])
        if pattern not in patterns:
            patterns[pattern] = col
            retention_map[col] = []  # Initialize the list of dropped columns for this pattern
        else:
            # Add the current column to the drop list and map it to the retained column
            columns_to_drop.append(col)
            retention_map[patterns[pattern]].append(col)
            if verbose:
                print(f"Duplicate found: '{col}' is identical to '{patterns[pattern]}' and will be dropped.")
    
    if columns_to_drop:
        # Drop duplicate columns
        data_df_clean = data_df_clean.drop(columns=columns_to_drop)
        if verbose:
            print(f"Dropped {len(columns_to_drop)} duplicate columns.")
    else:
        if verbose:
            print("No duplicate columns found to drop.")
    
    if verbose:
        print("DataFrame shape after dropping duplicate columns:", data_df_clean.shape)
    
    # Prepare the retention mapping DataFrame
    retention_records = [
        {'Retained Column': retained, 'Deleted Columns': ','.join(duplicates)}
        for retained, duplicates in retention_map.items() if duplicates
    ]
    retention_df = pd.DataFrame(retention_records, columns=['Retained Column', 'Deleted Columns'])
    
    # Define the retention mapping file path
    retention_file_path = os.path.join(output_dir, f'retention_df__{modellabel}.csv')
    
    # Write the retention mapping to a CSV file
    retention_df.to_csv(retention_file_path, index=False)
    if verbose:
        print(f"Retention mapping saved to '{retention_file_path}'")
        print("Retention Mapping DataFrame:")
        print(retention_df.head())
    
    return data_df_clean


def remove_one_edit_distance_features(
    data_df_clean,
    modellabel,
    output_dir=".",
    verbose=True
):
    """
    Remove features that are one binary edit away from other features, keeping the one with most non-zero values.
    
    This function performs the following steps:
    1. Identifies groups of binary features that differ by exactly one edit (one bit flip).
    2. For each group, keeps the feature with the highest count of non-zero values.
    3. Removes all other features in the group.
    4. Creates a retention mapping that records which features were retained and which were dropped.
    5. Saves the retention mapping to a CSV file.
    
    Parameters:
    - data_df_clean (pd.DataFrame): The input DataFrame containing binary features.
    - modellabel (str): A label to include in the retention mapping filename for identification.
    - output_dir (str, default="."): The directory where the retention mapping CSV will be saved.
    - verbose (bool, default=True): If True, prints progress information.
    
    Returns:
    - data_df_clean (pd.DataFrame): The cleaned DataFrame with one-edit-distance features removed.
    """
    
    print("STEP remove_one_edit_distance_features")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("Initial DataFrame shape:", data_df_clean.shape)
    
    # Get numeric columns only (exclude 'medium' and other non-feature columns)
    feature_columns = [col for col in data_df_clean.columns if col != 'medium']
    numeric_df = data_df_clean[feature_columns]
    
    if verbose:
        print(f"Processing {len(feature_columns)} feature columns for one-edit distance filtering")
    
    # Convert to numpy array for faster computation
    feature_matrix = numeric_df.values
    feature_names = numeric_df.columns.tolist()
    
    # Calculate non-zero counts for each feature
    nonzero_counts = np.sum(feature_matrix != 0, axis=0)
    
    # Track which features to keep and which to drop
    features_to_drop = set()
    retention_map = {}
    
    # Find all pairs of features that are one edit apart
    one_edit_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            # Calculate Hamming distance (number of differing bits)
            hamming_distance = np.sum(feature_matrix[:, i] != feature_matrix[:, j])
            
            # If features are exactly one edit apart
            if hamming_distance == 1:
                one_edit_pairs.append((i, j, nonzero_counts[i], nonzero_counts[j]))
    
    if verbose and one_edit_pairs:
        print(f"Found {len(one_edit_pairs)} pairs of features with one-edit distance")
    
    # Process pairs to determine which features to drop
    # Sort pairs by the maximum non-zero count in each pair (descending)
    # This ensures we prioritize keeping features with more non-zero values
    one_edit_pairs.sort(key=lambda x: max(x[2], x[3]), reverse=True)
    
    for i, j, count_i, count_j in one_edit_pairs:
        name_i, name_j = feature_names[i], feature_names[j]
        
        # Skip if either feature is already marked for dropping
        if name_i in features_to_drop or name_j in features_to_drop:
            continue
            
        # Keep the feature with more non-zero values
        if count_i >= count_j:
            # Keep feature i, drop feature j
            features_to_drop.add(name_j)
            if name_i not in retention_map:
                retention_map[name_i] = []
            retention_map[name_i].append(name_j)
            if verbose:
                print(f"One-edit distance found: '{name_j}' (nonzero: {count_j}) "
                      f"will be dropped in favor of '{name_i}' (nonzero: {count_i})")
        else:
            # Keep feature j, drop feature i
            features_to_drop.add(name_i)
            if name_j not in retention_map:
                retention_map[name_j] = []
            retention_map[name_j].append(name_i)
            if verbose:
                print(f"One-edit distance found: '{name_i}' (nonzero: {count_i}) "
                      f"will be dropped in favor of '{name_j}' (nonzero: {count_j})")
    
    # Convert to list for dropping
    columns_to_drop = list(features_to_drop)
    
    if columns_to_drop:
        # Drop the identified columns
        data_df_clean = data_df_clean.drop(columns=columns_to_drop)
        if verbose:
            print(f"Dropped {len(columns_to_drop)} features due to one-edit distance similarity.")
    else:
        if verbose:
            print("No features found with one-edit distance similarity.")
    
    if verbose:
        print("DataFrame shape after one-edit distance filtering:", data_df_clean.shape)
    
    # Prepare the retention mapping DataFrame
    retention_records = [
        {'Retained Column': retained, 'Deleted Columns': ','.join(duplicates)}
        for retained, duplicates in retention_map.items() if duplicates
    ]
    retention_df = pd.DataFrame(retention_records, columns=['Retained Column', 'Deleted Columns'])
    
    # Define the retention mapping file path
    retention_file_path = os.path.join(output_dir, f'one_edit_retention_df__{modellabel}.csv')
    
    # Write the retention mapping to a CSV file
    retention_df.to_csv(retention_file_path, index=False)
    if verbose:
        print(f"One-edit distance retention mapping saved to '{retention_file_path}'")
        if len(retention_df) > 0:
            print("One-edit distance retention mapping (first 10 entries):")
            print(retention_df.head(10))
    
    return data_df_clean


def name_columns(
    data_nodes,
    X,
    modellabel,
    output_dir=".",
    verbose=True
):
    """
    Rename columns in DataFrame X based on unique naming derived from data_nodes.

    This function performs the following steps:
    1. Cleans the 'id' and 'name' columns in data_nodes to handle NaN and empty strings.
    2. Strips '.-' from EC ids in both data_nodes and X columns for accurate matching.
    3. Creates a unique 'name_unique' for each 'id' by appending a suffix to duplicate names.
    4. Renames the columns in DataFrame X using the created mapping.
    5. Saves the mapping dictionary to a CSV file for reference.

    Parameters:
    - data_nodes (pd.DataFrame): DataFrame containing at least 'id' and 'name' columns.
    - X (pd.DataFrame): DataFrame whose columns are to be renamed based on the mapping.
    - modellabel (str): Label to include in the output mapping filename.
    - output_dir (str, default="."): Directory to save the mapping CSV file.
    - verbose (bool, default=True): If True, prints progress and summary information.

    Returns:
    - X_renamed (pd.DataFrame): The DataFrame X with renamed columns.
    - id_to_unique_name (dict): Dictionary mapping original 'id' to 'name_unique'.
    """

    print("STEP name_columns")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1: Create a DataFrame with 'id' and 'name'
    mapping_df = data_nodes[['id', 'name']].copy()
    if verbose:
        print("Initial mapping_df:")
        print(mapping_df.head())

    # Remove rows where 'name' is NaN, None, or empty string
    mapping_df = mapping_df[~mapping_df['name'].isnull() & (mapping_df['name'] != '')]
    if verbose:
        print("\nMapping_df after removing rows with NaN or empty 'name':")
        print(mapping_df.head())

    # STEP 2: Generate a count for each occurrence of a name to handle duplicates
    mapping_df['name_count'] = mapping_df.groupby('name').cumcount()
    if verbose:
        print("\nMapping_df after adding 'name_count':")
        print(mapping_df.head())

    # STEP 3: Extract prefix from 'id' (e.g., 'CHEBI:', 'GO:', etc.)
    mapping_df['prefix'] = mapping_df['id'].str.extract(r'^([^:]+:)')[0]
    if verbose:
        print("\nMapping_df after extracting 'prefix':")
        print(mapping_df.head())

    # STEP 4: Strip '.-' from 'id' for EC ids
    def strip_ec_dot_dash(id_str):
        if pd.isnull(id_str):
            return id_str
        if id_str.startswith('EC:'):
            return id_str.replace('.-', '')
        else:
            return id_str

    mapping_df['id_stripped'] = mapping_df['id'].apply(strip_ec_dot_dash)
    if verbose:
        print("\nMapping_df after stripping '.-' from EC ids:")
        print(mapping_df.head())

    # STEP 5: Create unique names by appending a suffix to duplicates and prepending prefix
    def create_name_unique(row):
        if pd.isnull(row['name']) or row['name'] == '':
            return None
        else:
            if row['name_count'] > 0:
                return f"{row['prefix']}{row['name']}_{row['name_count']}"
            else:
                return f"{row['prefix']}{row['name']}"

    mapping_df['name_unique'] = mapping_df.apply(create_name_unique, axis=1)
    if verbose:
        print("\nMapping_df after creating 'name_unique':")
        print(mapping_df.head())

    # Remove rows where 'name_unique' is NaN or empty
    mapping_df = mapping_df[~mapping_df['name_unique'].isnull() & (mapping_df['name_unique'] != '')]
    if verbose:
        print("\nMapping_df after removing rows with NaN or empty 'name_unique':")
        print(mapping_df.head())

    # STEP 6: Create the mapping dictionary from 'id_stripped' to 'name_unique'
    id_to_unique_name = mapping_df.set_index('id_stripped')['name_unique'].to_dict()
    if verbose:
        print("\nMapping Dictionary (id_to_unique_name):")
        for key, value in list(id_to_unique_name.items())[:5]:  # Show first 5 mappings
            print(f"{key} => {value}")

    # STEP 7: Rename the columns in X using the unique mapping
    # Adjust X.columns by stripping '.-' from EC columns
    X_columns_stripped = X.columns.map(strip_ec_dot_dash)

    # Map X_columns_stripped to new names
    X_renamed_columns = X_columns_stripped.map(id_to_unique_name)

    # Identify columns that were not mapped (resulting in NaN) and handle them
    unmapped_columns = X.columns[X_renamed_columns.isna()].tolist()
    if unmapped_columns:
        if verbose:
            print(f"\nUnmapped columns found: {unmapped_columns}")
        # Keep the original names for unmapped columns
        X_renamed_columns = [
            original if pd.isna(new) else new
            for original, new in zip(X.columns, X_renamed_columns)
        ]
        if verbose:
            print("Unmapped columns have been retained with original names.")
    else:
        X_renamed_columns = X_renamed_columns.tolist()

    # Apply the new column names to X_renamed
    X_renamed = X.copy()
    X_renamed.columns = X_renamed_columns
    
    # Check for duplicate column names after renaming and ensure all are unique
    column_counts = X_renamed.columns.value_counts()
    duplicate_cols = column_counts[column_counts > 1].index.tolist()
    
    if duplicate_cols:
        if verbose:
            print(f"\nWarning: Found {len(duplicate_cols)} duplicate column names after renaming")
            print(f"Duplicates: {duplicate_cols[:5]}...")  # Show first 5
        
        # Fix duplicates by ensuring truly unique names
        cols = pd.Series(X_renamed.columns)
        used_names = set()
        
        # Create a mapping of new unique names
        for i, col in enumerate(cols):
            original_col = col
            counter = 0
            
            # If this column name is already used, find a unique variant
            while col in used_names:
                counter += 1
                col = f"{original_col}_{counter}"
            
            used_names.add(col)
            cols[i] = col
        
        X_renamed.columns = cols
        
        # Verify all columns are now unique
        final_counts = X_renamed.columns.value_counts()
        remaining_duplicates = final_counts[final_counts > 1]
        
        if len(remaining_duplicates) > 0:
            print(f"ERROR: Still have {len(remaining_duplicates)} duplicates after fix!")
            print(remaining_duplicates.head())
        elif verbose:
            print("Successfully fixed all duplicate columns with unique numeric suffixes")

    if verbose:
        print("\nDataFrame X after renaming columns:")
        print(X_renamed.head())

    return X_renamed, id_to_unique_name


def extract_taxonomic_groups(taxa_ids, data_nodes, taxonomic_level="family", data_edges=None):
    """
    Extract taxonomic groups for stratification based on specified taxonomic level.
    Uses local knowledge graph data to traverse taxonomic hierarchy.
    
    Parameters:
    - taxa_ids (pd.Index): Index of taxa IDs (e.g., NCBITaxon:12345)
    - data_nodes (pd.DataFrame): Node metadata with taxonomic information
    - taxonomic_level (str): Taxonomic level to extract (e.g., "family", "genus", "species")
    - data_edges (pd.DataFrame): Edge data for traversing strain -> NCBITaxon relationships
    
    Returns:
    - pd.Series: Taxonomic groups for each taxa ID
    """
    print(f"\nExtracting {taxonomic_level}-level taxonomic groups using local KG data...")
    
    # Load rank information if available
    rank_file_path = "../kg-microbe/kg_microbe/utils/NCBITaxon_rank.tsv"
    rank_info = {}
    
    try:
        rank_df = pd.read_csv(rank_file_path, sep='\t')
        rank_info = dict(zip(rank_df['identifier'], rank_df['rank']))
        print(f"Loaded rank information for {len(rank_info)} taxa")
    except Exception as e:
        print(f"Warning: Could not load rank file {rank_file_path}: {e}")
    
    # Create mapping from nodes data
    node_names = {}
    if 'id' in data_nodes.columns and 'name' in data_nodes.columns:
        node_names = dict(zip(data_nodes['id'], data_nodes['name']))
        print(f"Loaded names for {len(node_names)} nodes")
    
    # Create strain-to-NCBITaxon mapping from edges if available
    strain_to_ncbi = {}
    if data_edges is not None:
        print("Building strain-to-NCBITaxon mapping from edges...")
        strain_edges = data_edges[
            (data_edges['subject'].str.startswith('strain:', na=False)) &
            (data_edges['object'].str.startswith('NCBITaxon:', na=False)) &
            (data_edges['predicate'] == 'biolink:subclass_of')
        ]
        strain_to_ncbi = dict(zip(strain_edges['subject'], strain_edges['object']))
        print(f"Found {len(strain_to_ncbi)} strain-to-NCBITaxon mappings")
    
    taxonomic_groups = []
    
    for taxa_id in taxa_ids:
        group = extract_taxonomic_group_local(taxa_id, taxonomic_level, rank_info, node_names, strain_to_ncbi)
        taxonomic_groups.append(group)
    
    groups_series = pd.Series(taxonomic_groups, index=taxa_ids)
    
    # Print summary
    unique_groups = groups_series.nunique()
    largest_groups = groups_series.value_counts().head(5)
    print(f"Created {unique_groups} taxonomic groups at {taxonomic_level} level")
    print(f"Largest groups: {largest_groups.to_dict()}")
    
    # Check for potential issues
    single_member_groups = sum(1 for count in groups_series.value_counts() if count == 1)
    if single_member_groups > len(groups_series.unique()) * 0.7:
        print(f"Warning: {single_member_groups} groups have only 1 member")
        print(f"Consider using a higher taxonomic level (e.g., 'order' instead of '{taxonomic_level}')")
    
    return groups_series


def extract_taxonomic_group_local(taxa_id, taxonomic_level, rank_info, node_names, strain_to_ncbi=None):
    """
    Extract taxonomic group from local data for a single taxon.
    
    Parameters:
    - taxa_id (str): Taxon identifier (e.g., NCBITaxon:12345)
    - taxonomic_level (str): Target taxonomic level
    - rank_info (dict): Mapping of taxon ID to rank
    - node_names (dict): Mapping of taxon ID to scientific name
    - strain_to_ncbi (dict): Mapping of strain IDs to parent NCBITaxon IDs
    
    Returns:
    - str: Taxonomic group name
    """
    import re
    
    # Handle strain cases first - resolve to parent NCBITaxon
    if str(taxa_id).startswith('strain:'):
        if strain_to_ncbi and taxa_id in strain_to_ncbi:
            parent_ncbi = strain_to_ncbi[taxa_id]
            print(f"Resolving strain {taxa_id} -> {parent_ncbi}")
            # Recursively call with the parent NCBITaxon
            return extract_taxonomic_group_local(parent_ncbi, taxonomic_level, rank_info, node_names, strain_to_ncbi)
        else:
            # Fallback if no parent mapping found
            strain_base = str(taxa_id).split(':')[1].split('_')[0][:10]
            return f"strain_{strain_base}"
    
    # Handle other non-NCBI taxa
    if not str(taxa_id).startswith('NCBITaxon:'):
        return f"unknown_{hash(str(taxa_id)) % 1000}"
    
    # Check rank information
    current_rank = rank_info.get(taxa_id, 'unknown')
    
    # If this taxon is already at the target level, use its name
    if current_rank == taxonomic_level:
        taxon_name = node_names.get(taxa_id, f"unknown_{taxa_id}")
        return clean_taxonomic_name(taxon_name)
    
    # If we have the name, try to infer family from naming patterns
    taxon_name = node_names.get(taxa_id, '')
    
    if taxonomic_level == 'family':
        # Look for family-like patterns in the name or use heuristics
        family_name = infer_family_from_name(taxon_name, taxa_id)
        return family_name
    elif taxonomic_level == 'genus':
        # For genus, typically the first word of the scientific name
        if taxon_name and ' ' in taxon_name:
            genus = taxon_name.split()[0]
            if re.match(r'^[A-Z][a-z]+$', genus):
                return genus
        return f"unknown_genus_{hash(str(taxa_id)) % 1000}"
    else:
        # For other levels, use a simple fallback
        return f"{taxonomic_level}_{hash(str(taxa_id)) % 10000}"


def infer_family_from_name(taxon_name, taxa_id):
    """
    Infer family name from taxon name using biological naming conventions.
    
    Parameters:
    - taxon_name (str): Scientific name of the taxon
    - taxa_id (str): Taxon identifier for fallback
    
    Returns:
    - str: Inferred family name
    """
    import re
    
    if not taxon_name:
        return f"unknown_family_{hash(str(taxa_id)) % 1000}"
    
    # Clean the name
    name = taxon_name.strip()
    
    # Common family naming patterns in bacteria
    family_patterns = {
        # Genus -> Family mappings for common bacterial genera
        'Escherichia': 'Enterobacteriaceae',
        'Salmonella': 'Enterobacteriaceae', 
        'Shigella': 'Enterobacteriaceae',
        'Klebsiella': 'Enterobacteriaceae',
        'Enterobacter': 'Enterobacteriaceae',
        'Serratia': 'Enterobacteriaceae',
        'Citrobacter': 'Enterobacteriaceae',
        'Proteus': 'Enterobacteriaceae',
        'Yersinia': 'Enterobacteriaceae',
        
        'Bacillus': 'Bacillaceae',
        'Clostridium': 'Clostridiaceae',
        'Staphylococcus': 'Staphylococcaceae',
        'Streptococcus': 'Streptococcaceae',
        'Lactobacillus': 'Lactobacillaceae',
        'Pseudomonas': 'Pseudomonadaceae',
        'Vibrio': 'Vibrionaceae',
        'Acinetobacter': 'Acinetobacteraceae',
        'Mycobacterium': 'Mycobacteriaceae',
        'Streptomyces': 'Streptomycetaceae',
        'Bacteroides': 'Bacteroidaceae',
        'Prevotella': 'Prevotellaceae',
        'Bifidobacterium': 'Bifidobacteriaceae',
        'Enterococcus': 'Enterococcaceae',
        'Listeria': 'Listeriaceae',
        'Corynebacterium': 'Corynebacteriaceae',
        'Actinomyces': 'Actinomycetaceae',
        'Fusobacterium': 'Fusobacteriaceae',
        'Campylobacter': 'Campylobacteraceae',
        'Helicobacter': 'Helicobacteraceae',
        'Neisseria': 'Neisseriaceae',
        'Haemophilus': 'Haemophilaceae',
        'Moraxella': 'Moraxellaceae',
        'Burkholderia': 'Burkholderiaceae',
        'Ralstonia': 'Burkholderiaceae',
        'Alcaligenes': 'Alcaligenaceae',
        'Bordetella': 'Alcaligenaceae',
        'Brucella': 'Brucellaceae',
        'Francisella': 'Francisellaceae',
        'Legionella': 'Legionellaceae',
        'Coxiella': 'Coxiellaceae',
        'Rickettsia': 'Rickettsiaceae',
        'Chlamydia': 'Chlamydiaceae',
        'Treponema': 'Treponemataceae',
        'Borrelia': 'Borreliaceae',
        'Leptospira': 'Leptospiraceae'
    }
    
    # Extract genus (first word)
    words = name.split()
    if words:
        genus = words[0]
        
        # Direct lookup
        if genus in family_patterns:
            return family_patterns[genus]
        
        # Pattern-based inference for -aceae families
        potential_family = genus + 'aceae'
        return potential_family
    
    # Fallback to hash-based grouping
    return f"unknown_family_{hash(str(taxa_id)) % 1000}"


def clean_taxonomic_name(name):
    """
    Clean taxonomic name for use as a group identifier.
    
    Parameters:
    - name (str): Raw taxonomic name
    
    Returns:
    - str: Cleaned name
    """
    if not name:
        return "unknown"
    
    # Remove special characters and standardize
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
    cleaned = re.sub(r'_+', '_', cleaned)  # Remove multiple underscores
    cleaned = cleaned.strip('_')
    
    return cleaned if cleaned else "unknown"


def group_based_train_val_test_split(X, y, groups, train_size=0.7, val_size=0.2, test_size=0.1, 
                                    random_state=42, verbose=True):
    """
    Split data ensuring that entire groups stay together in one split.
    This is the correct way to do taxonomic stratification.
    
    Parameters:
    - X: Feature matrix
    - y: Target labels
    - groups: Group assignments for each sample
    - train_size, val_size, test_size: Split proportions
    - random_state: Random seed
    - verbose: Print progress
    
    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    import numpy as np
    
    if verbose:
        print(f"Performing group-based splitting (train={train_size}, val={val_size}, test={test_size})")
    
    # Get unique groups and their sizes
    group_info = groups.value_counts().to_dict()
    unique_groups = list(group_info.keys())
    total_samples = len(X)
    
    if verbose:
        print(f"Total groups: {len(unique_groups)}")
        print(f"Group size distribution: min={min(group_info.values())}, max={max(group_info.values())}, mean={np.mean(list(group_info.values())):.1f}")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Shuffle groups for random assignment
    np.random.shuffle(unique_groups)
    
    # Assign groups to splits trying to match target proportions
    train_groups = []
    val_groups = []
    test_groups = []
    
    train_samples = 0
    val_samples = 0 
    test_samples = 0
    
    target_train = int(total_samples * train_size)
    target_val = int(total_samples * val_size)
    target_test = total_samples - target_train - target_val
    
    for group in unique_groups:
        group_size = group_info[group]
        
        # Decide which split to assign this group to
        # Choose the split that's furthest from its target
        train_deficit = target_train - train_samples
        val_deficit = target_val - val_samples
        test_deficit = target_test - test_samples
        
        # Assign to the split with the largest deficit (as proportion)
        train_prop_deficit = train_deficit / target_train if target_train > 0 else 0
        val_prop_deficit = val_deficit / target_val if target_val > 0 else 0
        test_prop_deficit = test_deficit / target_test if target_test > 0 else 0
        
        if train_prop_deficit >= val_prop_deficit and train_prop_deficit >= test_prop_deficit:
            train_groups.append(group)
            train_samples += group_size
        elif val_prop_deficit >= test_prop_deficit:
            val_groups.append(group)
            val_samples += group_size
        else:
            test_groups.append(group)
            test_samples += group_size
    
    if verbose:
        print(f"Final split sizes:")
        print(f"  Train: {train_samples} samples ({train_samples/total_samples:.2%}) in {len(train_groups)} groups")
        print(f"  Val:   {val_samples} samples ({val_samples/total_samples:.2%}) in {len(val_groups)} groups")
        print(f"  Test:  {test_samples} samples ({test_samples/total_samples:.2%}) in {len(test_groups)} groups")
    
    # Create boolean masks for each split
    train_mask = groups.isin(train_groups)
    val_mask = groups.isin(val_groups)
    test_mask = groups.isin(test_groups)
    
    # Split the data
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    # Verify no group overlap
    if verbose:
        train_group_set = set(train_groups)
        val_group_set = set(val_groups)
        test_group_set = set(test_groups)
        
        overlap_train_val = train_group_set & val_group_set
        overlap_train_test = train_group_set & test_group_set
        overlap_val_test = val_group_set & test_group_set
        
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print(f"❌ ERROR: Group overlaps detected!")
            print(f"  Train-Val: {len(overlap_train_val)}")
            print(f"  Train-Test: {len(overlap_train_test)}")
            print(f"  Val-Test: {len(overlap_val_test)}")
        else:
            print(f"✅ SUCCESS: Perfect group separation achieved!")
            print(f"  No taxonomic groups appear in multiple splits")
    
    return X_train, X_val, X_test, y_train, y_val, y_test




def check_and_split_data(
    X,
    y,
    mediumid,
    modellabel,
    output_dir=".",
    random_seed=42,
    verbose=True,
    taxonomic_stratify=False,
    data_nodes=None,
    taxonomic_level="family",
    data_edges=None
):
    """
    Check for duplicated column names in DataFrame X, split the data into training and test sets,
    convert labels to binary format based on a specified medium, and save the split datasets.

    This function performs the following steps:
    1. Verifies that all column names in X are unique.
    2. Displays the first few column names for confirmation.
    3. Splits X and y into training and test sets with stratification based on y and optionally taxonomic groups.
    4. Converts the labels to binary format, marking the specified medium as the positive class.
    5. Prints the unique classes and their distributions in both training and test sets.
    6. Concatenates the features and labels into separate DataFrames for training and testing.
    7. Saves the resulting DataFrames to compressed TSV files.
    8. Prints the shapes of the resulting training and test sets.

    Parameters:
    - X (pd.DataFrame): Feature DataFrame with unique column names.
    - y (pd.Series or pd.DataFrame): Target labels corresponding to the rows in X.
    - mediumid (str or int): Identifier for the medium to classify as the positive class.
    - modellabel (str): Label to include in the output filenames for identification.
    - output_dir (str, default="."): Directory to save the output TSV files.
    - random_seed (int, default=42): Seed for random number generator to ensure reproducibility.
    - verbose (bool, default=True): If True, prints detailed information during processing.
    - taxonomic_stratify (bool, default=False): If True, perform taxonomic stratification to prevent data leakage.
    - data_nodes (pd.DataFrame, default=None): Node metadata with taxonomic information (required if taxonomic_stratify=True).
    - taxonomic_level (str, default="family"): Taxonomic level to use for stratification (e.g., "family", "genus", "species").

    Returns:
    - X_train_full, y_train_full_binary, X_test, y_test, y_test_binary, X_val, y_val, y_val_binary: Training, validation, and test sets with features and binary labels.
    """

    print("STEP check_and_split_data")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1: Verify that all column names are unique
    if X.columns.duplicated().any():
        duplicated_columns = X.columns[X.columns.duplicated()].unique()
        warning_message = f"Warning: The following column names are duplicated: {duplicated_columns}"
        print(warning_message) if verbose else None
    else:
        success_message = "All column names have been successfully renamed and are unique."
        print(success_message) if verbose else None

    # (Optional) Display the first few column names to confirm
    first_n = 10
    columns_display = X.columns[:first_n].tolist()
    print(f"First {first_n} column names after renaming: {columns_display}") if verbose else None

    # STEP 2: Split into training and test sets
    try:
        if taxonomic_stratify:
            if data_nodes is None:
                raise ValueError("data_nodes must be provided when taxonomic_stratify=True")
            
            # Extract taxonomic groups for stratification
            taxonomic_groups = extract_taxonomic_groups(X.index, data_nodes, taxonomic_level, data_edges)
            
            if verbose:
                print(f"Using taxonomic stratification at {taxonomic_level} level")
                print(f"Found {len(taxonomic_groups.unique())} unique taxonomic groups")
            
            # Validate taxonomic groups before stratification
            group_counts = taxonomic_groups.value_counts()
            single_member_groups = sum(1 for count in group_counts if count == 1)
            if single_member_groups > len(group_counts) * 0.8:
                print(f"Warning: {single_member_groups}/{len(group_counts)} groups have only 1 member")
                print("Consider using a higher taxonomic level for better stratification")
            
            # Check for extremely small groups that might cause stratification issues
            very_small_groups = sum(1 for count in group_counts if count <= 2)
            if very_small_groups > len(group_counts) * 0.5:
                print(f"Warning: {very_small_groups} groups have ≤2 members, may cause stratification issues")
            
            # Use proper group-based splitting for taxonomic stratification
            # This ensures entire taxonomic groups stay together in one split
            X_train_full, X_val, X_test, y_train_full, y_val, y_test = group_based_train_val_test_split(
                X, y, taxonomic_groups, 
                train_size=0.7, val_size=0.2, test_size=0.1,
                random_state=random_seed, verbose=verbose
            )
            
        else:
            # Standard stratification by target only
            X_train_full, X_temp, y_train_full, y_temp = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=random_seed
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=random_seed
            )

        if verbose:
            print("Data successfully split into training, validation, and test sets.")
            print(f"Training set size: {X_train_full.shape[0]} samples")
            print(f"Validation set size: {X_val.shape[0]} samples") 
            print(f"Test set size: {X_test.shape[0]} samples")
    except ValueError as e:
        print(f"Error during train_test_split: {e}") if verbose else None
        raise

    # STEP 3: Define the positive class label
    positive_class_label = 'medium:' + str(mediumid)

    # STEP 4: Convert labels to binary format
    y_train_full_binary = y_train_full.apply(
        lambda x: 1 if x == positive_class_label else 0
    ).astype(int)
    y_test_binary = y_test.apply(
        lambda x: 1 if x == positive_class_label else 0
    ).astype(int)

    # STEP 3: Define the positive class label
    positive_class_label = 'medium:' + str(mediumid)

    # STEP 4: Convert labels to binary format
    y_val_binary = y_val.apply(
        lambda x: 1 if x == positive_class_label else 0
    ).astype(int)

    # STEP 5: Print unique classes and their distributions
    if verbose:
        print("\nUnique classes in y_train_full_binary:", y_train_full_binary.unique())
        print("Class distribution in y_train_full_binary:")
        print(y_train_full_binary.value_counts())

        print("\nUnique classes in y_test_binary:", y_test_binary.unique())
        print("Class distribution in y_test_binary:")
        print(y_test_binary.value_counts())

    # STEP 6: Concatenate the features and labels for each set
    train_df = pd.concat(
        [X_train_full, y_train_full_binary.rename('label')],
        axis=1
    )
    test_df = pd.concat(
        [X_test, y_test_binary.rename('label')],
        axis=1
    )
    val_df = pd.concat(
        [X_val, y_val_binary.rename('label')],
        axis=1
    )

    # STEP 7: Define file paths
    train_file_path = os.path.join(
        output_dir, f'taxa_to_media__{modellabel}_data_df_clean__train.tsv.gz'
    )
    test_file_path = os.path.join(
        output_dir, f'taxa_to_media__{modellabel}_data_df_clean__test.tsv.gz'
    )
    val_file_path = os.path.join(
        output_dir, f'taxa_to_media__{modellabel}_data_df_clean__val.tsv.gz'
    )

    # STEP 8: Save the DataFrames to TSV files with gzip compression
    try:
        train_df.to_csv(
            train_file_path, sep='\t', index=True, header=True, compression='gzip'
        )
        test_df.to_csv(
            test_file_path, sep='\t', index=True, header=True, compression='gzip'
        )
        val_df.to_csv(
            val_file_path, sep='\t', index=True, header=True, compression='gzip'
        )
        if verbose:
            print(f"\nTraining set saved to '{train_file_path}'")
            print(f"Validation set saved to '{val_file_path}'")
            print(f"Test set saved to '{test_file_path}'")
    except Exception as e:
        print(f"Error saving TSV files: {e}") if verbose else None
        raise

    # STEP 9: Print the shapes of the datasets
    if verbose:
        print("\nFinal Shapes:")
        print(f"Training data shape: {X_train_full.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")

    return X_train_full, y_train_full_binary, X_test, y_test, y_test_binary, X_val, y_val, y_val_binary

def perform_permutation_knockoff_analysis(
    X_train_full,
    y_train_full_binary,
    X_val,
    y_val_binary,
    RANDOM_SEEDS,
    CV_FOLDS=5,
    ITERATIONS=10000,
    modellabel="model",
    SHAP_PERM_PERCENTILE_THRESHOLD=0.95,
    SIGNIFICANCE_LEVEL=0.05,
    output_dir=".",
    GPU=False
):
    """
    Perform permutation knockoff analysis for feature selection.
    ...

    """

    print("STEP perform_permutation_knockoff_analysis")
    
    # Validate input data
    if X_train_full.empty or X_train_full.shape[1] == 0:
        print("ERROR: X_train_full is empty or has no features!")
        print(f"X_train_full shape: {X_train_full.shape}")
        return {
            'metrics_df': pd.DataFrame(),
            'shap_df': pd.DataFrame(),
            'selected_features': [],
            'error': 'No features available for analysis'
        }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to collect metrics and SHAP DataFrames
    metrics_list = []
    shap_df_list = []
    
    if CV_FOLDS > 0:
        # Prepare StratifiedKFold only once (we'll reuse it if we want CV)
        kf = StratifiedKFold(
            n_splits=CV_FOLDS, 
            shuffle=True, 
            random_state=RANDOM_SEEDS[0] if RANDOM_SEEDS else 42
        )

    print("Starting permutation knockoff runs")
    
    # -----------------------------
    # The main loop over RANDOM_SEEDS
    # -----------------------------
    for seed in RANDOM_SEEDS:
        print(f"Processing permutation seed: {seed}")

        # 1) Permute X_train_full
        X_train_permuted_full = X_train_full.copy().reset_index(drop=True)
        permuted_train_cols = []
        for col in X_train_full.columns:
            permuted_col_name = f"{col}_perm"
            permuted_col = (
                X_train_full[col]
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )
            permuted_col.name = permuted_col_name
            permuted_train_cols.append(permuted_col)

        # Concatenate permuted columns to training DataFrame
        if not permuted_train_cols:
            print(f"WARNING: No columns to permute! X_train_full has {len(X_train_full.columns)} columns")
            continue
        train_permuted_df = pd.concat(permuted_train_cols, axis=1)
        X_train_permuted_full = pd.concat(
            [X_train_permuted_full, train_permuted_df],
            axis=1
        )
        print(f"X_train_permuted_full shape: {X_train_permuted_full.shape}")

        # 2) Permute X_val in the same manner
        X_val_permuted_full = X_val.copy().reset_index(drop=True)
        permuted_val_cols = []
        for col in X_val.columns:
            permuted_col_name = f"{col}_perm"
            permuted_col = (
                X_val[col]
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )
            permuted_col.name = permuted_col_name
            permuted_val_cols.append(permuted_col)

        if not permuted_val_cols:
            print(f"WARNING: No columns to permute in validation set! X_val has {len(X_val.columns)} columns")
            continue
        val_permuted_df = pd.concat(permuted_val_cols, axis=1)
        X_val_permuted_full = pd.concat(
            [X_val_permuted_full, val_permuted_df],
            axis=1
        )
        print(f"X_val_permuted_full shape: {X_val_permuted_full.shape}")

        print(f"After permutation - X_train_permuted_full shape: {X_train_permuted_full.shape}")
        print(f"After permutation - y_train_full_binary shape: {y_train_full_binary.shape}")

        # ------------------------------------------------------
        # A) If CV_FOLDS=0, train a single model per seed
        # ------------------------------------------------------
        if CV_FOLDS == 0:
            print(f"CV_FOLDS=0, training a single model for seed={seed}")
            
            unique_classes = np.unique(y_train_full_binary)
            if len(unique_classes) < 2:
                print("Skipping seed due to lack of both classes.")
                continue  # valid 'continue' inside the for seed loop

            class_weights_perm = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y_train_full_binary
            )
            class_weights_perm_dict = dict(zip(unique_classes, class_weights_perm))
            print(f"Single model - Class Weights: {class_weights_perm_dict}")

            # Build CatBoost Pools
            train_data = Pool(X_train_permuted_full, label=y_train_full_binary)
            val_data = Pool(X_val_permuted_full, label=y_val_binary)

            perm_params = {
                'iterations': ITERATIONS,
                'depth': 4,
                'learning_rate': 0.05,
                'l2_leaf_reg': 4,
                'bagging_temperature': 1,
                'random_strength': 6,
                'loss_function': 'Logloss',
                'random_seed': seed,
                'verbose': 100,
                'early_stopping_rounds': 50,
                'use_best_model': True,
                # 'class_weights': class_weights_perm_dict
            }
            if GPU:
                perm_params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })

            modelperm = CatBoostClassifier(**perm_params)
            modelperm.fit(train_data, eval_set=val_data)

            # Evaluate on the entire (permuted) training set
            y_pred = modelperm.predict(X_train_permuted_full)
            y_pred_proba = modelperm.predict_proba(X_train_permuted_full)[:, 1]

            accuracy = accuracy_score(y_train_full_binary, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_train_full_binary, y_pred)
            classification_rep = classification_report(y_train_full_binary, y_pred, output_dict=True)
            auc_roc = roc_auc_score(y_train_full_binary, y_pred_proba)

            metrics_list.append({
                'random_seed': seed,
                'fold': 1,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'auc_roc': auc_roc,
                'classification_report': classification_rep
            })

            # SHAP
            explainer = shap.TreeExplainer(modelperm)
            shap_values = explainer(X_train_permuted_full)

            if isinstance(shap_values, shap.Explanation):
                shap_values_positive_class = shap_values.values
            elif isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values_positive_class = shap_values[1]
                else:
                    shap_values_positive_class = shap_values
            elif isinstance(shap_values, np.ndarray):
                shap_values_positive_class = shap_values
            else:
                print("Error: Unexpected SHAP values structure.")
                shap_values_positive_class = np.zeros(
                    (X_train_permuted_full.shape[0], X_train_permuted_full.shape[1])
                )

            print(f"Single-model - shap_values_positive_class shape: {shap_values_positive_class.shape}")
            shap_df = pd.DataFrame(shap_values_positive_class, columns=X_train_permuted_full.columns)
            shap_df_list.append(shap_df)

        # ------------------------------------------------------
        # B) Else, do cross-validation
        # ------------------------------------------------------
        else:
            print(f"Performing {CV_FOLDS}-fold cross-validation for seed={seed} ...")
            # kf is already defined if CV_FOLDS > 0
            for fold, (train_index, val_index) in enumerate(
                kf.split(X_train_permuted_full, y_train_full_binary),
                start=1
            ):
                print(f"\n  [Seed={seed}] Fold {fold}/{CV_FOLDS}")
                
                X_train_fold = X_train_permuted_full.iloc[train_index]
                X_val_fold   = X_train_permuted_full.iloc[val_index]
                y_train_fold = y_train_full_binary.iloc[train_index]
                y_val_fold   = y_train_full_binary.iloc[val_index]
                
                unique_classes = np.unique(y_train_fold)
                if len(unique_classes) < 2:
                    print(f"Fold {fold} skipped due to lack of both classes.")
                    continue

                class_weights_perm = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes,
                    y=y_train_fold
                )
                class_weights_perm_dict = dict(zip(unique_classes, class_weights_perm))
                print(f"Fold {fold} - Class Weights perm: {class_weights_perm_dict}")
                
                train_data = Pool(X_train_fold, label=y_train_fold)
                val_data   = Pool(X_val_fold,   label=y_val_fold)
                
                perm_params = {
                    'iterations': ITERATIONS,
                    'depth': 4,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 4,
                    'bagging_temperature': 1,
                    'random_strength': 6,
                    'loss_function': 'Logloss',
                    'random_seed': seed,
                    'verbose': 100,
                    'early_stopping_rounds': 50,
                    'use_best_model': True,
                    # 'class_weights': class_weights_perm_dict
                }
                if GPU:
                    perm_params.update({'task_type': 'GPU', 'devices': '0'})

                modelperm = CatBoostClassifier(**perm_params)
                modelperm.fit(train_data, eval_set=val_data)
                
                y_val_pred = modelperm.predict(val_data)
                y_val_pred_proba = modelperm.predict_proba(val_data)[:, 1]
                
                accuracy = accuracy_score(y_val_fold, y_val_pred)
                balanced_accuracy = balanced_accuracy_score(y_val_fold, y_val_pred)
                classification_rep = classification_report(y_val_fold, y_val_pred, output_dict=True)
                auc_roc = roc_auc_score(y_val_fold, y_val_pred_proba)
                
                metrics_list.append({
                    'random_seed': seed,
                    'fold': fold,
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_accuracy,
                    'auc_roc': auc_roc,
                    'classification_report': classification_rep
                })
                
                # SHAP
                explainer = shap.TreeExplainer(modelperm)
                shap_values = explainer(X_train_fold)

                if isinstance(shap_values, shap.Explanation):
                    shap_values_positive_class = shap_values.values
                elif isinstance(shap_values, list):
                    if len(shap_values) == 2:
                        shap_values_positive_class = shap_values[1]
                    else:
                        shap_values_positive_class = shap_values
                elif isinstance(shap_values, np.ndarray):
                    shap_values_positive_class = shap_values
                else:
                    print("Error: Unexpected SHAP values structure.")
                    shap_values_positive_class = np.zeros((X_train_fold.shape[0], X_train_fold.shape[1]))

                shap_df = pd.DataFrame(shap_values_positive_class, columns=X_train_fold.columns)
                shap_df_list.append(shap_df)
    
    print(f"Number of SHAP DataFrames collected: {len(shap_df_list)}")
    if len(shap_df_list) == 0:
        print("No SHAP DataFrames were added to shap_df_list. Check SHAP computation steps.")
    
    # --------------------------------------------------------------------
    # Below is the same logic for thresholds, p-values, and final selection
    # --------------------------------------------------------------------
    
    # Total number of runs (sum of folds if CV_FOLDS>0, else # of seeds)
    total_runs = len(shap_df_list)
    
    # Initialize selection count dictionaries
    feature_selection_counts_pos = {}
    feature_selection_counts_neg = {}
    
    # Iterate over each SHAP DataFrame
    for shap_df in shap_df_list:
        # Identify permuted and original features
        permuted_features = [col for col in shap_df.columns if '_perm' in col]
        original_features = [col for col in shap_df.columns if '_perm' not in col]
        
        if not permuted_features:
            print("No permuted features found in this SHAP DataFrame. Skipping to the next.")
            continue
        if not original_features:
            print("No original features found in this SHAP DataFrame. Skipping to the next.")
            continue
        
        # Mean SHAP for permuted features (positive)
        shap_df_positive = shap_df[permuted_features][shap_df[permuted_features] > 0]
        mean_shap_permuted_pos = shap_df_positive.mean()
        
        # Mean SHAP for permuted features (negative)
        shap_df_negative = shap_df[permuted_features][shap_df[permuted_features] < 0]
        mean_shap_permuted_neg = shap_df_negative.mean().abs()
        
        # Calculate thresholds
        if mean_shap_permuted_pos.empty:
            print("No positive SHAP values found in permuted features. Setting positive threshold to 0.")
            permuted_threshold_pos = 0
        else:
            permuted_threshold_pos = mean_shap_permuted_pos.quantile(SHAP_PERM_PERCENTILE_THRESHOLD)
        
        if mean_shap_permuted_neg.empty:
            print("No negative SHAP values found in permuted features. Setting negative threshold to 0.")
            permuted_threshold_neg = 0
        else:
            permuted_threshold_neg = mean_shap_permuted_neg.quantile(SHAP_PERM_PERCENTILE_THRESHOLD)
        
        print(f"permuted_threshold_neg: -{permuted_threshold_neg}, permuted_threshold_pos: {permuted_threshold_pos}")
        
        # Mean SHAP for original features
        mean_shap_original = shap_df[original_features].mean()
        print(f"mean_shap_original min: {mean_shap_original.min()}, max: {mean_shap_original.max()}")
        
        # Identify which features exceed thresholds
        selected_features_pos = mean_shap_original[mean_shap_original > permuted_threshold_pos].index.tolist()
        selected_features_neg = mean_shap_original[mean_shap_original < -permuted_threshold_neg].index.tolist()
        
        print(f"selected_features_pos: {selected_features_pos}")
        print(f"selected_features_neg: {selected_features_neg}")
        
        # Update counts
        for feature in selected_features_pos:
            feature_selection_counts_pos[feature] = feature_selection_counts_pos.get(feature, 0) + 1
        for feature in selected_features_neg:
            feature_selection_counts_neg[feature] = feature_selection_counts_neg.get(feature, 0) + 1
    
    # Compute selection frequencies
    feature_selection_frequencies_pos = {
        feature: count / total_runs 
        for feature, count in feature_selection_counts_pos.items()
    }
    feature_selection_frequencies_neg = {
        feature: count / total_runs 
        for feature, count in feature_selection_counts_neg.items()
    }

    permuted_feature_frequencies_pos = [count / total_runs for count in feature_selection_counts_pos.values()]
    permuted_feature_frequencies_neg = [count / total_runs for count in feature_selection_counts_neg.values()]
    
    # p-values for positive side
    p_values_pos = []
    for feature, freq in feature_selection_frequencies_pos.items():
        p_value = np.mean([pfreq >= freq for pfreq in permuted_feature_frequencies_pos])
        p_values_pos.append((feature, p_value))
    
    # p-values for negative side
    p_values_neg = []
    for feature, freq in feature_selection_frequencies_neg.items():
        p_value = np.mean([pfreq >= freq for pfreq in permuted_feature_frequencies_neg])
        p_values_neg.append((feature, p_value))
    
    p_values_df_pos = pd.DataFrame(p_values_pos, columns=['Feature', 'p_value_pos'])
    p_values_df_neg = pd.DataFrame(p_values_neg, columns=['Feature', 'p_value_neg'])
    
    p_values_df = pd.merge(p_values_df_pos, p_values_df_neg, on='Feature', how='outer').fillna(0)
    
    # Multiple test correction
    if not p_values_df_pos.empty:
        adjusted_p_values_pos = multipletests(p_values_df['p_value_pos'], method='fdr_bh')[1]
        p_values_df['adjusted_p_value_pos'] = adjusted_p_values_pos
    else:
        p_values_df['adjusted_p_value_pos'] = 1.0
        
    if not p_values_df_neg.empty:
        adjusted_p_values_neg = multipletests(p_values_df['p_value_neg'], method='fdr_bh')[1]
        p_values_df['adjusted_p_value_neg'] = adjusted_p_values_neg
    else:
        p_values_df['adjusted_p_value_neg'] = 1.0
    
    # Save p-values
    p_values_file_path = os.path.join(output_dir, f'p_values_results_separate_{modellabel}.tsv')
    p_values_df.to_csv(p_values_file_path, sep='\t', index=False)
    print(f"P-values saved to {p_values_file_path}")
    
    print("Top 10 features based on adjusted p-values:")
    print(p_values_df.sort_values(by=['adjusted_p_value_pos', 'adjusted_p_value_neg']).head(10))
    
    # Final feature selection
    selected_features_pos = p_values_df[p_values_df['adjusted_p_value_pos'] <= SIGNIFICANCE_LEVEL]['Feature'].tolist()
    selected_features_neg = p_values_df[p_values_df['adjusted_p_value_neg'] <= SIGNIFICANCE_LEVEL]['Feature'].tolist()
    selected_features = list(set(selected_features_pos + selected_features_neg))

    print("\nSelected Features based on Permutation Knockoff Analysis:")
    print(selected_features)

    # Save selected features
    selected_features_df = pd.DataFrame(selected_features, columns=['Selected_Features'])
    selected_features_file = os.path.join(output_dir, f'selected_features_{modellabel}.csv')
    selected_features_df.to_csv(selected_features_file, index=False)
    print(f"Selected features saved to {selected_features_file}")
    
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected Features:")
    print(selected_features)
    
    return selected_features, metrics_list, p_values_df



def perform_cross_validation(
    X_train,
    y_train,
    cv_params,
    n_splits=5,
    random_seed=42,
    modellabel="model",
    output_dir="."
):
    """
    Example cross-validation function that uses CatBoost. 
    Returns aggregated precision, recall, and F1 across folds.
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    all_precisions = []
    all_recalls = []
    all_f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val   = X_train.iloc[val_idx]
        y_fold_val   = y_train.iloc[val_idx]

        model_cv = CatBoostClassifier(**cv_params)
        # Here we set eval_set for early stopping if desired
        model_cv.fit(
            X_fold_train,
            y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            verbose=0
        )

        # Predictions on the fold's hold-out portion
        y_val_pred = model_cv.predict(X_fold_val)
        
        # Evaluate fold metrics
        prec = precision_score(y_fold_val, y_val_pred, zero_division=0)
        rec  = recall_score(y_fold_val,  y_val_pred, zero_division=0)
        f1   = f1_score(y_fold_val,      y_val_pred, zero_division=0)

        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1_scores.append(f1)

        print(f"[Fold {fold_idx}/{n_splits}] Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    cv_metrics = {
        'mean_precision': np.mean(all_precisions),
        'std_precision': np.std(all_precisions),
        'mean_recall': np.mean(all_recalls),
        'std_recall': np.std(all_recalls),
        'mean_f1': np.mean(all_f1_scores),
        'std_f1': np.std(all_f1_scores),
    }
    return cv_metrics



def prob_calibration(
    model,
    X_val_final,
    y_val_binary,
    X_test,
    y_test,
    y_test_binary,
    selected_feature_names,
    mediumid,
    modellabel,
    RANDOM_SEED=42,
    output_dir=".",
    verbose=True
):
    """
    Calibrate model probabilities using isotonic regression and evaluate performance.

    This function performs the following steps:
    1. Predicts probabilities on the validation set.
    2. Calibrates the model using isotonic regression based on validation probabilities.
    3. Saves the calibration model.
    4. Predicts probabilities on the test set.
    5. Applies calibration to test set probabilities.
    6. Computes and prints various evaluation metrics for both calibrated and uncalibrated probabilities.
    7. Plots and saves calibration curves.

    Parameters:
    - model: Trained classifier with a `predict_proba` method.
    - X_val_final (pd.DataFrame): Validation feature set.
    - y_val_final (pd.Series or pd.DataFrame): Validation labels.
    - X_test (pd.DataFrame): Test feature set.
    - y_test (pd.Series or pd.DataFrame): Test labels.
    - selected_feature_names (list of str): List of feature names to use from X_test.
    - mediumid (str or int): Identifier for the positive class (e.g., 'Glucose').
    - modellabel (str): Label to include in output filenames.
    - RANDOM_SEED (int, default=42): Seed for reproducibility.
    - output_dir (str, default="."): Directory to save output files.
    - verbose (bool, default=True): If True, prints progress and metrics.

    Returns:
    - results (dict): Dictionary containing calibration metrics and file paths.
    """

    print("STEP prob_calibration")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1: Predict probabilities on the validation set
    if verbose:
        print("Predicting probabilities on the validation set...")
    val_pred_proba = model.predict_proba(X_val_final)[:, 1]

    # STEP 2: Calibrate the model using isotonic regression
    if verbose:
        print("Calibrating probabilities using isotonic regression...")
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(val_pred_proba, y_val_binary)

    # STEP 3: Save the calibration model
    iso_reg_filename = f'iso_reg_calibration_model_{modellabel}.pkl'
    iso_reg_path = os.path.join(output_dir, iso_reg_filename)
    joblib.dump(iso_reg, iso_reg_path)
    if verbose:
        print(f"Calibration model saved to '{iso_reg_path}'")

    # STEP 4: Predict probabilities on test data
    if verbose:
        print("Predicting probabilities on the test set...")
    X_test_selected = X_test[selected_feature_names]
    y_test_pred_proba = model.predict_proba(X_test_selected)[:, 1]

    # STEP 5: Calibrate the test set probabilities
    if verbose:
        print("Applying calibration to test set probabilities...")
    calibrated_test_pred_proba = iso_reg.transform(y_test_pred_proba)

    # STEP 6: Compute test metrics using calibrated probabilities
    if verbose:
        print("\nEvaluating calibrated probabilities: "+str(calibrated_test_pred_proba))
    test_accuracy = accuracy_score(y_test_binary, (calibrated_test_pred_proba >= 0.5).astype(int))
    test_balanced_accuracy = balanced_accuracy_score(y_test_binary, (calibrated_test_pred_proba >= 0.5).astype(int))
    test_classification_report = classification_report(y_test_binary, (calibrated_test_pred_proba >= 0.5).astype(int))
    test_auc_roc = roc_auc_score(y_test_binary, calibrated_test_pred_proba)

    if verbose:
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Balanced Accuracy: {test_balanced_accuracy}")
        print("\nTest Classification Report:\n", test_classification_report)
        print(f"Test AUC-ROC: {test_auc_roc}")

    # Calculate calibration metrics for calibrated probabilities
    brier = brier_score_loss(y_test_binary, calibrated_test_pred_proba)
    logloss_calibrated = log_loss(y_test_binary, calibrated_test_pred_proba)

    if verbose:
        print(f"Brier score (Calibrated): {brier}")
        print(f"Log loss (Calibrated): {logloss_calibrated}")

    # STEP 7: Compute test metrics using uncalibrated probabilities
    if verbose:
        print("\nEvaluating uncalibrated probabilities:")
    test_accuracy_uncalibrated = accuracy_score(y_test_binary, (y_test_binary >= 0.5).astype(int))
    test_balanced_accuracy_uncalibrated = balanced_accuracy_score(y_test_binary, (y_test_pred_proba >= 0.5).astype(int))
    test_classification_report_uncalibrated = classification_report(y_test_binary, (y_test_pred_proba >= 0.5).astype(int))
    test_auc_roc_uncalibrated = roc_auc_score(y_test_binary, y_test_pred_proba)

    if verbose:
        print(f"Uncalibrated Test Accuracy: {test_accuracy_uncalibrated}")
        print(f"Uncalibrated Test Balanced Accuracy: {test_balanced_accuracy_uncalibrated}")
        print("\nUncalibrated Test Classification Report:\n", test_classification_report_uncalibrated)
        print(f"Uncalibrated Test AUC-ROC: {test_auc_roc_uncalibrated}")

    # Calculate calibration metrics for uncalibrated probabilities
    brier_uncalibrated = brier_score_loss(y_test_binary, y_test_pred_proba)
    logloss_uncalibrated = log_loss(y_test_binary, y_test_pred_proba)

    if verbose:
        print(f"Brier score (Uncalibrated): {brier_uncalibrated}")
        print(f"Log loss (Uncalibrated): {logloss_uncalibrated}")

    # STEP 8: Compare Brier scores and log loss
    if verbose:
        print(f"\nComparison of Calibration Metrics:")
        print(f"Calibrated Brier score: {brier}")
        print(f"Uncalibrated Brier score: {brier_uncalibrated}")
        print(f"Calibrated Log loss: {logloss_calibrated}")
        print(f"Uncalibrated Log loss: {logloss_uncalibrated}")

        if logloss_uncalibrated > logloss_calibrated:
            print("CALIBRATED PROB: Calibrated probabilities are more reliable.")
        else:
            print("UNCALIBRATED PROB: Uncalibrated probabilities are more reliable.")

    # STEP 9: Compute calibration curves
    if verbose:
        print("\nComputing calibration curves...")
    prob_true_uncalibrated, prob_pred_uncalibrated = calibration_curve(y_test_binary, y_test_pred_proba, n_bins=10)
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(y_test_binary, calibrated_test_pred_proba, n_bins=10)

    # STEP 10: Plot the calibration curves
    if verbose:
        print("Plotting calibration curves...")
    plt.figure(figsize=(10, 5))
    plt.plot(prob_pred_uncalibrated, prob_true_uncalibrated, marker='o', label='Uncalibrated')
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title('Calibration Curves')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.tight_layout()
    calibration_plot_filename = f'calibration_curves_{modellabel}.png'
    calibration_plot_path = os.path.join(output_dir, calibration_plot_filename)
    plt.savefig(calibration_plot_path)
    plt.close()  # Close the figure to free memory
    if verbose:
        print(f"Calibration curves saved to '{calibration_plot_path}'")

    # Prepare results to return
    results = {
        'calibrated_metrics': {
            'accuracy': test_accuracy,
            'balanced_accuracy': test_balanced_accuracy,
            'classification_report': test_classification_report,
            'auc_roc': test_auc_roc,
            'brier_score': brier,
            'log_loss': logloss_calibrated
        },
        'uncalibrated_metrics': {
            'accuracy': test_accuracy_uncalibrated,
            'balanced_accuracy': test_balanced_accuracy_uncalibrated,
            'classification_report': test_classification_report_uncalibrated,
            'auc_roc': test_auc_roc_uncalibrated,
            'brier_score': brier_uncalibrated,
            'log_loss': logloss_uncalibrated
        },
        'calibration_model_path': iso_reg_path,
        'calibration_plot_path': calibration_plot_path,
        'calibrated_test_pred_proba': calibrated_test_pred_proba,
        'uncalibrated_test_pred_proba': y_test_pred_proba
    }

    return results, iso_reg




def train_final_model(
    X_train_full,
    y_train_full_binary,
    X_val,
    y_val,
    y_val_binary,
    X_test,
    y_test_binary,
    selected_feature_names,
    modellabel,
    formatted_date,
    iterations_final,
    random_seed,
    cv_folds,
    output_dir=".", 
    GPU=False
):
    """
    Trains the final CatBoost model on (train+val) data and evaluates it on test data.
    Also performs cross-validation on the combined (train+val) set before final training,
    so that CV metrics match better with final model performance.

    Parameters:
        X_train_full (pd.DataFrame): Full training features.
        y_train_full_binary (pd.Series): Full training binary labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels (NOT used in final separate holdout sense).
        y_val_binary (pd.Series): Same as y_val but binary form.
        X_test (pd.DataFrame): Test features.
        y_test_binary (pd.Series): Test binary labels.
        selected_feature_names (list): List of selected feature names.
        modellabel (str): Label for the model to use in output file naming.
        formatted_date (str): Date string for file naming.
        iterations_final (int): Number of iterations for the final model.
        random_seed (int): Random seed for reproducibility.
        cv_folds (int): Number of cross-validation folds.
        output_dir (str): Directory to save output files.
        GPU (bool): Whether to use GPU training.

    Returns:
        tuple: (test_metrics, cv_metrics, model, X_trainval_selected)
    """

    print("STEP train_final_model_unified_cv")

    # -------------------------------------------------------
    # 1) Combine train + val sets for cross-validation + final training
    # -------------------------------------------------------
    X_trainval = pd.concat([X_train_full, X_val], axis=0)
    y_trainval = pd.concat([y_train_full_binary, y_val_binary], axis=0)

    # Keep only selected features
    X_train_selected = X_train_full[selected_feature_names]
    X_val_selected = X_val[selected_feature_names]
    X_trainval_selected = X_trainval[selected_feature_names]
    X_test_selected = X_test[selected_feature_names]

    print(f"X_trainval.shape: {X_trainval.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"X_trainval_selected.shape: {X_trainval_selected.shape}")
    print(f"X_test_selected.shape: {X_test_selected.shape}")

    # -------------------------------------------------------
    # 2) Compute class weights (optional)
    # -------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_trainval),
        y=y_trainval
    )
    class_weights_dict = dict(zip(np.unique(y_trainval), class_weights))
    print("Class Weights final (train+val):", class_weights_dict)

    # -------------------------------------------------------
    # 3) Define CatBoost parameters
    # -------------------------------------------------------
    final_mod_params = {
        'iterations': iterations_final,
        'depth': 4,
        'learning_rate': 0.05,
        'l2_leaf_reg': 4,
        'bagging_temperature': 1,
        'random_strength': 6,
        'loss_function': 'Logloss',
        'random_seed': random_seed,
        'verbose': 100,
        # We'll disable early_stopping_rounds for final training on entire dataset
        # but you can keep it if you want:
        'early_stopping_rounds': 50,
        'use_best_model': True,
        # 'class_weights': class_weights_dict,  # Optionally re-enable
    }

    if GPU:
        final_mod_params.update({
            'task_type': 'GPU',
            'devices': '0'
        })
    # NOTE: If you do want early stopping, you must pass an eval_set. 
    # Because we're training on the entire train+val, there's no separate val left,
    # so you could hold out a small portion or skip early stopping.

    # -------------------------------------------------------
    # 4) (Optional) Perform cross-validation on the combined set
    # -------------------------------------------------------
    cv_metrics = None
    if cv_folds > 0:
        print(f"\nPerforming {cv_folds}-fold Cross-Validation on Train+Val set...\n")
        cv_metrics = perform_cross_validation(
            X_train=X_trainval_selected,
            y_train=y_trainval,
            cv_params=final_mod_params,
            n_splits=cv_folds,
            random_seed=random_seed,
            modellabel=modellabel,
            output_dir=output_dir,
        )
        print("\nAggregated Cross-Validation Metrics (Train+Val):")
        print(f" Precision: {cv_metrics['mean_precision']:.4f} ± {cv_metrics['std_precision']:.4f}")
        print(f" Recall:    {cv_metrics['mean_recall']:.4f} ± {cv_metrics['std_recall']:.4f}")
        print(f" F1-Score:  {cv_metrics['mean_f1']:.4f} ± {cv_metrics['std_f1']:.4f}\n")

    # -------------------------------------------------------
    # 5) Train final model on the entire (train+val) dataset
    # -------------------------------------------------------
    print("Training final model on the entire Train+Val set...")
    train_data_final = Pool(data=X_train_selected, label=y_train_full_binary)
    val_data_final = Pool(data=X_val_selected, label=y_val_binary)
    model = CatBoostClassifier(**final_mod_params)

    # NOTE: If you do not want to do early stopping here, remove 'eval_set'.
    # Alternatively, hold out a small slice of data as the eval_set. 
    model.fit(train_data_final, eval_set=val_data_final, verbose=100)

    # -------------------------------------------------------
    # 6) Evaluate on test data
    # -------------------------------------------------------
    y_test_pred = model.predict(X_test_selected)
    y_test_pred_proba = model.predict_proba(X_test_selected)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test_binary, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test_binary, y_test_pred),
        "auc_roc": roc_auc_score(y_test_binary, y_test_pred_proba),
        "classification_report": classification_report(y_test_binary, y_test_pred, output_dict=True),
    }

    print("TEST Metrics:")
    print(f" Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f" Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f" AUC-ROC:           {test_metrics['auc_roc']:.4f}")
    print("\nClassification Report:\n", 
          classification_report(y_test_binary, y_test_pred))

    # Save classification report to TSV file
    classification_report_path = f"{output_dir}/classification_report_selected_{modellabel}.tsv"
    report_df = pd.DataFrame(test_metrics['classification_report']).transpose()
    report_df.to_csv(classification_report_path, sep='\t', index=True)
    print(f"Classification report saved to {classification_report_path}")

    # -------------------------------------------------------
    # 7) Save final model
    # -------------------------------------------------------
    model_file_path = f"{output_dir}/catboost_model_{modellabel}_{formatted_date}.cbm"
    model.save_model(model_file_path)
    print(f"Final model saved to {model_file_path}")

    # Return test and cross-validation metrics, plus the model
    return test_metrics, cv_metrics, model, X_train_selected


def final_shap_analysis(
    model,
    X_train_final,
    y_train_final,
    selected_feature_names,
    mediumid,
    modellabel,
    output_dir=".",
    topn=40,
    verbose=True
):
    """
    Perform SHAP analysis on a trained model, identify top features, and save relevant plots and data.

    This function performs the following steps:
    1. Calculates SHAP values for the training data.
    2. Determines the mean SHAP values for each feature.
    3. Identifies the top N features based on absolute mean SHAP values.
    4. Plots and saves SHAP summary plots for the top features.
    5. Saves the feature importance data to CSV files.
    6. Prints relevant statistics and information based on verbosity.

    Parameters:
    - model: Trained CatBoost model with a `predict_proba` method.
    - X_train_final (pd.DataFrame): Feature set used for training.
    - y_train_final (pd.Series or pd.DataFrame): Labels corresponding to `X_train_final`.
    - selected_feature_names (list of str): List of feature names to use from `X_test` for prediction.
    - mediumid (str or int): Identifier for the positive class (e.g., 'Glucose').
    - modellabel (str): Label to include in output filenames for identification.
    - output_dir (str, default="."): Directory to save output files (plots and CSVs).
    - topn (int, default=40): Number of top features to select based on SHAP values.
    - verbose (bool, default=True): If True, prints detailed information during processing.

    Returns:
    - shap_summary_df (pd.DataFrame): DataFrame containing mean SHAP values for all features.
    - top_features_df (pd.DataFrame): DataFrame containing mean SHAP values for the top N features.
    """

    print("STEP final_shap_analysis")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Create Pool object for SHAP values calculation
    train_data_for_shap = Pool(data=X_train_final, label=y_train_final)
    
    if verbose:
        print("Calculating SHAP values...")
    
    # STEP 2: Create an explainer object
    explainer = shap.TreeExplainer(model)
    if verbose:
        print(f"Final Type of explainer.expected_value: {type(explainer.expected_value)}")
        print(f"Final explainer.expected_value: {explainer.expected_value}")
    
    # STEP 3: Calculate SHAP values
    shap_values = explainer.shap_values(X_train_final)
    
    # STEP 4: Determine the structure of SHAP values
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            # Binary classification with separate SHAP values for each class
            shap_values_positive_class = shap_values[1]
            if verbose:
                print("SHAP values are a list with two elements (binary classification).")
        else:
            # Multi-class classification or other configurations
            shap_values_positive_class = shap_values
            if verbose:
                print(f"SHAP values are a list with {len(shap_values)} elements (multi-class or other).")
    elif isinstance(shap_values, np.ndarray):
        # Binary classification where shap_values is a single array
        shap_values_positive_class = shap_values
        if verbose:
            print("SHAP values are a single ndarray (binary classification).")
    else:
        if verbose:
            print("Error: Unexpected SHAP values structure.")
        shap_values_positive_class = np.zeros((X_train_final.shape[0], X_train_final.shape[1]))
    
    # STEP 5: Calculate the mean SHAP values across all samples
    mean_shap_values = np.mean(shap_values_positive_class, axis=0)
    
    # STEP 6: Get base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]  # For binary classification, take the positive class
    if verbose:
        print("Base value (expected value):", base_value)
    
    # STEP 7: Get model outputs (raw margins)
    try:
        model_outputs = model.predict(train_data_for_shap, prediction_type='RawFormulaVal')
        if verbose:
            print("Model outputs (raw margins) statistics:")
            print(pd.Series(model_outputs).describe())
    except Exception as e:
        if verbose:
            print(f"Error obtaining model outputs: {e}")
        model_outputs = np.array([])
    
    # STEP 8: Retrieve feature names
    try:
        feature_names = model.feature_names_
    except AttributeError:
        feature_names = X_train_final.columns.tolist()
        if verbose:
            print("Model does not have 'feature_names_'. Using DataFrame column names.")
    
    # STEP 9: Create a DataFrame of mean SHAP values
    shap_summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP_Value': mean_shap_values
    })
    
    # STEP 10: Calculate the absolute mean SHAP values
    shap_summary_df['Abs_Mean_SHAP_Value'] = shap_summary_df['Mean_SHAP_Value'].abs()
    
    # STEP 11: Sort the features by absolute mean SHAP value in descending order
    shap_summary_df = shap_summary_df.sort_values('Abs_Mean_SHAP_Value', ascending=False)
    
    # STEP 12: Select the top N features
    top_features_df = shap_summary_df.head(topn)
    
    # STEP 13: Extract the list of top feature names
    top_features = top_features_df['Feature'].tolist()
    
    # STEP 14: Get the indices of the top features in the training data
    top_feature_indices = [X_train_final.columns.get_loc(f) for f in top_features]
    
    # STEP 15: Subset the SHAP values to include only the top features
    shap_values_top = shap_values_positive_class[:, top_feature_indices]
    
    # STEP 16: Subset the feature matrix to include only the top features
    X_top = X_train_final[top_features]
    
    # STEP 17: Create a plot for SHAP summary
    plt.figure(figsize=(12, 8))  # Adjust the figsize as needed
    
    # Check if there are any features to plot
    if not top_features_df.empty:
        if verbose:
            print("Generating SHAP summary plot for top features...")
        # Generate SHAP summary plot
        shap.summary_plot(
            shap_values_top,                # SHAP values for the top features
            features=X_top,                 # Feature matrix for the top features
            feature_names=top_features,     # Names of the top features
            plot_type="dot",
            color_bar=True,
            color="coolwarm",
            show=False                      # Prevent SHAP from displaying the plot immediately
        )
        
        # Set the plot title to the model label
        plt.title(f"SHAP Summary Plot for {modellabel}", fontsize=16)
        
        # Remove any existing legends (if present)
        ax = plt.gca()  # Get current axes
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # Define filenames for the plots
        figure_name_base = f"top_{topn}_features_catboost_model_{modellabel}"
        pdf_path = os.path.join(output_dir, f"{figure_name_base}.pdf")
        png_path = os.path.join(output_dir, f"{figure_name_base}.png")
        
        # Save the figure as PDF and PNG
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(png_path, format='png', bbox_inches='tight')
        if verbose:
            print(f"SHAP summary plot saved to '{pdf_path}' and '{png_path}'")
        
        # Save the top features to a CSV file
        top_features_csv = os.path.join(output_dir, f"{modellabel}_top{topn}_feature_importance.csv")
        top_features_df.to_csv(top_features_csv, index=False)
        if verbose:
            print(f"Top {topn} features saved to '{top_features_csv}'")
    else:
        if verbose:
            print("No features to plot.")
    
    # STEP 18: Print the top features with their SHAP values
    if verbose:
        print("\nTop features with their mean SHAP values:")
        print(top_features_df[['Feature', 'Mean_SHAP_Value']])
    
    # STEP 19: Filter and display features with negative SHAP values
    negative_features_df = shap_summary_df[shap_summary_df['Mean_SHAP_Value'] < 0]
    
    if not negative_features_df.empty and verbose:
        print(f"Features with negative SHAP values: {negative_features_df.shape}")
        #for index, row in negative_features_df.iterrows():
        #    print(f"{row['Feature']}: {row['Mean_SHAP_Value']}")
    elif verbose:
        print("\nNo features have negative SHAP values.")
    
    # STEP 20: Save the full SHAP values DataFrame
    full_shap_csv = os.path.join(output_dir, f"{modellabel}_feature_importance_full.csv")
    shap_summary_df.to_csv(full_shap_csv, index=False)
    if verbose:
        print(f"\nFull SHAP feature importance saved to '{full_shap_csv}'")
    
    return shap_summary_df, top_features_df


def process_predictions(
    data_df_orig, 
    data_df_clean, 
    data_nodes, 
    model, 
    iso_reg, 
    mediumid, 
    modellabel, 
    cutoff, 
    EC_annot_df=None,
    RHEA_annot_df=None,
    tax_annot_df=None,
    egg_annot_df=None,
    traithop_annot_df=None,
    formatted_date="",
    y=None
):
    """
    Process predictions for taxa-media relationships, including calibration, label assignment, 
    and data filtering, by re-running the same pipeline steps on data_df_nomed that were 
    run on data_df_clean. This ensures columns are aligned for model inference.

    Parameters:
        data_df_orig (pd.DataFrame): Original pivoted DataFrame (with 'medium' column).
        data_df_clean (pd.DataFrame): Cleaned + annotated + pivoted training DataFrame.
        data_nodes (pd.DataFrame): Data nodes for label/name mapping.
        model (CatBoostClassifier): Trained CatBoost model.
        iso_reg (IsotonicRegression): Fitted isotonic regression calibrator.
        mediumid (str): Medium ID for predictions (e.g. 'Glucose').
        modellabel (str): Label for the current model.
        cutoff (float): Probability cutoff for filtering predictions.
        EC_annot_df (pd.DataFrame, optional): Crosstab of NCBITaxon vs. EC expansions.
        RHEA_annot_df (pd.DataFrame, optional): Crosstab of NCBITaxon vs. RHEA expansions.
        tax_annot_df (pd.DataFrame, optional): Single DataFrame containing taxonomy annotations.
        formatted_date (str, optional): Date/time string for naming output files.
        y (pd.Series, optional): Original label series for count comparisons.
    
    Returns:
        pd.DataFrame: Merged DataFrame with ratio counts for predictions vs. training.
    """

    print("STEP process_predictions")

    # 1) Identify rows WITHOUT medium
    data_df_nomed = data_df_orig[data_df_orig['medium'].isna()].copy()

    # 2) If we have a taxonomy DataFrame, join that. Otherwise, if we have EC + RHEA, join those.
    if tax_annot_df is not None:
        print(type(tax_annot_df))
        # Perform the taxonomy join
        data_df_nomed = data_df_nomed.join(
            tax_annot_df, how='left', lsuffix='_trait', rsuffix='_annot'
        )
    elif (EC_annot_df is not None) and (RHEA_annot_df is not None):
        # Join EC
        EC_columns = [col for col in EC_annot_df.columns if col.startswith('EC:')]
        EC_annot_sub = EC_annot_df[EC_columns]  # subset columns if needed

        # Join EC
        data_df_nomed = data_df_nomed.join(
            EC_annot_sub, how='left', lsuffix='_trait', rsuffix='_annot'
        )

        # Join RHEA with a new (or no) suffix
        data_df_nomed = data_df_nomed.join(
            RHEA_annot_df, how='left', lsuffix='_trait2', rsuffix='_annot'
        )

    # else: no annotation DataFrames provided, so skip joins

    # 3) Because data_df_nomed does not have 'medium' defined (these are no‐medium rows),
    #    add a placeholder column so removal steps match data_df_clean's pipeline.
    data_df_nomed['medium'] = np.nan

    # 4) Re-run the same cleaning pipeline that was applied in the application code
    #    to data_df_clean. This ensures data_df_nomed ends up with the same column set.
    data_df_nomed = remove_singleton_row_and_col(
        data_df_clean=data_df_nomed,
        exclude_cols=['medium'],
        verbose=False
    )
    data_df_nomed = remove_identical_pattern(
        data_df_clean=data_df_nomed,
        modellabel=modellabel,
        output_dir=".",
        verbose=False
    )

    # 5) Now we align columns exactly with data_df_clean
    #    so that the model sees the same features. 
    data_df_nomed = data_df_nomed.reindex(columns=data_df_clean.columns, fill_value=0)

    # 6) Create a copy for inference and drop 'medium' before passing to the model
    X2 = data_df_nomed.copy(deep=True)
    X2.drop(columns=['medium'], inplace=True)
    # If your pipeline renames columns, do it here:
    # X2, id_to_unique_name = name_columns(data_nodes, X2, modellabel, verbose=False)

    # 7) Predict probabilities on the new data
    probabilities = model.predict_proba(X2)
    calibrated_probabilities = iso_reg.transform(probabilities[:, 1])

    # 8) Build a predictions DataFrame
    predictions = pd.DataFrame({
        'row_id': X2.index,
        'prob': probabilities[:, 1],
        'calibrated_prob': calibrated_probabilities,
        'predicted_class': (calibrated_probabilities >= 0.5).astype(int)
    })
    predictions['media_id_label'] = predictions['predicted_class'].apply(
        lambda x: f'medium:{mediumid}' if x == 1 else 'Other'
    )

    # 9) Sort and merge with descriptive labels
    predictions = predictions.sort_values(by='prob', ascending=False)
    predictions = predictions.merge(
        data_nodes[['id', 'name']], left_on='row_id', right_on='id', how='left'
    ).drop(columns='id').rename(columns={'name': 'row_id_label'})
    predictions = predictions.merge(
        data_nodes[['id', 'name']], left_on='media_id_label', right_on='id', how='left'
    ).drop(columns='id').rename(columns={'name': 'media_id_descr'})

    # 10) Helper functions for linking
    def extract_ncbi_id(row_id):
        return str(row_id).split(':')[-1]
    def extract_media_id(media_id):
        return str(media_id).split(':')[-1]
    def extract_media_link(media_id_label):
        if media_id_label == 'Other':
            return ''
        return f'https://mediadive.dsmz.de/medium/{extract_media_id(media_id_label)}'

    predictions['ncbi_link'] = (
        'https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=' 
        + predictions['row_id'].apply(extract_ncbi_id)
    )
    predictions['media_link'] = predictions['media_id_label'].apply(extract_media_link)
    new_column_order = [
        'row_id', 'media_id_label', 'prob', 'calibrated_prob',
        'row_id_label', 'media_id_descr', 'ncbi_link', 'media_link'
    ]
    predictions = predictions[new_column_order]

    # 11) Save all predictions
    if formatted_date:
        predictions.to_csv(
            f'pred{formatted_date}__{modellabel}.tsv',
            sep='\t', index=False
        )

    # 12) Filter predictions by probability cutoff & label
    filtered_predictions = predictions[
        ((predictions['prob'] > cutoff) | (predictions['calibrated_prob'] > cutoff)) 
        & (predictions['media_id_label'] == f'medium:{mediumid}')
    ]
    if formatted_date:
        filtered_predictions.to_csv(
            f'taxa_media_filtered_predictions_{modellabel}_{cutoff}.tsv',
            sep='\t', index=False
        )
        filtered_predictions.to_csv(
            f'pred{formatted_date}__{modellabel}__prob_{cutoff}.tsv',
            sep='\t', index=False
        )

    # 13) Compare new predictions vs training distribution (optional)
    if y is not None:
        value_counts_pred = predictions[predictions['prob'] > 0.85]['media_id_label'].value_counts()
        df_pred = value_counts_pred.reset_index()
        df_pred.columns = ['media_id', 'pred_count']  

        value_counts_train = y.value_counts()
        df_train = value_counts_train.reset_index()
        df_train.columns = ['media_id', 'train_count']

        merged_df = pd.merge(df_pred, df_train, on='media_id', how='inner')
        merged_df['ratio'] = merged_df['pred_count'] / merged_df['train_count']

        print(merged_df)
        return merged_df
    else:
        return predictions  # or return whatever is relevant if y is not provided


def process_physical_parameter_data_pairs(data, parameter_name, modellabel, output_dir=".", verbose=True):
    """
    Process data pairs to extract physical parameter-related relationships (oxygen, temperature, pH, salinity).
    
    Parameters:
        data (pd.DataFrame): Input edges data
        parameter_name (str): Name of the parameter (e.g., 'oxygen', 'temperature', 'pH', 'salinity')
        modellabel (str): Label for the current model
        output_dir (str): Output directory for saving results
        verbose (bool): Whether to print verbose output
        
    Returns:
        tuple: (data_pairs_clean, data_pairs_rest) - parameter data and other trait data
    """
    if verbose:
        print(f"STEP process_{parameter_name}_data_pairs")
    
    # Filter for taxa-parameter relationships
    data_pairs_clean = data[data['subject'].str.contains('NCBITaxon:|strain:')]
    # Special handling for pH which uses pH_range, pH_opt, pH_delta instead of just pH:
    if parameter_name.lower() == 'ph':
        data_pairs_clean = data_pairs_clean[data_pairs_clean['object'].str.contains('pH_range:|pH_opt:|pH_delta:', case=False)]
    else:
        data_pairs_clean = data_pairs_clean[data_pairs_clean['object'].str.contains(f'{parameter_name}:', case=False)]
    data_pairs_clean.to_csv(f"{output_dir}/NCBITaxon_to_{parameter_name}_v2.tsv", sep="\t", header=True, index=False)
    
    # Add Value column for pivot table operations
    data_pairs_clean = data_pairs_clean.copy()
    data_pairs_clean['Value'] = 1
    
    if verbose:
        print(f"{parameter_name.capitalize()} data pairs shape: {data_pairs_clean.shape}")
    
    # Process additional trait data for feature enrichment
    data_pairs_rest_all = data[data['subject'].str.contains('NCBITaxon:|strain:')]
    data_pairs_rest = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('carbon_substrates:')]
    
    # Add various trait categories for physical parameter models
    trait_categories = [
        'pathways:', 'trophic_type:', 'metabolic_type:', 'sporulation:', 'motility:',
        'cell_shape:', 'gram_stain:', 'cell_length:', 'cell_width:', 'pH:',
        'temperature:', 'salinity:', 'oxygen:', 'pathogen:', 'isolation_source:',
        'assay:', 'cell_arrangement:', 'doubling_time:', 'antibiotic_sensitivity:',
        'cell_wall:', 'pigmentation:'
    ]
    
    for category in trait_categories:
        data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains(category)]
        data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    
    # Add Value column for pivot table operations  
    data_pairs_rest['Value'] = 1
    
    if verbose:
        print(f"Additional trait data shape: {data_pairs_rest.shape}")
    
    return data_pairs_clean, data_pairs_rest


def process_physical_parameter_predictions(
    data_df_orig, 
    data_df_clean, 
    data_nodes, 
    model, 
    iso_reg, 
    parameter_name,
    modellabel, 
    cutoff=0.9, 
    formatted_date="",
    output_dir="."
):
    """
    Process physical parameter predictions for taxa, including calibration, label assignment, and filtering.
    
    Parameters:
        data_df_orig (pd.DataFrame): Original pivoted DataFrame (with parameter column).
        data_df_clean (pd.DataFrame): Cleaned + annotated + pivoted training DataFrame.
        data_nodes (pd.DataFrame): Data nodes for label/name mapping.
        model (CatBoostClassifier): Trained CatBoost model.
        iso_reg (IsotonicRegression): Fitted isotonic regression calibrator.
        parameter_name (str): Name of the physical parameter (e.g., 'oxygen', 'temperature', 'pH', 'salinity')
        modellabel (str): Label for the current model.
        cutoff (float): Probability cutoff for filtering predictions.
        formatted_date (str): Formatted timestamp for file naming.
        output_dir (str): Output directory for saving results.
        
    Returns:
        pd.DataFrame: Predictions DataFrame with parameter class assignments.
    """
    print(f"Processing {parameter_name} predictions...")
    
    # 1) Create a version without the parameter column for prediction
    data_df_nomed = data_df_orig.drop(columns=[parameter_name])
    
    # 2) Ensure we have the same columns and processing as training data
    data_df_nomed = remove_singleton_row_and_col(
        data_df_clean=data_df_nomed,
        exclude_cols=[parameter_name],
        verbose=False
    )
    data_df_nomed = remove_identical_pattern(
        data_df_clean=data_df_nomed,
        modellabel=modellabel,
        output_dir=output_dir,
        verbose=False
    )
    
    # 3) Align columns exactly with training data
    data_df_nomed = data_df_nomed.reindex(columns=data_df_clean.columns, fill_value=0)
    
    # 4) Create inference data and drop parameter column
    X2 = data_df_nomed.copy(deep=True)
    X2.drop(columns=[parameter_name], inplace=True)
    
    # 5) Predict probabilities
    probabilities = model.predict_proba(X2)
    
    # Handle both binary and multiclass scenarios
    if probabilities.shape[1] == 2:
        # Binary classification
        prob_positive = probabilities[:, 1]
        predicted_classes = model.predict(X2)
    else:
        # Multiclass - get max probability and predicted class
        prob_positive = np.max(probabilities, axis=1)
        predicted_classes = model.predict(X2)
    
    # 6) Calibrate probabilities
    if hasattr(iso_reg, 'transform'):
        calibrated_probabilities = iso_reg.transform(prob_positive.reshape(-1, 1)).flatten()
    else:
        calibrated_probabilities = iso_reg.predict(prob_positive.reshape(-1, 1)).flatten()
    
    # 7) Build predictions DataFrame
    predictions = pd.DataFrame({
        'row_id': X2.index,
        'prob': prob_positive,
        'calibrated_prob': calibrated_probabilities,
        'predicted_class': predicted_classes
    })
    
    # 8) Sort and merge with descriptive labels
    predictions = predictions.sort_values(by='prob', ascending=False)
    predictions = predictions.merge(
        data_nodes[['id', 'name']], left_on='row_id', right_on='id', how='left'
    ).drop(columns='id').rename(columns={'name': 'row_id_label'})
    
    # 9) Helper functions for linking
    def extract_ncbi_id(row_id):
        return str(row_id).split(':')[-1]
    
    predictions['ncbi_link'] = (
        'https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=' 
        + predictions['row_id'].apply(extract_ncbi_id)
    )
    
    new_column_order = [
        'row_id', 'predicted_class', 'prob', 'calibrated_prob',
        'row_id_label', 'ncbi_link'
    ]
    predictions = predictions[new_column_order]
    
    # 10) Save all predictions
    if formatted_date:
        predictions.to_csv(
            f'{output_dir}/pred{formatted_date}__{parameter_name}_FINAL_v2.tsv',
            sep='\t', index=False
        )
    
    # 11) Filter predictions by probability cutoff
    filtered_predictions = predictions[
        (predictions['prob'] > cutoff) | (predictions['calibrated_prob'] > cutoff)
    ]
    
    if formatted_date:
        filtered_predictions.to_csv(
            f'{output_dir}/pred{formatted_date}__{parameter_name}_prob{cutoff}.tsv',
            sep='\t', index=False
        )
        filtered_predictions.to_csv(
            f'{output_dir}/taxa_{parameter_name}_filtered_predictions_{cutoff}_v2.tsv',
            sep='\t', index=False
        )
    
    return predictions

def remove_singleton_row_and_col(data_df_clean, exclude_cols=None, verbose=False):
    """
    Placeholder for your function that removes rows/columns with a single value.
    """
    # Your existing logic goes here...
    return data_df_clean

def remove_identical_pattern(data_df_clean, modellabel, output_dir=".", verbose=False):
    """
    Placeholder for your function that removes columns with identical patterns.
    """
    # Your existing logic goes here...
    return data_df_clean


def str2bool(v: str) -> bool:
    """
    Convert a string to a boolean.
    Interprets 'false', '0', 'no' as False. Everything else -> True.
    
    Parameters:
        v (str): String value to convert to boolean
        
    Returns:
        bool: True if string represents a true value, False otherwise
    """
    if isinstance(v, bool):
        return v
    if v.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif v.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    else:
        return bool(v)
