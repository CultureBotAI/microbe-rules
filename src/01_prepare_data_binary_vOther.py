#prepare data for binary scoring

"""
Data preparation script for binary classification of microbial growth on specific media.

This script processes KG-Microbe knowledge graph data to create binary feature matrices
for predicting whether microbial taxa can grow in specific media.

Usage:
    python 01_prepare_data_binary.py --medium 65
    python 01_prepare_data_binary.py --medium 514
    python 01_prepare_data_binary.py  # processes both 65 and 514
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import os


src_path = "src"
data_path = "data"

data_link = "https://zenodo.org/records/15106978"


def clean_singleton_and_constant_features(data_df, max_iterations=5):
    """
    Remove singleton and constant features iteratively.

    Removes:
    - Columns with sum <= 1 (singletons)
    - Columns that are all 1's (constants)
    - Rows with sum <= 1 (singletons)
    - Rows that are all 1's (constants)

    Iterates until no more rows/columns are removed or max_iterations reached.

    Parameters:
    - data_df: DataFrame to clean
    - max_iterations: Maximum number of cleaning passes (default: 5)

    Returns:
    - Cleaned DataFrame
    """
    for iteration in range(max_iterations):
        initial_shape = data_df.shape
        print(f"Iteration {iteration + 1}: Shape = {initial_shape}")

        numeric_cols = data_df.select_dtypes(include=['number']).columns

        # Remove columns with sum <= 1
        col_sums = data_df[numeric_cols].sum(axis=0)
        cols_to_drop = data_df[numeric_cols].columns[col_sums <= 1]
        data_df = data_df.drop(columns=cols_to_drop)

        # Remove columns that are all 1's
        numeric_cols = data_df.select_dtypes(include=['number']).columns
        col_sums = data_df[numeric_cols].sum(axis=0)
        n_rows = data_df.shape[0]
        cols_to_drop = data_df[numeric_cols].columns[col_sums == n_rows]
        data_df = data_df.drop(columns=cols_to_drop)

        # Remove rows with sum <= 1
        numeric_cols = data_df.select_dtypes(include=['number']).columns
        row_sums = data_df[numeric_cols].sum(axis=1)
        rows_to_drop = data_df.index[row_sums <= 1]
        data_df = data_df.drop(index=rows_to_drop)

        # Remove rows that are all 1's
        numeric_cols = data_df.select_dtypes(include=['number']).columns
        row_sums = data_df[numeric_cols].sum(axis=1)
        n_cols = len(numeric_cols)
        rows_to_drop = data_df.index[row_sums == n_cols]
        data_df = data_df.drop(index=rows_to_drop)

        # Check if shape changed
        if data_df.shape == initial_shape:
            print(f"Converged after {iteration + 1} iterations")
            break

    print(f"Final shape: {data_df.shape}")
    return data_df


def run_data_prep(mediumid: int) -> None:
    """
    Prepare binary classification data for a specific growth medium.

    This function:
    1. Loads KG-Microbe knowledge graph edges
    2. Filters subject-object pairs for taxa and various feature types
    3. Creates one-hot encoded binary feature matrix
    4. Generates binary classification targets (growth vs. no-growth)
    5. Removes singleton and constant features iteratively
    6. Removes duplicate feature patterns
    7. Saves processed data to compressed TSV

    Parameters:
    - mediumid: Medium identifier (e.g., 65 or 514)

    Returns:
    - None (saves output files to data/ directory)

    Output files:
    - NCBITaxon_to_medium_binary_permute_{mediumid}.tsv
    - taxa_media_classification_binary_permute_{mediumid}.tsv
    - data_df__taxa_to_media__NCBITaxon__binary_permute_{mediumid}.tsv
    - retention_df__binary_permute_{mediumid}.csv
    - taxa_to_media__binary_permute_{mediumid}_data_df_clean.tsv.gz
    """
    print(f"Preparing data for medium {mediumid}")

    if not os.path.exists(data_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(data_path)

    if not os.path.exists(src_path):
        print("Folder with source data does not exists. Please create the folder "+src_path+" and download data from the link into the folder and unpack the file to that folder. Link: "+data_link)
        exit(1)

    if not Path(os.path.join(src_path,"merged-kg_edges.tsv")).is_file():
       print("File with KG Microbe data does not exists. Please create the folder " + src_path + " and download data from the link into the folder and unpack the file to that folder. Link: " + data_link)
       exit(1)

    modellabel = "binary_permute_" + str(mediumid)

    data = pd.read_csv(os.path.join(src_path,"merged-kg_edges.tsv"), header=0, sep="\t", encoding = "ISO-8859-1")

    data_pairs = data[['subject', 'object']].drop_duplicates()

    # Subset the DataFrame based on the substring in subject
    data_pairs_clean = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:')]
    # Subset the DataFrame based on the substring in object
    data_pairs_clean = data_pairs_clean[data_pairs_clean['object'].str.contains('medium:')]
    fname = os.path.join(data_path,f"NCBITaxon_to_medium_{modellabel}.tsv")
    data_pairs_clean.to_csv(fname, sep="\t", header=True, index=False)

    # TODO add closure

    data_pairs_chem = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:')]
    data_pairs_chem = data_pairs_chem[data_pairs_chem['object'].str.contains('CHEBI:')]

    # TODO add closure
    ###
    ### ESPECIALLY for Taxonomy subClassOf >> one hot

    data_pairs_go = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:')]
    data_pairs_go = data_pairs_go[data_pairs_go['object'].str.contains('GO:')]
    data_pairs_go.shape

    # Collect all filtered dataframes in a list for efficient concatenation
    data_pairs_rest_all = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:')]

    # Define patterns to filter
    object_patterns = [
        'carbon_substrates:', 'pathways:', 'trophic_type:', 'production:', 'CAS-RN:',
        'CHEBI:', 'EC:', 'GO:', 'cell_shape:', 'cell_length:', 'cell_width:',
        'motility:', 'sporulation:', 'pigment:', 'gram_stain:', 'gc:',
        'pH_.*:', 'temp_.*:', 'temperature:', 'salinity:', 'NaCl_.*:',
        'oxygen:', 'pathogen:', 'isolation_source:', 'ENVO:', 'UBERON:', 'PO:'
    ]

    # Collect filtered dataframes
    filtered_dfs = []
    for pattern in object_patterns:
        filtered = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains(pattern)]
        if not filtered.empty:
            filtered_dfs.append(filtered)

    # Handle swapped subject/object patterns
    data_pairs_rest_all2 = data_pairs[data_pairs['object'].str.contains('NCBITaxon:|strain:')]
    # Swap 'subject' and 'object' for the filtered DataFrame
    data_pairs_rest_all2_swapped = data_pairs_rest_all2.copy()
    data_pairs_rest_all2_swapped['subject'], data_pairs_rest_all2_swapped['object'] = \
        data_pairs_rest_all2_swapped['object'], data_pairs_rest_all2_swapped['subject']

    swapped_patterns = ['PATO:', 'UBERON:', 'FOODON:', 'CHEBI:', 'ENVO:', 'PO:', 'assay:']
    for pattern in swapped_patterns:
        filtered = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains(pattern)]
        if not filtered.empty:
            filtered_dfs.append(filtered)

    # Single concatenation operation - much more efficient
    data_pairs_rest = pd.concat(filtered_dfs, ignore_index=True)

    data_pairs_rest['Value'] = 1

    # Step 2: Pivot the old DataFrame to form the new DataFrame structure
    data_df = data_pairs_rest.pivot_table(index='subject', columns='object', values='Value', aggfunc='sum',
                                          fill_value=0)
    # Optionally, convert the filled NaN values to integers if they were floats after pivot
    data_df = data_df.astype(int)

    # Copy the original DataFrame
    data_pairs_clean_filtered = data_pairs_clean.copy(deep=True)

    # Ensure that all NCBITaxon: and strain: subject values are considered
    all_subjects = data_pairs_clean_filtered['subject'].unique()

    # Group by 'subject' and list all 'object' (medium)
    taxon_media_groups = data_pairs_clean_filtered.groupby('subject')['object'].agg(list).reindex(all_subjects,
                                                                                                  fill_value=[])

    # Classify taxa based on association with medium:X
    def classify_taxa(media_list):
        medstr = 'medium:' + str(mediumid)
        if medstr in media_list:
            return medstr
        else:
            return 'other'

    # Apply classification function
    classified_taxa = taxon_media_groups.apply(classify_taxa)

    # Prepare the final DataFrame
    final_df = classified_taxa.reset_index()
    final_df.columns = ['NCBITaxon', 'medium']

    # Export to CSV
    fname = os.path.join(data_path,f'taxa_media_classification_{modellabel}.tsv')
    final_df.to_csv(fname, index=False, sep="\t")

    final_df.index = final_df['NCBITaxon']
    final_df.drop(columns=['NCBITaxon'], inplace=True)


    data_df = data_df.merge(final_df, left_index=True, right_index=True, how='left')

    data_df_orig = data_df.copy(deep=True)
    data_df = data_df[data_df['medium'].notna()]

    index_series = pd.Series(data_df.index.values)

    # Save this series to TSV file
    fname = os.path.join(data_path,'data_df__taxa_to_media__NCBITaxon__' + modellabel + '.tsv')
    index_series.to_csv(fname, sep='\t', index=False,
                        header=False)

    total_sum_numeric = data_df.select_dtypes(include=['number']).sum().sum()
    print(f"Total sum of numeric features: {total_sum_numeric}")

    # Clean singleton and constant features iteratively
    data_df_clean = data_df.copy()
    data_df_clean = clean_singleton_and_constant_features(data_df_clean)

    patterns = {}
    columns_to_drop = []
    retention_map = {}

    # Iterate over columns
    for col in data_df_clean.columns:
        pattern = tuple(data_df_clean[col])
        if pattern not in patterns:
            patterns[pattern] = col
            retention_map[col] = []  # Initialize the list of dropped columns for this pattern
        else:
            # Add the current column to the drop list and map it to the retained column
            columns_to_drop.append(col)
            retention_map[patterns[pattern]].append(col)

    # Drop duplicate columns
    data_df_clean = data_df_clean.drop(columns=columns_to_drop)

    # Prepare to write the mapping to a file
    retention_df = pd.DataFrame(
        [(retained, ','.join(duplicates)) for retained, duplicates in retention_map.items() if duplicates],
        columns=['Retained Column', 'Deleted Columns']
    )

    # Write to CSV file
    fname = os.path.join(data_path,'retention_df__' + modellabel + '.csv')
    retention_df.to_csv(fname, index=False)

    file_path = 'taxa_to_media__' + modellabel + '_data_df_clean.tsv.gz'
    file_path = os.path.join(data_path,file_path)
    print(f"Dataset has {len(data_df_clean.index)} rows and {len(data_df_clean.columns)} columns")
    print(f"Saving final data for medium {mediumid}")
    data_df_clean.to_csv(file_path, sep='\t', index=True, header=True, compression='gzip')




def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Prepare binary classification data for microbial growth medium prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single medium
  python 01_prepare_data_binary.py --medium 65

  # Process multiple media
  python 01_prepare_data_binary.py --medium 65 514

  # Process default media (65 and 514)
  python 01_prepare_data_binary.py
        """
    )

    parser.add_argument(
        '--medium', '-m',
        type=int,
        nargs='*',
        default=[65, 514],
        help='Medium ID(s) to process (default: 65 514)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data',
        help='Output directory for processed data (default: data)'
    )

    parser.add_argument(
        '--src-path',
        type=str,
        default='src',
        help='Directory containing source KG-Microbe data (default: src)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Update global paths if provided
    if args.data_path:
        data_path = args.data_path
    if args.src_path:
        src_path = args.src_path

    # Process each medium
    for mediumid in args.medium:
        print(f"\n{'='*80}")
        print(f"Processing medium {mediumid}")
        print(f"{'='*80}\n")
        run_data_prep(mediumid)
        print(f"\n{'='*80}")
        print(f"Completed medium {mediumid}")
        print(f"{'='*80}\n")

