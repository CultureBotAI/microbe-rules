#prepare data for binary scoring


from pathlib import Path

import pandas as pd
import os
import requests
import tarfile

src_path = "src"
data_path = "data"

data_link = "https://zenodo.org/records/15106978"

data_download_link = data_link + "/files/merged-kg.tar.gz?download=1"



def download_file_requests(url, folder_path, filename):
    """
    Downloads a file from a URL and saves it to a specified folder.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the full local file path
    local_filename = os.path.join(folder_path, filename)
    
    try:
        # Send a GET request to the URL
        with requests.get(url, stream=True) as r:
            # Raise an HTTPError for bad responses (4xx or 5xx)
            r.raise_for_status() 
            
            # Write the content to the local file
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"Downloaded '{filename}' to '{folder_path}'")
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
        exit(1)

def unpack_targz(file_path, destination_folder):
    """
    Unpacks a .tar.gz file to a specified destination folder.
    """
    
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    try:
        with tarfile.open(file_path, "r:gz") as tar:           
            tar.extractall(path=destination_folder, filter='data')
            
        print(f"Successfully unpacked '{file_path}' to '{destination_folder}'")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except tarfile.TarError as e:
        print(f"Error during unpacking: {e}")

def run_data_prep(mediumid):
    print(f"Preparing data for medium {mediumid}")

    if not os.path.exists(data_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(data_path)

    if not os.path.exists(src_path):
        print("Folder with source data does not exists. Will create it.")
        os.makedirs(src_path)
        #print("Folder with source data does not exists. Please create the folder "+src_path+" and download data from the link into the folder and unpack the file to that folder. Link: "+data_link)
        #exit(1)

    if not Path(os.path.join(src_path,"merged-kg_edges.tsv")).is_file():
       print("File with KG Microbe data does not exists.")
       archive_name = "merged-kg.tar.gz"
       local_archive_filename = os.path.join(src_path,archive_name)
       if not Path(local_archive_filename).is_file():
           print("Archive file package does not exists. Will download it.")
           download_file_requests(data_download_link,src_path,archive_name)
       print("Now, will npack the archive") 
       unpack_targz(local_archive_filename,src_path)
                            
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

    data_pairs_rest_all = data_pairs[data_pairs['subject'].str.contains('NCBITaxon:|strain:')]
    data_pairs_rest = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('carbon_substrates:')]
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('pathways:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('trophic_type:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('production:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('CAS-RN:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('CHEBI:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('EC:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('GO:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)

    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('cell_shape:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('cell_length:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('cell_width:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('motility:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('sporulation:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('pigment:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('gram_stain:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)

    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('gc:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)

    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('pH_.*:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('temp_.*:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('temperature:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('salinity:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('NaCl_.*:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('oxygen:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)

    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('pathogen:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('isolation_source:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('ENVO:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('UBERON:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all[data_pairs_rest_all['object'].str.contains('PO:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)

    data_pairs_rest_all2 = data_pairs[data_pairs['object'].str.contains('NCBITaxon:|strain:')]
    # Swap 'subject' and 'object' for the filtered DataFrame
    data_pairs_rest_all2_swapped = data_pairs_rest_all2.copy()
    data_pairs_rest_all2_swapped['subject'], data_pairs_rest_all2_swapped['object'] = data_pairs_rest_all2['object'], \
    data_pairs_rest_all2['subject']
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('PATO:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('UBERON:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('FOODON:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('CHEBI:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('ENVO:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('PO:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)
    data_pairs_rest2 = data_pairs_rest_all2_swapped[data_pairs_rest_all2_swapped['object'].str.contains('assay:')]
    data_pairs_rest = pd.concat([data_pairs_rest, data_pairs_rest2], ignore_index=True)

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

    data_df_clean = data_df.copy()

    # Select only numeric columns for the operation
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns

    # Remove columns with sum < 1, excluding the 'medium' column or any other non-numeric columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2 = data_df_clean[numeric_cols].sum(axis=0)
    cols_to_drop = data_df_clean[numeric_cols].columns[sum_less2 <= 1]
    data_df_clean = data_df_clean.drop(columns=cols_to_drop)

    # Remove cols which are all 1's
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2 = data_df_clean[numeric_cols].sum(axis=0)
    cols_to_drop = data_df_clean[numeric_cols].columns[sum_less2 == dimnow[1]]
    data_df_clean = data_df_clean.drop(columns=cols_to_drop)

    # Correct the approach to remove rows with sum < 1, ensuring to only sum over the updated numeric columns
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2_row = data_df_clean[numeric_cols].sum(axis=1)
    rows_to_drop = data_df_clean.index[sum_less2_row <= 1]
    data_df_clean = data_df_clean.drop(index=rows_to_drop)

    # Remove rows which are all 1's
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2_row = data_df_clean[numeric_cols].sum(axis=1)
    # print(sum_less2_row[sum_less2_row == dimnow[0]])
    rows_to_drop = data_df_clean.index[sum_less2_row == dimnow[0]]
    # print(rows_to_drop)
    data_df_clean = data_df_clean.drop(index=rows_to_drop)
    print(dimnow)

    # Select only numeric columns for the operation
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns

    # Remove columns with sum < 1, excluding the 'medium' column or any other non-numeric columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2 = data_df_clean[numeric_cols].sum(axis=0)
    cols_to_drop = data_df_clean[numeric_cols].columns[sum_less2 <= 1]
    data_df_clean = data_df_clean.drop(columns=cols_to_drop)

    # Remove cols which are all 1's
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2 = data_df_clean[numeric_cols].sum(axis=0)
    cols_to_drop = data_df_clean[numeric_cols].columns[sum_less2 == dimnow[1]]
    data_df_clean = data_df_clean.drop(columns=cols_to_drop)

    # remove rows with sum < 1, ensuring to only sum over the updated numeric columns
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2_row = data_df_clean[numeric_cols].sum(axis=1)
    rows_to_drop = data_df_clean.index[sum_less2_row <= 1]
    data_df_clean = data_df_clean.drop(index=rows_to_drop)

    # Remove rows which are all 1's
    numeric_cols = data_df_clean.select_dtypes(include=['number']).columns
    dimnow = data_df_clean.shape
    print(dimnow)
    sum_less2_row = data_df_clean[numeric_cols].sum(axis=1)
    # print(sum_less2_row[sum_less2_row == dimnow[0]])
    rows_to_drop = data_df_clean.index[sum_less2_row == dimnow[0]]
    # print(rows_to_drop)
    data_df_clean = data_df_clean.drop(index=rows_to_drop)
    print(dimnow)

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




if __name__ == '__main__':
    models = [65,514]

    for m in models:
        run_data_prep(m)

