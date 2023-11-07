import pandas as pd

# File paths
file_paths = {
    'archs4': '/Users/jasmingadestovlbaek/DeepLearning_project/_raw/archs4_gene_expression_norm_transposed.tsv',
    'gtex_gene': '/Users/jasmingadestovlbaek/DeepLearning_project/_raw/gtex_gene_expression_norm_transposed.tsv',
    'gtex_isoform': '/Users/jasmingadestovlbaek/DeepLearning_project/_raw/gtex_isoform_expression_norm_transposed.tsv'
}

# Cleaning function - modify as needed for each file
def clean_data(chunk):
    # Implement your cleaning code here
    # For example: chunk = chunk.dropna()
    return chunk

# Define chunk size
chunk_size = 10000  # Adjust based on your available memory

# Process each file
for file_name, file_path in file_paths.items():
    cleaned_data = pd.DataFrame()  # Empty DataFrame to hold cleaned data
    
    # Reading the file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, sep='\t', compression='gzip'):
        cleaned_chunk = clean_data(chunk)
        cleaned_data = pd.concat([cleaned_data, cleaned_chunk], ignore_index=True)
    
    # Save the cleaned data to a new file
    cleaned_file_path = f'/Users/jasmingadestovlbaek/DeepLearning_project/data/02_clean_{file_name}.tsv'
    cleaned_data.to_csv(cleaned_file_path, sep='\t', index=False)
    print(f'Cleaned data for {file_name} saved to {cleaned_file_path}')
