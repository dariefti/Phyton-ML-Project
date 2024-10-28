# -*- coding: utf-8 -*-

# Phyton version: Phyton 3
pip install fuzzywuzzy
pip install sentence-transformers
import pandas as pd
import re
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Load datasets
foreignnames_df = pd.read_csv('ForeignNames_2019_2020.csv').dropna()
country_iso_df = pd.read_csv('Country_Name_ISO3.csv')

# Standardize country names for better matching
foreignnames_df['foreigncountry_cleaned'] = foreignnames_df['foreigncountry_cleaned'].str.strip().str.upper()
country_iso_df['country_name'] = country_iso_df['country_name'].str.strip().str.upper()

# Merge dataframes on country name to get ISO3 codes
merged_df = pd.merge(
    foreignnames_df,
    country_iso_df,
    left_on='foreigncountry_cleaned',
    right_on='country_name',
    how='left'
).drop(columns=['country_name'])

# Clean and normalize firm names by removing non-alphabet characters, common suffixes, and extra whitespace
def clean_firm_name(name):
    if pd.isna(name):  # Check for NA and replace with an empty string
        name = ""
    name = str(name).lower()  # Convert to string and lowercase
    name = re.sub(r'[^a-zA-Z\s]', '', name)  # Remove non-alphabet characters
    # Remove common suffixes related to company types
    suffixes = [
    "ltd", "limited", "inc", "llc", "corp", "corporation", "company", "co", "pte",
    "bhd", "srl", "sa", "gmbh", "ag", "plc", "nv", "bv", "sl", "srl", "srl",
    "as", "ab", "oy", "sp zoo", "spa", "sas", "sasu", "kk"]
    for suffix in suffixes:
        name = re.sub(r'\b' + suffix + r'\b', '', name)
    # Remove extra whitespace
    return " ".join(name.split())

merged_df['normalized_name'] = merged_df['foreign'].apply(clean_firm_name)

# Fuzzy matching within each country
def assign_fuzzy_cleaned_name(group):
    unique_names = list(group['normalized_name'].unique())
    cleaned_mapping = {}

    for name in unique_names:
        if name not in cleaned_mapping:
            # Find the best match within 95% similarity
            matches = process.extractBests(name, unique_names, scorer=fuzz.token_sort_ratio, score_cutoff=95)
            best_match = matches[0][0]
            cleaned_mapping.update({match[0]: best_match for match in matches})

    # Map cleaned names back to group
    group['cleaned_name'] = group['normalized_name'].map(cleaned_mapping)
    return group

# Apply fuzzy matching by country
merged_df = merged_df.groupby('country_iso3', group_keys=False).apply(assign_fuzzy_cleaned_name)

# Create unique ID by combining country code and cleaned name
merged_df['cleaned_ID'] = merged_df.groupby(['country_iso3', 'cleaned_name']).ngroup()
merged_df['cleaned_ID'] = merged_df['country_iso3'] + merged_df['cleaned_ID'].astype(str).str.zfill(4)

# Display final dataframe
merged_df[['foreign', 'foreigncountry_cleaned', 'country_iso3', 'cleaned_name', 'cleaned_ID']].head()

# Now for the machine learning model: BERT-based matching
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(merged_df['normalized_name'].tolist(), convert_to_tensor=True)

# Function for BERT-based similarity search
def bert_based_cleaned_name(group, embeddings, names):
    cleaned_mapping = {}
    for idx, name in enumerate(group['normalized_name']):
        if name not in cleaned_mapping:
            query_embedding = embeddings[idx].unsqueeze(0)  # Get embedding for the current name
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings).flatten()
            top_match_idx = torch.argmax(cosine_scores).item()
            best_match = names[top_match_idx]
            cleaned_mapping[name] = best_match
    return group['normalized_name'].map(cleaned_mapping)

# Apply BERT-based matching
merged_df['cleaned_name_bert'] = bert_based_cleaned_name(merged_df, embeddings, merged_df['normalized_name'].tolist())

# Create unique IDs for BERT-based method
merged_df['cleaned_ID_bert'] = merged_df.groupby(['country_iso3', 'cleaned_name_bert']).ngroup()
merged_df['cleaned_ID_bert'] = merged_df['country_iso3'] + merged_df['cleaned_ID_bert'].astype(str).str.zfill(4)

# Count distinct values for both methods
fuzzy_unique_names = merged_df['cleaned_name'].nunique()
fuzzy_unique_ids = merged_df['cleaned_ID'].nunique()
bert_unique_names = merged_df['cleaned_name_bert'].nunique()
bert_unique_ids = merged_df['cleaned_ID_bert'].nunique()

# Display the distinct counts
#I assume that a smaller number indicates greater efficiency, as it suggests better classification of the names
print(f"Distinct cleaned_name (FuzzyWuzzy): {fuzzy_unique_names}")
print(f"Distinct cleaned_ID (FuzzyWuzzy): {fuzzy_unique_ids}")
print(f"Distinct cleaned_name (BERT): {bert_unique_names}")
print(f"Distinct cleaned_ID (BERT): {bert_unique_ids}")

# Save the full dataset with all original fields plus cleaned_name and cleaned_ID
columns_to_save = ['foreign', 'foreigncountry_cleaned', 'country_iso3', 'cleaned_name', 'cleaned_ID']
outputfile_full = "outputfile_daniera_1.csv"
merged_df.to_csv(outputfile_full, columns=columns_to_save, index=False)

# Create a dataset with only changed names
# Keep only rows where the original name and cleaned name differ
changed_names_df = merged_df[merged_df['normalized_name'] != merged_df['cleaned_name']][['foreign', 'cleaned_name']]

# Rename columns for clarity
changed_names_df = changed_names_df.rename(columns={'foreign': 'original_name', 'cleaned_name': 'cleaned_name'})

# Save the changed names dataset
outputfile_changed = "outputfile_daniera_1_changed.csv"
changed_names_df.to_csv(outputfile_changed, index=False)

# Load the 2021 dataset
foreignnames_2021_df = pd.read_csv('ForeignNames_2021.csv').dropna()

# Standardize country names in the 2021 dataset
foreignnames_2021_df['foreigncountry_cleaned'] = foreignnames_2021_df['foreigncountry_cleaned'].str.strip().str.upper()

# Merge with country ISO codes to get the country code for 2021 data
merged_2021_df = pd.merge(
    foreignnames_2021_df,
    country_iso_df,
    left_on='foreigncountry_cleaned',
    right_on='country_name',
    how='left'
).drop(columns=['country_name'])

# Clean and normalize firm names for the 2021 data using the same cleaning function
merged_2021_df['normalized_name'] = merged_2021_df['foreign'].apply(clean_firm_name)

# Dictionary of existing cleaned names and IDs by country from the 2019-2020 dataset
existing_names_by_country = merged_df.groupby('country_iso3')['cleaned_name'].apply(set).to_dict()
existing_ids_by_name = merged_df.set_index(['country_iso3', 'cleaned_name'])['cleaned_ID'].to_dict()

# Fuzzy match new data against existing cleaned names and assign IDs
def fuzzy_match_with_existing(cleaned_name, country_code, existing_names):
    if country_code in existing_names:
        match, score = process.extractOne(cleaned_name, list(existing_names[country_code]), scorer=fuzz.token_sort_ratio)
        return match if score >= 95 else None
    return None

# Apply fuzzy matching to 2021 data
merged_2021_df['cleaned_name'] = merged_2021_df.apply(
    lambda row: fuzzy_match_with_existing(row['normalized_name'], row['country_iso3'], existing_names_by_country),
    axis=1
)

# Assign IDs: If a match is found, use the existing ID; otherwise, assign a new ID
new_id_start = merged_df['cleaned_ID'].str.extract('(\d+)').astype(int).max()[0] + 1
merged_2021_df['cleaned_ID'] = merged_2021_df.apply(
    lambda row: existing_ids_by_name.get((row['country_iso3'], row['cleaned_name']))
    if pd.notna(row['cleaned_name']) else f"{row['country_iso3']}{str(new_id_start + row.name).zfill(4)}",
    axis=1
)

# Mark new firms and set old names where applicable
merged_2021_df['new'] = merged_2021_df['cleaned_name'].isna().astype(int)
merged_2021_df['old_name'] = merged_2021_df.apply(
    lambda row: merged_df[(merged_df['country_iso3'] == row['country_iso3']) & (merged_df['cleaned_name'] == row['cleaned_name'])]['foreign'].values[0]
    if row['new'] == 0 else None,
    axis=1
)

# Fill in cleaned_name from normalized_name for new firms without a match
merged_2021_df['cleaned_name'] = merged_2021_df.apply(
    lambda row: row['cleaned_name'] if pd.notna(row['cleaned_name']) else row['normalized_name'],
    axis=1
)

# Save the final output for 2021 with the specified columns
outputfile_2 = "outputfile_daniera_2.csv"
merged_2021_df[['foreign', 'cleaned_name', 'cleaned_ID', 'new', 'old_name']].to_csv(outputfile_2, index=False)
