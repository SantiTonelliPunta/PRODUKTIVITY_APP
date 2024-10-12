
# Import necessary libraries
import spacy
from collections import Counter
import pandas as pd
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CSV file containing embeddings
data_file_path = os.path.join(script_dir, '1000_embeddings_store.csv')
embeddings_df = pd.read_csv(data_file_path)

# Load SpaCy model for Named Entity Recognition (NER)
# Make sure to install the SpaCy model by running: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Function to extract brand names (organizations)
def extract_brand_names(text):
    doc = nlp(text)
    brand_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]  # Extract organizations (potential brand names)
    return brand_names

# Apply the function to the 'text' column to extract brand names
embeddings_df['brand_names'] = embeddings_df['text'].apply(extract_brand_names)

# Convert list of brand names to a comma-separated string for better readability in Excel
embeddings_df['brand_names'] = embeddings_df['brand_names'].apply(lambda x: ', '.join(x))

# Save the output to an Excel file
output_file_path = os.path.join(script_dir, 'brand_names_extracted.xlsx')
embeddings_df.to_excel(output_file_path, index=False)

print(f"Archivo Excel generado con los nombres de las marcas: {output_file_path}")
