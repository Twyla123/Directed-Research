import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to compute BERT embeddings for a given text
def compute_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the CLS token embedding, which represents the entire sentence
    return outputs.last_hidden_state[:, 0, :].numpy()

# Function to process AI data and compute embeddings
def process_ai_data(df):
    # Add a new column 'Verb-Noun Pairs Embedding' that stores the BERT embedding for each title
    df['Embedding'] = df['Title'].apply(lambda x: compute_bert_embedding(x) if isinstance(x, str) else np.zeros((1, 768)))
    return df

# Function to process occupation data and compute embeddings
def process_occupation_data(df):
    # Add a new column 'Verb-Noun Pairs Embedding' that stores the BERT embedding for each task description
    df['Embedding'] = df['Task Description'].apply(lambda x: compute_bert_embedding(x) if isinstance(x, str) else np.zeros((1, 768)))
    return df

# Function to compute cosine similarity between two embeddings
def compute_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2).item()

# === Processing AI data ===
# Load AI data from 'AI_verb_noun.csv'
ai_df = pd.read_csv('1_crawl_collect/AI_titles_dict.csv')

# Process the AI data to compute BERT embeddings
ai_df = process_ai_data(ai_df)

# === Processing Occupation data ===
# Load occupation data from 'occupation_verb_noun.csv'
occupation_df = pd.read_csv('/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/sample_occupation_verb_noun.csv')
occupation_df = pd.read_csv('/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/occupation_verb_noun.csv')

# Process the occupation data to compute BERT embeddings
occupation_df = process_occupation_data(occupation_df)

# Initialize a list to store similarity scores and matched pairs
similarity_scores = []
matched_pairs = []

# Compute cosine similarity between each occupation task embedding and all AI title embeddings
for occ_idx, occ_row in occupation_df.iterrows():
    occ_embedding = occ_row['Embedding']
    max_similarity = 0  # Track the maximum similarity for the current occupation task
    best_match = None  # Track the best matching AI verb-noun pair for this occupation task
    for ai_idx, ai_row in ai_df.iterrows():
        ai_embedding = ai_row['Embedding']
        # Compute similarity
        similarity = compute_similarity(occ_embedding, ai_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = ai_row['Title']  # Save the best matching AI title
    
    # Store the highest similarity and the matched AI title for this occupation task
    similarity_scores.append(max_similarity)
    matched_pairs.append((occ_row['Task Description'], best_match))

# Add the similarity scores to the occupation DataFrame
occupation_df['Max Similarity with AI'] = similarity_scores

# Save the processed occupation data with similarity scores to a CSV file named 'exposure_score_Bert.csv'
occupation_df.to_csv('2_extract_verb_noun/exposure_score_Bert.csv', index=False)

# Create a DataFrame to store matched pairs for verification
matched_df = pd.DataFrame(matched_pairs, columns=['Occupation Task Description', 'Matched AI Title'])

# Save the matched pairs to a CSV file for verification
matched_df.to_csv('2_extract_verb_noun/matched_verb_noun_pairs.csv', index=False)

# === Final Output Previews ===
# Print a preview of the processed AI data with embeddings
print("\nProcessed AI Data with Embeddings:")
print(ai_df[['Title', 'Embedding']].head())

# Print a preview of the processed Occupation data with embeddings and similarity scores
print("\nProcessed Occupation Data with Similarity Scores:")
print(occupation_df[['Task Description', 'Max Similarity with AI']].head())

# Print a preview of the matched verb-noun pairs
print("\nMatched Verb-Noun Pairs:")
print(matched_df.head())

#conda activate spacy_env
#python "/Users/twylazhang/Desktop/Directed Research/code_output/compare_BERT.py"
