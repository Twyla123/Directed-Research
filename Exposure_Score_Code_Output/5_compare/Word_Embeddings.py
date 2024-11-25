import pandas as pd
import ast
import spacy
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load spaCy model with word vectors
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")
print("spaCy model loaded.")

# Cache dictionary for word vectors
vector_cache = {}

# Optimized cosine similarity using numpy with zero norm and NaN check
def cosine_similarity_np(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0  # Return similarity of 0 if any vector is zero
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Function to get the cached vector or compute it if not available
def get_vector(text):
    if text in vector_cache:
        return vector_cache[text]
    else:
        vector = nlp(text).vector
        vector_cache[text] = vector
        return vector

# Function to compute the average vector for a verb-noun pair
def compute_pair_vector(verb, noun):
    verb_vector = get_vector(verb)
    noun_vector = get_vector(noun)
    return np.mean([verb_vector, noun_vector], axis=0)

# Function to compare two verb-noun pairs using cosine similarity
def compare_pairs(verb1, noun1, verb2, noun2):
    pair1_vector = compute_pair_vector(verb1, noun1)
    pair2_vector = compute_pair_vector(verb2, noun2)
    return cosine_similarity_np(pair1_vector, pair2_vector)

# Function to match occupation tasks with AI verb-noun pairs
def match_occupation_tasks_with_embeddings(occupation_row, ai_verb_noun_pairs, similarity_threshold=0.85):
    occupation_name = occupation_row['Occupation Name']
    task_importance = occupation_row['Importance']
    verb_noun_pairs = occupation_row['Meaningful Verb-Noun Pairs']
    
    individual_matches = []
    
    for verb1, noun1 in verb_noun_pairs:
        matched_ai_pairs = []
        for verb2, noun2 in ai_verb_noun_pairs:
            similarity = compare_pairs(verb1, noun1, verb2, noun2)
            if similarity >= similarity_threshold:
                matched_ai_pairs.append(f"{verb2} {noun2}")
        
        if matched_ai_pairs:
            # Create a row for each verb-noun pair with its corresponding details
            individual_matches.append({
                'Occupation Name': occupation_name,
                'Task Verb-Noun': f"{verb1} {noun1}",
                'Task Importance': task_importance,
                'Matched Verb-Noun Pairs from AI': ', '.join(matched_ai_pairs)
            })
    
    return individual_matches if individual_matches else None

# Function to process occupations in parallel
def process_in_batches(ai_file, occupation_df, similarity_threshold=0.85):
    print(f"Loading AI verb-noun pairs from {ai_file}...")
    ai_df = pd.read_csv(ai_file)
    ai_verb_noun_pairs = list(zip(ai_df['Verb'], ai_df['Noun']))  # Ensure to use 'Verb' and 'Noun' columns
    print(f"Loaded {len(ai_verb_noun_pairs)} AI verb-noun pairs.")
    
    all_matched_pairs = []
    
    print("Starting to process occupation tasks...")
    
    # Process occupation tasks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_matches = [
            executor.submit(match_occupation_tasks_with_embeddings, row, ai_verb_noun_pairs, similarity_threshold)
            for idx, row in occupation_df.iterrows()
        ]
        
        for i, future in enumerate(future_matches):
            result = future.result()
            if result:
                all_matched_pairs.extend(result)  # Append each individual task's match
            if i % 100 == 0:  # Print progress every 100 rows
                print(f"Processed {i} occupation rows.")
    
    print("Finished processing all occupation tasks.")
    return all_matched_pairs

# === Main Execution ===
ai_synonym_pairs_file = '/Users/twylazhang/Desktop/Directed Research/code_output/filtered_AI_verb_noun_meaningful.csv'
occupation_df = pd.read_csv('/Users/twylazhang/Desktop/Directed Research/code_output/filtered_occupation_pairs.csv')

# Convert 'Meaningful Verb-Noun Pairs' column from string to list of tuples
print("Converting 'Meaningful Verb-Noun Pairs' to list of tuples...")
occupation_df['Meaningful Verb-Noun Pairs'] = occupation_df['Meaningful Verb-Noun Pairs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
print("Conversion complete.")

# Process AI verb-noun pairs and match with the entire occupation dataset
print("Starting matching process for the entire dataset...")
matched_pairs = process_in_batches(ai_synonym_pairs_file, occupation_df)
print("Matching process complete.")

# Convert matched pairs to DataFrame and save
matched_pairs_df = pd.DataFrame(matched_pairs, columns=['Occupation Name', 'Task Verb-Noun', 'Task Importance', 'Matched Verb-Noun Pairs from AI'])

# Save to CSV and Excel
output_csv = '/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/compare_WordEmbedding.csv'
output_excel = '/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/compare_WordEmbedding.xlsx'

print(f"Saving matched pairs to CSV: {output_csv}")
matched_pairs_df.to_csv(output_csv, index=False)

print(f"Saving matched pairs to Excel: {output_excel}")
matched_pairs_df.to_excel(output_excel, index=False)

print("Files saved successfully.")

print("\nMatched Verb-Noun Pairs with Word Embeddings and Similarity Preview:")
print(matched_pairs_df.head())


# conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/Word_Embeddings.py"