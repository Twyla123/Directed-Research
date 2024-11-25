import pandas as pd
import ast
from collections import defaultdict
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Word2Vec or GloVe pre-trained word embeddings
# Ensure you have a pre-trained model available (e.g., Word2Vec or GloVe)
# Here, we assume it's Word2Vec
word_vectors = KeyedVectors.load_word2vec_format('path_to_pretrained_word_vectors.bin', binary=True)

# Function to read verb-noun pairs from a CSV and convert string representations of lists into actual lists of tuples
def read_verb_noun_pairs(file_name, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(file_name)

    # Convert string representations of lists to actual lists of tuples using ast.literal_eval
    if 'Verb-Noun Pairs' in df.columns:
        df['Verb-Noun Pairs'] = df['Verb-Noun Pairs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    else:
        print("Error: No valid 'Verb-Noun Pairs' column found.")
    return df

# Function to get the vector representation for a word using pre-trained embeddings
def get_word_embedding(word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros((word_vectors.vector_size,))

# Function to compute the average vector for a verb-noun pair
def get_pair_embedding(verb, noun):
    verb_embedding = get_word_embedding(verb)
    noun_embedding = get_word_embedding(noun)
    return (verb_embedding + noun_embedding) / 2

# Function to match occupation tasks with AI verb-noun pairs based on cosine similarity
def match_occupation_tasks_with_embeddings(occupation_tasks, ai_pairs):
    total_pairs = 0
    total_similarity = 0
    matched_pairs = []  # Store matched pairs for verification

    # Iterate over the tasks and compute cosine similarity between verb-noun pairs
    for task_pairs in occupation_tasks:
        if not task_pairs:  # Skip empty task pairs
            continue

        for verb, noun in task_pairs:
            total_pairs += 1
            task_embedding = get_pair_embedding(verb, noun)
            
            # Find the best matching AI verb-noun pair based on cosine similarity
            best_similarity = 0
            best_match = None
            for ai_verb, ai_noun in ai_pairs:
                ai_embedding = get_pair_embedding(ai_verb, ai_noun)
                similarity = cosine_similarity([task_embedding], [ai_embedding])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (ai_verb, ai_noun)

            # Store the best match for verification
            if best_match:
                matched_pairs.append(((verb, noun), best_match))
                total_similarity += best_similarity

    # Return average similarity and matched pairs
    return (total_similarity / total_pairs if total_pairs > 0 else 0), matched_pairs

# Function to calculate the exposure score based on cosine similarity and task-level weights
def calculate_exposure_score_with_embeddings(occupation_tasks, ai_pairs, task_importances):
    exposure_score = 0
    total_weight = 0
    all_matched_pairs = []  # To store all matched pairs across tasks

    # Iterate over each task in the occupation and apply task importance
    for task_idx, task_pairs in enumerate(occupation_tasks):
        if not task_pairs:
            continue

        task_importance = task_importances[task_idx]  # Importance for the current task

        if task_importance == 'Not available':
            continue
        
        try:
            task_importance = float(task_importance)
        except ValueError:
            continue

        match_percentage, matched_pairs = match_occupation_tasks_with_embeddings(task_pairs, ai_pairs)
        exposure_score += task_importance * match_percentage
        total_weight += task_importance

        all_matched_pairs.extend(matched_pairs)  # Collect matched pairs for later verification

    return (exposure_score / total_weight if total_weight > 0 else 0), all_matched_pairs

# Load AI patent verb-noun pairs and occupation task data
ai_titles_df = read_verb_noun_pairs('2_extract_verb_noun/AI_verb_noun.csv', file_type='csv')
#occupation_df = read_verb_noun_pairs('2_extract_verb_noun/occupation_verb_noun.csv', file_type='csv')
occupation_df = read_verb_noun_pairs('/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/sample_occupation_verb_noun.csv', file_type='csv')


# Debug: Print the first few rows of the AI Titles DataFrame to check if the verb-noun pairs are correctly parsed
print("\nFirst 5 rows of AI Titles Data:")
print(ai_titles_df.head())

# Flatten the list and remove empty lists for AI verb-noun pairs
ai_verb_noun_pairs = [pair for sublist in ai_titles_df['Verb-Noun Pairs'] if sublist for pair in sublist]

# Initialize a dictionary to store the exposure scores and a list to store matched pairs
exposure_scores = {}
all_matched_pairs = []

# Calculate the exposure score for each occupation
for idx, row in occupation_df.iterrows():
    occupation_name = row['Occupation Name']
    occupation_tasks = row['Verb-Noun Pairs']
    task_importance = row['Importance']

    task_importances = [task_importance] * len(occupation_tasks) if isinstance(task_importance, (int, float)) else [task_importance]

    print(f"\nProcessing occupation: {occupation_name}")

    # Calculate the exposure score using cosine similarity with word embeddings
    exposure_score, matched_pairs = calculate_exposure_score_with_embeddings(occupation_tasks, ai_verb_noun_pairs, task_importances)
    exposure_scores[occupation_name] = exposure_score
    all_matched_pairs.extend(matched_pairs)

# Convert exposure scores into a DataFrame for better visualization
exposure_scores_df = pd.DataFrame(list(exposure_scores.items()), columns=['Occupation Name', 'Exposure Score'])

# Save the exposure scores and matched pairs to CSV files
exposure_scores_df.to_csv('occupation_exposure_scores_with_word_embeddings.csv', index=False)

# Save matched pairs for verification
matched_pairs_df = pd.DataFrame(all_matched_pairs, columns=['Occupation Verb-Noun Pair', 'Matched AI Verb-Noun Pair'])
matched_pairs_df.to_csv('matched_verb_noun_pairs_with_word_embeddings.csv', index=False)

# Print a preview of the result
print("\nExposure Scores Preview:")
print(exposure_scores_df.head())

print("\nMatched Verb-Noun Pairs Preview:")
print(matched_pairs_df.head())