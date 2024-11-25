import pandas as pd
import ast
from collections import defaultdict
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

# Function to calculate the relative frequency for each verb-noun pair in AI patent data
def calculate_relative_frequencies(patent_pairs):
    pair_counts = defaultdict(int)
    total_count = 0

    # Iterate over the list of pairs
    for pairs in patent_pairs:
        for pair in pairs:
            pair_counts[pair] += 1
            total_count += 1

    # If no pairs were found
    if total_count == 0:
        print("No valid verb-noun pairs found in AI patent data.")
        return {}
    
    # Calculate relative frequency for each pair
    relative_frequencies = {pair: count / total_count for pair, count in pair_counts.items()}
    return relative_frequencies

# Function to initialize the PhraseMatcher with verb-noun pairs from AI patent data
def create_phrase_matcher(verb_noun_pairs):
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")  # Initialize PhraseMatcher based on lemmas
    
    # Convert each verb-noun pair into a doc and add it as a pattern to the matcher
    patterns = [nlp(f"{verb} {noun}") for verb, noun in verb_noun_pairs if len(verb_noun_pairs) == 2]
    matcher.add("VERB_NOUN_PAIRS", patterns)
    return matcher

# Function to match verb-noun pairs in occupation tasks using PhraseMatcher
def match_occupation_tasks(matcher, occupation_tasks):
    total_matches = 0
    total_pairs = 0

    # Iterate over the tasks and match verb-noun pairs
    for task_pairs in occupation_tasks:
        # Skip empty task pairs
        if not task_pairs:
            continue

        # Check if task_pairs is a valid list of tuples with exactly two elements
        valid_pairs = [(verb, noun) for pair in task_pairs if isinstance(pair, tuple) and len(pair) == 2]
        
        # Convert pairs into a string if they are valid
        task_text = " ".join([f"{verb} {noun}" for verb, noun in valid_pairs])
        
        if task_text:  # Only process if task_text is not empty
            doc = nlp(task_text)
            matches = matcher(doc)

            # Count the number of matched phrases
            total_matches += len(matches)
            total_pairs += len(valid_pairs)

    # Return match percentage
    return total_matches / total_pairs if total_pairs > 0 else 0

# Function to calculate the exposure score based on phrase matching and task-level weights (importance)
def calculate_exposure_score(occupation_tasks, matcher, task_importances):
    exposure_score = 0
    total_weight = 0

    # Iterate over each task in the occupation and apply task importance
    for task_idx, task_pairs in enumerate(occupation_tasks):
        # Skip empty task pairs
        if not task_pairs:
            print(f"Skipping task {task_idx + 1} due to empty verb-noun pairs.")
            continue

        # Handle the case when only one task importance is provided but multiple tasks exist
        if len(task_importances) == 1:
            task_importance = task_importances[0]
        else:
            task_importance = task_importances[task_idx]  # Importance for the current task

        # Skip or handle 'Not available' values
        if task_importance == 'Not available':
            print(f"Skipping task {task_idx + 1} due to 'Not available' task importance.")
            continue
        
        try:
            task_importance = float(task_importance)
        except ValueError:
            print(f"Invalid task importance for task {task_idx + 1}: {task_importance}")
            continue

        match_percentage = match_occupation_tasks(matcher, [task_pairs])

        print(f"\nTask {task_idx + 1} (task importance: {task_importance}): Match Percentage: {match_percentage:.6f}")

        # Apply task importance to the match percentage
        exposure_score += task_importance * match_percentage
        total_weight += task_importance

    return exposure_score / total_weight if total_weight > 0 else 0

# Load AI patent verb-noun pairs and occupation task data
ai_titles_df = read_verb_noun_pairs('2_extract_verb_noun/AI_verb_noun.csv', file_type='csv')
#occupation_df = read_verb_noun_pairs('2_extract_verb_noun/occupation_verb_noun.csv', file_type='csv')
occupation_df = read_verb_noun_pairs('/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/sample_occupation_verb_noun.csv', file_type='csv')

# Debug: Print the first few rows of the AI Titles DataFrame to check if the verb-noun pairs are correctly parsed
print("\nFirst 5 rows of AI Titles Data:")
print(ai_titles_df.head())

# Flatten the list and remove empty lists for AI verb-noun pairs
ai_verb_noun_pairs = [pair for sublist in ai_titles_df['Verb-Noun Pairs'] if sublist for pair in sublist]

# Create a PhraseMatcher using verb-noun pairs from AI patent data
matcher = create_phrase_matcher(ai_verb_noun_pairs)

# Initialize a dictionary to store the exposure scores
exposure_scores = {}

# Calculate the exposure score for each occupation
for idx, row in occupation_df.iterrows():
    occupation_name = row['Occupation Name']
    occupation_tasks = row['Verb-Noun Pairs']
    task_importance = row['Importance']  # Importance is a single numeric value or 'Not available'

    # If there are multiple tasks but only one importance value, repeat the importance for each task
    task_importances = [task_importance] * len(occupation_tasks) if isinstance(task_importance, (int, float)) else [task_importance]

    print(f"\nProcessing occupation: {occupation_name}")
    print(f"Verb-Noun Pairs for {occupation_name}: {occupation_tasks}")

    # Calculate the exposure score for this occupation using phrase matching and task importance
    exposure_score = calculate_exposure_score(occupation_tasks, matcher, task_importances)  
    exposure_scores[occupation_name] = exposure_score

# Convert exposure scores into a DataFrame for better visualization
exposure_scores_df = pd.DataFrame(list(exposure_scores.items()), columns=['Occupation Name', 'Exposure Score'])

# Save the exposure scores to a CSV file
exposure_scores_df.to_csv('occupation_exposure_scores_with_phrase_matching.csv', index=False)

# Print a preview of the result
print("\nExposure Scores Preview:")
print(exposure_scores_df.head())

#conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/compare_PhraseMatcher.py"