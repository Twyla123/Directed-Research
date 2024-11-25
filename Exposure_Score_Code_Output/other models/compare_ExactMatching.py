import spacy
import pandas as pd
import ast
from collections import defaultdict

# Load spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Function to read verb-noun pairs from a CSV or Excel file and convert string representations of lists into actual lists of tuples
def read_verb_noun_pairs(file_name, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(file_name)
    elif file_type == 'excel':
        df = pd.read_excel(file_name)

    # Convert string representations of lists to actual lists of tuples using ast.literal_eval
    if 'verb_noun_pairs' in df.columns:
        df['verb_noun_pairs'] = df['verb_noun_pairs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    elif 'Verb-Noun Pairs' in df.columns:
        df['Verb-Noun Pairs'] = df['Verb-Noun Pairs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    else:
        print("Error: No valid 'verb_noun_pairs' column found.")

    return df

# Function to lemmatize verb-noun pairs for better matching
def lemmatize_pairs(pairs):
    lemmatized_pairs = []
    for verb, noun in pairs:
        doc = nlp(verb + " " + noun)
        lemmatized_pairs.append((doc[0].lemma_, doc[1].lemma_))
    return lemmatized_pairs

# Function to calculate the relative frequency for each verb-noun pair in AI patent data
def calculate_relative_frequencies(occupation_pairs, ai_verb_noun_pairs):
    pair_counts = defaultdict(int)
    total_ai_pairs = len(ai_verb_noun_pairs)

    # Iterate over the occupation verb-noun pairs
    for occupation_pair in occupation_pairs:
        # Count how many times this occupation pair appears in the AI data
        count = sum(1 for ai_pair in ai_verb_noun_pairs if ai_pair == occupation_pair)
        pair_counts[occupation_pair] = count

    # If no AI pairs exist, return empty dictionary
    if total_ai_pairs == 0:
        print("No valid verb-noun pairs found in AI patent data.")
        return {}
    # Calculate relative frequency for each occupation pair (count / total number of AI pairs)
    relative_frequencies = {pair: count / total_ai_pairs for pair, count in pair_counts.items()}

    return relative_frequencies

# Function to calculate the exposure score and store matched pairs
def calculate_exposure_score(occupation_tasks, relative_frequencies, task_weights, matched_pairs):
    exposure_score = 0
    total_weight = 0

    # Iterate over each task in the occupation and calculate relative frequencies
    for task_idx, task_pairs in enumerate(occupation_tasks):
        task_weight = task_weights[task_idx]
        task_score = 0

        # Iterate through pairs, lemmatize and compare them to the AI pairs
        for pair in task_pairs:
            lemmatized_pair = (nlp(pair[0])[0].lemma_, nlp(pair[1])[0].lemma_)
            relative_frequency = relative_frequencies.get(lemmatized_pair, 0)
            task_score += relative_frequency

            # Store matched pairs (occupation and AI patent) if both verb and noun match exactly
            if relative_frequency > 0:
                matched_pairs.append((pair, lemmatized_pair))

        # Normalize by number of pairs in the task and apply task weight
        if len(task_pairs) > 0:
            exposure_score += task_weight * (task_score / len(task_pairs))
        total_weight += task_weight

    return exposure_score / total_weight if total_weight > 0 else 0

# Debugging function to print the first few rows of the DataFrame
def print_debug_info(df, name):
    print(f"\n{name} - First 5 rows:")
    print(df.head())

# Load AI patent verb-noun pairs and occupation task data
ai_titles_df = read_verb_noun_pairs('AI_verb_noun.csv', file_type='csv')
occupation_df = read_verb_noun_pairs('occupation_verb_noun_pairs.xlsx', file_type='excel')

# Print AI Titles and Occupation Data Columns
print("AI Titles Columns: ", ai_titles_df.columns)
print("Occupation Data Columns: ", occupation_df.columns)

# Debug: Print the first few rows of the AI Titles DataFrame to check if the verb-noun pairs are correctly parsed
print_debug_info(ai_titles_df, "AI Titles Data")
print_debug_info(occupation_df, "Occupation Data")

# Flatten the list and remove empty lists
ai_verb_noun_pairs = [pair for sublist in ai_titles_df['verb_noun_pairs'] if sublist for pair in sublist]
ai_verb_noun_pairs_lemmatized = lemmatize_pairs(ai_verb_noun_pairs)

# Calculate the relative frequencies of verb-noun pairs in AI patents
relative_frequencies = calculate_relative_frequencies(ai_verb_noun_pairs_lemmatized)

# If no relative frequencies are calculated, stop the process
if not relative_frequencies:
    print("No relative frequencies were calculated. Exiting.")
else:
    # Initialize a dictionary to store the exposure scores
    exposure_scores = {}
    matched_pairs = []  # List to store matched verb-noun pairs

    # Calculate the exposure score for each occupation
    for idx, row in occupation_df.iterrows():
        occupation_name = row['Occupation Name']
        occupation_tasks = row['Verb-Noun Pairs']
        occupation_tasks_lemmatized = lemmatize_pairs(occupation_tasks)

        print(f"\nProcessing occupation: {occupation_name}")

        # Generate task weights to match the number of tasks (equal weights)
        task_weights = [1.0 / len(occupation_tasks)] * len(occupation_tasks)

        # Calculate the exposure score for this occupation and save matched pairs
        exposure_score = calculate_exposure_score(occupation_tasks_lemmatized, relative_frequencies, task_weights, matched_pairs)
        exposure_scores[occupation_name] = exposure_score

    # Convert exposure scores into a DataFrame for better visualization
    exposure_scores_df = pd.DataFrame(list(exposure_scores.items()), columns=['Occupation Name', 'Exposure Score'])

    # Save the exposure scores to a CSV file
    exposure_scores_df.to_csv('occupation_exposure_scores.csv', index=False)

    # Convert matched pairs to a DataFrame and save to CSV
    matched_pairs_df = pd.DataFrame(matched_pairs, columns=['Occupation Verb-Noun Pair', 'AI Patent Verb-Noun Pair'])
    matched_pairs_df.to_csv('matched_verb_noun_pairs.csv', index=False)

    # Print a preview of the result
    print(exposure_scores_df.head())
    print("\nMatched Verb-Noun Pairs:")
    print(matched_pairs_df.head())



#conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/compare.py"