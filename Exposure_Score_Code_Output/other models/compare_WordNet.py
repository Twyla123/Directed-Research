import pandas as pd
import ast

# Function to read the data from a CSV and convert string representations of lists into actual lists of tuples
def read_occupation_data(file_name, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(file_name)

    # Convert string representations of lists to actual lists of tuples using ast.literal_eval
    if 'Matched Verb-Noun Pairs from AI' in df.columns:
        df['Matched Verb-Noun Pairs from AI'] = df['Matched Verb-Noun Pairs from AI'].apply(
            lambda x: [tuple(pair.split(': ')) for pair in x.split(',')] if isinstance(x, str) else []
        )
    else:
        print("Error: No valid 'Matched Verb-Noun Pairs from AI' column found.")
    return df

# Function to calculate the exposure score for an occupation based on matched verb-noun pairs and task importance
def calculate_exposure_score(occupation_tasks, task_importances, matched_verb_noun_pairs):
    exposure_score = 0
    total_weighted_pairs = 0

    # Iterate over each task in the occupation
    for task_idx, task_pairs in enumerate(matched_verb_noun_pairs):
        if not task_pairs:  # Skip if there are no matched pairs
            continue

        # Task importance for the current task
        task_importance = task_importances[task_idx]
        try:
            task_importance = float(task_importance)
        except ValueError:
            continue  # Skip tasks with invalid importance values

        # Calculate the sum of frequencies for the matched verb-noun pairs
        sum_matched_frequencies = sum([float(freq) for _, freq in task_pairs])

        # Add the weighted task score to the total exposure score
        exposure_score += task_importance * sum_matched_frequencies

        # Keep track of the total number of matched pairs, weighted by task importance
        total_weighted_pairs += task_importance * len(task_pairs)

    # Normalize the score by dividing by the total weighted pairs
    return exposure_score / total_weighted_pairs if total_weighted_pairs > 0 else 0

# Load the occupation task data
occupation_df = read_occupation_data('/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/compare_WordEmbedding.csv', file_type='csv')

# Initialize a dictionary to store the exposure scores
exposure_scores = {}

# Calculate the exposure score for each occupation
for idx, row in occupation_df.iterrows():
    occupation_name = row['Occupation Name']
    task_importances = [row['Task Importance']] * len(row['Matched Verb-Noun Pairs from AI']) if isinstance(row['Task Importance'], (int, float)) else [row['Task Importance']]

    print(f"\nProcessing occupation: {occupation_name}")

    # Calculate the exposure score based on matched verb-noun pairs and task importance
    exposure_score = calculate_exposure_score(row['Task Verb-Noun'], task_importances, row['Matched Verb-Noun Pairs from AI'])
    exposure_scores[occupation_name] = exposure_score

# Convert exposure scores into a DataFrame for better visualization
exposure_scores_df = pd.DataFrame(list(exposure_scores.items()), columns=['Occupation Name', 'Exposure Score'])

# Save the exposure scores to a CSV file
exposure_scores_df.to_csv('occupation_exposure_scores.csv', index=False)
exposure_scores_df.to_excel('occupation_exposure_scores.xlsx', index=False)

# Print a preview of the result
print("\nExposure Scores Preview:")
print(exposure_scores_df.head())



#conda activate spacy_env
#python "/Users/twylazhang/Desktop/Directed Research/code_output/compare_WordNet.py"
