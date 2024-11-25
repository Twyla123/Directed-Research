import pandas as pd

# Function to calculate the frequency of each AI verb-noun pair in the uncleaned AI data
def calculate_ai_pair_frequencies(ai_file):
    # Read AI verb-noun pairs from the CSV file (uncleaned data)
    ai_df = pd.read_csv(ai_file)

    # Create a list of AI verb-noun pairs (using 'Verb' and 'Noun' columns)
    ai_verb_noun_pairs = list(zip(ai_df['Verb'], ai_df['Noun']))
    
    # Calculate total number of pairs
    total_ai_pairs = len(ai_verb_noun_pairs)
    
    # Calculate the frequency of each AI verb-noun pair
    frequency_dict = pd.Series(ai_verb_noun_pairs).value_counts(normalize=True).to_dict()  # Normalize to get relative frequency
    
    return frequency_dict, total_ai_pairs

# Function to read the data from a CSV and convert string representations of lists into actual lists of strings
def read_occupation_data(file_name, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(file_name)

    # Convert string representations of lists into actual lists of verb-noun pairs
    if 'Matched Verb-Noun Pairs from AI' in df.columns:
        df['Matched Verb-Noun Pairs from AI'] = df['Matched Verb-Noun Pairs from AI'].apply(
            lambda x: x.split(', ') if isinstance(x, str) else []
        )
    else:
        print("Error: No valid 'Matched Verb-Noun Pairs from AI' column found.")
    return df

# Function to replace 'Not Available' in Task Importance with the average of available task importance values for each occupation
def fill_missing_importance(df):
    # Replace 'Not Available' with NaN
    df['Task Importance'] = df['Task Importance'].replace('Not available', pd.NA)

    # Convert Task Importance to numeric, treating NaN as missing values
    df['Task Importance'] = pd.to_numeric(df['Task Importance'], errors='coerce')

    # Group by Occupation Name and fill NaN with the average task importance for that occupation
    df['Task Importance'] = df.groupby('Occupation Name')['Task Importance'].transform(lambda x: x.fillna(x.mean()))

    return df

# Function to calculate the exposure score using AI pair frequencies
def calculate_exposure_score(task_importances, matched_verb_noun_pairs, frequency_dict):
    weighted_sum_frequencies = 0
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

        # Process the matched verb-noun pairs
        for pair in set(task_pairs):  # Use set to ensure unique pairs
            # Split the pair into verb and noun
            verb_noun = pair.split(' ')
            if len(verb_noun) != 2:
                print(f"Skipping invalid pair: {pair}")
                continue  # Skip invalid pairs if not exactly two parts

            verb, noun = verb_noun
            frequency = frequency_dict.get((verb, noun), 0)  # Get the frequency from the dict
            weighted_sum_frequencies += task_importance * frequency

        # Add to the total weighted pairs (just counting total verb-noun pairs)
        total_weighted_pairs += task_importance * len(task_pairs)

    # Calculate and return the exposure score using the weighted sum formula
    return weighted_sum_frequencies / total_weighted_pairs if total_weighted_pairs > 0 else 0

# === Main Execution ===
# Step 1: Calculate frequencies of AI verb-noun pairs from the uncleaned data
ai_frequencies, total_extracted_pairs = calculate_ai_pair_frequencies('/Users/twylazhang/Desktop/Directed Research/code_output/3_filter_prepare_compare/filtered_AI_verb_noun_meaningful.csv')

# Step 2: Load the cleaned occupation task data
occupation_df = read_occupation_data('/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/cleaned_matched_pairs.csv', file_type='csv')

# Step 3: Fill missing task importance values with the average importance for the same occupation
occupation_df = fill_missing_importance(occupation_df)

# Initialize dictionaries to store cumulative exposure values
weighted_sum_frequencies = {}
weighted_total_pairs = {}

# Step 4: Calculate the exposure score for each occupation
for idx, row in occupation_df.iterrows():
    occupation_name = row['Occupation Name']
    task_importances = [row['Task Importance']] * len(row['Matched Verb-Noun Pairs from AI']) if isinstance(row['Task Importance'], (int, float)) else [row['Task Importance']]

    print(f"\nProcessing occupation: {occupation_name}")

    # Initialize the weighted sum and total pairs for this occupation if not already done
    if occupation_name not in weighted_sum_frequencies:
        weighted_sum_frequencies[occupation_name] = 0
        weighted_total_pairs[occupation_name] = 0
    
    # Calculate the weighted sums for the current task based on matched verb-noun pairs, task importance, and AI frequencies
    current_weighted_sum = 0
    current_total_weighted_pairs = 0

    # Iterate over the matched verb-noun pairs for the task
    for pair in set(row['Matched Verb-Noun Pairs from AI']):  # Use set to avoid duplicates
        verb_noun = pair.split(' ')
        if len(verb_noun) != 2:
            print(f"Skipping invalid pair: {pair}")
            continue  # Skip invalid pairs if not exactly two parts

        verb, noun = verb_noun
        frequency = ai_frequencies.get((verb, noun), 0)  # Get the frequency from the dictionary (or 0 if not found)

        # Add the weighted frequency to the current weighted sum (task importance * frequency)
        current_weighted_sum += row['Task Importance'] * frequency
    
    # Add the weighted count of verb-noun pairs to the total weighted pairs for this task
    current_total_weighted_pairs += row['Task Importance'] * len(row['Matched Verb-Noun Pairs from AI'])

    # Update cumulative sums for the occupation
    weighted_sum_frequencies[occupation_name] += current_weighted_sum
    weighted_total_pairs[occupation_name] += current_total_weighted_pairs

# After processing all rows, calculate the exposure score for each occupation
exposure_scores = {
    occupation: weighted_sum_frequencies[occupation] / weighted_total_pairs[occupation] if weighted_total_pairs[occupation] > 0 else 0
    for occupation in weighted_sum_frequencies
}

# Convert exposure scores into a DataFrame for better visualization
exposure_scores_df = pd.DataFrame(list(exposure_scores.items()), columns=['Occupation Name', 'Exposure Score'])

# Save the exposure scores to a CSV file
exposure_scores_df.to_csv('/Users/twylazhang/Desktop/Directed Research/code_output/6_calculate/occupation_exposure_scores.csv', index=False)
exposure_scores_df.to_excel('/Users/twylazhang/Desktop/Directed Research/code_output/6_calculate/occupation_exposure_scores.xlsx', index=False)

# Print a preview of the result
print("\nExposure Scores Preview:")
print(exposure_scores_df.head())