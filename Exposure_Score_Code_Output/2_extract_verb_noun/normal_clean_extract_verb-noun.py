import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from langdetect import detect, DetectorFactory
import ast

# Ensure reproducibility for language detection
DetectorFactory.seed = 0

# Initialize stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define a function to preprocess text: tokenize, remove punctuation, and stopwords
def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        # Remove non-alphabetic tokens and stopwords
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens
    else:
        return []

# Define a function to extract verb-noun pairs from tokens using POS tagging
def extract_verb_noun_pairs(tokens):
    pos_tags = pos_tag(tokens)
    verb_noun_pairs = []
    invalid_pairs = []
    
    for i in range(len(pos_tags) - 1):
        word, tag = pos_tags[i]
        next_word, next_tag = pos_tags[i + 1]
        
        # Check for verb-noun pairs
        if tag.startswith('VB') and next_tag.startswith('NN'):
            verb_noun_pairs.append((word, next_word))
        else:
            invalid_pairs.append((word, next_word))
    
    return verb_noun_pairs, invalid_pairs

# Define a function to detect if a text is in English
def is_english(text):
    try:
        # Detect the language of the text
        return detect(text) == 'en'
    except:
        # If detection fails, assume it's not English
        return False

# 1. Process AI Titles (AI_titles_dict.csv)
print("Processing AI titles...")

# Load the file into a pandas DataFrame
ai_titles_df = pd.read_csv('sample_AI.csv')
print(f"Loaded {len(ai_titles_df)} AI titles.")

# Filter out non-English titles
ai_titles_df = ai_titles_df[ai_titles_df['Title'].apply(is_english)]
print(f"Filtered to {len(ai_titles_df)} English titles.")

# Preprocess the titles and extract verb-noun pairs
ai_titles_df['tokens'] = ai_titles_df['Title'].apply(preprocess_text)

# Initialize lists to store invalid pairs
ai_invalid_pairs = []

# Extract verb-noun pairs and collect invalid pairs
ai_titles_df['verb_noun_pairs'] = ai_titles_df['tokens'].apply(
    lambda tokens: extract_verb_noun_pairs(tokens)[0])
ai_titles_df['invalid_pairs'] = ai_titles_df['tokens'].apply(
    lambda tokens: extract_verb_noun_pairs(tokens)[1])

# Collect invalid pairs across all rows
ai_invalid_pairs.extend(ai_titles_df['invalid_pairs'].sum())

# Remove rows where no valid verb-noun pairs were found
ai_titles_df = ai_titles_df[ai_titles_df['verb_noun_pairs'].map(len) > 0]
print(f"Remaining AI titles after filtering: {len(ai_titles_df)}")

# Print out the invalid AI verb-noun pairs
if ai_invalid_pairs:
    print("\nInvalid AI Verb-Noun Pairs:")
    for pair in ai_invalid_pairs:
        print(f"Invalid Pair: {pair}")

# Save AI verb-noun pairs to a new CSV file
ai_titles_verb_noun_df = ai_titles_df[['Title', 'verb_noun_pairs']]
ai_titles_verb_noun_df.to_csv('AI_verb_noun.csv', index=False)
print("Saved AI verb-noun pairs to AI_verb_noun.csv")

# 2. Process Occupation Tasks (occupation_tasks_with_names.csv)
print("\nProcessing occupation tasks...")

# Load the occupation tasks data
occupation_df = pd.read_csv('sample_job.csv')
print(f"Loaded {len(occupation_df)} occupations.")

# Clean the "Occupation Name" column to remove extra spaces and newlines
occupation_df['Occupation Name'] = occupation_df['Occupation Name'].apply(lambda x: str(x).strip().replace("\n", " ").replace("\r", ""))

# Drop rows with missing or invalid 'Occupation Name'
occupation_df = occupation_df.dropna(subset=['Occupation Name'])
print(f"Filtered to {len(occupation_df)} valid occupations with non-empty names.")

# Initialize a dictionary to store results where key is 'Occupation Name' and value is verb-noun pairs
occupation_verb_noun_dict = defaultdict(list)
occupation_invalid_pairs = defaultdict(list)

# Iterate through each row to process the list of tasks
for idx, row in occupation_df.iterrows():
    occupation_name = row['Occupation Name']
    
    try:
        # Use ast.literal_eval instead of eval for safer list conversion
        tasks = ast.literal_eval(row['Tasks'])
        
        if isinstance(tasks, list):  # Check if tasks are properly formatted as a list
            for task in tasks:
                tokens = preprocess_text(task)
                verb_noun_pairs, invalid_pairs = extract_verb_noun_pairs(tokens)
                # Store valid verb-noun pairs
                if verb_noun_pairs:
                    occupation_verb_noun_dict[occupation_name].extend(verb_noun_pairs)
                # Store invalid pairs for review
                if invalid_pairs:
                    occupation_invalid_pairs[occupation_name].extend(invalid_pairs)
        else:
            print(f"Skipping row {idx}: Tasks column is not a list.")
    
    except (ValueError, SyntaxError):
        # Print only the skipped rows due to error in parsing
        print(f"Skipping row {idx}: Error parsing tasks for occupation '{occupation_name}'. Row content: {row}")

# Remove occupations where no valid verb-noun pairs were found
occupation_verb_noun_dict = {k: v for k, v in occupation_verb_noun_dict.items() if v}

# Save the dictionary to an Excel file for better readability
occupation_verb_noun_df = pd.DataFrame([(k, v) for k, v in occupation_verb_noun_dict.items()], columns=['Occupation Name', 'Verb-Noun Pairs'])
occupation_verb_noun_df.to_excel('occupation_verb_noun_pairs.xlsx', index=False)
print("Saved occupation verb-noun pairs to occupation_verb_noun_pairs.xlsx")

# Print a preview of the result
print("\nHere is a preview of the result:")
print(occupation_verb_noun_df.head())