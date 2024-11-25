import pandas as pd
import spacy

# Load the spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Function to extract verb-noun pairs using dependency parsing
def extract_verb_noun_pairs(text):
    doc = nlp(text)
    pairs = []

    # Iterate through the dependency parse of the sentence
    for token in doc:
        # Look for verbs (root) and their direct objects (nouns)
        if token.pos_ == 'VERB':
            for child in token.children:
                if child.dep_ == 'dobj' and child.pos_ == 'NOUN':
                    pairs.append((token.lemma_, child.lemma_))

    return pairs

# Function to process AI data
def process_ai_data(df):
    # Add a new column 'Verb-Noun Pairs' that stores the extracted verb-noun pairs for each description
    df['Verb-Noun Pairs'] = df['Title'].apply(extract_verb_noun_pairs)
    return df

# Function to process occupation data
def process_occupation_data(df):
    # Add a new column 'Verb-Noun Pairs' that stores the extracted verb-noun pairs for each task description
    df['Verb-Noun Pairs'] = df['Task Description'].apply(extract_verb_noun_pairs)
    return df

# Function to print the first few rows of the DataFrame for debugging
def print_debug_info(df, name):
    print(f"\n{name} - First 5 rows:")
    print(df.head())

# === Processing AI data ===
# Load AI data from 'AI_verb_noun.csv'
ai_df = pd.read_csv('1_crawl_collect/AI_titles_dict.csv')

# Print AI Data Columns for debugging
print("AI Data Columns: ", ai_df.columns)

# Debug: Print the first few rows of the AI DataFrame
print_debug_info(ai_df, "AI Data")

# Process the AI data to extract verb-noun pairs from the 'Description' column
ai_df = process_ai_data(ai_df)

# Save the processed AI data (including original columns and new 'Verb-Noun Pairs' column) to a CSV file
ai_df.to_csv('2_extract_verb_noun/AI_verb_noun.csv', index=False)

# === Processing Occupation data ===
# Load occupation data from 'occupation_tasks_with_importance.xlsx'
occupation_df = pd.read_excel('1_crawl_collect/occupation_tasks_with_importance.xlsx')

# Print Occupation Data Columns for debugging
print("Occupation Data Columns: ", occupation_df.columns)

# Debug: Print the first few rows of the Occupation DataFrame
print_debug_info(occupation_df, "Occupation Data")

# Process the occupation data to extract verb-noun pairs from the 'Task Description' column
occupation_df = process_occupation_data(occupation_df)

# Save the processed Occupation data (including original columns and new 'Verb-Noun Pairs' column) to a CSV file
occupation_df.to_csv('2_extract_verb_noun/occupation_verb_noun.csv', index=False)

# === Final Output Previews ===
# Print a preview of the processed AI data
print("\nProcessed AI Data with Verb-Noun Pairs:")
print(ai_df[['Title', 'Verb-Noun Pairs']].head())

# Print a preview of the processed Occupation data
print("\nProcessed Occupation Data with Verb-Noun Pairs:")
print(occupation_df[['Task Description', 'Verb-Noun Pairs']].head())

#conda activate spacy_env
#python "/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/dependency_extract_verb_noun.py"