import pandas as pd
from nltk.corpus import wordnet as wn
import spacy
import ast
import numpy as np

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")
print("spaCy model loaded.")

# Function to check if a word has a valid meaning in WordNet
def has_meaning_in_wordnet(word, pos_tag):
    return bool(wn.synsets(word, pos=pos_tag))

# Function to get the vector for a word in spaCy
def get_vector_in_spacy(word):
    doc = nlp(word)
    vector = doc.vector
    # If the vector has zero norm, return None
    if np.linalg.norm(vector) == 0:
        return None
    return vector

# Function to filter both meaningful and unmeaningful verb-noun pairs based on WordNet meaning and spaCy vectors
def filter_verb_noun_pairs(pairs):
    meaningful_pairs = []
    unmeaningful_pairs = []
    
    for verb, noun in pairs:
        # Check WordNet meanings
        has_wordnet_meaning = has_meaning_in_wordnet(verb, wn.VERB) and has_meaning_in_wordnet(noun, wn.NOUN)
        # Check spaCy vectors
        has_spacy_vector = get_vector_in_spacy(verb) is not None and get_vector_in_spacy(noun) is not None
        
        # Add to meaningful pairs only if both checks pass, otherwise add to unmeaningful pairs
        if has_wordnet_meaning and has_spacy_vector:
            meaningful_pairs.append((verb, noun))
        else:
            unmeaningful_pairs.append((verb, noun))
    
    return meaningful_pairs, unmeaningful_pairs

# Function to apply the filtering to the occupation data
def process_occupation_data(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)
    
    # Convert the string representation of lists into actual lists of tuples
    df['Verb-Noun Pairs'] = df['Verb-Noun Pairs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Create new columns for meaningful and unmeaningful pairs by applying the filter function
    df['Meaningful Verb-Noun Pairs'], df['Unmeaningful Verb-Noun Pairs'] = zip(*df['Verb-Noun Pairs'].apply(filter_verb_noun_pairs))
    
    # Identify rows where the 'Meaningful Verb-Noun Pairs' column contains an empty list
    empty_verb_noun_rows = df[df['Meaningful Verb-Noun Pairs'].apply(lambda x: len(x) == 0)]
    
    # Save these rows to a separate file
    empty_verb_noun_rows.to_csv('empty_occupation_verb_noun_pairs.csv', index=False)
    
    # Remove rows where 'Meaningful Verb-Noun Pairs' is empty
    df = df[df['Meaningful Verb-Noun Pairs'].apply(lambda x: len(x) > 0)]
    
    # Save the remaining rows with non-empty 'Meaningful Verb-Noun Pairs'
    df.to_csv('filtered_occupation_pairs.csv', index=False)

    # Preview the output data
    print("Output with Meaningful and Unmeaningful Pairs (Filtered):")
    print(df.head())

# File containing occupation data
file_name = '/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/occupation_verb_noun.csv'

# Process the occupation data
process_occupation_data(file_name)


# conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/3_filter_prepare_compare/filter_occupation_hybrid.py"


# conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/3_filter_prepare_compare/filter_occupation_hybrid.py"