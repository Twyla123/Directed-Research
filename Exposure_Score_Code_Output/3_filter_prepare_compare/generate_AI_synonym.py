import pandas as pd
import ast
import spacy
import pickle
from nltk.corpus import wordnet as wn
from spacy.matcher import PhraseMatcher
from functools import lru_cache
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Cache file path to store the synonyms
SYNONYM_CACHE_FILE = 'synonym_cache.pkl'

# Load the synonym cache from a file, if it exists
if os.path.exists(SYNONYM_CACHE_FILE):
    with open(SYNONYM_CACHE_FILE, 'rb') as cache_file:
        synonym_cache = pickle.load(cache_file)
else:
    synonym_cache = {}  # Initialize cache if file doesn't exist

# Function to save the synonym cache to a file
def save_synonym_cache():
    with open(SYNONYM_CACHE_FILE, 'wb') as cache_file:
        pickle.dump(synonym_cache, cache_file)

# Function to get synonyms with POS filtering for more accurate matching
@lru_cache(maxsize=None)
def get_synonyms_pos_filtered(word, pos_tag):
    synonyms = set()
    for syn in wn.synsets(word):
        if syn.pos() == pos_tag:  # Only consider synonyms with the correct part of speech
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
    return synonyms

# Function to generate all possible synonym combinations for verb-noun pairs with POS filtering
def get_synonym_pairs_pos(verb, noun):
    # Check if we've already generated synonyms for this pair
    if (verb, noun) in synonym_cache:
        return synonym_cache[(verb, noun)]
    
    # Otherwise, generate synonyms for the verb and noun
    verb_synonyms = get_synonyms_pos_filtered(verb, wn.VERB)
    noun_synonyms = get_synonyms_pos_filtered(noun, wn.NOUN)
    
    synonym_pairs = [(v, n) for v in verb_synonyms for n in noun_synonyms]
    
    # Cache the result and save it to file
    synonym_cache[(verb, noun)] = synonym_pairs
    save_synonym_cache()  # Save the updated cache
    
    return synonym_pairs

# Load your CSV file with 'Verb' and 'Noun' columns
file_path = '/Users/twylazhang/Desktop/Directed Research/code_output/filtered_AI_verb_noun_meaningful.csv'  # Replace with the correct path
ai_titles_df = pd.read_csv(file_path)

# Combine 'Verb' and 'Noun' columns into 'Verb-Noun Pairs' column
ai_titles_df['Verb-Noun Pairs'] = list(zip(ai_titles_df['Verb'], ai_titles_df['Noun']))

# Extract the verb-noun pairs from the newly created 'Verb-Noun Pairs' column
ai_verb_noun_pairs = ai_titles_df['Verb-Noun Pairs'].tolist()

# Create a list to store all synonym pairs
all_synonym_pairs = []

# Generate synonym pairs for each verb-noun pair
for pair in ai_verb_noun_pairs:
    verb, noun = pair
    synonym_pairs = get_synonym_pairs_pos(verb, noun)
    for synonym_pair in synonym_pairs:
        all_synonym_pairs.append((verb, noun, synonym_pair[0], synonym_pair[1]))

# Save the synonym pairs to a CSV file
synonym_pairs_df = pd.DataFrame(all_synonym_pairs, columns=['Original Verb', 'Original Noun', 'Synonym Verb', 'Synonym Noun'])
synonym_pairs_df.to_csv('ai_synonym_pairs.csv', index=False)

print("\nSynonym pairs saved to 'ai_synonym_pairs.csv'.")


# conda activate spacy_env
# python /Users/twylazhang/Desktop/Directed Research/code_output/3_filter_prepare_compare/generate_AI_synonym.py"