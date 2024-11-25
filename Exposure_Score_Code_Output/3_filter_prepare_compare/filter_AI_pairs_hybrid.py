import pandas as pd
from nltk.corpus import wordnet as wn
import spacy
import numpy as np

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_md")
print("spaCy model loaded.")

# Function to get the vector for a word in spaCy
def get_vector(word):
    vector = nlp(word).vector
    # If the vector has a zero norm, return None
    if np.linalg.norm(vector) == 0:
        return None
    return vector

# Function to check if a word has a valid meaning in WordNet
def has_meaning_in_wordnet(word, pos_tag):
    return bool(wn.synsets(word, pos=pos_tag))

# Function to filter verb-noun pairs based on WordNet meaning and spaCy vector
def filter_verb_noun_pairs(df):
    def is_meaningful(row):
        verb, noun = row['Verb'], row['Noun']
        # Check WordNet meanings
        has_wordnet_meaning = has_meaning_in_wordnet(verb, wn.VERB) and has_meaning_in_wordnet(noun, wn.NOUN)
        # Check spaCy vectors
        has_spacy_vector = get_vector(verb) is not None and get_vector(noun) is not None
        return has_wordnet_meaning and has_spacy_vector
    
    # Filter the rows that are meaningful
    meaningful_df = df[df.apply(is_meaningful, axis=1)]
    # Filter the rows that are unmeaningful
    unmeaningful_df = df[~df.apply(is_meaningful, axis=1)]
    
    return meaningful_df, unmeaningful_df

# Load the verb-noun pairs from your file
df = pd.read_csv('all_AI_verb_noun.csv')

# Filter the verb-noun pairs into meaningful and unmeaningful
meaningful_df, unmeaningful_df = filter_verb_noun_pairs(df)

# Drop duplicate verb-noun pairs based on both columns
meaningful_df = meaningful_df.drop_duplicates(subset=['Verb', 'Noun'])

# Save the meaningful pairs to one CSV
meaningful_df.to_csv('filtered_AI_verb_noun_meaningful.csv', index=False)

# Save the unmeaningful pairs to another CSV
unmeaningful_df.to_csv('filtered_AI_verb_noun_unmeaningful.csv', index=False)

# Preview both the filtered data
print("Meaningful Pairs Preview:")
print(meaningful_df.head())

print("\nUnmeaningful Pairs Preview:")
print(unmeaningful_df.head())


# conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/3_filter_prepare_compare/filter_AI_pairs_hybrid.py"


# conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/3_filter_prepare_compare/filter_AI_pairs_hybrid.py"