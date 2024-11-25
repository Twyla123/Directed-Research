import pandas as pd
import ast
import spacy
from spacy.matcher import PhraseMatcher
from concurrent.futures import ThreadPoolExecutor

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Optimized: Batch size for nlp.pipe
BATCH_SIZE = 500

# Function to create a PhraseMatcher from a batch of synonym pairs
def add_synonym_batch_to_matcher(matcher, synonym_batch):
    for index, row in synonym_batch.iterrows():
        original_verb = row['Original Verb']
        original_noun = row['Original Noun']
        synonym_verb = row['Synonym Verb']
        synonym_noun = row['Synonym Noun']
        
        # Create patterns from both original and synonym verb-noun pairs
        original_pattern = nlp(f"{original_verb} {original_noun}")
        synonym_pattern = nlp(f"{synonym_verb} {synonym_noun}")
        
        # Add both original and synonym patterns to the matcher
        matcher.add("VERB_NOUN_PAIRS", [original_pattern, synonym_pattern])
    return matcher

# Function to process occupation tasks in parallel
def match_occupation_tasks_with_synonyms(matcher, occupation_tasks, similarity_threshold=0.5):
    matched_pairs = []
    
    # Prepare task texts for batch processing
    task_texts = []
    for task_pairs in occupation_tasks:
        if isinstance(task_pairs, list) and task_pairs:
            valid_pairs = [(verb, noun) for pair in task_pairs if isinstance(pair, tuple) and len(pair) == 2]
            if valid_pairs:
                task_texts.append(" ".join([f"{verb} {noun}" for verb, noun in valid_pairs]))

    task_docs = list(nlp.pipe(task_texts, batch_size=BATCH_SIZE))  # Larger batch size for efficiency
    
    for doc in task_docs:
        matches = matcher(doc)
        for match_id, start, end in matches:
            matched_phrase = doc[start:end]
            
            # Calculate semantic similarity
            for token in doc:
                similarity = matched_phrase.similarity(token)
                if similarity >= similarity_threshold:
                    matched_pairs.append((matched_phrase.text, doc.text, similarity))
    
    return matched_pairs

# Function to load synonym pairs in batches and match tasks in parallel
def process_in_batches(synonym_file, occupation_df, batch_size=10000):
    synonym_df = pd.read_csv(synonym_file)
    all_matched_pairs = []
    
    # Split synonym pairs into batches to avoid memory overload
    for batch_start in range(0, len(synonym_df), batch_size):
        batch_end = min(batch_start + batch_size, len(synonym_df))
        synonym_batch = synonym_df[batch_start:batch_end]
        print(f"Processing synonym batch {batch_start}-{batch_end} out of {len(synonym_df)}")

        # Create a new matcher for each batch to limit memory usage
        matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        matcher = add_synonym_batch_to_matcher(matcher, synonym_batch)

        # Process occupation tasks in parallel
        with ThreadPoolExecutor() as executor:
            future_matches = [
                executor.submit(match_occupation_tasks_with_synonyms, matcher, row['Meaningful Verb-Noun Pairs'])
                for idx, row in occupation_df.iterrows()
            ]

            for future in future_matches:
                all_matched_pairs.extend(future.result())
    
    return all_matched_pairs

# === Main Execution ===
ai_synonym_pairs_file = '/Users/twylazhang/Desktop/Directed Research/code_output/ai_synonym_pairs.csv'
occupation_df = pd.read_csv('/Users/twylazhang/Desktop/Directed Research/code_output/filtered_occupation_pairs.csv')
occupation_df['Meaningful Verb-Noun Pairs'] = occupation_df['Meaningful Verb-Noun Pairs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Process AI synonym pairs and match occupations in batches
all_matched_pairs = process_in_batches(ai_synonym_pairs_file, occupation_df, batch_size=100000)

# Save matched pairs for verification
matched_pairs_df = pd.DataFrame(all_matched_pairs, columns=['Matched Phrase', 'Occupation Task', 'Similarity'])
matched_pairs_df.to_csv('/Users/twylazhang/Desktop/Directed Research/code_output/sample_run/matched_verb_noun_pairs.csv', index=False)

print("\nMatched Verb-Noun Pairs with Synonyms and Similarity Preview:")
print(matched_pairs_df.head())

# conda activate spacy_env
# python "/Users/twylazhang/Desktop/Directed Research/code_output/sample_run/wordNet.py"
# spaCy’s PhraseMatcher and semantic similarity calculations to compare verb-noun pairs between AI synonym pairs and occupation tasks