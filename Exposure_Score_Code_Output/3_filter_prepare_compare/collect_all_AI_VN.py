import pandas as pd
import ast

# Load the dataframe from the CSV file
df = pd.read_csv('/Users/twylazhang/Desktop/Directed Research/code_output/2_extract_verb_noun/AI_verb_noun.csv')

# Initialize an empty list to store all verb-noun pairs
all_AI_verb_noun = []

# Iterate through the DataFrame
for index, row in df.iterrows():
    # Convert the string representation of the list of tuples into an actual list of tuples
    verb_noun_pairs = ast.literal_eval(row['Verb-Noun Pairs'])
    
    # Add the verb-noun pairs to the final list
    all_AI_verb_noun.extend(verb_noun_pairs)

# Create a new dataframe from the list of verb-noun pairs
output_df = pd.DataFrame(all_AI_verb_noun, columns=['Verb', 'Noun'])

# Save the dataframe to a CSV file
output_df.to_csv('all_AI_verb_noun.csv', index=False)

print("Output saved to all_AI_verb_noun.csv")