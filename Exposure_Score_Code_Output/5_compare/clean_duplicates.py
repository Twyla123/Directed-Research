import pandas as pd

matched_pairs_df = pd.read_csv('/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/compare_WordEmbedding.csv')

# Example of 'Matched Verb-Noun Pairs from AI' column
# matched_pairs_df['Matched Verb-Noun Pairs from AI'] contains values like "book availability, distribute book, reproduce book, distribute book"

# Function to clean and drop duplicate pairs from the 'Matched Verb-Noun Pairs from AI' column
def clean_matched_pairs(df, column_name='Matched Verb-Noun Pairs from AI'):
    # Apply function to split the string, convert to set (to remove duplicates), and then rejoin into a cleaned string
    df[column_name] = df[column_name].apply(
        lambda x: ', '.join(set(x.split(', '))) if isinstance(x, str) else x
    )
    return df

# Apply the cleaning function to the DataFrame
matched_pairs_df = clean_matched_pairs(matched_pairs_df)

# Save the cleaned DataFrame
matched_pairs_df.to_csv('/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/cleaned_matched_pairs.csv', index=False)
matched_pairs_df.to_excel('/Users/twylazhang/Desktop/Directed Research/code_output/5_compare/cleaned_matched_pairs.xlsx', index=False)

# Print a preview of the cleaned DataFrame
print(matched_pairs_df.head())