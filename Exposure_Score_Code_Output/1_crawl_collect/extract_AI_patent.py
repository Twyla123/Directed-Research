import os
import csv

download_dir = "/Users/twylazhang/Desktop/Directed Research/code_output/AI_Patent"

# List of keywords
keywords = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Neural Networks",
    "Natural Language Processing",
    "Reinforcement Learning",
    "Cognitive Computing",
    "AI Automation"
]

# Initialize an empty dictionary to store the titles
titles_dict = {}

# Iterate over each file in the download directory
for filename in os.listdir(download_dir):
    if filename.endswith(".csv"):  # Ensure the file is a CSV
        # Identify which keyword this file corresponds to by its filename
        for keyword in keywords:
            if keyword in filename:  # Check if the keyword (with spaces) is in the filename
                # Open and read the CSV file
                csv_path = os.path.join(download_dir, filename)
                with open(csv_path, newline='', encoding='utf-8') as csvfile:
                    # Skip the first row (which contains the search URL)
                    next(csvfile)

                    # Now process the actual data starting from the second row
                    reader = csv.DictReader(csvfile)  # Using DictReader to parse rows into dictionaries

                    # Debug: print headers and rows to inspect the structure
                    headers = reader.fieldnames
                    print(f"Processing file: {filename}")
                    print(f"Headers: {headers}")  # Print the actual headers to verify 'title' exists

                    # Ensure the 'title' column is present
                    for row in reader:
                        title = row.get('title', 'Unknown Title')  # Using lowercase 'title' as the correct column name
                        
                        # Add the title to the dictionary
                        if keyword in titles_dict:
                            titles_dict[keyword].append(title)
                        else:
                            titles_dict[keyword] = [title]
                break  # Break the loop after finding the matching keyword

# Print the resulting dictionary of titles
print(titles_dict)

# Path to save the CSV file
csv_output_file = "/Users/twylazhang/Desktop/Directed Research/code_output/AI_titles_dict.csv"

# Save the dictionary to a CSV file
with open(csv_output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Keyword", "Title"])  # Write the header
    for keyword, titles in titles_dict.items():
        for title in titles:
            writer.writerow([keyword, title])  # Write each title under the corresponding keyword

print(f"Dictionary saved to {csv_output_file}")