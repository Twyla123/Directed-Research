import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from playwright.sync_api import sync_playwright

# Step 1: Scrape the occupation names and codes from the base URL
base_url = 'https://www.onetonline.org/find/all'
response = requests.get(base_url)

occupation_list = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all occupation links that contain '/link/summary/'
    occupation_links = soup.find_all('a', href=True)
    
    # Extract occupation codes and names only from the summary URLs
    for link in occupation_links:
        href = link['href']
        full_url = 'https://www.onetonline.org' + href if href.startswith('/link/summary/') else href
        
        if 'https://www.onetonline.org/link/summary/' in full_url:
            occupation_code = full_url.split('/')[-1]
            occupation_name = link.text.strip()
            occupation_list.append({'Occupation Name': occupation_name, 'Occupation Code': occupation_code})
    
    # Convert the occupation data to a DataFrame for further processing
    occupation_df = pd.DataFrame(occupation_list)
    print(f"Successfully scraped {len(occupation_list)} occupations.")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")

# Function to scrape task importance and descriptions for a given occupation
def scrape_task_details(occupation_code):
    """Scrape task importance and descriptions for a given occupation."""
    tasks_data = []
    expanded_task_count = 0  # Counter to keep track of how many tasks are expanded
    
    url = f"https://www.onetonline.org/link/details/{occupation_code}"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url)
            time.sleep(2)  # Allow some time for the page to load
            
            # Expand the tasks section if the button exists
            expand_button_selector = 'button[data-bs-target=".long_Tasks"]'
            expand_button = page.query_selector(expand_button_selector)
            if expand_button and expand_button.get_attribute('aria-expanded') == 'false':
                print(f"Expanding tasks for occupation code {occupation_code}...")
                expand_button.click()
                page.wait_for_selector('ul.list-unstyled.m-0', timeout=10000)  # Wait for the tasks to load
            
            # Scrape task descriptions and importance
            task_rows = page.query_selector_all('tr')
            for row in task_rows:
                importance = row.query_selector('td[data-title="Importance"]')
                task_description = row.query_selector('td[data-title="Task"]')
                if importance and task_description:
                    tasks_data.append({
                        'Task Description': task_description.inner_text().strip(),
                        'Importance': importance.inner_text().strip()
                    })
                    expanded_task_count += 1  # Increment counter for each task expanded

            print(f"Expanded {expanded_task_count} tasks for occupation code {occupation_code}.")
            
        except Exception as e:
            print(f"Failed to scrape data for {occupation_code}: {e}")
        
        browser.close()
    
    return tasks_data

# Step 3: Iterate over each occupation code and scrape task importance
all_occupation_tasks = {}

for index, row in occupation_df.iterrows():
    occupation_name = row['Occupation Name']
    occupation_code = row['Occupation Code']
    
    print(f"Scraping tasks for {occupation_name} ({occupation_code})...")
    task_data = scrape_task_details(occupation_code)
    
    # Store the data in a dictionary with occupation name as the key
    all_occupation_tasks[occupation_name] = task_data
    
    # Add a delay between requests to avoid overloading the server
    time.sleep(2)

# Step 4: Convert the task data into a DataFrame
task_data_list = []
for occupation_name, tasks in all_occupation_tasks.items():
    for task in tasks:
        task_data_list.append({
            'Occupation Name': occupation_name,
            'Task Description': task['Task Description'],
            'Importance': task['Importance']
        })

task_df = pd.DataFrame(task_data_list)

# Save the tasks and importance data to a CSV file
task_df.to_csv('occupation_tasks_with_importance.csv', index=False)

# Optionally, save it as an Excel file as well
task_df.to_excel('occupation_tasks_with_importance.xlsx', index=False)

print("Task scraping completed and data saved.")