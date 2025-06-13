import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

# Base URL for Simpsons episodes
BASE_URL = "https://www.springfieldspringfield.co.uk/view_episode_scripts.php?tv-show=the-simpsons&episode="

# Create folder to save scripts
os.makedirs('simpsons_scripts', exist_ok=True)

# Loop through seasons 1–36 and episodes 1–25
for season in tqdm(range(1, 37), desc="Seasons"):
    for episode in range(1, 26):  # Up to 25 episodes per season
        ep_code = f"s{season:02d}e{episode:02d}"
        full_url = BASE_URL + ep_code
        
        try:
            # Send a request to the episode URL
            response = requests.get(full_url)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find the script container
            script_container = soup.find('div', class_='scrolling-script-container')
            
            # Check if the container is found and has text
            if script_container and script_container.get_text(strip=True):
                script_text = script_container.get_text(separator='\n').strip()

                # Save the script to a file
                filename = os.path.join('simpsons_scripts', f"{ep_code}.txt")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(script_text)
            else:
                print(f"No script found for {ep_code}")
                
        except Exception as e:
            print(f"Failed to download {ep_code}: {e}")
        
        # Be polite with a delay between requests
        time.sleep(0.5)
