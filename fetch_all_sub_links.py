import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

# Get the directory where the current script is located
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(PROJECT_DIR, "fetched_urls.txt")

BASE_URL = "https://docs.chaicode.com/youtube/getting-started/"

def get_youtube_urls(base_url):
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {base_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    found_urls = set()
    found_urls.add(base_url)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/youtube/") and len(href.strip("/").split("/")) > 2:
            full_url = urljoin(base_url, href)
            found_urls.add(full_url)

    return sorted(found_urls)

def save_urls_to_file(urls, filename):
    with open(filename, "w") as f:
        for url in urls:
            f.write(f"{url}\n")
    print(f"Saved {len(urls)} URLs to {filename}")

# Run and store
urls_list = get_youtube_urls(BASE_URL)
if __name__=="__main__":
    save_urls_to_file(urls_list, OUTPUT_FILE)