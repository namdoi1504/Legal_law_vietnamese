import httpx
from bs4 import BeautifulSoup
import time


class LegalCrawler:
    def __init__(self, delay=1.5):
        self.delay = delay
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; LegalRAGBot/1.0)"}

    def fetch(self, url: str) -> str:
        with httpx.Client(headers=self.headers, timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()
        time.sleep(self.delay)
        return resp.text

    def save_raw(self, html: str, filename: str):
        path = f"data/raw/{filename}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
