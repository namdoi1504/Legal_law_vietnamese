"""
Crawl HTML Bộ luật Lao động 2019 từ thuvienphapluat.vn
Lưu raw HTML vào data/raw/
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.crawler.base_crawler import LegalCrawler

LABOR_LAW_URL = "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-luat-lao-dong-2019-333670.aspx"

OUTPUT_DIR = "data/raw"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    crawler = LegalCrawler(delay=2.0)

    print(f"Crawling: {LABOR_LAW_URL}")
    html = crawler.fetch(LABOR_LAW_URL)

    output_path = os.path.join(OUTPUT_DIR, "labor_law_2019.html")
    crawler.save_raw(html, "labor_law_2019")

    print(f"Saved to {output_path}")
    print(f"HTML size: {len(html):,} characters")


if __name__ == "__main__":
    main()
