"""
Parse HTML → JSON chunks
Input:  data/raw/labor_law_2019.html
Output: data/processed/labor_law_2019.json
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.labor_law_parser import LaborLawParser
from src.parser.validator import validate_chunks

INPUT_PATH = "data/raw/labor_law_2019.html"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "labor_law_2019.json")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read HTML
    print(f"Reading: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Parse
    parser = LaborLawParser()
    print("Parsing...")
    chunks = parser.parse_html(html)

    # Validate
    print("\n--- Validation Report ---")
    validate_chunks(chunks)
    print("-------------------------\n")

    # Confirm when saving if there are issues
    parser.save_json(chunks, OUTPUT_PATH)
    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()