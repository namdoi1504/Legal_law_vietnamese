"""
Build BM25 index từ parsed JSON
Input:  data/processed/labor_law_2019.json
Output: data/processed/bm25_index.pkl
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever.bm25_retriever import BM25LegalRetriever

INPUT_PATH = "data/processed/labor_law_2019.json"
OUTPUT_PATH = "data/processed/bm25_index.pkl"


def main():
    # Load chunks
    print(f"Loading chunks from: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Build index
    retriever = BM25LegalRetriever()
    retriever.build_index(chunks)

    # Save
    retriever.save_index(OUTPUT_PATH)
    print(f"BM25 index saved to: {OUTPUT_PATH}")

    # Quick smoke test
    print("\n--- Smoke Test ---")
    test_queries = [
        "nghỉ thai sản bao nhiêu tháng",
        "lương tối thiểu vùng",
        "làm thêm giờ tối đa"
    ]
    for q in test_queries:
        results = retriever.search(q, top_k=3)
        top = results[0]["metadata"]["article"] if results else "No result"
        print(f"  '{q}' → {top}")
    print("------------------\n")


if __name__ == "__main__":
    main()