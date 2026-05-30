# # check_chunks.py
# import json

# with open('data/processed/labor_law_2019.json', encoding='utf-8') as f:
#     chunks = json.load(f)

# print(f'Total chunks: {len(chunks)}')

# # Xem Điều 35
# print("\n--- Điều 35 ---")
# for c in chunks:
#     if c['metadata']['article'] == 'Điều 35':
#         print(c['chunk_id'], '|', c['metadata'].get('clause'))

# # Xem Điều 105
# print("\n--- Điều 105 ---")
# for c in chunks:
#     if c['metadata']['article'] == 'Điều 105':
#         print(c['chunk_id'], '|', c['metadata'].get('clause'))




# check_content.py
import json

with open('data/processed/labor_law_2019.json', encoding='utf-8') as f:
    chunks = json.load(f)

for dieu in ['Điều 125', 'Điều 112', 'Điều 163', 'Điều 124']:
    print(f"\n{'='*50}")
    print(f"{dieu}:")
    for c in chunks:
        if c['metadata']['article'] == dieu:
            print(f"  chunk_id: {c['chunk_id']}")
            print(f"  title: {c['metadata']['article_title']}")
            print(f"  content[:150]: {c['content'][:150]}")
            break


