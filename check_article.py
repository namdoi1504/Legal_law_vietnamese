# scripts/print_articles.py
import json
import sys

# Fix encoding cho Windows
sys.stdout.reconfigure(encoding='utf-8')

with open('data/processed/labor_law_2019.json', encoding='utf-8') as f:
    chunks = json.load(f)

seen = set()
articles = []
for c in chunks:
    article = c['metadata']['article']
    if article not in seen:
        seen.add(article)
        articles.append({
            "article": article,
            "title": c['metadata']['article_title'],
            "content_preview": c['content'][:120].replace('\n', ' ')
        })

print(f"Tổng số Điều: {len(articles)}\n")
print(f"{'Điều':<12} {'Tiêu đề':<45} {'Nội dung mẫu'}")
print("-" * 120)
for a in articles:
    print(f"{a['article']:<12} {a['title'][:45]:<45} {a['content_preview']}")