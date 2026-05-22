# Chạy kiểm tra chất lượng parse
def validate_chunks(chunks):
    issues = []
    for c in chunks:
        if len(c.content) < 20:
            issues.append(f"Too short: {c.chunk_id}")
        if not c.article_number.isdigit():
            issues.append(f"Bad article number: {c.chunk_id}")
        if len(c.embed_text) > 2000:
            issues.append(f"Too long (may need re-split): {c.chunk_id}")

    print(f"Total chunks: {len(chunks)}")
    print(f"Issues found: {len(issues)}")
    for i in issues[:10]:
        print(f"  - {i}")