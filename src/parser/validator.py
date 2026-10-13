def validate_chunks(chunks, expected_articles=None):
    issues, seen = [], set()
    for c in chunks:
        if c.chunk_id in seen:
            issues.append(f"Duplicate: {c.chunk_id}")
        seen.add(c.chunk_id)
        if not c.content.strip() or not c.article_number.isdigit():
            issues.append(f"Invalid: {c.chunk_id}")
        if c.metadata.get("point") and not c.metadata.get("parent_id"):
            issues.append(f"Missing parent: {c.chunk_id}")
    if expected_articles is not None:
        actual = {int(c.article_number) for c in chunks}
        if actual != set(expected_articles):
            issues.append(
                f"Article mismatch: missing={set(expected_articles)-actual}, extra={actual-set(expected_articles)}"
            )
    if issues:
        raise ValueError("\n".join(issues))
    return {
        "chunks": len(chunks),
        "articles": len({c.article_number for c in chunks}),
        "issues": [],
    }
