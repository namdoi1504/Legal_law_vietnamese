import math


def matches_selector(chunk, selector):
    meta = chunk["metadata"]
    for key, value in selector.items():
        if key == "point" and chunk.get("expanded_parent"):
            # A complete parent clause contains its original top-level points.
            from src.parser.labor_law_parser import LaborLawParser

            points = LaborLawParser._outside_quotes(
                LaborLawParser.POINT_PATTERN, chunk["content"]
            )
            if value not in {match.group(1) for match in points}:
                return False
        elif meta.get(key) != value:
            return False
    return True


def retrieval_metrics(item, chunks):
    groups = item.get("expected_evidence_groups")
    if not groups:
        groups = [[{"article": a}] for a in item.get("expected_articles", [])]
        unique = []
        seen = set()
        for c in chunks:
            key = (c["metadata"].get("document_id"), c["metadata"].get("article"))
            if key not in seen:
                unique.append(c)
                seen.add(key)
        chunks = unique
    if not groups:
        return {"hit": None, "recall": None, "mrr": None, "ndcg": None}
    seen = set()
    gains = []
    for c in chunks:
        new = {
            i
            for i, g in enumerate(groups)
            if i not in seen and any(matches_selector(c, s) for s in g)
        }
        gains.append(len(new))
        seen.update(new)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(groups), len(chunks))))
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    return {
        "hit": int(bool(seen)),
        "recall": len(seen) / len(groups),
        "mrr": next((1 / (i + 1) for i, g in enumerate(gains) if g), 0),
        "ndcg": None if any(g > 1 for g in gains) else dcg / ideal if ideal else 0,
    }
