import json


def token_upper_bound(text):
    # Conservative byte-level bound: no provider tokenizer download is required.
    # Deliberately under-fills some models rather than silently truncating evidence.
    return len(text.encode("utf-8"))


def source_label(meta):
    parts = [meta.get("article", "")]
    if meta.get("clause"):
        parts.append(f"Khoản {meta['clause']}")
    if meta.get("point"):
        parts.append(f"Điểm {meta['point']}")
    parts.append(f"{meta.get('law_name','')} {meta.get('law_year','')}")
    return " — ".join(parts)


def serialize_context(chunks):
    return json.dumps(
        [
            {
                "source_id": c["source_id"],
                "citation": source_label(c["metadata"]),
                "content": c["content"],
            }
            for c in chunks
        ],
        ensure_ascii=False,
    )


def build_context(chunks, budget, expand_parents=True):
    selected, seen = set(), set()
    result = []
    for original in chunks:
        meta = dict(original["metadata"])
        key = meta["chunk_id"]
        if key in selected or meta.get("parent_id") in seen:
            continue
        candidates = []
        if expand_parents and meta.get("point") and meta.get("parent_content"):
            parent_meta = {**meta, "chunk_id": meta["parent_id"], "point": None}
            candidates.append(
                {
                    **original,
                    "content": meta["parent_content"],
                    "metadata": parent_meta,
                    "expanded_parent": True,
                }
            )
        candidates.append({**original, "metadata": meta, "expanded_parent": False})
        for candidate in candidates:
            candidate = {**candidate, "source_id": f"S{len(result)+1}"}
            if token_upper_bound(serialize_context(result + [candidate])) <= budget:
                result.append(candidate)
                selected.add(key)
                if candidate["expanded_parent"]:
                    seen.add(meta["parent_id"])
                break
    return result
