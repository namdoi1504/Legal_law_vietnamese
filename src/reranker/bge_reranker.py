from configs.settings import settings


class BGEReranker:
    def __init__(self, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity

            model = CrossEncoder(
                settings.RERANKER_MODEL,
                device=settings.MODEL_DEVICE,
                activation_fn=Identity(),
            )
        self.model = model

    def rerank(self, query, chunks, top_k=None):
        top_k = settings.TOP_K_RERANK if top_k is None else top_k
        if not chunks or top_k <= 0:
            return []
        pairs = [
            (
                query,
                f"{c['metadata'].get('article','')} — {c['metadata'].get('article_title','')}\n{c['content']}",
            )
            for c in chunks
        ]
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        ranked = sorted(
            [
                {**c, "metadata": dict(c["metadata"]), "rerank_score": float(s)}
                for s, c in zip(scores, chunks)
            ],
            key=lambda c: c["rerank_score"],
            reverse=True,
        )
        cap = settings.RERANK_MAX_PER_ARTICLE
        if not cap:
            return ranked[:top_k]
        selected, overflow, counts = [], [], {}
        for c in ranked:
            m = c["metadata"]
            key = (m.get("document_id"), m.get("document_version"), m.get("article"))
            if counts.get(key, 0) < cap:
                selected.append(c)
                counts[key] = counts.get(key, 0) + 1
            else:
                overflow.append(c)
        return sorted(
            (selected[:top_k] + overflow[: max(0, top_k - len(selected))]),
            key=lambda c: c["rerank_score"],
            reverse=True,
        )
