from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, docs):

        pairs = []

        for d in docs:
            text = d.get(
                "Answer",
                d.get("Topic", "")
            )

            pairs.append([query, text])

        scores = self.model.predict(pairs)

        for d, s in zip(docs, scores):
            d["_rerank"] = float(s)

        docs.sort(
            key=lambda x: x["_rerank"],
            reverse=True
        )

        return docs