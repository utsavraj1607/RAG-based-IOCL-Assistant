class AnswerValidator:

    @staticmethod
    def validate(results):

        if not results:
            return False

        score = results[0].get(
            "_score",
            0
        )

        rerank = results[0].get(
            "_rerank",
            0
        )

        confidence = (
            score * 0.5 +
            rerank * 0.5
        )

        return confidence > 0.5