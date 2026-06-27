import re

class QueryUnderstanding:

    @staticmethod
    def analyze(query):
        query = query.lower()

        intent = "general"

        if any(x in query for x in ["what", "which"]):
            intent = "fact"

        elif any(x in query for x in ["explain", "describe"]):
            intent = "explanation"

        elif any(x in query for x in ["compare"]):
            intent = "comparison"

        entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)

        return {
            "intent": intent,
            "entities": entities
        }