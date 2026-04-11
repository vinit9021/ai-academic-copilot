from rapidfuzz import fuzz

def is_similar(query, keywords, threshold=80):
    for word in keywords:
        if fuzz.partial_ratio(query, word) > threshold:
            return True
    return False


def route_query(query: str):
    query = query.lower()

    keywords = ["quiz", "question", "questions", "test", "exam"]

    if is_similar(query, keywords):
        return "examiner"
    else:
        return "tutor"