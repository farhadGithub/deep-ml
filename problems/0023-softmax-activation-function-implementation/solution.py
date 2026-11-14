import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    results = []
    max_score = max(scores)
    for score in scores:
        results.append(math.exp(score-max_score))
    total = sum(results)
    for i, result in enumerate(results):
        results[i] = round(result/total,4)
    return results