import math


def hit_rate_k(
        recommendations: list[int], 
        relevant_movies: set[int],
        k: int,
          ) -> float:
    """ 
    from recommended movieId for the user, 
    check matching between top k recommendded movie and relevant movie
    """
    top_k = set(recommendations[:k])

    return float(bool(top_k & relevant_movies))

def ndcg_k(
        recommendations: list[int],
        relevant_movies: set[int],
        k: int
        )->float:
    """
    Ranking quality metric. Compares rankings to an ideal order
      where all relevant items are at the top of the list.
      cumulative gain is hit count of recommendations.
      DCG@K=sum over rank k movies {hit count/log_2(rank + 1)}
      IDGC@K= DCG with ideal recommendation order. every relevent movie place at the top
      DCG@K/IDCG@K
    """
    dcg = 0
    for i in range(k):
        if recommendations[i] in relevant_movies:
            dcg += (1/math.log2(i+2))

    ideal_relevant_count = min(len(relevant_movies), k)
    idcg = sum((1/math.log2(i+2)) for i in range(ideal_relevant_count))

    if idcg == 0:
        return 0
    else:
        return dcg/idcg
