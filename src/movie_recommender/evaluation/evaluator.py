import pandas as pd

from movie_recommender.evaluation.metrics import (
    hit_rate_k,
    ndcg_k,
)


def evaluate_model(
    model,
    model_name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,  # set of movies for each user tested on, assumed to be relevant
    k: int,
    split_name: str = "validation",
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one fitted recommendation model."""

    if k <= 0:
        raise ValueError("k must be positive.")

    candidate_catalog = set(
        train["movieId"].astype(int)
    )

    seen_by_user = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    relevant_by_user = (
        evaluation.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    evaluation_rows = []
    cold_start_users = 0

    for user_id, relevant_movies in relevant_by_user.items():
        # The hidden relevant movie must be in the training
        # catalog for a warm-start evaluation.
        if not relevant_movies.issubset(candidate_catalog):
            cold_start_users += 1
            continue

        if user_id not in seen_by_user:
            raise ValueError(
                f"User {user_id} has no training history."
            )

        seen_movies = seen_by_user[user_id]

        recommendations = model.recommend(
            user_id=int(user_id),
            seen_movies=seen_movies,
            k=k,
        )

        # Validate the model output
        # check if model recommend same movie
        if len(recommendations) != len(set(recommendations)):
            raise ValueError(
                f"Duplicate recommendations for user {user_id}."
            )

        if set(recommendations) & seen_movies:
            raise ValueError(
                f"Seen movies were recommended to user {user_id}."
            )

        outside_catalog = (
            set(recommendations) - candidate_catalog
        )

        if outside_catalog:
            raise ValueError(
                f"Model recommended movies outside the catalog: "
                f"{sorted(outside_catalog)[:10]}"
            )

        user_hit_rate = hit_rate_k(
            recommendations,
            relevant_movies,
            k,
        )

        user_ndcg = ndcg_k(
            recommendations,
            relevant_movies,
            k,
        )

        evaluation_rows.append({
            "userId": int(user_id),
            "relevant_movies": relevant_movies,
            "recommendations": recommendations,
            f"hit_rate@{k}": user_hit_rate,
            f"ndcg@{k}": user_ndcg,
        })

    per_user_results = pd.DataFrame(evaluation_rows)

    if per_user_results.empty:
        raise ValueError("No users could be evaluated.")

    summary = {
        "model": model_name,
        "split": split_name,
        "k": k,
        "users_available": len(relevant_by_user),
        "users_evaluated": len(per_user_results),
        "cold_start_users_excluded": cold_start_users,
        "hit_rate": float(
            per_user_results[f"hit_rate@{k}"].mean()
        ),
        "ndcg": float(
            per_user_results[f"ndcg@{k}"].mean()
        ),
    }

    return summary, per_user_results