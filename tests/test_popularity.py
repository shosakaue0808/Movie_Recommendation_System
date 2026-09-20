import pandas as pd
import pytest

from movie_recommender.models.popularity import (
    PopularityRecommender,
)


def make_training_data() -> pd.DataFrame:
    return pd.DataFrame({
        "userId": [1, 2, 3, 1, 2, 1],
        "movieId": [10, 10, 10, 20, 20, 30],
        "rating": [5.0, 4.0, 4.5, 4.0, 5.0, 4.0],
    })


def test_fit_creates_correct_popularity_ranking() -> None:
    train = make_training_data()

    model = PopularityRecommender().fit(train)

    assert model.movie_ranking == [10, 20, 30]


def test_recommend_removes_seen_movies() -> None:
    train = make_training_data()
    model = PopularityRecommender().fit(train)

    recommendations = model.recommend(
        seen_movies={10},
        k=2,
    )

    assert recommendations == [20, 30]


def test_recommend_requires_fit() -> None:
    model = PopularityRecommender()

    with pytest.raises(RuntimeError):
        model.recommend(seen_movies=set(), k=2)


def test_k_must_be_positive() -> None:
    train = make_training_data()
    model = PopularityRecommender().fit(train)

    with pytest.raises(ValueError):
        model.recommend(seen_movies=set(), k=0)