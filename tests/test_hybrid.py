import pandas as pd
import pytest

from movie_recommender.models.hybrid import (
    HybridContentRecommender,
)


@pytest.fixture
def sample_train() -> pd.DataFrame:
    return pd.DataFrame({
        "userId": [1, 2, 3, 4],
        "movieId": [1, 3, 3, 2],
    })


@pytest.fixture
def sample_features() -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "feature_action": [1.0, 1.0, 0.0],
            "feature_drama": [0.0, 0.0, 1.0],
        },
        index=[1, 2, 3],
    )

    features.index.name = "movieId"
    return features


def test_invalid_content_weight() -> None:
    with pytest.raises(ValueError):
        HybridContentRecommender(content_weight=1.5)


def test_recommend_excludes_seen_movies(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = HybridContentRecommender(content_weight=0.8)
    model.fit(sample_train, sample_features)

    recommendations = model.recommend(
        user_id=1,
        seen_movies={1},
        k=2,
    )

    assert 1 not in recommendations
    assert len(recommendations) == 2


def test_pure_content_prefers_similar_movie(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = HybridContentRecommender(content_weight=1.0)
    model.fit(sample_train, sample_features)

    recommendations = model.recommend(
        user_id=1,
        seen_movies={1},
        k=1,
    )

    assert recommendations == [2]


def test_pure_popularity_prefers_popular_movie(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = HybridContentRecommender(content_weight=0.0)
    model.fit(sample_train, sample_features)

    recommendations = model.recommend(
        user_id=1,
        seen_movies={1},
        k=1,
    )

    assert recommendations == [3]