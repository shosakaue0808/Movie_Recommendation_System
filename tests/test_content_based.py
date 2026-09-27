import pandas as pd
import pytest

from movie_recommender.models.content_based import (
    ContentBasedRecommender,
)


@pytest.fixture
def sample_train() -> pd.DataFrame:
    return pd.DataFrame({
        "userId": [1, 1, 2],
        "movieId": [10, 20, 30],
        "rating": [5.0, 4.0, 5.0],
    })


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_action": [1.0, 1.0, 0.0, 0.8],
            "feature_drama": [0.0, 0.0, 1.0, 0.2],
        },
        index=pd.Index(
            [10, 20, 30, 40],
            name="movieId",
        ),
    )


def test_fit_creates_user_profiles(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = ContentBasedRecommender().fit(
        sample_train,
        sample_features,
    )

    assert set(model.user_profiles.index) == {1, 2}

    assert set(model.user_profiles.columns) == {
        "feature_action",
        "feature_drama",
    }

    assert model.user_profiles.loc[1].tolist() == [
        1.0,
        0.0,
    ]

    assert model.user_profiles.loc[2].tolist() == [
        0.0,
        1.0,
    ]

    assert model.is_fitted
    
def test_recommend_ranks_similar_movie_first(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = ContentBasedRecommender().fit(
        sample_train,
        sample_features,
    )

    recommendations = model.recommend(
        user_id=1,
        seen_movies={10, 20},
        k=2,
    )

    assert recommendations == [40, 30]


def test_recommend_excludes_seen_movies(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = ContentBasedRecommender().fit(
        sample_train,
        sample_features,
    )

    recommendations = model.recommend(
        user_id=1,
        seen_movies={10, 20},
        k=2,
    )

    assert set(recommendations).isdisjoint({10, 20})


def test_recommend_requires_fit() -> None:
    model = ContentBasedRecommender()

    with pytest.raises(RuntimeError):
        model.recommend(
            user_id=1,
            seen_movies=set(),
            k=10,
        )


def test_unknown_user_raises_error(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = ContentBasedRecommender().fit(
        sample_train,
        sample_features,
    )

    with pytest.raises(ValueError):
        model.recommend(
            user_id=999,
            seen_movies=set(),
            k=10,
        )


def test_k_must_be_positive(
    sample_train: pd.DataFrame,
    sample_features: pd.DataFrame,
) -> None:
    model = ContentBasedRecommender().fit(
        sample_train,
        sample_features,
    )

    with pytest.raises(ValueError):
        model.recommend(
            user_id=1,
            seen_movies=set(),
            k=0,
        )