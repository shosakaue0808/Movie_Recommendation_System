import pandas as pd
import pytest

from movie_recommender.features.item_features import (
    build_genre_features,
    parse_genres,
)


def test_parse_multiple_genres() -> None:
    result = parse_genres("Action|Adventure|Sci-Fi")

    assert result == ["Action", "Adventure", "Sci-Fi"]


def test_parse_no_genres_listed() -> None:
    result = parse_genres("(no genres listed)")

    assert result == []


def test_build_genre_features() -> None:
    movies = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": [
            "Action|Comedy",
            "Drama",
            "(no genres listed)",
        ],
    })

    features = build_genre_features(
        movies=movies,
        catalog={1, 2, 3},
    )

    assert list(features.index) == [1, 2, 3]

    assert features.loc[1, "genre_Action"] == 1.0
    assert features.loc[1, "genre_Comedy"] == 1.0
    assert features.loc[1, "genre_Drama"] == 0.0

    assert features.loc[2, "genre_Action"] == 0.0
    assert features.loc[2, "genre_Drama"] == 1.0

    assert features.loc[3].sum() == 0.0


def test_build_features_only_for_catalog_movies() -> None:
    movies = pd.DataFrame({
        "movieId": [1, 2, 3],
        "genres": ["Action", "Comedy", "Drama"],
    })

    features = build_genre_features(
        movies=movies,
        catalog={1, 3},
    )

    assert list(features.index) == [1, 3]


def test_empty_catalog_raises_error() -> None:
    movies = pd.DataFrame({
        "movieId": [1],
        "genres": ["Action"],
    })

    with pytest.raises(ValueError):
        build_genre_features(movies, catalog=set())

from movie_recommender.features.item_features import (
    build_genre_features,
    build_tag_features,
    parse_genres,
)


def test_build_tag_features() -> None:
    tags = pd.DataFrame({
        "userId": [1, 2, 3, 4],
        "movieId": [10, 10, 20, 30],
        "tag": [
            "space adventure",
            "space",
            "romance",
            "dark comedy",
        ],
    })

    features = build_tag_features(
        tags=tags,
        catalog={10, 20, 30},
        min_document_frequency=1,
        max_features=100,
    )

    assert list(features.index) == [10, 20, 30]

    assert "tag_space" in features.columns

    assert features.loc[10, "tag_space"] > 0
    assert features.loc[20, "tag_space"] == 0
    assert features.loc[30, "tag_space"] == 0


def test_tag_features_only_include_catalog_movies() -> None:
    tags = pd.DataFrame({
        "movieId": [10, 20, 30],
        "tag": ["action", "comedy", "drama"],
    })

    features = build_tag_features(
        tags=tags,
        catalog={10, 30},
        min_document_frequency=1,
    )

    assert list(features.index) == [10, 30]


def test_movie_without_tags_has_zero_vector() -> None:
    tags = pd.DataFrame({
        "movieId": [10],
        "tag": ["action"],
    })

    features = build_tag_features(
        tags=tags,
        catalog={10, 20},
        min_document_frequency=1,
    )

    assert features.loc[20].sum() == 0.0