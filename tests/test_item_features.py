import numpy as np
import pandas as pd
import pytest

from movie_recommender.features.item_features import (
    build_genre_features,
    build_tag_features,
    parse_genres,
)


# ------------------------------------------------------------------
# parse_genres
# ------------------------------------------------------------------


def test_parse_genres_with_multiple_genres() -> None:
    result = parse_genres("Action|Adventure|Sci-Fi")

    assert result == ["Action", "Adventure", "Sci-Fi"]


def test_parse_genres_with_single_genre() -> None:
    result = parse_genres("Drama")

    assert result == ["Drama"]


def test_parse_genres_with_no_genres_listed() -> None:
    result = parse_genres("(no genres listed)")

    assert result == []


# ------------------------------------------------------------------
# build_genre_features
# ------------------------------------------------------------------


@pytest.fixture
def sample_movies() -> pd.DataFrame:
    return pd.DataFrame({
        "movieId": [30, 10, 20, 40],
        "title": [
            "No Genre Movie",
            "Action Comedy Movie",
            "Drama Movie",
            "Excluded Movie",
        ],
        "genres": [
            "(no genres listed)",
            "Action|Comedy",
            "Drama",
            "Horror",
        ],
    })


def test_build_genre_features_creates_multi_hot_vectors(
    sample_movies: pd.DataFrame,
) -> None:
    features = build_genre_features(
        movies=sample_movies,
        catalog={10, 20, 30},
    )

    assert features.loc[10, "genre_Action"] == 1.0
    assert features.loc[10, "genre_Comedy"] == 1.0
    assert features.loc[10, "genre_Drama"] == 0.0

    assert features.loc[20, "genre_Action"] == 0.0
    assert features.loc[20, "genre_Comedy"] == 0.0
    assert features.loc[20, "genre_Drama"] == 1.0


def test_build_genre_features_sorts_movie_ids(
    sample_movies: pd.DataFrame,
) -> None:
    features = build_genre_features(
        movies=sample_movies,
        catalog={10, 20, 30},
    )

    assert list(features.index) == [10, 20, 30]


def test_build_genre_features_only_includes_catalog_movies(
    sample_movies: pd.DataFrame,
) -> None:
    features = build_genre_features(
        movies=sample_movies,
        catalog={10, 30},
    )

    assert set(features.index) == {10, 30}
    assert 20 not in features.index
    assert 40 not in features.index


def test_movie_with_no_genres_has_zero_vector(
    sample_movies: pd.DataFrame,
) -> None:
    features = build_genre_features(
        movies=sample_movies,
        catalog={10, 20, 30},
    )

    assert features.loc[30].sum() == 0.0
    assert "genre_(no genres listed)" not in features.columns


def test_genre_features_are_binary(
    sample_movies: pd.DataFrame,
) -> None:
    features = build_genre_features(
        movies=sample_movies,
        catalog={10, 20, 30},
    )

    unique_values = set(
        features.to_numpy().ravel()
    )

    assert unique_values.issubset({0.0, 1.0})


def test_genre_feature_index_is_named_movie_id(
    sample_movies: pd.DataFrame,
) -> None:
    features = build_genre_features(
        movies=sample_movies,
        catalog={10, 20, 30},
    )

    assert features.index.name == "movieId"


def test_build_genre_features_rejects_missing_columns() -> None:
    movies = pd.DataFrame({
        "movieId": [1, 2],
        "title": ["Movie A", "Movie B"],
    })

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        build_genre_features(
            movies=movies,
            catalog={1, 2},
        )


def test_build_genre_features_rejects_empty_catalog(
    sample_movies: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Catalog cannot be empty",
    ):
        build_genre_features(
            movies=sample_movies,
            catalog=set(),
        )


def test_build_genre_features_rejects_no_matching_movies(
    sample_movies: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="No catalog movie IDs",
    ):
        build_genre_features(
            movies=sample_movies,
            catalog={999},
        )


# ------------------------------------------------------------------
# build_tag_features
# ------------------------------------------------------------------


@pytest.fixture
def sample_tags() -> pd.DataFrame:
    return pd.DataFrame({
        "userId": [1, 2, 3, 4, 5, 6],
        "movieId": [10, 10, 20, 20, 30, 40],
        "tag": [
            " Space ",
            "space adventure",
            "SPACE",
            "romance",
            "dark comedy",
            "horror",
        ],
        "timestamp": [
            100,
            101,
            102,
            103,
            104,
            105,
        ],
    })


def test_build_tag_features_includes_all_catalog_movies(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20, 30, 50},
        min_document_frequency=1,
        max_features=100,
    )

    assert list(features.index) == [10, 20, 30, 50]


def test_build_tag_features_only_includes_catalog_movies(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20},
        min_document_frequency=1,
        max_features=100,
    )

    assert set(features.index) == {10, 20}
    assert 30 not in features.index
    assert 40 not in features.index


def test_tag_cleaning_combines_case_variants(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20, 30},
        min_document_frequency=1,
        max_features=100,
    )

    assert "tag_space" in features.columns
    assert features.loc[10, "tag_space"] > 0.0
    assert features.loc[20, "tag_space"] > 0.0


def test_movie_without_tags_has_zero_tag_vector(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20, 50},
        min_document_frequency=1,
        max_features=100,
    )

    assert features.loc[50].sum() == pytest.approx(0.0)


def test_minimum_document_frequency_removes_rare_terms() -> None:
    tags = pd.DataFrame({
        "movieId": [10, 20, 30],
        "tag": [
            "space action",
            "space comedy",
            "romance",
        ],
    })

    features = build_tag_features(
        tags=tags,
        catalog={10, 20, 30},
        min_document_frequency=2,
        max_features=100,
    )

    assert "tag_space" in features.columns
    assert "tag_action" not in features.columns
    assert "tag_comedy" not in features.columns
    assert "tag_romance" not in features.columns


def test_tag_features_are_nonnegative(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20, 30},
        min_document_frequency=1,
        max_features=100,
    )

    assert np.isfinite(features.to_numpy()).all()
    assert (features.to_numpy() >= 0.0).all()


def test_tag_feature_names_have_prefix(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20, 30},
        min_document_frequency=1,
        max_features=100,
    )

    assert all(
        column.startswith("tag_")
        for column in features.columns
    )


def test_tag_feature_index_is_named_movie_id(
    sample_tags: pd.DataFrame,
) -> None:
    features = build_tag_features(
        tags=sample_tags,
        catalog={10, 20, 30},
        min_document_frequency=1,
        max_features=100,
    )

    assert features.index.name == "movieId"


def test_max_features_limits_vocabulary_size() -> None:
    tags = pd.DataFrame({
        "movieId": [1, 2, 3],
        "tag": [
            "action adventure",
            "comedy romance",
            "crime thriller",
        ],
    })

    features = build_tag_features(
        tags=tags,
        catalog={1, 2, 3},
        min_document_frequency=1,
        max_features=2,
    )

    assert features.shape[1] <= 2


def test_build_tag_features_rejects_missing_columns() -> None:
    tags = pd.DataFrame({
        "movieId": [1, 2],
        "userId": [10, 20],
    })

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        build_tag_features(
            tags=tags,
            catalog={1, 2},
        )


def test_build_tag_features_rejects_empty_catalog(
    sample_tags: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Catalog cannot be empty",
    ):
        build_tag_features(
            tags=sample_tags,
            catalog=set(),
        )


def test_build_tag_features_rejects_no_usable_tags() -> None:
    tags = pd.DataFrame({
        "movieId": [10, 20, 30],
        "tag": [None, "", "   "],
    })

    with pytest.raises(
        ValueError,
        match="No usable tags",
    ):
        build_tag_features(
            tags=tags,
            catalog={10, 20, 30},
            min_document_frequency=1,
        )

