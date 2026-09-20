import pytest

from movie_recommender.evaluation.metrics import (
    hit_rate_k,
    ndcg_k,
)


def test_hit_rate_when_relevant_movie_is_present() -> None:
    assert hit_rate_k(
        recommendations=[10, 20, 30],
        relevant_movies={20},
        k=3,
    ) == 1.0


def test_hit_rate_when_relevant_movie_is_missing() -> None:
    assert hit_rate_k(
        recommendations=[10, 20, 30],
        relevant_movies={40},
        k=3,
    ) == 0.0


def test_hit_rate_respects_k() -> None:
    assert hit_rate_k(
        recommendations=[10, 20, 30],
        relevant_movies={30},
        k=2,
    ) == 0.0


def test_ndcg_is_one_at_first_position() -> None:
    assert ndcg_k(
        recommendations=[20, 10, 30],
        relevant_movies={20},
        k=3,
    ) == pytest.approx(1.0)


def test_ndcg_at_third_position() -> None:
    assert ndcg_k(
        recommendations=[10, 30, 20],
        relevant_movies={20},
        k=3,
    ) == pytest.approx(0.5)


def test_ndcg_is_zero_when_movie_is_missing() -> None:
    assert ndcg_k(
        recommendations=[10, 20, 30],
        relevant_movies={40},
        k=3,
    ) == pytest.approx(0.0)