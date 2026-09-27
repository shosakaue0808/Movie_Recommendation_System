

import pandas as pd
from abc import ABC, abstractmethod



class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.

    Every recommender must:
    1. Learn from positive training interactions using fit().
    2. Recommend unseen movies using recommend().
    """

    REQUIRED_TRAIN_COLUMNS = {"userId", "movieId"}

    def __init__(self) -> None:
        """Initialize state shared by every recommender."""
        self.is_fitted = False

    def _validate_training_data(
        self,
        train: pd.DataFrame,
    ) -> None:
        """
        validate given training data matches fit function
        """
        missing_columns = (
            self.REQUIRED_TRAIN_COLUMNS
            - set(train.columns)
        )

        if missing_columns:
            raise ValueError(
                "Training data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if train.empty:
            raise ValueError(
                "Training data cannot be empty."
            )

    def _validate_recommendation_request(
        self,
        k: int,
    ) -> None:
        """
        validate if recommend function input are valid
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Call fit() before generating recommendations."
            )

        if k <= 0:
            raise ValueError(
                "k must be positive."
            )
    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        item_features: pd.DataFrame | None = None,
    ) -> "BaseRecommender":
        """
        Fit the recommendation model.

        Parameters
        ----------
        train:
            Positive training interactions containing userId and movieId.

        item_features:
            Optional movie-feature matrix indexed by movieId.
            Content-based models require it, while models such as
            popularity do not.

        Returns
        -------
        Self
            The fitted model.
        """
        raise NotImplementedError

    def recommend(
        self,
        user_id: int,
        seen_movies: set[int],
        k: int,
    ) -> list[int]:
        """
        Return the top-k unseen movie IDs for one user.

        Parameters
        ----------
        user_id:
            User for whom recommendations are generated.

        seen_movies:
            Movie IDs that must be excluded.

        k:
            Maximum number of recommendations.

        Returns
        -------
        list[int]
            Ranked unseen movie IDs.
        """
        raise NotImplementedError