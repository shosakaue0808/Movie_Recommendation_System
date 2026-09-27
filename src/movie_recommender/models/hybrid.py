import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from movie_recommender.models.base_model import BaseRecommender

REQUIRED_TRAIN_COLUMNS = {"userId", "movieId"}


class HybridContentRecommender(BaseRecommender):
    """
    Recommend unseen movies using a weighted combination of:

    1. Cosine similarity between a user profile and movie features.
    2. Normalized global movie popularity.
    """

    def __init__(self, content_weight: float = 0.8) -> None:
        super().__init__()
        if not 0 <= content_weight <= 1:
            raise ValueError(
                "content_weight must be between 0 and 1."
            )

        self.content_weight = content_weight
        self.item_features = pd.DataFrame()
        self.user_profiles = pd.DataFrame()
        self.popularity_scores = pd.Series(dtype=float)

    def fit(
        self,
        train: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> "HybridContentRecommender":
        """
        Build user content profiles and normalized popularity scores.

        Parameters
        ----------
        train:
            Positive training interactions containing userId and movieId.

        item_features:
            DataFrame indexed by movieId, with feature columns such as
            genres and TF-IDF tag features.
        """

        self._validate_training_data(train)


      
        if item_features.index.has_duplicates:
            raise ValueError(
                "Item feature index contains duplicate movie IDs."
            )

        if item_features.isna().any().any():
            raise ValueError(
                "Item features cannot contain missing values."
            )

        self.item_features = item_features.copy()
        self.item_features.index = (
            self.item_features.index.astype(int)
        )

        # Keep only interactions for movies with available features.
        filtered_train = train.loc[
            train["movieId"].isin(self.item_features.index)
        ].copy()

        if filtered_train.empty:
            raise ValueError(
                "No training movies have corresponding item features."
            )

        # Build one average content profile for each user.
        user_profiles: dict[int, np.ndarray] = {}

        for user_id, user_interactions in filtered_train.groupby(
            "userId"
        ):
            liked_movie_ids = (
                user_interactions["movieId"]
                .astype(int)
                .unique()
            )

            liked_movie_features = self.item_features.loc[liked_movie_ids]

            # take average over the rows
            user_profiles[int(user_id)] = (
                liked_movie_features.mean(axis=0).to_numpy()
            )

        self.user_profiles = pd.DataFrame.from_dict(
            user_profiles,
            orient="index",
            columns=self.item_features.columns,
        )

        self.user_profiles.index.name = "userId"

        # Count the number of distinct users who liked each movie.
        popularity = (
            filtered_train.groupby("movieId")["userId"]
            .nunique()
            .reindex(self.item_features.index, fill_value=0)
            .astype(float)
        )

        # Reduce the influence of extremely popular movies.
        log_popularity = np.log1p(popularity)

        # Scale popularity to the interval [0, 1]. normalization because genre vectors are also normalized 
        self.popularity_scores = (log_popularity / log_popularity.max())

        self.popularity_scores.index = (
            self.popularity_scores.index.astype(int)
        )
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        seen_movies: set[int],
        k: int,
    ) -> list[int]:
        """
        Recommend the top-k unseen movies for one user.
        """
        self._validate_recommendation_request(k)
        if user_id not in self.user_profiles.index:
            raise ValueError(
                f"User {user_id} does not have a training profile."
            )

        # store unseen movies by user
        candidate_movie_ids = self.item_features.index[
            ~self.item_features.index.isin(seen_movies)
        ]

        if len(candidate_movie_ids) == 0:
            return []

        candidate_features = self.item_features.loc[
            candidate_movie_ids
        ]

        # by double bracket, return 1* num of features dataframe not series
        user_profile = self.user_profiles.loc[[user_id]]

        # get matrix of 1*num of candidate movies and turn into [num of candidates] array
        similarity_scores = cosine_similarity(
            user_profile,
            candidate_features,
        )[0]

        candidate_popularity = (
            self.popularity_scores
            .reindex(candidate_movie_ids, fill_value=0.0)
            .to_numpy()
        )

        hybrid_scores = (
            self.content_weight * similarity_scores
            + (1 - self.content_weight) * candidate_popularity
        )

        ranking = pd.DataFrame({
            "movieId": candidate_movie_ids.astype(int),
            "similarity": similarity_scores,
            "popularity": candidate_popularity,
            "hybrid_score": hybrid_scores,
        })

        ranking = ranking.sort_values(
            by=["hybrid_score", "movieId"],
            ascending=[False, True],
        )

        return (
            ranking["movieId"]
            .head(k)
            .astype(int)
            .tolist()
        )