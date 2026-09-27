import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from movie_recommender.models.base_model import BaseRecommender

REQUIRED_TRAIN_COLUMNS = {'userId', 'movieId'}

class ContentBasedRecommender(BaseRecommender):
    """
    Recommend movies similar to a user's liked training movies.
    """

    def __init__(self):
        super().__init__()
        self.item_features = pd.DataFrame() # features matrix (genre, tags, combined)
        self.user_profiles = pd.DataFrame()

    def fit(self, train: pd.DataFrame, item_features: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build user profiles by averaging liked-movie vectors.
        """
        self._validate_training_data(train)
        if item_features is None:
            raise ValueError(
                "item_features is required for "
                "ContentBasedRecommender."
            )

        if item_features.empty:
            raise ValueError(
                "Item features cannot be empty."
            )

        self.item_features = item_features.copy()
        self.item_features.index = (
            self.item_features.index.astype(int)
        )

        if self.item_features.index.has_duplicates:
            raise ValueError(
                "Item feature index contains duplicate movie IDs."
            )

        if self.item_features.isna().any().any():
            raise ValueError(
                "Item features cannot contain missing values."
            )

        self.item_features = self.item_features.astype(
            np.float32
        )

        filtered_train = train.loc[
            train["movieId"].isin(self.item_features.index)
        ].copy()

        if filtered_train.empty:
            raise ValueError(
                "No training movies have corresponding "
                "item features."
            )

        user_profiles: dict[int, np.ndarray] = {}

        for user_id, user_interactions in filtered_train.groupby(
            "userId"
        ):
            liked_movie_ids = (
                user_interactions["movieId"]
                .astype(int)
                .unique()
            )

            liked_movie_features = self.item_features.loc[
                liked_movie_ids
            ]

            user_profiles[int(user_id)] = (
                liked_movie_features
                .mean(axis=0)
                .to_numpy()
            )

        self.user_profiles = pd.DataFrame.from_dict(
            user_profiles,
            orient="index",
            columns=self.item_features.columns,
        )

        self.user_profiles.index.name = "userId"

        self.is_fitted = True
        return self
    

    def recommend(
        self,
        user_id: int,
        seen_movies: set[int],
        k: int,
    ) -> list[int]:
        """Rank unseen movies by cosine similarity."""

        self._validate_recommendation_request(k)

        if user_id not in self.user_profiles.index:
            raise ValueError(
                f"User {user_id} does not have a "
                "training profile."
            )

        candidate_movie_ids = self.item_features.index[
            ~self.item_features.index.isin(seen_movies)
        ]

        if len(candidate_movie_ids) == 0:
            return []

        candidate_features = self.item_features.loc[
            candidate_movie_ids
        ]

        # Double brackets preserve a two-dimensional DataFrame.
        user_profile = self.user_profiles.loc[[user_id]]

        similarity_scores = cosine_similarity(
            user_profile,
            candidate_features,
        )[0]

        ranking = pd.DataFrame({
            "movieId": candidate_movie_ids.astype(int),
            "similarity": similarity_scores,
        })

        ranking = ranking.sort_values(
            by=["similarity", "movieId"],
            ascending=[False, True],
        )

        return (
            ranking["movieId"]
            .head(k)
            .astype(int)
            .tolist()
        )