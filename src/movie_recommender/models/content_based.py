import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_TRAIN_COLUMNS = {'userId', 'movieId'}

class ContentBasedRecommender:
    """
    Recommend movies similar to a user's liked training movies.
    """

    def __init__(self):
        self.item_features: pd.DataFrame # features matrix (genre, tags, combined)
        self.user_profiles: dict[int, np.ndarray] = {}
        self.is_fitted = False

    def fit(self, train: pd.DataFrame, item_features: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build user profiles by averaging liked-movie vectors.
        """
        missing_columns = REQUIRED_TRAIN_COLUMNS - set(train.columns)

        if missing_columns:
            raise ValueError(
                f"Training data is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if item_features.empty:
            raise ValueError("Item features cannot be empty.")

        if not item_features.index.is_unique:
            raise ValueError(
                "Item-feature index must contain unique movie IDs."
            )

        if item_features.isna().any().any():
            raise ValueError(
                "Item features cannot contain missing values."
            )

        training_movie_ids = set(
            train["movieId"].astype(int)
        )

        feature_movie_ids = set(
            item_features.index.astype(int)
        )

        missing_movies = (
            training_movie_ids - feature_movie_ids
        )

        if missing_movies:
            raise ValueError(
                "Some training movies are missing item features: "
                f"{sorted(missing_movies)[:10]}"
            )

        self.item_features = (
            item_features
            .copy()
            .sort_index()
            .astype(np.float32)
        )

        self.user_profiles = {}

        for user_id, user_interactions in train.groupby("userId"):
            liked_movie_ids = (
                user_interactions["movieId"]
                .astype(int)
                .unique()
            )

            liked_movie_features = self.item_features.loc[
                liked_movie_ids
            ]

            user_profile = (
                liked_movie_features
                .mean(axis=0)
                .to_numpy(dtype=np.float32)
            )

            self.user_profiles[int(user_id)] = user_profile

        self.is_fitted = True

        return self
    
    def recommend(
        self,
        user_id: int,
        seen_movies: set[int],
        k: int,
        ) -> list[int]:
        """Return top-K unseen movies ranked by cosine similarity."""

        if not self.is_fitted or self.item_features is None:
            raise RuntimeError(
                "Call fit() before generating recommendations."
            )

        if k <= 0:
            raise ValueError("k must be positive.")

        if user_id not in self.user_profiles:
            raise ValueError(
                f"No profile is available for user {user_id}."
            )

        candidate_movie_ids = [
            int(movie_id)
            for movie_id in self.item_features.index
            if movie_id not in seen_movies
        ]

        if not candidate_movie_ids:
            return []

        user_profile = self.user_profiles[user_id]

        if np.linalg.norm(user_profile) == 0:
            raise ValueError(
                f"User {user_id} has a zero feature profile."
            )

        candidate_features = self.item_features.loc[
            candidate_movie_ids
        ]

        similarity_scores = cosine_similarity(
            user_profile.reshape(1, -1),
            candidate_features,
        ).ravel()

        ranking = pd.DataFrame({
            "movieId": candidate_movie_ids,
            "score": similarity_scores,
        })

        ranking = ranking.sort_values(
            ["score", "movieId"],
            ascending=[False, True],
        )

        return (
            ranking["movieId"]
            .head(k)
            .astype(int)
            .tolist()
        )