import pandas as pd
from movie_recommender.models.base_model import BaseRecommender

class PopularityRecommender(BaseRecommender):
    """
    recommend globally popular movie that a user has not seen. 
    """
    def __init__(self):
        """
        movie_ranking: global popularity ranking of movies
        is_fitted: flag if model is fitted
        """
        super().__init__()
        self.movie_ranking: list[int] =[]

    def fit(self, train: pd.DataFrame, item_features: pd.DataFrame | None = None)->"PopularityRecommender":
        """
        create global popularity ranking and assign list of movieIds to movie_ranking.
        popularity score is based on global count of being seen
        """
        self._validate_training_data(train)
        # make popularity series with movieId as tie breaker
        popularity_score = train.groupby('movieId')['userId'].nunique().sort_index()
        # sort series by popularity but keep movieId order when it's tie
        self.movie_ranking = popularity_score.sort_values(ascending=False, kind='stable').index.astype(int).tolist()
        self.is_fitted = True
        return self

    def recommend(self, user_id: int, seen_movies: set[int], k: int)->list[int]:
        """
        for each user, recommend unseen global top-k movies
        """
        self._validate_recommendation_request(k)

        personal_recommend = []
        for movieId in self.movie_ranking:
            if not (movieId in seen_movies):
                personal_recommend.append(movieId)
            if len(personal_recommend) == k:
                return personal_recommend
        return personal_recommend


