import pandas as pd

class PopularityRecommender:
    """
    recommend globally popular movie that a user has not seen. 
    """
    def __init__(self):
        """
        movie_ranking: global popularity ranking of movies
        is_fitted: flag if model is fitted
        """
        self.movie_ranking: list[int] =[]
        self.is_fitted = False

    def fit(self, train: pd.DataFrame)->"PopularityRecommender":
        """
        create global popularity ranking and assign list of movieIds to movie_ranking.
        popularity score is based on global count of being seen
        """
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
        if not self.is_fitted:
            raise RuntimeError('Call fit() before generating recommendations')

        if k <= 0:
            raise ValueError("rank needs to be positive")

        personal_recommend = []
        for movieId in self.movie_ranking:
            if not (movieId in seen_movies):
                personal_recommend.append(movieId)
            if len(personal_recommend) == k:
                return personal_recommend
        return personal_recommend


