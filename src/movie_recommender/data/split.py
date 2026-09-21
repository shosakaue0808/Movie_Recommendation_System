import pandas as pd

def temporal_positive_split(ratings: pd.DataFrame, 
                            positive_threshold: float = 3.6, 
                            min_pos_rev: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split positive interactions chronologically for each user"""
    # filters ratings by threshold
    positive = ratings[ratings['rating'] >= positive_threshold].copy()

    # get count of positive rating for each user and make new column for count and assign count by map
    positive_counts = positive.groupby('userId').size()
    positive['_user_count'] = positive['userId'].map(positive_counts)
   

    # remove user with less than 5 reviews
    eligible_positive = positive[positive['_user_count']>=min_pos_rev].copy()
    # sort by userId, timestamp, movieId in this priority
    sorted_positive = eligible_positive.sort_values(by=['userId', 'timestamp', 'movieId'])
    # assign index for row in each user
    sorted_positive['_position'] = sorted_positive.groupby('userId').cumcount()
    
    # Split positive reviews in past, current, and future (train, validation, and test)
    train = sorted_positive.loc[sorted_positive['_position'] < sorted_positive['_user_count'] - 2].copy()
    validation = sorted_positive.loc[sorted_positive['_position'] == sorted_positive['_user_count'] - 2].copy()
    test = sorted_positive.loc[sorted_positive['_position'] == sorted_positive['_user_count'] -1].copy()
    return(train, validation, test)

