import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


REQUIRED_MOVIE_COLUMNS = {"movieId", "genres"}
REQUIRED_TAG_COLUMNS = {"movieId", "tag"}

def parse_genres(genre_string: str) -> list[str]:
    """
    convert genre of the movie separated by | into list
    """
    # remove no genres listed
    if genre_string == "(no genres listed)":
        return []
    return genre_string.split('|')


def build_genre_features(
        movies: pd.DataFrame,
        catalog: set[int]
        )->pd.DataFrame:
    """
    create multi-hot genre features for movies in catalog
    """
    missing_columns = REQUIRED_MOVIE_COLUMNS - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Movies data is missing columns: {sorted(missing_columns)}"
        )
    
    if not catalog:
        raise ValueError("Catalog cannot be empty.")
    catalog_movies = movies.loc[movies['movieId'].isin(catalog)].copy()

    if catalog_movies.empty:
        raise ValueError(
            "No catalog movie IDs were found in the movies dataframe."
        )
    
    genre_lists = catalog_movies['genres'].apply(parse_genres)

    # make multihot encoder
    encoder = MultiLabelBinarizer()
    genre_matrix = encoder.fit_transform(genre_lists)

    feature_names = [
        f"genre_{genre}"
        for genre in encoder.classes_
    ]

    genre_features = pd.DataFrame(
        genre_matrix,
        index=catalog_movies["movieId"].astype(int),
        columns=feature_names,
        dtype=float,
    )

    genre_features.index.name = "movieId"

    return genre_features

def build_tag_features(
    tags: pd.DataFrame,
    catalog: set[int],
    min_document_frequency: int = 2,
    max_features: int = 1000,
    ) -> pd.DataFrame:
    """
    Create TF-IDF tag features for movies in the catalog to give weights based on helpfulness of tags. 
    Ex) Funny, Cyberpunk. Funny is repeated word, but cyberpunk is more distinguishable word
    TF-IDF is numerical statistic that reflects the significance of a word within a document relative to a collection of documents. 
    TF (term frequency) measures the frequency of a term within a doc  
    TF(t, d)= num(t) in d/num of total terms in d
    IDF (inverse document frequency) measure teh rarity of a term within collection of docs. 
    IDF(t, D) = log(total num of docs in D / number of docs contains t) gives small weights to common words (helpful)
    TF-IDF(t, d, D) = TF(t, d) * IDF(t, D) importance = frequency in one doc d * rarelity in D
    """

    missing_columns = REQUIRED_TAG_COLUMNS - set(tags.columns)

    if missing_columns:
        raise ValueError(
            f"Tags data is missing columns: {sorted(missing_columns)}"
        )

    if not catalog:
        raise ValueError("Catalog cannot be empty.")

    # take row of movie that is in catalog, and keep movieId and tag columns
    catalog_tags = (
        tags.loc[
            tags["movieId"].isin(catalog),
            ["movieId", "tag"],
        ]
        .dropna(subset=["tag"])
        .copy()
    )

    # clean and standardize tags
    catalog_tags["tag"] = (
        catalog_tags["tag"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    catalog_tags = catalog_tags.loc[
        catalog_tags["tag"] != ""
    ]

    if catalog_tags.empty:
        raise ValueError(
            "No usable tags were found for catalog movies."
        )

    # Combine every tag assigned to the same movie into one document
    tag_documents = (
        catalog_tags.groupby("movieId")["tag"]
        .apply(" ".join)
    )

    # Include catalog movies that have no tags
    movie_ids = sorted(catalog)

    # fill empty string so that they get zero tag vector
    tag_documents = tag_documents.reindex(
        movie_ids,
        fill_value="",
    )

    vectorizer = TfidfVectorizer(
        min_df=min_document_frequency,
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        dtype=np.float32,
    )

    tag_matrix = vectorizer.fit_transform(tag_documents)

    feature_names = [
        f"tag_{term}"
        for term in vectorizer.get_feature_names_out()
    ]

    tag_features = pd.DataFrame(
        tag_matrix.toarray(),
        index=movie_ids,
        columns=feature_names,
        dtype=np.float32,
    )

    tag_features.index.name = "movieId"

    return tag_features