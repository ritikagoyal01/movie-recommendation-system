from pathlib import Path
import numpy as np
import pandas as pd

# locate the datset 
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

# load data
ratings = pd.read_csv(DATA_DIR/"ratings.csv")
movies = pd.read_csv(DATA_DIR/"movies.csv")

# rows = movies,columns = users
movie_user_matrix = ratings.pivot(
    index="movieId",
    columns="userId",
    values="rating"
)

print("Movie-user matrix shape: ",movie_user_matrix.shape)

def cosine_similarity_items(movie_a,movie_b,matrix):
    """Calculate cosine similarity using users who rated both movies."""

    ratings_a = matrix.loc[movie_a]
    ratings_b = matrix.loc[movie_b]

    # select users who rated both movies
    common_users = ratings_a.notna() & ratings_b.notna()

    vector_a = ratings_a[common_users].to_numpy(dtype=float)
    vector_b = ratings_b[common_users].to_numpy(dtype=float)

    if len(vector_a) == 0:
        return np.nan

    denominator = np.linalg.norm(vector_a)*np.linalg.norm(vector_b)

    if denominator==0:
        return np.nan

    return np.dot(vector_a,vector_b)/denominator

def recommend_similar_movies(movie_id, matrix, movies, top_n=10):
    """find movies with similar rating patterns to a given movie. """

    if movie_id not in matrix.index:
        raise ValueError(f"movie ID {movie_id} was not found")

    similarities = []

    for other_movie in matrix.index:
        if other_movie == movie_id:
            continue

        score = cosine_similarity_items(
            movie_id,
            other_movie,
            matrix
        )

        if np.isfinite(score) and score>0:
            similarities.append((other_movie, score))

    # sort from most similar to least similar
    similarities.sort(key=lambda item: item[1],reverse=True)
    similarities=similarities[:top_n]

    result = pd.DataFrame(
        similarities,
        columns=["movieId","similarity"]
    )

    return result.merge(
        movies[["movieId","title"]],
        on="movieId",
        how="left"
    )[["movieId","title","similarity"]]

def recommend_for_user(user_id, ratings, matrix, movies, top_n=10):
    """Recommend unrated movies based on movies a user has rated """

    user_ratings = ratings[ratings["userId"] == user_id]

    if user_ratings.empty:
        raise ValueError(f"No ratings found for user {user_id}")

    # movies the user has already rated
    rated_movie_ids = set(user_ratings["movieId"])

    # candidate movies are those not rated by the user
    candidates = [
        movie_id for movie_id in matrix.index
        if movie_id not in rated_movie_ids
    ]
    predictions = []

    for candidate in candidates:
        weighted_sum = 0.0
        similarity_sum = 0.0

        for _,row in user_ratings.iterrows():
            known_movie = row["movieId"]
            user_rating = row["rating"]

            if known_movie not in matrix.index:
                continue

            similairty = cosine_similarity_items(
                candidate,
                known_movie,
                matrix
            )

            if np.isfinite(similairty) and similairty>0:
                weighted_sum += similairty*user_rating
                similarity_sum += similairty

            if similarity_sum>0:
                predictions.append({
                    "movieId":candidate,
                    "predicted_rating":weighted_sum/similarity_sum
                })

        result = pd.DataFrame(predictions)

        if result.empty:
            return pd.DataFrame(
                columns=["movieId","title","predicted_rating"]
            )

        result = result.merge(
            movies[["movieId","title"]],
            on="movieId",
            how="left"
        )

        return(
            result[["movieId","title","predicted_rating"]]
            .sort_values("predicted_rating",ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

# example 1: find movies similar to toy story(movieId 1)
print("\n MOVIES SIMILAR TO TOY STORY")
print(
    recommend_similar_movies(
        movie_id=1,
        matrix=movie_user_matrix,
        movies=movies,
        top_n=10
    ).to_string(index=False)
)

# exapmle2 recommend movies for user
print("\n ITEM BASED RECOMMENDATIONS FOR USER 1")
print(
    recommend_for_user(
        user_id=1,
        ratings=ratings,
        matrix=movie_user_matrix,
        movies=movies,
        top_n=10
    ).to_string(index=False)
)