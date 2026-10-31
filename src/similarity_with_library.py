from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# loacte the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

ratings = pd.read_csv(DATA_DIR/"ratings.csv")
movies = pd.read_csv(DATA_DIR/"movies.csv")

# build the user-item matrix
user_item_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

# for the baseline, represent the missing ratings as zero
filled_matrix = user_item_matrix.fillna(0)

# claculate the cosine similarity between all users
similarity_matrix = cosine_similarity(filled_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=filled_matrix.index,
    columns=filled_matrix.index
)
print("Similarity matrix shape: ",similarity_df.shape)

# show users most similar to user 1
target_user = 1

similar_users = (
    similarity_df.loc[target_user]
    .drop(target_user)
    .sort_values(ascending=False)
    .head(5)
)

print("\n TOP 5 SIMILAR USERS TO USER 1")
print(similar_users)

# recommend movies using similar users
def recommend_movies(target_user, top_n=10, k=20):
    if target_user not in user_item_matrix.index:
        raise ValueError(f"User {target_user} was not found")

    # select the k most similar users
    neighbors = (
        similarity_df.loc[target_user]
        .drop(target_user)
        .sort_values(ascending=False)
        .head(k)
    )

    # only consider movies the target user has not rated
    target_ratings = user_item_matrix.loc[target_user]
    unrated_movies = target_ratings[target_ratings.isna()].index

    # weighted average of neighbors ratings
    neighbor_ratings = user_item_matrix.loc[neighbors.index]

    weighted_ratings = neighbor_ratings.mul(neighbors,axis=0)

    numerator = weighted_ratings.sum(axis=0)
    denominator = neighbor_ratings.notna().mul(neighbors,axis=0).sum(axis=0)

    predictions = numerator.div(denominator.replace(0,np.nan))
    predictions = predictions.loc[unrated_movies].dropna()

    # select the highest predicted ratings
    result = predictions.nlargest(top_n).rename("predicted_rating")
    result = result.reset_index()
    result.columns = ["movieId","predicted_rating"]

    return result.merge(
        movies[["movieId","title"]],
        on="movieId",
        how="left"
    )[["movieId","title","predicted_rating"]]

print("\n LIBRARY-BASED RECOMMENDATIONS FOR USER 1")
print(recommend_movies(target_user=1).to_string(index=False))