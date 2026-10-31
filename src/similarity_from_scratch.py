from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

ratings = pd.read_csv(DATA_DIR/"ratings.csv")
movies = pd.read_csv(DATA_DIR/"movies.csv")

user_item_matrix = ratings.pivot(
    index = "userId",
    columns = "movieId",
    values = "rating"
)

print("user-iem matrix shape: ",user_item_matrix.shape)
print("\n First 5 users and 5 movies: ")
print(user_item_matrix.iloc[:5,:5])

total_cells = user_item_matrix.shape[0]*user_item_matrix.shape[1]
missing_cells = user_item_matrix.isna().sum().sum()
sparsity = missing_cells/total_cells * 100

print(f"\n Missing cells: {missing_cells}")
print(f"Matrix sparsity: {sparsity:.2f}%")

example_movie_id = 1

if example_movie_id in user_item_matrix.columns:
    movie_ratings = user_item_matrix[example_movie_id].dropna()
    print(f"\n Ratings for movie ID {example_movie_id}")
    print(movie_ratings.head(10))

def cosine_similarity_users(user_a,user_b,matrix):

    ratings_a = matrix.loc[user_a]
    ratings_b = matrix.loc[user_b]

    common_movies = ratings_a.notna() & ratings_b.notna()

    vector_a = ratings_a[common_movies].to_numpy(dtype=float)
    vector_b = ratings_b[common_movies].to_numpy(dtype=float)

    if len(vector_a)==0:
        return np.nan

    denominator = np.linalg.norm(vector_a)*np.linalg.norm(vector_b)

    if denominator == 0:
        return np.nan

    return np.dot(vector_a,vector_b)/denominator

# testing the function
user_a = 1
user_b =2 

similarity = cosine_similarity_users(
    user_a,
    user_b,
    user_item_matrix
)

print(f"\nSimilarity between {user_a} and user {user_b} : {similarity}")

def recommend_movies_user_based(target_user, matrix, movies, top_n = 10, k=20):
    """
    Recommend movies using the ratings of similar user.
    target_user : ID of the user recieving recommendations
    matrix: user-item rating matrix
    movies: movie detauls dataframe
    top_n : number of recommmendations to return
    k: number of similar users to consider
    """
    if target_user not in matrix.index:
        raise ValueError(f"User {target_user} was not found")

    similarities = {}

    # step1 : calculate similarity with every other user
    for other_user in matrix.index:
        if other_user == target_user:
            continue

        score = cosine_similarity_users(
            target_user,
            other_user,
            matrix
        )
        if np.isfinite(score) and score > 0:
            similarities[other_user] = score

    # step 2: select the k most similar users
    similar_users = sorted(
        similarities.items(),
        key=lambda item : item[1],
        reverse=True
    )[:k]

    if not similar_users:
        return pd.DataFrame(
            columns = ["movieId","title","predicted_ratings"]
        )

    # step3 : consider movies the target user has not rated
    unrated_movies = matrix.loc[target_user]
    unrated_movies = unrated_movies[unrated_movies.isna()].index

    recommendations = []

    # step4: predict a rating using similar user's ratings
    for movie_id in unrated_movies:
        weighted_sum = 0.0
        similarity_sum = 0.0

        for other_users, similarity in similar_users:
            rating = matrix.loc[other_user,movie_id]

            if pd.notna(rating):
                weighted_sum +=similarity*rating
                similarity_sum += similarity

        if similarity_sum > 0:
            predicted_rating = weighted_sum/similarity_sum

            recommendations.append({
                "movieId":movie_id,
                "predicted_rating":predicted_rating
            })

    # step5: sort recommendations and attach movie titles
    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        return pd.DataFrame(
            columns = ["movieId","title","predicted_rating"]
        )
    recommendations_df=recommendations_df.merge(
        movies[["movieId","title"]],
        on="movieId",
        how="left"
    )

    return(
        recommendations_df[
            ["movieId","title","predicted_rating"]
        ]
        .sort_values("predicted_rating",ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
# generate recommendations for user 1
recommendations = recommend_movies_user_based(
    target_user=1,
    matrix=user_item_matrix,
    movies=movies,
    top_n=10,
    k=20
)

print("\n TOP 10 RECOMMENDED MOVIES FOR USER 1 ")
print(recommendations.to_string(index=False))