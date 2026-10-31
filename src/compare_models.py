from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error,mean_squared_error

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as suprise_split
from surprise import accuracy

from matrix_factorization_scratch import MatrixFactorization

# locate data
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

ratings = pd.read_csv(DATA_DIR/"ratings.csv")
ratings = ratings[["userId","movieId","rating"]]

train_df, test_df = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

global_mean = train_df["rating"].mean()

print("Training ratings: ",len(train_df))
print("Testing ratings: ",len(test_df))

#  USER BASED COLLABORATIVE FILTERING

user_matrix= train_df.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

similarity_values = cosine_similarity(
    user_matrix.fillna(0)
)

similarity_df = pd.DataFrame(
    similarity_values,
    index=user_matrix.index,
    columns=user_matrix.index
)

def predict_user_based(user_id,movie_id,k=20):
    if user_id not in user_matrix.index:
        return global_mean

    if movie_id not in user_matrix.columns:
        return global_mean

    similarities = similarity_df.loc[user_id].drop(user_id)

    movie_ratings = user_matrix[movie_id].dropna()
    similarities = similarities.loc[
        similarities.index.intersection(movie_ratings.index)
    ]

    similarities = similarities[similarities>0]
    similarities = similarities.nlargest(k)

    if similarities.empty:
        return global_mean

    neighbor_ratings = movie_ratings.loc[similarities.index]

    prediction = np.dot(
        similarities.to_numpy(),
        neighbor_ratings.to_numpy()
    ) / similarities.sum()

    return float(np.clip(prediction,0.5,5.0))

print("\nTraining user-based evaluation...")
user_predictions = np.array([
    predict_user_based(row.userId,row.movieId)
    for row in test_df.itertuples(index=False)
])

# SURPRISE SVD
reader = Reader(rating_scale=(0.5,5.0))
surprise_data = Dataset.load_from_df(
    train_df[["userId","movieId","rating"]],
    reader
)

surprise_trainset = surprise_data.build_full_trainset()

svd_model = SVD(
    n_factors = 50,
    n_epochs = 20,
    random_state = 42
)

print("\nTraining Surprise SVD..")
svd_model.fit(surprise_trainset)

svd_predictions = np.array([
    svd_model.predict(row.userId,row.movieId).est
    for row in test_df.itertuples(index=False)
])

# MATRIX FACTORIZATION
all_user_ids = sorted(ratings["userId"].unique())
all_movie_ids = sorted(ratings["movieId"].unique())

user_to_idx = {
    user_id: idx for idx, user_id in enumerate(all_user_ids)
}
movie_to_idx = {
    movie_id: idx for idx, movie_id in enumerate(all_movie_ids)
}

train_users = train_df["userId"].map(user_to_idx).to_numpy()
train_movies = train_df["movieId"].map(movie_to_idx).to_numpy()
train_ratings = train_df["rating"].to_numpy(dtype=float)

known_train_users = set(train_df["userId"])
known_train_movies = set(train_df["movieId"])

scratch_model = MatrixFactorization(
    n_users=len(all_user_ids),
    n_movies=len(all_movie_ids),
    n_factors=50,
    learning_rate=0.005,
    regularization=0.02,
    epochs=20,
    random_state=42
)

print("\nTaining matrix factorization from scratch...")
scratch_model.fit(
    train_users,
    train_movies,
    train_ratings
)

scratch_predictions =[]

for row in test_df.itertuples(index=False):
    if (row not in known_train_users or row.movieId not in known_train_movies):
        prediction = global_mean

    else:
        prediction = scratch_model.predict(
            user_to_idx[row.userId],
            movie_to_idx[row.movieId],

        )
    scratch_predictions.append(prediction)

scratch_predictions = np.array(scratch_predictions)

# COMPARE ALL THREEE MODELS ON THE SAME TEST SET

actual_ratings = test_df["rating"].to_numpy()

def evaluate_model(name,predictions):
    rmse = np.sqrt(
        mean_squared_error(actual_ratings,predictions)
    )
    mae = mean_absolute_error(actual_ratings,predictions)

    return {
        "Model": name,
        "RMSE": round(rmse,4),
        "MAE": round(mae,4)
    }

results = pd.DataFrame([
    evaluate_model("User-based similarity",user_predictions),
    evaluate_model("SUrprise SVD ",svd_predictions),
    evaluate_model("scratch matrix factorization",scratch_predictions)
])

results = results.sort_values("RMSE").reset_index(drop=True)
print("\n MODEL COMPRISON")
print(results.to_string(index=False))
print("\nLower RMSE and MAE indicate better rating predictions.")