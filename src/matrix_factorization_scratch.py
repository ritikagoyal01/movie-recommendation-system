from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load and prepare the data
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

ratings = pd.read_csv(DATA_DIR/"ratings.csv")
movies_df = pd.read_csv(DATA_DIR/"movies.csv")

train_df,test_df = train_test_split(
    ratings[["userId","movieId","rating"]],
    test_size = 0.2,
    random_state=42
)

# convert ids into zero-based matrix indices
all_user_ids = sorted(ratings["userId"].unique())
all_movie_ids = sorted(ratings["movieId"].unique())

user_to_idx = {
    user_id: idx for idx, user_id in enumerate(all_user_ids)

}

movie_to_idx = {
    movie_id: idx for idx, movie_id in enumerate(all_movie_ids)
}

# converting training and testing data into integer indices
train_users = train_df["userId"].map(user_to_idx).to_numpy()
train_movies = train_df["movieId"].map(movie_to_idx).to_numpy()
train_ratings = train_df["rating"].to_numpy(dtype=float)

test_users = test_df["userId"].map(user_to_idx).to_numpy()
test_movies = test_df["movieId"].map(movie_to_idx).to_numpy()
test_ratings = test_df["rating"].to_numpy(dtype=float)

# defining the matrix factorization model
class MatrixFactorization:
    def __init__(
            self,
            n_users,
            n_movies,
            n_factors=50,
            learning_rate=0.005,
            regularization=0.02,
            epochs=20,
            random_state=42
    ):
        rng = np.random.default_rng(random_state)

        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs

        self.P = rng.normal(
            0,0.1,(n_users,n_factors)
        )
        self.Q = rng.normal(
            0,0.1,(n_movies,n_factors)
        )

        self.user_bias = np.zeros(n_users)
        self.movie_bias = np.zeros(n_movies)

        self.global_mean = 0.0

    def predict(self, user_idx, movie_idx):
        prediction = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.movie_bias[movie_idx]
            + np.dot(
                self.P[user_idx],
                self.Q[movie_idx]
            )
        )
        return float(np.clip(prediction,0.5,5.0))

    def fit(self, users, movies, ratings):
        self.global_mean = float(np.mean(ratings))

        for epoch in range(self.epochs):
            order = np.random.permutation(len(ratings))

            for j in order:
                u = users[j]
                i = movies[j]
                actual = ratings[j]

                prediction = (
                    self.global_mean
                    + self.user_bias[u]
                    + self.movie_bias[i]
                    + np.dot(self.P[u], self.Q[i])
                )

                error = actual-prediction

                old_user_factors = self.P[u].copy()

                self.user_bias[u] += self.learning_rate* (
                    error
                    - self.regularization * self.user_bias[u]
                )

                self.movie_bias[i] += self.learning_rate * (error-self.regularization * self.movie_bias[i])

                self.P[u] += self.learning_rate * (error * self.Q[i]-self.regularization*self.P[u])
                self.Q[i] += self.learning_rate*(error*old_user_factors-self.regularization*self.Q[i])

                print(f"Completed epoch {epoch+1}/{self.epochs}")

        return self

# train the model

model = MatrixFactorization(
    n_users=len(all_user_ids),
    n_movies=len(all_movie_ids),
    n_factors=50,
    learning_rate=0.005,
    regularization=0.02,
    epochs=20
)

model.fit(train_users,train_movies,train_ratings)

#  evaluate on held-out ratings

test_predictions = np.array([
    model.predict(u,i)
    for u,i in zip(test_users,test_movies)
])

rmse = np.sqrt(mean_squared_error(
    test_ratings,test_predictions
))

mae = mean_absolute_error(
    test_ratings,test_predictions
)

print("\n FROM-SCRATCH MODEL EVALUATION")
print(f"RMSE : {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# recommend movies for a user

def recommend_movies(user_id,top_n=10):
    if user_id not in user_to_idx:
        raise ValueError(f"Unknown user ID: {user_id}")

    user_idx = user_to_idx[user_id]

    rated_movie_ids = set(
        ratings.loc[
            ratings["userId"]==user_id,"movieId"
        ]
    )

    # candidates = movies_df[
    #     ~movies_df["movieId"].isin(rated_movie_ids)
    # ].copy()
    candidates = movies_df[
        (~movies_df["movieId"].isin(rated_movie_ids))
        & (movies_df["movieId"].isin(movie_to_idx))
    ].copy()

    candidates["predicted_rating"] = candidates["movieId"].map(
        lambda movie_id: model.predict(
            user_idx,
            movie_to_idx[movie_id]
        )
    )

    return (
        candidates[
            ["movieId","title","genres","predicted_rating"]
        ]
        .sort_values("predicted_rating",ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

print("\n FROM SCRATCH RECOMMENDATIONS FOR USER 1")
print(recommend_movies(1).to_string(index=False))


            

        
        
        

