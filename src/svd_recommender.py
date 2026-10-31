from pathlib import Path
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# loacte the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

ratings = pd.read_csv(DATA_DIR/"ratings.csv")
movies = pd.read_csv(DATA_DIR/"movies.csv")

# surprise expects user id, item id, and rating columns
ratings_for_model= ratings[
    ["userId","movieId","rating"]
].copy()

# 2. tell surprise the valid rating range
reader =Reader(rating_scale=(0.5,5.0))

data = Dataset.load_from_df(
    ratings_for_model,
    reader
)

# 3. split ratings into training and testing sets
trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# 4. create and train the svd model
model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)
model.fit(trainset)
print("SVD model trained successfully ")

# 5. evaluate predictions on unseen test ratings
predictions = model.test(testset)

print("\n MODEL EVALUATION")
accuracy.rmse(predictions,verbose=True)
accuracy.mae(predictions,verbose=True)

# 6.recommend movies for a selected user
def recommend_movies(user_id, top_n=10):
    if user_id not in set(ratings["userId"]):
        raise ValueError(f"User {user_id} was not found")

    rated_movie_ids = set(
        ratings.loc[
            ratings["userId"] == user_id,
            "movieId"
        ]
    )

    # candidte movies that exist in the datset
    candidates = movies[
        ~movies["movieId"].isin(rated_movie_ids)
    ].copy()

    # predict ratings for the candidate movies
    candidates["predicted_rating"] = candidates["movieId"].apply(
        lambda movie_id: model.predict(
            uid = user_id,
            iid = movie_id
        ).est
    )

    return (
        candidates[
            ["movieId","title","predicted_rating"]
        ]
        .sort_values("predicted_rating",ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

# 7. generate recommendations for user 1
print("\n TOP 10 SVD RECOMMENDATIONS FOR USER 1")

recommendations = recommend_movies(
    user_id=1,
    top_n=10
)

print(recommendations.to_string(index=False))