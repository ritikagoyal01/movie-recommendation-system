from pathlib import Path
import pandas as pd

# locate the dataset
PROJECT_DIR =  Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

# load the csv files
movies = pd.read_csv(DATA_DIR/"movies.csv")
ratings = pd.read_csv(DATA_DIR/"ratings.csv")

# display basic infromation

print("\n DATASET SHAPE ")
print("Movies: ",movies.shape)
print("Ratings: ",ratings.shape)

print("\n FIRST FIVE MOVIES")
print(movies.head())


print("\n FIRST FIVE RATING ")
print(ratings.head())


print("UNIQUE COUNTS")
print(f"\nUnique Users< {ratings['userId'].nunique()}")
print(f"Unique rated movies: {ratings['movieId'].nunique()}")

print("RATING SATISTICS")
print(ratings['rating'].describe())

# most rated movies
movie_rating_counts = (
    ratings.groupby("movieId")
    .size()
    .reset_index(name="rating_count")
    .merge(movies[["movieId","title"]],on="movieId")
    .sort_values("rating_count",ascending=False)
)

print("\n TAP 10 MOST RATED MOVIES")
print(movie_rating_counts[["title","rating_count"]].head(10).to_string(index=False))

# average ratings by movie
movie_average_ratings = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean","count"])
    .reset_index()
    .merge(movies[["movieId","title"]],on="movieId")
    .rename(columns={"mean": "average_rating","count": "rating_count"})
)

# to avoid ranking movies with very few ratings
popular_movies = movie_average_ratings[
    movie_average_ratings["rating_count"]>=50
].sort_values("average_rating",ascending=False)

print(
    popular_movies[["title","average_rating","rating_count"]]
    .head(10)
    .to_string(index=False)
)

print("\n MISSING VALUES")
print("Movies:\n",movies.isnull().sum())
print("Ratings:\n",ratings.isnull().sum())