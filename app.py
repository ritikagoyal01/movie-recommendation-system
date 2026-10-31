from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from surprise import Dataset, Reader, SVD

# PAGE CONFIGURATION
st.set_page_config(
    page_title="MovieMind | movie Recommendations",
    layout = "wide"
)



PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR/"data"/"ml-latest-small"

# LOAD THE DATA
@st.cache_data
def load_data():
    movies = pd.read_csv(DATA_DIR/"movies.csv")
    ratings=pd.read_csv(DATA_DIR/"ratings.csv")

    return movies,ratings

# TRAIN THE MODEL ONCE AND THEN CACHE IT
@st.cache_resource
def train_model():
    _,ratings=load_data()

    reader = Reader(rating_scale=(0.5,5.0))

    data = Dataset.load_from_df(
        ratings[["userId","movieId","rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    model = SVD(
        n_factors = 50,
        n_epochs = 20,
        random_state=42
    )

    model.fit(trainset)
    return model

# RECOMMENDATIONS
def get_recommendations(user_id,model,movies,ratings,top_n):
    rated_movie_ids = set(
        ratings.loc[
            ratings["userId"]==user_id,
            "movieId"
        ]
    )
    rated_by_someone = set(ratings["movieId"])

    candidates = movies[
        movies["movieId"].isin(rated_by_someone)
        & ~movies["movieId"].isin(rated_movie_ids)
    ].copy()

    candidates["predicted_rating"] = candidates["movieId"].apply(
        lambda movie_id:model.predict(
            uid = user_id,
            iid = movie_id
        ).est
    )

    ratings_counts = ratings.groupby("movieId").size()

    candidates["rating_count"] = (
        candidates["movieId"]
        .map(ratings_counts)
        .fillna(0)
        .astype(int)
    )

    return (
        candidates.sort_values(
            "predicted_rating",
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

# APP INTERFACE
st.title("MovieMind")
st.subheader("Your personalized movie recommendation system")

st.write(
    "Discover movies you might enjoy using collaborative filtering"
    "and matrix factorization"
)

st.divider()

try:
    movies,ratings = load_data()
except FileNotFoundError:
    st.error(
        "Dataset files were not found . Please check that"
        "data/ml-latest-small contains movies.csv and ratings.csv."
    )
    st.stop()

# DATASET OVERVIEW
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Movies",f"{len(movies):,}")
with col2:
    st.metric("Users",f"{ratings['userId'].nunique():,}")
with col3:
    st.metric("Ratings",f"{len(ratings):,}")

st.divider()

st.header("Find your next favourite movie")

user_ids = sorted(ratings["userId"].unique())

selected_user = st.selectbox(
    "Select a user profile",
    options = user_ids,
    index = 0,
    format_func = lambda user_id:f"User {user_id}"
)

top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=5
)

if st.button("Get Recommendations",type="primary"):
    with st.spinner("Training or loading the recommendation model...."):
        model = train_model()

    with st.spinner("Finding movies for your profile..."):
        recommendations = get_recommendations(
            selected_user,
            model,
            movies,
            ratings,
            top_n
        )

    if recommendations.empty:
        st.warning("No recommendations are available for this user.")
    else:
        st.success(f"Recommendations for User {selected_user}")

        display_recommendations = recommendations.copy()

        display_recommendations["predicted_rating"]=(
            display_recommendations["predicted_rating"].round(2)
        )
        display_recommendations = display_recommendations.rename(
            columns={
                "movieId":"Movie ID",
                "title":"Movie",
                "genres":"Genres",
                "predicted_rating":"Predicted Rating",
                "rating_count":"Number of Ratings"
            }
        )

        st.dataframe(
            display_recommendations[
                [
                    "Movie",
                    "Genres",
                    "Predicted Rating",
                    "Number of Ratings"
                ]
            ],
            use_container_width = True,
            hide_index=True
        )

        st.caption(
            "Predicted ratings are model estimates, not actual user ratings."
        )
st.divider()

# POPULAR MOVIES SECTION
st.header("Popular movies")

popular_movies = (
    ratings.groupby("movieId")
    .agg(
        average_rating=("rating","mean"),
        rating_count=("rating","count")
    )
    .reset_index()
    .merge(movies,on="movieId",how="left")
)

popular_movies=popular_movies[
    popular_movies["rating_count"]>=50
]

popular_movies=popular_movies.sort_values(
    ["average_rating","rating_count"],
    ascending=[False,False]
).head(10)

popular_movies["average_rating"]=(
    popular_movies["average_rating"].round(2)
)

st.dataframe(
    popular_movies[
        ["title","genres","average_rating","rating_count"]
    ].rename(
        columns={
            "title":"Movie",
            "genres":"Genres",
            "average_rating":"Average Rating",
            "rating_count":"Number of Ratings"
        }
    ),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Popular movies are ranked by average rating amount movies"
    "with at least 50 ratings."
)

st.divider()

st.caption(
    "Built with Python. Pandas, Streamlit, and Surprise SVD"
    "using the MovieLenns dataset."
)