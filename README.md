# MovieMind — Movie Recommendation System

MovieMind is a machine learning project that recommends movies based on user rating patterns using collaborative filtering and matrix factorization.

## Features

- User-based collaborative filtering
- Item-based collaborative filtering
- Matrix factorization using Surprise SVD
- Matrix factorization implemented from scratch using NumPy
- Model evaluation using RMSE and MAE
- Interactive Streamlit application
- Popular movie exploration

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Scikit-Surprise
- Streamlit

## Dataset

This project uses the [MovieLens Small dataset](https://grouplens.org/datasets/movielens/latest/).

Download the dataset and place `movies.csv` and `ratings.csv` inside:

`data/ml-latest-small/`

## Setup

Clone the repository:

```bash
git clone https://github.com/ritikagoyal01/movie-recommendation-system
cd movie-recommendation-system
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Model Evaluation

The models were evaluated using RMSE and MAE on a held-out test set.

| Model | RMSE | MAE |
|---|---:|---:|
| Surprise SVD | 0.8808 | 0.6739 |
| User-based similarity | 0.9755 | 0.7536 |
| Scratch matrix factorization | 1.0488 | 0.8316 |

Lower values indicate better rating prediction accuracy on the evaluated test set.

## Future Improvements

- React and FastAPI frontend/backend architecture
- Movie poster integration
- User-provided movie preferences
- Recommendation ranking metrics
