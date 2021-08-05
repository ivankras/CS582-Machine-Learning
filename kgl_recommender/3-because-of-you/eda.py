import pandas as pd

ratings_small_input_file = '../input/ratings_small.csv'
ratings_input_file = '../input/ratings.csv'
movies_input_file = '../input/movies_metadata.csv'

def get_ratings_small():
    ratings = pd.read_csv(ratings_small_input_file)
    return ratings

def get_ratings(size=-1):
    ratings = pd.read_csv(ratings_input_file)
    if size > 0:
        ratings = ratings[:size]
    return ratings

def _drop_odd_ids(movies):
    # Delete odd ids (e.g.: dates)
    odd_ids = [id for id in movies['id'] if len(id) > 6]
    movies = movies.drop(movies.loc[movies['id'].isin(odd_ids)].index, axis=0)
    return movies

def get_movies_ids():
    movies = pd.read_csv(movies_input_file)
    movies = _drop_odd_ids(movies)
    movies_ids = movies['id'].apply(int)

    return movies_ids

def get_movies_from_ids(movies_ids):
    movies = pd.read_csv(movies_input_file)
    movies = _drop_odd_ids(movies)
    movies['id'] = movies['id'].apply(int)
    return movies.loc[movies['id'].isin(movies_ids)]
