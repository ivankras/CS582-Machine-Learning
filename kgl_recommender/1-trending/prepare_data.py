import pandas as pd

def get_trending(num):
    # Load data
    imdb_movies = pd.read_csv('../IMDb movies.csv')
    imdb_ratings = pd.read_csv('../IMDb ratings.csv')

    # Drop columns
    X = imdb_movies[['imdb_title_id', 'title']]

    X_ratings = imdb_ratings[['imdb_title_id', 'weighted_average_vote']]

    X = X.merge(X_ratings, on='imdb_title_id').sort_values('weighted_average_vote', ascending=False)[0:num]

    return X

def get_trending_10():
    return get_trending(10)