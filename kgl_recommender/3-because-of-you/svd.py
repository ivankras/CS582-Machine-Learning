import numpy as np
import pandas as pd
from concurrent import futures
from joblib import (load, dump)
from surprise import (
    Reader,
    Dataset,
    SVD
)
# from surprise.model_selection import cross_validate

from eda import (
    get_movies_ids,
    get_ratings_small,
    get_ratings
)

svd_model_file = 'models/svd'
svd_small_model_file = 'models/svd_small'

def generate_model_svd(small=True):
    # ---------
    # Load data
    # ---------
    reader = Reader()
    ratings = get_ratings_small() if small else get_ratings()
    data = Dataset.load_from_df(df=ratings[['userId', 'movieId', 'rating']], reader=reader)
    
    # ---------
    # Generate SVD
    # ---------
    svd = SVD()
    # Evaluate the model for the data
    # cross_validate(algo=svd, data=data, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, verbose=True)
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    dump(svd, svd_small_model_file if small else svd_model_file, 9)

    return svd

def predict_score_svd(user_id, movie_id, svd=None):
    svd = svd if svd is not None else generate_model_svd()
    return svd.predict(user_id, movie_id)

def get_all_predictions_svd(user_id, small=True):
    # svd = generate_model_svd()
    svd = load(svd_small_model_file if small else svd_model_file)

    movies_ids = get_movies_ids()

    ratings = get_ratings_small() if small else get_ratings()
    ratings = ratings.loc[ratings['userId'] == user_id]

    predictions_table = []

    with futures.ThreadPoolExecutor(max_workers=None) as executor:
        threads = []
        movies_ids = [movie for movie in movies_ids if len(ratings.loc[ratings['movieId'] == movie]) == 0]
        for movie in movies_ids:
            threads.append(executor.submit(predict_score_svd, user_id, int(movie), svd))
        
        for future in futures.as_completed(threads):
            predictions_table.append(future.result())

    return predictions_table

def get_top_k_predictions(predictions_table, k):
    predictions_table = sorted(predictions_table, key=lambda k: k.est, reverse=True)
    return predictions_table[:k]
