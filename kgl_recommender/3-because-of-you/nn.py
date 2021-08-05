import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import (Model, load_model)
from tensorflow.keras.layers import (
    Input, Reshape, Activation,
    Dropout, Dense, Concatenate, Lambda, Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from eda import (
    get_ratings_small,
    get_ratings,
    get_movies_ids
)

n_factors_embedding = 50
nn_model_dir = 'models/nn'
nn_small_model_dir = 'models/nn_small'

class EmbeddingLayer:
    # Taken from https://medium.com/@jdwittenauer/deep-learning-with-keras-recommender-systems-e7b99cb29929
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors
    
    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x

def _prepare_data_nn(small=True):
    # ---------
    # Load data
    # ---------
    ratings = get_ratings_small() if small else get_ratings()

    # ---------
    # Prepare input
    # ---------
    user_enc = LabelEncoder()
    ratings['user'] = user_enc.fit_transform(ratings['userId'].values)
    n_users = ratings['user'].nunique()

    item_enc = LabelEncoder()
    ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)
    n_movies = ratings['movie'].nunique()

    ratings['rating'] = ratings['rating'].values.astype(np.float32)
    min_rating = min(ratings['rating'])
    max_rating = max(ratings['rating'])

    X = ratings[['user', 'movie']].values
    y = ratings['rating'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1 if small else 0.25, random_state=12)

    X_train_array = [X_train[:, 0], X_train[:, 1]]
    X_test_array = [X_test[:, 0], X_test[:, 1]]

    data = {
        'X_train': X_train_array,
        'X_test': X_test_array,
        'y_train': y_train,
        'y_test': y_test
    }

    return data, n_users, n_movies, min_rating, max_rating

def _get_recommender_nn(n_users, n_movies, n_factors, min_rating, max_rating):
    user = Input(shape=(1,))
    u = EmbeddingLayer(n_users+1, n_factors)(user)

    movie = Input(shape=(1,))
    m = EmbeddingLayer(n_movies+1, n_factors)(movie)

    x = Concatenate()([u, m])
    x = Dropout(0.05)(x)
    
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = Model(inputs=[user, movie], outputs=x)
    opt = Adam(learning_rate=0.001)

    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

def _train_nn(model, data, small=True):
    X_train, X_test, y_train, y_test = data.values()
    training_history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=64,
        epochs=4 if small else 8,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    nn_dump_dir = nn_small_model_dir if small else nn_model_dir
    model.save(nn_dump_dir)

    return training_history, model

def _get_predictor_nn(train=False, small=True):
    if train:
        data, n_users, n_movies, min_rating, max_rating = _prepare_data_nn(small)
        model = _get_recommender_nn(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=n_factors_embedding,
            min_rating=min_rating,
            max_rating=max_rating
        )

        training_history, trained_model = _train_nn(model, data, small)
    else:
        nn_load_file = nn_small_model_dir if small else nn_model_dir
        trained_model = load_model(nn_load_file)
        training_history = None

    return training_history, trained_model

def predict_score_nn(user_ids, movie_ids, trained_model):
    predictions = trained_model.predict(x=[np.array(user_ids), np.array(movie_ids)])
    return trained_model, predictions

def get_top_scores(user_id, k, train=False, small=True):
    _, trained_model = _get_predictor_nn(train, small)

    ratings = get_ratings_small() if small else get_ratings()
    ratings = ratings.loc[ratings['userId'] == user_id]

    movies_ids = get_movies_ids()
    print('All movies: ', len(movies_ids))
    movies_ids = [movie for movie in movies_ids if len(ratings.loc[ratings['movieId'] == movie]) == 0]
    movies_ids = [movie for movie in movies_ids if movie in ratings['movieId']]
    print('All movies: ', len(movies_ids))

    user_input = [user_id for _ in range(len(movies_ids))]
    _, predictions = predict_score_nn(user_input, movies_ids, trained_model)

    movies = {movies_ids[i] : predictions[i] for i in range(len(movies_ids))}
    movies = sorted(movies.items(), key=lambda k: k[1], reverse=True)

    return movies[:k]
