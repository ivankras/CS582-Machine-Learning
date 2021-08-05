import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _get_similarity_by_description(dataset):
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    dataset['description'] = dataset['description'].fillna('')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(dataset['description'])

    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def _get_movies_by_description(dataset, imdb_title_id):
    similarity = _get_similarity_by_description(dataset)

    dataset = dataset.reset_index()
    # titles = dataset['imdb_title_id']
    indices = pd.Series(dataset.index, index=dataset['imdb_title_id'])

    # print(dataset['imdb_title_id'][0:10])
    index = indices[imdb_title_id]
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores]

    return dataset.iloc[movie_indices]
    

def _get_movies_by_gcpt(dataset, imdb_title_id):
    # TODO: Similarity by genre, country, production_company and title_principals
    return range(1,11)


def get_related_movies(imdb_title_id, num, criterion='simple'):
    # Load data
    imdb_movies = pd.read_csv('../IMDb movies.csv')
    # imdb_names = pd.read_csv('../IMDb names.csv')
    # imdb_ratings = pd.read_csv('../IMDb ratings.csv')
    imdb_ppals = pd.read_csv('../IMDb title_principals.csv')

    X = imdb_movies.drop(
        [
            'title',
            'date_published',
            'director',
            'writer',
            'actors',
            'usa_gross_income',
            'worlwide_gross_income',
            'budget',
            'metascore'
        ],
        axis=1
    )

    # Transform string years into integer
    X['year'] = X['year'].apply(lambda x: int(x.split(' ')[-1]) if not isinstance(x, int) else x)

    X_ppals = imdb_ppals[['imdb_title_id', 'imdb_name_id', 'category']]

    # TODO: remove constraint if model can be applied somewhere else
    # Discard old movies
    X = X[X['year'] >= 1992]
    if criterion == 'simple':
        X['description'] = X['description'].fillna('')
        return _get_movies_by_description(X, imdb_title_id)[0:num]
    else:
        X = X.merge(X_ppals, on='imdb_title_id')
        return _get_movies_by_gcpt(X, imdb_title_id)[0:num]

    # TODO: cache results


def get_10_related_movies(imdb_title_id):
    return get_related_movies(imdb_title_id, 10)
    # return get_related_movies(imdb_title_id, 10, 'complex')
