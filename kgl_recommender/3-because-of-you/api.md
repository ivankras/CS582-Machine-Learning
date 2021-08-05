# Collaborative Model - API Description
## (Because of you)
----

_NOTE: Input files (.csv) are located in "../input/" directory_

### SVD (svd.py)
```py
generate_model_svd(small=True)
# set small=False for using ratings.csv instead of ratings_small.csv
# returns SVD() (model) after saving it in "models/svd" (or "models/svd_small")

# -------------------------------------

predict_score_svd(user_id, movie_id, svd=None)
# set svd for indicating an already trained model; otherwise, it will call generate_model_svd()
# returns Prediction object
#   movie id on "iid", prediction score in "est"
#   Prediction(uid=2, iid=194880, r_ui=None, est=3.4515549984825915, details={'was_impossible': False})

# -------------------------------------

get_all_predictions_svd(user_id=n, small=True)
# depends on pre-trained SVD() model ("models/svd" or "models/svd_small", depending on small argument)
# returns all movie predictions (Prediction object) for user_id

# -------------------------------------

get_top_k_predictions(predictions_user_n, k=10)
# predictions_user_n is the result from get_all_predictions_svd()
# returns top k predictions (Prediction object)


# -------------------------------------
# -------------------------------------

# Additional: to convert result from get_top_k_predictions() into movieId:movieTitle
top_preds = get_top_k_predictions(predictions_user_n, k=10)

# Discard scores and everything that's not the movieId
top_preds = [int(pred.iid) for pred in top_preds]

# Get information for top movies
top_movies = get_movies_from_ids(top_preds)
print(top_movies['original_title'])

```
----
### NN (nn.py)
```py
get_top_scores(user_id, k, train=False, small=True)
# set small=False for using ratings.csv instead of ratings_small.csv
# set train=True if the model is not already trained; it will call _get_predictor_nn()
# returns SVD() (model) after saving it in "models/svd" (or "models/svd_small")

# -------------------------------------
# -------------------------------------

# Additional: to convert result from get_top_scores() into movieId:movieTitle
from eda import get_movies_from_ids

# Discard scores and everything that's not the movieId
top_movies = [movie for movie, _ in top_preds]

# Get information for top movies
top_movies = get_movies_from_ids(top_movies)
top_movies['original_title']

```