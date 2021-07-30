Real-world recommender systems are often composed of two stages:

1. The retrieval stage is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.
2. The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

In this tutorial, we're going to focus on the first stage, retrieval. If you are interested in the ranking stage, have a look at our ranking tutorial.

Retrieval models are often composed of two sub-models:

1. A query model computing the query representation (normally a fixed-dimensionality embedding vector) using query features.
2. A candidate model computing the candidate representation (an equally-sized vector) using the candidate features

The outputs of the two models are then multiplied together to give a query-candidate affinity score, with higher scores expressing a better match between the candidate and the query.

In this tutorial, we're going to build and train such a two-tower model using the Movielens dataset.

We're going to:

1. Get our data and split it into a training and test set.
2. Implement a retrieval model.
3. Fit and evaluate it.
4. Export it for efficient serving by building an approximate nearest neighbours (ANN) index.

## The dataset

It contains a set of ratings given to movies by a set of users, and is a workhorse of recommender system research

The data can be treated in two ways:

1. It can be interpreted as expressesing which movies the users watched (and rated), and which they did not. This is a form of implicit feedback, where users' watches tell us which things they prefer to see and which they'd rather not see.
2. It can also be seen as expressesing how much the users liked the movies they did watch. This is a form of explicit feedback: given that a user watched a movie, we can tell roughly how much they liked by looking at the rating they have given.

In this tutorial, we are focusing on a retrieval system: a model that predicts a set of movies from the catalogue that the user is likely to watch. Often, implicit data is more useful here, and so we are going to treat Movielens as an implicit system. This means that every movie a user watched is a positive example, and every movie they have not seen is an implicit negative example.

## Imports

```
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs
```

## Preparing the dataset

We use the MovieLens dataset from Tensorflow Datasets. Loading `movielens/100k_ratings` yields a `tf.data.Dataset` object containing the ratings data and loading `movielens/100k_movies` yields a `tf.data.Dataset` object containing only the movies data.

```
# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")
```

The ratings dataset returns a dictionary of movie id, user id, the assigned rating, timestamp, movie information, and user information:

```
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)
```

The movies dataset contains the movie id, movie title, and data on what genres it belongs to. Note that the genres are encoded with integer labels.

```
for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)
```

In this example, we're going to focus on the ratings data. Other tutorials explore how to use the movie information data as well to improve the model quality.

We keep only the `user_id`, and `movie_title` fields in the dataset.

```
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])
```

In this simple example, however, let's use a random split, putting 80% of the ratings in the train set, and 20% in the test set.

```
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
```

Let's also figure out unique user ids and movie titles present in the data.

This is important because we need to be able to map the raw values of our categorical features to embedding vectors in our models. To do that, we need a vocabulary that maps a raw feature value to an integer in a contiguous range: this allows us to look up the corresponding embeddings in our embedding tables.

```
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]
```

## Implementing a model

Choosing the architecture of our model is a key part of modelling.

Because we are building a two-tower retrieval model, we can build each tower separately and then combine them in the final model.

### The query tower

The first step is to decide on the dimensionality of the query and candidate representations:

```
embedding_dimension = 32
```

Higher values will correspond to models that may be more accurate, but will also be slower to fit and more prone to overfitting.

The second is to define the model itself. Here, we're going to use Keras preprocessing layers to first convert user ids to integers, and then convert those to user embeddings via an Embedding layer. Note that we use the list of unique user ids we computed earlier as a vocabulary:

```
user_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])
```

A simple model like this corresponds exactly to a classic matrix factorization approach. While defining a subclass of tf.keras.Model for this simple model might be overkill, we can easily extend it to an arbitrarily complex model using standard Keras components, as long as we return an embedding_dimension-wide output at the end.

### The candidate tower

```
movie_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])
```

### Metrics

In our training data we have positive (user, movie) pairs. To figure out how good our model is, we need to compare the affinity score that the model calculates for this pair to the scores of all the other possible candidates: if the score for the positive pair is higher than for all other candidates, our model is highly accurate.

To do this, we can use the tfrs.metrics.FactorizedTopK metric. The metric has one required argument: the dataset of candidates that are used as implicit negatives for evaluation.

In our case, that's the movies dataset, converted into embeddings via our movie model:

```
metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)
```

### Loss

In this instance, we'll make use of the Retrieval task object: a convenience wrapper that bundles together the loss function and metric computation:

```
task = tfrs.tasks.Retrieval(
  metrics=metrics
)
```

The task itself is a Keras layer that takes the query and candidate embeddings as arguments, and returns the computed loss: we'll use that to implement the model's training loop.

### The full model

TFRS exposes a base model class (tfrs.models.Model) which streamlines building models: all we need to do is to set up the components in the __init__ method, and implement the compute_loss method, taking in the raw features and returning a loss value.

The base model will then take care of creating the appropriate training loop to fit our model.

```
class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)
```

## Fitting and evaluating

Let's first instantiate the model.

```
model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
```

Then shuffle, batch, and cache the training and evaluation data.

```
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
```

Then train the model:

```
model.fit(cached_train, epochs=3)
```

As the model trains, the loss is falling and a set of top-k retrieval metrics is updated. These tell us whether the true positive is in the top-k retrieved items from the entire candidate set. For example, a top-5 categorical accuracy metric of 0.2 would tell us that, on average, the true positive is in the top 5 retrieved items 20% of the time.

Note that, in this example, we evaluate the metrics during training as well as evaluation. Because this can be quite slow with large candidate sets, it may be prudent to turn metric calculation off in training, and only run it in evaluation.

Finally, we can evaluate our model on the test set:

```
model.evaluate(cached_test, return_dict=True)
```

Test set performance is much worse than training performance. This is due to two factors:

1. Our model is likely to perform better on the data that it has seen, simply because it can memorize it. This overfitting phenomenon is especially strong when models have many parameters. It can be mediated by model regularization and use of user and movie features that help the model generalize better to unseen data.
2. The model is re-recommending some of users' already watched movies. These known-positive watches can crowd out test movies out of top K recommendations.

The second phenomenon can be tackled by excluding previously seen movies from test recommendations. This approach is relatively common in the recommender systems literature, but we don't follow it in these tutorials. If not recommending past watches is important, we should expect appropriately specified models to learn this behaviour automatically from past user history and contextual information. Additionally, it is often appropriate to recommend the same item multiple times (say, an evergreen TV series or a regularly purchased item).

## Making predictions

Now that we have a model, we would like to be able to make predictions. We can use the tfrs.layers.factorized_top_k.BruteForce layer to do this.

```
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index(movies.batch(100).map(model.movie_model), movies)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")
```

Of course, the BruteForce layer is going to be too slow to serve a model with many possible candidates. The following sections shows how to speed this up by using an approximate retrieval index.

## Model serving

In a two-tower retrieval model, serving has two components:

- a serving query model, taking in features of the query and transforming them into a query embedding, and
- a serving candidate model. This most often takes the form of an approximate nearest neighbours (ANN) index which allows fast approximate lookup of candidates in response to a query produced by the query model.

In TFRS, both components can be packaged into a single exportable model, giving us a model that takes the raw user id and returns the titles of top movies for that user. This is done via exporting the model to a SavedModel format, which makes it possible to serve using TensorFlow Serving.

To deploy a model like this, we simply export the `BruteForce` layer we created above:

```
# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  # Save the index.
  index.save(path)

  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.keras.models.load_model(path)

  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])

  print(f"Recommendations: {titles[0][:3]}")
```

We can also export an approximate retrieval index to speed up predictions. This will make it possible to efficiently surface recommendations from sets of tens of millions of candidates.

```
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index(movies.batch(100).map(model.movie_model), movies)
```

This layer will perform approximate lookups: this makes retrieval slightly less accurate, but orders of magnitude faster on large candidate sets

```
# Get recommendations.
_, titles = scann_index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")
```

Exporting it for serving is as easy as exporting the `BruteForce` layer:

```
# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  # Save the index.
  scann_index.save(
      path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )

  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.keras.models.load_model(path)

  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])

  print(f"Recommendations: {titles[0][:3]}")
```

代码链接: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/NLP_recommend/Recommending%20movies:%20retrieval.ipynb

