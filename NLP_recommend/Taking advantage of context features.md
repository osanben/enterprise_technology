In the featurization tutorial we incorporated multiple features beyond just user and movie identifiers into our models, but we haven't explored whether those features improve model accuracy.

Many factors affect whether features beyond ids are useful in a recommender model:

1. Importance of context: if user preferences are relatively stable across contexts and time, context features may not provide much benefit. If, however, users preferences are highly contextual, adding context will improve the model significantly. For example, day of the week may be an important feature when deciding whether to recommend a short clip or a movie: users may only have time to watch short content during the week, but can relax and enjoy a full-length movie during the weekend. Similarly, query timestamps may play an important role in modelling popularity dynamics: one movie may be highly popular around the time of its release, but decay quickly afterwards. Conversely, other movies may be evergreens that are happily watched time and time again.
2. Data sparsity: using non-id features may be critical if data is sparse. With few observations available for a given user or item, the model may struggle with estimating a good per-user or per-item representation. To build an accurate model, other features such as item categories, descriptions, and images have to be used to help the model generalize beyond the training data. This is especially relevant in cold-start situations, where relatively little data is available on some items or users.

```
import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs
```

We follow the featurization tutorial and keep the user id, timestamp, and movie title features.

```
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])
```

We also do some housekeeping to prepare feature vocabularies.

```
timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))
```

## Model definition

### Query model

We start with the user model defined in the featurization tutorial as the first layer of our model, tasked with converting raw input examples into feature embeddings. However, we change it slightly to allow us to turn timestamp features on or off. This will allow us to more easily demonstrate the effect that timestamp features have on the model. In the code below, the use_timestamps parameter gives us control over whether we use timestamp features.

```
class UserModel(tf.keras.Model):

  def __init__(self, use_timestamps):
    super().__init__()

    self._use_timestamps = use_timestamps

    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
    ])

    if use_timestamps:
      self.timestamp_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
          tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
      ])
      self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

      self.normalized_timestamp.adapt(timestamps)

  def call(self, inputs):
    if not self._use_timestamps:
      return self.user_embedding(inputs["user_id"])

    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        self.normalized_timestamp(inputs["timestamp"]),
    ], axis=1)
```

Note that our use of timestamp features in this tutorial interacts with our choice of training-test split in an undesirable way. Because we have split our data randomly rather than chronologically (to ensure that events that belong to the test dataset happen later than those in the training set), our model can effectively learn from the future. This is unrealistic: after all, we cannot train a model today on data from tomorrow.

This means that adding time features to the model lets it learn future interaction patterns. We do this for illustration purposes only: the MovieLens dataset itself is very dense, and unlike many real-world datasets does not benefit greatly from features beyond user ids and movie titles.

This caveat aside, real-world models may well benefit from other time-based features such as time of day or day of the week, especially if the data has strong seasonal patterns.

### Candidate model

For simplicity, we'll keep the candidate model fixed. Again, we copy it from the featurization tutorial:

```
class MovieModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
          vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
    ])

    self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_tokens)

    self.title_text_embedding = tf.keras.Sequential([
      self.title_vectorizer,
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

    self.title_vectorizer.adapt(movies)

  def call(self, titles):
    return tf.concat([
        self.title_embedding(titles),
        self.title_text_embedding(titles),
    ], axis=1)
```

### Combined model

With both UserModel and MovieModel defined, we can put together a combined model and implement our loss and metrics logic.

Note that we also need to make sure that the query model and candidate model output embeddings of compatible size. Because we'll be varying their sizes by adding more features, the easiest way to accomplish this is to use a dense projection layer after each model:

```
class MovielensModel(tfrs.models.Model):

  def __init__(self, use_timestamps):
    super().__init__()
    self.query_model = tf.keras.Sequential([
      UserModel(use_timestamps),
      tf.keras.layers.Dense(32)
    ])
    self.candidate_model = tf.keras.Sequential([
      MovieModel(),
      tf.keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "user_id": features["user_id"],
        "timestamp": features["timestamp"],
    })
    movie_embeddings = self.candidate_model(features["movie_title"])

    return self.task(query_embeddings, movie_embeddings)
```

## Experiments

### Prepare the data

We first split the data into a training set and a testing set.

```
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()
```

### Baseline: no timestamp features

We're ready to try out our first model: let's start with not using timestamp features to establish our baseline.

```
model = MovielensModel(use_timestamps=False)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")
```

This gives us a baseline top-100 accuracy of around 0.2

### Capturing time dynamics with time features

Do the result change if we add time features?

```
model = MovielensModel(use_timestamps=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")
```

This is quite a bit better: not only is the training accuracy much higher, but the test accuracy is also substantially improved.

代码链接: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/NLP_recommend/Taking%20advantage%20of%20context%20features.ipynb