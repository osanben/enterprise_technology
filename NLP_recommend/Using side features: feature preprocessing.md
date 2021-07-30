One of the great advantages of using a deep learning framework to build recommender models is the freedom to build rich, flexible feature representations.

These need to be appropriately transformed in order to be useful in building models:

+ User and item ids have to be translated into embedding vectors: high-dimensional numerical representations that are adjusted during training to help the model predict its objective better.
+ Raw text needs to be tokenized (split into smaller parts such as individual words) and translated into embeddings.
+ Numerical features need to be normalized so that their values lie in a small interval around 0.

## The MovieLens dataset

Let's first have a look at what features we can use from the MovieLens dataset:

```
import pprint

import tensorflow_datasets as tfds

ratings = tfds.load("movielens/100k-ratings", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)
```

There are a couple of key features here:

+ Movie title is useful as a movie identifier.
+ User id is useful as a user identifier.
+ Timestamps will allow us to model the effect of time.

The first two are categorical features; timestamps are a continuous feature.

## Turning categorical features into embeddings

A categorical feature is a feature that does not express a continuous quantity, but rather takes on one of a set of fixed values.

Most deep learning models express these feature by turning them into high-dimensional vectors. During model training, the value of that vector is adjusted to help the model predict its objective better.
For example, suppose that our goal is to predict which user is going to watch which movie. To do that, we represent each user and each movie by an embedding vector. Initially, these embeddings will take on random values - but during training, we will adjust them so that embeddings of users and the movies they watch end up closer together.
Taking raw categorical features and turning them into embeddings is normally a two-step process:

1. Firstly, we need to translate the raw values into a range of contiguous integers, normally by building a mapping (called a "vocabulary") that maps raw values ("Star Wars") to integers (say, 15)

2. Secondly, we need to take these integers and turn them into embeddings.

### Defining the vocabulary

The first step is to define a vocabulary. We can do this easily using Keras preprocessing layers.

```
import numpy as np
import tensorflow as tf

movie_title_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
```

The layer itself does not have a vocabulary yet, but we can build it using our data.

```
movie_title_lookup.adapt(ratings.map(lambda x: x["movie_title"]))

print(f"Vocabulary: {movie_title_lookup.get_vocabulary()[:3]}")
```

Once we have this we can use the layer to translate raw tokens to embedding ids:

```
movie_title_lookup(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"])
```

Note that the layer's vocabulary includes one (or more!) unknown (or "out of vocabulary", OOV) tokens. This is really handy: it means that the layer can handle categorical values that are not in the vocabulary. In practical terms, this means that the model can continue to learn about and make recommendations even using features that have not been seen during vocabulary construction.

### Using feature hashing

We can take this to its logical extreme and rely entirely on feature hashing, with no vocabulary at all. This is implemented in the tf.keras.layers.experimental.preprocessing.Hashing layer.

```
# We set up a large number of bins to reduce the chance of hash collisions.
num_hashing_bins = 200_000

movie_title_hashing = tf.keras.layers.experimental.preprocessing.Hashing(
    num_bins=num_hashing_bins
)
```

We can do the lookup as before without the need to build vocabularies:

```
movie_title_hashing(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"])
```

### Defining the embeddings

Now that we have integer ids, we can use the Embedding layer to turn those into embeddings.

An embedding layer has two dimensions: the first dimension tells us how many distinct categories we can embed; the second tells us how large the vector representing each of them can be.

When creating the embedding layer for movie titles, we are going to set the first value to the size of our title vocabulary (or the number of hashing bins). The second is up to us: the larger it is, the higher the capacity of the model, but the slower it is to fit and serve.

```
movie_title_embedding = tf.keras.layers.Embedding(
    # Let's use the explicit vocabulary lookup.
    input_dim=movie_title_lookup.vocab_size(),
    output_dim=32
)
```

We can put the two together into a single layer which takes raw text in and yields embeddings.

```
movie_title_model = tf.keras.Sequential([movie_title_lookup, movie_title_embedding])
```

Just like that, we can directly get the embeddings for our movie titles:

```
movie_title_model(["Star Wars (1977)"])
```

We can do the same with user embeddings:

```
user_id_lookup = tf.keras.layers.experimental.preprocessing.StringLookup()
user_id_lookup.adapt(ratings.map(lambda x: x["user_id"]))

user_id_embedding = tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32)

user_id_model = tf.keras.Sequential([user_id_lookup, user_id_embedding])
```

## Normalizing continuous features

Continuous features also need normalization. For example, the timestamp feature is far too large to be used directly in a deep model

```
for x in ratings.take(3).as_numpy_iterator():
  print(f"Timestamp: {x['timestamp']}.")
```

We need to process it before we can use it. While there are many ways in which we can do this, discretization and standardization are two common ones.

### Standardization

Standardization rescales features to normalize their range by subtracting the feature's mean and dividing by its standard deviation. It is a common preprocessing transformation.

This can be easily accomplished using the tf.keras.layers.experimental.preprocessing.Normalization layer:

```
timestamp_normalization = tf.keras.layers.experimental.preprocessing.Normalization()
timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))

for x in ratings.take(3).as_numpy_iterator():
  print(f"Normalized timestamp: {timestamp_normalization(x['timestamp'])}.")
```

### Discretization

Another common transformation is to turn a continuous feature into a number of categorical features. This makes good sense if we have reasons to suspect that a feature's effect is non-continuous.
To do this, we first need to establish the boundaries of the buckets we will use for discretization. The easiest way is to identify the minimum and maximum value of the feature, and divide the resulting interval equally:

```
max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()
min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000)

print(f"Buckets: {timestamp_buckets[:3]}")
```

Given the bucket boundaries we can transform timestamps into embeddings:

```
timestamp_embedding_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
  tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)
])

for timestamp in ratings.take(1).map(lambda x: x["timestamp"]).batch(1).as_numpy_iterator():
  print(f"Timestamp embedding: {timestamp_embedding_model(timestamp)}.")
```

## Processing text features

We may also want to add text features to our model. Usually, things like product descriptions are free form text, and we can hope that our model can learn to use the information they contain to make better recommendations, especially in a cold-start or long tail scenario.
While the MovieLens dataset does not give us rich textual features, we can still use movie titles. This may help us capture the fact that movies with very similar titles are likely to belong to the same series.
The first transformation we need to apply to text is tokenization (splitting into constituent words or word-pieces), followed by vocabulary learning, followed by an embedding.

The Keras tf.keras.layers.experimental.preprocessing.TextVectorization layer can do the first two steps for us:

```
title_text = tf.keras.layers.experimental.preprocessing.TextVectorization()
title_text.adapt(ratings.map(lambda x: x["movie_title"]))
```

Let's try it out:

```
for row in ratings.batch(1).map(lambda x: x["movie_title"]).take(1):
  print(title_text(row))
```

Each title is translated into a sequence of tokens, one for each piece we've tokenized.

We can check the learned vocabulary to verify that the layer is using the correct tokenization:

```
title_text.get_vocabulary()[40:45]
```

This looks correct: the layer is tokenizing titles into individual words.
To finish the processing, we now need to embed the text. Because each title contains multiple words, we will get multiple embeddings for each title. For use in a donwstream model these are usually compressed into a single embedding. Models like RNNs or Transformers are useful here, but averaging all the words' embeddings together is a good starting point.

## Putting it all together

With these components in place, we can build a model that does all the preprocessing together.

### User model

The full user model may look like the following:

```
class UserModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        user_id_lookup,
        tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32),
    ])
    self.timestamp_embedding = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
      tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
    ])
    self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

  def call(self, inputs):

    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        self.normalized_timestamp(inputs["timestamp"])
    ], axis=1)
```

Let's try it out:

```
user_model = UserModel()

user_model.normalized_timestamp.adapt(
    ratings.map(lambda x: x["timestamp"]).batch(128))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {user_model(row)[0, :3]}")
```

### Movie model

We can do the same for the movie model:

```
class MovieModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = tf.keras.Sequential([
      movie_title_lookup,
      tf.keras.layers.Embedding(movie_title_lookup.vocab_size(), 32)
    ])
    self.title_text_embedding = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=max_tokens),
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      # We average the embedding of individual words to get one embedding vector
      # per title.
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

  def call(self, inputs):
    return tf.concat([
        self.title_embedding(inputs["movie_title"]),
        self.title_text_embedding(inputs["movie_title"]),
    ], axis=1)
```

Let's try it out:

```
movie_model = MovieModel()

movie_model.title_text_embedding.layers[0].adapt(
    ratings.map(lambda x: x["movie_title"]))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {movie_model(row)[0, :3]}")
```

代码地址: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/NLP_recommend/Using%20side%20features:%20feature%20preprocessing.ipynb