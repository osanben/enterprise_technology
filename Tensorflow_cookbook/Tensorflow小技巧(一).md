## how-do-i-select-rows-from-a-dataframe-based-on-column-values

To select rows whose column value equals a scalar, `some_value`, use `==`:

```py
df.loc[df['column_name'] == some_value]
```

To select rows whose column value is in an iterable, `some_values`, use `isin`:

```py
df.loc[df['column_name'].isin(some_values)]
```

## how-do-i-sort-a-dictionary-by-value

```py
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
dict(sorted(x.items(), key=lambda item: item[1]))
```

## how-can-i-count-the-occurrences-of-a-list-item

```py
from collections import Counter

l = ["a","b","b"]
Counter(l)
```

## pandas.DataFrame.drop_duplicates

```
df = pd.DataFrame({
...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
...     'rating': [4, 4, 3.5, 15, 5]
... })

df.drop_duplicates(subset=['brand'])
```

## tf.data.Dataset-----as_numpy_iterator()

Returns an iterator which converts all elements of the dataset to numpy.

```
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in dataset.as_numpy_iterator():
  print(element)
```

## tf.data.Dataset

The `tf.data.Dataset` API supports writing descriptive and efficient input pipelines. `Dataset` usage follows a common pattern:

1. Create a source dataset from your input data.
2. Apply dataset transformations to preprocess the data.
3. Iterate over the dataset and process the elements.

Iteration happens in a streaming fashion, so the full dataset does not need to fit into memory.

The simplest way to create a dataset is to create it from a python list:

```
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
for element in dataset:
  print(element)
```

Once you have a dataset, you can apply transformations to prepare the data for your model:

```
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(lambda x: x*2)
list(dataset.as_numpy_iterator())
```