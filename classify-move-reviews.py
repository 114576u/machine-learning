

# Sample taken from
# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub

# This notebook classifies movie reviews as positive or negative using the text of the review. This is an example of
#   binary—or two-class—classification, an important and widely applicable kind of machine learning problem.
# The tutorial demonstrates the basic application of transfer learning with TensorFlow Hub and Keras.
# We'll use the IMDB dataset that contains the text of 50,000 movie reviews from the Internet Movie Database. These
#   are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are
#   balanced, meaning they contain an equal number of positive and negative reviews.
# This notebook uses tf.keras, a high-level API to build and train models in TensorFlow, and TensorFlow Hub, a library
#   and platform for transfer learning. For a more advanced text classification tutorial using tf.keras, see the
#   MLCC Text Classification Guide.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

# !pip install -q tensorflow-hub
# !pip install -q tensowrflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# DOWNLOAD THE IMDB dataset
# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# EXPLORE THE DATA
# print first 10 examples
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print("-----train_examples_batch--------------")
print(train_examples_batch)

print("-----train_labels_batch----------------")
print(train_labels_batch)

# BUILD THE MODEL
# in this case the input data consists of sentences and the labels to predict are either 0 or 1
# we are using in this example a pre-trained text embedding model from TensorFlow Hub called google/t2-preview/gnews-swivel-20dim/1
#   there are other pre-trained models:
#   google/tf2-preview/gnews-swivel-20dim-with-oov/1
#   google/tf2-preview/nnlm-en-dim50/1  => larger model with 1M vocabulary size and 50 dimensions
#   google/tf2-preview/nnlm-en-dim128/1 => 1M vocabulary size and 128 dimensions

# we first create a Keras layer that uses TensorFlow Hub model to embed sentences
# no matter the length of the input tet, the output shape of the embeddings is (num_examples, embedding_dimension)
embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'
hub_layer = hub.KerasLayer(embedding,
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True)
print("hub_layer(train_examples_batch:")
hub_layer(train_examples_batch[:3])

# let's now build the full model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
print("model.summary(): ")
model.summary()

# layers are stocked sequentially to build the classifier

# COMPILE THE MODEL
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# TRAIN THE MODEL
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# EVALUATE THE MODEL
results = model.evaluate(test_data.batch(512),
                         verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))