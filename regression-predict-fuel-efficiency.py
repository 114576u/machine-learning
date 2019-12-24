
# Sample taken from
# https://www.tensorflow.org/tutorials/keras/regression

# when in regression problems, we aim to PREDICT the output of a continuous value (vs SELECT when in classification problems)

# this example uses a car database with models of 70's and 80's

# use:
# pip install -q searborn
# pip install -q git+https://github.com/tensorflow/docs

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print(tfdocs)

# get the Auto MPG dataset
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

# import using pandas
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values= "?",
                          comment='\t',
                          sep=" ",
                          skipinitialspace=True
                          )
dataset = raw_dataset.copy()
print(dataset.tail())

# check whether there are unknown values:
print(dataset.isna().sum())

# drop those rows
dataset = dataset.dropna()

# Origin column is categorical, not numeric, so converting doing this:
dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

# split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# inspect the data: have a quick look at the joint distribution of a few pairs of columns from the training set
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()
# ====> interesting note: seaborn plots are shown as if these were usual matplotlib plots, so simply calling plt.show

# getting overall statistics
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# split features from labels
# Separate the target value, or "label", from the features. This label is the value
# that we will train the model to predict.
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# normalize data => is the data we'll use to train the model
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# THE MODEL

# build the model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()

# inspect the model
print(model.summary())

# try out the mode: take a batch of 20 exampels from the training data and call model.predict:
example_batch = normed_train_data[:20]
example_result = model.predict(example_batch)
print(example_result)

# TRAIN THE MODEL
EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

# Visualize the model's training progress using the stats stored in the history object.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.show()

# This graph shows little improvement, or even degradation in the validation error after
# about 100 epochs. Let's update the model.fit call to automatically stop training when
# the validation score doesn't improve. We'll use an EarlyStopping callback that tests a
# training condition for every epoch. If a set amount of epochs elapses without showing
# improvement, then automatically stop the training.

model = build_model()

# the patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data,
                          train_labels,
                          epochs=EPOCHS,
                          validation_split = 0.2,
                          verbose=0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()]
                          )

plotter.plot({'Early Stopping': early_history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

# Let's see how well the model generalizes by using the test set, which we did
# not use when training the model. This tells us how well we can expect the model
# to predict when we use it in the real world.
loss, mae, mse = model.evaluate(normed_test_data,
                                test_labels,
                                verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# MAKE PREDICTIONS
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()