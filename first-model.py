
# Sample taken from
# https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#0

import tensorflow as tf
import numpy as np
from tensorflow import keras

# create the simples possible neural network, with 1 layer and 1 neuron. Input shape to it is just 1 value
model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
)

# loss function measures the guessed answers against the known correct answers and measures how well or how badly it did
# optimizer is used to make another guess. Based on the loss function's result, it will try to minimize the loss
#       sgd => stochastic gradient descent
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# provide the known data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# train the neural network
model.fit(xs, ys, epochs=500)

# use the model
print(model.predict([10.0]))
