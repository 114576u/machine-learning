

import tensorflow as tf
import numpy as np
from tensorflow import keras

# create the simples possible neural network, with 1 layer and 1 neuron. Input shape to it is just 1 value
model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
)

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# train the neural network
model.fit(xs, ys, epochs=500)

# use the model
print(model.predict([10.0]))
