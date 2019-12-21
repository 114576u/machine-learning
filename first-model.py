

import tensorflow as tf
import numpy as np
from tensorflow import keras

# create the simples possible neural network, with 1 layer and 1 neuron. Input shape to it is just 1 value
model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
)

