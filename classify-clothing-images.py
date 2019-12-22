
# Sample taken from
# https://www.tensorflow.org/tutorials/keras/classification

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# help libraries
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version is ", tf.__version__)

# Fashion MNIST contain 60000 gray images used to train the network and 10000 images to evaluate accuracy
# each image is represented as 28 x 28 pixels
fashion_minst = keras.datasets.fashion_mnist

# train_images, train_labels arrays are the training set, the data the model uses to learn
# test_images, test_labels are the arrays the model is tested against
(train_images, train_labels), (test_images, test_labels) = fashion_minst.load_data()

# image mapping class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# before training, we explore the dataset format
print("train_images.shape:")
print(train_images.shape)

# check number of labels in the training set
print("len(train_labels):")
print(len(train_labels))

# each label is an integer 0 to 9
print("train_labels:")
print(train_labels)

# There are 10000 images in the test set, each 28 x 28 pixels
print("test_images.shape:")
print(test_images.shape)

# and the test set contains 10000 image labels
print("len(test_labels):")
print(len(test_labels))

# preprocess the data
# data must be preprocessed before training the network. We inspect first image in the training set, and we'll see
# that the pixel values fall in the range 0 to 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scale the values to a range of 0 and 1 before feeding into the neural network model
# we achieve that by dividing by 255 and this is applied both to training and testing set
train_images = train_images / 255.0
test_images = test_images / 255.0

# we check we have the expected results, by showing the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# BUILD THE MODEL
# building the neural network requires configuring the layers of the model, then compiling the model

# set-up the layers
# layers extract representations from the data fed into them
#   Flatten transforms the format of the images from a two-dimensional array to a one-dimensional array. It's like unstacking rows of pixels and lining them up.
#   First Dense layer contains 128 nodes (=neurons)
#   Second Dense layer contains 10 nodes that returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the
#       probability that the current image belongs to one of the 10 classes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=10)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# make predictions
predictions = model.predict(test_images)
predictions[0]
#   a prediction is an array of 10 numbers. They represent the model's confidence that the image corresponds
#   to each of the 10 different articles of clothing, To se which label has the highest confidence value:
np.argmax(predictions[0])

# and check the value of the label for [0]:
print("test_labels[0]:", test_labels[0])

# graph this to look at the full set of 10 class predictions:
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img=predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


# plot the first X test images, their predicted labels and the true labels.
# color correct predictions in blue and incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# finally, we use the trained model to make a prediction about a single image
img = test_images[1]
print(img.shape)

# add the image to a batch where it's the only member
img = (np.expand_dims(img, 0))
print(img.shape)

# now predict the correct label for this image
predictions_single = model.predict(img)
print("predictions_single: ", predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# grab the predictions for our only image in the batch
pred = np.argmax(predictions_single[0])
print("finally, prediction is ", pred, " which is a ", class_names[pred])