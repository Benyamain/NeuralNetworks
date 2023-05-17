import tensorflow as tf
# Keras: An API for tf that makes things easier for us such as providing boilerplate code
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Flatten the data
# Input layer: (28 x 28) -> 784 neurons
# Output layer: (0 to 9) -> 10 neurons
# Add a hidden layer since (784 x 10) -> 7840 weight(s) and biase(s)
# Add an arbitary amount of neurons for your hidden layer (15 to 20 percent of your input layer neuron size): 128 neurons
# This adds more complexity to the network that can look for more patterns undetected by the less complex input and
# output layers. We do not know what the hidden layer will do but we have hopes for it to make the model far more accurate

# Each image corresponds with a label (10)
dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize the data to a ratio between 0 and 1
# Labels are between 0 and 9 so normalizing is not necessary
train_images = train_images / 255.0
test_images = test_images / 255.0
# Array of pixels and their grayscale values (28 x 28)
# print(train_images[7])

# Dense layers is a fully-connected network: A neuron from one layer is connected to another layer neuron exactly one time
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = 'relu'),
    # Pick values for each output neuron that sum up to 1
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Show an image
# Cmap removes the 'fluff' that plt adds
# plt.imshow(train_images[7], cmap = plt.cm.binary)
# plt.show()