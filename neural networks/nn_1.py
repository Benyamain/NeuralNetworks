import tensorflow as tf
# Keras: An API for tf that makes things easier for us such as providing boilerplate code
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Each image corresponds with a label (10)
dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize the data to a ratio between 0 and 1
# Labels are between 0 and 9 so normalizing is not necessary
train_images = train_images / 255.0
test_images = test_images / 255.0
# Array of pixels and their rgb values
# print(train_images[7])


# Show an image
# Cmap removes the 'fluff' that plt adds
plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()