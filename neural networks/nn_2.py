import tensorflow as tf
from tensorflow import keras
import numpy as np

dataset = keras.datasets.imdb
# Only take words that are the 10,000 most frequent
(train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words = 10000)
# print(train_data[0])

# Finding the mappings for the integer values
word_index = dataset.get_word_index()
# Key: Word
# Value: Integer
word_index = {key: (value + 3) for key, value in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] =  1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Swap all the keys for the values
# Integer points to a word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Redefining our dataset into a form that our model can tolerate/accept
# Movie reviews can vary in their length ...
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value = word_index['<PAD>'], padding = 'post', maxlen = 256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value = word_index['<PAD>'], padding = 'post', maxlen = 256)
print(len(train_data), len(test_data))

# Return all of the keys that we want (readable to us)
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# print(decode_review(test_data[0]))
# Can add the layers into sequential as a list but doing so through add achieves the same goal
# Final output of this model: 'Positive' or 'Negative' movie review (between 0 and 1)
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
# Normalizes the data between values 0 and 1
model.add(keras.layers.Dense(1, activation = 'sigmoid'))