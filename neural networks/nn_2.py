import tensorflow as tf
from tensorflow import keras
import numpy as np

dataset = keras.datasets.imdb
# Only take words that are the 10,000 most frequent
(train_data, train_labels), (test_data, test_labels) = dataset.load_data(num_words = 88000)
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

# 'Have a great day' : 'Have a good day'
# [0, 1, 2, 3] : [0, 1, 4, 3]
# As an example, these phrases in our dataset are integer encoded in a list
# Although 2 and 4 are different, the phrases still hold similar meanings. Thus, embedding layer is introduced.
# Embedding layer groups words in a similar manner to dictate which ones are similar to each other
# Generates word vectors (16D) for each word onto a space (all of the words are drawn on this space)
# Find the difference in the word vectors for each word based on their shared angle ; the greater the angle, the greater
# the difference. However, we want to associate them more closely based on context: The words that surround word you are
# looking at.
# Word embeddings become learned (get 'great' and 'good' closer together [smaller angle])
# When fed into the network, the output for those words would be similar
# N-dimensions represents the number of coefficients to each vector (Ax + By + Cz + ... + ...)
# However this is a lot of dimensions. Therefore, we should scale it down by using GlobalAveragePooling.
# Average layer: Takes whatever n-dimension our dataset is in and puts it in a lower dimension
# Dense layer: Add an arbitary amount of neurons for your hidden layer (15 to 20 percent of your input layer neuron size): 16 neurons
# The dense layer will look at these patterns of words and attempt to classify them as either positive or negative movie reviews
# made possible by modifying the weight(s) and biase(s).
# Output layer: A single output (0 or 1)
# Dense layers is a fully-connected network: A neuron from one layer is connected to another layer neuron exactly one time
'''
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
# Normalizes the data between values 0 and 1
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()
# Two options for the neuron: 0 or 1
# Recall that loss calculates the difference
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
x_validation_data = train_data[:10000]
x_train = train_data[10000:]
y_validation_data = train_labels[:10000]
y_train = train_labels[10000:]

# Batch size: How many reviews are we loading at once
fit_model = model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_validation_data, y_validation_data), verbose = 1)
results = model.evaluate(test_data, test_labels)
print('Results: ', results)

# Extension for a saved model in keras and tensorflow
# Save it in binary. Doing so would save time to keep retraining the model
model.save('./neural networks/model.h5') '''

def review_encode(s):
    # Setting a start tag
    encoded = [1]
    
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            # Unknown tag
            encoded.append(2)
        
    return encoded

model = keras.models.load_model('./neural networks/model.h5')
with open('./neural networks/review.txt', encoding = 'utf-8') as f:
    # Can read multiple lines (or reviews in this case) but for right now we only have one
    for line in f.readlines():
        # Strip the \n
        new_line = line.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace('\"', '').strip().split(' ')
        encode = review_encode(new_line)
        # Expecting a list of lists
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index['<PAD>'], padding = 'post', maxlen = 256)
        predict = model.predict(encode)
        print('Line: ', line, '\n\nEncoded values: ', encode, '\n\nPrediction: ', predict[0])

# test_review = test_data[0]
# model.predict() does not accept normal lists so use np.array()
# prediction = model.predict(np.array([test_review]))
# print('Review: ', decode_review(test_review))
# print('\nPrediction: ', str(prediction[0]))
# print('\nActual: ', str(test_labels[0]))
# print('\nResults: ', results)