# Jacob Huckleberry
# Tutorial workthough
# 2023-02-24

############################################
# Main imports
import json
import numpy as np
import pickle
import random

# Natural Language Toolkit
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# Tensorflow
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout

############################################

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize words, classes, documents
words = []
classes = []
documents = []

# Initialize ignore words
ignore_symbols = ['?', '!', '.', ',', '-', '\'', '\"', ':', ';']

# Load json file
intents = json.loads(open('/Users/jacobhuckleberry/Desktop/Code/Chatbots/Alvin/intents.json').read())

# Loop through each sentence in the intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# Lemmatize words into words list
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]
# Remove duplicates from words list
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
####################################################

#### Machine learning ####
# Preprocess: initialize training data
training = []
output_empty = [0] * len(classes)

"""
converting document data into 1 or 0
and appending this into the training list 
"""
for document in documents:

    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle training data
random.shuffle(training)

# convert training data to numpy array
training = np.array(training)

# train 1st dimension and 2nd dimension of np array
train_x = list(training[:, 0])
train_y = list(training[:, 1])


#### build neural network ####
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# define SDG
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('alvin_chatbot.h5')
print('Done')