#File for training the ChatBox

import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk  #natural language toolkit
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer  #work working works worked all these words will be reduced to the same root word using this moduleeee

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:   #we deal with intents as a dictionary now
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)   #tokenize basically converts the sentences into seperate words
        words.extend(word_list)     #we just need to append the content and not the whole list
        documents.append((word_list,intent['tag']))   #to bascially keep track of which tag does the current word_list belong to
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]    #removing the ignored letters
words = sorted(set(words))      #elimintaing the repeated words/ duplicates
classes = sorted(set(classes))

#saving the words and classes for later use
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#using bag of words technique
#since we cant use the words itself we will sign them the values 0 or 1 based on whether that word is occuring in the dataset or not
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:    #if the word occurs append 1 otherwise 0
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


#Machine Learning Part - converting into a nump array
random.shuffle(training)
training = np.array(training)

#training data
train_x = list(training[:,0])
train_y = list(training[:,1])

#creating the neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]),activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model_save = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',model_save)
print('Done')