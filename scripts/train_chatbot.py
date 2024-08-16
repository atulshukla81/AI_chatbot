import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import pickle

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open('F:/Python Projects/chatbot_project/data/intents.json') as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Loop through each sentence in intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add documents
        documents.append((word_list, intent['tag']))
        # Add to classes if not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print the total classes and words
print(f"Total classes: {len(classes)}, words: {len(words)}")

# Create training data
training = []
output_empty = [0] * len(classes)

# Create training set, bag of words for each sentence
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle the features
random.shuffle(training)

# Convert training data to np.array with dtype=object to avoid shape issues
training = np.array(training, dtype=object)

# Create train and test lists
train_x = np.array([np.array(item[0]) for item in training])
train_y = np.array([np.array(item[1]) for item in training])

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save('F:/Python Projects/chatbot_project/models/chatbot_model.h5')

# Save words and classes
with open('F:/Python Projects/chatbot_project/data/words.pkl', 'wb') as file:
    pickle.dump(words, file)

with open('F:/Python Projects/chatbot_project/data/classes.pkl', 'wb') as file:
    pickle.dump(classes, file)


# import nltk
# from nltk.stem import WordNetLemmatizer
# import json
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import SGD
# import random
# import pickle

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load intents
# intents = json.loads(open('../data/intents.json').read())

# # Initialize lists
# words = []
# classes = []
# documents = []
# ignore_words = ['?', '!', '.', ',']

# # Loop through each sentence in intents patterns
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # Tokenize each word
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         # Add documents
#         documents.append((word_list, intent['tag']))
#         # Add to classes if not already there
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize and lower each word and remove duplicates
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))

# # Sort classes
# classes = sorted(list(set(classes)))

# # Print the total classes and words
# print(f"Total classes: {len(classes)}, words: {len(words)}")

# # Create training data
# training = []
# output_empty = [0] * len(classes)

# # Create training set, bag of words for each sentence
# for doc in documents:
#     bag = []
#     word_patterns = doc[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)
    
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
    
#     training.append([bag, output_row])

# # Shuffle the features and convert to np.array
# random.shuffle(training)
# training = np.array(training)

# # Create train and test lists
# train_x = list(training[:, 0])
# train_y = list(training[:, 1])

# # Create model
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# # Compile model
# sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # Train the model
# model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# # Save model
# model.save('../models/chatbot_model.h5')

# # Save words and classes
# pickle.dump(words, open("../data/words.json", "wb"))
# pickle.dump(classes, open("../data/classes.json", "wb"))
