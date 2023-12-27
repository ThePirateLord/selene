# Run this cell so you do not see GPU availibility errors from tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Import the Required Libraries
import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Load the Data
nltk.download('omw-1.4')
with open(r'intents.json') as data:
    intents = json.loads(data.read())

# Tokenization
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        classes.append(intent['tag'])
        documents.append((w, intent['tag']))
        
# Lemmatization
lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create Data for Training
training = []
output_empty = [0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(words.lower()) for words in pattern_words] 
    
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])]=1
    training.append([bag, output_row])

random.shuffle(training)
a = int(0.7*len(training))
training = np.array(training, dtype='object')
x_train = list(training[:a, 0])
y_train = list(training[:a, 1])
x_val = list(training[a:,0])
y_val = list(training[a:,1])

# Design the Model
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Train and Save the Model¶
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
             optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, 
                 batch_size=5, validation_data=(x_val, y_val), verbose=1)
model.save('trained_model_selene.h5', hist)

# Print the training curves
plt.rcParams["figure.figsize"] = (12,8)
N = np.arange(0,200)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, hist.history["loss"], label="train_loss")
plt.plot(N, hist.history["val_loss"], label="val_loss")
plt.plot(N, hist.history['accuracy'], label="accuracy")
plt.plost(N, hist.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# Predict the Response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w==s:
                bag[i]=1
                if show_details:
                    print("Found in bag: %s" %w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result


