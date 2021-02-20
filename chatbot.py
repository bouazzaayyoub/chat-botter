
import nltk
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import wordnet
import fr_core_news_md,en_core_web_sm
import re,json,random
from urllib.request import urlopen

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# nltk.download('punkt')
# nltk.download('wordnet')

"""# Useful functions

### remove duplication
"""

def remove_duplication(word):
  """
    if the input word is similar to an english word return the input word 
    else remove duplications and search again for similar english words
  """
  repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
  repl = r'\1\2\3'
  if wordnet.synsets(word):
    return word
  repl_word = repeat_regexp.sub(repl, word)
  if repl_word != word:
    return remove_duplication(repl_word)
  else:
    return repl_word
remove_duplication("hello"),remove_duplication("blaaaaaabla")

"""### Lemmetization & Stemming"""

# Les fonctions pour séparer les mots et les transformer  vers leurs origin gramatical 
spacy_fr=fr_core_news_md.load()
spacy_en = en_core_web_sm.load()
# Convertir les francais mots vers leurs origin
fr_lemmatizer = lambda w:spacy_fr(w)[0].lemma_
# Convertir les mots anglais vers leurs origin
eng_lemmatizer = lambda w:spacy_en(w)[0].lemma_
# Convertir les mots arabe vers leurs origin
ar_lemmatizer = ISRIStemmer().stem
lemmatizer = lambda word: ar_lemmatizer(fr_lemmatizer(eng_lemmatizer(remove_duplication(word))))

lemmatizer("السلااام"),lemmatizer("donne"),lemmatizer("yeeux")

"""# Importing dataset"""

data_file = urlopen('https://raw.githubusercontent.com/DadiAnas/AI-Chatbot-FlaskServer/master/datasets/intents.json').read() #dataset_import
intents = json.loads(data_file) #dataset_JsonParser

"""#Preparing Dataset

## Oranize data in lists
"""

#les variables utilisés
words=[] #words
classes = [] #tag
documents = [] # (pattern,tag)
ignore_words = ['?', '!',';','.',','] #words to ignore

#mettre les mot dans words
#catégorisation des patterne selon tag 
#ajouter tag dans la list classes 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))
        # adding classes to our class list
        classes.append(intent['tag'])

"""## Lemmetazering & steeming words list"""

#changer les mot vers leurs origine et ignorer "?,!
words = [lemmatizer(w.lower()) for w in words if w not in ignore_words]

"""## remove duplication & sort"""

#trier (pour le training) et éviter la redondance
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print (len(documents), "documents",documents)

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

"""### Make Training dataset"""

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
      bag.append(1 if w in pattern_words else 0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
X = list(training[:,0])
y = list(training[:,1])
print("Training data created")
print(X[0])
print(y[0])

len(X)

"""## Split Dataset"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
len(X_train),len(X_test)

"""#Build ANN model

### Initialize the model
"""

model = Sequential()

"""### Add input layer 128 neurons, relu activation | Adding Droupout to avoid overfitting"""

model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))#avoid overfitting

"""### Add hidden layer 64 neurons, relu activation | Adding Droupout to avoid overfitting"""

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

"""### Add output layer number of neurons equal to number of intents, softmax activation """

model.add(Dense(len(y_train[0]), activation='softmax'))

"""### Compile model. Stochastic gradient descent with Nesterov to accelerated gradient """

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

"""Build the model"""

model.build()

model.summary()

"""#Training model"""

#fitting the model
model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=10,  verbose=1)

"""# Evaluate the model"""

def find_best_threshold():
  ERROR_THRESHOLD = 0.001
  accuracy,step =0,0
  while step < 1:
    y_pred = model.predict(X_test) > step
    if metrics.accuracy_score(y_test,y_pred) > accuracy:
      accuracy = metrics.accuracy_score(y_test,y_pred)
      ERROR_THRESHOLD = step
    step += 0.01
  return ERROR_THRESHOLD
ERROR_THRESHOLD = find_best_threshold()
print(ERROR_THRESHOLD)

y_pred = model.predict(X_test) > ERROR_THRESHOLD
metrics.accuracy_score(y_test,y_pred)

"""# Use the model"""

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer(word.lower()) for word in sentence_words]
   
    return sentence_words

def bow(sentence, words, show_details=True):
    """
    return bag of words array: 0 or 1 for each word in the bag that match 60% another word in the sentence
    """
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array([np.array(bag)]))

def predict_class(sentence, model):
    """
      looking for the class of the sentence
    """
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(p)[0]
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    """
      search for response in the predicted class 
    """
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = (random.choice(i['responses']),tag)
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

resp = chatbot_response("slaaam")[0]
resp

def make_conversation():
  resp = ('','')
  while resp[1] != 'good_bye':
    user_msg = str(input('Me:'))
    resp = chatbot_response(user_msg)
    print(f'chatbot:{resp[0]}')
make_conversation()