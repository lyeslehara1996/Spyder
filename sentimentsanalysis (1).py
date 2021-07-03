

#declaration des module 

import pandas as pd 
import re 
import nltk

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout

from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
import string as st

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report

nltk.download('stopwords')
from tensorflow.keras.utils import to_categorical
nltk.download('punkt')

SAVE_FILE = False

#affichage de data frame 

df=pd.read_excel('/content/SemEval2017.xlsx')
df.head()

df.drop("Unnamed: 3", axis=1, inplace=True)
df.drop("Unnamed: 4", axis=1, inplace=True)
df.drop("Unnamed: 5", axis=1, inplace=True)
df.drop("Unnamed: 6", axis=1, inplace=True)
df.drop("Unnamed: 7", axis=1, inplace=True)
df.drop("Unnamed: 8", axis=1, inplace=True)
df.drop("Unnamed: 9", axis=1, inplace=True)
df.drop("Unnamed: 10", axis=1, inplace=True)
df.drop("Unnamed: 11", axis=1, inplace=True)
df.drop("Unnamed: 12", axis=1, inplace=True)
df.drop("Unnamed: 13", axis=1, inplace=True)
df.drop("Unnamed: 14", axis=1, inplace=True)
df.drop("Unnamed: 15", axis=1, inplace=True)

df.head()

df.polariy.value_counts()

df.info()

#supprimer les lignes qui contient des valeur null 
df.polariy.unique()
df.dropna(subset=['polariy'], inplace=True)
df.polariy.unique()
df.info()

plt.figure(figsize=(8,6))
df.polariy.hist(xlabelsize=14)
plt.show()

#df_clean = df
df['Comments']=df.Comments.str.lower()
df

"""**Prétraitement**

Suppression des symbole
"""

df.Comments = df.Comments.apply(lambda x: re.sub(r'https?:\/\/\S+', ' ', str(x)))
df.Comments = df.Comments.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", ' ', str(x)))
df.Comments = df.Comments.apply(lambda x: re.sub(r'{link}', ' ', str(x)))
df.Comments = df.Comments.apply(lambda x: re.sub(r'&[a-z]+;', ' ', str(x)))
df.Comments = df.Comments.apply(lambda x: re.sub(r"[^a-z]", ' ', str(x)))
df.Comments = df.Comments.apply(lambda x: re.sub(r'@mention', ' ', x))
df.Comments = df.Comments.apply(lambda x: " ".join(x.lower() for x in str(x).split()  if len(x)>3 ))
df

"""Suppression des stop word"""

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
df['Comments']=df['Comments'].apply(remove_stopwords)

corpus= []
for text in df['Comments']:
    words= [word.lower() for word in word_tokenize(text)]
    
    corpus.append(words)

num_words=len(corpus)
print(num_words)

df.info()

reviews =  df[['Comments']]
labels =  df[['polariy']]
reviews

revue_sans_ponctuation=[]
for sentence in reviews['Comments']:

    revue_sans_ponctuation.append(' '.join(Word.strip(st.punctuation) for Word in sentence.split()))

reviews_cleaned = np.asarray(revue_sans_ponctuation)
reviews_cleaned

review_array = np.asarray(reviews['Comments'])
label_array = np.asarray(labels['polariy'])
reviews_labels = np.stack((review_array, label_array), axis = 1)

list_index=[]

for i, text in enumerate(reviews_labels[:,0:1]):
    if(text == "\n"):
        list_index.append(i)

reviews_labels = np.delete(reviews_labels, list_index, axis=0)
reviews_labels

##Encoder les polarity 

encoder = LabelEncoder()
encoder.fit(labels['polariy'])
encoded_labels = encoder.transform(labels['polariy'])
encoded_labels = to_categorical(encoded_labels)

encoded_labels

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews_cleaned)

reviews = tokenizer.texts_to_sequences(reviews_cleaned)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

reviews = pad_sequences(reviews, padding='post', maxlen=maxlen)

from sklearn.model_selection import train_test_split,KFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []
for train, test in kfold.split(reviews, encoded_labels):
  # create model
  model = Sequential()
  model.add(Embedding(input_dim=num_words,output_dim=100,input_length=100,trainable=True))
  model.add(LSTM(100,dropout=0.1,return_sequences=True ))
  model.add(LSTM(100,dropout=0.1))
  model.add(Dense(1,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(reviews[train], encoded_labels[train], epochs=150, batch_size=10, verbose=0)

	# evaluate the model
  scores = model.evaluate(reviews[test], encoded_labels[test], verbose=0)
  
  
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))