# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 19:11:02 2021

@author: CBS
"""

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
import string 

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


SAVE_FILE = False


df=pd.read_excel('4a-english/4A-English/SemEval2017.xlsx')
df.head()


df.drop("Unnamed: 3", axis=1, inplace=True)
df.drop("Unnamed: 4", axis=1, inplace=True)
df.drop("Unnamed: 5", axis=1, inplace=True)
df.drop("Unnamed: 6", axis=1, inplace=True)
df.head()


df.Polarity.value_counts()
df.shape

plt.figure(figsize=(8,6))
df.Polarity.hist(xlabelsize=14)
plt.show()

df_clean = df
df_clean.Comments=df_clean.Comments.str.lower()
df_clean.head()


df_clean.Comments = df_clean.Comments.apply(lambda x: re.sub(r'https?:\/\/\S+', ' ', str(x)))
df_clean.Comments = df_clean.Comments.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", ' ', str(x)))
df_clean.Comments = df_clean.Comments.apply(lambda x: re.sub(r'{link}', ' ', str(x)))
df_clean.Comments = df_clean.Comments.apply(lambda x: re.sub(r'&[a-z]+;', ' ', str(x)))
df_clean.Comments = df_clean.Comments.apply(lambda x: re.sub(r"[^a-z]", ' ', str(x)))
df_clean.Comments = df_clean.Comments.apply(lambda x: re.sub(r'@mention', ' ', x))
df_clean.Comments = df_clean.Comments.apply(lambda x: " ".join(x.lower() for x in str(x).split()  if len(x)>3 ))
df_clean

df_clean.Polarity = df_clean.Polarity.apply(lambda x: re.sub(r'https?:\/\/\S+><', ' ', str(x))) 
  
df.Polarity.value_counts()

df_clean['Polarity'].replace('', np.nan, inplace=True)

df_clean.dropna(subset=['Polarity'], inplace=True)

df_clean.Polarity.value_counts()

reviews =  df_clean[['Comments']]
labels =  df_clean[['Polarity']]
reviews

encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
# encoded_labels = to_categorical(encoded_labels)

encoded_labels

#convertir au chiffre pour avoir implementer le model 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_word =40000
maxlen=128
tokenizer = Tokenizer(num_words=40000, split=' ')
tokenizer.fit_on_texts(reviews['Comments'])
X = tokenizer.texts_to_sequences(reviews['Comments'])
X = pad_sequences(X,padding='post', maxlen=maxlen)
X[:2]


X_train, X_test, Y_train, Y_test =  train_test_split(X, encoded_labels, test_size=0.20, random_state=42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()

model.add(Embedding(input_dim=max_word,output_dim=100,input_length=128,trainable=True))
model.add(LSTM(100,dropout=0.4,return_sequences=True ))
model.add(LSTM(100,dropout=0.4))
model.add(Dense(1,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

history=model.fit(X_train,Y_train,batch_size=128, epochs=3, verbose=1,validation_data=(X_test,Y_test))