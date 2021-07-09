# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:56:19 2021

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
import string as st
SAVEd = False

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

review_train=pd.read_excel("train/review_train.xlsx")
label_train=pd.read_excel("train/label_train.xlsx")
review_test=pd.read_excel("test/review_test.xlsx")
label_test=pd.read_excel("F:/PFE/dossier de travail/test/label_test.xlsx")

review_train.drop("Unnamed: 0", axis=1, inplace=True)
review_test.drop("Unnamed: 0", axis=1, inplace=True)
label_train.drop("Unnamed: 0", axis=1, inplace=True)
label_test.drop("Unnamed: 0", axis=1, inplace=True)

print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

review_train=np.array(review_train)
review_test=np.array(review_test)
label_train=np.array(label_train)
label_test=np.array(label_test)

label_train
#Model_2


model = Sequential()
model.add(Embedding( 22814,100,input_length=100))

model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(16, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(8, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(4, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(LSTM(16, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(8, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(4, return_sequences= False))

model.add(Dense(3,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history=model.fit(review_train, label_train, epochs=5,  validation_split=0.2)

