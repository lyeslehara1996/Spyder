# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:54:27 2021

@author: CBS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 08:46:30 2021

@author: CBS
"""

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
import string as st
SAVEd = False

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report
from keras.utils.np_utils import to_categorical
nltk.download('stopwords')

""" Dataset"""

df=pd.read_excel("SemEval2017.xlsx", engine="openpyxl")

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

df.Comments=df.Comments.str.lower()
df.head() 


###################STOP WORDS################
#STOP WORDS
#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

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

############supprission des caractere spiciaux Dans Commantaire #########


df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'https?:\/\/\S+', ' ', x))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", ' ', x))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'{link}', ' ', x))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'&[a-z]+;', ' ', x))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r"[^a-z]", ' ', x))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'@mention', ' ', x))
df['Comments'] = df['Comments'].apply(lambda x: " ".join(x.lower() for x in str(x).split()  if len(x)>3 ))

#################Supprission des caractere speciaux dans Polarity#################


df['polariy'] = df['polariy'].apply(lambda x: re.sub(r'https?:\/\/\S+><', ' ', str(x)))


reviews =  df[['Comments']]
labels =  df[['polariy']]

revue_sans_ponctuation=[]
for sentence in reviews['Comments']:

    revue_sans_ponctuation.append(' '.join(Word.strip(st.punctuation) for Word in sentence.split()))

reviews_cleaned = np.asarray(revue_sans_ponctuation)
reviews_cleaned

review_array = np.asarray(reviews['Comments'])
label_array = np.asarray(labels['polariy'])

reviews_labels = np.stack((review_array, label_array), axis = 1)


########Encoder les polarity ##############
encoder = LabelEncoder()
encoder.fit(label_array)
encoded_labels = encoder.transform(label_array)
encoded_labels

##### Train and Test
review_train, review_test, label_train, label_test = train_test_split(reviews_cleaned, encoded_labels, test_size=0.20, random_state=42)

print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

#########tokenizer review ###########
maxlen=100
tokenizer =Tokenizer(num_words=40000)
tokenizer.fit_on_texts(review_train)
review_train=tokenizer.texts_to_sequences(review_train)
review_train=pad_sequences(review_train, maxlen=maxlen, truncating='post', padding='post')


review_test=tokenizer.texts_to_sequences(review_test)
review_test=pad_sequences(review_test, maxlen=maxlen, truncating='post', padding='post')


##### Model #####

model = Sequential()
#model.add(Embedding(input_dim=4000,output_dim=50,input_length=100,trainable=True))
model.add(LSTM(units=100, return_sequences=True, input_shape=(maxlen,3)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history=model.fit(review_train,label_train, epochs=100,batch_size=64,validation_data=(review_test,label_test))  