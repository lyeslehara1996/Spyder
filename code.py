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

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical
nltk.download('stopwords')
nltk.download('punkt')
""" Dataset"""

df=pd.read_excel("SemEval2017A.xlsx")

df.drop("Unnamed: 3", axis=1, inplace=True)
df.drop("Unnamed: 4", axis=1, inplace=True)
df.drop("Unnamed: 5", axis=1, inplace=True)
df.drop("Unnamed: 6", axis=1, inplace=True)
df.head()


df.info()

#supprimer les lignes qui contient des valeur null 
df.Polarity.unique()
df.dropna(subset=['Polarity'], inplace=True)
df.Polarity.unique()
df.info()


plt.figure(figsize=(8,6))
df.Polarity.hist(xlabelsize=14)
plt.show()

#### transformet les mots en miniscule ######
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
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'https?:\/\/\S+', ' ', str(x)))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", ' ', str(x)))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'{link}', ' ', str(x)))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'&[a-z]+;', ' ', str(x)))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r"[^a-z]", ' ', str(x)))
df['Comments'] = df['Comments'].apply(lambda x: re.sub(r'@mention', ' ', str(x)))
df['Comments'] = df['Comments'].apply(lambda x: " ".join(x.lower() for x in str(x).split()  if len(x)>3 ))



#######deviser en review and labels ######


reviews =  df[['Comments']]
labels =  df[['Polarity']]

corpus= []
for text in reviews['Comments']:
    words= [word.lower() for word in word_tokenize(text)]
    corpus.append(words)

num_words=len(corpus)
print(num_words)

####reviews sans ponctuation #######

revue_sans_ponctuation=[]
for sentence in reviews['Comments']:

    revue_sans_ponctuation.append(' '.join(Word.strip(st.punctuation) for Word in sentence.split()))

reviews_cleaned = np.asarray(revue_sans_ponctuation)
reviews_cleaned



review_array = np.asarray(revue_sans_ponctuation)
label_array = np.asarray(labels['Polarity'])

reviews_labels = np.stack((review_array, label_array), axis = 1)

reviews_labels

########Encoder les polarity ##############
encoder = LabelEncoder()
encoder.fit(label_array)
encoded_labels = encoder.transform(label_array)
encoded_labels = to_categorical(encoded_labels)
encoded_labels

##### Train and Test
review_train, review_test, label_train, label_test = train_test_split(reviews_cleaned, encoded_labels, test_size=0.20, random_state=42)

print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

#########tokenizer review ###########
maxlen=100
tokenizer =Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(review_train)
review_train=tokenizer.texts_to_sequences(review_train)
review_train=pad_sequences(review_train, maxlen=maxlen, truncating='post', padding='post')


review_test=tokenizer.texts_to_sequences(review_test)
review_test=pad_sequences(review_test, maxlen=maxlen, truncating='post', padding='post')

review_train


print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

vocab_size = len(tokenizer.word_index) + 1
vocab_size




#### Model_1 #####
""" Ce model ca marche pas """
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(maxlen,3)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))   
model.add(Dense(3,activation='softmax'))
model.summary()

history=model.fit(review_train, label_train, batch_size=100, epochs=5, verbose=1, validation_split=0.2)
model.evaluate(review_test, label_test, verbose=1)

### Model_2###

""" Ce model ca marche mais l'erreur s'affiche lorsque on ajoute des couche supplementaire
    et le resultat de accuracy ne s'améliore pas (reste stable) meme si on augmente le nombre de neurone et le nombre d'epoch utilise 
    et aussi j'ai pas compris bien  le principe de la couche Embedding 
 """


Model_2 = Sequential()
Model_2.add(Embedding(input_dim=vocab_size,output_dim=100,input_length=100,trainable=True))

Model_2.add(LSTM(100,dropout=0.2,return_sequences=True ))

Model_2.add(LSTM(100,dropout=0.2))
Model_2.add(Dense(3,activation='softmax'))
Model_2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

Model_2.summary()

history=Model_2.fit(review_train, label_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)


plt.figure(figsize=(16,5))
epoch=range(1,len(history.history['accuracy'])+1)
plt.plot(epoch,history.history['loss'],'b',label='training', color='red')
plt.plot(epoch,history.history['val_loss'],'b',label='validation Loss')
plt.legend()
plt.show()

Model_2.evaluate(review_test, label_test, verbose=1)