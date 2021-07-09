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

from keras import backend as K
from keras import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


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


######## exporter les dataframe #######

reviews.to_excel("F:/PFE/dossier de travail/data/reviews.xlsx", index=None)
labels.to_excel("F:/PFE/dossier de travail/data/labels.xlsx", index=None)



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

########Encoder les polarity avec One-Hot encoding  ##############


encoder = LabelEncoder()
encoder.fit(label_array)
encoded_labels = encoder.transform(label_array)
encoded_labels = to_categorical(encoded_labels)
encoded_labels

##### Train and Test######


review_train, review_test, label_train, label_test = train_test_split(reviews_cleaned, encoded_labels, test_size=0.20, random_state=42)

print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

#########tokenizer review ###########



tokenizer = Tokenizer(num_words=4000)
tokenizer.fit_on_texts(review_train)

review_train = tokenizer.texts_to_sequences(review_train)
review_test = tokenizer.texts_to_sequences(review_test)

vocab_size = len(tokenizer.word_index) + 1
vocab_size
maxlen = 100

review_train = pad_sequences(review_train, padding='post', maxlen=maxlen)
review_test = pad_sequences(review_test, padding='post', maxlen=maxlen)



print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

label_train[0]


########### Create Validation Set ################

review_train_Emb, review_Val_Emb, label_train_Emb, label_Val_Emb = train_test_split(review_train, label_train, test_size=0.1, random_state=42)

print(review_train_Emb.shape, label_train_Emb.shape)
print(review_Val_Emb.shape, label_Val_Emb.shape)



#######################################



######## enregistrer des donnees d'entrainement  et les donnees de test###########

"""  
np.review_train.to_excel("F:/PFE/dossier de travail/train/review_train.xlsx")
review_test.to_excel("F:/PFE/dossier de travail/test/review_test.xlsx")
label_train.to_excel("F:/PFE/dossier de travail/train/label_train.xlsx")
label_test.to_excel("F:/PFE/dossier de travail/test/label_test.xlsx")


"""




#### Model_1 #####

""" Ce model ca marche pas """

model = Sequential()

model.add(Embedding(input_dim=vocab_size,output_dim=100,input_length=100,trainable=True))

model.add(LSTM(units=100, return_sequences=True, input_shape=(maxlen,3)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=25, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=15, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=5, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.summary()

history=model.fit(review_train_Emb, label_train_Emb, batch_size=128, epochs=5, verbose=1, validation_data=(review_Val_Emb,label_Val_Emb))
model.evaluate(review_test, label_test, verbose=1)






### Model_2###

model_2 = Sequential()

model_2.add(Embedding(input_dim=vocab_size,output_dim=100,input_length=100,trainable=True))

model_2.add(LSTM(units=200, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=150, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=100, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=50, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=32, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=16, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=8, return_sequences=True))
model_2.add(Dropout(0.2))

model_2.add(LSTM(units=4, return_sequences=False))


model_2.add(Dense(3,activation='softmax'))

model_2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model_2.summary()

history=model_2.fit(review_train, label_train, batch_size=10, epochs=5, verbose=2, validation_split=0.2)



score,acc=model.evaluate(review_test, label_test, verbose=1)


print("score: %.2f" % (score))
print("acc: %.2f" % (acc))






plt.figure(figsize=(16,5))
epoch=range(1,len(history.history['accuracy'])+1)
plt.plot(epoch,history.history['loss'],'b',label='training', color='red')
plt.plot(epoch,history.history['val_loss'],'b',label='validation Loss')
plt.legend()
plt.show()
