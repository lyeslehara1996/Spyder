

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
nltk.download('punkt')
nltk.download('stopwords')

from keras.utils.np_utils import to_categorical

df=pd.read_excel('/content/SemEval2017.xlsx')
df

"""

netoyage de dataset

"""

df.drop("Unnamed: 3", axis=1, inplace=True)
df.drop("Unnamed: 4", axis=1, inplace=True)
df.drop("Unnamed: 5", axis=1, inplace=True)
df.drop("Unnamed: 6", axis=1, inplace=True)

df.head()

df.Polarity.value_counts()

plt.figure(figsize=(8,6))
df.Polarity.hist(xlabelsize=14)
plt.show()

"""**Prétraitement de dataSet**

transformer les mots en miniscule
"""

df_clean = df
df_clean.Comments=df_clean.Comments.str.lower()
df_clean

"""Suppression de tout les symbole  """

df_clean['Comments'] = df_clean['Comments'].apply(lambda x: re.sub(r'https?:\/\/\S+', ' ', str(x)))
df_clean['Comments'] = df_clean['Comments'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", ' ', str(x)))
df_clean['Comments'] = df_clean['Comments'].apply(lambda x: re.sub(r'{link}', ' ', str(x)))
df_clean['Comments'] = df_clean['Comments'].apply(lambda x: re.sub(r'&[a-z]+;', ' ', str(x)))
df_clean['Comments'] = df_clean['Comments'].apply(lambda x: re.sub(r"[^a-z]", ' ', str(x)))
df_clean['Comments'] = df_clean['Comments'].apply(lambda x: re.sub(r'@mention', ' ', str(x)))
df_clean['Comments'] = df_clean['Comments'].apply(lambda x: " ".join(x.lower() for x in str(x).split()  if len(x)>3 ))
df_clean[['Comments']].head()

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

"""Suppression de stop words

"""

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
df_clean['Comments']=df_clean['Comments'].apply(remove_stopwords)

"""Décomposition de dataset"""

reviews =  df_clean[['Comments']]
labels =  df_clean[['Polarity']]

"""suppression de ponctuation"""

revue_sans_ponctuation=[]
for sentence in reviews['Comments']:

    revue_sans_ponctuation.append(' '.join(Word.strip(st.punctuation) for Word in sentence.split()))

reviews_cleaned = np.asarray(revue_sans_ponctuation)
reviews_cleaned

review_array = np.asarray(reviews)
label_array = np.asarray(labels)

reviews_labels = np.stack((review_array, label_array), axis = 1)
reviews_labels

list_index=[]

for i, text in enumerate(reviews_labels[:,0:1]):
    if(text == "\n"):
        list_index.append(i)

reviews_labels = np.delete(reviews_labels, list_index, axis=0)

"""Encoder les labels multi class"""

"""
Ici est une erreur car dans la case polarity il y a un caracter speciaux "<" et lorsque j'ai essayer  de le supprimer
 avec l'instruction mis en commantaire il va creer une autre valeur polarity vide 'nan' ce qui devient 4 polarity dans
 notre dataset (positive,negative,neutral et nan )
et lorsque on decode les labels on obtient des arrays de 4 val ce qui afficher dans la cellule suivante.
 mais ce resultat est faut on doit avoir seulement 3  

"""

le=LabelEncoder()
#labels['Polarity'] = le.fit_transform( labels['Polarity'].astype(str))
encoded_labels= le.fit_transform( labels['Polarity'])
encoded_labels = to_categorical(encoded_labels)
encoded_labels

le=LabelEncoder()
labels['Polarity'] = le.fit_transform( labels['Polarity'].astype(str))
encoded_labels= le.fit_transform( labels['Polarity'])
encoded_labels = to_categorical(encoded_labels)
encoded_labels

"""Déviser les donnés de test et d'entraînement"""

review_train, review_test, label_train, label_test = train_test_split(reviews_cleaned,encoded_labels ,test_size=0.2, random_state=42)

print(review_train.shape, label_train.shape)
print(review_test.shape, label_test.shape)

"""Encoder les reviews"""

tokenizer = Tokenizer(num_words=4000)
tokenizer.fit_on_texts(review_train)

review_train = tokenizer.texts_to_sequences(review_train)
review_test = tokenizer.texts_to_sequences(review_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

review_train = pad_sequences(review_train, padding='post', maxlen=maxlen)
review_test = pad_sequences(review_test, padding='post', maxlen=maxlen)

review_train.shape
review_train

"""Model 

"""

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(maxlen,3)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

history=model.fit(review_train,label_train, epochs=100,batch_size=64,validation_data=(review_test,label_test))