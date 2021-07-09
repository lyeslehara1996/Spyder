

#declaration des module 

import pandas as pd 
import re 
import nltk
import pickle

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
from keras.models import load_model


from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report

nltk.download('stopwords')
from tensorflow.keras.utils import to_categorical
nltk.download('punkt')

#affichage de data frame 

df=pd.read_excel('SemEval2017A.xlsx')
df.head()

df.drop("Unnamed: 3", axis=1, inplace=True)
df.drop("Unnamed: 4", axis=1, inplace=True)
df.drop("Unnamed: 5", axis=1, inplace=True)
df.drop("Unnamed: 6", axis=1, inplace=True)


df.head()

df.Polarity.value_counts()

df.info()

#supprimer les lignes qui contient des valeur null 


df.Polarity.unique()
df.dropna(subset=['Polarity'], inplace=True)
df.Polarity.unique()
df.info()

plt.figure(figsize=(8,6))
df.Polarity.hist(xlabelsize=14)
plt.show()

#df_clean = df


df['Comments']=df.Comments.str.lower()


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

######decomposition de dataframe #########

reviews =  df[['Comments']]
labels =  df[['Polarity']]
reviews


#######supprimer de ponctuation ########


revue_sans_ponctuation=[]
for sentence in reviews['Comments']:

    revue_sans_ponctuation.append(' '.join(Word.strip(st.punctuation) for Word in sentence.split()))

reviews_cleaned = np.asarray(revue_sans_ponctuation)
reviews_cleaned


###### convertir to array ###########

review_array = np.asarray(reviews['Comments'])
label_array = np.asarray(labels['Polarity'])
reviews_labels = np.stack((review_array, label_array), axis = 1)



list_index=[]

for i, text in enumerate(reviews_labels[:,0:1]):
    if(text == "\n"):
        list_index.append(i)

reviews_labels = np.delete(reviews_labels, list_index, axis=0)
reviews_labels

##### Encoder les polarity  avec le codage ordinal  ###########

encoder = LabelEncoder()
encoder.fit(labels['Polarity'])
encoded_labels = encoder.transform(labels['Polarity'])
encoded_labels = to_categorical(encoded_labels)

encoded_labels

########## Encoder  les Texts #########

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(reviews_cleaned)

reviews = tokenizer.texts_to_sequences(reviews_cleaned)
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

reviews = pad_sequences(reviews, padding='post', maxlen=maxlen)

from sklearn.model_selection import KFold
from keras.models import Sequential,save_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


############ evaluation du model1 avec la technique de K-fold cross validation #########

acc_par_fold = []
loss_par_fold = []
# Define the K-fold Cross Validator
kfold = KFold(n_splits=3, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(reviews, encoded_labels):

  # Define the model architecture
  model = Sequential()
  model.add(Embedding(input_dim=num_words,output_dim=100,input_length=100,trainable=True))
  model.add(LSTM(100,dropout=0.2,return_sequences=True ))
  model.add(Dropout(0.2))
  model.add(LSTM(50,dropout=0.2,return_sequences=True ))
  model.add(Dropout(0.2))
  model.add(LSTM(25,dropout=0.2))

  model.add(Dense(3,activation='softmax'))

  # Compile the model
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(reviews[train], encoded_labels[train],
              batch_size=100,
              epochs=5,
              verbose=0)
  
  # Generate generalization metrics
  scores = model.evaluate(reviews[test], encoded_labels[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_par_fold.append(scores[1] * 100)
  loss_par_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
  
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score par fold')
for i in range(0, len(acc_par_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_par_fold[i]} - Accuracy: {acc_par_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_par_fold)} (+- {np.std(acc_par_fold)})')
print(f'> Loss: {np.mean(loss_par_fold)}')
print('------------------------------------------------------------------------')

######### Sauvegarder le model###########

filepath = './Models/model2.h5'
save_model(model, filepath,save_format='h5')

########## load models ##########

model = load_model ( 'F:/PFE/dossier de travail/Models/model2.h5 ')

plt.figure(figsize=(16,5))
epoch=range(1,len(history.history['accuracy'])+1)
plt.plot(epoch,history.history['accuracy'],'b',label='training', color='b')
plt.legend()
plt.show()




########## models 2 #########


acc_par_fold = []
loss_par_fold = []
# Define the K-fold Cross Validator
kfold = KFold(n_splits=3, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(reviews, encoded_labels):

  # Define the model architecture
  model = Sequential()
  model.add(Embedding(input_dim=num_words,output_dim=100,input_length=100,trainable=True))
  model.add(LSTM(100,dropout=0.2,return_sequences=True ))
  model.add(Dropout(0.2))
  model.add(LSTM(50,dropout=0.2,return_sequences=True ))
  model.add(Dropout(0.2))
  model.add(LSTM(32,dropout=0.2,return_sequences=True ))
  model.add(Dropout(0.2))
  model.add(LSTM(16,dropout=0.2,return_sequences=True ))
  model.add(Dropout(0.2))
  model.add(LSTM(8,dropout=0.2,return_sequences=False ))

  model.add(Dense(3,activation='softmax'))

  # Compile the model
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(reviews[train], encoded_labels[train],
              batch_size=100,
              epochs=5,
              verbose=0)
  
  # Generate generalization metrics
  scores = model.evaluate(reviews[test], encoded_labels[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_par_fold.append(scores[1] * 100)
  loss_par_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1
  
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score par fold')
for i in range(0, len(acc_par_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_par_fold[i]} - Accuracy: {acc_par_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_par_fold)} (+- {np.std(acc_par_fold)})')
print(f'> Loss: {np.mean(loss_par_fold)}')
print('------------------------------------------------------------------------')

######### Sauvegarder le model###########

filepath = './Models/model2.h5'
save_model(model, filepath,save_format='h5')














"""
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
cvscores = []
for train, test in kfold.split(reviews, encoded_labels):
  # create model
  model = Sequential()
  model.add(Embedding(input_dim=num_words,output_dim=100,input_length=100,trainable=True))
  model.add(LSTM(100,dropout=0.1,return_sequences=True ))
  model.add(LSTM(100,dropout=0.1))
  model.add(Dense(1,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(reviews[train], encoded_labels[train], epochs=3, batch_size=10, verbose=0)

	# evaluate the model
  scores = model.evaluate(reviews[test], encoded_labels[test], verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

"""