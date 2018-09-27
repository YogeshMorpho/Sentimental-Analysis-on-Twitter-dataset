import numpy
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.optimizers import adam
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



data=pd.read_csv('Sentiment.csv')

max_words=2000
tokenizer=Tokenizer(num_words=max_words,split=' ')
tokenizer.fit_on_texts(data['text'].values)
X_train=tokenizer.texts_to_sequences(data['text'].values)
X_train=sequence.pad_sequences(X_train)

Y_train=pd.get_dummies(data['sentiment']).values

model=Sequential()
model.add(Embedding(2000,32,input_length=29))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(3))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=5,batch_size=64,validation_split=0.3)
