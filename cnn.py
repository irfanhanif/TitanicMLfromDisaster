# import tflearn
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam

ftr = np.load("npy/ftr.npy")
cls = np.load("npy/cls.npy")

test_df = pd.read_csv("data/test.csv", delimiter=',')

cls_mod = []
for i in cls:
    if i == 0:
        cls_mod.append(np.array([1,0]))
    else:
        cls_mod.append(np.array([0,1]))
cls_mod = np.array(cls_mod)

shuffle(ftr, cls)
X, X_test, Y, Y_test = train_test_split(ftr, cls_mod, test_size=0.20)

X = np.expand_dims(X, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# embedding_size = 50
# max_word = 1
# max_feature = max_word * embedding_size
model= Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu',input_shape=(len(ftr[0]), 1)))

model.add(Flatten())
model.add(Dropout(0.3))

model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(X, Y,
          batch_size=32,
          shuffle=True,
          epochs=2000,
          validation_data=(X_test, Y_test))

ftr_test = np.load("npy/ftr_test.npy")
ftr_test = np.expand_dims(ftr_test, axis=2)
predicted_test = model.predict(ftr_test)
predicted_test = map(lambda x: x.argmax(axis=0), predicted_test)

test_id = test_df["PassengerId"].values
result = {"PassengerId": test_id, "Survived": predicted_test}
result = pd.DataFrame(data=result)
result.to_csv("submission.csv", sep=',', index=False)