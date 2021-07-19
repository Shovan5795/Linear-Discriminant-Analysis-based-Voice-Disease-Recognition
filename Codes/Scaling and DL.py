# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:38:08 2021

@author: shovon5795
"""

import pandas as pd
dataset = pd.read_csv(r"E:\Research\SAM\Dataset\final_version.csv")

X = dataset.iloc[0:368,0:2770]
y = dataset.iloc[:,-1]


#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_new = scaler.fit_transform(X)

'''
#MinmaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_new = scaler.fit_transform(X)


#Plot function
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Vs Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Vs Validation loss')
    plt.legend()
'''

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=None)
X_PCA = lda.fit_transform(X_new, y)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_PCA2= pca.fit_transform(X_new)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPool1D


#model.add(Conv1D(filters = 40, kernel_size = 1, activation = 'relu', padding = 'same', input_shape = (3,1)))
#model.add(Conv1D(filters = 20, kernel_size = 1, activation = 'relu', padding = 'same'))
#model.add(Conv1D(filters = 10, kernel_size = 1, activation = 'relu', padding = 'same'))
    

def create_model():
    model = Sequential()
    model.add(Dense(256, activation = 'relu', input_shape = (367,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer= Adam(lr=0.0001),metrics=['accuracy'])
    print(model.summary())
    return model

history = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 10)

from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, cross_val_score
kf = KFold(n_splits = 10)
skf = StratifiedKFold(n_splits = 10)
shs = ShuffleSplit(n_splits = 10)
sshs = StratifiedShuffleSplit(n_splits = 10)
import datetime
start_time = datetime.datetime.now()
acc1 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=kf, n_jobs=1)
acc2 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=skf, n_jobs=1)
acc3 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=shs, n_jobs=1)
acc4 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=sshs, n_jobs=1)
print(acc1.mean())
print(acc2.mean())
print(acc3.mean())
print(acc4.mean())
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = (time_diff.total_seconds() * 1000)/4




















