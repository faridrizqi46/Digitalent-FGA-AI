# -*- coding: utf-8 -*-
"""Training Keras Sequential.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16gF6Ky_QkgBOFYi0djVjWQuenLcIYtQZ
"""

import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('data.csv')
df.head()

x_input = df[['x1','x2']].values
y_label = df[['label']].values

model = Sequential()
model.add(Dense(units=1, input_dim=2, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_input, y_label, epochs=1000)

model.evaluate(x_input, y_label)