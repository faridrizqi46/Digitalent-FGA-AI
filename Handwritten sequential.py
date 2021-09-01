import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Import Keras libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from pandas import get_dummies

mnist = tf.keras.datasets.mnist
(train_features, train_labels), (test_features, test_labels) = mnist.load_data()

train_features, test_features = train_features / 255.0, test_features / 255.0 #dinormalisasi

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units = 50, activation="relu")) #buat hidden layer 1 dengan 50 neuron
model.add(Dense(units = 20, activation="relu")) #buat hidden layer 2 dengan 20 neuron
model.add(Dense(units = 10, activation="softmax")) #output layer yg berisi 10 neuron (angka 0 - 9)

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_features,train_labels,epochs=20)
model.evaluate(test_features,test_labels)

#Menguji model dengan image di lokasi 200
loc = 200
test_image = test_features[loc]
test_image = test_image.reshape(1,28,28)

result = model.predict(test_image)

print("")
print(result)
print("")
print(result.argmax())

test_labels[loc]
plt.imshow(test_features[loc])
plt.show()







