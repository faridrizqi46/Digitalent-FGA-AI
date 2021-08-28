# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:43:37 2021

@author: Muhammad S.Haris ^l^
"""

import pandas as pd
import sklearn as sl
import tensorflow as tf
 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from pandas import get_dummies

df = pd.read_csv('iris.csv')
df.head()
#plt.scatter(df[df['species'] == 0]['sepallength'], df[df['species'] == 0]['sepalwidth'], marker='*')
#plt.scatter(df[df['species'] == 1]['sepallength'], df[df['species'] == 1]['sepalwidth'], marker='<')
#plt.scatter(df[df['species'] == 2]['sepallength'], df[df['species'] == 2]['sepalwidth'], marker='o')
#plt.show()
x = df[['petallength', 'petalwidth', 'sepallength', 'sepalwidth']].values
y = df['species'].values
y = get_dummies(y)
y = y.values
x = tf.Variable(x, dtype=tf.float32)
Number_of_features = 4
Number_of_units = 3
weight = tf.Variable(tf.zeros([Number_of_features, Number_of_units]))  
bias = tf.Variable(tf.zeros([Number_of_units]))

def perceptron(x):
    z = tf.add(tf.matmul(x, weight), bias)
    output = tf.nn.softmax(z)
    return output

optimizer = tf.keras.optimizers.Adam(.01)
def train(i):
    for n in range(i):
        loss=lambda: abs(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=perceptron(x))))
        optimizer.minimize(loss, [weight, bias])

train(1000)

tf.print(weight)

ypred = perceptron(x)
ypred = tf.round(ypred)
print(accuracy_score(y, ypred))
