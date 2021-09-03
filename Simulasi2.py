# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:12:00 2021

@author: Hazel
"""

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.read_csv('data.csv')
df.head()
plt.scatter(df[df['label'] == 0]['x1'], df[df['label'] == 0]['x2'], marker='*')
plt.scatter(df[df['label'] == 1]['x1'], df[df['label'] == 1]['x2'], marker='<')
plt.show()

x_input = df[['x1','x2']].values
y_label = df[['label']].values
x = tf.Variable(x_input, dtype=tf.float32)
y = tf.Variable(y_label, dtype=tf.float32)
number_features = 2
number_units = 1
learning_rate = 0.1
weight = tf.Variable(tf.zeros([number_features, number_units]))
bias = tf.Variable(tf.zeros([number_units]))
optimizer = tf.optimizers.SGD(learning_rate)

def perceptron(x):
    z = tf.add(tf.matmul(x,weight),bias)
    output = tf.sigmoid(z)
    return output

def train(i):
    for n in range(i):
        loss = lambda:abs(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=perceptron(x))))
        optimizer.minimize(loss, [weight, bias])

train(1000)
tf.print(weight,bias)
ypred = perceptron(x)
ypred = tf.round(ypred)
acc = accuracy_score(y.numpy(), ypred.numpy())
print(acc)

cnf_matrix = confusion_matrix(y.numpy(), ypred.numpy())
print(cnf_matrix)

    
 


