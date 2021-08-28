# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:09:41 2021

@author: Muhammad S.Haris ^l^
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

m = tf.Variable(0.0)
b = tf.Variable(0.0)

def regression(x):
    model = m*x+b
    return model

# Input Data
x=[1.,2.,3.,4.]
y=[0.,-1.,-2.,-3.]

fig=plt.figure()
ax=fig.add_subplot(111)
plt.plot(x,y,'bo-')
#plt.show()
loss = lambda:abs(regression(x)-y)
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
for i in range(1001):
    opt.minimize(loss,[m,b])
    if i %10==0:
        plt.plot(x,regression(x))
        ax.set_xlim(0.9,4.1)
        ax.set_ylim(-3.1,0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        plt.suptitle(f'i = {i}, m = {m.numpy():.3f}, b = {b.numpy():.3f}')
        plt.pause(0.01)
plt.show()
