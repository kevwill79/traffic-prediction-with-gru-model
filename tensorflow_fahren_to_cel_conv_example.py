# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:24:09 2025

@author: kwill
"""

'''
Example how to train a model to convert temperatures from fahrenheit to celsius

Remember: This is the algorithm the model is training to learn 
          f = c * 1.8 + 32

Input:   0,  8, 15, 22,  38
Output: 32, 46, 59, 72, 100
'''

import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

# Only display errors
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

'''
 Set up training data
 
 Features (inputs) 
 labels (outputs)
 Training Examples (input/output used during training)
'''
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

'''
Create a simple model (1 layer and 1 neuron)

input_shape=[1] — This specifies that the input to this layer is a single value.

units=1 — This specifies the number of neurons in the layer. The number of neurons 
defines how many internal variables the layer has to try to learn how to solve 
the problem

Dense - Every neuron in each layer is connected to all the neurons in the previous
layer. These types of layers are fully connected or dense layers. So in keras, a
dense layer is a layer that the neurons in that layer is fully connected to the 
neurons in the previous layer.
'''
# build a layer (layer 0)
#l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# assemble layers into the model
#model = tf.keras.Sequential([l0])

# Usually the layer is defined inside the model definition
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

'''
Compile model, with loss and optimizer functions

loss function - a way of measuring how far off predictions are from the desired outcome. 
(The measure difference is called the loss)
optimizer function - a way of adjusting internal values in order to reduce the loss

0.1 is the learning rate. This is the step size taken when adjusting the values 
in the model. 

If the value is too small, it will take too many iterations to train the model.
If the value is too large, and the accuracy goes down.

Finding a good value often involves some trial and error, bt the range is usually 
within 0.001 (default), and 0.1
'''
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

'''
Train the model by calling the fit method.

During training, the model takes in celsius values, performs a calculation using 
the current internal variables (called weights) and outputs values which are meant
to be fahrenheit equivalent. The weights are initially set randomly, so the output 
will not be close to the correct value. 

fit method
arg 1 - inputs
arg 2 - desired outputs
epochs - how many times this cycle should be run
We're training the model with 3500 examples (7 pairs, over 500 epochs)
verbose - controls how much output the method produces
return - history object (we can use it to plot how the loss of our model goes down 
after each training epoch) a high loss means that the fahrenheit degrees the 
model predicts is far from the corresponding value in fahrenheit_a.
'''
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")


# use matplotlib to visualize
#plt.xlabel('Epoch Number')
#plt.ylabel("Loss Magnitude")
#plt.plot(history.history['loss'])

# now use the model to make predictions
print(model.predict(x=np.array([100.0])))

# let's check out the layer weights (print the internal variables of the dense layer)
#print("These are the layer variables: {}".format(l0.get_weights()))

# the weights represent the 1.8 and the 32 in the celsius to fahrenheit formula
#  f = c * 1.8 + 32
for layer in model.layers:
    print(layer.get_weights())
    
# experiment with more dense layers
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model2 = tf.keras.Sequential([l0, l1, l2])
model2.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model2.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
#print(model.predict(x=np.array([100.0])))
#print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([100.0])))
#print("These are the l0 variables: {}".format(l0.get_weights()))
#print("These are the l1 variables: {}".format(l1.get_weights()))
#print("These are the l2 variables: {}".format(l2.get_weights()))

for layer in model2.layers:
    print(layer.get_weights())