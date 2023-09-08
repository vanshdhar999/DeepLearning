# So far we have defined our deep neural network with, some hidden layers and actiavtion
# function. But our model doesnt know what to do. In order to predict some values 
# we need our model to train on the training data set. 

# We need two things in general, am optimizer and a loss function. 

# The loss function will tell us how good the model is in predicting the values. The optimizer is the actual
# algorithm that will drive our model to near perfect predictions by making changes in the weights and biases
# We use an algorithm called Stochastic gradient descent for the optimizer and MAE (Mean Average Loss) or MSE
#(Mean Squared Error). 

#Each iteration's sample of training data is called a minibatch (or often just "batch"), 
#while a complete round of the training data is called an epoch. 
#The number of epochs you train for is how many times the network will see each training example.



# Learning rate and mini batch sizes are two very important features that decide our model's accuracy

#Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning",
# in a sense). Adam is a great general-purpose optimizer.

#Stochastic means "determined by chance." Our training is stochastic because the minibatches are random samples from the dataset.
# And that's why it's called SGD!


#After defining a model, you can add a loss function and optimizer with the model's compile method:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

#Define a model

model  = keras.Sequential([ layers.Dense(units=1, input_shape=[3]) ])

model.compile(optimizer ="adam", loss="mae") 