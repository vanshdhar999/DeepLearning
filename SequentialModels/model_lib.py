import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import matplotlib 
import pandas as pd

#made a library for defining models. 

def single_neuron_model(units, input):

    #define a model with user inputs.
    training_model = keras.Sequential([layers.Dense(units= units, activation = "relu", input_shape = [input]), 
                                       layers.Dense(1)
                                                    
                                       ])

    return training_model


def wider_model(units, input):

    training_model  = keras.Sequential([layers.Dense(units = units, activation = "relu", input_shape = [input]), 
                                        layers.Dense(1)])
    return training_model


def deeper_model(units, input):

        training_model  = deeper = keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1),
                        ])
        
        return training_model
