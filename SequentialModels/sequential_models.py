# building sequential models 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras import layers 
import pandas as pd



plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)


model = keras.Sequential([
        # Insert the hidden layers, with activation function ReLu. 
        layers.Dense(units = 4, activation = 'relu', input_shape = 2),
        layers.Dense(units = 3, activation ='relu'),
        # Insert the output layer 
        layers.Dense(units = 1),
        ])
# Store the data set 
# And print some of the rows using .head()
concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
concrete.head()

#Input Shape 

input_shape = [8] # Except the target column all other features will be used as inputs.

#Define the hidden layers and the output layer. Each hidden layer should have 512 units with activation 
# Function as relu. Also define the output layer with unit defined as 1. 

model = keras.Sequential([
    layers.Dense(units = 512, activation = 'relu', input_shape = input_shape), 
    layers.Dense(units = 512,activation = 'relu'), 
    layers.Dense(units = 512,activation = 'relu'), 
    
    layers.Dense(units = 1)
])

# We can sometimes define the activation layers separately. 

model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])

