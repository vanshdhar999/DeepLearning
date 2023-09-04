from tensorflow import keras 
#How to create a linear unit

from tensorflow.keras.layers import Dense

from tensorflow.keras import layers

# Create a network with 1 linear unit

model  = keras.Sequential([ layers.Dense(units=1, input_shape=[3]) ])

#units is the number of outputs we want and input_shape is the number of inputs

