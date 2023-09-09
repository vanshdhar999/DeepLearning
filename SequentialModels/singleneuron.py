#We will make a linear unit model

from tensorflow import keras

from tensorflow.keras import layers


import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',titleweight='bold', titlesize=18, titlepad=10)

import pandas as pd

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')

red_wine.head()

#We can get the shape of data set from

red_wine.shape

# Since we need to guess the quality of the wine we exclude the quality 
# fromn the features and define the input_shape on the rest of the features

input_shape = [11]


# Now define the model

model = keras.Sequential([layers.Dense(units=1, input_shape=input_shape)])

#Now we define the weights for the neuron

#By default keras stores the weights in the form of tensors. 

# We can get the weights by printing the weights


# We can use object.attribute to get the weights of the data set

w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w,b))

#The weights are randomly initialized and the bias is set to zero




