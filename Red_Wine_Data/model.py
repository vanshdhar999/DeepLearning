# We will use our sequential model to predict the qualtiy of wine, with certain features
# given in the data set red-wine.csv 

#This dataset consists of physiochemical measurements from about 1600 Portuguese red wines.

#Also included is a quality rating for each wine from blind taste-tests.

import pandas as pd
from IPython.display import display 
from tensorflow import keras
from tensorflow.keras import layers

# read the data from the csv file and store it in red_wine
red_wine = pd.read_csv('./red-wine.csv')

#Creating training and validation splits 

df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1], works much better for some reason we will learn about. 

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

# Determine how many inputs should we have for our model. excluding the target. 
print(X_train.shape)

#We will define our model with 3 hidden layers and over 1500 neurons. This should be sufficient for our purpose.

model = keras.Sequential([
    layers.Dense(units = 512, activation ='relu', input_shape= [11]),
    layers.Dense(units = 512, activation='relu'),
    layers.Dense(units = 512, activation='relu'),

    layers.Dense(units = 1)
])

# define the loss function and optimizer 

model.compile(loss="mae", optimizer="adam")




