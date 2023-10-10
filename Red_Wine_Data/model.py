# We will use our sequential model to predict the qualtiy of wine, with certain features
# given in the data set red-wine.csv 

#This dataset consists of physiochemical measurements from about 1600 Portuguese red wines.

#Also included is a quality rating for each wine from blind taste-tests.
import matplotlib
import pandas as pd
from IPython.display import display 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



# read the data from the csv file and store it in red_wine
red_wine = pd.read_csv('./red-wine.csv')

# This data set does not have missing values



#Creating training and validation splits 

df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
#display(df_train.head(4))

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
#print(X_train.shape)

#We will define our model with 3 hidden layers and over 1500 neurons. This should be sufficient for our purpose.

model = keras.Sequential([
    layers.Dense(units = 1024, activation ='relu', input_shape= [11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(units = 1024, activation ='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(units = 1024, activation ='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    
    layers.Dense(units = 1)
])

# define the loss function and optimizer 

model.compile(
    loss="mse", 
    optimizer="adam")

# Let's train our data now. We will define our batch size to be 256, and epochs to be 10. 
# That is the model will run through the data set ten times. 
print(y_train, y_valid)
print(X_train.shape, X_valid.shape)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=256, epochs=10, verbose=0)

# the keras api keeps us updated with the loss in each iteration.

#it is better to store the loss values in a pandas dataframe. and plot it to visualize the loss function over each run

# convert the training history to pandas dataframe

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()

# Use pandas native  method to plot the loss



# Hurray ! You just trained your first deep neural network.










