#Now we're going to apply neural networks to another common machine learning problem: classification.
#The main difference is in the loss function we use and in what kind of outputs we want the final layer to produce.

#In your raw data, the classes might be represented by strings like "Yes" and "No", or "Dog" and "Cat". Before using this data we'll assign a class label: one class will be 0 and the other will be 1.
#Assigning numeric labels puts the data in a form a neural network can use.


# Accuracy and Cross Entropy
#Accuracy is one of the many metrics in use for measuring success on a classification problem. Accuracy is the ratio of correct predictions to total predictions: accuracy = number_correct / total. 
#A model that always predicted correctly would have an accuracy score of 1.0.

#For classification, what we want instead is a distance between probabilities, and this is what cross-entropy provides.
# Cross-entropy is a sort of measure for the distance from one probability distribution to another.

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'), # We add sigmoid function to get the probability between 0 and 1.
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

#Add the cross-entropy loss and accuracy metric to the model with its compile method.