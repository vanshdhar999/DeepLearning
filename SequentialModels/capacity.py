# This concept deals with the capacity of the model. It is generally defined by number of neurons or how well neurons are connected to each ohter. 

import model_lib
from tensorflow.keras.callbacks import EarlyStopping 

single_model = model_lib.single_neuron_model(16, 1)


wider_model = model_lib.wider_model(32, 1)


deeper_model = model_lib.deeper_model(32, 1)


# We can add a early stopper in our model that makes sure that we dont overfit our data. We can define more epochs than needed Keras's early stopping feature will take care of the rest.

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

