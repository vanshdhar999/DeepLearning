# In order to prevent overfitting, we add a dropout layer before the layer we want the
#dropout feature to work on. 
# This is done to make sure our model does to learn suprious patterns, therefore we 
#break the conspiracy of the weights, by randomly dropping some percentage of the nodes.

#Adding dropout 
import model_lib
model = model_lib.single_neuron_model()
model = keras.Sequential([
        # ...s
        layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
        
        layers.Dense(16),

        # ....
])

# Adding Batch Normalization

layers.Dense(16, activation='relu'),
layers.BatchNormalization(),