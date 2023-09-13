# This concept deals with the capacity of the model. It is generally defined by number of neurons or how well neurons are connected to each ohter. 

import model_lib

single_model = model_lib.single_neuron_model(16, 1)


wider_model = model_lib.wider_model(32, 1)


deeper_model = model_lib.deeper_model(32, 1)