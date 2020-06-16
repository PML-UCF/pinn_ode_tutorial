# Euler Integration of First Order Ordinary Differential Equations

This folder contains the implementation of the Euler integrator to solve a fatigue crack propagation problem as first order ODE example. 

The files in this folder can be briefly described as follows:

- euler_example is a complete code implementation including the creation of the EulerIntegratorCell class, the creation and training of the model as well as prediction on test data
- euler_save_training trains the imported model on training data and saves the model weights 
- euler_predict_only loads the model weights and predicts on test data
- model contains the EulerIntegratorCell class, the Normalization layer and the create_model function
- The data folder contains the far-field stress data, target crack length and initial crack length for training and prediction 
