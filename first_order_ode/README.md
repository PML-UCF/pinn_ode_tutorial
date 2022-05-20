# Euler Integration of First Order Ordinary Differential Equations

This folder contains the implementation of the Euler integrator to solve a fatigue crack propagation problem as first order ODE example. 

The files in this folder can be briefly described as follows:

TENSORFLOW

- euler_example.py is a complete code implementation including the creation of the EulerIntegratorCell class, the creation and training of the model as well as prediction on test data
- euler_predict_only.py loads the model weights and predicts on test data
- model.py contains the EulerIntegratorCell class, the Normalization layer and the create_model function

PYTORCH

- euler_example_PyTorch.py is a simplified PyTorch implementation of euler_example.py

DATA:
- The data folder contains the far-field stress data, target crack length and initial crack length for training and prediction 
