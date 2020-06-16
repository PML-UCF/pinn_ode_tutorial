# Runge-Kutta implementation of second order ordinary differential equation

This folder contains the implementation of the Runge-Kutta integration of a two degree of freedom vibrations system as example for a second order ordinary differential equation. 

The python files can be described as follows:

- runge_kutta_example.py is a complete implementation of a Runge-Kutta integrator including model training and prediction 
- runge_kutta_save_training.py trains the trainable coefficients on the training data and saves the model weights
- runge_kutts_predict_only.py loads the saved model weights and predicts on test data
- model.py contains the RungeKuttaIntegratorCell class and the create_model function
- The data folder contains the input force data and target displacement file for system identification as well as seperate data file for prediction

