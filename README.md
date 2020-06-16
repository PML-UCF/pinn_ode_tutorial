# Python Implementation of Ordinary Differential Equations Solvers using Hybrid Physics-informed Neural Networks

This repository is provided as a tutorial for the implementation of integration algorithms of first and second order ODEs through recurrent neural networks in Python. The first order example implements an Euler forward integrator used to solve a fatigue crack growth problem. The second order example implements a Runge-Kutta integrator used for the system identification of a two degree of freedom vibrations problem.  

The repository consists of the following two folders:

### first_order_ode:
- euler_example is a complete code implementation including the EulerIntegratorCell class, creation and training of the model as well as prediction on test data
- euler_save_training trains the imported model on training data and saves the model weights 
- euler_predict_only loads the model weights and predicts on test data
- model contains the EulerIntegratorCell class, the Normalization layer and the create_model function

### second_order_ode:
- runge_kutta_example is a complete implementation of a Runge-Kutta integrator including model training and prediction 
- runge_kutta_save_training trains the trainable coefficients on the training data and saves the model weights
- runge_kutts_predict_only loads the saved model weights and predicts on test data
- model contains the RungeKuttaIntegratorCell class and the create_model function

In order to run the codes, you can clone the repository:

``` 
$ git clone https://github.com/PML-UCF/pinn_code_tutorial.git
```
## Citing this repository
Please cite this repository using:

```
@misc{2020_pinn_educational,
	Author = {Kajetan Fricke and Renato G. Nascimento and Felipe A. C. Viana},
	Doi = {10.5281/zenodo.3356877},
	Howpublished = {https://github.com/PML-UCF/pinn\_code\_tutorial},
	Month = {May},
	Publisher = {Zenodo},
	Title = {Python Implementation of Ordinary Differential Equations Solvers using Hybrid Physics-informed Neural Networks},
	Url = {https://github.com/PML-UCF/pinn\_code\_tutorial},
	Version = {0.0.1},
	Year = {2020},
	Bdsk-Url-1 = {https://github.com/PML-UCF/pinn\_code\_tutorial},
	Bdsk-Url-2 = {https://doi.org/10.5281/zenodo.3356877}}
```

The corresponding reference entry should look like: 

K. Fricke, R. G. Nascimento, and F. A. C. Viana, Python Implementation of Ordinary Differential Equations Solvers using Hybrid Physics-informed Neural Networks, v0.0.1, Zenodo, https://github.com/PML-UCF/pinn_code_tutorial, 10.5281/zenodo.3356877.
