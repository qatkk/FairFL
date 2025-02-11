## Introduction 

This is a simple implementation of ["FairFed: Enabling Group Fairness in Federated Learning," *AAAI Technical Track on Machine Learning I*](https://doi.org/10.1609/aaai.v37i6.25911) in order to analyze its behaviour under attacks, specifically fairness-based targeted attacks. 
The implementation uses Flower framework and Census income dataset. This federated dataset is created using Dirichlet's theorem with Alpha=0.2 (for changing this parameter for your tests, refer to fairnessattack/task.py: load_data, save_dataset). 
In the main file you can set the number of clients, the beta ( the gain to which the fairness metric will be taken into account), and the convergence threshold for your training. The server will first initialize the training with the statistics of the clients and then apply the training algorithm. 

## Setup 

In the directory of the project first run the command below to install the dependancies: 

`poetry install`

After setting your desired training values in the "main.py" run the project as below: 
 `mkdir data`
`cd fairnessattack`
`poetry run python main.py`

