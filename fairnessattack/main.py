from server import Server
from matplotlib import pyplot as plt 
import numpy as np 

def main():
    global_fairness_values = []
    global_accuracy_values = []
    convergence_threshold = 0.001
    number_of_clients = 51
    beta = 5
    server_handler = Server(number_of_clients=number_of_clients, convergence_threshold=convergence_threshold, beta=beta, set_new_test=False, dataset='census', verbose=False)
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values.append(server_handler.get_fairness_values()["global hist"])
    global_accuracy_values.append(server_handler.get_accuracy_values()['global hist'])
    print(global_fairness_values, global_accuracy_values)
    print("finished test")

if __name__ == "__main__":
    main()