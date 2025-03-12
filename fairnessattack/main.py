from server import Server
from matplotlib import pyplot as plt 
import numpy as np 

def main():
    global_fairness_values = []
    global_accuracy_values = []
    convergence_threshold = 0.001
    number_of_clients = 3
    beta = 1
    set_new_test = {'load model': False , 'load data': False}
    server_handler = Server(number_of_clients=number_of_clients, convergence_threshold=convergence_threshold, beta=beta, set_new_test=set_new_test, dataset='census', verbose=True)
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values.append(server_handler.get_fairness_values()["global hist"])
    global_accuracy_values.append(server_handler.get_accuracy_values()['global hist'])
    print(global_fairness_values, global_accuracy_values)
    print("finished test")

    attack_scenario = {"metric": 'stats', "goal": 'fairness', "ratio": 5}
    server_handler = Server(number_of_clients=number_of_clients, convergence_threshold=convergence_threshold, beta=beta, set_new_test=set_new_test, dataset='census', attack=True, attack_scenario=attack_scenario, verbose=True)
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values.append(server_handler.get_fairness_values()["global hist"])
    global_accuracy_values.append(server_handler.get_accuracy_values()['global hist'])
    print("finished test")
    print(f"""Fairness values with no change are {global_fairness_values[0]}
            the attacked values are {global_fairness_values[1]}
            the change percentage between the rounds are {np.array(global_fairness_values[1][len(global_fairness_values[1])-1])/ np.array(global_fairness_values[0][len(global_fairness_values[0])-1]) * 100} """)
    print(f"""Accuracies values with no change are {global_accuracy_values[0]}
            the attacked values are {global_accuracy_values[1]}
            the change percentage between the rounds are {np.array(global_accuracy_values[1][len(global_accuracy_values[1])-1])/ np.array(global_accuracy_values[0][len(global_accuracy_values[0])-1]) * 100} """)


if __name__ == "__main__":
    main()