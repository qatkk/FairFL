from server import Server
from matplotlib import pyplot as plt 
from task import save_dataset, prepare_dataset

def main():
    global_fairness_values = []
    global_accuracy_values = []
    attack_scenario = {"metric": 'size', "goal": 'contribution', "ratio": 10}
    beta = 5
    server_handler = Server(number_of_clients=51, convergence_threshold=0.0005, beta=beta, set_new_test=False, dataset='census')
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values.append(server_handler.get_fairness_values()["global hist"])
    global_accuracy_values.append(server_handler.get_accuracy_values()['global hist'])
    print("finished test")
    server_handler = Server(number_of_clients=51, convergence_threshold=0.0005, beta=beta, set_new_test=False, dataset='census', attack_scenario=attack_scenario, attack=True)
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values.append(server_handler.get_fairness_values()["global hist"])
    global_accuracy_values.append(server_handler.get_accuracy_values()['global hist'])
    print(f"final fairness values are {global_fairness_values}")
    print(f"final accuracies are {global_accuracy_values}")



if __name__ == "__main__":
    main()