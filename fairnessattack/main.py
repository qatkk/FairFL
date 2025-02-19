from server import Server
from matplotlib import pyplot as plt 

def main():
    global_fairness_values = []
    global_accuracy_values = []
    attack_scenario = {"metric": 'size', "goal": 'fairness', "ratio": 1}
    beta = 5
    server_handler = Server(number_of_clients=51, convergence_threshold=0.001, beta=beta, set_new_test=False, dataset='census')
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values.append(server_handler.get_fairness_values()["global hist"])
    global_accuracy_values.append(server_handler.get_accuracy_values()['global hist'])
    print("finished test")



if __name__ == "__main__":
    main()