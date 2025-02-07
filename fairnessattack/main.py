from client import Client 
from server import Server
from task import load_data, IncomeClassifier, save_dataset, prepare_dataset, save_model, load_model
from matplotlib import pyplot as plt 

def main():
    server_handler = Server(number_of_clients=5, convergence_threshold=0.0001, beta=5, set_new_test=True)
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values = server_handler.get_fairness_values()["global hist"]
    plt.plot(global_fairness_values)
    plt.show()


if __name__ == "__main__":
    main()


