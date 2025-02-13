from server import Server
from matplotlib import pyplot as plt 
from task import save_dataset, prepare_dataset

def main():
    server_handler = Server(number_of_clients=51, convergence_threshold=0.0001, beta=1, set_new_test=False, dataset= 'census')
    server_handler.initilize() 
    server_handler.training()
    global_fairness_values = server_handler.get_fairness_values()["global hist"]
    plt.plot(global_fairness_values)
    plt.show()


if __name__ == "__main__":
    main()