from client import Client 
from task import load_data, IncomeClassifier

def main():
    clients = [] 
    initialize_values = []
    total_number_of_datapoints = 0 
    privileged_statistics = 0 
    unprivileged_statistics  = 0 
    number_of_clients = 3
    for client_id in range(3):
        train_holder, test_holder, sensitive_attr_index, privileged_value = load_data(client_id, num_partitions=number_of_clients)
        net = IncomeClassifier() 
        clients.append(Client(net, trainloader=train_holder, testloader=test_holder, sensitive_attr=sensitive_attr_index, privileged_value=privileged_value))
        initialize_values.append(list(clients[client_id].initialize_round()))
        total_number_of_datapoints += initialize_values[client_id][0]
    for client_id in range(3):
        privileged_statistics += initialize_values[client_id][1]*initialize_values[client_id][0]/total_number_of_datapoints
        unprivileged_statistics += initialize_values[client_id][2]*initialize_values[client_id][0]/total_number_of_datapoints
    print(f"total data points are, {total_number_of_datapoints}, with privileged statistics as {privileged_statistics} and unprivileged statistics as {unprivileged_statistics}")
        
if __name__ == "__main__":
    main()