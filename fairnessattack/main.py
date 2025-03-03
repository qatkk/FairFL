from client import Client 
from task import load_data, IncomeClassifier

def main():
    clients = [] 
    initialize_values = []
    total_number_of_datapoints = 0 
    privileged_counts = 0 
    unprivileged_counts  = 0 
    total_labels = 0
    number_of_clients = 3
    number_of_rounds = 5
    fairness = {"global value": 0, "global hist":[], "local hist": [[]], "local differences": [[]], "global delta hist": [], "global delta": 0}
    accuracy = {"global value": 0, "global hist":[], "local hist": [[]], "local differences": [[]], "global delta hist": [], "global delta": 0}
    net = IncomeClassifier() 

    for client_id in range(number_of_clients):
        train_holder, test_holder, sensitive_attr_index, privileged_value = load_data(client_id, num_partitions=number_of_clients)
        clients.append(Client(net, trainloader=train_holder, testloader=test_holder, 
                              sensitive_attr=sensitive_attr_index, privileged_value=privileged_value))
        # ///////////////////  Initialization  
        initialize_values.append(list(clients[client_id].initialize_round()))
        total_number_of_datapoints += initialize_values[client_id][0]
        total_labels += initialize_values[client_id][3]
        privileged_counts += initialize_values[client_id][1]
        unprivileged_counts += initialize_values[client_id][2]
    for client_id in range(number_of_clients): 
        clients[client_id].initialize_weights(total_number_of_datapoints)

    print(f"Initialization round done! \n Number of data points in total are, {total_number_of_datapoints}, with privileged statistics as {privileged_counts/total_labels} and unprivileged statistics as {unprivileged_counts/total_labels}")

    for round in range(number_of_rounds):
        print(f"Starting round {round}")
        print("Starting fariness computations: ")
        #  //////////////////////   Fairness Computation  
        fairness["local hist"].append([])
        accuracy["local hist"].append([])
        for client_id in range(number_of_clients):
            fairness["local hist"][round].append(clients[client_id].fairness_evaluate(privileged_counts/total_labels, unprivileged_counts/total_labels, total_number_of_datapoints, privileged_counts, unprivileged_counts))
            fairness["global value"] += fairness["local hist"][round][client_id]
            _, _, client_accuracy = clients[client_id].evaluate()
            accuracy["local hist"][round].append(client_accuracy["weighted"]/total_number_of_datapoints)
            accuracy["global value"] += accuracy["local hist"][round][client_id]

        # fairness["local hist"][round] = fairness_round
        fairness["global hist"].append(fairness["global value"])
        fairness["global value"] = 0 
        # accuracy["local hist"][round] = accuracy_round
        accuracy["global hist"].append(accuracy["global value"])
        accuracy["global value"] = 0

        print(f"Fairness values are: \n {fairness['local hist'][round]} \n and accuracies are: \n {accuracy["local hist"][round]}")
        if round >= 1: 
            print(f"Global fairness for this round is {fairness["global hist"][round]} and for the previous round {fairness['global hist'][round - 1]}")
        print("Starting local difference computations: ")
        #  ///////////////////////  Delta Computation 
        fairness["local differences"].append([])
        accuracy["local differences"].append([])
        for client_id in range(number_of_clients):
            fairness["local differences"][round].append(abs(fairness["global hist"][round] - fairness["local hist"][round][client_id]))
            fairness["global delta"] += fairness["local differences"][round][client_id]/number_of_clients
            accuracy["local differences"][round].append(abs(accuracy["global hist"][round] - accuracy["local hist"][round][client_id]))
            accuracy["global delta"] += accuracy["local differences"][round][client_id]/number_of_clients

        fairness["global delta hist"].append(fairness["global delta"])
        accuracy["global delta hist"].append(accuracy["global delta"])
        fairness["global delta"] = 0
        accuracy["global delta"] = 0

        # ///////////////////// Local Weight Update
        print("Updating local weights based on local/global differences: ")
        beta = 0.1 
        for client_id in range(number_of_clients):
            clients[client_id].update_weights(beta, fairness["global delta hist"][round])
        
        # ///////////////////// Model Aggregation 
        print("Aggregating models with respect to their weights:")
        aggregated_weights = 0 
        global_parameters = clients[0].get_client_parameters(weighted = True)
        for client_id in range(number_of_clients): 
            aggregated_weights += clients[client_id].get_weight()
            if (client_id != 1):
                global_parameters = [arr1 + arr2 for arr1, arr2 in zip(global_parameters, clients[client_id].get_client_parameters(weighted = True))]
        aggregated_parameters = [arr * aggregated_weights for arr in global_parameters]

        print ("Fitting new parameters: ")
        for client_id in range(number_of_clients):
            clients[client_id].fit(aggregated_parameters)

if __name__ == "__main__":
    main()