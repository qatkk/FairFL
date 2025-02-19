from client import Client 
from task import load_data, IncomeClassifier, save_dataset, prepare_dataset, save_model, load_model, set_weights, get_weights
import sys

class Server():
    def __init__(self, convergence_threshold, beta, set_new_test, number_of_clients, dataset = 'adult', attack = False, attack_scenario = {"metric": None, "goal": None, "ratio": 1}, verbose = False):
        self.clients = [] 
        self.verbos = verbose
        self.dataset = dataset
        self.total_number_of_datapoints = 0 
        self.privileged_counts = 0 
        self.unprivileged_counts  = 0 
        self.convergence_threshold = convergence_threshold
        self.attack = attack
        self.attack_scenario = attack_scenario
        self.total_labels = 0
        self.number_of_clients = number_of_clients
        self.beta = beta
        self.global_model = IncomeClassifier(dataset = dataset)
        self.set_dataset = set_new_test
        self.fairness_values = {"global value": 0, "global hist":[], "local hist": [[]], "local differences": [[]], "global delta hist": [], "global delta": 0}
        self.accuracy_values = {"global value": 0, "global hist":[], "local hist": [[]], "local differences": [[]], "global delta hist": [], "global delta": 0}
    
    def initilize(self): 
        initialize_values = []
        if (self.set_dataset):
            save_model(self.global_model)
            if self.dataset == 'census' :
                save_dataset()
            else : 
                load_data(self.number_of_clients)
        else: 
            try:
                self.global_model = load_model(dataset= self.dataset)
            except Exception as e:
                sys.exit("Initialize the dataset first -> set_new_test = True")

        for client_id in range(self.number_of_clients):
            train_holder, test_holder, sensitive_attr_index, privileged_value = prepare_dataset(client_id, dataset= self.dataset)
            self.clients.append(Client(client_id, self.global_model, trainloader=train_holder, testloader=test_holder, 
                                sensitive_attr=sensitive_attr_index, privileged_value=privileged_value, malicious= self.attack, attack_scenario = self.attack_scenario))
            # ///////////////////  Initialization  
            initialize_values.append(list(self.clients[client_id].initialize_round()))
            self.total_number_of_datapoints += initialize_values[client_id][0]
            self.total_labels += initialize_values[client_id][3]
            self.privileged_counts += initialize_values[client_id][1]
            self.unprivileged_counts += initialize_values[client_id][2]
        for client_id in range(self.number_of_clients): 
            self.clients[client_id].initialize_weights(self.total_number_of_datapoints)


    def training(self):
        round = 0 
        fairness_epsilon = 1000
        while(fairness_epsilon > self.convergence_threshold):
            print(f"Starting round {round}")
            #  //////////////////////   Fairness Computation  
            self.fairness_values["local hist"].append([])
            self.accuracy_values["local hist"].append([])
            for client_id in range(self.number_of_clients):
                client_fairness = self.clients[client_id].fairness_evaluate(privileged_probability=float(self.privileged_counts/self.total_number_of_datapoints), 
                                                                            unprivileged_probability=float(self.unprivileged_counts/self.total_number_of_datapoints),
                                                                            total_data_points=self.total_number_of_datapoints)
                self.fairness_values["local hist"][round].append(client_fairness)
                self.fairness_values["global value"] += self.fairness_values["local hist"][round][client_id]
                _, _, client_accuracy = self.clients[client_id].evaluate()
                self.accuracy_values["local hist"][round].append(client_accuracy["weighted"]/self.total_number_of_datapoints)
                self.accuracy_values["global value"] += self.accuracy_values["local hist"][round][client_id]

            self.fairness_values["global hist"].append(self.fairness_values["global value"])
            self.fairness_values["global value"] = 0 
            self.accuracy_values["global hist"].append(self.accuracy_values["global value"])
            self.accuracy_values["global value"] = 0
            if (round>=1):
                fairness_epsilon = abs(self.fairness_values["global hist"][round] - self.fairness_values["global hist"][round-1])
            if self.verbos: 
                print(f"Fairness values are: \n {self.fairness_values['local hist'][round]} \n and accuracies are: \n {self.accuracy_values["local hist"][round]}")

            #  ///////////////////////  Delta Computation 
            self.fairness_values["local differences"].append([])
            self.accuracy_values["local differences"].append([])
            self.fairness_values["global delta"] = 0
            self.accuracy_values["global delta"] = 0
            for client_id in range(self.number_of_clients):
                self.fairness_values["local differences"][round].append(abs(self.fairness_values["global hist"][round] - self.fairness_values["local hist"][round][client_id]))
                self.fairness_values["global delta"] += self.fairness_values["local differences"][round][client_id]/self.number_of_clients
                self.accuracy_values["local differences"][round].append(abs(self.accuracy_values["global hist"][round] - self.accuracy_values["local hist"][round][client_id]))
                self.accuracy_values["global delta"] += self.accuracy_values["local differences"][round][client_id]/self.number_of_clients

            self.fairness_values["global delta hist"].append(self.fairness_values["global delta"])
            self.accuracy_values["global delta hist"].append(self.accuracy_values["global delta"])


            # ///////////////////// Local Weight Update
            for client_id in range(self.number_of_clients):
                self.clients[client_id].update_weights(self.beta, self.fairness_values['local differences'][round][client_id], self.fairness_values["global delta hist"][round])
            
            # ///////////////////// Model Aggregation 
            aggregated_weights = 0 
            global_parameters = self.clients[0].get_client_parameters(weighted = True)
            for client_id in range(self.number_of_clients): 
                aggregated_weights += self.clients[client_id].get_weight()
                if (client_id != 0):
                    global_parameters = [arr1 + arr2 for arr1, arr2 in zip(global_parameters, self.clients[client_id].get_client_parameters(weighted = True))]
            aggregated_parameters = [arr / aggregated_weights for arr in global_parameters]
            set_weights(self.global_model, aggregated_parameters)
            # ///////////////////// Fitting the new global model parameters 
            for client_id in range(self.number_of_clients):
                self.clients[client_id].fit(aggregated_parameters)

            round += 1 

    def get_fairness_values(self):
        return self.fairness_values
    
    def get_accuracy_values(self):
        return self.accuracy_values
    
    def get_global_model(self):
        return get_weights(self.global_model)