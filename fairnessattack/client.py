from task import IncomeClassifier,  evaluate, get_weights, load_data, set_weights, train, fairness
import torch
import warnings

class Client():
    def __init__(self, id, net, trainloader, testloader, sensitive_attr, privileged_value, dataset = 'census', malicious = False, attack_scenario = {"metric": None, "goal": None, "ratio": 1}):
        self.net = net
        self.id = id
        self.trainloader = trainloader
        self.testloader = testloader
        self.sensitive_attr = sensitive_attr
        self.labeled_privileged = 0 
        self.labeled_unprivileged = 0 
        self.weight = 0
        self.malicious = malicious
        self.dataset = dataset
        data_size = 0
        self.fairness = 0
        self.scenario = attack_scenario
        self.attack_scenario = attack_scenario
        for X_batch, y_batch in trainloader:
            data_size += y_batch.size(0)
        self.data_size = data_size
        self.privileged_value = privileged_value

    def fit(self, parameters):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self):
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy, "weighted": accuracy*self.data_size}
    
    def fairness_evaluate(self, privileged_probability, unprivileged_probability,
                total_data_points, total_privilege_label_count, total_unprivilege_label_count):
        self.fairness = fairness(model=self.net, test_loader=self.testloader, global_privileged_prob=privileged_probability, 
                global_unprivilege_prob=unprivileged_probability, attr_index=self.sensitive_attr, 
                privileged_value=self.privileged_value, client_priviledge_label_count=self.labeled_privileged, 
                client_unprivilege_label_count=self.labeled_unprivileged, client_points=self.data_size, total_points=total_data_points, 
                total_privilege_label_count=total_privilege_label_count, total_unprivilege_label_count=total_unprivilege_label_count)
        
        return self.fairness

    
    def initialize_round(self):
        privileged = 0 
        unprivilege = 0
        labeled = 0 
        if (self.attack_scenario['metric'] == 'size') & (self.malicious == True): 
            self.data_size = manipulate_parameter(self.attack_scenario, self.data_size, self.id)

        with torch.no_grad():
            for X_batch, y_batch in self.trainloader:
                mask = (y_batch.view(-1) == 1) 
                if mask.any():  
                    try: 
                        privileged_count = (X_batch[mask, self.sensitive_attr] == self.privileged_value).sum().item()
                    except Exception as e:
                        privileged_count = 0  
                    try:
                        unprivileged_count = (X_batch[mask, self.sensitive_attr] != self.privileged_value).sum().item()
                    except Exception as e:
                        unprivileged_count = 0  
                    privileged += privileged_count
                    unprivilege += unprivileged_count
                    labeled += (y_batch == 1).sum().item()
                else:
                    labeled += 0             
        if labeled !=0 :   
            self.labeled_privileged = privileged
            self.labeled_unprivileged = unprivilege
            return self.data_size, privileged, unprivilege, labeled
        else: 
            warnings.warn(f"Client {self.id} doesn't have any true labeled data")
            return self.data_size, privileged, unprivilege, labeled

    
    def initialize_weights(self, total_size):
        self.weight = self.data_size / total_size
    
    def update_weights(self, beta, client_delta, global_delta):
        self.weight = self.weight - (beta * (client_delta - global_delta))

    def get_client_parameters(self, weighted = False):
        if weighted: 
            return get_weights(self.net, self.weight)
        else:
            return get_weights(self.net)
    
    def get_weight(self):
        return self.weight
    
def manipulate_parameter(attack_scenario, manipulated_param, client_id):
        if (attack_scenario['goal'] == 'contribution' and client_id == 49) : # The id for Wyoming state with the least datasize
            manipulated_param = int(manipulated_param * attack_scenario['ratio'])
        elif attack_scenario['goal'] == 'fairness' and client_id == 10 : # Client id for Hawaii with least privileged people
            manipulated_param = int(manipulated_param * attack_scenario['ratio'])
        return manipulated_param