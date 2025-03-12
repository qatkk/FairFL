from task import IncomeClassifier,  evaluate, get_weights, set_weights, train, fairness, test_updated_model 
import torch
import warnings

class Client():
    def __init__(self, id, net, trainloader, sensitive_attr, privileged_value, dataset = 'census', malicious = False, attack_scenario = {"metric": None, "goal": None, "ratio": 1}):
        self.id = id
        self.trainloader = trainloader
        self.sensitive_attr = sensitive_attr
        self.labeled_privileged = 0 
        self.labeled_unprivileged = 0 
        self.weight = 0
        self.malicious = malicious
        self.dataset = dataset
        data_size = 0
        self.fairness = 0
        self.attack_scenario = attack_scenario
        for X_batch, y_batch in trainloader:
            data_size += y_batch.size(0)
        self.data_size = data_size
        self.privileged_value = privileged_value
        # Fix this for the other dataset 
        model_parameters = get_weights(net=net)
        custom_weights = torch.from_numpy(model_parameters[0]).float() 
        custom_bias = torch.from_numpy(model_parameters[1]).float() 
        self.net = IncomeClassifier(dataset=dataset, custom_weights=custom_weights, custom_bias=custom_bias)

    def fit(self, parameters):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, total_number_of_points = 1):
        loss, accuracy = evaluate(self.net, self.trainloader)
        return loss, len(self.trainloader), {"accuracy": accuracy, "weighted": accuracy* (self.data_size/total_number_of_points)}
    
    def fairness_evaluate(self, privileged_count, unprivileged_count, total_data_points, global_fairness):  
        self.fairness = fairness(model=self.net, test_loader=self.trainloader, global_privileged_labeled_count=privileged_count, 
                global_unprivilege_labeled_count=unprivileged_count, attr_index=self.sensitive_attr, 
                privileged_value=self.privileged_value, client_priviledge_label_count=self.labeled_privileged, 
                client_unprivilege_label_count=self.labeled_unprivileged, client_points=self.data_size, total_points=total_data_points)
        if (self.attack_scenario['metric'] == 'fairness') & (self.malicious == True): 
            return manipulate_parameter(attack_scenario=self.attack_scenario, client_id=self.id, manipulated_param=[global_fairness, self.fairness])
        return self.fairness * (self.data_size/total_data_points)

    
    def initialize_round(self):
        privileged = 0 
        unprivilege = 0
        labeled = 0 
        data_size = self.data_size
        if (self.attack_scenario['metric'] == 'size') & (self.malicious == True): 
            self.data_size, data_size = manipulate_parameter(attack_scenario=self.attack_scenario, manipulated_param=self.data_size, client_id=self.id)
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
        self.labeled_privileged = privileged
        self.labeled_unprivileged = unprivilege 
        if (self.attack_scenario['metric'] == 'stats') and self.malicious == True : 
            privileged, unprivilege  = manipulate_parameter(attack_scenario=self.attack_scenario, manipulated_param={'unpriv':unprivilege, 'priv':privileged}, client_id=self.id)
        if labeled !=0 :   
            return data_size, privileged, unprivilege, labeled
        else: 
            warnings.warn(f"Client {self.id} doesn't have any true labeled data")
            return data_size, privileged, unprivilege, labeled

    def return_delta(self, global_parameters):
        _, _, accuracy =  self.evaluate()
        delta = {"fairness": abs(global_parameters['fairness'] - self.fairness), "accuracy": abs(global_parameters['accuracy'] - accuracy['accuracy'])}
        if self.malicious == True and self.attack_scenario['metric'] == 'delta': 
            delta = manipulate_parameter(attack_scenario=self.attack_scenario, client_id=self.id, 
                                         manipulated_param={'client delta':delta, 
                                                            'fairness global delta': global_parameters['fairness delta'], 
                                                            'accuracy global delta': global_parameters['accuracy delta']})
        return delta

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
    
    def test_model_updates(self):
        return test_updated_model(self.net, self.testloader)
    

def manipulate_parameter(attack_scenario, client_id, manipulated_param = 0 ):
        if attack_scenario['metric'] == 'size' : 
            if (attack_scenario['goal'] == 'contribution' and client_id == 0) :
                return int(manipulated_param * attack_scenario['ratio']), 0
            else :
                return  manipulated_param ,  manipulated_param
        elif attack_scenario['metric'] == 'stats' : 
            if attack_scenario['goal'] == 'fairness' and client_id == 1 :
                return 0 , int(manipulated_param['unpriv'] * attack_scenario['ratio'])
            else : 
                return manipulated_param['priv'], manipulated_param['unpriv']
        elif attack_scenario['metric'] == 'fairness' : 
            if attack_scenario['goal'] == 'convergence' and client_id == 1 :
                return manipulated_param[0] * attack_scenario['ratio']
            else : 
                return manipulated_param[1]
        elif attack_scenario['metric'] == 'delta' : 
            if attack_scenario['goal'] == 'fairness' and client_id == 0 :
                if not len(manipulated_param['fairness global delta']):
                    delta = manipulated_param['client delta']
                    return {key: value * attack_scenario['ratio'] for key, value in delta.items()}
                else: 
                    delta = {'fairness': manipulated_param['fairness global delta'][len(manipulated_param['fairness global delta'])-1], 
                             'accuracy': manipulated_param['accuracy global delta'][len(manipulated_param['accuracy global delta'])-1]}
                    return {key: value * attack_scenario['ratio'] for key, value in delta.items()}
            else :
                return manipulated_param['client delta']