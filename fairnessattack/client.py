from task import IncomeClassifier,  evaluate, get_weights, load_data, set_weights, train, fairness
import torch


class Client():
    def __init__(self, net, trainloader, testloader, sensitive_attr, privileged_value):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.sensitive_attr = sensitive_attr
        data_size = 0
        for X_batch, y_batch in trainloader:
            data_size += y_batch.size(0)
        self.data_size = data_size
        self.privileged_value = privileged_value

    def fit(self, parameters):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}
    
    def initialize_round(self):
        privileged = 0 
        unprivilege = 0
        labeled = 0 
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
            return self.data_size, privileged/labeled, unprivilege/labeled
        else: 
            return -1
     

    
    def initialize_weights(self, total_size):
        state_dict = self.net.state_dict()  # Get the current state dict
        for key in state_dict.keys():
            state_dict[key] = torch.full_like(state_dict[key], self.data_size/total_size)  # Fill with a single value
        self.net.load_state_dict(state_dict)


    def get_client_weights(self):
        return get_weights(self.net)
    
def client_test():
    train_holder, test_holder, sensitive_attr_index, privileged_value = load_data(2, 3)
    net = IncomeClassifier() 
    first_client = Client(net, trainloader=train_holder, testloader=test_holder, sensitive_attr=sensitive_attr_index, privileged_value=privileged_value)
    first_client.initialize_round()

client_test()