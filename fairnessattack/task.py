from collections import OrderedDict
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from folktables import ACSDataSource, ACSIncome
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

fds = None  # Cache FederatedDataset


def load_data(num_partitions: int):
    dataset = load_dataset("mstz/adult", "income")
    
    X_train = dataset['train'].to_pandas() 
    file_path = './data/'
    X_test = dataset['test'].to_pandas()
    X = pd.concat([X_train, X_test], ignore_index=True)
    X.dropna(inplace=True)

    categorical_cols = X.select_dtypes(include=["object"]).columns

    ordinal_encoder = OrdinalEncoder()
    X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
    client_partitions = dirichlet_partition(X, num_clients=num_partitions, alpha=0.2, label_column='is_male')

    for client_id in range(num_partitions):
        client_data = client_partitions[client_id]
    
        X_client = client_data.drop(columns=["over_threshold"])
        y_client = client_data["over_threshold"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.20, random_state=42
        )
        train_data = X_train.copy()
        train_data['over_threshold'] = y_train

        for _ in range(10):
            train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X_train = train_data.drop(columns=["over_threshold"])
        y_train = train_data["over_threshold"]
        
        X_train.to_csv(file_path + str(client_id) + "_train_dataset.csv", index=False)
        y_train.to_csv(file_path + str(client_id) + "_train_label_dataset.csv", index=False)
        X_test.to_csv(file_path + str(client_id) + "_test_dataset.csv", index=False)
        y_test.to_csv(file_path + str(client_id) + "_test_label_dataset.csv", index=False)
        

def dirichlet_partition(df, label_column, num_clients, alpha=0.2):
    labels = df[label_column].values
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    # Generate Dirichlet distributions per class
    partitions = {i: [] for i in range(num_clients)}
    for c in unique_classes:
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample Dirichlet distribution for this class
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Ensure all data is assigned
        proportions[-1] += len(class_indices) - sum(proportions)
        
        # Distribute indices to partitions
        start_idx = 0
        for i, count in enumerate(proportions):
            partitions[i].extend(class_indices[start_idx:start_idx + count])
            start_idx += count
    
    client_datasets = {i: df.iloc[indices] for i, indices in partitions.items()}
    
    return client_datasets

def save_dataset(save_path: str = "./data/"):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    state_abbreviations = {
        0: "AL", 1: "AK", 2: "AZ", 3: "AR", 4: "CA", 5: "CO", 6: "CT", 7: "DE", 8: "FL",
        9: "GA", 10: "HI", 11: "ID", 12: "IL", 13: "IN", 14: "IA", 15: "KS", 16: "KY", 17: "LA", 
        18: "ME", 19: "MD", 20: "MA", 21: "MI", 22: "MN", 23: "MS", 24: "MO", 25: "MT", 26: "NE", 
        27: "NV", 28: "NH", 29: "NJ", 30: "NM", 31: "NY", 32: "NC", 33: "ND", 34: "OH", 35: "OK", 
        36: "OR", 37: "PA", 38: "RI", 39: "SC", 40: "SD", 41: "TN", 42: "TX", 43: "UT", 44: "VT", 
        45: "VA", 46: "WA", 47: "WV", 48: "WI", 49: "WY", 50: "PR"
    }
    for state_index in range(51):
        print(state_abbreviations[state_index])
        acs_data = data_source.get_data(states=[state_abbreviations[state_index]], download=True)
        features, label, _ = ACSIncome.df_to_numpy(acs_data)

        X = pd.DataFrame(features, columns=ACSIncome.features)
        X_train, X_test, y_train, y_test = train_test_split(
            X, label, test_size=0.20, random_state=42
        )
        # train_data = X_train.copy()
        # train_data['income'] = y_train
        # for _ in range(10):
        #     train_data = train_data.sample(frac=1, random_state=99).reset_index(drop=True)
        # X_train = pd.DataFrame(train_data.drop(columns=["income"]))
        # y_train = train_data["income"]
        pd.DataFrame(X_train).to_csv(save_path + str(state_index)+ "_train_dataset.csv" , index=False)
        pd.DataFrame(y_train).to_csv(save_path + str(state_index)+ "_train_label_dataset.csv" , index=False)
        pd.DataFrame(X_test).to_csv(save_path + str(state_index)+ "_test_dataset.csv", index=False)
        pd.DataFrame(y_test).to_csv(save_path + str(state_index)+ "_test_label_dataset.csv", index=False)




def prepare_dataset(partition_id, dataset, file_path: str = "./data/"):
    X_train = pd.read_csv(file_path + str(partition_id)+ "_train_dataset.csv")
    y_train = pd.read_csv(file_path + str(partition_id)+ "_train_label_dataset.csv" )
    X_test = pd.read_csv(file_path + str(partition_id)+ "_test_dataset.csv")
    y_test = pd.read_csv(file_path + str(partition_id)+ "_test_label_dataset.csv")
    X = pd.concat([X_train, X_test], ignore_index=True)
    if dataset == 'adult':
        attribute = {"name": 'is_male', 'value':True}
    else: 
        attribute = {"name": 'RAC1P', 'value':1}
    attr_index = X.columns.get_loc(attribute["name"])
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    X_train = preprocessor.fit_transform(X_train)

    try:
        row = X_test[X_test[attribute["name"]] == attribute["value"]].iloc[0]
        row_df = pd.DataFrame([row])
        privileged_transformed = preprocessor.transform(row_df)[:, attr_index][0]
    except: 
        privileged_transformed = -1 
    X_test = preprocessor.transform(X_test)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=35, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=35, shuffle=False)
    return train_loader, test_loader, attr_index, privileged_transformed


class IncomeClassifier(nn.Module):
    def __init__(self, dataset):
        if dataset == 'adult':
            input_dim = 12 
        elif dataset == 'census': 
            input_dim = 10
        super(IncomeClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def save_model(model, filepath="income_classifier.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")\
    
def load_model(dataset, filepath="income_classifier.pth"):
    if dataset == 'adult' : 
        input_dim = 12 
    elif dataset == 'census' :
        input_dim = 10
    model = IncomeClassifier(dataset= dataset)  # Create an instance
    model.load_state_dict(torch.load(filepath)) 
    return model 


def train(model, train_loader, num_epochs=1):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader):
    model.eval()
    criterion = nn.BCELoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    return loss, accuracy

def fairness(
        model, test_loader, global_privileged_prob,
        global_unprivilege_prob, attr_index, privileged_value, 
        client_priviledge_label_count, client_unprivilege_label_count, 
        client_points, total_points, total_unprivilege_label_count, total_privilege_label_count):
    model.eval()
    privileged = 0 
    unprivilege = 0 
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            labeled = (y_batch.view(-1) == 1) 
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            mask = (labeled & (predicted.view(-1) == 1))
            if mask.any():  
                try: 
                    privileged_count = (X_batch[mask, attr_index] == privileged_value).sum().item()
                except Exception as e:
                    privileged_count = 0  
                try:
                    unprivileged_count = (X_batch[mask, attr_index] != privileged_value).sum().item()
                except Exception as e:
                    unprivileged_count = 0  
                privileged += privileged_count
                unprivilege += unprivileged_count
    if client_priviledge_label_count == 0 :
        p_4 = 0 
    else : 
        p_4 = privileged / client_priviledge_label_count

    if client_unprivilege_label_count == 0 : 
        p_1 = 0
    else  :
        p_1 = unprivilege / client_unprivilege_label_count
        
    p_2 = client_unprivilege_label_count / client_points 
    p_3 = global_unprivilege_prob
    p_5 = client_priviledge_label_count / client_points 
    p_6 = global_privileged_prob
    client_fairness = (client_points/total_points) * (((p_1 * p_2 )/p_3) - ((p_4 * p_5)/p_6))
    return client_fairness


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)



def get_weights(net, scalar = 1.0):
    ndarrays = [val.cpu().numpy() * scalar for _, val in net.state_dict().items()]
    return ndarrays
