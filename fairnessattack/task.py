from collections import OrderedDict
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flwr_datasets import FederatedDataset
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets.partitioner import DirichletPartitioner


fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    global fds
    if fds is None:
            partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.2, partition_by='income', min_partition_size=300)
            fds = FederatedDataset(
                dataset="scikit-learn/adult-census-income",
                partitioners={"train": partitioner},
            )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset.dropna(inplace=True)

    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    X = dataset.drop("income", axis=1)
    y = dataset["income"]
    attr_index = X.columns.get_loc('race')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    row = X_test[X_test['race'] == 4]
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    privileged_transformed = preprocessor.transform(row)[:, attr_index][0]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=14, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False)
    return train_loader, test_loader, attr_index, privileged_transformed

def save_dataset(partition_id: int, num_partitions: int, save_path: str = "dataset.csv"):
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.2, partition_by='income', min_partition_size=300)
        fds = FederatedDataset(
            dataset="scikit-learn/adult-census-income",
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset.dropna(inplace=True) 
    categorical_cols = dataset.select_dtypes(include=["object"]).columns
    ordinal_encoder = OrdinalEncoder()
    dataset[categorical_cols] = ordinal_encoder.fit_transform(dataset[categorical_cols])

    X = dataset.drop("income", axis=1)
    y = dataset["income"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Save the processed dataset
    X_train.to_csv(str(partition_id)+ "_train_" +save_path, index=False)
    y_train.to_csv(str(partition_id)+ "_train_label_" +save_path, index=False)
    X_test.to_csv(str(partition_id)+ "_test_" +save_path, index=False)
    y_test.to_csv(str(partition_id)+ "_test_label_" +save_path, index=False)

    print(f"Dataset saved to {save_path}")


def prepare_dataset(partition_id, file_path: str = "dataset.csv"):
    X_train = pd.read_csv(str(partition_id)+ "_train_" +file_path)
    y_train = pd.read_csv(str(partition_id)+ "_train_label_" +file_path)
    X_test = pd.read_csv(str(partition_id)+ "_test_" +file_path)
    y_test = pd.read_csv(str(partition_id)+ "_test_label_" +file_path)
    X = pd.concat([X_train, X_test], ignore_index=True)
    attr_index = X.columns.get_loc('race')

    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    row = X_test[X_test['race'] == 4]
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    privileged_transformed = preprocessor.transform(row)[:, attr_index][0]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=14, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False)
    return train_loader, test_loader, attr_index, privileged_transformed


class IncomeClassifier(nn.Module):
    def __init__(self, input_dim: int = 14):
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
    
def load_model(filepath="income_classifier.pth", input_dim=14):
    model = IncomeClassifier(input_dim=input_dim)  # Create an instance
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
        
    p_2 = client_unprivilege_label_count / total_unprivilege_label_count 
    p_3 = global_unprivilege_prob
    p_5 = client_priviledge_label_count / total_privilege_label_count 
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
