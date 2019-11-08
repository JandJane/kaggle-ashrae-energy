import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

from config import *


class Net(nn.Module):
    def __init__(self, d_in=10, k=2, n_hidden=1, batch_norm=False, dropout=False):
        super(Net, self).__init__()
        d_cur = d_in
        self.layers = []
        for i in range(n_hidden):
            self.layers.append(nn.Linear(d_cur, d_cur // k))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(d_cur // k))
            self.layers.append(nn.ReLU())
            if dropout:
                self.layers.append(nn.Dropout())
            d_cur //= k
        self.layers.append(nn.Linear(d_cur, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Scaler:
    def __init__(self, preprocessor, batch_size=1024):
        self.df = preprocessor.df
        self.train_idx = preprocessor.train_idx

        self.scaler_labels = None
        self.scaler_features = None
        self.encoders = {}

        self.create_scalers()

        cat_train, num_train, labels_train = self.transform(self.df.loc[preprocessor.train_idx])
        cat_test, num_test, labels_test = self.transform(self.df.loc[preprocessor.test_idx])
        self.d_in = cat_train.shape[1] + num_train.shape[1]

        #         if prod:
        #             self.testloader = torch.FloatTensor(np.concatenate([cat_test, num_test], axis=1)),
        # self.df[~self.df.index.isin(self.train_idx)].row_id
        #         else:
        self.testloader = self.create_dataloader(cat_test, num_test, labels_test, batch_size)
        self.trainloader = self.create_dataloader(cat_train, num_train, labels_train, batch_size)

    def create_scalers(self):
        self.scaler_features = StandardScaler()
        self.scaler_labels = StandardScaler()

        self.scaler_features.fit(self.df.loc[self.train_idx, columns_config['numerical']])
        self.scaler_labels.fit(self.df.loc[self.train_idx, 'meter_reading'].values.reshape(-1, 1))
        for col in columns_config['categorical']:
            self.encoders[col] = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.encoders[col].fit(self.df.loc[self.train_idx, col].values.reshape(-1, 1))

    def transform(self, data):
        num_features = self.scaler_features.transform(data.loc[:, columns_config['numerical']])
        labels = self.scaler_labels.transform(data.loc[:, 'meter_reading'].values.reshape(-1, 1))
        cat_features = []
        for col in columns_config['categorical']:
            cat_features.append(self.encoders[col].transform(data[col].values.reshape(-1, 1)))
        cat_features = np.concatenate(cat_features, axis=1)
        return cat_features, num_features, labels

    @staticmethod
    def create_dataloader(cat, num, labels, batch_size, shuffle=True, add_row_ids=False, row_ids=None):
        data = [cat, num]
        if add_row_ids:
            data.append(row_ids.astype(int))
        dataset = TensorDataset(torch.Tensor(np.concatenate(data, 1)), torch.Tensor(labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader


class Trainer:
    def __init__(self, scaler, net_config, lr=0.001):
        self.scaler = scaler
        self.trainloader = scaler.trainloader
        self.testloader = scaler.testloader
        self.scaler_labels = scaler.scaler_labels

        self.optimizer = None
        self.criterion = None
        self.net = None

        net_config['d_in'] = scaler.d_in
        self.create_models(net_config, lr)

        self.train_losses = []
        self.test_losses = []
        self.metrics = []

    def create_models(self, net_config, lr):
        self.net = Net(**net_config).to(device)
        print('Net architecture:')
        print(self.net)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def metric(self, pred, labels):
        pred_raw = self.scaler_labels.inverse_transform(pred.detach().cpu().numpy())
        labels_raw = self.scaler_labels.inverse_transform(labels.detach().cpu().numpy())
        #         print(pred_raw.shape, labels_raw.shape)
        loss = np.mean((pred_raw - labels_raw) ** 2) ** 0.5
        return loss

    def train(self, n_epochs, verbose=True, do_val=True):
        for epoch in range(n_epochs):
            self.net.train()
            losses = []
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                if inputs.size(0) <= 1:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if verbose:
                print('[%d] Train loss: %.3f' % (epoch + 1, np.mean(losses)))
            self.train_losses.append(np.mean(losses))

            if do_val:
                self.net.eval()
                losses = []
                metrics = []
                for i, data in enumerate(self.testloader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        outputs = self.net(inputs)
                    if np.isnan(outputs.sum().item()):
                        continue
                    loss = self.criterion(outputs, labels)
                    losses.append(loss.item())
                    metrics.append(self.metric(outputs, labels))
                if verbose:
                    print('[%d] Test loss: %.3f' % (epoch + 1, np.mean(losses)))
                    print('[%d] Test metric: %.3f' % (epoch + 1, np.mean(metrics)))
                self.test_losses.append(np.mean(losses))
                self.metrics.append(np.mean(metrics))

    def predict(self, submission):
        self.net.eval()
        inputs, row_ids = self.testloader
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.net(inputs)
        pred_raw = self.scaler_labels.inverse_transform(outputs.detach().cpu().numpy())
        pred_raw[pred_raw < 0] = 0
        submission = np.concatenate([submission,
                                     np.concatenate([row_ids.values.reshape(-1, 1), pred_raw], axis=1)
                                     ], axis=0)
        print(submission[:, 0].shape, np.unique(submission[:, 0]).shape)
        return submission

    def predict(self, test_df, submission, batch_size=100000):
        self.net.eval()
        for i in range(0, test_df.shape[0], batch_size):
            cat_test, num_test, labels_test = self.scaler.transform(test_df[i: min(i + batch_size, test_df.shape[0])])
            inputs = torch.FloatTensor(np.concatenate([cat_test, num_test], axis=1)).to(device)
            print(inputs)
            row_ids = test_df.row_id[i: min(i + batch_size, test_df.shape[0])]
            with torch.no_grad():
                outputs = self.net(inputs)
            print(outputs)
            pred_raw = self.scaler_labels.inverse_transform(outputs.detach().cpu().numpy())
            print(pred_raw)
            #             pred_raw = np.exp(pred_raw) - 1
            submission = np.concatenate([submission,
                                         np.concatenate([row_ids.values.reshape(-1, 1), pred_raw], axis=1)
                                         ], axis=0)
        return submission

    def plot(self):
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
        ax1.plot(self.train_losses, color='b')
        ax1.plot(self.test_losses, color='y')
        ax2.plot(self.metrics, color='y')
        plt.show()

    def save_model(self, name):
        torch.save(self.net, 'models/' + name)

    def load_model(self, name):
        self.net = torch.load('models/' + name)