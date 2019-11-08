import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import pickle

from config import columns_config
from trainer import Trainer

device = 'cpu'


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
    def __init__(self, preprocessor, batch_size=1024, from_state=False, state_path=None):
        self.df = preprocessor.df
        self.train_idx = preprocessor.train_idx

        self.df.wind_direction_cat = self.df.wind_direction_cat.fillna(0)  # TODO delete this once datasets are regenerated
        self.df.wind_direction = self.df.wind_direction.fillna(0)

        if from_state:
            state = pickle.load(state_path)
            self.scaler_labels = state['scaler_labels']
            self.scaler_features = state['scaler_features']
            self.encoder = state['encoder']
            self.testloader = state['testloader']
            self.trainloader = state['trainloader']
            self.d_in = state['d_in']
        else:
            self.scaler_labels = StandardScaler()
            self.scaler_features = StandardScaler()
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)

            self.create_scalers()

            cat_train, num_train, labels_train = self.transform(self.df.loc[preprocessor.train_idx])
            cat_test, num_test, labels_test = self.transform(self.df.loc[preprocessor.test_idx])
            self.d_in = cat_train.shape[1] + num_train.shape[1]

            #         if prod:
            #             self.testloader = torch.FloatTensor(np.concatenate([cat_test, num_test], axis=1)), self.df[~self.df.index.isin(self.train_idx)].row_id
            #         else:
            self.testloader = self.create_dataloader(cat_test, num_test, labels_test, batch_size)
            self.trainloader = self.create_dataloader(cat_train, num_train, labels_train, batch_size)

    def dump_state(self, path):
        state = {
            'scaler_labels': self.scaler_labels,
            'scaler_features': self.scaler_features,
            'encoder': self.encoder,
            'testloader': self.testloader,
            'trainloader': self.trainloader,
            'd_in': self.d_in
        }
        pickle.dump(state, open(path, 'wb'))

    def create_scalers(self):
        self.scaler_features.fit(self.df.loc[self.train_idx, columns_config['numerical']])
        self.scaler_labels.fit(self.df.loc[self.train_idx, 'meter_reading'].values.reshape(-1, 1).astype(np.float))
        self.encoder.fit(self.df.loc[self.train_idx, columns_config['categorical']])

    def transform(self, data):
        num_features = self.scaler_features.transform(data[columns_config['numerical']])
        labels = self.scaler_labels.transform(data.loc[:, 'meter_reading'].values.reshape(-1, 1).astype(np.float))
        cat_features = self.encoder.transform(data[columns_config['categorical']])
        print('NaNs in scaled arrays:', np.isnan(labels).sum(), np.isnan(num_features).sum(),
              np.isnan(cat_features.todense()).sum())
        return cat_features, num_features, labels

    @staticmethod
    def create_dataloader(cat, num, labels, batch_size, shuffle=True, add_row_ids=False, row_ids=None):
        data = [cat.todense(), num]
        if add_row_ids:
            data.append(row_ids.astype(int))
        dataset = TensorDataset(torch.Tensor(np.concatenate(data, 1)), torch.Tensor(labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader


class NetTrainer(Trainer):
    def __init__(self, scaler=None, net_config=None, lr=0):
        super(Trainer, self).__init__()

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
        loss = np.mean((pred_raw - labels_raw) ** 2) ** 0.5
        return loss

    def train(self, n_epochs=0, verbose=True, do_val=True):
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
                    loss = self.criterion(outputs, labels)
                    losses.append(loss.item())
                    metrics.append(self.metric(outputs, labels))
                if verbose:
                    print('[%d] Test loss: %.3f' % (epoch + 1, np.mean(losses)))
                    print('[%d] Test metric: %.3f' % (epoch + 1, np.mean(metrics)))
                self.test_losses.append(np.mean(losses))
                self.metrics.append(np.mean(metrics))

    def predict(self, test_df, submission, batch_size=100000):
        self.net.eval()
        for i in range(0, test_df.shape[0], batch_size):
            cat_test, num_test, labels_test = self.scaler.transform(test_df[i: min(i + batch_size, test_df.shape[0])])
            inputs = torch.FloatTensor(np.concatenate([cat_test.todense(), num_test], axis=1)).to(device)
            row_ids = test_df.row_id[i: min(i + batch_size, test_df.shape[0])]
            with torch.no_grad():
                outputs = self.net(inputs)
            pred_raw = self.scaler_labels.inverse_transform(outputs.detach().cpu().numpy())
            pred_raw = np.exp(pred_raw) - 1
            submission = np.concatenate([submission,
                                         np.concatenate([row_ids.values.reshape(-1, 1), pred_raw], axis=1)
                                         ], axis=0)
        return submission

    def plot(self):  # TODO
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
        ax1.plot(self.train_losses, color='b')
        ax1.plot(self.test_losses, color='y')
        ax2.plot(self.metrics, color='y')
        plt.show()

    def save_model(self, path):
        torch.save(self.net, path)

    def load_model(self, path):
        self.net = torch.load(path)
