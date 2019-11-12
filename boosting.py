import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from matplotlib import pyplot as plt

from config import columns_config, device
from trainer import Trainer, CV


class CatBoostCV(CV):
    def __init__(self, preprocessor):
        super(CatBoostCV, self).__init__()
        self.df = preprocessor.df
        self.train_idx = preprocessor.train_idx

    @staticmethod
    def create_pool(X, y):
        return cb.Pool(X, y, list(range(len(columns_config['categorical']))))

    def iter_cv(self):
        X = self.df.loc[self.train_idx, columns_config['categorical'] + columns_config['numerical']]
        y = self.df.loc[self.train_idx, 'meter_reading']
        cv_groups = self.df.loc[self.train_idx, 'cv_group']

        logo = LeaveOneGroupOut()

        for train_idx, test_idx in logo.split(X, y, cv_groups):
            yield self.create_pool(X.loc[train_idx], y.loc[train_idx]), \
                  self.create_pool(X.loc[test_idx], y.loc[test_idx])


class CatBoostTrainer(Trainer):
    def __init__(self, trainpool, testpool, model_config=None):
        super(CatBoostTrainer, self).__init__()

        self.trainpool = trainpool
        self.testpool = testpool

        self.boosting = None
        self.create_model(model_config)

    def create_model(self, model_config):
        self.boosting = cb.CatBoostRegressor(**model_config)

    def train(self, do_val=True, verbose=True):
        self.boosting.fit(self.trainpool, eval_set=self.testpool, verbose=verbose, use_best_model=True)

    def plot(self, pic_name):
        eval_results = self.boosting.get_evals_result()
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(eval_results['learn']['RMSE'], color='b')
        ax.plot(eval_results['validation']['RMSE'], color='y')
        f.savefig('plots/%s.png' % pic_name)

    def predict(self, test_df, submission, batch_size=100000):
        for i in range(0, test_df.shape[0], batch_size):
            pool = CatBoostCV.create_pool(
                test_df[i: min(i + batch_size, test_df.shape[0])][columns_config['categorical'] + columns_config['numerical']],
                test_df[i: min(i + batch_size, test_df.shape[0])].meter_reading)
            row_ids = test_df.row_id[i: min(i + batch_size, test_df.shape[0])]
            pred_raw = self.boosting.predict(pool)
            pred_raw = np.exp(pred_raw) - 1
            submission = np.concatenate([submission,
                                         np.concatenate([row_ids.values.reshape(-1, 1), pred_raw.reshape(-1, 1)], axis=1)
                                         ], axis=0)
        return submission

    def save_model(self, name):
        self.boosting.save_model(name)

    def load_model(self, name):
        self.boosting.load_model(name)







