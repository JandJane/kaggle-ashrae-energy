import pickle
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")  # TODO filter warnings once

from config import *


METERS = [3]
FIRST = [True, False]
PREPARE_DATA = False

# TODO have all this in a separate utils file
# Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def select_meter(df, meter=1):
    df = df[df.meter == meter]
    return df


def filter_wind(weather_df):
    weather_df.loc[weather_df.wind_direction + weather_df.wind_speed == 0, ['wind_direction', 'wind_speed']] = np.NaN
    return weather_df


def merge(data, weather, meta):
    df = meta.merge(data, on='building_id')
    df = df.merge(weather, on=['site_id', 'timestamp'])
    return df


def filter_zero_targets(df):
    df = df[df.meter_reading != 0]
    return df


def create_new_features(df):
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df['month'] = df.timestamp.apply(lambda x: x.month)
    df['hour'] = df.timestamp.apply(lambda x: x.hour)
    df['weekday'] = df.timestamp.apply(lambda x: x.weekday())

    df['season'] = df['month'] % 12 // 3
    df['daytime'] = df['hour'] // 5
    return df


def filter_outliers(train_df):
    train_df = train_df.drop(index=train_df[train_df.meter == 0][train_df.meter_reading > 1145].index)
    train_df = train_df.drop(index=train_df[train_df.meter == 1][train_df.meter_reading > 4178].index)
    train_df = train_df.drop(index=train_df[train_df.meter == 2][train_df.meter_reading > 13125].index)
    train_df = train_df.drop(index=train_df[train_df.meter == 3][train_df.meter_reading > 2388].index)
    return train_df


def filter_shit(train_df):
    # site 0 meter 0 up to june
    train_df = train_df.drop(index=train_df[train_df.site_id == 0][train_df.meter == 0][train_df.month < 6].index)
    return train_df


def fill_nans(df):
    for col in columns_config['numerical']:
        if col == 'mean_target':
            continue
        df[col] = df[col].fillna(df[col].mean())
        df[col] = df[col].fillna(0)  # make sure we have no NaNs
    return df


def fill_infs(df):
    for col in columns_config['numerical']:
        if col == 'mean_target':
            continue
        df[col] = df[col].replace(np.inf, 0)
        df[col] = df[col].replace(-np.inf, 0)
    return df


def create_wind_cat(df):
    # TODO deal with 360 = 0
    df['wind_direction_cat'] = df['wind_direction'] // 45
    return df


def prepare_data(meter=1, fast_debug=False):
    meta = pd.read_csv('data/building_metadata.csv')
    train = pd.read_csv('data/train.csv')
    weather = pd.read_csv('data/weather_train.csv')

    train = train[train.building_id != 1099]
    train = filter_outliers(train)

    train = select_meter(train, meter)
    weather = filter_wind(weather)
    df = merge(train, weather, meta)

    if fast_debug:
        # building_ids = [1109, 1130, 1363, 1377]
        site_ids = np.random.choice(df.site_id.unique(), 2, replace=False)
        df = df[df.site_id.isin(site_ids)]

    df = create_new_features(df)
    df = filter_shit(df)
    return df


def prepare_test_data(meter=1):
    meta = pd.read_csv('data/building_metadata.csv')
    test = pd.read_csv('data/test.csv')
    weather = pd.read_csv('data/weather_test.csv')

    test = select_meter(test, meter)
    weather = filter_wind(weather)
    df = merge(test, weather, meta)

    df['meter_reading'] = np.NaN
    df = create_new_features(df)
    return df


def combine_train_test(df, test_df):
    df['row_id'] = np.NaN
    df_all = pd.concat([df, test_df], axis=0)
    df_all = df_all.reset_index()
    df_all = fill_nans(df_all)
    df_all = fill_infs(df_all)
    df_all = create_wind_cat(df_all)
    return df_all


class Preprocessor:
    def __init__(self, df, random_test_split=True, test_idx=None):
        self.df = df

        if not random_test_split:
            self.train_idx = \
                self.df[np.logical_and(~pd.isna(self.df.meter_reading), ~self.df.index.isin(test_idx))].index
        else:
            self.train_idx = self.train_test_split()
        self.test_idx = \
            self.df[np.logical_and(~pd.isna(self.df.meter_reading), ~self.df.index.isin(self.train_idx))].index
        self.prod_idx = self.df[pd.isna(self.df.meter_reading)].index

        self.df = self.create_mean_target()
        self.df = self.transform_target()

    def train_test_split(self):
        #         if 'row_id' in self.df.columns:
        #             train_idx = self.df[~pd.isna(self.df.meter_reading)].index
        #         else:
        train_idx = np.random.choice(self.df[~pd.isna(self.df.meter_reading)].index,
                                     len(self.df[~pd.isna(self.df.meter_reading)]) // 2, replace=False)
        return train_idx

    def create_mean_target(self):
        mean_targets = pd.DataFrame(
            data=self.df[self.df.index.isin(self.train_idx)].groupby('building_id').meter_reading.mean())
        mean_targets.columns = ['mean_target']

        self.df['tmp_index'] = self.df.index
        self.df.index = self.df.building_id
        self.df['mean_target'] = mean_targets
        self.df.index = self.df.tmp_index
        self.df.mean_target = self.df.mean_target.replace((np.inf, -np.inf, np.NaN), 0)

        return self.df

    def transform_target(self):
        self.df['meter_reading'] = np.log(self.df['meter_reading'] + 1)
        return self.df


if __name__ == '__main__':
    for meter in METERS:
        for first in FIRST:
            if PREPARE_DATA:
                t0 = time.time()
                df = prepare_data(meter=meter, fast_debug=False)
                df = reduce_mem_usage(df)
                print('Prepared df of length %d' % len(df))

                test_df = prepare_test_data(meter=meter)
                test_df = reduce_mem_usage(test_df)
                print('Prepared test df of length %d' % len(test_df))

                df_all = combine_train_test(df, test_df)
                df_all.to_pickle('data/prepared_data_%d.pkl' % meter)
                print('Saved prepared data. Elapsed time %.1f seconds' % (time.time() - t0))
            else:
                df_all = pd.read_pickle('data/prepared_data_%d.pkl' % meter)

            # Do train test split, create mean target column, log target
            if first:
                preprocessor = Preprocessor(df_all, random_test_split=True)
                with open('models/train_indices_%d_first.pkl' % meter,'wb') as f:
                    pickle.dump(preprocessor.train_idx, f)
                print('Saved %d train indices for meter %d part %d of dataset' %
                      (len(preprocessor.train_idx), meter, int(first) + 1))
            else:
                with open('models/train_indices_%d_first.pkl' % meter,'rb') as f:
                    x = pickle.load(f)
                preprocessor = Preprocessor(df_all, random_test_split=False, test_idx=x)

            # Dump preprocessor
            with open('models/'
                      'preprocessor_%d_%d' % (meter, int(not first) + 1), 'wb') as f:
                pickle.dump(preprocessor, f)

            print('---------------------------------------------------------------')
