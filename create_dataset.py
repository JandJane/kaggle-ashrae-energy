import pickle
import pandas as pd
import time
import warnings
from data_preprocessing_utils import *

warnings.filterwarnings("ignore")  # TODO filter warnings once


METERS = [1, 2, 3]
N_FOLDS = 5
PREPARE_DATA = False

if __name__ == '__main__':
    for meter in METERS:
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
        preprocessor = Preprocessor(df_all, n_folds=N_FOLDS)

        print(preprocessor.df.cv_group.value_counts())

        # Dump preprocessor
        with open('models/preprocessor_%d' % meter, 'wb') as f:
            pickle.dump(preprocessor, f)

        print('---------------------------------------------------------------')
