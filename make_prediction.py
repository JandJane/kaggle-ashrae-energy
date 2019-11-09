import os
import pickle
import time
import numpy as np

from nn import *
from config import models_config, DIR
from create_dataset import Preprocessor

PREPROCESSORS = ['models/preprocessor_2_2']  # must be "models/preprocessor_%d_%d"
MODELS = ['models/nn_2_2_2019-11-09-02-02.pth']
MODEL_TYPE = 'nn'

if __name__ == '__main__':
    for MODEL, PREPROCESSOR in zip(MODELS, PREPROCESSORS):
        print('Making prediciton for model %s with preprocessor %s' % (MODEL[7:], PREPROCESSOR[15:]))

        init_config = models_config[MODEL_TYPE]['init_config']

        if MODEL_TYPE == 'nn':
            preprocessor = pickle.load(open(os.path.join(DIR, PREPROCESSOR), 'rb'))
            scaler = Scaler(preprocessor)
            init_config['scaler'] = scaler
            trainer = NetTrainer(**init_config)
            trainer.load_model(MODEL)
            submission = np.array([]).reshape(-1, 2)
            t0 = time.time()
            submission = trainer.predict(preprocessor.df.loc[preprocessor.prod_idx], submission)
            print('Generated predictions in %d seconds' % (time.time() - t0))
            print('Samples prediction:', submission[10, 1], 'for row', submission[10, 0])
            print('Number of NaNs in prediction:', np.isnan(submission[:, 1]).sum())
            with open(os.path.join(DIR, 'results/prediction_%s.pkl' % MODEL[7:]), 'wb') as f:
                pickle.dump(submission, f)
