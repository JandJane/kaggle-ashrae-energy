import pickle
import time
import numpy as np

from nn import *
from config import models_config

PREPROCESSOR = 'models/..'  # must be "models/preprocessor_%d_%d"
MODEL = 'models/..'
MODEL_TYPE = 'nn'
SCALER = 'models/scaler_'

if __name__ == '__main__':
    init_config = models_config[MODEL_TYPE]['init_config']

    if MODEL_TYPE == 'nn':
        preprocessor = pickle.load(PREPROCESSOR)
        scaler = Scaler(preprocessor.df, from_state=True, state_path=SCALER)
        init_config['scaler'] = scaler
        trainer = NetTrainer(**init_config)
        trainer.load_model(MODEL)
        submission = np.array([]).reshape(-1, 2)
        t0 = time.time()
        submission = trainer.predict(preprocessor.df[preprocessor.prod_idx], submission)
        print('Generated predictions in %d seconds' % (time.time() - t0))
        print('Samples prediction:', submission[10, 1], 'for row', submission[10, 0])
        with open('results/prediction_%s.pkl' % MODEL[7:], 'wb') as f:
            pickle.dump(submission, f)
