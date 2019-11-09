import os
import pickle
import time
from datetime import datetime

from config import models_config, DIR
from create_dataset import *
from nn import *

MODEL_TYPE = 'nn'
PREPROCESSORS = ['models/preprocessor_0_2']  # must be "models/preprocessor_%d_%d"

if __name__ == '__main__':
    for PREPROCESSOR in PREPROCESSORS:
        print('Training %s model with preprocessor %s' % (MODEL_TYPE, PREPROCESSOR[20:]))

        with open(os.path.join(DIR, PREPROCESSOR), 'rb') as f:
            preprocessor = pickle.load(f)

        init_config = models_config[MODEL_TYPE]['init_config']
        train_config = models_config[MODEL_TYPE]['train_config']

        if MODEL_TYPE == 'nn':
            scaler = Scaler(preprocessor)
            # scaler.dump_state('models/scaler_%s.pkl' % PREPROCESSOR[20:])
            init_config['scaler'] = scaler
            trainer = NetTrainer(**init_config)

        t0 = time.time()
        trainer.train(**train_config, verbose=True, do_val=True)
        print('Finished training. Train time %d seconds' % (time.time() - t0))

        model_name = '%s_%s_%s' % (MODEL_TYPE, PREPROCESSOR[20:], datetime.today().strftime('%Y-%m-%d-%H-%M'))
        trainer.plot(model_name)
        trainer.save_model(os.path.join(DIR, 'models/%s.pth' % model_name))
        with open(os.path.join(DIR, 'models/%s_config.txt' % model_name), 'w+') as f:
            f.write(str({**init_config, **train_config}))
            f.close()
