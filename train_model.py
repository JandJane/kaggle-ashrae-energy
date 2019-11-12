import os
import pickle
import time
from datetime import datetime

from config import models_config, DIR
from data_preprocessing_utils import Preprocessor
from nn import *
from boosting import *

MODEL_TYPE = 'cb'
PREPROCESSORS = ['models/preprocessor_3']  # must be "models/preprocessor_%d"

if __name__ == '__main__':
    for PREPROCESSOR in PREPROCESSORS:
        print('Training %s model with preprocessor %s' % (MODEL_TYPE, PREPROCESSOR[20:]))

        with open(os.path.join(DIR, PREPROCESSOR), 'rb') as f:
            preprocessor = pickle.load(f)

        init_config = models_config[MODEL_TYPE]['init_config']
        train_config = models_config[MODEL_TYPE]['train_config']

        if MODEL_TYPE == 'nn':
            cv = Scaler(preprocessor)
            init_config['scaler'] = cv
            TrainClass = NetTrainer
        elif MODEL_TYPE == 'cb':
            cv = CatBoostCV(preprocessor)
            TrainClass = CatBoostTrainer

        i_group = 0
        for trainloader, testloader in cv.iter_cv():
            trainer = TrainClass(trainloader, testloader, **init_config)

            t0 = time.time()
            trainer.train(**train_config, verbose=True, do_val=True)
            print('Finished training. Train time %d seconds' % (time.time() - t0))

            model_name = '%s_%s_%d_%s' % \
                         (MODEL_TYPE, PREPROCESSOR[20:], i_group, datetime.today().strftime('%Y-%m-%d-%H-%M'))
            trainer.plot(model_name)
            trainer.save_model(os.path.join(DIR, 'models/%s.pth' % model_name))
            with open(os.path.join(DIR, 'models/%s_config.txt' % model_name), 'w+') as f:
                f.write(str({**init_config, **train_config}))
                f.close()

            i_group += 1
