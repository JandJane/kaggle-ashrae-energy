import pickle
import time
from datetime import datetime

from config import models_config
from create_dataset import *
from nn import *

MODEL_TYPE = 'nn'
PREPROCESSOR = 'models/preprocessor_0_2'  # must be "models/preprocessor_%d_%d"
SCALER = None


if __name__ == '__main__':
    with open(PREPROCESSOR, 'rb') as f:
        preprocessor = pickle.load(f)

    init_config = models_config[MODEL_TYPE]['init_config']
    train_config = models_config[MODEL_TYPE]['train_config']

    if MODEL_TYPE == 'nn':
        if SCALER:
            scaler = Scaler(preprocessor, from_state=True, state_path=SCALER)
        else:
            scaler = Scaler(preprocessor)
            # scaler.dump_state('models/scaler_%s.pkl' % PREPROCESSOR[20:])
        init_config['scaler'] = scaler
        trainer = NetTrainer(**init_config)

    t0 = time.time()
    trainer.train(**train_config, verbose=True, do_val=True)
    print('Finished training. Train time %d seconds' % (time.time() - t0))

    model_name = '%s_%s_%s' % (MODEL_TYPE, PREPROCESSOR[20:], datetime.today().strftime('%Y-%m-%d-%H-%M'))
    trainer.plot(model_name)
    trainer.save_model('models/%s.pth' % model_name)
    with open('models/%s_config.txt' % model_name, 'w+') as f:  # TODO just rewrite all these ugly 'with' constructions
        f.write(str({**init_config, **train_config}))
        f.close()
