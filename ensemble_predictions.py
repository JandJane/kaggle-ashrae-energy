import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from config import DIR

# TODO train a separate model for ensembling

# predictions to ensemble
PREDS = [
    ('results/a.pkl', 'results/b.pkl'),
]

if __name__ == '__main__':
    results = []
    for preds in PREDS:
        to_ensemble = []
        for pred in preds:
            x = pickle.load(os.path.join(DIR, pred))
            to_ensemble.append(x)
        # TODO assert all the vectors have the same height
        ensembled = to_ensemble[0]
        ensembled[:, 1] = np.mean(
            np.concatenate([x[:, 1].reshape(-1, 1) for x in to_ensemble], axis=1),
            axis=1
        )
        results.append(ensembled)
        print('Length', ensembled.shape[0])

    results = np.concatenate(results, axis=0)
    df = pd.DataFrame(data=results)
    df.columns = ['row_id', 'meter_reading']
    df.row_id = df.row_id.astype(int)
    df.index = df.row_id
    df = df.drop(columns='row_id')
    print('Indices in resulting prediction %d, of which unique indices %d' % (len(df), len(df.index.unique())))
    print('Number of NaNs:', len(df[df.meter_reading.isna()]))

    baseline = pd.read_csv(os.path.join(DIR, 'results/baseline.csv'))
    baseline.loc[df.index, 'meter_reading'] = df.meter_reading
    baseline.index = baseline.row_id
    baseline = baseline.drop(columns='row_id')
    baseline.to_csv(os.path.join(DIR, 'results/submission_%s.csv' % datetime.today().strftime('%Y-%m-%d-%H-%M')))
