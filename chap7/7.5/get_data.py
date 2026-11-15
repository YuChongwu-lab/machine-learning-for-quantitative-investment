import pandas as pd
import numpy as np
from paths import data_file


def get_data():
    raw_data = pd.read_csv(data_file('data_ml.csv'), index_col='stock_id')
    features = raw_data.iloc[:,1:-4]
    label = raw_data.iloc[:,-4]
    data = np.hstack([features, label.values.reshape(-1,1)])
    data = pd.DataFrame(data).dropna()
    return data
