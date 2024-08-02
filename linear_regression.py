import numpy as np


def compute_RMSE(y_true, y_pred):

    squared_diff = np.square(y_true - y_pred)
    
    mean_squared_diff = np.mean(squared_diff)
    
    rmse = np.sqrt(mean_squared_diff)

    return rmse
