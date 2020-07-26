import pandas as pd
import numpy as np


def detrend_array(array):
  a = pd.DataFrame(array)
  a = a - a.shift(1)
  a = a.dropna()
  return a.values


def make_data_great_again(data, detrend_cols, divide_100, num_cols, detrend_min=0.1):
  data_detrend = np.zeros((data.shape[0] - 1, data.shape[1]))
  for i in range(0, num_cols):
    if i in detrend_cols: data_detrend[:, i] = np.add(detrend_array(data[:, i]), detrend_min)[:, 0]
    elif i in divide_100: data_detrend[:, i] = data[:1, i] / 100.0
    else: data_detrend[:, i] = data[:1, i]
  return data_detrend

    