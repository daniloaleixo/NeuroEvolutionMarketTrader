import math
import numpy as np
import pandas as pd
from core.technical_analysis import add_technical_indicators_with_intervals
from tqdm import tqdm

class CNNDataLoaderBuySell():

  def __init__(self, df_train, indicators=[], intervals=[], df_test=[], true_value=False, 
  verbose=False, window_size=10):

    self.dataframe = df_train
    if len(df_test): self.test_dataframe = df_test

    self.dataframe = add_technical_indicators_with_intervals(
      self.dataframe, 
      indicators=indicators, 
      intervals=intervals,
      true_value=true_value,
      verbose=verbose
    )
    self.len_train = len(self.dataframe) - 1
    self.train_labels = self._generate_buy_sell_labels(self.dataframe['Close'].values, window_size=window_size)

    if len(df_test):
      self.test_dataframe = add_technical_indicators_with_intervals(
        self.test_dataframe, 
        indicators=indicators, 
        intervals=intervals,
        true_value=true_value,
        verbose=verbose
      )
      self.len_test = len(self.test_dataframe) - 1
      self.test_labels = self._generate_buy_sell_labels(self.test_dataframe['Close'].values, window_size=window_size)


  def get_test_data(self, cols, window_size=15, class_unbalacing=True):
    y_test = self.test_labels

    x_test = self.test_dataframe.get(cols).values
    x_test = x_test.reshape((x_test.shape[0], window_size, window_size))
    x_test = np.stack((x_test,) * 3, axis=-1)

    if class_unbalacing: return self._handle_class_unbalacing(x_test, y_test)
    return np.array(x_test), np.array(y_test)

  def get_train_data(self, cols, window_size=15, class_unbalacing=True):
    y_train = self.train_labels

    x_train = self.dataframe.get(cols).values
    x_train = x_train.reshape((x_train.shape[0], window_size, window_size))
    x_train = np.stack((x_train,) * 3, axis=-1)


    if class_unbalacing: return self._handle_class_unbalacing(x_train, y_train)
    return np.array(x_train), np.array(y_train)

  def _generate_buy_sell_labels(self, close, window_size=10):
    # Generate data labels
    labels = np.zeros((len(close), ))
    for i in range(len(close) - window_size):
      if close[i] < close[i+1:i+window_size].min(): 
          labels[i] = 1
      if close[i] > close[i+1:i+window_size].max(): 
          labels[i] = 2
    return labels


  def _handle_class_unbalacing(self, x, y):
    '''
    For up and down movements with have class unbalacing, this function makes the same 
    number of ups and downs
    '''
    min_size = 9999999999
    num_labels = int(np.max(y)) + 1

    for label in [i for i in range(num_labels)]:
      if min_size > np.sum(np.array(y == label)): min_size = np.sum(np.array(y == label))

    labels_count = np.zeros((num_labels, ))
    i = 0
    new_x, new_y = [], []

    tqdm_e = tqdm(range(len(y)))
    while np.sum(labels_count) < num_labels * min_size:
      if labels_count[int(y[i])] < min_size:
        new_x.append(x[i])
        new_y.append(y[i])
        labels_count[int(y[i])] += 1

      i += 1
      if i > 0 and i % 10000 == 0:
        tqdm_e.update(10000)
        tqdm_e.refresh()
    tqdm_e.close()

    return np.array(new_x), np.array(new_y)