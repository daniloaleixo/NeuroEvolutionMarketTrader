import math
import numpy as np
import pandas as pd
from core.technical_analysis import add_technical_indicators_with_intervals

class CNNDataLoader():

  def __init__(self, df_train, indicators=[], intervals=[], df_test=[], true_value=False, verbose=False):

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

    if len(df_test):
      self.test_dataframe = add_technical_indicators_with_intervals(
        self.test_dataframe, 
        indicators=indicators, 
        intervals=intervals,
        true_value=true_value,
        verbose=verbose
      )
      self.len_test = len(self.test_dataframe) - 1


  def get_test_data(self, cols, window_size=15):
    y_test_movement = (self.test_dataframe["Close"] - self.test_dataframe["Close"].shift()).dropna().values
    y_test = np.array(y_test_movement > 0).astype(int)

    x_test = self.test_dataframe.get(cols).values[:-1]
    x_test = x_test.reshape((x_test.shape[0], window_size, window_size))
    x_test = np.stack((x_test,) * 3, axis=-1)

    return self._handle_class_unbalacing(x_test, y_test, y_test_movement)

  def get_train_data(self, cols, window_size=15):
    y_train_mov = (self.dataframe["Close"] - self.dataframe["Close"].shift()).dropna().values
    y_train = np.array(y_train_mov > 0).astype(int)

    x_train = self.dataframe.get(cols).values[:-1]
    x_train = x_train.reshape((x_train.shape[0], window_size, window_size))
    x_train = np.stack((x_train,) * 3, axis=-1)

    return self._handle_class_unbalacing(x_train, y_train, y_train_mov)


  def generate_train_batch(self, cols, batch_size=256, window_size=48, stacked=False, classifier=True):
    '''Yield a generator of training data from filename'''
    i = 0 if not stacked else 2
    while i < (self.len_train):
      x_batch = []
      y_batch = []
      for b in range(batch_size):
        if i >= self.len_train:
          # stop-condition for a smaller final batch if data doesn't divide evenly
          yield np.array(x_batch), np.array(y_batch)
          i = 0
        x, y = self._next_window(cols, i, window_size=window_size, classifier=classifier)
        x_batch.append(x)
        y_batch.append(y)
        i += 1
      yield self._handle_class_unbalacing(x_batch, y_batch)

  def generate_test_batch(self, cols, batch_size, window_size=48, stacked=False, classifier=True):
    '''Yield a generator of test data from filename'''
    i = 0 if not stacked else 2
    while i < (self.len_test):
      x_batch = []
      y_batch = []
      for b in range(batch_size):
        if i >= self.len_test:
          # stop-condition for a smaller final batch if data doesn't divide evenly
          yield np.array(x_batch), np.array(y_batch)
          i = 0
        x, y = self._next_window(cols, i, train=False, window_size=window_size, classifier=classifier)
        x_batch.append(x)
        y_batch.append(y)
        i += 1
      yield self._handle_class_unbalacing(x_batch, y_batch)

  def _next_window(self, cols, i, train=True, window_size=48, stacked=False, classifier=True):
    '''Generates the next data window from the given index location i'''
    if train:
      if stacked:
        x_train1 = self.dataframe.iloc[i-2].get(cols).values
        x_train1 = x_train1.reshape((1, window_size, window_size))
        x_train2 = self.dataframe.iloc[i-1].get(cols).values
        x_train2 = x_train2.reshape((1, window_size, window_size))
        x_train3 = self.dataframe.iloc[i].get(cols).values
        x_train3 = x_train3.reshape((1, window_size, window_size))
        x_train = np.zeros((window_size, window_size, 3))
        x_train[:, :, 0] = x_train1[0]
        x_train[:, :, 1] = x_train2[0]
        x_train[:, :, 2] = x_train3[0]
      else:
        x_train = self.dataframe.iloc[i].get(cols).values
        x_train = x_train.reshape((1, window_size, window_size))
        x_train = np.stack((x_train,) * 3, axis=-1)[-1]

      y_train_mov = self.dataframe["Close"].iloc[i+1] - self.dataframe["Close"].iloc[i]
      y_train = int(y_train_mov > 0) if classifier else y_train_mov
      return x_train, y_train
    else:
      if stacked:
        x_test1 = self.test_dataframe.iloc[i-2].get(cols).values
        x_test1 = x_test1.reshape((1, window_size, window_size))
        x_test2 = self.test_dataframe.iloc[i-1].get(cols).values
        x_test2 = x_test2.reshape((1, window_size, window_size))
        x_test3 = self.test_dataframe.iloc[i].get(cols).values
        x_test3 = x_test3.reshape((1, window_size, window_size))
        x_test = np.zeros((window_size, window_size, 3))
        x_test[:, :, 0] = x_test1[0]
        x_test[:, :, 1] = x_test2[0]
        x_test[:, :, 2] = x_test3[0]
      else:
        x_test = self.test_dataframe.iloc[i].get(cols).values
        x_test = x_test.reshape((1, window_size, window_size))
        x_test = np.stack((x_test,) * 3, axis=-1)[-1]

      y_test_mov = self.test_dataframe["Close"].iloc[i+1] - self.test_dataframe["Close"].iloc[i]
      y_test = int(y_test_mov > 0) if classifier else y_test_mov
      return x_test, y_test




  def _handle_class_unbalacing(self, x, y, y_mov=[]):
    '''
    For up and down movements with have class unbalacing, this function makes the same 
    number of ups and downs
    '''
    size = 0
    y_mov_flag = len(y_mov) > 0

    # has more down than ups
    if np.sum(np.array(y == 0)) > np.sum(np.array(y == 1)):
      size = np.sum(np.array(y == 1))
    # has more upds than downs
    else:
      size = np.sum(np.array(y == 0))

    ups = 0
    downs = 0
    i = 0
    new_x, new_y, new_y_mov = [], [], []
    while ups < size or downs < size:
      if y[i] == 0 and downs < size:
        new_x.append(x[i])
        new_y.append(y[i])
        if y_mov_flag: new_y_mov.append(y_mov[i])
        downs += 1
      if y[i] == 1 and ups < size:
        new_x.append(x[i])
        new_y.append(y[i])
        if y_mov_flag: new_y_mov.append(y_mov[i])
        ups += 1
      i += 1


    if y_mov_flag: return np.array(new_x), np.array(new_y), np.array(new_y_mov)
    return np.array(new_x), np.array(new_y)