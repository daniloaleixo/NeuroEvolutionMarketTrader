import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, cols, split=0, 
                start_train_index=0, end_train_index=0, start_test_index=0, end_test_index=0,
                test_file=None, detrend=False):

        self.dataframe = pd.read_csv(filename, index_col=0)
        if test_file: self.test_dataframe = pd.read_csv(test_file, index_col=0)

        # Remove trends
        if detrend:
            self.dataframe = self.detrend(self.dataframe)
            if test_file: self.test_dataframe = self.detrend(self.test_dataframe)


        # Se passamos o split vamos separar pelo indice
        if (split):
            i_split = int(len(self.dataframe) * split)
            self.data_train = self.dataframe.get(cols).values[:i_split]
            self.data_test  = self.dataframe.get(cols).values[i_split:]
        else:
            # Se não a gente faz pelos indices
            self.data_train = self.dataframe.get(cols).loc[(self.dataframe.index > start_train_index) & (self.dataframe.index < end_train_index)].values
            if test_file:
                self.data_test = self.test_dataframe.get(cols).loc[(self.test_dataframe.index > start_test_index) & (self.test_dataframe.index < end_test_index)].values
            else:
                self.data_test = self.dataframe.get(cols).loc[(self.dataframe.index > start_test_index) & (self.dataframe.index < end_test_index)].values
        

        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise, debug=False, only_close=False):
        '''
        Create x, y test data windows
        When passing only_close=False we generate another output sized as input 
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        real_data = data_windows
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        if not only_close: y = data_windows[:, -1]

        return x, y, real_data[:, -1, [0]], real_data[:, 0, [0]], real_data[:, -2, [0]]

    def get_train_data(self, seq_len, normalise, only_close=False):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        data_close = []
        first_close = []
        last_close = []
        for i in range(self.len_train - seq_len):
            x, y, close, f_close, l_close = self._next_window(i, seq_len, normalise, only_close=only_close)
            data_x.append(x)
            data_y.append(y)
            data_close.append(close)
            first_close.append(f_close)
            last_close.append(l_close)
        return np.array(data_x), np.array(data_y), np.array(data_close), np.array(first_close), np.array(last_close)


    # def get_train_data_files(self, seq_len, normalise):
    #     '''
    #     Create x, y train data windows
    #     Warning: batch method, not generative, make sure you have enough memory to
    #     load data, otherwise use generate_training_window() method.
    #     '''

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            data_close = []
            first_close = []
            last_close = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y, close, f_close, l_close = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                data_close.append(close)
                first_close.append(f_close)
                last_close.append(l_close)
                i += 1
            yield np.array(x_batch), np.array(y_batch), np.array(data_close), np.array(first_close), np.array(last_close)
            
    def generate_test_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of test data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_test - seq_len):
            x_batch = []
            y_batch = []
            data_close = []
            first_close = []
            last_close = []
            for b in range(batch_size):
                if i >= (self.len_test - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y, close, f_close, l_close = self._next_window(i, seq_len, normalise, False)
                x_batch.append(x)
                y_batch.append(y)
                data_close.append(close)
                first_close.append(f_close)
                last_close.append(l_close)
                i += 1
            yield np.array(x_batch), np.array(y_batch), np.array(data_close), np.array(first_close), np.array(last_close)

    def _next_window(self, i, seq_len, normalise, train=True, only_close=False):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len] if train else self.data_test[i:i+seq_len]
        real_data = window
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1] # Remove the last interation which is what we have to predict
        y = window[-1, [0]] # Get the prediction
        if not only_close: y = window[-1]
        return x, y, real_data[-1, [0]], real_data[0, [0]],  real_data[-2, [0]]

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / (float(window[0, col_i]) + 0.00000001) ) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)


    def get_agent_data(self, seq_len, lstm_x_train, predictions_train, cols=[6]):
        '''
        Get all the data we'll use in the agent data
        '''
        train_data = self._mount_agent_data(lstm_x_train, predictions_train, cols)
        x_train = self._create_agent_batches(seq_len, lstm_x_train, train_data, cols)

        return x_train


    def _mount_agent_data(self, x, predictions, cols=[6]):
        '''
        Get (close, prediction) tuple 
        '''
        data = np.zeros((len(x), cols))
        data[:, 0] = x[:,-1, 0]               # Close
        data[:, 1] = x[:,-1, 1]               # Open
        data[:, 2] = x[:,-1, 2]               # High
        data[:, 3] = x[:,-1, 3]               # Low
        # data[:, 4] = x[:,-1, 4]               # Volume

        for i in cols:
          data[:, i] = predictions[i]              # Predictions
        return data

    def _create_agent_batches(self, seq_len, lstm_x, ohlcv, num_cols=6):
        x_train = np.zeros((len(lstm_x) - seq_len, seq_len - 1, num_cols))
        for i in range(len(lstm_x) - seq_len):
            x_train[i, :] = ohlcv[i:i+seq_len-1]
        return x_train

    def get_y_train_data(self, seq_len, normalise, close_col=0, debug=False):
        '''
        Get y train data in memory
        (Doing this because x is too large and I just need the y values)
        '''
        y = []
        for i in range(self.len_train - seq_len):
            if debug: print("Mounting y ", i + 1, " of ", self.len_train - seq_len)
            y.append(self.data_train[i+seq_len][0])
    
        y = np.array(y)
        
        first_value = y[0]
        y = np.true_divide(y, first_value) if normalise else y # Normalize

        if debug: print("Finished y")
        return y

    def get_y_test_data(self, seq_len, normalise, close_col=0, debug=False):
        '''
        Get y test data in memory
        (Doing this because x is too large and I just need the y values)
        '''
        y = []
        for i in range(self.len_test - seq_len):
            if debug: print("Mounting y ", i + 1, " of ", self.len_test - seq_len)
            y.append(self.data_test[i+seq_len][0])
    
        y = np.array(y)
        
        first_value = y[0]
        y = np.true_divide(y, first_value) if normalise else y # Normalize

        if debug: print("Finished y")
        return y
            

    def detrend(self, df):
        df = df.shift() - df
        return df.dropna()
