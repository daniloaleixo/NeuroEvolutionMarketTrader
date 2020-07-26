from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow as tf
import os
import datetime as dt
import keras_metrics


def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class CNNModel():

  def __init__(self):
    self.model = Sequential()

  def load_weights(self, path):
    self.model.load_weights(path)

  def load_model(self, filepath):
    print('[Model] Loading model from file %s' % filepath)
    self.model = tf.keras.models.load_model(filepath, custom_objects={
      "f1_score": f1_score,
      "binary_precision": keras_metrics.precision(),
      "binary_recall": keras_metrics.recall(),
    })

  def create_model_cnn(self, params, input_shape):
    '''
    Create CCN model
    '''
    for layer in params['layers']:
      neurons = layer['neurons'] if 'neurons' in layer else None
      dropout_rate = layer['rate'] if 'rate' in layer else None
      activation = layer['activation'] if 'activation' in layer else None
      return_seq = layer['return_seq'] if 'return_seq' in layer else None
      input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
      input_dim = layer['input_dim'] if 'input_dim' in layer else None
      kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
      strides = layer['strides'] if 'strides' in layer else None
      padding = layer['padding'] if 'padding' in layer else None
      use_bias = layer['use_bias'] if 'use_bias' in layer else None
      pool_size = layer['pool_size'] if 'pool_size' in layer else None

      if layer['type'] == 'dense':
        if input_dim: self.model.add(Dense(neurons, activation=activation, input_shape=(input_dim,)))
        else: self.model.add(Dense(neurons, activation=activation))
      if layer['type'] == 'lstm':
        self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
      if layer['type'] == 'Conv2D':
        self.model.add(Conv2D(neurons, kernel_size, strides=strides, input_shape=input_shape, padding=padding, activation=activation, use_bias=use_bias, kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2, seed=None),kernel_regularizer=l2(2e-4)))
      if layer['type'] == 'MaxPool2D':
        self.model.add(MaxPool2D(pool_size=pool_size))
      if layer['type'] == 'BatchNormalization':
        self.model.add(BatchNormalization())
      if layer['type'] == 'flatten':
        self.model.add(Flatten())
      if layer['type'] == 'dropout':
        self.model.add(Dropout(dropout_rate))

    # Optimizer
    if params["optimizer"] == 'rmsprop': 
      optimizer = optimizers.RMSprop(lr=params["lr"])
    elif params["optimizer"] == 'sgd':
      optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
    elif params["optimizer"] == 'adam':
      optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)

    metrics = []
    for metric in params['metrics']:
      if metric == 'accuracy': metrics.append(metric)
      if metric == 'f1_score': metrics.append(f1_score)
      if metric == 'precision': metrics.append(keras_metrics.precision())
      if metric == 'recall': metrics.append(keras_metrics.recall())

    self.model.compile(loss=params['loss'], optimizer=optimizer, metrics=metrics)
    self.model.summary()


  def train(self, x, y, epochs=3000, batch_size=64, save_dir='', x_val=[], y_val=[], verbose=True):
    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
    
    validation_data = (x_val, y_val) if len(x_val) and len(y_val) else None
    save_fname = os.path.join(save_dir, 'cnn-%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

    callbacks = [
      ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True, save_weights_only=False),
      ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.0001)
    ]
    hist = self.model.fit(
      x,
      y,
      epochs=epochs,
      batch_size=batch_size,
      # callbacks=callbacks,
      validation_data=validation_data,
      verbose=verbose,
      shuffle=True,
    )
    self.model.save(save_fname)

    print('[Model] Training Completed. Model saved as %s' % save_fname)
    return hist

  def train_generator(self, data_gen, epochs=3000, batch_size=64, steps_per_epoch=0, save_dir='', validation_data=None,     validation_steps=None):
    '''
    Training out of memory
    '''
    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
    
    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
      ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                        min_delta=0.001, cooldown=1, min_lr=0.0001)
    ]
    hist = self.model.fit_generator(
      data_gen,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      callbacks=callbacks,
      workers=1,
      validation_data=validation_data,
      validation_steps=validation_steps
    )
    
    print('[Model] Training Completed. Model saved as %s' % save_fname)
    return hist

  def predict(self, x, verbose=True):
    return self.model.predict(x, verbose=verbose)

  def predict_generator(self, data_gen, steps=0):
    predicted = self.model.predict_generator(
      data_gen,
      steps=steps,
      verbose=1
    )
    return predicted

  def predict_next_candle(self, x, last_close):
    predictions = self.predict(x)
    return predictions[-1, 0] + last_close