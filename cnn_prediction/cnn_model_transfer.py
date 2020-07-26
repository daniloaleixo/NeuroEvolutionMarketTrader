from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
import tensorflow.keras as keras
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

class CNNModelTransfer():

  def __init__(self, input_shape, layers=[]):
    new_input = Input(shape=input_shape)
    self.base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

    # add a global spatial average pooling layer
    x = self.base_model.output
    x = GlobalAveragePooling2D()(x)

    for layer in layers:
      neurons = layer['neurons'] if 'neurons' in layer else None
      dropout_rate = layer['rate'] if 'rate' in layer else None
      activation = layer['activation'] if 'activation' in layer else None
      return_seq = layer['return_seq'] if 'return_seq' in layer else None
      input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
      input_dim = layer['input_dim'] if 'input_dim' in layer else None

      if layer['type'] == 'dense':
        x = Dense(neurons, activation=activation)(x)
      if layer['type'] == 'dropout':
        x = Dropout(dropout_rate)(x)

    # this is the model we will train
    self.model = Model(inputs=self.base_model.input, outputs=x)
    self.model.summary()

  def load_weights(self, path):
    self.model.load_weights(path)

  def freeze_top_layers(self):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in self.base_model.layers:
      layer.trainable = False

  def unfreeze_layers(self, num_layers_to_freeze=100):
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in self.model.layers[:num_layers_to_freeze]:
      layer.trainable = False
    for layer in self.model.layers[num_layers_to_freeze:]:
      layer.trainable = True

  def compile_model(self, lr=0.001):
    optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    self.model.compile(
      loss='sparse_categorical_crossentropy', 
      optimizer=optimizer,
      metrics=['accuracy', f1_score, keras_metrics.precision(), keras_metrics.recall()]
    )

  def compile_model_for_fine_tuning(self, lr=0.0001):
    # we need to recompile the model for these modifications to take effect
    self.model.compile(
      optimizer=optimizers.SGD(lr=lr, momentum=0.9), 
      loss='sparse_categorical_crossentropy', 
      metrics=['accuracy', f1_score, keras_metrics.precision(), keras_metrics.recall()]
    )

  def print_model_layers(self):
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(self.base_model.layers):
      print(i, layer.name)
  
  def train(self, x, y, epochs=3000, batch_size=64, save_dir='', x_val=[], y_val=[]):
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
      callbacks=callbacks,
      validation_data=validation_data,
      verbose=1,
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
    timer.stop()
    return hist


  def predict(self, x, verbose=True):
    return self.model.predict(x, verbose=verbose)

  def predict_generator(self, data_gen, steps=0):
    return self.model.predict_generator(
      data_gen,
      steps=steps,
      verbose=1
    )

  def predict_next_candle(self, x, last_close):
    predictions = self.predict(x)
    return predictions[-1, 0] + last_close