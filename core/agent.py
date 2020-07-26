from core.utils import Timer

from core.memory import PER

import keras.backend as K
from keras.utils.generic_utils import get_custom_objects

import keras
from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

import random
from collections import deque
import numpy as np
import tensorflow as tf



def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links:     https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    def __init__(self, configs, state_size, action_size, learning_rate, gamma=0.95, 
                 explore_start=1.0, explore_stop=0.01, decay_rate=0.00007, possible_actions = [], name='Agent', 
                 is_eval=False, memory_size=1000, reset_every=1000, load_model_name=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        
        self.memory = PER(memory_size)


        self.is_eval = is_eval
        
        self.gamma = gamma
        
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.decay_step = 0
        self.explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * self.decay_step)
        
        self.possible_actions = possible_actions

        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        get_custom_objects().update({ "huber_loss": huber_loss })


        # If is eval 
        if is_eval:
            self.model = load_model("saved_models/" + name)
        else:
            # Se não quero fazer o load
            if load_model_name == None:
                self.model = self._model(configs)
            else:
                print("Loading pretrained model to train a lil more")
                self.model = load_model("saved_models/" + load_model_name)
        self.model.summary()
        

        # double q
        self.n_iter = 1
        self.reset_every = reset_every

        # target network
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        
    def _model(self, configs):
        model = Sequential()

        for layer in configs['agent']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            is_input = layer['is_input'] if 'is_input' in layer else None
            is_output = layer['is_output'] if 'is_output' in layer else None

            # Dense
            if layer['type'] == 'dense':
                if is_input:
                    # Input layer
                    if isinstance(self.state_size, int): model.add(Dense(neurons, input_dim=self.state_size, activation=activation))
                    else: model.add(Dense(neurons, input_shape=self.state_size, activation=activation))
                elif is_output:
                    # Output layer
                    model.add(Dense(self.action_size, activation=activation))
                else:
                    model.add(Dense(neurons, activation=activation))

            # LSTM
            if layer['type'] == 'lstm':
                model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))

            # Dropout
            if layer['type'] == 'dropout':
                model.add(Dropout(dropout_rate))
                

            # Flatten
            if layer['type'] == 'flatten':
                if not isinstance(self.state_size, int): model.add(Flatten())


        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model
    
    def _predict_action(self, data):
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_action(self, state):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        self.explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)

        # print("comparar ", self.explore_probability > exp_exp_tradeoff, self.explore_probability, exp_exp_tradeoff)
        if not self.is_eval and self.explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = np.argmax(random.choice(self.possible_actions))
            # print("random ", action)
        else:
            predicted = self._predict_action(state)
            # print("array predicted ", predicted)
            action = np.argmax(predicted)
            # print("predicted ", action)

        return action, self.explore_probability



    #
    # Memory
    #


    def memorize(self, state, action, reward, next_state, done, loss):
        self.memory.add((state, action, reward, next_state, done), loss)

    def replay(self, batch_size):
        mini_batch = self.memory.batch(batch_size)
        X_train, y_train = [], []


        # Reset target model weights
        if self.n_iter % self.reset_every == 0:
            self.target_model.set_weights(self.model.get_weights())
        else: 
            self.n_iter += 1

            
        for state, action, reward, next_state, done in mini_batch:
            if done:
                    target = reward
            else:
                 # approximate double deep q-learning equation
                t = self.target_model.predict(next_state)[0]
                target = reward + self.gamma * t[np.argmax(self.model.predict(next_state)[0])]


            # estimate q-values based on current state
            q_values = self.model.predict(state)

            # update the target for current action based on discounted reward
            q_values[0][action] = target

            X_train.append(state[0])
            y_train.append(q_values[0])
            
        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]
        
        self.decay_step += 1
            
        if not self.is_eval and self.explore_probability > self.explore_stop:
            self.explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)

        return loss