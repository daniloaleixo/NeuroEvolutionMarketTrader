import numpy as np
import gym
from .functions import *
from time import sleep
import pickle
from random import random



# Lets define a new neural network class that can interact with gym
class NeuralNet():
    
    def __init__(self, n_units=None, parent1=None, parent2=None, var=0.02, episodes=50):

        # Testing if we need to copy a network
        if parent1 is None:
            # Saving attributes
            self.n_units = n_units
            self.genesis()

        elif parent2 is None:
            # Only one parent
            self.copy_parent(parent1)
            self.mutate(var)
        else:
            # Get architecture from either parent (they are the same)
            self.n_units = parent1.n_units
            self.crossover(parent1, parent2)
            self.mutate(var)
            
    def act(self, X):
        # Grabbing weights and biases
        weights = self.params['weights']
        biases = self.params['biases']
        # First propgating inputs
        a = relu((X@weights[0])+biases[0])
        # Now propogating through every other layer
        for i in range(1,len(weights)):
            a = relu((a@weights[i])+biases[i])
        # Getting probabilities by using the softmax function
        probs = softmax(a)
        return np.argmax(probs)
        
    # Defining the evaluation method
    def evaluate(self, env, episodes, render_env, record, max_env_steps=10_000, debug=False):
        # Creating empty list for rewards
        rewards = []
        # Recording video if we need to 
        if record is True:
            env = gym.wrappers.Monitor(env, "recording")
        for i_episode in range(episodes):
            
            observation = env.reset()
            cum_rewards = 0
            done = False
            count = 0

            # If the number of steps is larger than the max we choose a sample to evaluate inside the env
            if env.num_steps > max_env_steps:
                env.step_value = int(random() * (env.num_steps - max_env_steps - 1))

            while not done and count < max_env_steps:
                if render_env is True:
                    sleep(1e-2) # Time in seconds                
                    env.render()
                
                observation, r, done, _ = env.step(self.act(np.array(observation)), debug=debug)
                cum_rewards += r
                count += 1

            # penalize no trades
            if env.total_trades == 0: cum_rewards += env.no_trades_penalty 
                
            rewards.append(cum_rewards)
            

        # Closing our enviroment
        env.close()
        # Getting our final reward
        if len(rewards) == 0:
            return 0
        else:
            if debug: print(rewards)
            return np.array(rewards).mean()

    def genesis(self):
        '''
        First generation, create the weights for itself
        '''
        # Initializing empty lists to hold matrices
        weights = []
        biases = []
        # Populating the lists
        for i in range(len(self.n_units)-1):
            weights.append(np.random.normal(loc=0,scale=1,size=(self.n_units[i], self.n_units[i+1])))
            biases.append(np.zeros(self.n_units[i+1]))

        # Creating dictionary of parameters
        self.params = {'weights':weights,'biases':biases}


    def crossover(self, parent1, parent2):
        '''
        Crossover between two parents
        '''

        # Copy parent 1
        self.params = {
            'weights':np.copy(parent1.params['weights']),
            'biases':np.copy(parent1.params['biases'])
        }
        parent2 = {
            'weights':np.copy(parent2.params['weights']),
            'biases':np.copy(parent2.params['biases'])
        }

        # And then modify and include weights from parent 2
        for i in range(len(parent1.params['weights'])):
            for j in range(len(parent1.params['weights'][i])):
                for k in range(len(parent1.params['weights'][i][j])):
                    if random() > 0.5:
                        self.params['weights'][i][j][k] = parent2['weights'][i][j][k]
        for i in range(len(parent1.params['biases'])):
            for j in range(len(parent1.params['biases'][i])):
                if random() > 0.5:
                    self.params['biases'][i][j] = parent2['biases'][i][j]

    def copy_parent(self, copy_network):
        self.n_units = copy_network.n_units
        self.params = {'weights':np.copy(copy_network.params['weights']),
                      'biases':np.copy(copy_network.params['biases'])}


    def mutate(self, var):
        '''
        Mutate the weights
        '''
        self.params['weights'] = [x+np.random.normal(loc=0,scale=var,size=x.shape) for x in self.params['weights']]
        self.params['biases'] = [x+np.random.normal(loc=0,scale=var,size=x.shape) for x in self.params['biases']]
        
    def get_env_info(self, env, render_env):
        observation = env.reset()
        done = False

        while not done:
            if render_env is True:
                sleep(1e-2) # Time in seconds                
                env.render()
            
            a = self.act(np.array(observation))
            observation, r, done, _ = env.step(a)
            
        # Closing our enviroment
        env.close()

        return env.get_info()


    def load(self, path):
        f = open(path, "rb")
        info = pickle.load(f)
        n_units, params = info
        self.n_units = n_units
        self.params = {'weights': params['weights'],
                        'biases': params['biases']}
        f.close()

    def save(self, path):
        f = open(path + ".pkl", "wb")
        info = (self.n_units, self.params)
        pickle.dump(info, f)
        f.close()