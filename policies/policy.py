import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Policy(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def get_next(self, *args, **kwargs):
        pass
    
    def get_train(self, *args, **kwargs):
        pass
    
    def get_evo(self, *args, **kwargs):
        pass

class RandomPolicy(Policy):
    def __init__(self, output_shape, low=-1, high=1):
        self.output_shape = output_shape
        self.low = low
        self.high = high

    def get_next(self):
        return (lambda _ : np.random.uniform(self.low, self.high, self.output_shape))

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 2)

def mutate_weights(m, prob, mean=0, std=1):
    if type(m) == nn.Linear:
        x = np.random.normal(mean, std, m.weight.data.shape)
        r = np.random.uniform(0, 1, m.weight.data.shape)

        f = lambda x : 0 if x < prob else 1
        vf = np.vectorize(f)

        mutate_filter = vf(r)
        mutation = np.multiply(x, mutate_filter)
        m.weight.data += torch.FloatTensor(mutation)

        
class Evo_MLP(nn.Module, Policy):
    def __init__(self, input_shape, num_outputs, num_units=16):
        super(Evo_MLP, self).__init__()
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(input_shape, num_units)
        self.fc2 = nn.Linear(num_units, num_outputs)
        
        self.apply(init_weights)

    def forward(self, x):
        x = Variable(torch.FloatTensor(x))
        x = F.relu(self.fc1(x))
        y = F.tanh(self.fc2(x))
        return y
            
    def get_next(self):
        return (lambda x : self.forward(x))

    def get_evo(self, prob=0.9, mean=0, std=1):
        mutation_f = lambda m : mutate_weights(m, prob, mean, std)
        return (lambda : self.apply(mutation_f))

class CCEA(object):
   
    def __init__(self, population, fitness, update, selection):
        """
        :param population: 2D numpy array
        :param fitness: 1D numpy array of policies -> 1D numpy array of doubles of results
        """
        self.pop_size  = population.shape[0]
        self.team_size = population.shape[1]

        self.population = population
        self.update     = update
        self.selection  = selection

    def evolve(self):
        idx_n = []

        for _ in range(0, self.pop_size):
            idx = range(0, self.team_size)
            np.random.shuffle(idx)
            idx_n.append(idx)

        results = []
        for t in range(0, self.team_size):
            policies = []
            for p in range(0, self.pop_size):
                index = idx_n[p][t]
                policies.append(self.population[p][t])

            scores = self.fitness(np.array(policies))
            results.append(scores)

        results = results.transpose()

        for i, pool in enumerate(self.population):
            pool_results = results[i]
            self.population[i] = self.selection(pool, pool_results)
        
