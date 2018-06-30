import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from torch import Tensor

class Policy(object):
    def __init__(self, *args, **kwargs):
        pass
    
    def get_next(self, state, *args, **kwargs):
        pass
    
    def get_train(self, *args, **kwargs):
        pass
    

class RandomPolicy(Policy):
    def __init__(self, output_shape, low=-1, high=1):
        self.output_shape = output_shape
        self.low = low
        self.high = high

    def get_next(self):
        return np.random.uniform(self.low, self.high, self.output_shape)
        
class Evo_MLP(nn.Module, Policy):
    def __init__(self, input_shape, num_outputs, num_units=16):
        super(Evo_MLP, self).__init__()
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(input_shape, num_units)
        self.fc2 = nn.Linear(num_units, num_outputs)
        for param in self.parameters():
            param.requires_grad = False

    def get_next(self, state):
        x = Variable(torch.FloatTensor(state))
        x = F.relu(self.fc1(x))
        y = F.tanh(self.fc2(x))
        return y

    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 2)

    def mutate(self):
        random_w1 = np.random.uniform(0.8, 1.2, list(self.fc1.weight.size()))
        random_w2 = np.random.uniform(0.8, 1.2, list(self.fc2.weight.size()))
        self.fc1.weight *= Tensor(random_w1)
        self.fc2.weight *= Tensor(random_w2)

def CCEA(population, fitness, retain=0.2):
    '''
    Implements the basic CCEA algorithm with binary tournaments.
    Maintains 20% of the population each time by default.
    '''
    population_size = population.num_agents
    #print(population, fitness)
    scored_pop = []
    for agent_id in population.agent_policies:
        scored_pop.append((population.agent_policies[agent_id], fitness[agent_id]))
    scored_pop = sorted(scored_pop, key=lambda x: x[1],reverse=True)
    scored_pop = [x[0] for x in scored_pop]
    # retain x% of the population
    scored_pop = scored_pop[:int(population_size*retain)]
    for _ in range(population_size - len(scored_pop)):
        # We only choose from the unmutated survivors
        choice = random.choice(scored_pop)
        #print(choice)
        choice.mutate()
        scored_pop.append(choice)
    # Return the evaluated population.
    return scored_pop
