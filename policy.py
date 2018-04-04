import numpy as np

class Policy(object):

    def __init__(self, *args, **kwargs):
        pass

    def get_next(self, *args, **kwargs):
        pass

    def get_train(self, *args, **kwargs):
        pass

class RandomPolicy(Policy):

    def __init__(self, output_shape, low=-1, high=1):
        self.output_shape = output_shape
        self.low = low
        self.high = high

    def get_next(self):
        return (lambda _ : np.random.uniform(self.low, self.high, self.output_shape))
