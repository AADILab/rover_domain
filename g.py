""" g.py

Class to calculate g in the rover domain.

LICENSE GOES HERE
"""

from reward import Reward

class G(Reward):
    """ G Reward Class
    Calculates the G reward for the multiagent domain.

    G is a path based reward, so this class keeps a history of the entire path
    of all agents. Reward is the scales value of the observed POIs.

    Equation goes here once final one decided.
    """
    def __init__(self):
        super(G, self).__init__()
        self.clear()

    def accept_jointstate(self, jointstate):
        self.history.append(jointstate)

    def calculate_reward(self):
        pass

    def clear(self):
        self.history = []
