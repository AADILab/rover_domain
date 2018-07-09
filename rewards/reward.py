""" reward.py

Abstract base class for reward.

LICENSE GOES HERE
"""

class Reward(object):
    """ Reward Base Class
    Reward base class for multiagent domain. Accepts a jointstate and
    calculates a vector of rewards (one per agent). If required (ie, if the
    reward is not Markovian), will store jointstate history.

    Calculates the reward on a per agent basis (so any counterfactuals will be
    dealt with in this class, not the learning).
    """

    def __init__(self, *args, **kwargs):
        """ Initializes the reward.

        Initialization may involve connecting to a queue or message
        passing system, in which case record_history will be the queue
        consumer.
        """
        self.history = []

    def record_history(self, jointstate):
        """ record_history

        Reads in a jointstate to calculate the reward. Repeated application of
        this function gives Reward all information needed to calculate the
        final reward of a simulation run.

        :param jointstate: A single instance in time of all information
        needed to calculate the reward (possibly with history).
        :returns: Nothing
        """
        self.history.append(jointstate)

    def calculate_reward(self):
        """ calculate_reward

        Based on the stored state, calculates and returns a reward.

        :returns: array of rewards, one per agent.
        """
        pass

    def clear(self):
        """ clear
        Clear the history of the reward.

        :returns: Nothing
        """
        self.history = []
