""" learn.py

Abstract base class for learning.

LICENSE GOES HERE
"""

class Learner(object):
    """ Learn Base Class
    Learner base class for multiagent domain. Learning algorithms will inherit from this class

    """

    def __init__(self, *args, **kwargs):
        """ Initializes the learning algorithm.

        Initialization may involve passing in learning parameters 
        such as learning rate.
        """
        pass

    def get_action(self, *args, **kwargs):
        """ get_action

        Returns the next action to be taken by the agent. 
        May take in state to do so.
        """
        pass

    def update_learning(self, *args, **kwargs):
        """ update_learning

        Accepts the reward for doing the action at the state
        Updates state and policy
        """
        pass

    # def get_action_exploit(self):
    #     """ get_action

    #     Returns the next action to be taken by the agent. 
    #     Only returns exploitative action
    #     May take in state to do so.
    #     """
    #     pass
