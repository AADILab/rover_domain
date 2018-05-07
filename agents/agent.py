""" agent.py

Abstract base class for team.

LICENSE GOES HERE
"""

class Agent(object):
    """ Agent Base Class
    Agent base class for multiagent domain. Acts given an observation, and
    has a UUID.
    """

    def __init__(self, *args, **kwargs):
        pass

    def act(self, *args):
        """ act
        Given an input observation, agent will return an action.

        :param *args: All information returned from a Team. It is used, coupled
        with any internal state, to calculate the action.
        :returns: action
        """
        pass

    def can_act(self):
        """ can_act
        Blocking function. For asyncronous agents, they can only act if they
        are free.

        :returns: True if the agent can act this timestep, false if the agent
        is in the process of a multi timestep action.
        """
        pass
