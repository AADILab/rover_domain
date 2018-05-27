""" team.py

Abstract base class for team.

LICENSE GOES HERE
"""

class Team(object):
    """ Team Base Class
    Team base class for multiagent domain. Reads a jointstate of a simulation
    and sends appropriate agent level information to each agent. Collects agent
    actions to send a joint action to simulation.

    If domain is asyncronous, joint action will not include certain agent's
    actions. Agents are responsible for can_act() function, otherwise their
    action will be included in the joint action.
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_jointaction(self, jointstate):
        """ get_jointaction
        Determines the action for all agents that are free to act given
        a jointstate of the system.

        :param jointstate: A dictionary of all relevant information
        :returns: A dictionary mapping between agents and actions, only agents
        that are not blocking return an action.
        """
        pass

    def get_agent_observation(self, jointstate, agent_id, world_to_agent):
        """ get_agent_observations
        For debugging and testing purposes, returns the observation given
        to an individual agent that they use to determine their actions.

        :param jointstate: A dictionary of all relevant information
        :returns: The observation given to a single agent.
        """
        pass
