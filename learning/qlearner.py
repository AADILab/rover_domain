""" rover_team.py

Rover team for rover domain. Team class controls observation of individual
agents from the jointstate and transforms individual agent actions into
the global frame for use in joint action.

LICENSE GOES HERE
"""

from learning.Learner import LearnAlg


class QLearner(Learner):
    
    def __init__(self):
        """Create Q Learner
        If seed is passed, will initialize randomly. Otherwise, will initialize
        according to passed starting positions.

        Note that this class has too many parameters for PEP 8. Possible
        these args can be moved into kwargs, and have inheritance hierachy
        handle them (ie, random world, bounded world for seed, world_width)

        :param seed: Random seed. If not specified, will be generated and
        stored for reset.
        :param initial_poi_locs: Initial poi locations. If None, will be
        initialized randomly. A POI location is (x,y).
        :param initial_agent_poses: Initial agent poses. If None, will be
        initialized randomly. And agent pose is [(x,y), theta].
        :param number_agents: The number of rovers in the domain. Default is 1.
        :param number_pois: The number of pois in the domain. Default is 1.
        :param world_width: The length and width of the square rover domain.
        """
        super(QLearner, self).__init__()



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