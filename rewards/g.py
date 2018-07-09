""" g.py

Class to calculate g in the rover domain.

LICENSE GOES HERE
"""
from math import sqrt
from heapq import heappush, heappop
import numpy as np
from rewards.reward import Reward


class GlobalReward(Reward):
    """ G Reward Class
    Calculates the G reward for the multiagent domain.

    G is a path based reward, so this class keeps a history of the entire path
    of all agents. Reward is the scales value of the observed POIs.

    Equation goes here once final one decided.
    :param coupling: the number of agents that must observe a POI before it is counted
    :param observation_radius: how close an agent must be for an observation to count
    :param min_dist: the minimum distance between an agent and POI, 
    if an agent is closer than min_dist, we count it as though it were min_dist away

    Note: Check if the above description is correct
    """
    def __init__(self, coupling=1, observation_radius=4.0, min_dist=1.0):
        super(GlobalReward, self).__init__()
        self.clear()
        self.coupling = coupling
        self.observation_radius = observation_radius
        self.min_dist = min_dist

    def calculate_reward(self):
        """ calculate_reward
        """
        reward = {}
        if not self.history:
            raise "No history yet. Cannot determine vector size."

        poi_reward = {}

        # for each time step, check to see if there are any agents 
        # within observation_radius of each POI
        #
        # If there are more than coupling agents within range,
        # compute the POI reward for that time step by 
        # taking POI value/average distance away of each agent
        #
        # The overall reward from that POI is the maximum reward that has been
        # achieved over all time steps.
        
        for timestep in self.history:
            for poi_id, poi_info in timestep['pois'].items():
                poi_obs = []
                for agent_id, agent_info in timestep['agents'].items():
                    dist = self.distance(poi_info['loc'], agent_info['loc'])
                    if dist < self.observation_radius:
                        heappush(poi_obs, max(self.min_dist, dist))
                if len(poi_obs) >= self.coupling:
                    dists = [heappop(poi_obs) for _ in range(self.coupling)]
                    current_poi_reward = poi_info['value'] / np.mean(dists)
                else:
                    current_poi_reward = 0
                last = poi_reward[poi_id] if poi_id in poi_reward else 0
                poi_reward[poi_id] = max(current_poi_reward, last)

        g_reward = sum([r for _, r in poi_reward.items()])

        # Since this is global reward, all agents get the same reward
        for agent_id in self.history[0]['agents']:
            reward[agent_id] = g_reward

        return reward

    @staticmethod
    def distance(loc_1, loc_2):
        """ distance MOVE TO UTILITY CLASS
        L2 Norm of first two positions in loc_1 and loc_2. Unsafe.

        :returns: Euclidean distance between first two dimensions of input
        vectors.
        """
        square = lambda x: x**2
        return sqrt(square(loc_1[0] - loc_2[0]) + square(loc_1[1] - loc_2[1]))
