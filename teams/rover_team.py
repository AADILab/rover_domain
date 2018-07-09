""" rover_team.py

Rover team for rover domain. Team class controls observation of individual
agents from the jointstate and transforms individual agent actions into
the global frame for use in joint action.

LICENSE GOES HERE
"""

from math import cos, sin, sqrt
import numpy as np
from teams.team import Team

class RoverTeam(Team):
    """ RoverTeam
    Rover team class for multiagent domain. Accepts jointstate from Rover
    Domain simulation and sends the appropriate quadrant input to agents, and
    combines the agent outputs into a (global frame) joint action.
    """

    def __init__(self, agent_policies):
        """ __init__
        :param agent_policies: dictionary keyed by agent id of functions that
        take a numpy array and return a dx, dy vector
        """
        super(RoverTeam, self).__init__()
        self.agent_policies = agent_policies
        self.num_agents = len(self.agent_policies)

    def get_jointaction(self, jointstate):
        """ get_jointaction
        Accepts a dictionary of agent identifiers to actions and returns the
        agent actions keyed by that identifier.

        The jointstate and the actions are in the global frame. This means
        that this class will transform into the agent frame before querying
        the agent for its action.

        :param jointstate: {agents : { agent_id : info }, pois:{poi_id : info}}
        poi info is just a location, agent info is a tuple of location, heading
        :returns: {agents : { agent_id : action }}
        """
        actions = {}
        agents = jointstate['agents']
        for agent_id, info in agents.items():
            w2a, a2w = self.get_transforms(info)
            obs = self.get_agent_observation(jointstate, agent_id, w2a)
            action = self.agent_policies[agent_id].get_next(obs)
            dx_a, dy_a = action[0].item(), action[1].item()
            actions[agent_id] = a2w(dx_a, dy_a)

        return {'agents' : actions}

    @staticmethod
    def get_transforms(agent_info):
        """ get_transforms

        Two functions for transforming a location into the coordinate frame
        of the agent. This agent is assumed to be the origin with the x axis
        (theta = 0) along the heading of the agent.

        :param agent_info: (loc: (x,y), theta : theta) tuple of agent state info
        :returns: world_to_agent and agent_to_world - two coordinate transform
        functions that accept a tuple as input. The first transforms a vector
        into another vector for use in determining quadrants, the second only
        shifts a movement unit vector
        """
        loc = agent_info['loc']
        theta = agent_info['theta']
        xin, yin = loc

        cost = cos(theta)
        sint = sin(theta)

        nct = cos(-theta)
        nst = sin(-theta)

        world_to_agent = lambda xo, yo: (cost * (xo - xin) - sint * (yo - yin),
                                         sint * (xo - xin) + cost * (yo - yin))

        agent_to_world = lambda xo, yo: (nct * xo - nst * yo,
                                         nst * xo + nct * yo)
        return world_to_agent, agent_to_world

    def get_agent_observation(self, jointstate, agent_id, world_to_agent):
        """ get_agent_observation

        Determines the input to the agent policy. In this case, the observation
        is an 8-vector with values representing the count of pois and agents
        in the quadrants in the frame of the agent (optionally scaled by
        the distances of those objects to the agent and the value of the POI).

        :param jointstate: Dictionary containing both agent and poi info,
        should be {agents : { agent_id : info }, pois : { poi_id : info }}
        :returns: Observation is numpy array representing quadrant counts
        (in agent frame) of pois and agents.
        """
        agent_obs = np.zeros(8)
        agent_loc = jointstate['agents'][agent_id]['loc']

        for _, poi in jointstate['pois'].items():
            new_loc = world_to_agent(*poi['loc'])
            quad = self.get_quad(new_loc)
            increment = poi['value'] / max(self.distance(poi['loc'], agent_loc), 0.01)
            agent_obs[quad] += increment

        for other_id, agent in jointstate['agents'].items():
            if agent_id == other_id:
                continue
            new_loc = world_to_agent(*agent['loc'])
            quad = self.get_quad(new_loc)
            increment = 1 / max(self.distance(agent['loc'], agent_loc), 0.01)
            agent_obs[quad + 4] += increment

        return agent_obs

    @staticmethod
    def get_quad(loc):
        """ get_quad
        Determines the quadrant the location (2 vector) is located in.

        :param loc: 2D Position (x,y)
        :returns: 0 - 3 for quadrant, numbered CCW starting from +/+ = 0
        """
        loc_x, loc_y = loc
        if loc_x >= 0 and loc_y >= 0:
            return 0
        if loc_x >= 0 and loc_y <= 0:
            return 3
        if loc_x <= 0 and loc_y >= 0:
            return 1
        if loc_x <= 0 and loc_y <= 0:
            return 2

        raise "Location not in quadrant (shouldn't be possible)"


    @staticmethod
    def distance(loc_1, loc_2):
        """ distance MOVE TO UTILITY CLASS
        L2 Norm of first two positions in loc_1 and loc_2. Unsafe.

        :returns: Euclidean distance between first two dimensions of input
        vectors.
        """
        square = lambda x: x**2
        return sqrt(square(loc_1[0] - loc_2[0]) + square(loc_1[1] - loc_2[1]))
