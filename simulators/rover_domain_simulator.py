""" rover_domain_simulator.py

Simulator for rover domain (collection of moving agents with POIs).

LICENSE GOES HERE
"""
import math
import random
import sys
from simulators.simulator import Simulator

class RoverDomain(Simulator):
    """ Rover Domain
    Multiagent domain that has a collection of POIs and moving agents.
    Jointstate is position / pose of all objects, joint action is agent
    action.
    """
    def __init__(self, seed=None, initial_poi_locs=None,
                 initial_agent_poses=None, number_agents=1,
                 number_pois=1, world_width=30, world_length=30):
        """ Create rover domain.
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
        :param world_width: The width of the rover domain.
        :param world_length: The length of the rover domain.
        """
        super(RoverDomain, self).__init__()

        self.number_agents = number_agents
        self.number_pois = number_pois
        self.world_width = world_width
        self.world_length = world_length

        # Check if initial poi and agent locations are within world bounds.
        # Assumes locations inputted are all either randomized or specified.
        if initial_poi_locs is not None:
            for loc in range(self.number_pois):
                if initial_poi_locs[loc][0] < 0 or initial_poi_locs[loc][0] > self.world_width or \
                   initial_poi_locs[loc][1] < 0 or initial_poi_locs[loc][1] > self.world_length:
                    print("POI " + str(loc) + " Location is Out of World Bounds")
                    exit()
        if initial_agent_poses is not None:
            for pos in range(self.number_agents):
                if initial_agent_poses[pos][0] < 0 or initial_agent_poses[pos][0] > self.world_width or \
                   initial_agent_poses[pos][1] < 0 or initial_agent_poses[pos][1] > self.world_length:
                    print("Agent " + str(pos) + " Location is Out of World Bounds")
                    exit()

        self.initial_vals = (initial_poi_locs, initial_agent_poses)
        self.seed_random(seed)
        self.initialize()
    
    def __repr__(self):
        return "POI: {0!r} Rovers: {1!r}".format(self.pois, self.agents)

    def reset(self):
        """ reset
        Resets the rover domain simulation. It will restore all values (poi
        locations, agent poses, and random seed) to their respective values
        at the start of the simulation. Note that the random values (if
        applicable) will be recreated to ensure the same random values are
        generated during the simulation.
        """
        self.seed_random(self.seed)
        self.initialize()

    def apply_actions(self, actions):
        """ apply_actions
        Moves all agents with actions specified. Updates internal state, but
        does not publish anything.

        Actions are assumed to be in the global frame, as are all stored
        agent poses. The only things in the agent frame are the agent specific
        observations, but those are handled by the Team class outside the
        simulation.

        :param actions: dictionary mapping agent_id (same key as
        self.agent_poses) and [dx, dy]
        """

        # unwrap agents dictionary
        actions = actions['agents']

        for agent_id, action in actions.items():
            loc = self.agents[agent_id]['loc']

            # Check if action moves agent within the world bounds.
            if loc[0] + action[0] < 0:
                x = 0
            elif loc[0] + action[0] > self.world_width:
                x = self.world_width
            else:
                x = loc[0] + action[0]

            if loc[1] + action[1] < 0:
                y = 0
            elif loc[1] + action[1] > self.world_length:
                y = self.world_length
            else:
                y = loc[1] + action[1]

            loc = (x, y)
            theta = math.atan2(action[1], action[0])
            self.agents[agent_id] = {'loc': loc, 'theta': theta}


    def get_jointstate(self):
        return {"agents": self.agents, "pois": self.pois}

    def initialize(self):
        """ initialize
        Initializes rover domain. The initial poi location and agent poses are
        determined either by passed in values or random values.

        If the rover domain is re-initialized, the rover domain will re-create
        any values (either the pois, agents, or both) if not originally passed.
        The values are recreated instead of stored so that when reset, the
        seed is also reset.
        """
        initial_poi_locs = self.initial_vals[0]
        if initial_poi_locs is None:
            initial_poi_locs = self.random_pois()

        initial_agent_poses = self.initial_vals[1]
        if initial_agent_poses is None:
            initial_agent_poses = self.random_agents()

        self.pois = {}
        for i, poi in enumerate(initial_poi_locs):
            self.pois["poi_" + str(i)] = poi

        self.agents = {}
        for i, agent in enumerate(initial_agent_poses):
            self.agents["agent_" + str(i)] = agent

    def seed_random(self, seed):
        """ seed_random
        Seeds random with either provided seed or a random number and stores
        the new seed.
        :returns: Returns new seed
        """
        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed
        random.seed(self.seed)
        return self.seed

    def random_pois(self):
        """ random_pois
        Creates a set of random POIs (currently just a location)
        """
        pois = []
        for _ in range(self.number_pois):
            poi = self.random_loc()
            val = random.uniform(1, 10)
            pois.append({'loc' : poi, 'value' : val})
        return pois

    def random_agents(self):
        """ random_agents
        Creates a set of random agents. An agent is a tuple of location
        and angle.
        :returns: List of agents. [[[x,y], theta], [[x,y], theta]]
        """
        agents = []
        for _ in range(self.number_agents):
            loc = self.random_loc()
            angle = random.uniform(-math.pi, math.pi)
            agents.append({'loc' : loc, 'theta' : angle})
        return agents

    def random_loc(self):
        """ random_loc
        Returns a random location within the bounds of the world
        """
        return [random.uniform(0, self.world_width),
                random.uniform(0, self.world_length)]
