"""
d.py
Calculates D given a G function which uses a 
labeled history of the world.
"""
from rewards.reward import Reward
from rewards.g import GlobalReward

class DifferenceReward(Reward):
    '''
    Calculates a domain-independent version of the difference
    reward. Uses a single Global reward, which must use a history
    of the world and actions taken. Thus, by removing an agent
    from the history, D is calculated via
        G(w) - D(w-agent)
    '''
    def __init__(self, G):
        super(DifferenceReward, self).__init__()
        self.clear()
        self.alternate_history = []
        self.G = G
        self.world_G = None

    def calculate_reward(self, agent_id):
        '''
        Calculates an alternative history *without* the agent_id
        in it, and evaluates both on the global reward associated
        with this instance.
        '''
        for timestep in self.history:
            agents = {}
            # Copy POI location
            poi = timestep[1]
            # Copy over all but the identified agent
            for k in timestep[0]:
                if k == agent_id:
                    continue
                else:
                    agents[k] = timestep[0][k]
            new_timestep = {agents, poi}
            self.alternate_history.append(new_timestep)
        # Calculate original G
        if self.world_G is not None:
            self.G.history = self.history
            self.world_G = self.G.calculate_reward()
        self.G.history = self.alternate_history
        alternate_G = self.G.calculate_reward()
        # Clear the alternate history for another call to D
        self.alternate_history = []
        return self.world_G - alternate_G
        
    def clear(self):
        self.history = []
        self.alternate_history = []
