""" simulator.py

Abstract base class for simulator.

LICENSE GOES HERE
"""

class Simulator(object):
    """ Simulator Base Class
    Simulator base class for multiagent domain. Applies a joint action and
    publishes a joint state for agent use.

    Can be asyncronous. Can accept variable number of actions.
    """
    def __init__(self, *args, **kwargs):
        """ Initialize the simulator.
        For random initialization, should pass a random seed. Simulator should
        store the seed to allow for deterministic reset.
        """
        pass

    def reset(self):
        """ Resets the simulator.

        Random initialization is re-initialized from the same random seed.
        """
        pass

    def apply_actions(self, actions):
        """ apply_actions accepts a number of agent actions

        :param actions: A dictionary mapping between agent uuid and action,
        allowing for the application of individual agent actions, potentially
        asyncronously.
        :returns: Nothing. May or may not update the interal simulator state
        """
        pass

    def update_jointstate(self):
        """ update_jointstate applies any unapplied agent actions and publishes
        an updated jointstate. Publishing can be modifying the accessed
        public member variable of this Simulator, or it can be literally
        publishing to a message passing queue.

        :returns: Updated joint state
        """
        pass
