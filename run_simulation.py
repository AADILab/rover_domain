from simulators.rover_domain_simulator import RoverDomain
from teams.rover_team import RoverTeam
from policies.policy import RandomPolicy



def main():
    """ main

        This function contains the initial trial to run the rover domain.
    """

    # Initialize the rover domain.

    # Rover domain parameters
    seed = 1
    initial_poi_locs = None
    initial_agent_poses = None
    number_agents = 1
    number_pois = 1
    world_width = 30

    domain = RoverDomain(seed, initial_poi_locs,
                 initial_agent_poses, number_agents,
                 number_pois, world_width)


    # Intialize the rover team.

    # Rover team parameters
    agent_policies = dict()
    agent_policies_actions = dict()

    # For every agent, assign it a policy that maps input numpy vector x to dx,dy, currently just Random policy
    for i in range(number_agents):
        agent_policies["agent_"+str(i)] = RandomPolicy(2)
        agent_policies_actions["agent_"+str(i)] = agent_policies["agent_"+str(i)].get_next()

    use_distance=True

    team = RoverTeam(agent_policies_actions, use_distance)


    # Simulation loop
    steps = 100

    for step in range(steps):
        print("Step:", step)

        # Get Agent States from Rover Doman
        joint_state = domain.update_jointstate()

        print(joint_state['agents'])

        # Get the actions from the team
        actions = team.get_jointaction(joint_state)

        # Pass actions to domain to update
        domain.apply_actions(actions)






if __name__ == '__main__':
    main()