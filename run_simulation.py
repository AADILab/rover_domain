from simulators.rover_domain_simulator import RoverDomain
from teams.rover_team import RoverTeam
from policies.policy import RandomPolicy
from policies.policy import CCEA
from policies.policy import Evo_MLP
from rewards.g import GlobalReward
import yaml
import sys

def EvaluateTeam(team, domain, reward, steps):
    for step in range(steps):
        #print(domain)
        # Get States from Rover Doman
        joint_state = domain.get_jointstate()

        # Get the actions from the team
        actions = team.get_jointaction(joint_state)

        # Pass actions to domain to update
        domain.apply_actions(actions)

        # Update the joint state
        joint_state = domain.get_jointstate()
        reward.record_history(joint_state)

    # Compute the Global Reward
    reward_G = reward.calculate_reward()
    print("Reward: ", reward_G)
    return team, domain, reward_G

def main():
    """
    """
    # Read and store parameters from configuration file.
    if len(sys.argv) is 1:
        config_f = "config.yml"
    else:
        config_f = sys.argv[1]
    with open(config_f, 'r') as f:
        config = yaml.load(f)

    # Initialize the rover domain.
    domain = RoverDomain(
        config["Seed"],
        config["Initial POI Locations"],
        config["Initial Agent Positions"],
        config["Number of Agents"],
        config["Number of POIs"],
        config["World Width"],
        config["World Length"])

    agent_policies = {}
    for i in range(config["Number of Agents"]):
        agent_policies["agent_"+str(i)] = Evo_MLP(8, 2)
    team = RoverTeam(agent_policies)

    # Initialize the reward function
    global_reward = GlobalReward(
        config["Coupling"],
        config["Observation Radius"],
        config["Minimum Distance"])

    for generation in range(config["Epochs"]):
        team, domain, fitness = EvaluateTeam(team, domain, global_reward, config["Steps"])
        # CCEA Evaluation
        CCEA(team, fitness)


if __name__ == '__main__':
    main()
