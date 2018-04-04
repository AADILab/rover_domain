import argparse
import numpy as np
import os
import random

from rover_domain import Task_Rovers
        
def parse_args():
    parser = argparse.ArgumentParser("Rover domain argument parser")
    # Environment
    parser.add_argument("--dim_x", type=int, default=20, help="width of env")
    parser.add_argument("--dim_y", type=int, default=20, help="height of env")
    parser.add_argument("--num_poi", type=int, default=1, help="number of pois")
    parser.add_argument("--num_rover", type=int, default=1, help="number of agents")
    parser.add_argument("--action_dim", type=int, default=4, help="Action dimensions")
    parser.add_argument("--unit_test", type=bool, default=False, help="Unit test")
    # Core training parameters
    parser.add_argument("--angle_res", type=int, default=90, help="Angle res")
    parser.add_argument("--num_timestep", type=int, default=25, help="Number of timesteps")
    parser.add_argument("--poi_rand", type=bool, default=True, help="POI Initialization")
    parser.add_argument("--render", type=bool, default=True, help="Whether to render episode")
    parser.add_argument("--act_dist", type=float, default=4.0, help="Distance agent must be within")
    parser.add_argument("--coupling", type=int, default=1, help="POI's required for observation")
    parser.add_argument("--obs_radius", type=float, default=4.0,
                        help="POI observation radius")
    return parser.parse_args()

def random_policy(input):
    return np.random.uniform(-1,1,2)

def joint_state(joint_input):
    return [random_policy(i) for i in joint_input]

if __name__ == "__main__":
    arglist = parse_args()
    domain = Task_Rovers(arglist)
    obs = domain.reset()

    for _ in range(arglist.num_timestep):
        next_obs, reward = domain.step(joint_state(obs))
        obs = next_obs

    if arglist.render == True:
        domain.render()
