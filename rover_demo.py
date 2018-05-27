import argparse
import numpy as np
import os
import random
import logging

from rover_domain import Task_Rovers
from policy import RandomPolicy, Evo_MLP

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
    parser.add_argument("--render", type=bool, default=False, help="Whether to render each episode step")
    parser.add_argument("--render_path", type=bool, default=False, help="Whether to display full path")
    parser.add_argument("--act_dist", type=float, default=4.0, help="Distance agent must be within")
    parser.add_argument("--coupling", type=int, default=1, help="POI's required for observation")
    parser.add_argument("--obs_radius", type=float, default=4.0,
                        help="POI observation radius")
    parser.add_argument("--sensor_model", type=bool, default=True,
                        help="True for density sensor, false for minimum")
    return parser.parse_args()

def joint_action(policies, joint_input):
    return [f(x).data.numpy() for f,x in zip(policies, joint_input)]

if __name__ == "__main__":
    arglist = parse_args()
    domain = Task_Rovers(arglist)
    obs = domain.reset()

    policy = RandomPolicy(output_shape=2, low=-1, high=1)
    policies = [policy.get_next() for _ in range(arglist.num_rover)]

    networks = [Evo_MLP(12,2) for _ in range(arglist.num_rover)]
    policies = [net.get_next() for net in networks]
    updates  = [net.get_evo() for net in networks]

    obs = domain.reset()
    for _ in range(arglist.num_timestep):
        print ("Step")
        action = joint_action(policies, obs)
        print (action)
        for f in updates:
            f()
        action = joint_action(policies, obs)
        print (action)
        
        next_obs, reward, done, info  = domain.step(action)
        obs = next_obs

        if arglist.render:
            domain.visualize()

    if arglist.render_path:
        domain.render()
