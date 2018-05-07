""" rover_team_test.py

Unit tests for rover_team's RoverTeam object. Run with pytest

LICENSE GOES HERE
"""

import numpy as np
from math import cos, sin, sqrt, pi
from teams.rover_team import RoverTeam

tol = 0.00001

def is_iter(x):
    return type(x) is list or type(x) is tuple

def assertFloat(x, y, tolerance):
    if is_iter(x) and is_iter(y):
        for xi, yi in zip(x,y):
            assert abs(xi - yi) < tolerance
    elif not is_iter(x) and not is_iter(y):
        assert abs(x - y) < tolerance
    else:
        assert x == y

def test_distance():
    loc_1 = (3,3)
    loc_2 = (5,5)
    assertFloat(RoverTeam.distance(loc_1, loc_2), sqrt(8), tol)

def test_get_quad_first():
    in_first = (1, 1)
    assert RoverTeam.get_quad(in_first) == 0

def test_get_quad_second():
    in_second = (-1, 1)
    assert RoverTeam.get_quad(in_second) == 1

def test_get_quad_third():
    in_third = (-1, -1)
    assert RoverTeam.get_quad(in_third) == 2

def test_get_quad_fourth():
    in_fourth = (1, -1)
    assert RoverTeam.get_quad(in_fourth) == 3

def test_quad_border():
    on_xaxis = (0, 1)
    assert RoverTeam.get_quad(on_xaxis) == 0

def test_agent_observation_agent():
    agent_loc = (3, 3)
    agent_theta = 0
    agent_info = {'loc' : agent_loc, 'theta' : agent_theta}
    w2a, _ = RoverTeam.get_transforms(agent_info)

    other_loc = (2, 2)
    other_theta = 0
    other_info = {'loc' : other_loc, 'theta' : other_theta}

    agents = {'agent_1' : agent_info, 'agent_2' : other_info}
    jointstate = {'agents' : agents, 'pois' : {}}

    team = RoverTeam(None)
    obs = team.get_agent_observation(jointstate, 'agent_1', w2a)

    expected = np.zeros(8)
    expected[6] = 1 / sqrt (2)
    assert obs[6] == expected[6]

def test_agent_observation_poi():
    agent_loc = (3, 3)
    agent_theta = 0
    agent_info = {'loc' : agent_loc, 'theta' : agent_theta}
    w2a, _ = RoverTeam.get_transforms(agent_info)

    other_loc = (2, 2)
    other_info = {'loc' : other_loc}

    agents = {'agent_1' : agent_info}
    pois = {'poi_1' : other_info}
    jointstate = {'agents' : agents, 'pois' : pois}

    team = RoverTeam(None)
    obs = team.get_agent_observation(jointstate, 'agent_1', w2a)

    expected = np.zeros(8)
    expected[2] = 1 / sqrt(2)
    assert obs[2] == expected[2]

def test_agent_observation_distance():
    agent_loc = (3, 3)
    agent_theta = 0
    agent_info = {'loc' : agent_loc, 'theta' : agent_theta}
    w2a, _ = RoverTeam.get_transforms(agent_info)

    other_loc = (5, 5)
    other_theta = 0
    other_info = {'loc' : other_loc, 'theta' : other_theta}

    agents = {'agent_1' : agent_info, 'agent_2' : other_info}
    jointstate = {'agents' : agents, 'pois' : {}}

    team = RoverTeam(None)
    obs = team.get_agent_observation(jointstate, 'agent_1', w2a)

    expected = np.zeros(8)
    expected[4] = 1 / sqrt(8)
    assert obs[4] == expected[4]

def test_get_transforms_origin():
    loc = (0, 0)
    theta = 0
    agent_info = {'loc' : loc, 'theta' : theta}
    w2a, _ = RoverTeam.get_transforms(agent_info)

    orig = (3, -4)
    expected = orig
    assert w2a(*orig) == expected

def test_get_transforms_origin_with_heading():
    loc = (0, 0)
    theta = pi / 2
    agent_info = {'loc' : loc, 'theta' : theta}
    w2a, _ = RoverTeam.get_transforms(agent_info)

    orig = (3, -4)
    expected = (4, 3)
    assertFloat (w2a(*orig), expected, tol)

def test_get_transforms_different_point():
    loc = (3, 3)
    theta = 0
    agent_info = {'loc' : loc, 'theta' : theta}
    w2a, _ = RoverTeam.get_transforms(agent_info)

    orig = (2, 2)
    expected = (-1, -1)
    assertFloat( w2a(*orig), expected, tol)
