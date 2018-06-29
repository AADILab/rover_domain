Notes
=====

Documentation Notes
===================
- Documentation on Policy Class




Implementation Questions and Suggestions
========================================
- One point of confusion is when initializing a Rover_Team, agent_policies is a dictionary of functions that map observations to dx,dy. However there is the whole Policy class that may be more suitable?

- Does the global reward function in `g.py` work? Does it need to go through the whole history to compute it?


