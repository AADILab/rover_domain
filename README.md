Notes
=====

Documentation Notes
===================
- Documentation on Policy Class




Implementation Questions and Suggestions
========================================
- Can Rover domain be only instantiated with a square world? Perhaps change to rectangular?

- Input validation for the POI/Rover initial locations parameters.

- rover_domain_simulator.update_jointstate() is actually a getter, maybe rename to get_state()

- rover_domain_simulator.apply_actions() does not work, modified it to grab out agents dictionary and iterate on that. Check to see if this is what was intended.

- One point of confusion is when initializing a Rover_Team, agent_policies is a dictionary of functions that map observations to dx,dy. However there is the whole Policy class that may be more suitable?



