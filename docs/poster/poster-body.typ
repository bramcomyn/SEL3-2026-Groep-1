#let figure-placeholder = rect(width: 100%, height: 150pt, fill: white, stroke: gray)[
  #set align(center + horizon)
  #text(gray)[(Figure Placeholder)]
]

= Introduction

Transitioning robots from safe test environments to real-world scenarios can be challenging, and might lead to unexpected damage.
Ideally, this damage would not render the robot unfunctional, but building robots that can withstand damage is a complex task.

Often, nature has already solved the problem at hand, as is the case with animals that can survive severe injuries.
For example, the brittle star can survive the loss a limb, which lead to the idea of building robots that can survive damage by mimicking the brittle star's structure and behavior.

#figure(
  image("../assets/brittle-stars.jpg"),
  caption: [
    Image showing three brittle stars.
    The one on the left and in the middle are undamaged, while the right one is showing clear signs of damage  (#link("https://ecology.wa.gov/blog/march-2018/eyes-under-puget-sound-critter-of-the-month-the", "source")).
  ]
)

We implement a Multi-Agent Reinforcement Learning (MARL) algorithm to train a brittle star-inspired robot to survive damage and continue performing its task.
The algorithm is trained to choose from a specific set of motor actions, which are implemented as central pattern generators (CPGs) that control the robot's movement to mimick a simplified rowing gait.

= Methodology

We built our robot using adapted versions of provided code and open-source packages, and implemented the following components:

- custom adaptation of provided CPG code, mostly remaining the same, but with some adjustments allow for vectorized environments;
- adapted rowing gait modulation: instead of one unique role per arm, each arm can pick from 5 different roles, allowing multiple arms to pick the same role;
- JAX-based implementation of Independent Q-Learning (IQL): MARL algorithm where each agent learns its own Q-function independently.

To handle simulation physics and environment interactions, we made use of the provided #link("https://pypi.org/project/biorobot/0.2.5/", `biorobot`)-package, which implements a brittle star-like robot and its environment in the #link("https://pypi.org/project/mujoco/", "MuJoCo physics engine").

In order to obtain faster training times, we made use of the vectorized environment support provided by the #link("https://mujoco.readthedocs.io/en/stable/mjx.html", `MJX`)-backend in MujoCo.

= Results

// TODO: add more details about the results of our experiments

= Conclusion

// TODO: add conclusion

= Acknowledgements

// TODO: add acknowledgements

= References

// TODO: add references

