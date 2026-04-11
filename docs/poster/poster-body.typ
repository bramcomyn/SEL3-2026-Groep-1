#let figure-placeholder = rect(width: 100%, height: 150pt, fill: white, stroke: gray)[
  #set align(center + horizon)
  #text(gray)[(Figure Placeholder)]
]

= Introduction

Transitioning robots from safe test environments to real-world scenarios can be challenging, and might lead to *unexpected damage*.
Ideally, this damage would not render the robot unfunctional, but building robots that can withstand damage is a complex task.

Often, nature has already solved the problem at hand, as is the case with animals that can survive severe injuries.
For example, the brittle star is able to sustain the loss of a limb, which lead to the idea of building robots that can *survive damage by mimicking the brittle star's structure and behavior*.

#figure(
  image("../assets/brittle-stars.jpg"),
  caption: [
    Image showing three brittle stars, where the one on the right is showing clear signs of damage  (#link("https://ecology.wa.gov/blog/march-2018/eyes-under-puget-sound-critter-of-the-month-the", "source")).
  ]
)

We implement a *Multi-Agent Reinforcement Learning* (MARL) algorithm to train a brittle star-inspired robot to survive damage and continue performing its task.
The algorithm is trained to choose from a specific set of motor actions, which are implemented as *central pattern generators* (CPGs) that control the robot's movement to *mimick a simplified rowing gait*.

= Methodology

We built our robot using adapted versions of provided code and open-source packages, and implemented the following components:

- provided CPG code, with some adjustments *allow for vectorized environments*;
- rowing gait modulation: each arm can *pick from 5 different roles*, allowing multiple arms to pick the same role;
- JAX-based implementation of *Independent Q-Learning* (IQL): MARL algorithm where each agent learns its own Q-function independently.

In order to make training simpler, we started with a single, fixed target, no damage and the following settings for the IQL algorithm:

- *reward:* distance moved closer to the target in each step, with an extra bonus of 10 for reaching a terminal state in the environment;
- *action space:* 1 discrete action per arm, with 5 possible actions (corresponding to 5 different roles in the rowing gait);
- *observation space:* angle (per arm), direction and distance to target.

= Results

After our first training runs, which used *single environments with a fixed random target and no damage*, the robot was *able to reach the target*, although these results were inconsistent across runs.

Training runs resulted in highly variable performance, which led us to believe our reward function might not show enough correlation with the actual task of reaching the target.

#figure(
  image("../assets/two-runs-single-env-trained.png"),
  caption: [
    Two training runs of 100 episodes each, showing the moving average of the reward per episode, with a window size of 10 episodes.
  ]
)

= Conclusion

= Acknowledgements

= References
