#import "@preview/xarrow:0.4.0": *

#let figure-placeholder(height: 600pt) = rect(width: 100%, height: height, fill: white, stroke: gray)[
  #set align(center + horizon)
  #text(gray)[(Figure Placeholder)]
]

#let content-block(content) = rect(
  width: 100%, 
  fill: white, 
  stroke: gray,
  inset: (x: 10mm, y: 10mm),
  radius: 5mm
)[
  #content
]

#let intro-figure-height = 120mm

#let brittle-star-figure = figure(
  image("../assets/brittle-stars.jpg", height: intro-figure-height),
  caption: [
    Two intact and one damaged brittle stars.
  ]
)

#let simulated-brittle-star-figure = figure(
  image("../assets/simulated-brittle-star.png", height: intro-figure-height),
  caption: [
    Brittle star-inspired robot in a simulated environment.
  ]
)

#let brittle-star-moving-figures = range(1, 5).map(i => {
  figure(
    image("../assets/brittle-star-moving-" + str(i) + ".png", height: 100mm),
    caption: [
      #text(black)[#i / 4]
    ],
  )
})

#let methodology-figure = figure(
  image("../assets/methodology.svg")
)

#let reward-function-figure = figure(
  image("../assets/reward-function.svg", width: 350mm),
  caption: [
    Average reward over 10 environment steps while training for 100 episodes
  ]
)

#let learning-curve-figure = figure(
  image("../assets/learning_curve_comparison.svg", width: 350mm)
)

#let trajectories = figure(
  image("../assets/position_trajectory.png")
)

#content-block[

  = Damage Robustness

  Real-world robots might sustain *damage* in operation.
  Nature already has solutions to this problem, e.g. in the form of *brittle stars*: sea creatures that are close relatives to starfish, but that can deal with the loss of limbs by *adapting their locomotion* and even regeneration.

  #align(center)[
    #grid(
      columns: (auto, auto),
      align: (center + horizon),
      column-gutter: 1em,
      brittle-star-figure,
      simulated-brittle-star-figure
    )
  ]

  #align(center)[brittle star damage robustness $xarrow(sym: -->, ?)$ damage-robust robots]

]

#content-block[

  = Research Goal

  Build damage-robust robots by learning brittle star-inspired locomotion gaits with the help of central pattern generators (CPGs) and multi-agent reinforcement learning (MARL).

]

#content-block[

  = Methodology

  *CPGs* are a *mathematical model for representing natural rythmic motions* without sensory feedback.
  We use them in a *MARL* pipeline where the CPG gaits are the chosen actions.
  The idea of MARL is to have *multiple agents interact independently* in the same environment.

  We make use of *Independent Q-Learning* (IQL), which is an adaption of Q-Learning in which each agent treats the *other agents as an extra source of stochasticity*.
  To keep training complexity lower, agents *share their networks' parameters*.

  #methodology-figure

  Agents pick one of *five preconfigured CPG gaits* which allow them to switch roles from leading arm to primary or secondary rower, both left and right.
  Damage is introduced by *zeroing out actuator inputs for one arm* at a random point in time during an episode.

]

#content-block[

  = Results

  #learning-curve-figure
  
]

#content-block[

  = Conclusions

  // + More research is required to *fix the learning process*.
  // + Reward function should be checked for *bugs*.
  // + Cannot say anything about *damage robustness*
  
  + Training to walk to a target on a larger distance requires more and longer training episodes.
  + Training with damage takes more than twice as long as training without damage to obtain similar results.
  
]

#content-block[

  = Future work

  - Explore more complex locomotion gates (without CPGs?)
  - Trying out other MARL algorithms

]
