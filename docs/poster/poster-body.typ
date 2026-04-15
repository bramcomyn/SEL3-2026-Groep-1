#import "@preview/xarrow:0.4.0": *

#let figure-placeholder(height: 150pt) = rect(width: 100%, height: height, fill: white, stroke: gray)[
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
    Two intact (left) and one damaged (right) brittle stars.
  ]
)

#let simulated-brittle-star-figure = figure(
  image("../assets/simulated-brittle-star.png", height: intro-figure-height),
  caption: [
    Brittle star-inspired robot with 5 arms in a simulated environment.
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
  image("../assets/methodology.svg", width: 300mm),
  caption: [
    Overview of the methodology.
  ]
)

#let reward-function-figure = figure(
  image("../assets/reward-function.svg", width: 350mm),
  caption: [
    Average reward over 10 environment steps while training for 100 episodes
  ]
)

#let random-reward-function-figure = figure(
  image("../assets/random-reward-function.svg", width: 350mm),
  caption: [
    Same training setup as above, but with random target positions
  ]
)

#content-block[

  = Damage Robustness

  Real-world robots might sustain damage in operation.
  Nature already has solutions to this problem, e.g. brittle stars:

  #align(center)[
    #grid(
      columns: (auto, auto),
      align: (center + horizon),
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

  Actions: rowing gait modulation through CPGs:

    #table(
      stroke: none,
      columns: (auto, auto),
      gutter: 5mm,
      inset: 5mm,
      align: (left + horizon),
      table.header([*Action*], [*Description*]),
      table.hline(stroke: gray + 1pt),
      [Leading arm],           [Pointing upward],
      [Left rower],            [Clockwise rotation],
      [Right rower],           [Counterclockwise rotation],
      [Secondary left rower],  [Smaller amplitude clockwise],
      [Secondary right rower], [Smaller amplitude counterclockwise]
    )

  Learning: *action selection* through MARL:
    - Observations: 
     + *Angle* between arm and target, 
     + *Direction* between body and target, 
     + *Distance* between body and target
    - Reward function: *distance reduction* toward target
    - *Fixed* target position

  #methodology-figure

]

#content-block[

  = Results

  *Can walk* in simulation with 5 arms, fixed target and no damage:

  #align(center)[
    #grid(
      columns: (auto, auto),
      column-gutter: 10mm,
      rows: (auto, auto),
      row-gutter: 10mm,
      align: (center + horizon),
      ..brittle-star-moving-figures
    )
  ]

  Reward functions shows a *small increase* in reward, but remains close to zero:

  #align(center)[
    #reward-function-figure
  ]

  Moving to random target positions shows the *same results, but with a lower average* reward:

  #align(center)[
    #random-reward-function-figure
  ]
  
]

#content-block[

  = Conclusions

  + More research is required to *fix the learning process*.
  + Reward function should be checked for *bugs*.
  + Cannot say anything about *damage robustness*
  
]
