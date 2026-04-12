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
    average reward over 10 environment steps while training for 100 episodes
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

  - actions: rowing gait modulation through CPGs:

    #table(
      stroke: none,
      columns: (auto, auto),
      gutter: 5mm,
      inset: 5mm,
      align: (left + horizon),
      table.header([*action*], [*description*]),
      table.hline(stroke: gray + 1pt),
      [leading arm],           [pointing upward],
      [left rower],            [clockwise rotation],
      [right rower],           [counterclockwise rotation],
      [secondary left rower],  [smaller amplitude clockwise],
      [secondary right rower], [smaller amplitude counterclockwise]
    )

  - learning: *action selection* through MARL:
    - observations: 
     + *angle* between arm and target, 
     + *direction* between body and target, 
     + *distance* between body and target
    - reward function: *distance reduction* toward target
    - *fixed* target

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

  However, this did not reflect in high rewards during training:

  #align(center)[
    #reward-function-figure
  ]
  
]

#content-block[

  = Conclusions
  
]

#content-block[

  = Acknowledgements
  
]

#content-block[

  = References
  
]
