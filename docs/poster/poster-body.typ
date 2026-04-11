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
    image("../assets/brittle-star-moving-" + str(i) + ".png", height: 80mm),
  )
})

#content-block[

  = Damage Robustness

  Real-world robots might sustain damage in operation.
  Nature already has solutions to this problem, e.g. brittle stars:

  #grid(
    columns: (auto, auto),
    brittle-star-figure,
    simulated-brittle-star-figure
  )

  #align(center)[brittle star damage robustness $xarrow(sym: -->, ?)$ damage-robust robots]

]

#content-block[
  
  = Research Goal

  Build damage-robust robots by learning brittle star-inspired locomotion gaits with the help of central pattern generators (CPGs) and multi-agent reinforcement learning (MARL).

]

#content-block[

  = Methodology

  + develop training algorithm for learning locomotion gaits
  + adding damage events during training
  + succes?
  
]

#content-block[

  = Results

  #grid(
    columns: (auto, auto),
    column-gutter: 10mm,
    rows: (auto, auto, auto, auto),
    row-gutter: 10mm,
    ..brittle-star-moving-figures
  )
  
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
