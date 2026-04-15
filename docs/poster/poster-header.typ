#let project-title = [
  Brittle Star-Inspired Damage Robustness:\ Multi-Agent Reinforcement Learning Approach
]

#let authors = [
  Bram Comyn, Brent Janssens, Nyah Van Wayenberge
]

#let affiliations = [
  Faculty of Science, Ghent University
]

#rect(
  width: 150%, 
  fill: rgb("#2D8CA8").lighten(20%), 
  stroke: gray,
  inset: (x: 20mm, y: 20mm),
  radius: 10mm
)[
  #set text(fill: white)
  #align(center)[
    #text(size: 86pt, weight: "bold")[#project-title]
    #v(2mm)
    #text(size: 48pt, weight: "medium")[#authors] \
    #v(2mm)
    #text(size: 48pt, weight: "light")[#affiliations]
  ]
]
