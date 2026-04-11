#let project-title = [
  Brittle Star-Inspired Damage Robustness:\ Multi-Agent Reinforcement Learning Approach
]

#let authors = [
  Bram Comyn#super[1], Brent Janssens#super[1], Nyah Van Wayenberge#super[1]
]

#let affiliations = [
  #super[1] Faculty of Science, Ghent University
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
    #text(size: 96pt, weight: "bold")[#project-title]
    #v(2mm)
    #text(size: 54pt, weight: "medium")[#authors] \
    #v(2mm)
    #text(size: 54pt, weight: "light")[#affiliations]
  ]
]
