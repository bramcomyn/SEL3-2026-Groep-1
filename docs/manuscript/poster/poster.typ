#let faculty-we-color = rgb("#2D8CA8")

#set page(
  paper: "a0",
  margin: (x: 20mm, top: 0mm, bottom: 0mm),
  fill: gray.lighten(80%),
  columns: 2,
)

#set text(font: "Red Hat Display", size: 36pt)
#set heading(numbering: "1")

#show heading: set text(font: "Red Hat Display", size: 48pt, weight: "bold", fill: faculty-we-color)
#show link: set text(fill: blue)
#show figure.caption: set text(size: 28pt, weight: "medium", fill: gray)
#show figure: set figure(supplement: none)

#let seperator = [
  #line(length: 100%, stroke: gray + 2pt)
]

#place(
  top + center,
  scope: "parent",
  float: true
)[
  #include "poster-header.typ"
]

#include "poster-body.typ"

#place(
  bottom + center,
  scope: "parent",
  float: true
)[
  #seperator
  #include "poster-footer.typ"
]
