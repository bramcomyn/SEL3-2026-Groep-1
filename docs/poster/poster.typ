#set page(
  paper: "a0",
  margin: (x: 40mm, top: 30mm, bottom: 10mm),
  fill: gray.lighten(90%),
  columns: 3,
)

#set text(font: "Red Hat Display", size: 28pt)

#set heading(numbering: "01")
#show heading: set text(font: "Red Hat Display", size: 48pt, weight: "bold")

#show link: set text(fill: blue)

#let seperator = [
  #v(2mm)
  #line(length: 100%, stroke: gray + 2pt)
  #v(2mm)
]

#place(
  top + center,
  scope: "parent",
  float: true
)[
  #include "poster-header.typ"
  #seperator
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
