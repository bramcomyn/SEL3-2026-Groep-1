#set page(
  paper: "a0",
  margin: (x: 40mm, top: 30mm, bottom: 10mm),
  fill: gray.lighten(90%)
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

#grid(
  columns: (100%),
  rows: (auto, 1fr, auto),
  inset: 0.5em,
  gutter: 2em,

  block[
    #include "poster-header.typ"
    #seperator
  ],

  block(height: 100%)[
    #columns(3, gutter: 0.5em)[
      #include "poster-body.typ"
    ]
  ],

  block[
    #seperator
    #include "poster-footer.typ"
  ]
)