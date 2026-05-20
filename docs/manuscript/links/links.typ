#import "../configuration.typ": *
#import "@preview/tiaoma:0.3.0"

#set document(
  author: "Bram Comyn",
  title: "SEL3 - Groep 1 - Links"
)

#show: configuration

#set figure(supplement: none)

#let readme-url = "https://github.com/bramcomyn/SEL3-2026-Groep-1#readme"
#let wiki-url = "https://github.com/bramcomyn/SEL3-2026-Groep-1/wiki"

#grid(
  columns: (1fr),
  rows: (1fr, 1fr),
  row-gutter: 5em,
  stroke: none,
  align: center + horizon,
  figure(
    tiaoma.qrcode(
      readme-url,
      options: (
        scale: 7.0
      )
    ),
    caption: [QR-code naar #link(readme-url)]
  ),
  figure(
    tiaoma.qrcode(
      wiki-url,
      options: (
        scale: 7.0
      )
    ),
    caption: [QR-code naar #link(wiki-url)]
  )
)
