#import "@preview/zebraw:0.6.1": *

#let fwe-blue = rgb(45, 140, 168)
#let header-footer-line = line(length: 100%, stroke: 0.5pt + fwe-blue)

#let header = {

  set text(9pt, fill: fwe-blue)
  context {
    let headings = document.title

    stack(
      spacing: 4pt,
      grid(
        columns: (1fr, 1fr),
        [#headings],
        align(right)[UGent - Faculteit Wetenschappen]
      ), header-footer-line
    )
  }
  
}

#let footer = {

  set text(9pt, fill: fwe-blue)
  context {
    let current-page = counter(page).at(here()).first()
    let page-total   = counter(page).final().first()

    stack(
      spacing: 4pt,
      header-footer-line,
      grid(
        columns: (1fr, 1fr),
        [Academiejaar 2025 - 2026],
        align(right)[Pagina #current-page van #page-total]
      )
    )
  }
  
}

#let appendix(content) = {
  set heading(numbering: "A.1")
  counter(heading).update(0)
  state("appendix").update(true)
  content
}

#let code-block(code, hl: "") = {
  zebraw(
    numbering-separator: true,
    highlight-lines: hl,
    comment-flag: $-->$,
    lang: false,
    code
  )
}

#let configuration(body) = {

  set outline(depth: 3)
  set text(
    font: "Latin Modern Sans",
    lang: "nl",
    region: "be"
  )
  
  set par(justify: true)
  set bibliography(style: "ieee")
  show link: set text(fill: fwe-blue)

  set page(
    header: header,
    footer: footer
  )

  set figure(numbering: it => {
    let appx = state("appendix", false).get()
    let alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    let header = counter(heading).get().at(0)
    if appx [#alph.at(header - 1).#it]
    else [#header.#it]
  })

  set heading(numbering: "1.1")

  show raw.where(block: false): set text(fill: rgb("#d03e3e"))
  show figure.where(kind: table): set block(breakable: true)
  show figure.where(kind: table): set text(size: 8pt)

  body
  
}
