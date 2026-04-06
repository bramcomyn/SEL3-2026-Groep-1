#import "configuration.typ": *

#let authors = ("Bram Comyn", "Brent Janssens", "Nyah Van Wayenberge")

#set document(
  title: "Software Engineering Lab 3 - Abstract",
  author: authors,
  date: datetime(day: 4, month: 4, year: 2026)
)

#show: configuration

#align(center)[
  #title()
  #v(0.5em)
  #authors.join(", ") \
  _Universiteit Gent: Faculteit Wetenschappen_
]

= Brittle Star-inspired Damage Robustness in Robotics: a Multi-Agent Reinforcement Learning Approach

Building robots that can deal with challenging real-world environments is a difficult task.
Inspired by the brittle star's ability to adapt its locomotion when one of its arms is damaged, we explore the concept of enabling robots to deal with damage while continue functioning.
We implement a control system based on central pattern generators (CPGs) to replicate a rowing gait and use multi-agent reinforcement learning (MARL) to allow the robot to adapt its gait in response to damage.
Ofcourse, the first step is to teach the robot how to walk without any damage, which turns out to be a non-trivial task in itself.
Learning a multi-arm rowing gait is a complex problem that requires significant effort and a complex coordination strategy between multiple oscillators, even when not considering damage at all.
