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

Building robots that can deal with the real world's often challenging environments is a difficult task.
In this lab, we explore the concept of enabling robots to deal with damage and continue functioning, inspired by the brittle star's ability to adapt its locomotion when one of its arms is damaged.
We implement a control system based on central pattern generators (CPGs) to replicate a rowing gait and use multi-agent reinforcement learning (MARL) to allow the robot to adapt its gait in response to damage.
The robot needs to be able to learn how to walk in the first place, before it can adapt to damage.
This turns out to be a non-trivial task, as the robot needs to learn how to coordinate its arms to achieve locomotion.
