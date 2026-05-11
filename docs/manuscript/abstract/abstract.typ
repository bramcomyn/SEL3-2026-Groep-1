#import "configuration.typ": *

#let authors = ("Bram Comyn", "Brent Janssens", "Nyah Van Wayenberge")

#set document(
  title: "Software Engineering Lab 3 - Abstract",
  author: authors,
  date: datetime(day: 4, month: 4, year: 2026)
)

#show: configuration
#set heading(numbering: "1.")

#align(center)[
  #title()
  #v(0.5em)
  #authors.join(", ") \
  _Universiteit Gent: Faculteit Wetenschappen_
]

This document contains our abstract, as well as our thinking process and the different iterations we went through to arrive at our final abstract.
The final version of our abstract is the one in @suggested-abstract-3.

= Suggested Title <suggested-title>

#quote(block: true, quotes: true)[
  Brittle Star-inspired Damage Robustness: Multi-Agent Reinforcement Learning Approach
]

= Suggested Abstract 1 <suggested-abstract-1>

Building robots that can deal with challenging real-world environments is a difficult task.
Inspired by the brittle star's ability to adapt its locomotion when one of its arms is damaged, we explore the concept of enabling robots to deal with damage while continue functioning.
We implement a control system based on central pattern generators (CPGs) to replicate a rowing gait and use multi-agent reinforcement learning (MARL) to allow the robot to adapt its gait in response to damage.
Of course, the first step is to teach the robot how to walk without any damage, which turns out to be a non-trivial task in itself.
Learning a multi-arm rowing gait is a complex problem that requires significant effort and a complex coordination strategy between multiple oscillators, even when not considering damage at all.

= Suggested Abstract 2 <suggested-abstract-2>

Achieving damage robustness in real-world environments remains a significant challenge for autonomous robotics.
To address this, we look to the brittle star, an organism capable of instantly adapting its locomotion when its limbs are compromised.
We implement a control system based on Central Pattern Generators (CPGs) to replicate a rowing gait, using Multi-Agent Reinforcement Learning (MARL) to enable adaptive coordination.
Our current work focuses on obtaining a stable baseline locomotion, by trying to teach the robot to pick from a small set of motor primitives, which turns out to be a non-trivial task in itself.

= Suggested Abstract 3 <suggested-abstract-3>

Building robots that can deal with challenging real-world environments is a difficult task.
Inspired by the brittle star's ability to adapt its locomotion when one of its arms is damaged, we explore the concept of enabling robots to deal with damage while remaining operational.
We implement a control system based on central pattern generators (CPGs) to replicate a rowing gait and use multi-agent reinforcement learning (MARL) to allow the robot to adapt its gait in response to damage.
Our current work focuses on obtaining a stable baseline locomotion, by teaching the robot to pick from a small set of motor primitives.
When trying this for a fixed-target environment, the robot manages to learn a good way of coordinating its arms to reach the target.
However, generalizing this to a random-target environment turns out to be a non-trivial task in itself.
