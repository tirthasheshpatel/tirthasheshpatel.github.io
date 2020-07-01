---
layout: post
title: Game Theory Introduction
subtitle: Let's play some games
permalink: drafts/game-theory-introduction
hide: true
tags: [Game Theory]
---

{: .box-warning}
**Warning:** This is a draft. It is yet to be posted

This article focuses at creating notes for the Game Theory course on Coursera.

### Self-Interested Agents and Utility

- ***Self interested*** means:
  - Not that they want to harm others but that they only care about themselves.
  - The agent has its own description of the states of the world and acts based on that description.

- Each Agent has its ***utility function***.
  - `quantifies` degree of preferance across alternatives.
  - explains the impact of `uncertainties`.
  - `Decition-Theoretic Rationality`: act to maximize expected utility.

### Defining Games

- **Some Terms**
  - `Players`: Decision Makers.
  - `Actions`: What can the players do.
  - `Payoffs`: What motives players.

- `Normal Form (Matrix Form)`: List what payoffs each player gers as a function of thier actions.
  - It is *as if* all players move simultaneously.
  - Strategies encode many things.

- `Extensive form` includes timing of the moves (actions).
  - Players move sequentially (represented as a tree!).
  - Keeps track of what each player knows when he/she makes each decision.

{: .box-note}
**Note**: In games represeted by a `Normal Form`, the players don't have the knowledge of what other players have chosen.

#### Normal form

- **Representation**: Finite, $n$ person `normal form` game: $\langle N, A, u \rangle$
  - `Players`: $N = \{1, 2, ..., n\}$ is a finite set of $n$ players, indexed by $i$.
  - `Action Set` for player $i$ is given by $A_i$
  - `Action Profile` is the cartesian cross product space of all the actions of each of the players.
    - Action Profile $a = (a_1, ..., a_i, ..., a_n) \in A = A_1 \times A_2 \times ... \times A_i \times ... \times A_n$
  - `Utility function` is a function mapping actions of each player to their utilities. Hence, it takes as input a `action profile` $a$ and outputs a utility vector of size $n$ representing the utility of each player for that action profile.
    - $u: A \to \mathbb{R}^n$
