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

***Self interested*** means:
  - Not that they want to harm others but that they only care about themselves.
  - The agent has its own description of the states of the world and acts based on that description.

Each Agent has its ***utility function***.
  - `quantifies` degree of preferance across alternatives.
  - explains the impact of `uncertainties`.
  - `Decition-Theoretic Rationality`: act to maximize expected utility.

### Defining Games

**Some Terms**
  - `Players`: Decision Makers.
  - `Actions`: What can the players do.
  - `Payoffs`: What motives players.

`Normal Form (Matrix Form)`: List what payoffs each player gers as a function of thier actions.
  - It is *as if* all players move simultaneously.
  - Strategies encode many things.

`Extensive form` includes timing of the moves (actions).
  - Players move sequentially (represented as a tree!).
  - Keeps track of what each player knows when he/she makes each decision.

{: .box-note}
**Note**: In games represeted by a `Normal Form`, the players don't have the knowledge of what other players have chosen.

#### Normal form

**Representation**: Finite, $n$ person `normal form` game: $\langle N, A, u \rangle$
  - `Players`: $N = \{1, 2, ..., n\}$ is a finite set of $n$ players, indexed by $i$.
  - `Action Set` for player $i$ is given by $A_i$
  - `Action Profile` is the cartesian cross product space of all the actions of each of the players.
    - Action Profile $a = (a_1, ..., a_i, ..., a_n) \in A = A_1 \times A_2 \times ... \times A_i \times ... \times A_n$
  - `Utility function` is a function mapping actions of each player to their utilities. Hence, it takes as input a `action profile` $a$ and outputs a utility vector of size $n$ representing the utility of each player for that action profile.
    - $u: A \to \mathbb{R}^n$

{: .box-note}
**Note**: The utility of the players doesn't only depend on their action but also on actions of other players.

### More Games

`Prisoners Delimma` is any game that has the following form.

| 1/2 |   C   |   D   |
|-----|-------|-------|
| C   | $a,a$ | $b,c$ |
| D   | $c,b$ | $d,d$ |

where $c > a > d > b$.

These games are well known for paradoxical properties we will explore later.

`Games of pure competition`: Players have `exactly opposed` interests!
  - There must be exactly $2$ players.
  - For all action profiles $a \in A$, $u_1(a) + u_2(a) = c$ for some constant $c$.
    - Special Case: `Zero Sum`.

Example 1: We toll two coins. If both are heads or tails, player 1 gets $1$ and player 2 gets $-1$ while if one is heads and the other is tails, player 2 gets $1$ and player 1 gets $-1$. This is a zero sum game.

| 1/2 |    H   |    T   |
|-----|--------|--------|
| H   | $1,-1$ | $-1,1$ |
| T   | $-1,1$ | $1,-1$ |

Example 2: Game of Rock-Paper-Scissors! Whoever wins gets $1$ other gets $-1$ and if it ties, both get $0$. This is also a game of pure competion and a zero sum game.

{: .box-note}
**Excersise**: Try to find more games of pure competition.

`Games of co-operation`: Players have excatly the same interests.
  - no conflict: all players want the same thing.
  - $\forall a \in A, \forall i,j, u_i(a) = u_j(a)$
  - Utility is represented by a real value instead of a vector (as all players get same payoffs.)

Example: Couple going to a movie!

| 1/2 |   A   |   B   |
|-----|-------|-------|
| A   | $1,2$ | $0,0$ |
| B   | $0,0$ | $2,1$ |

As you can see, if they choose different movies, they both get a zero (obviously because they want to go together). Player 1 likes movie B more, so he gets more payoffs for going to that movie while Player 2 likes movie A more and get more payoff for going to that movie.

### Strategic Reasoning

Let's play a game where you will have to rate a model on scale of 1-100 and you will win if you are closest to 2/3'rd of the average of ratings of all the other players.

Let's try to reason through this game and coose our best action:
  - Suppose the average is $X$.
  - This means you will win if you rate $\frac{2}{3}X$.
  - But wait, now other players will reason throught the same argument and rate the model $\frac{2}{3}X$.
  - This means you should rate $\left(\frac{2}{3}\right)^2 X$
  - This agument continues down the road and we reach at a conclusion that we should rate the model 1. If any player rates the model more that 1, he will immediately diverge from 2/3'rd of the average and lose.

This action profile (of each player rating the mode 1) is known as the `Nash Equilibrium`. Some properties of Nash Equilibrium are:
  - Provides a consistent list of actions.
  - Each player's action maxilizes his/her expected payoff, given the actions of the others.
  - Nobody has an incentive to *deviate* from their action if an equilibrium profile is played.
  - Somebody has an incentive to deviate from a profile of actions that do not form an equilibrium.

{: .box-note}
**Note**: Nash equilibrium implicitly assumes everyone plays the action that maximizes *their* utility but if that profile isn't played, nash equilibria fails to provide the best reponse to that profile of actions.

### Best Response

If you knew what everyone is going to do, it would be easy to pick the best action.

Let $a_{-i} = \langle a_1, a_2, ..., a_{i-1}, a_{i+1}, ..., a_n \rangle$. Hence, $a = (a_i, a_{-i})$.

{: .box-note}
Best Response Definition:
$$a_{i}^{*} \in \mathcal{BR}(a_{-i}) \text{ iff } u_i(a_{i}^{*}, a_{-i}) \ge u_i(a_i, a_{-i})$$
