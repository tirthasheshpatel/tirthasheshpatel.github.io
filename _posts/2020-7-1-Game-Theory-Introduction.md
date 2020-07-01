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

{: .box-warning}
**Note**: In games represeted by a `Normal Form`, the players don't have the knowledge of what other players have chosen.

#### Normal form

**Representation**: Finite, $n$ person `normal form` game: $\langle N, A, u \rangle$
  - `Players`: $N = \{1, 2, ..., n\}$ is a finite set of $n$ players, indexed by $i$.
  - `Action Set` for player $i$ is given by $A_i$
  - `Action Profile` is the cartesian cross product space of all the actions of each of the players.
    - Action Profile $a = (a_1, ..., a_i, ..., a_n) \in A = A_1 \times A_2 \times ... \times A_i \times ... \times A_n$
  - `Utility function` is a function mapping actions of each player to their utilities. Hence, it takes as input a `action profile` $a$ and outputs a utility vector of size $n$ representing the utility of each player for that action profile.
    - $u: A \to \mathbb{R}^n$

{: .box-warning}
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

{: .box-error}
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

{: .box-warning}
**Note**: Nash equilibrium implicitly assumes everyone plays the action that maximizes *their* utility but if that profile isn't played, nash equilibria fails to provide the best reponse to that profile of actions.

### Best Response

If you knew what everyone is going to do, it would be easy to pick the best action.

Let $a_{-i} = \langle a_1, a_2, ..., a_{i-1}, a_{i+1}, ..., a_n \rangle$. Hence, $a = (a_i, a_{-i})$.

{: .box-note}
**Best Response (Definition)**:
$$a_{i}^{*} \in \mathcal{BR}(a_{-i}) \text{ iff } \forall a_i \in A_i, u_i(a_{i}^{*}, a_{-i}) \ge u_i(a_i, a_{-i})$$

This definition leads us to the definition on `Nash Equilibrium` that says:

{: .box-note}
**Nash Equlibrium (Definition)**
Action profile $a = \langle a_1, ..., a_n \rangle$ is a *pure strategy* Nash Equilibrium iff $\forall i, a_i \in \mathcal{BR}(a_{-i})$

{: .box-warning}
There could be a unique or multiple or no Nash Equilibria of a game.

### Nash Equilibrium of Example Games

1. Prisoner's Delimma

|   |   C    |   D    |
|---|--------|--------|
| C | -1, -1 | 0, -4  |
| D | -4, 0  | -3, -3 |

The best move for the Player 1 is to choose D irrespective of what the other player chooses because he is bound to recieve either 0 or -3 unlike the case in which he chooses C where he may end up recieving -4! Same goes for the player 2 and hence there is a unique nash equilibrium which is the action profile $\\{D, D\\}$. If any one of them deviates from this profile, he/she will end up recieving -4 while the other player recieves 0.

2. Couple going to a movie!

| 1/2 |   A   |   B   |
|-----|-------|-------|
| A   | $1,2$ | $0,0$ |
| B   | $0,0$ | $2,1$ |

Action profiles $\\{A, A\\}$ and $\\{B, B\\}$ both are Nash Equlibrium profiles for this game.

3. Two coins toss game.

| 1/2 |    H   |    T   |
|-----|--------|--------|
| H   | $1,-1$ | $-1,1$ |
| T   | $-1,1$ | $1,-1$ |

Turns out thee are no nash equilibrium profiles for this game.

### Domination and Strategies

Let $s\_i$ and $s'\_i$ be two strategies for some player $i$ and let $S_{-i}$ be a set of **all** strategy profiles **other** players could take.

We define **domination** of strategies as follows:

{: .box-note}
**Strictly Dominant Strategy (Definition)**: If $u\_i(s\_i, S\_{-i}) > u\_i(s'\_{i}, S_{-i})$ then $s_i$ strictly dominates $s'_{i}$

{: .box-note}
**Very Weakly Dominant Strategy (Definition)**: If $u_i(s_i, S_{-i}) \ge u_i(s'_{i}, S\_{-i})$ then $s_i$ very weakly dominates $s'_{i}$

If one strategy dominates all others, then it is `dominant`.

A strategy profile where everone is playing a dominant strategy must be a Nash Equilibrium. An equilibrium of strictly dominant strategies must be unique.

Example: `Prisoner's Delimma`

|   |   C    |   D    |
|---|--------|--------|
| C | -1, -1 | 0, -4  |
| D | -4, 0  | -3, -3 |

The dominant strategy profile in this game is $\\{D, D\\}$ beasue no matter what the other player chooses, coosing D will lead to a "strictly" better outcome.

### Pareto Optimality

This concept asks us to look at the games from the outside as an observer and not as a player which is what we have been doing until now.

{: .box-note}
From the point of view of an outside, can some outcomes of a game be said to be `better` than others.

Idea: Sometimes, one outcome $o$ is at least as good for every agent as another outcome $o'$, and there is some agent who strictly prefers $o$ to $o'$.
  - in this case, it seems reasonable to say that $o$ is better than $o'$.
  - we say $o$ `pareto-dominates` $o'$.

{: .box-note}
**Pareto Optimality (Definition)**: An outcome $o^{*}$ is said to be Pareto-optimal if no other outcome Pareto-dominates it.

{: .box-warning}
The can be multiple pareto optimal outcomes but it is not possible to have a game in which there is no pareto optimal outcome!

So, why is "**Prisoner's *Delimma***" such a "delimma"?

|   |   C    |   D    |
|---|--------|--------|
| C | -1, -1 | 0, -4  |
| D | -4, 0  | -3, -3 |

As we have seen, the dominant strategy and nash equilibrium for this game is the profile $\\{D, D\\}$. Now, the pareto optimal profiles are $\\{C, C\\}$, $\\{C, D\\}$, and $\\{D, C\\}$. The Nash Equilibria which is also the dominant strategy is the only one that isn't a pareto optimal!! This is why this example is paradoxical and hence a "delimma".
