---
layout: post
title: ALAFF - Week 1 Notes
subtitle: Week 1 notes of Advanced Linear Algebra - Foundations to Frontier
hide: true
tags: [Notes]
comments: false
permalink: /alaff/week-01
---

This was a long-ass week with lots of problems and homework!

### Definition of a "norm" function

**Definition**: A functions $f: \mathbb{C}^{m} \to \mathbb{R}$ is a norm if, $\forall x, y \in \mathbb{C}^{m}$ and $\alpha \in \mathbb{R}$, it satisfies:

$$
\begin{align*}
f(x) > 0 \text{, } \forall x > 0 &&< \text{Positive Definiteness} \\
f(\alpha x) = |\alpha|x  &&< \text{Homogeniety} \\
f(x + y) \le f(x) + f(y) &&< \text{Triangular Inequality}
\end{align*}
$$
