---
layout: post
title: ALAFF - Week 1 Notes
subtitle: Week 1 notes of Advanced Linear Algebra - Foundations to Frontier
hide: true
tags: [Notes]
comments: false
permalink: /alaff/week-01
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['<span>$','$</span>'], ['\\(','\\)']],
      processEscapes: true
    }
  });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


This was a long-ass week with lots of problems and homework!

### Vector Norms

**Definition**: A functions <span>$f: \mathbb{C}^{m} \to \mathbb{R}$</span> is a norm if, <span>$\forall x, y \in \mathbb{C}^{m}$</span> and <span>$\alpha \in \mathbb{R}$</span>, it satisfies:

<div>
$$
\begin{align*}
f(x) > 0 \text{, } \forall x > 0 && \text{Positive Definiteness} \\
f(\alpha x) = |\alpha|x  && \text{Homogeniety} \\
f(x + y) \le f(x) + f(y) && \text{Triangular Inequality} \\
\end{align*}
$$
</div>

### Vector-2 Norm

**Definition**: <span>$\|\cdot\|_2: \mathbb{C}^{m} \to \mathbb{R}$</span>

<div>
$$
\begin{align*}
\|x\|_2 &= \sqrt{\sum_{i=0}^{m-1} |x_i|^2} \\
         &= \sqrt{x^H x} \\
\end{align*}
$$
</div>

**Lemma 1**: Cauchy Schwarz inequality says <span>$\forall x,y \in \mathbb{C}^m$</span>:

<div>
$$|x^H y| \le \|x\|_2 \|y\|_2$$
</div>

**Proof**: Assume that <span>$x \ne 0$</span> and <span>$y \ne 0$</span> otherwise the inequality becomes trivially true.

We can then choose <span>$\|x\|_2 = 1$</span> and <span>$\|y\|_2 = 1$</span>. That leaves us to prove <span>$|x^H y| \le 1$</span>

Pick

<div>
$$
\alpha =
\begin{cases}
1                     && x^H y = 0 \\
\frac{y^H x}{|x^H y|} && \text{otherwise} \\
\end{cases}
$$
</div>

Now, <span>$|\alpha| = 1$</span> and <span>$|\alpha x^H y|$</span> is real and non-negative (because <span>$|\y^H x| = |x^H y|$</span>)

<div>
$$
\begin{align*}
\alpha x^H y = \overline{\alpha x^H y} = \bar{\alpha} y^H x && (\overline{x^H y} = y^H x)
\end{align*}
$$
</div>

<div>
$$
\begin{align*}
0 &\le \|x - \alpha y\|_2^{2} \\
  &= (x - \alpha y)^H (x - \alpha y) \\
  &= (x^H - \bar{\alpha}y^H)(x - \alpha y) \\
  &= x^H x - \alpha x^H y - \bar{\alpha}y^H x + (\bar{\alpha}\alpha) (y^H y) \\
  &= 1 - 2 \alpha x^H y + (1) (1) && (x^H x=1, y^H y=1, |\alpha|=1, \bar{\alpha} y^H x=\alpha x^H y) \\
  &= 2 - 2 \alpha x^H y \\
  &= 2 - 2  \frac{y^H x}{|x^H y|} x^H y && ((x^H y)^H = y^H x, z^H z = \|z\|_2^{2}) \\
  &\le 2 - 2 |x^H y| \\
|x^H y| &\le 1 \\
\end{align*}
$$
</div>

Now if the <span>$\|x\|_2 \ne 1$</span> and <span>$\|y\|_2 \ne 1$</span> then we can normalize them to have norm 1. Hence, substituting the normalized vector in the above equation, we have:

<div>
$$
\begin{align*}
|\frac{x^H}{\|x\|_2} \frac{y}{\|y\|_2}| &\le 1 \\
|x^H y| &\le \|x\|_2 \|y\|_2 \\
\end{align*}
$$
</div>

Hence, we conclude our proof here.

<hr width="50%" style="text-align=center;">

**Theorem 1**: Vector-2 norm is a norm function.

**Proof**: We need to prove three properties of the norm functions to conclude vector-2 norm is a norm function

- Positive definiteness: Let the <span>$i^{\text{th}}$</span> entry of the vector <span>$x$</span> be non-zero. Then

<div>
$$
\begin{align*}
\|x\|_2 &= \sqrt{\sum_{j=0}^{m-1} |x_j|^2} \\
         &\ge \sqrt{|x_i|^2} \\
         &> 0 \\
\end{align*}
$$
</div>

Hence, vector-2 norm is positive definite.

- Homogeneity:

<div>
$$
\begin{align*}
\|\alpha x\|_2 &= \sqrt{\sum_{i=0}^{m-1} |\alpha x_i|^2} \\
                &= \sqrt{\sum_{i=0}^{m-1} |\alpha|^2|x_i|^2} \\
                &= |\alpha| \sqrt{\sum_{i=0}^{m-1} |x_i|^2} \\
                &= |\alpha| \|x\|_2 \\
\end{align*}
$$
</div>

vector-2 norm is homogeneous.

- Triangular Inequality:

<div>
$$
\begin{align*}
\|x+y\|_2^{2} &= (x+y)^H (x+y) \\
               &= (x^H+y^H) (x+y) \\
               &= x^H x + x^H y + y^H x + y^H y \\
               &= \|x\|_2^2 + \|y\|_2^2 + 2\text{Re}(x^H y) \\
               &\le \|x\|_2^2 + \|y\|_2^2 + 2|x^H y| \\
               &\le \|x\|_2^2 + \|y\|_2^2 + 2\|x\|_2\|y\|_2 && \text{(Cauchy Schwarz inequality)} \\
               &= (\|x\|_2 + \|y\|_2)^2 \\
\|x+y\|_2 &\le \|x\|_2 + \|y\|_2 \\
\end{align*}
$$
</div>

Hence, we conclude our proof that vector-2 norm is a norm!

