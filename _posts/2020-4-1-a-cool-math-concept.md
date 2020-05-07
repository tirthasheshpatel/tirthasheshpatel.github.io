---
type: post
title: A Cool Math Concept
subtitle: this concept is used in deep learning RNN models to analyze exploding and vanishing gradients
image: /images/random/alto1.png
tags: Random
---

![ALTO](/images/random/alto1.png)

### Some definitions and theorems

*Definition 1: A matrix whose colums sum up to 1 is called **Column Stochastic Matrix**.*

*Definition 2: If $\lambda_1, \lambda_2, ..., \lambda_n$ are eigenvalues of any matrix, then $\lambda_k$ is considered a **dominant eigenvalue** if $\mid \lambda_k \mid > \mid \lambda_i \mid, \forall i \in \{1, 2, ..., n\} - \{k\}$. The coressponding eigenvector is called **dominant eigenvector**.*

*Therem 1: Dominant eigenvector of a stochastic matrix is 1.*

*Theorem 2: For any matrix $A$ and some vector $v_0$, the series $Av_0, A^2v_0, ..., A^nv_0$ converges to a multiple of the dominant eigenvector of matix A.*

### Some observations

Say $e_d$ and $\lambda_d$ are the dominant eigenvector and eigenvalue of some stochastic matrix $M$. We can say from theorem 2 that

$$M^nv_0 = ke_d$$

For step $n+1$, we will have

$$
\begin{align*}
M^{n+1}v_0 &= M ( M^{n}v_0 )\\
           &= M ( ke_d )\\
           &= k ( Me_d )\\
           &= k ( \lambda_de_d )\\
           &= k ( 1 e_d ) & (\text{see theorem} 1)\\
           &= ke_d\\
\end{align*}
$$

We can see that the series has converged to a constant multiple of the dominant eigenvector of matrix $M$. Let's say that the matrix $M$ isn't a stochastic matrix. Then we would have something like

$$
\begin{align*}
M^{n+1}v_0 &= M ( M^{n}v_0 )\\
           &= M ( ke_d )\\
           &= k ( Me_d )\\
           &= k ( \lambda_de_d )\\
\end{align*}
$$

Similarly, for $(n+k)\text{'th}$ step, we would have

$$M^{n+k}v_0 = k ( \lambda_d^k e_d )$$

This is interesting! Because if $\lambda_d < 1$ then the above series *vanishes* to 0. If $\lambda_d > 1$, then it *explodes* to infinity. At last, we say that it converges to some finite value for $\lambda_d = 1$. We will see a use of this in RNN models to explain exploding and vanishing gradients.

Signing out!

Tirth Patel
