---
type: post
title: Restricted  Boltzmann Machines in Deep Learning
subtitle: Deep Learning Course - Part 2
image: /images/graphical_models/logo_rbm.jpeg
tags: [Deep Learning, Machine Learning]
---


## Influence and Reference

This article is highly influenced by the [NPTEL's Deep Learning - Part 2 course by Mitesh Khapra](https://nptel.ac.in/courses/106/106/106106201/) and uses its material reserving all the rights to their corresponding authors. This article contains a full implementation of Restricted Boltzmann Machines using pure NumPy and no other 3rd party frameworks.

### Table of Contents

- [Joint Distributions](#joint-distributions)
- [The concept of Latent Variables](#the-concept-of-latent-variables)
- [Restricted Boltzmann Machines](#restricted-boltzmann-machines)
- [RBMs as Stochastic Neural Networks](#rbms-as-stochastic-neural-networks)
- [Unsupervised Learning with RBMs](#unsupervised-learning-with-rbms)
- [Computing the gradient of the log likelihood](#computing-the-gradient-of-the-log-likelihood)
- [Sampling](#sampling)
- [Markov Chain](#markov-chain)
- [Why do we care about Markov Chains](#why-do-we-care-about-markov-chains)
- [Setting up a Markov Chain for RBMs](#setting-up-a-markov-chain-for-rbms)
- [Training RBMs using Gibbs Sampling](#training-rbms-using-gibbs-sampling)
- [Training RBMs using Contrastive Divergence](#training-rbms-using-contrastive-divergence)

### Joint Distributions

#### The Movie Critic Example

We have collected reviews of a movie that are $5$ words long and these words come from a vocabulary of $50$ words. Five such reviews are shown below:

- An unexpected and necessary masterpiece
- delightfully merged information and comedy
- Directors first true masterpiece movie
- Sci-fi perfection, truly mesmerizing film
- Waste of time and money
- Best lame historical movie ever

We create a Bayesian Network assuming that each word is a random variable and each word depends only on the previous two words in the review. One such graph is shown below.

![review graph](/images/graphical_models/reviews.svg)

Now, we can write the joint distribution as

$$P(X_1, X_2, X_3, X_4, X_5) = P(X_1)P(X_2 \mid X_1)P(X_3 \mid X_2, X_1)P(X_4 \mid X_3, X_2)P(X_5 \mid X_4, X_3)$$

Using this distribution, we can sample new reviews, we can complete the incomplete reviews and also determine if the review comes from the same author! The example of determining whether the review came from the same author is shown below.

![Review Example](/images/graphical_models/review_example.png)

Similarly, we can sample new reviews using a simple code shown below

```python
import numpy as np

review = [None, None]

words = np.array(
            [
                'the',
                'movie',
                'amazing',
                'useless',
                'was',
                'is',
                'masterpiece',
                'i',
                'liked',
                'descent'
            ]
        )

probs = dict()

probs[('the', 'movie')] = [.01, .01, .01, .03, .60, .30, .01, .01, .01, .01]

# Add conditional independencies for all users

for _ in range(5):
    outcomes = np.random.choice(np.arange(10), p=probs[(review[-2], review[-1])])
    review.append(words[outcome])

print(''.join(review[2:]))
```

The above code generates the following reviews:

1. I liked the amazing movie
2. The movie is a masterpiece
3. The movie was liked descent

#### The Bedroom Images Example

![The bedroom](/images/graphical_models/bedroom.jpg)

Suppose we are given some ${1024} \times {1024}$ images of different bedrooms like the one above and want to train the machine to generate new bedroom images. This task is very similar to the reviews example except that there is no sense of direction in case of images, unlike reviews. Assuming the pixels as our RVs, we need a Markov Network to represent the connections between the neighboring pixels instead of a Bayesian Network.

![Bedroom Network](/images/graphical_models/bedroom_network.png)

Using the above graph, we can factorize the joint as

$$P(X_1, X_2, ..., X_{1024}) = \frac{1}{Z}\prod_i\phi_1(D_i)$$

where $D_i$ is a set of variables that form a maximum clique (groups of neighboring pixels).

Using the joint distributions, again, we can generate images of a bedroom, denoise images and even impute the values of missing pixels in the image. Such models are called Generative Models and we are going to look at one such generative model called **Restricted Boltzmann Machines**.

### The concept of Latent Variables

> **Latent variables are hidden variables in a model which are responsible for the generation of the observed data.**

Such variables have no special meaning whatsoever. We can't predict that the hidden variable $h_1$ represents "sunny day", $h_2$ represents "trees", etc. We can just think of them as some variables that provide an ***abstract*** representation of the data we observe.

Such variables can also be used to ***generate*** an observation by tweaking their values a little.

Suppose, we have $32 \times 32$ images of a beach and we assume there are $n$ latent variables responsible for the generation of those images. More formally, we have a set $V$ of $\{V_1, V_2, ..., V_{1024}\}$ visible variables and a set $H$ of $\{H_1 H_2, ..., H_n\}$ latent variables. Can you now think of a Markov Network that has the joint distribution $P(V, H)$?

Our original Markov Network assumed we had connections between the neighboring pixels of an image. We can now assume connections between all the visible variables $V$ with all the hidden variables $H$, eliminating the original connections between neighboring pixels. This means that we try to capture the relations between neighboring pixels through the latent variables rather than direct connection them. This gives us the advantage of ***abstraction*** which is not possible to achieve by directly assuming connections between pixels. The intuition behind this is that images may vary differently on a per-pixel basis while being similar to each other in terms of what they represent. This behavior can be captured by the latent variables and not by assuming direct connections between pixels.

This concept is very similar to PCA and autoencoders.

For our case, we assume these visible and hidden variables to only take up binary values $\{0, 1\}$. In general, if we have $m$ visible variables and $n$ hidden variables then $V$ and $H$ can take up $2^m$ and $2^n$ unique values respectively and there are $2^{n+m}$ unique configurations possible.

$$V \in \{0, 1\}^{m}$$

$$H \in \{0, 1\}^{n}$$

### Restricted Boltzmann Machines

According to our previous discussion, we have the following Markov Network

![Markov Network](/images/graphical_models/markov_rbm.png)

There are connections between each visible and hidden variable but no connections between two visible or two hidden variables. So, we can write the joint distribution as the product of the following factors

$$P(V, H) = \frac{1}{Z}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)$$

We can also introduce factors tied to each visible and hidden unit until we normalize the whole expression appropriately.

$$P(V, H) = \frac{1}{Z}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j)$$

The normalization constant $Z$ is a partition function which is a sum of products over $2^{m+n}$ values as $V$ and $H$ can take up to $2^m$ and $2^n$ unique values respectively.

$$Z = \sum_V\sum_H\prod_{i}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j)$$

In our case, the visible variables in $V$ and hidden variables in $H$ can take on only binary values and partition function $Z$ is a sum over $2^{m+n}$ values.

Now, we need a representation that can be learned by a machine. And we know that machine learns... **parameters**. Hence, we have to introduce parameters to make the machine learn this joint distribution.

$$
\begin{gather*}
\phi_{ij}(v_i, h_j) = \exp(W_{ij}v_ih_j) \\
\psi_{i}(v_i) = \exp(b_iv_i) \\
\xi_{j}(h_j) = \exp(c_jh_j) \\
\end{gather*}
$$

This particular choice of parameters leads to a joint distribution of the following form

$$
\begin{align*}
P(V, H) &= \frac{1}{Z}\prod_{i}\prod_{j}\left[ \phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j) \right]\\
        &= \frac{1}{Z}\prod_{i}\prod_{j}\left[ \exp(W_{ij}v_ih_j)\exp(b_iv_i)\exp(c_jh_j) \right]\\
        &= \frac{1}{Z}\exp \left[ \sum_{i}\sum_{j} W_{ij}v_ih_j + \sum_{i} b_iv_i + \sum_{j} c_jh_j \right]\\
        &= \frac{1}{Z}\exp\left[ - E(V, H) \right]\\
\end{align*}
$$

where $E(V, H)$  is the energy function and is given by

$$ E(V, H) = - \sum_{i}\sum_{j} W_{ij}v_ih_j - \sum_{i} b_iv_i - \sum_{j} c_jh_j $$

The resulting joint distribution $P(V, H)$ is called a ***Boltzmann distribution*** or ***Gibb's Distribution***. We have further restricted our connections only between visible and hidden variables. Hence, this model is called **Restricted Boltzmann Machines**.

### RBMs as Stochastic Neural Networks

Let's derive some formulas that show that these RBMs are just plain old neural networks

The energy function is given by

$$E(V, H) =  - \sum_{i}\sum_{j} W_{ij}v_ih_j - \sum_{i} b_iv_i - \sum_{j} c_jh_j$$

Now, let's say $V_{-l}$ denote all the visible variables except the $l$'th variable. Then

$$
\begin{gather*}
\alpha_l(H) = - \sum_{j=1}^{n} W_{lj}h_j - b_l \\
\beta(V_{-l}, H) = - \sum_{i=1, i \neq l}^{m} \sum_{j=1}^{n} W_{ij}v_ih_j - \sum_{i=1, i \neq l}^{m} b_iv_i - \sum_{j=1}^{n} c_jh_j \\
E(V, H) = \alpha_l(H)v_l + \beta(V_{-l}, H) \\
\begin{align*}
P(v_l=1 \mid H) &= P(v_l=1 \mid V_{-l}, H) \\
                &= \frac{P(v_l=1, V_{-l}, H)}{P(v_l=0, V_{-l}, H) + P(v_l=1, V_{-l}, H)} \\
                &= \frac{e^{- \alpha_l(H)1 - \beta(V_{-l}, H)}}{e^{- \alpha_l(H)1 - \beta(V_{-l}, H)} + e^{- \alpha_l(H)0 - \beta(V_{-l}, H)}} \\
                &= \frac{1}{1 + e^{ \alpha_l(H) }} \\
                &= \sigma(- \alpha_l(H)) \\
                &= \sigma(\sum_{j=1}^{n} W_{lj}h_j + b_l) \\
\end{align*}
\end{gather*}
$$

Similarly, we have

$$
\begin{align*}
P(v_l=0 \mid H) &= \sigma(\alpha_l(H)) \\
                &= \sigma(- \sum_{j=1}^{n} W_{lj}h_j - b_l) \\
\end{align*}
$$

We can similarly calculate the P(h_l=1 \mid V, H_{-l}).

$$\alpha_l(V) = - \sum_{i=1}^{m} W_{il}v_i - c_l$$

$$\beta(V, H_{-l}) = - \sum_{i=1}^{m} \sum_{j=1, j \neq l}^{n} W_{ij}v_ih_j - \sum_{i=1}^{m} b_iv_i - \sum_{j=1, \neq l}^{n} c_jh_j$$

$$E(V, H) = \alpha_l(V)h_l + \beta(V, H_{-l})$$

$$
\begin{align*}
P(h_l=1 \mid V) &= P(h_l=1 \mid V, H_{-l}) \\
                &= \frac{P(h_l=1, V, H_{-l})}{P(h_l=0, V, H_{-l}) + P(h_l=1, V, H_{-l})} \\
                &= \frac{e^{- \alpha_l(V)1 - \beta(V, H_{-l})}}{e^{- \alpha_l(V)1 - \beta(V, H_{-l})} + e^{- \alpha_l(V)0 - \beta(V, H_{-l})}} \\
                &= \frac{1}{1 + e^{ \alpha_l(V) }} \\
                &= \sigma(- \alpha_l(V)) \\
                &= \sigma(\sum_{i=1}^{m} W_{il}v_i + c_l) \\
\end{align*}
$$

Similarly, we have

$$
\begin{align*}
P(h_l=0 \mid V) &= \sigma(\alpha_l(V)) \\
                &= \sigma(- \sum_{i=1}^{m} W_{il}v_i - c_l) \\
\end{align*}
$$

We have derived all the formulas we need to show that the model can be thought of as a neural network with each variable being a neuron! Let's go ahead to the last step of completely transforming this model into a neural network by deriving a loss function...

### Unsupervised Learning with RBMs

Just like in any other probabilistic framework, we want to maximize the ***likelihood*** which is the probability of observing the data. As $\log(.)$ a function is a concave function, we can maximize the log-likelihood instead.

$$
\begin{gather*}
& arg\,max_{W, b, c} \log P(V) \\
& arg\,max_{W, b, c} \log \frac{1}{Z} \sum_{H}P(V, H) \\
& arg\,max_{W, b, c} \log \sum_{H}P(V, H) - \log Z \\
& arg\,max_{W, b, c} \log \sum_{H}P(V, H) - \log \sum_{V, H} P(V, H) \\
& arg\,max_{W, b, c} \log \sum_{H} \exp(-E(V, H)) - \log \sum_{V, H} \exp(-E(V, H)) \\
\end{gather*}
$$

### Computing the gradient of the log-likelihood

Let's pretend for a second that $\theta$ is a collection of all the parameters and we want to maximize the above function wrt them. Let's start by evaluating the gradient first.

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial\theta} &= \frac{\partial}{\partial\theta}\left(\log \sum_{H} \exp(-E(V, H)) - \log \sum_{V, H} \exp(-E(V, H))\right) \\
                                                   &= \frac{1}{\sum_{H}\exp(-E(V, H))}\frac{\partial}{\partial\theta}\left( \sum_{H} \exp(-E(V, H)) \right) - \frac{1}{\sum_{V, H}\exp(-E(V, H))}\frac{\partial}{\partial\theta}\left( \sum_{H, V} \exp(-E(V, H)) \right) \\
                                                   &= -\frac{1}{\sum_{H}\exp(-E(V, H))}\left( \sum_{H} \exp(-E(V, H)) \frac{\partial E(V, H)}{\partial \theta} \right) + \frac{1}{\sum_{V, H} \exp(-E(V, H))}\left(\sum_{V, H} \exp(\exp(-E(V, H))\frac{\partial E(V, H)}{\partial \theta}\right) \\
                                                   &= - \sum_{H}\left( \frac{\exp(-E(V, H))}{\sum_{H} \exp(-E(V, H))}\frac{\partial E(V, H)}{\partial \theta} \right) + \sum_{V, H}\left( \frac{\exp(-E(V, H))}{\sum_{V, H} \exp(-E(V, H))}\frac{\partial E(V, H)}{\partial \theta} \right) \\
                                                   &= - \sum_{H}\left( P(H \mid V) \frac{\partial E(V, H)}{\partial \theta} \right) + \sum_{V, H}\left( P(V, H) \frac{\partial E(V, H)}{\partial \theta} \right) \\
                                                   &= - \mathbb{E}_{P(H \mid V)}\left( \frac{\partial E(V, H)}{\partial \theta} \right) + \mathbb{E}_{P(V, H)}\left( \frac{\partial E(V, H)}{\partial \theta} \right) \\
\end{align*}
$$

Now we can calculate the gradients with respect to our original parameters $W$, $b$, and $c$.

$$
\begin{gather*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} = \mathbb{E}_{P(H \mid V)}\left( v_ih_j \right) - \mathbb{E}_{P(V, H)}\left( v_ih_j \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial b_i} = \mathbb{E}_{P(H \mid V)}\left( v_i \right) - \mathbb{E}_{P(V, H)}\left( v_i \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial c_j} = \mathbb{E}_{P(H \mid V)}\left( h_j \right) - \mathbb{E}_{P(V, H)}\left( h_j \right) \\
\end{gather*}
$$

### Sampling

The gradients with respect to our parameters are expectations that are summations over an exponential number of terms and hence interactable. They can't be evaluated analytically and we have to fall back to some approximation methods. Expectations almost immediately point towards sampling. So, we need to sample from this multimodal probability distribution space of $2^{m+n}$ dimensions. This is a really difficult task that we will tackle using Gibbs Sampling. So, let's get started!

### Markov Chain

**Goal 1**: Given a random variable $X \in \mathbb{R}^n$, we want to sample from a joint distribution $P(X)$.

**Goal 2**: Given a function $f(X)$ of some random variable $X$, we want to calculate the expectation $\mathbb{E}_{P(X)}(f(X))$.

Suppose that instead of a single random variable $X$, we have a chain of random variables $X_1, ..., X_k$ where $X_i \in \mathbb{R}^n \forall i \in \{1, 2, ..., k\}$.

Instead of looking at the real values variables $X$, we will stick to our binary-valued variables.

Let's define the state of a random variable as any value from its domain. For our example, the RV $X_i$ is a $n$ dimensional vector and each dimension can take on a $0$ or a $1$. This means the size of the state space is $2^{n}$ for our case.

Suppose we are initially in some *state* $X_0$ at time $0$. We now *transition* from this state to some other state $X_1$ at time $1$. Similarly, we keep on transitioning from the previous state to some new state at each time step. Eventually, we would end up with a *chain* of $k$ random variables if we did this for $k$ time steps. We are interested in calculating the probability of transitioning to $X_{k+1}$ given all the previous states $X_1, X_2, ..., X_k$.

$$P(X_{k+1}=x_{k+1} \mid X_1=x_1, X_2, x_2, ..., X_k=x_k)$$

Assuming that the next state only depends on the previous state, we end up with a distribution

$$P(X_{k+1}=x_{k+1} \mid X_k=x_k)$$

This property of $X_{k+1} \perp \{X_1, X_2, ..., X_{k-1}\} \mid X_k$ is called the ***Markov Property*** and the resulting chain is called a ***Markov Chain***. As the variables we have are binary, we can specify the distribution $P(X_{k+1}=x_{k+1} \mid X_k=x_k)$ by defining its value for each value our RV can take in time step $k+1$ and in time step $k$. We can further arrange these probabilities in a matrix $\mathbb{T}$ where an entry $\mathbb{T}_{ab}$ denote the probability of moving from state $a$ to state $b$. Such a matrix is called a ***Transition Matrix*** and denoted by $\mathbb{T}$. For our case, the size of this matrix is $2^{n} \times 2^{n}$.

This transition matrix can change at each time step but, for our task, we assume it to be constant at all time steps. Such a Markov chain is called ***time-homogeneous Markov chain***.

Now suppose that the starting distribution at time step $0$ is given by $\mu_0$. Just to be clear, $mu^0$ is a $2^n$ dimensional vector such that $\mu^{0}_{a} = P(X_0=a)$ where $a$ is a value $X_0$ can take at time step $0$. Hence, $\mu^0$ contains probabilities of all the values $X$ can take at any time step and hence is $2^n$ dimensional vector. $\mu^{0}_a$ is the probability that the random variable takes on the value $a$ among all possible $2^n$ values. Now, we can evaluate $P(X_1=b)$ as

$$P(X_1=b) = \sum_{a}P(X_0=a, X_1=b)$$

The above formula tries to find every possible path from which we could have reached $b$ and sum up over all such probabilities in order to get the final probability. Quite intuitive, right?

$$
\begin{align*}
P(X_1=b) &= \sum_{a} P(X_0=a)P(X_1=b \mid X_0=a) \\
         &= \sum_{a} \mu^{0}_{a} \mathbb{T}_{ab}
\end{align*}
$$

We can nicely represent the distribution as a matrix multiplication shown below.

$$
\mu^{1} = P(X_1) = \mu^{0}\mathbb{T}
$$

Let's move on to $P(X_2=b)$

$$
\begin{align*}
P(X_2=b) &= \sum_{a} P(X_1=a)P(X_2=b \mid X_1=a) \\
         &= \sum_{a} \mu^{1}_{a} \mathbb{T}_{ab} \\
\end{align*}
$$

$$
\begin{align*}
P(X_2) &= \mu^{1}\mathbb{T} \\
       &= (\mu^{0}\mathbb{T})\mathbb{T} \\
       &= \mu^{0}\mathbb{T}^2
\end{align*}
$$

Similarly, for $k$'th time step

$$P(X_k) = \mu^{0}\mathbb{T}^k$$

Now, the above equation has exponential number of terms ($\mu^{k} \in \mathbb{R}^{2^n}$ and $\mathbb{T} \in \mathbb{R}^{2^n \times 2^n}$) and hence intractable to store or calculate. Let's keep that at the back of our minds and we will revisit it later.

If at certain time step $t$, $\mu^{t}$ reashes distribution $\pi$ such that $\pi \mathbb{T} = \pi$, then for all subsequent time steps

$$P(X_j) = \mu^{j} = \pi, \forall j \geq t$$

Such a distribution $\pi$ is called the stationary distribution of the Markov Chain. Now if we continue the chainby sampling $X_{t+1} = x_{t+1}, X_{t+1}=x_{t+1}, ...$ then we can think of $x_{t+1}, x_{t+2}$ as samples drawn from the same distribution $\pi$.

***Conclusion: If we run our chain from any initial distribution $\mu^{0}$ for large number of time steps then after some arbitary point $t$, we start getting samples $x_{t+1}, x_{t+2}, ...$ which come from the stationary distribution of the Markov chain.***

### Why do we care about Markov Chains

#### Our Goal

1. Sample from $P(X)$
2. Compute the expectation $\mathbb{E}_{P(X)}(f(X))$

#### Attack Plan

Now suppose we have a Markov Chain whose stationary distribution $\pi$ is our desired probability distribution $P(X)$ then
  - we can sample from the stationary distribution easily.
  - we can use those samples to calculate the emperical estimate for our expectation
    $$\mathbb{E}(f(X)) \approx \frac{1}{n} \sum_{k}^{k+n} f(X_i) $$
    where $X_i$ is drawn from the stationary distribution $\pi$ of the markov chain.

*Theorem 1: If $X_1, X_2, ..., X_n$ is a **irreducible** time homogeneous markov chain with stationary distribution $\pi$, then*

$$\frac{1}{t}\sum_{i=1}^{t} f(X_i) \xrightarrow{t \to \infty} \mathbb{E}_{P(X)}(f(X)) \text{ where } X \in \mathbb{X} \text{ and } X \sim \pi$$

*Further, if the Markov Chain is non-periodic then*

$$P(X_t=x_t \mid X_{0}=x_{0}) \to \pi \text{ as } t \to \infty, \forall x_{t}, x_{0} \in \mathbb{X}$$

#### Adding onto our goal list

1. Define the Markov Chain for our RBMs.
2. Define what the transition matrix for our Markov Chain is.
3. Show that it is easy to sample from this chain.
4. Show that the stationary distribution $\pi$ is the desired distribution $P(V, H)$.
5. Show that the chain is ***irreducible*** and ***aperiodic***.

Let's also define $X = \{V, H\}$ which means that $\{X_1, X_2, X_3, ..., X_{m+n}\} = \{V_1, V_2, ..., V_m, H_1, H_2, ..., H_n\}$.

### Setting up a Markov Chain for RBMs

#### Procedure

1. At time step $0$, we start be randomly initializing $X_0 \in \{0, 1\}^{m+n}$ where $X = \{V, H\}$
2. We randomly choose some $i \in {1, 2, ..., m+n}$ from a uniform distribution, say, $q(i)$
3. Fix values of all the variables except $X_i$.
4. Sample the new value for $X_i$ (could be a $V$ or a $H$) using the following conditional distribution.
    - $P(X_{i}=y_i \mid X_{-i}=X_{-i})$
5. Repeat the steps 2 to 4 for many many time steps.

#### Identifying the transition matrix

If we were to provide the entire transition matrix, then we would need $\mathbb{R}^{2^{m+n} \times 2^{m+n}}$ real values. Such a matrix is impossible to store in practice. So, in the procedure above, we have defined a very very sparse matrix by only allowing those transitions where the value of only one variable changes at a time step. It can formally be written as

$$
\mathbb{T}_{xy}
\begin{cases}
q(i)P(y_i \mid x_{-i}), & \text{where $y \in \{0, 1\}^{m+n}$ and $\exists i \in \{1, 2, ..., m+n\}$ such that $X_j=y_j \forall j \in \{1, 2, ..., m+n\} - \{i\}$} \\
0,                      & otherwise
\end{cases}
$$

The above expression says that the probability of the state of more than one random variable being different is $0$. While the probability that the $i'th$ random variable would change given values of all the other random variables is $q(i)P(y_i \mid x_{-i})$.

#### How is it easy to sample from this chain?

At eash step, we need to compute $P(X_i=y_i \mid X_{-i}=x_{-i})$.

We saw how to compute in the [RBMs as Stochastic Neural Networks section](#rbms-as-stochastic-neural-networks). Let's write those out assuming $y_i=1$. (Note that $y_i$ is also binary and take values $0$ or $1$)

$$
\begin{gather*}
P(v_l=1 \mid H) = \sigma(\sum_{j=1}^{n} W_{lj}h_j + b_l) \\
P(h_l=1 \mid V) = \sigma(\sum_{i=1}^{m} W_{il}v_i + c_l) \\
\end{gather*}
$$

We can now sample from a uniform distribution and use $P(v_l=1 \mid H)$ and $P(h_l=1 \mid V)$ values as thresholds to decide weather to assign a $1$ or a $0$ to the hidden or visible variable. Formally, we are just sampling from Bernoulli's distribution with probability of success $P(v_l=1 \mid H)$ or $P(h_l=1 \mid V)$ depending on $i \leq m$ or $i > m$.

#### How do we show that the stationary distribution is P(X)?

To prove that our chain converges to our desired distrbution, we need to define a theorem.

***Detailed Balance Theorem**: To show that the distribution $\pi$ of a Markov Chain described by the transition probabilities $\mathbb{T}_{xy}, x, y \in \Omega$, it is sufficient to show that $\forall x, y \in \Omega$, the following condition holds*

$$\pi(x)\mathbb{T}_{xy} = \pi(y)\mathbb{T}_{yx}$$

Let's prove this theorem for 3 different cases.

- ***Case 1: When $X$ and $Y$ are different in more than two dimensions.***

We clearly stated that we can only transition to those states where either one of the variable changes it value or all the variables remain the same. This means that, for this case, $\mathbb{T}_{xy} = 0$.

$$
\begin{gather*}
\pi(x)\mathbb{T}_{xy} \stackrel{?}{=} \pi(y)\mathbb{T}_{yx}\\
\pi(x)(0) \stackrel{?}{=} \pi(y)(0)\\
0 = 0 & (\text{prove me wrong hehe})\\
\end{gather*}
$$

- ***Case 2: When $X$ and $Y$ are the same***

$$
\begin{gather*}
\pi(x)\mathbb{T}_{xy} \stackrel{?}{=} \pi(y)\mathbb{T}_{yx}\\
\pi(x)\mathbb{T}_{xx} = \pi(x)\mathbb{T}_{xx}\\
\end{gather*}
$$

- ***Case 3: When $X$ and $Y$ differ in exactly one dimention***

$$
\begin{align*}
\pi(x)\mathbb{T}_{xy} &= P(x) (q(i)P(y_i \mid X_{-i}))\\
                      &= P(x_i, X_{-i}) (q(i)\frac{P(y_i, X_{-i})}{P(X_{-i})})\\
                      &= P(y_i, X_{-i}) (q(i)\frac{P(x_i, X_{-i})}{P(X_{-i})})\\
                      &= P(y) (q(i)P(x_i \mid X_{-i}))\\
                      &= P(y) (q(i)P(x_i \mid Y_{-i}))\\
                      &= \pi(y) \mathbb{T}_{yx}\\
\end{align*}
$$

The above case proves that the stationary distribution has converged to our distribution. All these three cases combined determine that the ***Detailed Balance Condition*** holds and we can use it as a proxy for our original distribution. Now, we have everything that we need to compute, or rather approximate, the gradient. So, let's move on to training and algorithm!

### Training RBMs using Gibbs Sampling

Let's first catch up to the last disscussion of gradients that led to all these fucken of Markov chains and Gibb's Sampling.

$$
\begin{gather*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} = \mathbb{E}_{P(H \mid V)}\left( v_ih_j \right) - \mathbb{E}_{P(V, H)}\left( v_ih_j \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial b_i} = \mathbb{E}_{P(H \mid V)}\left( v_i \right) - \mathbb{E}_{P(V, H)}\left( v_i \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial c_j} = \mathbb{E}_{P(H \mid V)}\left( h_j \right) - \mathbb{E}_{P(V, H)}\left( h_j \right) \\
\end{gather*}
$$

Let's try to simplify the gradient little bit more and then we can move on to the training part.

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} &= \mathbb{E}_{P(H \mid V)}\left( v_ih_j \right) - \mathbb{E}_{P(V, H)}\left( v_ih_j \right) \\
                                                    &= \sum_{H}\left(P(H \mid V)v_ih_j\right) - \sum_{V, H}\left(P(V, H) v_ih_j \right) \\
                                                    &= \sum_{H}\left(P(H \mid V)v_ih_j\right) - \sum_{V} P(V) \sum_{H}\left(P(H \mid V)v_ih_j\right) \\
\end{align*}
$$

We can further simplify $\sum_{H}\left(P(H \mid V)v_ih_j\right)$ as

$$
\begin{align*}
\sum_{H}\left(P(H \mid V)v_ih_j\right) &= \sum_{h_j}\sum_{H_{-j}}\left( P(h_j \mid H_{-j}, V)P(H_{-j} \mid V)v_ih_j \right) \\
                                       &= \sum_{h_j}\left( P(h_j \mid H_{-j}, V) v_ih_j \right) \left( \sum_{H_{-j}}P(H_{-j} \mid V) \right) \\
                                       &= \sum_{h_j}\left( P(h_j \mid H_{-j}, V) v_ih_j \right) \\
                                       &= P(h_j=1 \mid H_{-j}, V)v_i \\
                                       &= \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i \\
\end{align*}
$$

Substituting this in our gradient equation, we have

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} &= \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i - \sum_{V} P(V)\left( \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i \right) \\
                                                    &= \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i - \mathbb{E}_{P(V)}\left( \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i \right) \\
\end{align*}
$$

We can similarly derive the equations for the gradients with respect to $b_i$ and $c_j$

$$
\begin{gather*}
\frac{\partial\mathcal{L}(\theta)}{\partial b_i} = v_i - \mathbb{E}_{P(V)} v_i \\
\frac{\partial\mathcal{L}(\theta)}{\partial c_j} = \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right) - \mathbb{E}_{P(V)}\left( \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right) \right) \\
\end{gather*}
$$

We have removed the expectation with respet to $P(H, V)$ and $P(H \mid V)$ but still the expectation with respect to $P(V)$ remains and is exponential in time. Hence, we have to use the Gibb's sampling procedure to compute the emperical estimate of the gradient. But before we do that, let's vectorize the results we have until now.

### Training RBMs using Contrastive Divergence
