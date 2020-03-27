---
type: post
title: Graphical Models for Deep Learning
subtitle: Deep Learning Course - Part 2
overlay_image: /images/graphical_models/logo_graph.jpg
overlay_filter: 0.5
tags: [Machine Learning, Deep Learning]
---

![Cool Graph](/images/graphical_models/logo_graph.jpg)

### Table of Contents

- [Recap of Probability Theory](#recap-of-probability-theory)
- [Why are we interested in Joint Distributions](#why-are-we-interested-in-joint-distributions)
- [How to represent Joint Distributions](#how-to-represent-joint-distributions)
- [Can we represent joint distribution more compactly](#can-we-represent-joint-distribution-more-compactly)
- [Can we use a Graph to represent a Joint Distribution](#can-we-use-a-graph-to-represent-a-joint-distribution)
- [Different types of Reasoning encoded in a Bayesian Network](#different-types-of-reasoning-encoded-in-a-bayesian-network)
- [Independencies encoded by Bayesian Networks](#independencies-encoded-by-bayesian-networks)
- [Bayesian Networks](#bayesian-networks)
- [I Maps](#i-maps)
- [Markov Networks](#markov-networks)
- [Factors in Markov Networks](#factors-in-markov-networks)
- [Local Independencies in a Markov Network](#local-independencies-in-a-markov-network)

### Recap of Probability Thoery

**Marginal distribution** is a Probability distribution over a single Random Variable represented by $P(X)$.

**Joint distribution** is a probability distribution over all possble combinations of two or more RVs represented by $P(X, Y, ...)$

**Conditional distribution** is a probability distribution over some variables given other variables represented by $P(X \mid Y)$

We can also have a combination of joint and conditional called the **Joint Conditional Distribution** represented by $P(X, Y, ... \mid Z)$

Joint distribution can be factorized as a product of mariginals and some conditional distributions.

$$P(X, Y) = P(X \mid Y)P(Y)$$

In case of a conditional independence, the above factorization reduces to $P(X, Y)=P(X)P(Y)$. In a more general sense,

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-1} \mid X_n)P(X_n)$$

which can be further factorized into

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-2} \mid X_{n-1}, X_n)P(X_{n-1} \mid X_n)P(X_n)$$

and so on. Now, by eliminating the factors that are independent, we can end up with a compact factorization which nicely describes the system.

We can get a marginal distribution from a joint distribution by summing up the value of all possible combinations of all other variables other than the variable over which the marginal is desired. An example is shown below:

$$P(X) = \sum_{Y}P(X, Y)$$

Now, suppose that we have $n$ number of random variables $X_1, X_2, ..., X_n$. We can now determine the number of ***parameters*** required to completely determine a joint probability distribution as $2^n - 1$.

### Why are we interested in Joint Distributions

A Joint distribution encodes in it all the required information of a system to fully describe it and answer all sorts of different questions that could arise.

***Example***: Assume you are running a oil company and you want to determine the locations that have the maximum probability of finding oil. You can use one of the systems shown below to describe your setting.

![Oil Graph](/images/graphical_models/oil_graph.svg)

$Y$ is the binary random variable describing the availibility of oil in a particular location. Other parameters $X_1, X_2, ..., X_6$ are the factors on which it depends. We are primarily interested in $P(Y \mid X_1, X_2, ..., X_6)$ which can be determined using the joint probability over all the rvs in the system.

$$P(Y \mid X_1, X_2, ..., X_6) = \frac{P(Y, X_1, X_2, ..., X_6)}{\sum_{X_1, X_2, .., X_6}P(Y, X_1, X_2, ..., X_6)}$$

Using the joint distribution, we can find the marginal as

$$P(Y) = \sum_{X_1, X_2, ..., X_6}P(Y, X_1, X_2, ..., X_6)$$

We can also determine conditional independies as

$$P(X_1, Y) \stackrel{?}{=} P(X)P(Y)$$

The joint distributions can in turn be used to ask how high or low the temperature ($X_4$) is at some location given we found oil there. In general,

> The joint distribution is an encyclopedia containing all the possible information about a system.

### How to represent Joint Distributions

To determine a joint distributions, with say $n$ number of variables $X_1, X_2, X_3, ..., X_n$, we have to get the all the explicit probabilities of all the combinations of RVs. This means that we need to specify $\mid X_1 \mid \times \mid X_2 \mid \times \mid X_3 \mid \times ... \times \mid X_n \mid - 1$. This means for $n$ binary variables, we need $2^n-1$ such values.

#### Challenges of explicit representation

The number of parameters required to represent the joint distribution increases exponentially and hence becomes intractable even with a very small number of parameters. Formally writing

- **Computational**: It is $\mathcal{O}(2^n-1)$ in both space and time. We are quickly going to run out of computational resources to calculate and store the joint.

- **Cognitive**: Impossible to aquire so many numbers from a human or even a expert.

- **Statistical**: Need huge amount of prior data to calculate the joint.

### Can we represent joint distribution more compactly

Suppose, we have SAT scores and Intelligence and we want to determine the joint distributions over these two RVs. We can write it as either $P(S, I) = P(S \mid I)P(I)$ or $P(S, I) = P(I \mid S)P(S)$. The former seems to be a more natural way of writing the joint distribution as, naturally, intelligence of a student determines his/her SAT scores. Hence, it makes sense to ask what is the SAT score given the student's intelligence. Keep this notion of ***assumptions*** in mind!

Now consider one more RV Grade ($G$) of a student. Notice that none of the three variables are independent of each other. Grade and SAT scores are clearly corelated with the Intelligence, in the sense that, student's performance in SAT and his grade affect our belief of his intellignce. We can make a ***assumption*** that the grade of a student doesn't depend on his SAT score, given his intelligence. We essentially are saying

$$P(G \mid I, S) = P(G \mid I)$$

which can be represented as $$G \perp S \mid I$$.

These assumptions are modelling choice and once chosen cannot be changed. This choice of assumptions is made such that it agrees the most with the observed data out of all modelling choices.

Let's see the implications of the independence assumption we made in previous discussion. Suppose, $I$ and $S$ are binary variables and $G$ can take upto 3 values. Under no assumption, we would have $2 \times 2 \times 3 - 1 = 11$ parameters. Under our assumption, the joint distribution factorizes as follows

$$P(S, I, G) = P(G \mid I)P(S \mid I)P(I)$$

which needs only $7$ parameters to be represented. This means, we can reduce the number of parameters required to represent the joint distribution by assuming independence between our RVs.

#### Advantages of assuming the conditional independencies

1. **Natural**: The alternate parameterization is more natural than that of the joint distribution.

2. **Compact**: Needs fewer parameters to be represented.

3. **Modular**: If we want to introduce a new RV, it will mostly not affect the variables assumed to be independent of the newly introduced RV.

### Can we use a Graph to represent a Joint Distribution

![Student Graph](/images/graphical_models/student_graph.svg)

The graph shown above represents intelligence as $I$, grade as $G$, difficulty (of exams) as $D$, letter (of recommendation) as $L$ and the SAT score as $S$. The graph clearly shows all the conditional independencies made to make the joint distribution more compact. Such graphs are called **Bayesian Graphs**. Properties of Bayesian Graphs are:

1. Contains a node for each random variable

2. The edges denote dependencies between random variables.

3. Each variable depends directly on the parents in the network.

Each node is located with a local probability distribution which is a conditional or a marginal depending on weather the node has a parent or not. Such a graph that has coupled with each node, its local probability distribution, is called a **Bayesian Network**.

Bayesain Networks provide a very simple data structure to represent all the RVs in a system and store all the conditional dependencies using which a joint distribution can be calculated. Baysian Networks store all the factors needed to evaluate the joint distribution.

### Different types of Reasoning encoded in a Bayesian Network

#### 1. Causal Reasoning

Here, we try to predict the downstream effects of various factors. We ask questions like what is the 

### Independencies encoded by Bayesian Networks

### Bayesian Networks

### I Maps

### Markov Networks

### Factors in Markov Networks

### Local Independencies in a Markov Network
