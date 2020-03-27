---
type: post
title: Graphical Models for Deep Learning
subtitle: Deep Learning Course - Part 2
image: /images/graphical_models/logo_graph.jpg
tags: [Machine Learning, Deep Learning]
---

![Cool Graph](/images/graphical_models/logo_graph.jpg)

## Influence and Reference

This article is highly influenced by the [NPTEL's Deep Learning - Part 2 course by Mitesh Khapra](https://nptel.ac.in/courses/106/106/106106201/) and uses its material reserving all the rights to their corresponding authors.

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

### Recap of Probability Theory

**Marginal distribution** is a Probability distribution over a single Random Variable represented by $P(X)$.

**Joint distribution** is a probability distribution over all possible combinations of two or more RVs represented by $P(X, Y, ...)$

**Conditional distribution** is a probability distribution over some variables given other variables represented by $P(X \mid Y)$

We can also have a combination of joint and conditional called the **Joint Conditional Distribution** represented by $P(X, Y, ... \mid Z)$

Joint distribution can be factorized as a product of marginals and some conditional distributions.

$$P(X, Y) = P(X \mid Y)P(Y)$$

In case of a conditional independence, the above factorization reduces to $P(X, Y)=P(X)P(Y)$. In a more general sense,

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-1} \mid X_n)P(X_n)$$

which can be further factorized into

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-2} \mid X_{n-1}, X_n)P(X_{n-1} \mid X_n)P(X_n)$$

and so on. Now, by eliminating the independent factors, we can end up with a compact factorization which nicely describes the system.

We can get a marginal distribution from a joint distribution by summing up the value of all possible combinations of all other variables other than the variable over which the marginal is desired. An example is shown below:

$$P(X) = \sum_{Y}P(X, Y)$$

Now, suppose that we have $n$ number of random variables $X_1, X_2, ..., X_n$. We can now determine the number of ***parameters*** required to completely determine a joint probability distribution as $2^n - 1$.

### Why are we interested in Joint Distributions

A Joint distribution encodes in it all the required information of a system to fully describe it and answer all sorts of different questions that could arise.

***Example***: Assume you are running an oil company and you want to determine the locations that have the maximum probability of finding oil. You can use one of the systems shown below to describe your setting.

![Oil Graph](/images/graphical_models/oil_graph.svg)

$Y$ is the binary random variable describing the availability of oil in a particular location. Other parameters $X_1, X_2, ..., X_6$ are the factors on which it depends. We are primarily interested in $P(Y \mid X_1, X_2, ..., X_6)$ which can be determined using the joint probability over all the RVs in the system.

$$P(Y \mid X_1, X_2, ..., X_6) = \frac{P(Y, X_1, X_2, ..., X_6)}{\sum_{X_1, X_2, .., X_6}P(Y, X_1, X_2, ..., X_6)}$$

Using the joint distribution, we can find the marginal as

$$P(Y) = \sum_{X_1, X_2, ..., X_6}P(Y, X_1, X_2, ..., X_6)$$

We can also determine conditional independencies as

$$P(X_1, Y) \stackrel{?}{=} P(X)P(Y)$$

The joint distributions can, in turn, be used to ask how high or low the temperature ($X_4$) is at some location given we found oil there. In general,

> The joint distribution is an encyclopedia containing all the possible information about a system.

### How to represent Joint Distributions

To determine a joint distributions, with say $n$ number of variables $X_1, X_2, X_3, ..., X_n$, we have to get the all the explicit probabilities of all the combinations of RVs. This means that we need to specify $\mid X_1 \mid \times \mid X_2 \mid \times \mid X_3 \mid \times ... \times \mid X_n \mid - 1$. This means for $n$ binary variables, we need $2^n-1$ such values.

#### Challenges of explicit representation

The number of parameters required to represent the joint distribution increases exponentially and hence becomes intractable even with a very small number of parameters. Formally writing

- **Computational**: It is $\mathcal{O}(2^n-1)$ in both space and time. We are quickly going to run out of computational resources to calculate and store the joint.

- **Cognitive**: Impossible to acquire so many numbers from a human or even an expert.

- **Statistical**: Need a huge amount of prior data to calculate the joint.

### Can we represent joint distribution more compactly

Suppose, we have SAT scores and Intelligence and we want to determine the joint distributions over these two RVs. We can write it as either $P(S, I) = P(S \mid I)P(I)$ or $P(S, I) = P(I \mid S)P(S)$. The former seems to be a more natural way of writing the joint distribution as, naturally, the intelligence of a student determines his/her SAT scores. Hence, it makes sense to ask what is the SAT score given the student's intelligence. Keep this notion of ***assumptions*** in mind!

Now consider one more RV Grade ($G$) of a student. Notice that none of the three variables are independent of each other. Grade and SAT scores are correlated with Intelligence, in the sense that, student's performance in SAT and his grade affect our belief of his intelligence. We can make a ***assumption*** that the grade of a student doesn't depend on his SAT score, given his intelligence. We essentially are saying

$$P(G \mid I, S) = P(G \mid I)$$

which can be represented as $$G \perp S \mid I$$.

These assumptions are modeling choice and once chosen cannot be changed. This choice of assumptions is made such that it agrees the most with the observed data out of all modeling choices.

Let's see the implications of the independence assumption we made in the previous discussion. Suppose, $I$ and $S$ are binary variables and $G$ can take up to 3 values. Under no assumption, we would have $2 \times 2 \times 3 - 1 = 11$ parameters. Under our assumption, the joint distribution factorizes as follows

$$P(S, I, G) = P(G \mid I)P(S \mid I)P(I)$$

which needs only $7$ parameters to be represented. This means we can reduce the number of parameters required to represent the joint distribution by assuming independence between our RVs.

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

Each node is located with a local probability distribution which is a conditional or a marginal depending on whether the node has a parent or not. Such a graph that has coupled with each node, its local probability distribution, is called a **Bayesian Network**.

Bayesian Networks provide a very simple data structure to represent all the RVs in a system and store all the conditional dependencies using which a joint distribution can be calculated. Bayesian Networks store all the factors needed to evaluate the joint distribution.

### Different types of Reasoning encoded in a Bayesian Network

#### 1. Causal Reasoning

Here, we try to predict the downstream effects of various factors. Do we ask questions like what is the probability that the student will get a good recommendation letter ($P(L=good)$)? What is the probability that the student gets a good recommendation letter given he is highly intelligent ($P(L=good \mid I=high)$)?

This means we ask questions about the outcomes given the base knowledge.

#### 2. Evidential Reasoning

> Here we reason about the causes by looking at their effects.

This means that we want to update our belief of other variables based on the observations of some other variable. Do we ask questions like what is the probability of a student being intelligent ($P(I=high)$)? What is the probability of the course being difficult ($P(D=high)$)? What is the probability of the student being intelligent given his SAT score or given that he got a C grade ($P(I=high \mid S=low)$ or $P(I=high \mid G=C)$)?

#### 3. Explaining Away

> Here we see how different causes of the same effect can interact

It is possible that, given a low grade, the probability of the student being intelligent reduces. But what if it is also given that the exams were difficult? Would our belief about student's intelligence improve? Yes! We call this effect "***explaining away***" as exams being difficult explains why the student got a grade he got!

### Independencies encoded by Bayesian Networks

We care about independencies as they simplify the factors of the joint distributions and hence reduce the computational time. In general, given $n$ RVs, we are interested in knowing if:

1. $X_i \perp X_j$
2. $X_i \perp X_j \mid Z, Z \subseteq \{X_1, X_2, ..., X_n\} - \{X_i, X_j\}$

#### Case 1: Node and its parents

> **Rule 1: A node is not independent of its parents even when the values of other variables are given.**

![Student graph](/images/graphical_models/student_graph.svg)

For our graph, we have the following dependencies according to the rule mentioned above

- $$L \not\perp G$$
- $$G \not\perp I$$
- $$G \not\perp D$$
- $$S \not\perp I$$
- $$G \not\perp D, I \mid \{S, L\}$$
- $$S \not\perp I \mid \{D, G, L\}$$
- $$L \not\perp G \mid \{D, I, S\}$$

#### Case 2: Node and its non-parents

> **Rule 2: A node seems to be independent of other variables given the value of its parents.**

![Modified Student Graph](/images/graphical_models/student_graph_mod.svg)

In the graph (A), can we say that the SAT score is independent of whether the student will receive a recommendation letter or not, given the grade? Yes! It is because the grade provides full knowledge of whether the student will receive a recommendation letter or not irrespective of his performance in the SAT exams. But what if the instructor also asks for his SAT score and doesn't solely depend on his grade? In that case, we will end up in graph (B) and the independence will no longer hold. We can say that

1. For graph (A), $L \perp S \mid G$
2. For graph (B), $L \not\perp S \mid G$

This means that the node is independent of all other variables given the value of its parents.

NOTE: parent**s**, not parent.

For now, we will stick to the graph (A).

#### Case 3: Node and its decendents

If you look closely, I have sneaked in a ***seems to be*** in the previous rule which we have to give away now and hence discard that rule!

> **Rule 3: A node is independent of all the non-descendent variables, given its parents.**

The previous rule says that $G \perp L \mid \{D, I\}$. But what if the student got a bad recommendation letter? Is that going to change our belief about the grade of the student? Yes! Hence, the previous rule fails to capture these dependencies. If observed closely, we can come up with a new rule that, a node is independent of all the non-descendent variables, given its parents. This rule leads to the following independencies:

- $$S \perp G \mid I$$
- $$L \perp D, I, S \mid G$$
- $$G \not\perp L \mid \{D, I\}$$

This rule concludes our discussion on independence encoded by Bayesian Networks.

### Bayesian Networks

*Definition: A Bayesian Network structure G is a directed acyclic graph where nodes represent random variables $X_1, X_2, ..., X_n$. Let $Pa_{X_i}^{G}$ denote the parents of $X_i$ in $G$ and $\mathcal{ND}^G\left(X_i\right)$ denote all the variables that are non-decendents on $X_i$ in $G$. Then $G$ encodes the following set of conditional independence assumptions called the local independencies denoted by $I_i^G$ for each variable $X_i$.*

$$X_i \perp \mathcal{ND}^G\left(X_i\right) \mid Pa_{X_i}^G$$

### I Maps

*Definition: Let $G$ be a BN over a set of RVs $X$ and let P be a joint distribution over these variables. If $G$ is an I-Map of $P$, then $P$ factorizes according to $G$. Conversely, if $P$ factorized according to $G$, then $G$ is an I-Map of $P$.*

### Markov Networks

Suppose, there are four students $A, B, C$, and $D$. Now, $A$ and $B$ like to study together. Also, $B$ and $C$, $C$ and $D$, and $A$ and $D$ like to study together. But $A$ and $C$ and $B$ and $D$ don't get along together very well. This information can be represented as a graph shown below.

![Undirected Student Graph](/images/graphical_models/student_undirected.svg)

Suppose that there is a misconception in the last lecture taken by the university professor. Each student either has a misconception or has solved the misconception. We are interested in knowing everything about the weather a student has a misconception or not, given a lot of past data of such misconception. We need to evaluate the joint probability to determine the answers to all such questions:

$$P(A, B, C, D) = ?$$

Our independencies are given by:

- $A \perp C \mid \{B, D\}$
- $B \perp D \mid \{A, C\}$

Let's try to achieve this using Bayesian Networks.

![Baye to Markov](/images/graphical_models/baye_to_markov.svg)

The graph (A) encodes the independencies $A \perp C \mid \{B, D\}$ and $B \perp D \mid A$ but it fails to capture the independency $B \perp D \mid \{A, C\}$ because, given $C$ and $D$, our belief about $B$ is affected and vice versa.

The graph (B) captures the independencies $A \perp C \mid \{B, D\}$ and $B \perp D$ but not $B \perp D \mid \{A, C\}$.

We can never encode these independencies in a bayesian network, no matter how hard we try!

A sound reader would have also noticed that a directed model doesn't make sense in such a situation. There is no direction in the context of two students studying together to solve a misconception. Both the students contribute equally to the discussion. So we can't say that one student depends on the other. Moreover, we are interested in how strong the connections are between two or more students who choose to study together which will finally influence how the misconception flows among students.

The undirected form of a Bayesian Network is called a **Markov Network** and they capture exactly the independencies we desire.

### Factors in Markov Networks

We parametrize a Markov Network with some weight (or strength) associated with each edge which we call **factors** in the network opposed to considering a distribution over all the variables which is what we did in Bayesian Network. We can move from these weights to a probability distribution by simply normalizing the product of all the factors. The factors of a Markov Network capture ***affinity*** between connected RVs.

For our example, we can have factors $\phi_1(A, B), \phi_2(B, C), \phi_3(C, D) and \phi_4(A, D)$ that capture the affinity between corresponding nodes in our graph. A factor such as $\phi(A, B)$ cats like the joint distribution over A and B $P(A, B)$.

![student markov network](/images/graphical_models/student_markov.png)

As shown in the above figure, we will have to learn a value (strength/weight) for each combination of the connected RVs in the graph. Once, we have such values, we can write the joint distribution as

$$P(A, B, C, D) = \frac{1}{Z}\phi_1(A, B)\phi_2(B, C)\phi_3(C, D)\phi_4(A, D)$$

where $Z$ is the normalization constant. We can write the normalization constant as

$$Z = \sum_{A}\sum_{B}\sum_{C}\sum_{D}\phi_1(A, B)\phi_2(B, C)\phi_3(C, D)\phi_4(A, D)$$

![student markov network](/images/graphical_models/student_markov_2.png)

Let's explore one more Markov Network.

![student extended graph](/images/graphical_models/student_ext.svg)

We can see that some subgraphs are fully connected like $ADE$ and $ABF$ components. We can use a single factor to represent the connections between all the combinations of the RVs instead of using 3 different one of them. Meaning, we replace $\phi(A, D)$, $\phi(A, E)$ and $\phi(D, E)$ with $\phi(A, E, D)$. We can do this for more than 3 variables also.

### Local Independencies in a Markov Network

Let $U$ be a set of all the random variables in our joint distribution. Let $X, Y$, and $Z$ be some district subsets of $U$. The distribution $P$ over these variables would imply $X \perp Y \mid Z$ iff it factorizes as

$$P(X, Y, Z) = \frac{1}{Z}\phi_1(X, Z)\phi_2(Y, Z)$$

*Definition: For a given Markov network $H$, we define a Markov blanket $\mathcal{M}$ to be the neighborhood of an RV $X$ in $H$. We can further define the local dependencies associated with $H$ to be*

$$X \perp U - X - \mathcal{M} \mid \mathcal{M}$$

![THE END](/images/graphical_models/the_end.gif)
