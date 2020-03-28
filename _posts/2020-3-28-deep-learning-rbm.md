---
type: post
title: Restricted  Boltzmann Machines in Deep Learning
subtitle: Deep Learning Course - Part 2
image: on
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
- Sci-fi perfection, tryly mesmerizing film
- Waste of time and money
- Best lame historical movie ever

We create a Bayesian Network assuming that each word is a random variable and each word depends only on previous two words in the review. One such graph is shown below.

![review graph](/images/graphical_models/reviews.svg)

Now, we can write the joint distribution as

$$P(X_1, X_2, X_3, X_4, X_5)=P(X_1)P(X_2\midX_1)P(X_3\midX_2,X_1)P(X_4\midX_3,X_2)P(X_5\midX_4,X_3)$$

Using this distribution, we can sample new reviews, we can complete the incomplete reviews and also determine if the review comes from the same author! The example of determining weather the review came from the same author is shown below.

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

The above code generates following reviews:

1. I liked the amazing movie
2. The movie is a masterpiece
3. The movie was liked descent

#### The Bedroom Images Example

![The bedroom](/images/graphical_models/bedroom.jpg)

Suppose we are given some $1024\times1024$ images of different bedrooms like the one above and want to train the machine to generate new bedroom images. This task is very similar to the reviews example except that there is no sense of direction in case of images, unlike reviews. Hence, we have to use a Markov Network to represent the connections between the neighboring pixels instead of a Bayesian Netowrk.

### The concept of Latent Variables

### Restricted Boltzmann Machines

### RBMs as Stochastic Neural Networks

### Unsupervised Learning with RBMs

### Computing the gradient of the log likelihood

### Sampling

### Markov Chain

### Why do we care about Markov Chains

### Setting up a Markov Chain for RBMs

### Training RBMs using Gibbs Sampling

### Training RBMs using Contrastive Divergence
