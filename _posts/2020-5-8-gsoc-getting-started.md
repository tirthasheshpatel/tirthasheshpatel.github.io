---
layout: post
title: Pre-GSoC Period - I am excited to get started!
subtitle: I am very excited to get started with my GSoC project with PyMC3 team!
gsoc_post: true
tags: [GSoC 2020]
comments: true
---

## Getting started with GSoC '20

I am happy to say that I have been selected as a Student Developer under the GSoC 2020 program. I will be working with [pymc-devs][1], an organization under the [NUMFocus Umbrella][2], to develop a higher-level API for Gaussian Processes in [PyMC4][3] under the mentorship of [Christopher Fonnesbeck][4] and [Osvaldo Martin][5]. I am very excited to learn new tools and techniques from my mentors for the project. I aim to complete all the goals listed in my proposal as soon as possible so I can work on models and approximations that are not present in PyMC3.

### My Project Abstract

My goal for GSoC 2020 is to implement, test, and maintain a higher-level API for Gaussian Processes in PyMC4 using TensorFlow and TensorFlow Probability and write tutorials/articles and notebooks explaining their usage.

My work consists of implementing many Gaussian Process Models including Student's T Process Model and writing optimization methods like Neuton's Method and Powell's Method to find the maximum a-posteriori of the models that can be used to infer parameters of the model. My goal is also to implement at least one approximation technique when full GP modelling becomes impractical.

<small> [View my project abstract on GsoC page here][12] </small>

### My Journey so far...

I started using PyMC3 in June of 2018 while I was doing a course on [Bayesian Methods for Machine Learning][6] hoping to understand bayesian modelling and MCMC algorithms. It helped me to understand and write bayesian models in Python with ease without diving deep into the underlying algorithms. I never would have thought that I am going to help develop a tool as complex as this in the near future!

In January 2020, I discovered that PyMC3 is taking part in GSoC'20 and thought to myself that I may never get a chance to work with such a huge organization and with developers as experienced as [Christopher Fonnesbeck][4] and [Thomas Wiecki][7].

So, I gathered some hope in late January and started going through the codebase of PyMC3 and PyMC4. It took me a lot of time and effort to understand the codebase but I had acquainted a lot of knowledge of the basic structure of PyMC3 API by February. I also made [my first PR][8] during this period.

It was the time to choose a project from the PyMC3 wiki page. Gaussian Process Modelling excited me the most as I had worked a lot with it before to develop some projects. In late February, I decided to post a [discourse thread][9] on PyMC discourse introducing myself to the [pymc-devs][1] and proposing the project.

Having proposed the project, I started going through the Gaussian Process API in PyMC3. It was quite a large codebase to go through but I was well acquainted with the API by the end of February. It helped me to learn a lot about python as well as theano. I made a [series of PRs on PyMC3 project][10] making small changes to the GP API around this time.

Fully acquainted with the GP API of PyMC3 and basic flow of PyMC4 project, I felt confident enough to create a prototype of GP API for PyMC4. Following the advice of [Christopher Fonnesbeck][4] on the [discourse thread][9], I made a [PR proposing my prototype][11] (which very closely resembled the PyMC3 API) to the [pymc-devs][1] which was very well received by the developers and my mentors. Many core developers, including [Bill Engels](https://github.com/bwengals) (designer of GP API in PyMC3), provided reviews using which I was able to refine the API and my knowledge of GP Modelling in a bayesian workflow.It also led me to create [this blog post][13] explaining my work so far.

Having done so much work prior to GSoC, I decided to start working on a proposal to be submitted to the GSoC page. With the help of [Osvaldo Martin][5], I was able to create a very strong proposal that helped me organize the time and tasks for my project. I want to thank [Osvaldo Martin][5] to provide reviews and positive feedback on my proposal. I was able to submit my proposal on the GSoC page in time.

After a month of suspense and nervousness, I got the following mail from GSoC'20.

![gsoc mail of acceptance!!!](/images/gsoc-mail.png)

It was a relief and moment of pride to see this mail. It is the support from my mentors and parent that have brought me so far in my career that I can't thank them more! I am super excited to work with the PyMC team this summer. The journey so far has brought me so much knowledge about software development and Bayesian statistics that I can't wait to get started with my project this summer!

## Thank you, Mentors and my lovely Parents!

The journey so far has helped me learn a lot about bayesian modelling, software development and using tools such as TensorFlow and TensorFlow Probability. Along the way, I was also able to help some pymc users on discourse and I hope to do so in the future too.

I want to thank [Christopher Fonnesbeck][4], [Osvaldo Martin][5] and [pymc-devs][1] for giving me this opportunity to work with them on this awesome project! Lastly, I want to thank and dedicate this internship to my parents (especially my late mother), who made me capable of standing up for myself and educated me with unconditional love! I love you, mom and dad!

[1]: https://github.com/pymc-devs
[2]: https://numfocus.org/
[3]: https://github.com/pymc-devs/pymc4
[4]: https://github.com/fonnesbeck
[5]: https://github.com/aloctavodia
[6]: https://www.coursera.org/learn/bayesian-methods-in-machine-learning
[7]: https://github.com/twiecki
[8]: https://github.com/pymc-devs/pymc4/pull/213
[9]: https://discourse.pymc.io/t/adding-gaussian-processes-in-pymc4-this-summer/4538
[10]: https://github.com/pymc-devs/pymc3/pulls?q=is%3Apr+author%3Atirthasheshpatel
[11]: https://github.com/pymc-devs/pymc4/pull/235
[12]: https://summerofcode.withgoogle.com/projects/6135416450711552
[13]: https://tirthasheshpatel.github.io/2020-03-16-gaussian-process-in-pymc4/
