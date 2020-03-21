---
layout: post
title: The curious case of Expanding space
category: Astronomy
tags: [hubble, redshifts, astronomy, physics]
---

![cool image](/images/astronomy_files/header_img_1.png)

### Introduction

Let's step back to 1915 when the term **cosmology** was first termed as the study of the universe. Astronomers, when looke throught their telescopes, observed these little beads of light. They knew that they were stars like our sun but they also observed these clouds of light like cotton (which we now know as nebulae). The first person to explain these was **Vesto Melvin Slipher**, one of the founders of cosmology. Read more about him on [this wiki page](https://en.wikipedia.org/wiki/Vesto_Slipher).

![Vesto Slipher](https://upload.wikimedia.org/wikipedia/commons/a/a7/V.M._Slipher.gif)

### Distances in astrophysics

Distance is everything in astronomy. Without knowing how far away an object is, we can't say if its the size of a cell or the size of a galaxy. We can measure the energy released by an object in the universe by knowing its distance. Not only that, we can also know how old an object is based on its distance as light travels at **ONLY** $3 \times 10^8 ms^{-1}$ and astronomical objects are **millions** of light years away. This being said, how do we measure this distance. Well, to know that, we need to go deeper into how we see things arounf us. Our eye collects light from an area and projects it onto our retina where the image is formed. So, how does the projection work. See the image below.

![man will eat this bitch up!](https://i.pinimg.com/236x/5a/52/d6/5a52d68cea8dfd6ad10b035b0321cce2--picture-ideas-photo-ideas.jpg)

You would have seen many such illutions which you thought were cool but actually they have a elegant mathematical explanation. *Computer vision: A modern approach by D. A. Forsyth* has an excellent explanation of this phenomenon. I will not dive into gory mathematical details but instead look at it in a intuitive way.

Let's say you observe two astronomers that are about the same size but at different distances. In the image below, one astronomer is 10 steps away from the camera and the other one is 30 steps away. This creates an illution that the other astronomer is smaller in size which is due to **perspective projection**. Sometimes, this phenomenon is referred to as **perspective effect**. But is there a way to know how small the astronomer will appear given its distance from the other and vice versa? Well, let's go into some mathematical details.

![perspective effect](/images/astronomy_files/perspective_effect.png)

We observe the things around us in a circle. Means all the objects we see subtend some angle with our eye. Let's say that angle is $\theta$. Now, let's say that the length of the object we observe is $l$ and the distance between our eye and the object is $r$. Then the length of the arc is $r\theta$. If the distance between the object and our eye is very large, then we can approximate the length of the object to be equal to the length of the arc. Hence, we have the relationship

$$l \approx r\theta$$

$$\theta = \frac{l}{r}$$

The angle is what determines the how big the object appears to our eye and the actual length of the object remains fixed. The relationship above clearly shows that the apparent length of the object ($\theta$) is inversely propotional to the distance beteen the object and the eye. So, as the distance between the eye and the object increase, the aparent length becomes smaller and the object appears small. Voila! We solved the mystery of why the object appears smaller as the distance increases. In the previous image of the two astronomers, one of the astronomers was $30$ steps away while the other was $10$ steps away. If we are right, then the following condition must hold

$$\frac{\theta_1}{\theta_2} = \frac{r_2}{r_1}$$

where $r_2=30$, $r_1=10$. Hence, the astronomer far away from the camera appears to be 3 times smaller than the astronomer who is close to the camera. Let's match these results with the experiment.

![perspective effect observation](/images/astronomy_files/perspective_effect_obs.png)
