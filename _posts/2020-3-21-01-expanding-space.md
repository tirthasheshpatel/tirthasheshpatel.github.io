---
layout: post
title: The curious case of Expanding space
category: Astronomy
tags: [hubble, redshifts, astronomy, physics]
---

<!-- ![cool image](/images/astronomy_files/header_img_1.png) -->
![cool image](https://cdn.discordapp.com/attachments/277329934949679104/277330843653898240/99.jpg)

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

Aha! The experimental ratio of sizes of astronomers is approximately $3.03$ which is very close to our expected result $3$.

### Distances from Brightness

The universe is full of stars like our sun and galaxies like our milky way. They all share a common property of brightness => releasing energy in the form of light. If we can find the distance of a galaxy or star based on how bright it appears here on earth then we would have a universal way of measuring distance.

Can you think of a way to measure the distances from the apparent brightness of an object? I have an idea!

![bulb is the idea representing my idea! Puns intended](https://i2.wp.com/business-ethics.com/wp-content/uploads/2011/09/EarthTalkBULB.jpg?zoom=1.25&resize=948%2C572&ssl=1)

Imagine we have a light bulb and it releases some energy per second which we call **lunimosity**. The luminosity of a typical light bulb might be around $100W$ while it is around $3.86 \times 10^26W$ which is much much higher. Also notice that the light released by the light bulb spreads around it in a sphere (in the image it looks like a circle because its 2D).

Say, we have a telescope that collects light coming from an object at a distance $r$ from it. If the luminosity of the object is $L$ then we can give the equation for the apparent brightness as follows

$$P = \frac{LA}{4 \pi r^2}$$

where $A$ is the area of the lens of the telescope. This area can vary highly from telescope to telescope and hence we use **flux** which is the apparent brightness per unit area as a universal measure.

$$\phi = \fracP{L}{4 \pi r^2}$$

$$\phi \propto \frac{1}{r^2}$$

where $\phi$ is our flux. The equation above clearly shows that the apparent brightness of an object is inversely propotional to the square of its distance from the telescope. Meaning, if we know the brightness of the object, we can easily get its distance! That is amazing! Also, we can easily measure relative distance between objects by the following equation.

$$\frac{\phi_1}{\phi_2} = \frac{r_2^2}{r_1^2}$$

![worked out example](/images/astronomy_files/distance_using_brightness.png)

### Spectra

![spectra of all the elements](/images/astronomy_files/spectra.jpg)

In astronomy, the spectra is about $70$ to $90%$ of all the observations. It is of utmost importance because of the following properties:

1. Every element on the periodic table has unique spectral emission lines and hence we can detect presence of a particular element using the spectra of an astronomical object.
2. Spectral lines are very accurate and don't leave any windows for errors.
3. Easy to measure with current technology.

The telescope first collects light from the image plane which is fed into a **spectograph**. The light collected this way has many wavelengths present in it. This spectograph has a device like a prism (usually a diffraction grading) which splits light into its individual wavelengths and these wavelengths are recorded to form a graph which is typically showing the wavelength on the x-axis and the energy per unit wavelength on the y-axis. This is known as the **spectrograph**.

For a normal star, galaxy or nebulae, the graph first goes up and then down to roughly form a bell shaped curve like the one shown below. If it peaks at the beginning than its a hot star, at the middle than its a medium hot star and if it peaks at the end of the spectrograph than its a cold star.

![expected spectra](/images/astronomy_files/expected_spectra.svg)

It turns out that we normally observe some dips in between this bell at certain wavelengths called **absorption lines**. This means that the light coressponding to that wavemength was absorbed showing the presence of an element coressponding to that wavelength.

![reduce_spectrum_extract](/images/astronomy_files/reduce_spectrum_extract.jpg)

Using such a spectrograph, we can make out what a star, galaxy, or nebulae is made up of.

### Doppler Effect

``to be continued``

### Refrences

[The Solar-Stellar Spectrograph](http://www2.lowell.edu/users/jch/sss/article.php?r=t_datared_d_spectrum)
