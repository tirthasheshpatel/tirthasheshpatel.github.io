---
title: The Curious Case of the Expanding Universe
date: 2020-3-21
categories: 
  - Astronomy
  - Physics
tags: 
  - Astronomy
  - Physics
permalink: posts/the-curious-case-of-the-expanding-universe
---

<!-- ![cool image](/images/astronomy_files/header_img_1.png) -->
![cool image](https://cdn.discordapp.com/attachments/277329934949679104/277330843653898240/99.jpg)

### Table of Contents

- [Introduction](#introduction)
- [Distances in astrophysics](#distances-in-astrophysics)
- [Distances from Brightness](#distances-from-brightness)
- [Spectra](#spectra)
- [Doppler Effect](#doppler-effect)
- [Hubble's Law](#hubbles-law)
- [Refrences](#refrences)

### Introduction

Let's step back to 1915 when the term **cosmology** was first termed as the study of the universe. Astronomers, when looked through their telescopes, observed these little beads of light. They knew that they were stars like our sun but they also observed these clouds of light like cotton (which we now know as nebulae). The first person to explain these was **Vesto Melvin Slipher**, one of the founders of cosmology. Read more about him on [this wiki page](https://en.wikipedia.org/wiki/Vesto_Slipher).

![Vesto Slipher](https://upload.wikimedia.org/wikipedia/commons/a/a7/V.M._Slipher.gif)

### Distances in astrophysics

Distance is everything in astronomy. Without knowing how far away an object is, we can't say if its the size of a cell or the size of a galaxy. We can measure the energy released by an object in the universe by knowing its distance. Not only that, but we can also know how old an object is based on its distance as light travels at **ONLY** $3 \times 10^8 ms^{-1}$ and astronomical objects are **millions** of light-years away. This being said, how do we measure this distance. Well, to know that, we need to go deeper into how we see things around us. Our eye collects light from an area and projects it onto our retina where the image is formed. So, how does the projection work? See the image below.

![man will eat this bitch up!](https://i.pinimg.com/236x/5a/52/d6/5a52d68cea8dfd6ad10b035b0321cce2--picture-ideas-photo-ideas.jpg)

You would have seen many such illusions which you thought were cool but actually, they have an elegant mathematical explanation. *Computer vision: A modern approach by D. A. Forsyth* has an excellent explanation of this phenomenon. I will not dive into gory mathematical details but instead, look at it intuitively.

Let's say you observe two astronomers that are about the same size but at different distances. In the image below, one astronomer is 10 steps away from the camera and the other one is 30 steps away. This creates an illusion that the other astronomer is smaller in size which is due to **perspective projection**. Sometimes, this phenomenon is referred to as **perspective effect**. But is there a way to know how small the astronomer will appear given its distance from the other and vice versa? Well, let's go into some mathematical details.

![perspective effect](/images/astronomy_files/perspective_effect.png)

We observe the things around us in a circle. This means all the objects we see subtend some angle with our eye. Let's say that angle is $\theta$. Now, let's say that the length of the object we observe is $l$ and the distance between our eye and the object is $r$. Then the length of the arc is $r\theta$. If the distance between the object and our eye is very large, then we can approximate the length of the object to be equal to the length of the arc. Hence, we have a relationship

$$l \approx r\theta$$

$$\theta = \frac{l}{r}$$

The angle is what determines how big the object appears to our eye and the actual length of the object remains fixed. The relationship above clearly shows that the apparent length of the object ($\theta$) is inversely proportional to the distance between the object and the eye. So, as the distance between the eye and the object increase, the apparent length becomes smaller and the object appears small. Voila! We solved the mystery of why the object appears smaller as the distance increases. In the previous image of the two astronomers, one of the astronomers was $30$ steps away while the other was $10$ steps away. If we are right, then the following condition must hold

$$\frac{\theta_1}{\theta_2} = \frac{r_2}{r_1}$$

where $r_2=30$, $r_1=10$. Hence, the astronomer far away from the camera appears to be 3 times smaller than the astronomer who is close to the camera. Let's match these results with the experiment.

![perspective effect observation](/images/astronomy_files/perspective_effect_obs.png)

Aha! The experimental ratio of sizes of astronomers is approximately $3.03$ which is very close to our expected result $3$.

### Distances from Brightness

The universe is full of stars like our sun and galaxies like our milky way. They all share a common property of brightness => releasing energy in the form of light. If we can find the distance of a galaxy or star based on how bright it appears here on earth then we would have a universal way of measuring distance.

Can you think of a way to measure the distances from the apparent brightness of an object? I have an idea!

![bulb is the idea representing my idea! Puns intended](https://i2.wp.com/business-ethics.com/wp-content/uploads/2011/09/EarthTalkBULB.jpg?zoom=1.25&resize=948%2C572&ssl=1)

Imagine we have a light bulb and it releases some energy per second which we call **luminosity**. The luminosity of a typical light bulb might be around $100W$ while it is around $3.86 \times 10^26W$ which is much much higher. Also, notice that the light released by the light bulb spreads around it in a sphere (in the image it looks like a circle because it's 2D).

Say, we have a telescope that collects light coming from an object at a distance $r$ from it. If the luminosity of the object is $L$ then we can give the equation for the apparent brightness as follows

$$P = \frac{LA}{4 \pi r^2}$$

where $A$ is the area of the lens of the telescope. This area can vary highly from telescope to telescope and hence we use **flux** which is the apparent brightness per unit area as a universal measure.

$$\phi = \frac{L}{4 \pi r^2}$$

$$\phi \propto \frac{1}{r^2}$$

where $\phi$ is our flux. The equation above clearly shows that the apparent brightness of an object is inversely proportional to the square of its distance from the telescope. Meaning, if we know the brightness of the object, we can easily get its distance! That is amazing! Also, we can easily measure the relative distance between objects by the following equation.

$$\frac{\phi_1}{\phi_2} = \frac{r_2^2}{r_1^2}$$

![worked out example](/images/astronomy_files/distance_using_brightness.png)

### Spectra

![spectra of all the elements](/images/astronomy_files/spectra.jpg)

In astronomy, the spectra are about $70$ to $90\%$ of all the observations. It is of utmost importance because of the following properties:

1. Every element on the periodic table has unique spectral emission lines and hence we can detect the presence of a particular element using the spectra of an astronomical object.
2. Spectral lines are very accurate and don't leave any windows for errors.
3. Easy to measure with current technology.

The telescope first collects light from the image plane which is fed into a **spectograph**. The light collected this way has many wavelengths present in it. This spectrograph has a device like a prism (usually a diffraction grating) that splits light into its wavelengths and these wavelengths are recorded to form a graph which is typically showing the wavelength on the x-axis and the energy per unit wavelength on the y-axis. This is known as the **spectrograph**.

For a normal star, galaxy or nebulae, the graph first goes up and then down to roughly form a bell-shaped curve like the one shown below. If it peaks at the beginning than its a hot star, at the middle than its a medium-hot star and if it peaks at the end of the spectrograph than its a cold star.

![expected spectra](/images/astronomy_files/expected_spectra.svg)

It turns out that we normally observe some dips in between this bell at certain wavelengths called **absorption lines**. This means that the light corresponding to that wavelength was absorbed showing the presence of an element corresponding to that wavelength.

![reduce_spectrum_extract](/images/astronomy_files/reduce_spectrum_extract.jpg)

Using such a spectrograph, we can make out what a star, galaxy, or nebulae is made up of.

### Doppler Effect

![train example](https://upload.wikimedia.org/wikipedia/commons/9/90/Dopplerfrequenz.gif)

You may have encountered a train engine passing from right beside you while you are cursing why god sent in such a huge train while you were getting late to work. Well then, you must also have heard the pattern of sound that hits your ears from when the engine appears in sight moving towards you until when it goes out of the sight moving away... There is a steep increase in the sound while it is moving towards you which reaches its peak when it's just beside your ear and then a steep decrease until it moves away out of your sight. This effect is called **doppler effect**. It is seen in all the phenomena where waves are involved (for instance, here, the sound waves). Light can also take the form of a wave, meaning, doppler effect is observed in case of light too! Let's see what is the reason behind the effect and how does it relate to astronomy!

Let's look at the formal definition from Wikipedia.

> ***Definition: The Doppler effect (or the Doppler shift) is the change in frequency of a wave in relation to an observer who is moving relative to the wave source.[1] It is named after the Austrian physicist Christian Doppler, who described the phenomenon in 1842***

Below is a gif that shows the difference between the emission of waves by a stationary and a non-stationary source.

![stationary](https://upload.wikimedia.org/wikipedia/commons/e/e3/Dopplereffectstationary.gif)

![not stationary](https://upload.wikimedia.org/wikipedia/commons/c/c9/Dopplereffectsourcemovingrightatmach0.7.gif)

How does it relate to astronomy? Read the title of this article again! If we show that there is some shift (*doppler shift*) in the spectrograph we observe then we can say that the astronomical objects around us are moving relative to us. This is our first step towards claiming that the universe expands.

Let's say we have observed a spectrograph but found an absorption line that doesn't match any of the elements or the result we expected. This means that there is a shift in the wavelength and hence the object is moving relative to us.

![redshift example](/images/astronomy_files/redshift.png)

Suppose an absorption line was expected at $\lambda_l$ and an absorption line is observed at $\lambda_o$. Then we can give the relative shift as

$$z = \frac{\lambda_o-\lambda_l}{\lambda_l}$$

If the value of $z$ is positive, it means that the object is moving away from us and the shift is known as **redshift**. If the value of $z$ is negative, it means that the object is moving towards us (very rare) and the shift is known as **blueshift**. We can also calculate the relative velocity of the object using the theorem of relativity!

$$z = \sqrt{\frac{1+\frac{v}{c}}{1-\frac{v}{c}}}+1$$

$$z \approx \frac{v}{c}$$

So, Vesto Slipher went through and used a telescope to be the first person to take spectra of these galaxies or nebulae as he knew them. He observed a graph similar to what is shown below

![sliphers observation](/images/astronomy_files/sliphers_observation.png)

You can see from the image clearly that the wavelengths have been redshifted and that the galaxy is moving away from us. Slipher knew this and he tried to calculate the velocity at which the galaxy was moving away from us by manually calculating the redshifts from the image. What he observed was astonishing! The galaxy was moving at a velocity 1000's of kilometers per **seconds** (not hour). We can travel around the earth in seconds at such a huge speed. This is a supersonic baby!!

![super sonic baby!](/images/random/sonic_running.png)

That is beautiful! We can get the relative velocity of an astronomical object by measuring the shift in wavelength caused due to the doppler effect. It is experimentally observed that most of the galaxies are redshifted, meaning, moving away from us. This could mean that the universe is expanding! Mystery solved!!! But just when you thought we are done...

![bomb cat](/images/random/bomb_cat.png)

There is one more question before we say for final that the universe is expanding.

> Are we in some special place in the universe? Are we at the center and everything around us is expanding?

Well, Copernicus said that we shouldn't think of ourselves as a special place in the universe but just a perfectly average place in the universe but our observations seem contrary!

Let's answer these final questions in the following sections!

### Hubble's Law

![mah man smokin' a pipe](https://upload.wikimedia.org/wikipedia/commons/1/15/Studio_portrait_photograph_of_Edwin_Powell_Hubble_%28cropped%29.JPG)

The question of us being in a special place was answered by none other than Edwin Hubble. Edwin Hubble has the access to the largest telescope back then in 1915-1920. He observed the Slipher's nebulae and saw that it was composed of stars. So Hubble realized that the nebulae were galaxies as we appreciate them today.

Hubble has this idea to compare how bright stars would be in one galaxy and compare how big stars are in another galaxy under the assumption that the brightest star in each galaxy would be about the same brightness. So if the brightest stars in each galaxy had the same brightness, then we can measure the distance between them using the **inverse square law** ($\phi \propto r^{-2}$). So, Hubble did this for a bunch of Slipher's galaxies and in 1929 he showed the world his results.

![hubble's law aka amazing!](/images/astronomy_files/hubble_law.jpg)

This is arguably the most important result in the history of cosmology! What the plot shows is the distance along the X-axis and velocity along the Y-axis. It seems that the objects are moving faster and faster as we move farther and farther away from the earth. This, rather fuzzy, relationship is quite interesting even though it doesn't entirely fit on the graph. It because of the assumptions that Hubble made which makes the relationship look weak. Let's see what the modern graphs have to say.

![modern hubble's graph](/images/astronomy_files/modern_hubble.jpg)

As you can see, this graph explores the velocities as great as $4 \times 10^{4}km/s$ of astronomical objects as far as $700 Mpc$ or $2.170 \times 10^{22}km$ relative to us. You can see the relationship very clearly in this graph.

What this means is that everything around us is moving away from us and it gets faster as we move farther away into the depths of the universe. What this is telling us is that every other galaxy in the universe, pretty much, is moving away from us, and their speeds can be staggering. But how fast they're going away doesn't depend upon what sort of galaxy, it doesn't matter if you're a big galaxy or a small galaxy. **All it seems to depend on is how far it is away from us**.

> This means that we are, indeed, in a special place of the universe, or are we \>:)

Let's keep that question for sometime later...

![hubble deep field](/images/astronomy_files/hubble_deep_field.gif)

### Refrences

1. [Astrophysics on EdX](https://www.edx.org/xseries/astrophysics)
2. [The Solar-Stellar Spectrograph](http://www2.lowell.edu/users/jch/sss/article.php?r=t_datared_d_spectrum)
3. [A relation between distance and radial velocity among extra-galactic nebulae](https://www.pnas.org/content/15/3/168)
4. [Hubble's diagram and cosmic expansion](https://www.pnas.org/content/101/1/8)
5. [Doppler Effect](https://en.wikipedia.org/wiki/Doppler_effect)
6. [Vesto Slipher](https://en.wikipedia.org/wiki/Vesto_Slipher)
7. [Edwin Hubble](https://en.wikipedia.org/wiki/Edwin_Hubble)
