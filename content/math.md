Title: Writing some math using render_math plugin
Date: 2021-07-27 10:40
Category: Math
Tags: math, pelican-plugins, python, pelican
Author: Peach Bitch
Summary: Writing mathematical equations using this awesome plugin!!

## Normal Distribution

The PDF of the normal distribution is given by:

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean (`loc`) and $\sigma^2$ is the variance (`scale`) of the distribution.

### CDF and the Error Function

CDF of the Normal distribution is defined using the error function which doesn't have a solution in the closed form.

$$F(x) = \frac{1}{2}\left[1+\text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$$

where $\text{erf}(x)$ is the error function defined as:

$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{-\infty}^{x}e^{-t^2}dt$$

