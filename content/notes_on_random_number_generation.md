Title: Universal Methods for Random Variate Generation
Author: Tirth Patel
Date: 2021-07-29 17:30
Category: RNG
Tags: rng, maths, proabability theory
Modified: 2021-07-29 17:45

<h1>Rejection Methods</h1>

<h2>Transformed Density Rejection</h2>

* Most universal algorithms are **very slow** compared to algorithms that are specialized to that distribution. Algorithms that are fast have a **slow setup** and **large tables**.
* The aim is to provide a universal method which is **not too slow** and needs only a **short	 setup**.
* This method can be applied to **continuous unimodal** distributions with **bounded densities** which need to be **T-concave**.	
* **Algorithm (in words)**:
    * Transform the density using a suitable function $T(x)$ defined for $x > 0$.
    * Define $h(x) = T(f(x))$ and a piecewise linear function $l(x)$ such that $l(x) \ge h(x)$ for all $x$ in the support of $f$. $T^{-1}l(x)$ is then a dominating function for f(x) and rejection can be used to sample from the desired distribution.
    * Restrict $h(x) = T(f(x))$ to be concave.
    * Define $l(x)$ as a minimum of three lines touching $h(x)$ in the mode $m$, in $x_l < m$, and $x_r > m$. It is given by:
        * $l(x) = \min{(h(x_l) + h^{\prime}(x_l)(x - x_l), h(m), h(x_r) + h^{\prime}(x_l)(x - x_r))}$
    * **Constraints:**
        * **$\lim_{x \to 0} T(x) = -\infty$**
        * $T(x)$ is differential and $T^{\prime}(x) > 0$ which implies that $T^{-1}$ exists
        * $\int_{0}^{\infty} T^{-1}(h(m)-x)dx < \infty$
        * $h(x) = T(f(x))$ must be concave
    * **Assumptions** (in practice but not in theory):
        * Define $F(x) = \int{T^{-1}(x)dx}$
        * $F^{-1}$ exists.
        * $\lim_{x \to \infty}F(x) = 0$
    * Now to use rejection, we need to compute two intersection points $b_l$ and $b_r$ of the three parts of $l(x)$ and we also need to compute the areas between x-axis and $T^{-1}(l(x))$ (i.e. CDF) for the three intervals $v_l, v_c, v_r$. 
    * Sampling from the density $T^{-1}(l(x))$ is done by inverting the CDF of the left and right tail of the distribution. The middle part remains constant.
    * For many distributions, evaluation of PDF is expensive. So, we create a squeeze function that can be used as a proxy to the PDF. The squeeze function is defined piecewise as the line joining $x_l$ to $m$ and $m$ to $x_r$ (dashed lines in the figure). As the PDF is T-concave, the squeeze will always lie below the PDF.
* The figure shows:
    * $h(x)$ (left curve, thick line)
    * $l(x)$ (left curve, thin line)
    * $f(x)$ (right curve, think line)
    * $T^{-1}(l(x))$ (right curve, thin line)
    * Squeezes in dashed lines

![alt_text](/images/rng/image1.png)

* **Algorithm (mathematical):**
    * **Input**: $f(x)$ (<span style="text-decoration:underline;">density function</span> of the distribution. It doesn’t need to be normalized), $f^{\prime}(x)$ (<span style="text-decoration:underline;">derivative of the density function</span> w.r.t $x$), $T(x)$ (<span style="text-decoration:underline;">transformation function</span>)
    * **Step 1: Setup**
        * **$h(x) \leftarrow T(f(x))$** and $h^{\prime}(x) \leftarrow T^{\prime}(f(x))f^{\prime}(x)$
        * $F(x) \leftarrow \int{T^{-1}(x)dx}$
        * $m \leftarrow \text{ mode of the distribution}$
        * $i_l \leftarrow \inf\{x\mid f(x)>0\}, i_r\leftarrow \sup\{x\mid f(x)>0\}$
        * Choose $x_l \in (i_l,m)$ and $x_r \in (m, i_r)$
        * $b_l \leftarrow x_l + \frac{h(m) - h(x_l)}{h^{\prime}(x_l)}$
        * $b_r \leftarrow x_r + \frac{h(m) - h(x_r)}{h^{\prime}(x_r)}$
        * $v_l \leftarrow \frac{F(h(m))-F(h^{\prime}(x_l)(i_l-x_l)+h(x_l))}{h^{\prime}(x_l)}$
        * $v_c \leftarrow f(m)(b_r-b_l)$
        * $v_r \leftarrow \frac{F(h^{\prime}(x_r)(i_r-x_r)+h(x_r))-F(h(m))}{h^{\prime}(x_r)}$
        * $s_l \leftarrow \frac{h(m)-h(x_l)}{m-x_l}$
        * $s_r \leftarrow \frac{h(m)-h(x_r)}{m-x_r}$
    * **Step 2: Sampling**
        * Generate a uniform random number $U$
        * $U \leftarrow U(v_l + v_c + v_r)$
        * if $U \le v_l$
            * $l_x\leftarrow F^{-1}(F(h(m))-Uh^{\prime}(x_l))$
            * $X\leftarrow\frac{l_x-h(x_l)}{h^{\prime}(x_l)}+x_l$
        * elif $U \le v_l + v_c$:
            * $l_x\leftarrow h(m)$
            * $X\leftarrow \frac{U-v_l}{v_c}(b_r-b_l)+b_l$
        * else:
            * $l_x\leftarrow F^{-1}((U-(v_l+v_c))h^{\prime}(x_r)-F(h(m)))$
            * $X\leftarrow\frac{l_x-h(x_r)}{h^{\prime}(x_r)}+x_r$
        * $l_x\leftarrow T^{-1}(l_x)$
        * Generate a uniform random number $V$
        * $V \leftarrow Vl_x$
        * if $X < m$
            * if $X > x_l \text{ and } V \le T^{-1}(h(m)-(m-x)s_l)$:
                * **return X**
        * elif $X<x_r \text{ and } V \le T^{-1}(h(m)-(m-x)s_r)$:
            * **return X**
        * if $V>f(X)$:
            * reject the sample and restart **Step 2**.
        * else **return X**
* The algorithm is left with choosing $x_l$ and $x_r$. 
* Define $t_o = -\frac{F(T(1))}{F(T(1)-F(T(1)))-F(T(1))}$
* **Theorem**: With the choice of $x_l=m-\frac{t_o}{f(m)}$ and $x_r=m+\frac{t_o}{f(m)}$, the number of iterations of the TDR algorithm is lower or equal to $2t_o$ for **arbitrary** T-concave distributions. (_Theorem 2 in the paper_)
* As the **number of iterations of TDR are bounded uniformly over the class of all T-concave distributions with arbitrary support**, **it makes TDR a good candidate for automatic generation**.
* **Choice of the Transformation Function**:
    * The two most important choices for implementation are:
        * $T_0(x) = \log(x)$
        * $T_{-0.5}(x) = -\frac{1}{\sqrt{x}}$
    * In general, $T_c(x) = -x^{c}$ defines a family of such transformation functions that satisfy all our constraints.
    * $T_0$ and $T_{-0.5}$ are used because $T^{-1}, F, F^{-1}$ are very simple to obtain and compute.
    * Any PDF that is $T_{c_1}$ concave is always $T_{c_2}$ concave for all $c_2 \le c_1$. The table below shows the maximum T-concavity of some standard distributions.


![alt_text](/images/rng/image2.png)

* Also, it is enough for TDR algorithm to know the densities upto proportionalities i.e. <span style="text-decoration:underline;">the PDF need not integrate to 1</span>. **Hence, we can ignore the normalization constant while computing the PDF.**
* As compared to the Ratio-Of-Uniform methods, **TDR leads to fewer iterations per sample** with optimal points of contact. For example, expected iterations per sample for sampling from the Cauchy distribution using RoU method is around 1.27 while it is just around 1.1 for TDR (with optimal points of contact).

<h2>Universal Transformed Density Rejection</h2>

* Universal Transformed Density Rejection addresses the following points which aren’t addressed by Transformed Density Rejection:
1. Choice of $x_l$ and $x_r$ is sometimes **much too far away from the mode** and leads to a higher number of iterations per sample.
2. The **derivative can be inconvenient/slow** to compute. So, we should try to approximate the tangent at $x_l$ and $x_r$ without the need to compute the exact derivatives.
3. Using equations of **Theorem 2**, $x_l$ and $x_r$ can lie outside the support of the distribution, in which case, we will have to set the areas of the tails to 0 i.e. $v_l \leftarrow 0$ and $v_r \leftarrow 0$ and set the intersection points $b_l \leftarrow i_l$ and $b_r \leftarrow i_r$. This way, **we lose the squeeze function** and need to perform expensive PDF evaluation at each iteration which slows down the algorithm. 


![alt_text](/images/rng/image3.png)


![alt_text](/images/rng/image4.png)


* Universal TDR (UTDR) **fixes the transformation function** to $T_{-0.5}(x)$ which is $-\frac{1}{\sqrt{x}}$ and addresses the above points using the following method:
1. We start with a choice of $t_o = 0.664$ (which is the optimal value for the normal distribution as shown in the table above) and calculate the area under the dominating curve. If the area is less than 4, we continue with the choices of $x_l = m - \frac{t_o}{f(m)}$ and $x_r = m - \frac{t_o}{f(m)}$. But if the area is greater than 4, we fall back to the original constant $t_o$ for $T_{-0.5}(x)$, which is $2$ (see the table above).
2. We can approximate the derivative as $\frac{h(x_l+\Delta)-h(x_l)}{\Delta}$. The same can be done for $x_r$ with $-\Delta$. The choice of $\Delta$ is made in such a way that the precision is only lost upto maximum 5 decimal digits.
3. We just define the missing point for the squeeze function as a point whose distance from $m$ is 60% of the distance between $m$ and the border of the distribution.
* **Algorithm**:
    * **Step 1: Setup**
        * **$m \leftarrow \text{mode of the distribution}$**
        * $f_m \leftarrow f(m), h_m \leftarrow -\frac{1}{\sqrt{f_m}}$
        * $i_l \leftarrow \inf\{ x \mid f(x)>0 \}$
        * $i_r \leftarrow \sup\{ x \mid f(x)>0 \}$
        * if $i_l = \infty$, $t_l \leftarrow 0$ else $t_l \leftarrow 1$
        * if $i_r= \infty$, $t_r \leftarrow 0$ else $t_r \leftarrow 1$
        * $c \leftarrow 0.664$
    * **Step 1.1: Extra Setup**
        * **$c\leftarrow \frac{c}{f_m}$**, $x_l \leftarrow m-c$, $x_r\leftarrow m+c$
        * if $t_l=1 \text{ and } x_l < i_l$:
            * $b_l \leftarrow i_l, v_l \leftarrow 0$
            * if $i_l < m \text{ then } x_l \leftarrow m-(m-i_l)0.6$
            * $s_l \leftarrow \frac{h_m+\frac{1}{\sqrt{f(x_l)}}}{m-x_l}$
        * else:
            * $\tilde{y_l} \leftarrow -\frac{1}{\sqrt{f(x_l)}}$
            * $s_l \leftarrow \frac{h_m-\tilde{y_l}}{m-x_l}$
            * $\Delta \leftarrow \max(|x_l|, -\frac{\tilde{y_l}}{s_l}) \dot 10^{-5}$
            * $y_l \leftarrow -\frac{1}{\sqrt{f(x_l+\Delta)}}$
            * $a_l \leftarrow \frac{y_l-\tilde{y_l}}{\Delta}$
            * $b_l \leftarrow x_l\frac{h_m-y_l}{a_l}$
            * $d_l \leftarrow y_l-a_lx_l$
            * $v_l \leftarrow -\frac{1}{a_lh_m}, c_l \leftarrow v_l$
            * if $t_l=1 \text{ then } v_l \leftarrow v_l+\frac{1}{a_l(a_li_l+d_l)}$
        * Symmetric if...else statement for calculation of $x_r$, $b_r$, $a_r$, $d_r$, $v_r$, and $y_r$.
        * $v_c \leftarrow (b_r-b_l)f_m$
        * $v_{lc} \leftarrow v_l+v_c$
        * $v_t \leftarrow v_{lc}+v_r$
        * if $v_t > 4 \text{ then } c\leftarrow 2$ and **goto** **Step 1.1**
    * **Step 2: Sampling**
        * Generate a random uniform number $U$
        * $U \leftarrow Uv_t$
        * if $U < v_l$:
            * $X \leftarrow -\frac{d_l}{a_l}+\frac{1}{a_l^2(U-c_l)}$
            * $l_x \leftarrow (a_l(U-c_l))^2$
        * elif $U < v_{lc}$:
            * $X \leftarrow \frac{(U-v_l)(b_r-b_l)}{v_c}+b_l$
            * $l_x \leftarrow f_m$
        * else:
            * $X\leftarrow -\frac{d_r}{a_r}-\frac{1}{a_r^2(U-v_{lc}-c_r)}$
            * $l_x \leftarrow (a_r(U-v_{lc}-c_r))^2$
        * Generate a uniform random number $V$
        * $V \leftarrow Vl_x$
        * if $X < m$:
            * if $X \ge x_l \text{ and } V(h_m-(m-X)s_l)^2 \le 1$ **return X**
        * elif $X \le x_r \text{ and } V(h_m-(m-X)s_r)^2 \le 1$ **return X**
        * if $V \le f(X)$ **return X**
        * else **goto Step 2**

<h2>Inverse Transformed Density Rejection</h2>




* TODO

<h2>Adaptive Rejection Sampling</h2>




* TODO

<h2>Simple Setup Rejection</h2>




* TODO

	

<h1>Ratio-of-Uniforms Methods</h1>


<h2>Automatic Ratio-of-Uniforms</h2>




* TODO

<h2>Naive Ratio-of-Uniforms</h2>




* TODO

<h2>Simple Ratio-of-Uniforms</h2>




* TODO

<h1>CDF Inversion Methods</h1>


<h2>Fast Inversion using Hermite Interpolation</h2>




* TODO

<h2>Fast Inversion using Polynomial Interpolation</h2>




* TODO

<h2>Numerical Inversion</h2>




* TODO
