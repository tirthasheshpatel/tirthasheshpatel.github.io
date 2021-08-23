Title: Week #6: More Benchmarks
Author: Tirth Patel
Date: 2021-07-18
Category: GSoC 2021
Tags: gsoc2021
Alias: gsoc2021/week6.html gsoc2021/week6/index.html

<h2>What did you do this week?</h2>

I continued writing benchmarks for UNU.RAN methods to sample from continuous distributions. The results of these benchmarks are quite promising. They outperform the default <code>rvs</code> method in SciPy on an average by 100x better performance. In some cases, when the PDF/CDF are expensive to evaluate, UNU.RAN methods are 10000 times faster than the default rvs method. It would a nice future project to add a specialized rvs methods for some distributions where UNU.RAN performs significantly better than SciPy. For more details and some pretty plots, please look at <a href="https://github.com/tirthasheshpatel/scipy/pull/10">tirthasheshpatel#10</a>. With the help of Ralf Gommers, UNU.RAN submodule got transfered under the SciPy organization this week. The new submodule is now present at <a href="https://github.com/scipy/unuran">scipy/unuran</a>. 

<h2>What is coming up next?</h2>

I am hoping to get some reviews on my PR over the period of next week. Other than that, I don't have anything specific to do this week. I will maybe benchmark UNU.RAN method to sample from discrete distributions...

<h2>Did you get stuck anywhere?</h2>

Nope. Thankfully, another smooth week.
