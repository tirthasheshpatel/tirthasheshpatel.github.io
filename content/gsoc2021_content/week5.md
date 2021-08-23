Title: Week #5: First Phase Ends...
Author: Tirth Patel
Date: 2021-07-12
Category: GSoC 2021
Tags: gsoc2021
Alias: /gsoc2021/week5.html /gsoc2021/week5/index.html

<h2>What did you do this week?</h2>

The PR resolving the memory leaks is still up for reviews. While it was being reviewed, I thought it would be nice to think ahead of time and look at the potential performance of the methods I propose to add. So, I created a new branch <a href="https://github.com/tirthasheshpatel/scipy/tree/gsoc-unuran-bench"><code>gsoc-unuran-bench</code></a> on my fork and started wrapping the remaining methods I had proposed in the excel sheet that I wrote a couple of weeks ago. I then wrote a <a href="https://github.com/tirthasheshpatel/scipy/blob/gsoc-unuran-bench/unuran_perf.py">small Python script</a> to benchmark all the wrapped methods against NumPy random number generators. For now, I have only used two distributions: Standard Normal and Beta(2, 3). I plan to add more in the following weeks. Sampling was run 3 times per measurement. The results of the benchmark:

<ul>
<li>UNU.RAN's methods (namely <code>NumericalInversePolynomial</code> and <code>AutomaticRatioOfUniforms</code>) were 3x faster than the NumPy RNG for the Beta(2, 3) distribution.</li>
<li>NumPy RNG was slightly faster than UNU.RAN's methods (with <code>NumericalInversePolynomial</code> and <code>AutomaticRatioOfUniforms</code> being the closest to the performance of the NumPy RNG) to sample from the Standard Normal distribution.</li>
</ul>

It is good to see that there is a possibility of improving the performance of sampling from some distributions once the methods from UNU.RAN are integrated in SciPy.

<h2>What is coming up next?</h2>

There are already some reviews on the PR resolving memory leaks and I hope by the end of the next week, there would be even more and we could decide whether we want to use that approach in SciPy. It's a tricky and non-conventional approach so I am not sure how many reviews would be considered "enough" or how much time will it take for the maintainers to properly review it. But while that is going on, I hope to start wrapping methods to sample from discrete distributions and benchmark them against the NumPy RNG.

<h2>Did you get stuck anywhere?</h2>

No. This was more or less a smooth week...

<h2>Marking the end of Phase 1</h2>

GSoC page says that the first phase reviews would start from tomorrow. So, this seems like a good time to summarize all the progress of Phase 1 here:

<ul>
<li>PR filed: Tests pass.</li>
<li>SciPy builds with UNU.RAN.</li>
<li>Separated UNU.RAN in its own submodule.</li>
<li>Create wrappers for one continuous and one discrete generator.</li>
<li>Basic benchmarks written.</li>
<li>Basic tests written.</li>
<li>A strong documentation suite and tutorials written.</li>
<li>Extra benchmarks written on my fork for all continuous methods in UNU.RAN.</li>
</ul>

According to my proposal, I have successfully achieved my first and second milestone and encroached the third milestone ahead of time! Most of the points above are still under reviews and might be changed in the future if need be. It would still be a challenge, both for the maintainers and me, to resolve the memory leaks issue but I hope that is done before the end of the second phase so that we can merge and test out some of the new functionality and iterate on the design. Let's hope the best for what's coming!
