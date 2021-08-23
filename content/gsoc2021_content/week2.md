Title: Week #2: Working on the PR
Author: Tirth Patel
Date: 2021-06-20
Category: GSoC 2021
Tags: gsoc2021
Alias: /gsoc2021/week2.html, /gsoc2021/week2/index.html

<h2>What did you do this week?</h2>

This week was spent mostly polishing my pull request and addressing reviews. I got a few big things out of the way though. Firstly, I refactored the API to accept a single <code>dist</code> object containing all the required methods. Secondly, I wrote a tutorial to document the usage of the new API. And lastly, I wrote a benchmark suite to profile the setup and sampling stage of each sampler. Moreover, using a lot of help from Bas (@BvB93 on GitHub), I was able to resolve all the MyPy static typing errors and get the MyPy check passing. While adding tests for the <code>seed</code> parameter, I noticed that I had made a mistake in handling the old NumPy <code>RandomState</code> API. As I had used global variables to sample from the NumPy RNG, seeding a generator broke! This was because the underlying (global) NumPy RNG was overridden by a new RNG as soon as a new generator with a seed was created. Thankfully, I quickly found a way to avoid the use of global variables and tests started passing again. One of my mentors, Christoph, was interested in using the UNU.RAN's test suite to write strong tests in SciPy. I have started looking into its test suite and also ported a few tests but this is still a work in progress.

<h2>What is coming up next?</h2>

I have got a lot of work done on the PR and it's shaping nicely: Main components of the PR have been written; Most tests pass. With this, I hope to mark the PR as open for reviews soon. I will have to make sure that I have added sufficient tests and documentation. Also, the new code lacks comments which may give reviewers a difficult time. I aim to clean out the newly added code and write more comments to delineate certain parts that might be tricky to understand. I also need to clean up the license file. There was also interest in separating UNU.RAN in a submodule. I hope to address some of these points in the upcoming week.

<h2>Did you get stuck anywhere?</h2>

I faced a weird 32-bit Linux failure which was related to my changes. When the <code>randint</code> distribution is input to the DAU method, it fails with an "unknown error" in UNU.RAN. I was able to localize the error but failed to find a reason for the failure. I suspect floating-point errors but a deeper inspection needs to be done. For the time being, as this isn't inside SciPy (and also only exists on a very specific platform and an old NumPy version), I have skipped that test case. This also led to a squalid revelation: memory leaks :/ This is turning into more of a can of worms than I had initially expected. Sometimes UNU.RAN frees allocated memory <i>after</i> calling the error handler. But the error handler is designed to jump out of the C code and return to the Cython code where the error can be raised safely. But, then, the allocated memory is never freed leading to a memory leak. I am not sure how often this happens. But it might be something to investigate in more depth. I will see if this is substantial and look into what can be done.
