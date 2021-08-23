Title: Week #0: Polishing the Prototype
Author: Tirth Patel
Date: 2021-07-06
Category: GSoC 2021
Tags: gsoc2021
Alias: gsoc2021/week0.html gsoc2021/week0/index.html

<h2>A little introduction</h2>

Hello everyone! I am Tirth, a last year computer science undergraduate student at Nirma University in India. I have been using NumPy and SciPy since I started doing scientific computing in my first year of college. I have been contributing to SciPy since last year and hope to continue to do so :). I will we working this summer to integrate UNU.RAN library in the <code>scipy.stats</code> submodule. UNU.RAN is a C library for Universal Non-Uniform RANdom number generation. It has been used in the ROOT project by CERN and R bindings for the library (<code>Runuran</code>) have also been created. My goal would be to integrate methods for sampling from univariate continuous and discrete distributions.

<h2>What did you do this week?</h2>

I got to know my mentors, Christoph and Nicholas, in the first week of the community bonding period. Since then, we have been meeting regularly to discuss the API and have been exchanging a lot of design ideas. Over the span of last three weeks, I have been able to significantly enhance my prototype to the point where I feel confident enough to propose a PR on SciPy. I started out with <a href="https://github.com/tirthasheshpatel/scipy/pull/5">tirthasheshpatel/scipy#5</a> on my fork which was thread-unsafe and made my way up to <a href="https://github.com/tirthasheshpatel/scipy/pull/6">tirthasheshpatel/scipy#6</a> which seems in a very good shape. It builds with UNU.RAN on all the required platforms and tests pass with an exception of a flaky failure. During the last week, I have created <a href="https://docs.google.com/spreadsheets/d/1r36HypXwpit7sHt9YAe3K7yPL7HNNuQjTkbVYQIJ4xI/edit?usp=sharing">this excel sheet</a> with some information on the methods I propose to add in SciPy. It also contains the information about the methods to add and parameters to keep, etc. It will help me with coding those methods in the coming weeks and also document the decisions properly.

<h2>What is coming up next?</h2>

As the PR on my fork builds and tests pass, I hope to create a PR on SciPy by next week. I also aim to circulate a mail in the Mailing List regarding the imminent PR and try to get feedback from other devs on the design of the API. Nevertheless, the coming weeks are critical to the work I aim to finish during GSoC so I hope to get things done without much contention!

<h2>Did you get stuck anywhere?</h2>

There have not been any serious blockers during the community bonding period but the Windows CI failed due to some unrelated Pythran errors. After a few abortive attempts to resolve them, the discussion on <a href="https://github.com/scipy/scipy/issues/13717">#13717</a> helped me fix the failing builds. As pointed out in <a href="https://github.com/scipy/scipy/issues/13717#issuecomment-852388251">this comment</a>, I was missing LLVM and MinGW binaries in the <code>PATH</code> which caused some weird linking problems for 64-bit builds. It was moment of relief to see builds passing on windows, since windows failures worried me the most. Hopefully, everything passes on the SciPy PR that I aim to create by the end of this week :). Fingers crossed!