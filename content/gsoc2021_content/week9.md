Title: Week #9: CI failure on Python3.7-dbg
Author: Tirth Patel
Date: 2021-08-09
Category: GSoC 2021
Tags: gsoc2021
Alias: /gsoc2021/week9.html /gsoc2021/week9/index.html

<h2>What did you do this week?</h2>

I addressed more reviews this week and added pickling support to all the generators. Most of the changes focused on the documentation, tutorials, and the UI. We removed the <code>params</code> keyword (used to pass positional arguments to the passed callable) from the method constructor and had a discussion on renaming the <code>seed</code> parameter to something like <code>numpy_rng</code>. I also added some histogram plots to the docs and tutorials. Lastly, as all the generators are Cython extensions classes with a non-trivial <code>__cinit__</code> method, Cython was not able to generate code to pickle object automatically. Hence, I added a rudimentary <code>__reduce__</code> method which enables pickling support. Relevant tests have also been added.

<h2>What is coming up next?</h2>

We need to finalize renaming <code>seed</code> to something more apposite like <code>rng</code> or <code>numpy_rng</code>. All other blocking comments have been addressed, except a recent failure on CI. I will investigate the cause of the failure and try to resolve it by the end of this week. As the GSoC deadline is imminent, I hope to complete that by next week so we could merge the PR.

<h2>Did you get stuck anywhere?</h2>

I noticed a <a href="https://github.com/scipy/scipy/runs/3273674972">few failures</a> on the linux workflows on Python3.7-dbg build. For some reason, Python crashes when TransformedDensityRejection is called with a invalid PDF. It seems like there is some internal logic error in the C/Cython code. As this is not a Python error with a proper traceback, it makes it very difficult to even locate the error, let alone solving it! I am thankfully able to replicate it locally and hope to figure out the failures as soon as possible.
