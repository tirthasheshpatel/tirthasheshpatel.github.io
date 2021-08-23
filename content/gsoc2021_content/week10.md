Title: Week #10: Finishing up the PR
Author: Tirth Patel
Date: 2021-08-15
Category: GSoC 2021
Tags: gsoc2021
Alias: /gsoc2021/week10.html, /gsoc2021/week10/index.html

<h2>What did you do this week?</h2>

<p>Last week I got stuck at some failures on Linux Python3.7-dbg (debug builds). I was able to replicate it locally and thankfully also found a fix for it. Turns out, I was accessing a Python attribute while a live exception was set in Python. This is not allowed because the Python objects of the class might have been destroyed or garbage collected and my attribute lookup might fail with a segmentation fault. This was caught by the debug build of Python 3.7 and was fairly simple to fix: just removed the offending line and the tests passed again! To ensure this doesn't happen again, I refactored some code to make more careful use of <code>PyErr_Occurred()</code>.</p>

<p>There was also some discussion around what to name the seed parameter: the new Generator API favors something other than <code>random_state</code> which has been well-established in SciPy. As there was no consensus among what to name it - some favored consistency while others wanted to differ between the new and the old API - we decided to go with the plain old <code>random_state</code> and change its name in some other PR at all places in SciPy.</p>

<h2>What is coming up next?</h2>

Almost all the reviews have now been addressed, the CI is green, and I am happy with the current state of the PR. It seems like a good starting point and if things stay this way, we might merge by the end of this week. Once that's done, adding new methods is fairly straightforward. I will try to add as many methods as possible before the end of GSoC.

<h2>Did you get stuck anywhere?</h2>

No blockers this week.
