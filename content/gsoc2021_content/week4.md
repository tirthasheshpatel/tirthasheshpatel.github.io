Title: Week #4: Memory Leaks :(
Author: Tirth Patel
Date: 2021-07-04
Category: GSoC 2021
Tags: gsoc2021
Alias: /gsoc2021/week4.html /gsoc2021/week4/index.html

<h2>What did you do this week?</h2>

<p>This week was, by far, the <i>most</i> challenging week for me! Turns out, resolving the memory leaks issue is not as simple as patching UNU.RAN and changing the order of frees and calls to the error handler :/ I was able to discover this because of a very helpful suggestion by Nicholas (one of my mentors) to run <code>valgrind</code> on my changes (see the discussion on <a href="https://github.com/tirthasheshpatel/unuran/pull/1">tirthasheshpatel/unuran#1</a>).</p>

<pre>==25301== LEAK SUMMARY:
==25301==    definitely lost: 13,728 bytes in 58 blocks
==25301==    indirectly lost: 43,520 bytes in 387 blocks
==25301==      possibly lost: 166,439 bytes in 114 blocks
==25301==    still reachable: 3,553,979 bytes in 2,254 blocks
==25301==         suppressed: 0 bytes in 0 blocks
</pre>

<p>These memory leaks occur due to the non-local returns in the thunk and in the error handler. Unfortunately, this behavior is ingrained in the way I have designed the API to handle errors and a major refactor is required to fix the issue! <a href="https://github.com/tirthasheshpatel/scipy/pull/9">tirthasheshpatel/scipy#9</a> is my first (and hopefully last) attempt aimed at doing this. It refactors the Cython wrapper to use the <code>MessageStream</code> API to handle the errors occurring in UNU.RAN and <code>PyErr_Occurred()</code> to detect the errors occurring in Python callbacks.</p>

<p>The MessageStream API was introduced and written by Pauli Virtanen (@pv on GitHub) while writing wrappers for Qhull. <code>MessageStream</code> uses <code>FILE *</code> streams to log errors occurring in the C API to a temporary file (either in memory or on disk depending on whether <code>open_memstream</code> is available). Once the execution of the C function is complete, one can check the file for errors and raise them in Python. One of the downsides of this approach is that UNU.RAN contains a global <code>FILE *</code> variable which is not thread-safe. Hence, thread-safety needs to be provided in the Cython wrapper itself which further complicates things. I have used a module-level lock which is acquired every time before calling the <code>unur_set_stream</code> function (which is responsible for changing the global <code>FILE *</code> variable) and is released once the required set of C functions have been evaluated. Finally, <code>valgrind</code> seems to be happy with this and reports no memory leaks!</p>

<pre>==44175== LEAK SUMMARY:
==44175==    definitely lost: 1,128 bytes in 11 blocks
==44175==    indirectly lost: 0 bytes in 0 blocks
==44175==      possibly lost: 195,264 bytes in 124 blocks
==44175==    still reachable: 3,258,070 bytes in 2,112 blocks
==44175==         suppressed: 0 bytes in 0 blocks
</pre>

<p>(The <code>1128</code> lost bytes are inside <code>dlopen.c</code> library and not UNU.RAN wrappers.)</p>

<p>A very similar thing has been done by Pauli Virtanen in Qhull and getting his help and reviews on <a href="https://github.com/tirthasheshpatel/scipy/pull/9">tirthasheshpatel/scipy#9</a> would be invaluable! I hope that my approach is correct this time and the whole fuss about memory leaks resolves as soon as possible.</p>

<h2>What is coming up next?</h2>

The issue I delineate in the previous section is the bottleneck blocking <a href="https://github.com/scipy/scipy/pull/14215">#14215</a> from receiving more reviews and merging. I hope to continue my work on <a href="https://github.com/tirthasheshpatel/scipy/pull/9">tirthasheshpatel/scipy#9</a> and get it approved and/or merged by the end of this or next week. I also hope this doesn't bring more surprises down the line and I can continue my work on <a href="https://github.com/scipy/scipy/pull/14215">#14215</a> more smoothly!

<h2>Did you get stuck anywhere?</h2>

Not yet... :)
