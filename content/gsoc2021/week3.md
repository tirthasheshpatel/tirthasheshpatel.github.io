Title: Week #3: Separating UNU.RAN in its own submodule
Author: Tirth Patel
Date: 2021-06-27
Category: GSoC 2021
Tags: gsoc2021
Alias: gsoc2021/week3.html gsoc2021/week3/index.html

<h2>What did you do this week?</h2>

This week went by addressing last week's blockers and bifurcating UNU.RAN in its own submodule. I put the UNU.RAN source code in my PR which made that patch size over 100,000 LOC making it very difficult to review. So, I spent a couple of days creating a <a href="https://github.com/tirthasheshpatel/unuran">separate repository</a> for the UNU.RAN source code and using <code>git submodule</code> to clone it into SciPy. I also wrote a Python script to download and clean UNU.RAN for use in SciPy. This reduced the size of the patch from over 100,000 LOC to only about 4000 LOC. There were also some comments on the tutorial and benchmarks which I addressed this week.

<h2>What is coming up next?</h2>

Last week, I also noticed a memory leak so I decided to write a Python script to find more. The script reports 40 to 50 memory leaks throughout the source code but the part of the source code used by SciPy has only about 9 to 10 memory leaks. In the last meeting, we came up with two potential ways to get rid of them: (i) patch up UNU.RAN and reverse the order of frees and error, (ii) Instead of jumping out of the C function, use the return code of UNU.RAN functions to set and raise the error. The problem with approach (ii) is that some functions don't return an error code. So, I plan to test the first approach out this week and hopefully address all the memory leaks. We also decided to try out writing a higher-level API for the String API and get other devs opinions on it on the mailing list.

<h2>Did you get stuck anywhere?</h2>

No blockers this week!
