Title: Week #1: Creating a (big) PR
Author: Tirth Patel
Date: 2021-06-12
Category: GSoC 2021
Tags: gsoc2021
Alias: gsoc2021/week1.html gsoc2021/week1/index.html

<h2>What did you do this week?</h2>

This week I submitted an overview of the progress on the mailing list (<a href="https://mail.python.org/pipermail/scipy-dev/2021-June/024878.html">here</a>) and created a pull request on SciPy (<a href="https://github.com/scipy/scipy/pull/14215">#14215</a>). Thankfully, all the tests pass and SciPy builds with UNU.RAN on all the required platforms! I also created some <a href="https://drive.google.com/file/d/1TH70SSvvc5eF6-YmDO8kFNvLqKaGROW-/view?usp=sharing">flowcharts</a> to elucidate the design of the internal API and manifest how callbacks are acquired and released. I also tried to write a higher-level API (<a href="https://github.com/tirthasheshpatel/scipy/pull/8">tirthasheshpatel/scipy#8</a>) as suggested by one of my mentors.

<h2>What is coming up next?</h2>

We have discussed quite a lot of points to keep me busy for a couple of weeks down the line :). Here it is:

 <ul>
  <li>Generate/build UNU.RAN tests and try integrating into SciPy test suite.</li>
  <li>Maybe figure out a way to speed up the performance on NumPy < 1.19.</li>
  <li>Write better/stronger tests.</li>
  <li>Mock up API that uses same object interface i.e. bundle all functions together in a <code>dist</code> parameter.</li>
  <li>Address code reviews on my PR.</li>
  <li>Add relation [of the UNU.RAN API] to the <code>rv_discrete</code> and <code>rv_continuous</code> classes in tutorial. Add in docs that <code>rvs</code> of UNU.RAN methods and SciPy distributions differ.</li>
</ul>

<h2>Did you get stuck anywhere?</h2>

No blockers this week!
