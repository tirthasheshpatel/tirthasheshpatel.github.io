---
layout: post
title: A review of Process Scheduling in Operating Systems
category: Operating Systems Notes
tags: [process, scheduling, os]
---

### Abstract

In this article, I have collected and reviewed various algorithms present for short term scheduling. The first section introduces the concept of short term scheduling and its importance in efficient CPU utilization. In further sections, I introduce the scheduling algorithms and thier efficiency. This article ends with techniques for searching the values for free parameters present in various scheduling algorithms introduced in former sections. You may want to refer to my previous article on process creation to get a idea of the concept of processes in operation systems.

### 1. Introduction

Let's start where we left off previously:

> The objective of
multiprogramming is to have some process running at all times, to maximize CPU utilization.

Several processes are kept in memory at the same time while a processor can only execute one process at an instance. This causes a need for a mechanism to choose a process such that all the processes in memory get a fair amount of time to execute and terminate in finite time. This problem is known in all of computer science since years and is referred to as the scheduling problem. This problem can be viewed as both CSP (constraint satisfiction problem) or CAP (constrain optimization problem) according to the type of algorithm that a particular operation system chooses to schedule its processes.

#### CPU-I/O burst cycles

Process execution begins with a CPU burst.
That is followed by an I/O burst, which is followed by another CPU burst, then another I/O burst, and so on. The duration of CPU burst has been studied extensivel. Although the time taken by a process to execute may be highly system dependent, a pattern in the frequency curve is observed which is generally caracterisized as a exponential or hyperexponential distributions.

![cpu burst distribution](/images/os-process/cpu-burst-distribution.jpg)

An I/O-bound program typically has many short CPU bursts. A CPU-bound program might have a few long CPU bursts. This distribution can be important in the selection of an appropriate CPU-scheduling algorithm.

#### CPU Scheduler

***Definition 1**: The process of selecting a process from the ready queue to be allocated a CPU for execution is known as **short term scheduling**.*

A ready queue can be implemented as a FIFO queue, a priority queue, a tree or simply an unordered linked list.

#### Preemptive Scheduling

CPU scheduling decisions may take place under forllowing circumstances:

1. When a process switches from the running state to waiting state.
2. Whena process switches from running state to the ready state.
3. When a process switches from the waiting state to the ready state.
4. When a process terminates.


