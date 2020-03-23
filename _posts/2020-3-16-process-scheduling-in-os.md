---
layout: post
title: Chapter 6 - Process Scheduling in Operating Systems
category: Operating Systems Notes
tags: [process, scheduling, os]
---

### Abstract

In this article, I have collected and reviewed various algorithms present for short term scheduling. The first section introduces the concept of short term scheduling and its importance in efficient CPU utilization. In further sections, I introduce the scheduling algorithms and thier efficiency. This article ends with techniques for searching the values for free parameters present in various scheduling algorithms introduced in former sections. You may want to refer to my previous article on process creation to get a idea of the concept of processes in operation systems.

### 1. Introduction

Let's start where we left off previously:

> The objective of multiprogramming is to have some process running at all times, to maximize CPU utilization.

Several processes are kept in memory at the same time while a processor can only execute one process at an instance. This causes a need for a mechanism to choose a process such that all the processes in memory get a fair amount of time to execute and terminate in finite time. This problem is known in all of computer science since years and is referred to as the scheduling problem. This problem can be viewed as both CSP (constraint satisfiction problem) or CAP (constrain optimization problem) according to the type of algorithm that a particular operation system chooses to schedule its processes.

#### CPU-I/O burst cycles

Burst!!! Let the bomb cat handle this!

![bomb cat will blow this bitch up](/images/random/bomb_cat.png)

Process execution begins with a CPU burst. That is followed by an I/O burst, which is followed by another CPU burst, then another I/O burst, and so on. The duration of CPU burst has been studied extensivel. Although the time taken by a process to execute may be highly system dependent, a pattern in the frequency curve is observed which is generally caracterisized as a exponential or hyperexponential distributions.

![cpu burst distribution](/images/os-process/cpu-burst-distribution.jpg)

An I/O-bound program typically has many short CPU bursts. A CPU-bound program might have a few long CPU bursts. This distribution can be important in the selection of an appropriate CPU-scheduling algorithm.

#### CPU Scheduler

***Definition 1**: The process of selecting a process from the ready queue to be allocated a CPU for execution is known as **short term scheduling**.*

A ready queue can be implemented as a FIFO queue, a priority queue, a tree or simply an unordered linked list.

#### Preemptive Scheduling

CPU scheduling decisions may take place under forllowing circumstances:

1. When a process switches from the running state to waiting state.
2. When a process switches from running state to the ready state.
3. When a process switches from the waiting state to the ready state.
4. When a process terminates.

When scheduling takes place only under the circumstances 1 and 4, we say that the scheduling scheme is **nonpreemptive** or **cooperative**. Otherwise, it's **preemptive**.

Challenges of preemptive scheduling:

1. Preemptive scheduling can result in race conditions when data are shared among several processes.
2. A process may be preempted while changing some important kernel data and lead to CHAOS!
3. The **sections of data**/**device** (errata in the book) affected by interrupts must be guarded from simultaneous use.

#### Dispatcher

The dispatcher is the module that gives control of the CPU to the process selected by the short term scheduler. The involves the following:

- Switching context.
- Switching to user mode.
- Jumping to the proper location in the user program to restart the program.

The dispatcher should be as fast as possible, since it is invoked during every process switch. The time it takes for the dispatcher to stop one process and start another running is known as the **dispatch latency**.

### Scheduling Criteria

1. **CPU Utilization**: The CPU must be utilized ALL the time. Hammer the CPU baby!!
2. **Throughput**: One measure of work is the number of processes completed per unit time called **throughput**.
3. **Turnaround time**: The amount of time between start and termination of the process is called turnaround time. Turnaround time is the sum of the periods spent waiting to get into memory, waiting in the ready queue, executing on the CPU, and doing I/O.
4. **Waiting time**: Waiting time is the sum of all the periods spent waiting in different scheduling queues like the ready queue.
5. **Response time**: Another measure is the time from the submission of a request until the first response is produced. This measure, called response time, is the time it takes to start responding, not the time it takes to output the response. The turnaround time is generally limited by the speed of the output device.

#### First-Come, First-Serve Scheduling

> The process that requests CPU first, gets the CPU first.

It is a **non-preemptive** scheduling algorithm. Meaning, new process is allocated a CPU only when the process running either terminates or goes into waiting queue for I/O.

**Example**: Suppose processes $P_1$, $P_2$, and $P_3$ have $24ms$, $3ms$, $5ms$ CPU burst time respectively. Let's consider the following situations:

- $P_1$, $P_2$, $P_3$: Hence, the waiting time for $P_1$ is $0ms$, for $P_2$ is $24ms$ and for $P_3$ is $27ms$. Hence, the average waiting time is $\frac{0+24+27}{3}$ which is $17ms$

- $P_3$, $P_2$, $P_1$: The waiting time for $P_3$ is $0ms$, for $P_2$ is $5ms$, and for $P_1$ is $8ms$. Hence, the average waiting time is $\frac{0+5+8}{3}$ which is $4.33ms$.

- $P_2$, $P_3$, $P_1$: The waiting time for $P_2$ is $0ms$, for $P_3$ is $3ms$, and for $P_1$ is $8ms$. Hence, the average waiting time is $\frac{0+3+8}{3}$ which is $3.67ms$.

As you can see, the waiting time varies highly with respect to the order of execution of processes. Hence, if a process with higher CPU burst requests CPU before a process with less CPU burst time than the waiting time would likely be huge relative to what it would be if it executed the process with less CPU burst first. This choice leads to our next scheduling algorithm which is...

#### Shortest-Job-First Scheduling

> When the CPU is available, it is assigned to the process that has the smallest next CPU burst.

Note that the scheduling depends on the next CPU burst of a process and not its total length.

The ``SJF`` scheduling is provably optimal, in that it gives the minimum average waiting time for a given set of processes.

The real difficulty here is predicting the length of the next CPU burst of all the processes. The algorithm is $\mathcal{O}(n)$ time which makes it infeasible to use for short term scheduling. Plus, there is no way to know the length of the next CPU burst.

The next CPU burst is generally predicted as an **exponential avegare**/**moving average** of the measured lengths of previous CPU bounds.

$$\tau_{n+1} = \alpha t_n + (1-\alpha)\tau_{n}$$

where $\tau_{n+1}$ is the predicted CPU burst, $\tau_{n}$ is a moving average of the predicted CPU bursts, $t_{n}$ is actual length of $n$'th CPU burst and $\alpha$ is a weight factor that controls the smoothness of the curve. A good value of this parameter is generally $0.5$ or $0.9$. One more degree of freedom is $\tau_0$ which is the initial predicted CPU burst which is normally set to $10ms$.

$$\tau_{n+1} = \alpha t_{n}+(1-\alpha)\alpha t_{n-1}+...+(1-\alpha)^{j}\alpha t_{n-j}+...+(1-\alpha)^{n+1}\alpha \tau_0$$

The ``SJF`` algorithm can be either preemptive or non-preemptive. The preemptive ``SJF`` algorithm is sometimes called the **shortest-remaining-time-first scheduling**.

#### Priority Scheduling

> The CPU is assigned to the process with the highest priority process.

Throughout this article, the process with lowest value of priority is considered highest priority process.

The ``SJF`` is a special case of the general priority schduling algorithm where the priority is the lenght of CPU burst time. Priorities can be assigned internally or externally.

Prioirity scheduling can be either preemptive or non-preemptive. A preemptive priority scheduling algorithm will preempt the CPU if the priority of the newly arrived process is higher than the priority of the currently running process. A nonpreemptive priority scheduling algorithm will simply put the new process at the head of the ready queue.

A major problem with priority scheduling is **indefinite blocking** or **starvation**. Suppose there is a process with priority 10 and processes with priorities lower than 10 enter the ready queue and terminate before the process with priority 10 gets the CPU. This would lead to a situation where the process with priority 10 will never get the CPU. Such a situation is regarded as **indefinite blocking** or **starvation**.

Asolution to the problem of indefinite blockage of low-priority processes is **aging**. Aging involves gradually increasing the priority of processes that wait in the system for a long time.

#### Round-Robin Scheduling

The round-robin (RR) scheduling algorithm is designed especially for timesharing systems.

A small unit of time, called a **time quantum** or **time slice**, is defined. A time quantum is generally from 10 to 100 milliseconds in length. The ready queue is treated as a circular queue.

It is a preemptive scheduling algorithm.

The average waiting time under the RR is ofter too long.

If there are $n$ processes in the ready queue and the time quantum is $q$, then each process gets the CPU time in chunks of at most $q$ time units. Each process must wait no longer than $(n-1)q$ time units until its next time quantum.

The performance of the ``RR`` algorithm heavily depends on the length of the time quantum. Turnaround time also depends on the length of the time quantum. A rule of thumb is that $80%$ of the CPU bursts should be shorter than the time quantum.

#### Multilevel Queue Scheduling
