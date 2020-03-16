---
layout: post
title: Processes in Operating Systems -- Notes
category: Operating Systems Notes
tags: [process, os]
---

### Process VS Program

Program is a *passive* entity. Program is a *active* entitiy.
A program becomes a process when its executable file is loaded into memory.
Note that a process itself can be an execution environment for other code.

### Process States -- The 5 state diagram

As a process executes, it changes states. A process might be in
one of the following states:

1. New: Process is being created.
2. Running: Processor is executing instructions.
3. Waiting: Process is waiting for some event to occur. Can be a IO event, interrupt event, etc.
4. Ready: The process is ready to be assigned to a processor for execution.
5. Terminated: The process has finished execution.

Only one process can be running on a processor at any instant.

![process state diagram](/images/os-process/process-state-diagram.svg)

### Process Control Block

Each process is represented in an operating system by a process control block. It contains the following information:

- **Process state**: new, running, waiting, ...
- **Program counter**: which instruvtions is to be executed next time the process is assigned to any processor.
- **CPU registers**: sp, accumulator, index registers, other general purpose registers.
- **CPU scheduling information**: process priority, pointer to scheduling queue, etc.
- **Memory Management info**: value of base and limit registers, page tables, segement tables, etc.
- **Accounting info**: info that ``ps -eal`` displays.
- **I/O status info**: a list of open files, divices allocated, etc.

### Process scheduling

> The objective of multiprogramming is to have some processes running all the times, to maximize the CPU utilization.

#### Scheduling Queues

1. **Job Queue**: When a process enters, it is put into this queue
2. **Ready Queue**: The process is residing in the main memory and is ready to be executed.
3. **I/O queue or Device Queue**: Processes that are waiting for some I/O event to occur are put in here.

NOTE: In linux, job and ready queue are the same. All the new processes are loaded and put in the ready queue.

![queueing diagram](/images/os-process/queueing-diagram.svg)

#### Schedulers

1. **Long Term Schedulers**: Long term schedulers loads a bunch of processes from secondary memory to primary memory. Meaning, these schedulers have to make a choice of which processes to load in memory for maximum CPU and IO device utilization.
    - dicision is made in a few seconds or minutes.
    - selects a good *mix* of CPU and IO bound processes.
2. **Short Term Schedulers**: Long term schedulers select from among the processes that are ready to execute and allocated the CPU to them. Properties of such schedulers is:
    - dicision is made in a few milliseconds.
    - time taken to make the dicision is pure overhead.

NOTE: long term schedular is absent in linux and midium term scheduler is responsible for *swapping* some processes from main to secondary memory and vice versa to control the **degree of multiprogramming**.

#### Context Switch

When an interrupt occurs, the system needs to save the current context of the process running on the CPU so that it can restore that context when its processing is done, essentially suspending the process and then resuming it when the processing is done.

It includes the value of the CPU registers, the process state  and memory-management information. Generically, we perform a state save of the current state of the CPU, be it in kernel or user mode, and then a state restore to resume operations. This task is known as **context switch**. Context switch times are pure overhead as no useful work is done.

### Operations on Process

