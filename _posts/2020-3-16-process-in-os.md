---
layout: post
title: Chaper 3 - Processes in Operating Systems
category: Operating Systems Notes
tags: [process, os]
---

### Process VS Program

A Program is a *passive* entity. A process is an *active* entity.
A program becomes a process when its executable file is loaded into memory.
Note that a process itself can be an execution environment for other code.

### Process States -- The 5 state diagram

As a process executes, it changes states. A process might be in
one of the following states:

1. New: Process is being created.
2. Running: The processor is executing instructions.
3. Waiting: The process is waiting for some event to occur. It can be an IO event, interrupt event, etc.
4. Ready: The process is ready to be assigned to a processor for execution.
5. Terminated: The process has finished execution.

Only one process can be running on a processor at any instant.

![process state diagram](/images/os-process/process-state-diagram.svg)

### Process Control Block

Each process is represented in an operating system by a process control block. It contains the following information:

- **Process state**: new, running, waiting, ...
- **Program counter**: which instruction is to be executed next time the process is assigned to any processor.
- **CPU registers**: sp, accumulator, index registers, other general-purpose registers.
- **CPU scheduling information**: process priority, a pointer to scheduling queue, etc.
- **Memory Management info**: the value of base and limit registers, page tables, segment tables, etc.
- **Accounting info**: info that ``ps -eal`` displays.
- **I/O status info**: a list of open files, devices allocated, etc.

### Process scheduling

> The objective of multiprogramming is to have some processes running all the time, to maximize the CPU utilization.

#### Scheduling Queues

1. **Job Queue**: When a process enters, it is put into this queue
2. **Ready Queue**: The process is residing in the main memory and is ready to be executed.
3. **I/O queue or Device Queue**: Processes that are waiting for some I/O event to occur are put in here.

NOTE: In Linux, the job and ready queue are the same. All the new processes are loaded and put in the ready queue.

![queueing diagram](/images/os-process/queueing-diagram.svg)

#### Schedulers

1. **Long Term Schedulers**: Long term schedulers loads a bunch of processes from secondary memory to primary memory. Meaning, these schedulers have to choose which processes to load in memory for maximum CPU and IO device utilization.
    - a decision is made in a few seconds or minutes.
    - selects a good *mix* of CPU and IO bound processes.
2. **Short Term Schedulers**: Long term schedulers select from among the processes that are ready to execute and allocated the CPU to them. Properties of such schedulers are:
    - a decision is made in a few milliseconds.
    - time taken to make the decision is pure overhead.

NOTE: long term schedular is absent in Linux and the medium-term scheduler is responsible for *swapping* some processes from main to secondary memory and vice versa to control the **degree of multiprogramming**.

#### Context Switch

When an interrupt occurs, the system needs to save the current context of the process running on the CPU so that it can restore that context when its processing is done, essentially suspending the process and then resuming it when the processing is done.

It includes the value of the CPU registers, the process state, and memory-management information. Generically, we perform a state save of the current state of the CPU, be it in kernel or user mode, and then a state restores to resume operations. This task is known as **context switch**. Context switch times are pure overhead as no useful work is done.

### Operations on Process

#### Process Creation

Every process has a unique process identifier (PID). When a process creates a new process, two possibilities for execution exists:

1. The parent executes concurrently.
2. Waits until the chile terminates.

There are also two address spaces possible for the new process:

1. Copies the address space of parent.
2. The child process has a new program loaded in it.

The following C functions are a part of the UNIX API for process creating and termination.

- ``fork()`` -- Creates a new child process
- ``execlp(), execvp(), ...`` -- Loads a new program into address space of child process (hence destroying the address space of the parent process).
- ``wait()`` -- waits for any one of the children (or specified child) to finish execution.

Here's a code for creating a child process and running the `ls` command to print the files in a directory usign C:

```c
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
    pid_t pid, killed_pid;
    int killed_status;

    /* fork a child process */
    pid = fork();

    /* parent process will output a non-zero number while child process returns 0 */
    printf("process with pid: %d has reached this line!\n", pid);

    if(pid < 0) { /* error occured */
        fprintf(stderr, "Fork failed!\n\n");
        return 1;
    }
    else if ( pid == 0 ) { /* child process */
        execlp("/bin/ls", "ls");
    }
    else { /* parent process */
        /* The wait() system call is passed a parameter that
        allows the parent to obtain the exit status of the child */
        killed_pid = wait(&killed_status);
        printf("child with pid: %d exited with status: %d\n\n", killed_pid, killed_status);
    }

    return 0;
}

```

#### Process Termination

A process terminates when it makes a system call ``exit()``.

Some systems do not allow a child to exist if its parent has terminated. In such systems, if a process terminates (either normally or abnormally), then all its children must also be terminated. This phenomenon, referred to as cascading termination, is normally initiated by the operating system.

The ``wait()`` system call is passed a parameter that allows the parent to obtain the exit status of the child.

```c
pid_t PID;
int status;

pis = wait(&status);
```

A process that has terminated but whose parent has not yet called ``wait()`` is known as **zombie** process. If the parent terminates without calling ``wait()``, the child process becomes an **orphan** process. In Linux, the ``init`` process (launched during the bootstrap period) adopts all the orphan processes and periodically calls ``wait()`` until the child terminates. Below is an example that creates a zombie process. This zombie is alive for 30 seconds and its entry in the process table can be viewed using ``ps -eal`` command on the terminal.

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main(int argc, char **argv)
{
    pid_t PID;

    /* create a child process */
    pid = fork();

    if(pid == 0) { /* clild process */

        /* we will let the process run for 30 seconds
         * so we can later see its entry in `ps -ael` */
        sleep(30);

        return 0;
    }
    else if(pid > 0){ /* parent process */

        /* don't be like this parent */
        printf("I am leaving my child to become an orphan!\n Linux please adopt my child! Please!!!\n bye bye daughter...\n");
        return 0;
    }

    printf("WARNING: abnormal termination...\n");
    return 1;
}

```

### Interprocess Communication (IPC)

A process is *independent* if it cannot affect or be affected by other processes executing in the system.

A process is *co-operating* if it can affect or be affected by other processes running in the system.

Reasons for providing an environment for process co-operation:

1. Information sharing
2. Computational Speedup
3. Modularity
4. Convenience

Cooperating processes require an **interprocess communication** mechanism that allows them to exchange data and infeormation.

Two fundamental models of IPC are:

1. Shared Memory
2. Message passing

#### Shred Memory Systems

Shared memory region resides inside the address space of the process creating the shared memory segment. Any Other process which wish to communicate must first attach it to its address sapce.

Following program solves the classic producer consumer problem using shared memory. The producer produces ``"Hello World"`` message and places it in the shared memory space from where the consumer consumes and prints it on the terminal.

```c
/**
 * The producer process
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>

int main(int argc, char** argv)
{
    const int SIZE = 4096;
    const char *name = "OS";

    const char *message_0 = "Hello, ";
    const char *message_1 = "World!\n";

    int shm_fd;
    void *ptr;

    shm_fd = shm_open(name, O_CREAT|O_RDWR, 0666);
    ftruncate(shm_fd, SIZE);

    ptr = mmap(0, SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0);

    sprintf(ptr, "%s", message_0);
    ptr += strlen(message_0);
    sprintf(ptr, "%s", message_1);
    ptr += strlen(message_1);

    return 0;
}
```

```c
/**
 * The consumer process
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/shm.h>

int main(int argc, char** argv)
{
    const int SIZE = 4096;
    const char *name = "OS";
    int shm_fd;
    void *ptr;

    shm_fd = shm_open(name, O_RDONLY, 0666);
    ptr = mmap(0, SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    printf("%s", (char *)ptr);
    shm_unlink(name);

    return 0;
}
```

```c
/**
 * The manager process that launches
 * the producer and consumer
 */
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    pid_t producerp, consumerp;

    producerp = fork();
    if(producerp == 0) {
        printf("starting producer...\n");
        execlp("./producer", NULL);
    }

    consumerp = fork();
    if(consumerp == 0) {
        printf("starting consumer...\n");
        execlp("./consumer", NULL);
    }

    if(consumerp > 0 && producerp > 0) {
        wait(NULL);
        wait(NULL);
        printf("exiting...\n");
    }

    return 0;
}
```

#### Message Passing Systems

If processes `P` and `Q` want to communicate, they must send messages to and recienve messages from each other: a communication link must exist between them. Methods for implementing a link between processes are shown below:

1. **Direct Communication**: Under direct communication, each process that wants to communicate must explicitly name the recipient or sender of the communication.
    - ``send(P, message)``: Send a message to P.
    - ``recienve(Q, message)``: Recienve a message from Q.
    - Disadvantage: Limited modularity

2. **Indirect Communication**: With indirect communication, the messages are sent to and received from mailboxes, or ports.
    - ``send(A, message)``: Send a message to mailbox A.
    - ``receive(A, message)``: Receive a message from mailbox A.

3. **Synchronization**: Message passing may be either blocking or nonblocking: also known as synchronous and asynchronous.
    - Blocking send. The sending process is blocked until the message is received by the receiving process or by the mailbox.
    - Nonblocking send. The sending process sends the message and resumes operation.
    - Blocking receive. The receiver blocks until a message is available.
    - Nonblocking receive. The receiver retrieves either a valid message or a null.

4. **Buffering**: Whether communication is direct or indirect, messages exchanged by communicating processes reside in a temporary queue. Basically, such queues can be implemented in three ways:
    - Zero Capacity
    - Bounded Capacity
    - Unbounded Capacity

### PID Manager

The below program used bitmap to generate and allocate unique ``PID``s. This is somewhat analogous to what Linux does to allocate PID to a process.

``pid_manager.h``

```c
#ifndef GUARD_PID_MANAGER_H
#define GUARD_PID_MANAGER_H

#include <stdio.h>
#include <stdlib.h>

#define MIN_PID 30
#define MAX_PID 50

static short *pid_map = NULL;
static int curr;
static int stop;

int allocate_map()
{
    if(pid_map) {
        fprintf(stderr, "pid_map has already been initialized.\n");
        return -1;
    }

    pid_map = (short *)calloc(MAX_PID - MIN_PID + 1, sizeof(short));

    if(!pid_map) {
        fprintf(stderr, "error: memory allocation failed!\n");
        return -1;
    }

    curr = 0;
    stop = (curr + MAX_PID - MIN_PID) % (MAX_PID - MIN_PID + 1);

    return 1;
}

int allocate_pid()
{
    if(!pid_map) {
        fprintf(stderr, "error: pid_map not initialized!\n");
        return -1;
    }

    while(curr != stop) {
        if(!pid_map[curr]) {
            stop = (curr + MAX_PID - MIN_PID) % (MAX_PID - MIN_PID + 1);
            pid_map[curr] = 1;
            return curr + MIN_PID;
        }
        curr = (curr + 1) % (MAX_PID - MIN_PID + 1);
    }

    if(!pid_map[curr]) {
        stop = (curr + MAX_PID - MIN_PID) % (MAX_PID - MIN_PID + 1);
        pid_map[curr] = 1;
        return curr + MIN_PID;
    }

    curr = 0;
    stop = (curr + MAX_PID - MIN_PID) % (MAX_PID - MIN_PID + 1);
    fprintf(stderr, "error: pid_map is full. release some processes to assign new pid!\n");
    return -1;
}

int release_pid(int pid)
{
    pid = pid - MIN_PID;
    if(!pid_map) {
        fprintf(stderr, "error: pid_map not initialized!\n");
        return -1;
    }

    if(pid_map[pid]) {
        pid_map[pid] = 0;
        return 1;
    }

    fprintf(stderr, "error: tried releasing a non-existent pid!\n");
    return -1;
}

void display_pid_map()
{
    for(int i = 0;i < MAX_PID - MIN_PID + 1;i++) {
        printf("%d", pid_map[i]);
    }
    printf("\n");
}

#endif
```

Using the above header file, new pids can be allocated in \\( \mathcal{O}(n) \\) worst case time where \\( n \\) is the size of the bitmap. A example program is shown below:

```c
#include <stdio.h>
#include <stdlib.h>
#include "pid_manager.h"

#ifndef MIN_PID
    #warning "MIN_PID flag not loaded properly"
#endif

#ifndef MAX_PID
    #warning "MAX_PID flag not loaded properly"
#endif

int main(int argc, char **argv)
{
    /* status bit of map allocation */
    int status;

    /* allocate a bit map of pids */
    status = allocate_map();
    printf("allocation status: %d\n", status);

    /* try to get a pid,*/
    int pid;
    for(int i = MIN_PID;i < MAX_PID + 1;i++){
        /* allocate a pid */
        pid = allocate_pid();

        /* check and log abnormal pid allocation*/
        if(pid < 0) {
            printf("abnormal pid allocated: %d\n", pid);
        }
    }

    /* release some pids */
    for(int i = MAX_PID;i >= MIN_PID + 5;i--){
        status = release_pid(i);

        /* check and log failed attempt */
        if(status < 0) {
            printf("abnormal release status: %d\n", status);
        }
    }

    /* allocate the pids again just to test if release works fine! */
    for(int i = MIN_PID + 5;i < MAX_PID + 1;i++) {
        pid = allocate_pid();
        if(pid < 0) {
            printf("abnormal pid allocated: %d\t", pid);
            printf("trial: %d\n", i);
        }
    }

    display_pid_map();

    return 0;
}

```

Phew! You made it!

### References

[Operating Systems Concepts -- 9th Edition](https://codex.cs.yale.edu/avi/os-book/)
