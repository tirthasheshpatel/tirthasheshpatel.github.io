---
layout: post
title: Memory Management in OS
subtitle: This article describes how is the memory managed by the OS using paging and segmentation
tags: Operating Systems
hide: true
---

### Memory Management Goals

1. Allocate: Allocate the physical memory used by the process

2. Arbitrate: Translate the memory from logical address space to physical address space and validate for any illegal accesses.

#### Paging

1. Allocate: Logical memory is allocated in a fixed size containers called **pages** and coressponding physical memory is allocated in fixed size containers called **Page Frames**.

2. Arbitrate: A data structure called **Page Table** is used to traslate from logical address space to physical address space. It is also used to validate illegal accesses.

#### Segmentation

1. Allocate: Logical memory is allocated in a more flexible containers called **segments** and coressponding physical memory is allocated in variable sized containers called **segment registers**.

2. Arbitrate: Segement Talbes are used to translate and validate.

### Memory Management Unit

- Translate the virtual to physical addresses.
- report faults like illegal access, permission denied, etc.

#### Registers

- Pointers to Page Tables.
- base and limit registers for segment based methods.

#### Cache

- valid VA -> PA access => Translation Lookaside Buffer or TLB.

#### Translation

Actual PA translation done by the hardware!

### Page Tables

Page tables map the logical addresses to thier corressponding physical addresses.

The size of a page and a page frame is identical and so the number of entries in the page table becomes minimum. Now to actually translate, the logical addresses in pages are stored in the following format:

```c
typedef struct {
    long virtual_page_number;
    long offset;
} page_entry_t;
```

Here, the VPN (virtual page number) is present in a page table that gives its mapping to the physical access. Now, as the size of pages and page frmaes is same, we just add the offset to the mapped physical address to get the final physical address. Suppose for example:

Logical Address => ``VPN`` : 100 and ``offset`` : 5

Page Table => ``100 : 120000``

Then, the mapped physical address is $$120000 + 5 = 120005$$.

#### Smart Management

Now to save the physical memory as much as possible, the OS will allocate physical memory only when it is accessed.

Suppose you allocated an array `arr = (void *)malloc(1000 * sizeof(char))`. A total of `1000 * sizeof(char)` is allocated in the virtual memory but the coressponding physical memory is still not allocated. When first reference is made to some element in the array, only then, is the physical memory allocated. For example, suppose you accessed `arr[10]`. Only at this point is the physical memory allocated.

Now, there may be some data we don't access very often. Such data is swapped to fast temporary secondary memory.
