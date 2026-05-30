# 5 In-Kernel Parallel Runtime

MPK employs an *in-kernel parallel runtime* that executes the *t*Graph across all SMs within a single mega-kernel. This design eliminates kernel-launch overheads and enables finegrained control over scheduling, synchronization, and execution order. Once launched, the mega-kernel continuously manages both computation and communication until the inference workload completes.

To support this execution model, MPK partitions a GPU's SMs into *workers* and *schedulers*. Each worker runs on one physical SM and maintains an independent *task queue*. Workers execute a lightweight loop that repeatedly dequeues tasks, performs the associated computation or communication, and signals task completion by notifying the task's triggering event. This design ensures that workers are fully utilized while enabling asynchronous execution across operators.

<span id="page-6-1"></span>![](_page_6_Figure_7.jpeg)

Figure 7: MPK 's event-driven execution model. Circles denote events, and blue (or orange) rectangles denote compute (or communication) tasks, respectively. Edges from an event to a task correspond to task launches, while edges from a task to an event indicate that the task triggers the associated event. AT, MM, and AR refer to attention, matrix multiplication, and AllReduce, respectively.

Schedulers are organized at *warp* granularity, with each SM hosting four scheduler warps. Each scheduler maintains an event queue and repeatedly polls for newly activated events, dispatching the corresponding tasks to workers. The allocation of workers and schedulers is fixed at kernel-launch time and matches the GPU's physical SM count, avoiding any dynamic role-switching overhead inside the kernel.

The remainder of this section details the in-kernel runtime architecture. § [5.1](#page-6-0) describes MPK 's event-driven execution model. § [5.2](#page-7-0) introduces two complementary task-launch mechanisms, analyzes their trade-offs, and explains how MPK combines them to achieve low-latency and load-balanced execution. § [5.3](#page-7-1) describes MPK's additional system optimizations that further reduce overhead and improve throughput.

