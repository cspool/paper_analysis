# nested grid、stream在GPU上执行（TG、CTA、nested、runtime）

METHOD AND SYSTEM FOR PROCESSING NESTED STREAM EVENTS

THREAD GROUP SCHEDULER FOR COMPUTING ON A PARALLEL THREAD PROCESSOR

CONTROLLING WORK DISTRIBUTION FOR PROCESSING TASKS

SCHEDULING AND MANAGEMENT OF  COMPUTE TASKS WITH DIFFERENT EXECUTION PRIORITY LEVELS

Hardware Compute Partitioning on NVIDIA GPUs for Composable Systems

## 命令Stream（串行任务打包成命令Stream，减少CPU同步）

CPU将**串行的任务打包成命令流并通过pushbuffer传递**，来去除频繁的CPU-GPU同步，提高CPU和GPU的协作效率。

**pushbuffer传递串行执行的命令流Stream（cudaStream）。**

应用可能需要定义**多个cudaStream协作（包含kernel嵌套调用）**，不同Stream中任务执行要满足cross-stream依赖，一般需要CPU使用cuda函数显式调度任务执行，让CPU和GPU频繁的同步等待。

不同task group（channel）中不同Stream并发的执行模型不支持CTA**动态产生kernel grid**，引入GTMD queue和QTMD等结构来**解决for-if-do嵌套kernel**难以静态unrolling的问题，去除CPU干预。

扩展架构支持线程组动态生成cudaStream，支持**CTA动态产生cudaStream**，扩展item到task和event，让调度满足**多个cudaStream之间依赖**。

> **[图片提取文字 (image.png)]:**
> [0004] In conventional computing systems having both a central processing unit (CPU) and a graphics processing unit (GPU), the CPU determines which specific computational tasks are performed by the GPU and in what order. A GPU computational task typically comprises highly parallel, highly similar operations across a parallel dataset, such as an image or set of images. In a conventional GPU execution model, the CPU initiates a particular computational task by selecting a corresponding thread program and instructing the GPU to execute a set of parallel instances of the thread program. In the conventional GPU execution model, only the CPU may initiate execution of a thread program on the GPU. After all thread instances complete execution, the GPU must notify the CPU and wait for another computational task to be issued by the CPU. Notifying the CPU and waiting for the next computational task is typically a blocking, serialized operation that leaves certain resources within the GPU temporarily idle, thereby reducing overall system performance. [0005] Performance may be improved in certain scenarios by queuing sequential computational tasks in a pushbuffer, from which the GPU may pull work for execution without waiting for the CPU. Computational tasks that include fixed data-flow processing pipelines benefit from this pushbuffer model when the CPU is able to generate work for the GPU quickly enough to have work pending within the pushbuffer whenever the GPU is able to start a new task. However, data-dependent computational tasks are still left with a sequential dependence between GPU results, CPU task management, and subsequent GPU task execution, which must be launched by the CPU.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> [0006] Multi-threaded computation models conventionally organize work into ordered streams of tasks that must complete in a defined order. In such computation models, execution semantics dictate that a given task must complete before a dependent task may execute. In a simple scenario, a serial dependence among an arbitrary sequence of tasks may be queued within a pushbuffer for efficient execution by the GPU. However, certain computation models allow for cross stream dependencies, whereby a task in one stream depends on two or more different tasks completing, potentially across two or more different streams. In such scenarios, the CPU schedules tasks to avoid deadlock. The process of waiting for certain tasks to complete before scheduling other tasks to avoid deadlock creates additional serial dependencies between the CPU and GPU task execution, reducing overall efficiency.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%203.png)

## Fig2、3、4、5（channel、TMD、CTA、线程组）

**pushbuffer中命令流是cudaStream**。

TMD group（链表，应用channel）是相同优先级的TMD，RR或greedy从不同task group中调度TMD，不同group一般是不同应用。

TMU中若干task group属于硬件调度模块，也称为channel。

所有线程组的GTMQ queue、QTMD和TMDQs是定义Stream的数据结构（类似CPU内存中pushbuffer），保存在显存，缓存在TMD cache。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3A
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6A
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 4C
> 
> ![](_page_0_Figure_3.jpeg)
> 
> ![](_page_0_Figure_4.jpeg)
> 
> Figure 4D
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%206.png)

> **[图片提取文字 (image.png)]:**
> The task management unit 300 manages compute tasks to be scheduled as an array of TMD groups that are stored in the scheduler table 321. A TMD group is a set of compute tasks with the same scheduling priority. The number of TMD groups, or priority levels, may be one or more. Within each TMD group, the compute tasks at the respective priority level are stored in a list, which can be implemented with a linked list, and hereinafter a linked list is assumed. Each TMD in a linked list stores a pointer to the next TMD in the respective linked list. A head pointer and a tail pointer for the linked list are stored for each TMD. A TMD group having no tasks has a head pointer that equals the tail pointer and an empty bit is set TRUE.
> 
> When compute tasks are received from the host interface **206**, the task management unit **300** inserts the compute tasks into a TMD group. More specifically, a task pointer to the TMD corresponding to the compute task is added to the tail of the linked list for that group unless a special TMD bit is set which causes the task to be added to the head of the linked list. Even though all tasks within a TMD group have the same scheduling priority level, the head of the TMD group linked list is the first compute task that is selected by the task management unit **300** and scheduled for execution. Thus, the compute task at the head of the linked list has a relatively higher priority compared with other compute tasks at the same priority level. Similarly, each successive com-
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> The simplest scheduling scheme is for the task management unit 300 to schedule the compute task at the head of each group (if a compute task exists in the group) and rotate through the groups in round-robin order. Another scheduling technique is priority scheduling that selects the compute tasks in strict priority order. The task management unit 300 selects a compute task from the highest priority group that has at least one compute task, starting at the head of the group.
> 
> Each TMD 322 may be a large structure, e.g., 256 Bytes or more, that is typically stored in PP memory 204. Due to the large size, the TMDs 322 are expensive to access in terms of bandwidth. Therefore, the task/work unit 207 may be configured to include a cache (not shown) to store only the (relatively small) portion of the TMD 322 that is needed by the task management unit 300 for scheduling. The remainder of the TMD 322 may be fetched from PP memory 204 when the task is scheduled, i.e., transferred to the work distribution unit 340. The TMDs 322 are written under software control, and, when a compute task completes execution, the TMD associated with the completed compute task may be recycled to store information for a different compute task. Because a TMD 322 may be stored in the cache, the entries storing information for the completed compute task should be flushed from the TMD cache 405.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%208.png)

线程组warp是1个warp Ctx承载的线程负载，warp Ctx是SM执行warp的资源单位，warpSz是SIMT Cores执行SIMD指令的线程宽度。

SM并发多个CTA，包含所有warp**按照“年龄”顺序调度发射**，因为SM的SIMT Cores的并行warp数和SM的并发warp数不同。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3B
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3C
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 4
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2012.png)

> **[图片提取文字 (image.png)]:**
> 3A, according to one embodiment of the present invention. The SMP 310 includes an instruction L1 cache 370 that is configured to receive instructions and constants from memory via L1.5 cache 335. An instruction scheduler 312 receives instructions and constants from the instruction L1 cache 370 and controls local register file 304 and SMP 310 functional units according to the instructions and constants. The SMP 310 functional units include N exec (execution or processing) units 302 and P load-store units (LSU) 303. The instruction scheduler 312 is configured to schedule thread groups belonging to one or more different CTAs for execution
> 
> [0055] FIG. 3C is a block diagram of the SMP 310 of FIG.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2020.png)

> **[图片提取文字 (image.png)]:**
> by the exec units 302. The instruction scheduler 312 may cause each of the N exec units 302 to execute a different one of N threads belonging to a given thread group.
> 
> [0056] As described in greater detail below in conjunction with FIGS. 4 and 5, the instruction scheduler 312 performs a series of processing steps at each cycle of the SMP 310 in order to identify a particular thread group to issue for execution during the subsequent cycle. First, the instruction scheduler 312 identifies the thread groups that are currently available to issue. A particular thread group may not be available to issue when that thread group has been stalled by, e.g., a hardware-imposed latency, a math or memory latency, or a synchronization latency, as discussed in greater detail below in conjunction with FIG. 4.
> 
> [0057] Once a pool of available thread groups has been identified, the instruction scheduler 312 sorts the available thread groups based on a "seniority" value associated with the CTA to which each thread group belongs. As referred to herein, the seniority value of a given CTA may refer to the amount of time elapsed since the CTA was initially launched (i.e., the "age" of that CTA). However, as further discussed below in conjunction with FIG. 5, the seniority value of a CTA may be increased or decreased by the instruction scheduler 312, and, thus, may not necessarily reflect the actual age of that CTA.
> 
> that has the greatest seniority value and then sorts the thread groups within that CTA based on a credit value associated with each thread group. The credit value associated with a given thread group reflects the progress of the threads within the thread group towards completing the processing tasks assigned to those threads, as also discussed in greater detail below in conjunction with FIG. 5. The instruction scheduler 312 selects the thread group with the highest credit value, and then issues that thread group for execution during the subsequent cycle. At the subsequent cycle, the SMP 310 executes the threads within the selected thread group using the execunits 302 and the LSUs 303, as well as various different memory units within the SMP 310 and/or external to the SMP 310.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2021.png)

**grid TMD（计算任务）、queue TMD（嵌套任务、图形任务）**

grid TMD的CTA通过block定义，**grid TMD拆成CTA的三维阵列**。

可配置TMD按CTA串行，TMD抢占和恢复。

queue TMD初始化queue，queue中entry是输入数据vertices或子任务。queue TMD在满足条件后派发不同数量CTA处理queue中数据或子任务，即让CTA排队处理queue中数据。

> **[图片提取文字 (image.png)]:**
> TMD 322
> 
> ## Initialization Parameters 405
> 
> Scheduling Parameters 410
> 
> Execution Parameters 415
> 
> CTA State <u>420</u>
> 
> Work Distribution Parameters 422
> 
> Queue <u>425</u>
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## Compute Task Metadata
> 
> FIG. 4A is a conceptual diagram of the contents of a TMD 322 that is stored in PP memory 204, according to one embodiment of the invention. The TMD 322 is configured to store initialization parameters 405, scheduling parameters 410, execution parameters 415, CTA state 420, work distribution parameters 422, and a queue 425. The work distribution parameters 422 store values that control the distribution of work stored in the queue **425** to one or more CTAs. State that is common to all TMDs 322 is not included in each TMD 322. Because a TMD 322 is a data structure that is stored in PP memory 204, a compute program running on the CPU 102 or PPU 112 can create a TMD 322 structure in memory and then submit the TMD 322 for execution by sending a task pointer to the TMD 322 to the task/work unit **207.** 
> 
> The initialization parameters 405 are used to configure the GPCs 208 when the TMD 322 is launched and may include the starting program address and size of the queue 525. Note that the queue 425 may be stored separately from the TMD 322 in memory, in which case the TMD 322 includes a
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2014.png)

> **[图片提取文字 (image.png)]:**
> an index or a virtual address. The initialization parameters 405 may also include bits to indicate whether various caches, e.g., a texture header cache, a texture sampler cache, a texture data cache, data cache, constant cache, and the like, are invalidated when the TMD 322 is launched. Initialization parameters 405 may also include dimensions of a CTA in threads, a TMD version number, an instruction set version number, dimensions of a grid in terms of CTA width, height, and depth, memory bank mapping parameters, depth of a call stack as seen by an application program, and a size of the call-return stack for the TMD. The initialization parameters 505 may include constant buffer parameters, which are a set of descriptors of
> 
> pointer to the queue 425 (queue pointer) in place of the
> 
> actual queue 425. When entries of the queue 425 are
> 
> assigned to a CTA for processing, each entry is specified by
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> The execution parameters 415 for a TMD 322 may include a task pointer to a dependent TMD that is automatically launched when the TMD 322 completes. Semaphores may be executed by the TMDs 322 to ensure that dependencies between the different TMDs 322 and the CPU 102 are met. For example, the execution of a first TMD 322 may depend on a second TMD completing, so the second TMD generates a semaphore release, and the first TMD executes after the corresponding semaphore acquire succeeds. In some embodiments, the semaphore acquire is performed in the host interface 206 or the front end 212. The execution parameters 415 for a TMD 322 may store a plurality of semaphore releases, including the type of memory barrier, address of the semaphore data structure in memory, size of the semaphore data structure, payload, and enable, type, and format of a reduction operation. The data structure of the semaphore may be stored in the execution parameters 415 or may be stored outside of the TMD 322. The execution parameters 415 may also include the
> 
> starting address of the program to be executed for the TMD 322, the type of memory barrier operation that is performed when execution of the TMD 322 completes, a serial execution flag indicating whether only a single CTA is executed at a time (serially) for the TMD 322, and a throttle enable flag that controls whether or not the task/work unit 207 may limit the number of CTAs running concurrently based on the memory limitations specified for the TMD 322. The execution parameters 415 may also store various flags that control behaviors of arithmetic operations performed by the processing task that is executed for the TMD 322, e.g., not-anumber (NaN) handling, float-to-integer conversion, and rounding modes of various instructions.
> 
> The CTA state **420** for the TMD **322** may include the number of cycles that have elapsed since a CTA was launched for use in conjunction with the coalesce waiting time parameter. Additionally, when a process is preempted, processing of the TMD **322** may be stopped at an instruction boundary or a CTA boundary and identification of the CTA at which processing will be resumed is stored in the CTA state **420**. The state information needed to resume execution of the TMD **322** after preemption may be stored in the CTA
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2016.png)

**CTA state**包含TMD年龄、抢占时的instr boundary或CTA boundary。

**scheduler参数**是TMD往GPC调度的设置，TMD是grid还是queue，如何加入TMD group。

WDU将queue中entry发给CTA执行，queue中每N个entry指派M个CTA，**queue定义了dataflow或多线程协作，常用于图形pipeline或dataflow任务**。

> **[图片提取文字 (image.png)]:**
> number of cycles that have elapsed since a CTA was launched for use in conjunction with the coalesce waiting time parameter. Additionally, when a process is preempted, processing of the TMD 322 may be stopped at an instruction boundary or a CTA boundary and identification of the CTA at which processing will be resumed is stored in the CTA state 420. The state information needed to resume execution of the TMD 322 after preemption may be stored in the CTA
> 
> The CTA state 420 for the TMD 322 may include the
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2017.png)

> **[图片提取文字 (image.png)]:**
> system memory 104. The CTA state 420 also stores pointers to entries of the queue 425 and counter overflow flags indicating when each pointer increments past the end of the queue 425 and needs to wrap back to the start of the queue 425.
> 
> The scheduling parameters 410 control how the task/work unit 207 schedules the TMD 322 for execution. The sched-
> 
> state 420, or in a separate area in PP memory 204, or in
> 
> uling parameters 410 may include the TMD group ID, a bit to indicate where the TMD 322 is added to a linked list (head or tail), and a pointer to the next TMD 322 in the TMD group. The scheduling parameters 410 may also include masks that enable/disable specific streaming multiprocessors within the GPCs 208.
> 
> The scheduling parameters 410 may also include a bit
> 
> indicating whether the TMD 322 is a queue TMD or a grid
> 
> TMD. Alternate embodiments may have different structures
> 
> for a grid TMD and a queue TMD, or implement either grid
> 
> TMDs or queue TMDs. If the TMD 322 is a grid TMD, then
> 
> the queue feature of the TMD 322 is unused, and execution of the TMD 322 causes a fixed number of CTAs to be launched and executed. The number of CTAs is specified as the product of the grid width, height, and depth. When entries of the grid are assigned to a CTA for processing, each entry is specified by coordinates within the grid, but there is no explicit data pointer. The program defined by the TMD 322 may use any sequence of instructions and constant buffer data values to convert the grid coordinates into a pointer to a fixed amount of predefined data for the grid to be processed by a CTA.
> 
> If the TMD 322 is a queue TMD, then the queue feature
> 
> queue 425, as queue entries. Queue entries are input data to CTAs of the TMD 322. The queue entries may also represent child tasks that are generated by another TMD 322 during execution of a thread, thereby providing nested parallelism. Typically, execution of the thread, or CTA that includes the thread, is suspended until execution of the child task completes. The queue 425 may be implemented as a circular queue so that the total amount of data is not limited to the size of the queue 425. As previously described, the queue
> 
> of the TMD 322 is used, meaning that data are stored in the
> 
> 425 may be stored separately from the TMD 322 and the TMD 322 may store a queue pointer to the queue 425. Advantageously, queue entries for the child task may be written to the queue 425 while the TMD 322 representing the child task is executing.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> In one embodiment, a variable number of CTAs are executed for a queue TMD, where a CTA is launched for each a number of contiguous entries written to the queue 425 of the queue TMD. While a particular CTA may process multiple contiguous entries of the queue 425, each entry in the queue **425** is processed by only a single CTA. The work distribution parameters 422 for a queue TMD may include the number of contiguous entries (N) of queue 425 that are needed to launch a CTA. The number of CTAs depends on the number of entries written to the queue 425 of the queue TMD and N. For example, when N=10 and 50 entries are written to the queue 425, 5 CTAs will be executed for the queue TMD. If all 50 entries are written at the same time, all 5 CTAs may launch at the same time. If the 50 entries are written over several clock cycles, then the CTAs will be launched as each successive group of 10 contiguous entries is written.
> 
> In another embodiment, each of the N entries may be processed more than one CTA. When N entries are added to the queue **425**, a first set of CTAs are launched for the TMD **322**, where the number of CTAs in the set, M, is also
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2019.png)

> **[图片提取文字 (image.png)]:**
> specified by the work distribution parameters 422. For example, when N=6 and M=3 and 36 entries are written to the queue 425, 18 CTAs (M\*36/N) will be executed for the queue TMD. If all 36 entries are written at the same time, all 18 CTAs may launch at the same time. If the 36 entries are written over several clock cycles, then the CTAs will be launched as each successive group of 6 contiguous entries is written.
> 
> An example TMD 322 may be configured to perform
> 
> tessellation operations, where each of the M CTAs processes vertices that are written to the queue 425. Each CTA may be configured to process the same set of vertices differently based on the CTA identifier. Multiple CTAs may be used to process each data element, e.g., entry written to the queue 425, when the amount of processing will benefit from concurrency. Note that in an embodiment where each CTA executes on a single SM 310, launching M CTAs to process N data elements will allow the data elements to be processed by up to M SMs 310 (assuming M is not greater than the number of SMs 310). When the processing speed of a CTA is limited by the resources of an SM 310, multiple CTAs may be used that each process fewer threads, so that more resources are available for each thread. In another example, when threads are likely to diverge during execution, each thread (or a set of threads) may be executed by a different CTA so that serialization of execution due to divergence is reduced and the divergent threads are instead executed concurrently by each of M different CTAs.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2020.png)

> **[图片提取文字 (image.png)]:**
> control the distribution of work for processing according to a compute task that is executed in a multi-threaded system, e.g., PPU 202. The TMD 322 that encodes a compute task includes the work distribution parameters 422 and scheduling circuitry reads the work distribution parameters 422 when one or more entries of a work queue 425 for the compute task have been written. Multiple processing tasks may each be executed independently with a producer processing task dynamically generating work to be processed by a one or more consumer processing tasks. The distribution of the work to one or more consumer processing tasks may be controlled in terms of how much work is needed before processing of that work is initiated by the consumer processing tasks, the number of CTAs that will process the work, and a step size that controls the specific entries of the work queue that are distributed to each CTA. The distribution mechanism may be used to control the distribution of work multi-threaded systems and in data flow networks.
> 
> The work distribution parameters 422 are configured to
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4B
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2022.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4C
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2024.png)

## Fig4、5（CTA动态生成grid）

支持nested parallel需要设计硬件功能（GPU执行pipeline）、软件运行时功能（底层向上接口，如父亲线程同步）、编程语言支持（应用编程接口）。

运行时函数是辅助应用程序在硬件运行的函数，结合底层硬件接口实现某些功能。

> **[图片提取文字 (image.png)]:**
> which are therefore children of task 420(1). While only one level of parent-child hierarchy (nesting depth) is shown in FIG. 4, an arbitrary hierarchy may be implemented in practice. In one embodiment, nesting depth is limited by a number of scheduling groups. Priority may be assigned to child execution over parent execution within the scheduling groups. In one embodiment, tasks 420 and 430 each execute as at least one thread group, or at least one CTA within SM **310** of FIG. **3**B. To enable thread programs having a parent and child relationship to execute on SM 310, three system elements should be implemented, including hardware functionality for parallel processing subsystem 112, software runtime functionality for parallel processing subsystem 112, and language support constructs for programming parallel processing subsystem 112.
> 
> [0072] In this example, task 420(1) is a parent of tasks 430,
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2025.png)

> **[图片提取文字 (image.png)]:**
> parallel processing subsystem 112 includes launching a new grid or CTA of work from a request generated by SM 310 and queued for execution to task/work unit 207, saving execution state for SM 310, continuing execution within SM 310 from the saved execution state, and facilitating memory coherence between a parent and child task. The runtime features required to support a parent thread launching a child thread, CTA, or grid within processing subsystem 112 includes launching a new grid in response to a request from a thread executing within SM 310, enabling a parent thread to perform a thread synchronization barrier on a child thread group, ensuring memory coherence between the parent thread and
> 
> [0073] The hardware functionality required to support a
> 
> parent thread launching a child thread, CTA, or grid within
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2026.png)

> **[图片提取文字 (image.png)]:**
> the child group, scheduling work and continuation of synchronized thread groups for guaranteed forward computational progress, and ensuring proper execution semantics for parent threads and child groups. The language support constructs include a mechanism for specifying the launch of a child thread program from a parent thread, and executing a synchronization barrier on the child program.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2027.png)

> **[图片提取文字 (image.png)]:**
> the parent. Launching the child grid may be implemented using CUDA runtime calls, including a system memory barrier (membar.sys) to guarantee memory consistency between parent and child. Executing the system memory barrier has the effect of flushing all pending write data from the parent to memory, such as PP memory 204. Flushing all pending writes allows any thread within the child grid to safely read arbitrary data from the memory space of the parent while executing on an arbitrary SM 310. Those of ordinary skill in the art will understand that flushing caches is one means to ensuring memory consistency, but other approaches are possible.
> 
> [0081] Once the memory barrier has been executed, a
> 
> CUDA runtime call, referred to herein as cudaRTLaunch(),
> 
> may be executed to launch the child grid. In one embodiment,
> 
> the CUDA runtime call to cudaRTLaunch() queues a new
> 
> task for execution to the scheduler by presenting a posted compare and swap (PCAS) message to a memory management unit, such as MMU 328 of FIG. 3B, which reflects the message back to the scheduler. A PCAS operation represents one mechanism for SM 310 to schedule work. The PCAS operation is implemented as a blocking (posted) synchronization operation that is performed by MMU 328. The PCAS operation atomically compares a present value of memory at a specified memory location to a first specified value and over writes the memory location with a second specified value if, and only if, the present value of memory matches the first
> 
> specified value.
> 
> events to save the current execution state of the calling (soon to be parent) thread. In this example, the calling thread is foo(). The call to cudaThreadSynchronize() may explicitly reference a child grid being launched by the calling thread. In one embodiment, parallel processing subsystem 112 is configured to save all relevant execution state for a CTA executing within a GPC 208, including all relevant architectural
> 
> [0082] The thread synchronization barrier call cudaTh-
> 
> readSynchronize(), shown in Table 1, initiates a series of
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2028.png)

**pushbuffer中命令流是cudaStream**。

grid TMD是计算grid任务，queue TMD是图形任务，Stream包含多个命令（不同任务）。

计算task是一次kernel调用（grid），父亲CTA执行if、for、kernel嵌套可能**动态产生儿子grid。**产生的儿子grid执行完后，父亲CTA执行剩余。

父亲CTA运行时产生儿子grid A。

cuda runtime call是memory bar kernel，保证**内存一致**（无效当前Cache）。

cudaRTLaunch将**儿子grid加入queue**，具体是MMU执行PCAS指令，让scheduler（queue）接受SM生成的grid TMD指针，而不是pushbuffer中的grid TMD指针。

cudaThreadSync插入**barrier**让父亲线程等待儿子线程同步，**保存父亲Ctx**。插入barrier是因为**线程调用kernel后立刻返回**，不会等待调用完成。

儿子grid A结束恢复父亲CTA。

AtExit任务调用scheduler kernel，调度指向的CTA启动（某个之前挂起的CTA）。

AtEntry任务调用restoration kernel，来**恢复父亲CTA的Ctx**，保护**内存一致**（之前Cache失效），让父亲跳出**barrier**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE 1
> 
> ```
> _global_ void foo()
> void *ptr = malloc(1024);
> A <<<1, 1>>>(ptr); // child launch
> cudaThreadSynchronize(); // sync barrier
> do_stuff(ptr);
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 5
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2031.png)

## Table2、3、4、5、6（for-if-do嵌套kernel的动态展开）

kernel grid调用的嵌套，减少CPU参与判断和启动任务，main交互最外层kernel调用，深层kernel调用和其父亲交互。

T2中，100个foo线程分别生成bar grid，并分别等待自己bar grid完成，main等待100个foo线程完成，总计执行100个foo线程和100个bar线程。

T3中，只有第一个foo线程生成bar grid，并等待bar grid完成，main等待100个foo线程完成，总计执行100个foo线程和1个bar线程。

> **[图片提取文字 (image.png)]:**
> ## TABLE 2
> 
> ```
> global__void foo()
>     void *ptr = malloc(1024);
>     bar <<<1, 1>>> (ptr);
>     cudaThreadSynchronize();
>     do_stuff(ptr);
> void main()
>     // Launch 100 threads of "foo"
>     foo <<<1,100>>>();
>     cudaThreadSynchronize();
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE 3
> 
> ```
> _global__ void foo()
>     void *ptr = malloc(1024);
>     if(threadIdx.x == 0)
>          bar <<<1, 1>>>(ptr);
>     cudaThreadSynchronize();
>     do_stuff(ptr);
> void main()
>     // Launch 100 threads of "foo"
>     foo <<< 1, 100 >>>( );
>     cudaThreadSynchronize();
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2033.png)

block内所有foo线程等待父亲线程（block中第一个foo线程）创建父亲stream后，每个foo线程生成bar grid并将bar grid加入父亲stream（**100个bar grid在stream中串行**），所有foo线程等待父亲stream完成。总计100个foo线程和100个bar线程（处于1个父亲stream）。

> **[图片提取文字 (image.png)]:**
> ## TABLE 4
> 
> ```
> _shared__ cudaStream__t stream;
> __global__ void foo()
>     if(threadIdx.x == 0)
>          cudaStreamCreate(&stream);
>     _syncthreads();
>     bar <<< 1, 1, 0, stream >>>();
>     cudaThreadSynchronize();
> void main()
>     foo <<< 1, 100 >>>();
>     cudaThreadSynchronize();
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2034.png)

> **[图片提取文字 (image.png)]:**
> nels having a parent and arbitrary child kernels is structured to be composable, meaning only the parent kernel is visible from outside the hierarchy. In other words, when a parent kernel launches child kernels, the child kernels appear as part of the parent kernel. This means the parent kernel and all child kernels must complete execution for the parent to be considered as having completed execution. By way of example, all work associated with task 520(0) of FIG. 5 must complete before task 520(1) is launched. Similarly, all work associated with task 520(1) and child tasks 530(0) through 530(2) must complete before task **520**(1) is considered to have completed. Once task 520(1) has completed, task 520(2) may be launched.
> 
> [0088] In one embodiment, a hierarchy of executing ker-
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2035.png)

CPU程序main的for-if-do结构，需要CPU动态判断if是否执行if内的do任务，不能将for-if-do作静态unrolling成多个do-kernel线程。

Tab6：最内层if(cond) do定义成if-kernel线程，for-if-do作unrolling为if-kernel<<<forSz>>>，但外层多个嵌套for-if-do的条件全部需要CPU参与，频繁的CPU-GPU同步影响性能。

Tab7：每层for(forSz)-if-do作unrolling为if-kernel<<<forSz>>>调用，即生成if-kernel grid，if-kernel线程中if-do中嵌套for-if-do，对应if-kernel grid的动态生成。

> **[图片提取文字 (image.png)]:**
> ## TABLE 5
> 
> ```
> void main() {
> for(i...imax) {
>      if(condition1) for(j...jmax) {
>           if(condition2) for(k...kmax) {
>                if(condition3) do_stuff(data, i, j, k);
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE 6
> 
> ```
> _global_ void cuda_inner_loop(data, i, j) {
>     if(condition3) do_stuff(data, i, j, threadIdx.x);
> void main() {
>     for(i...imax) {
>           if(condition1) for(j...jmax) {
>                if(condition2) {
>                     cuda_inner_loop<<< kmax >>>(data, i, j);
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE 7
> 
> ```
> __global__ void cuda__inner__loop (data, i, j) {
>      if(condition) do_stuff(data, i, j, threadIdx.x);
>  _global__ void cnp__loop__j(data, i) {
>      if(condition) cuda_inner_loop<<< kmax >>>(data, i,
> threadIdx.x):
>  _global__ void cnp__loop__i(data, i) {
>      if(condition) cnp_loop_j <<< jmax >>> (data,
> threadIdx.x);
> void main() {
>      cnp loop i << \max >>> (data);
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2038.png)

## Fig6、7、8（grid中生成grid、grid中生成Stream）

命令流Stream中的计算命令流变成表层GTMD queue。每个GTMD调度到WDU并按CTA派发到SM（SM按CTA分配Ctx），即分配1个grid到SM执行。

Fig6的GTMD queue和QTMD是管理任务（CTA）中嵌套kernel的父、子、系统任务的queues数据结构，Fig7-8中不同TMDQ是管理任务（线程组）中嵌套Stream的queues数据结构，类似pushbuffer中存储的命令流。

每个TMDQ、GTMD queue、QTMD统称为**TMD group，不同TMD group优先级不同**。

GTMD queue是grid TMD队列，分为**应用表层grid队列和嵌套里层grid队列**。

应用表层grid队列，是驱动为应用静态生成的grid队列，包含main中表层的kernel调用。

嵌套里层grid队列，是SM执行CTA时**动态产生**的grid，包含kernel中嵌套的kernel调用。

QTMD是cuda kernel TMD队列，是scheduler和restoration的TMD队列。

scheduler kernel指向某个之前挂起的CTA，调度挂起的CTA启动。

restoration kernel为启动的CTA恢复Ctx，重置Cache，跳出barrier。

不同kernel可能分别处于不同queue。

> **[图片提取文字 (image.png)]:**
> a grid task metadata descriptor (GTMD) queue 650 for receiving and storing application work 612, for example from CPU 102 of FIG. 1. The application work 612 comprises an ordered sequence of GTMDs, labeled task1 through taskN. Scheduler 610 is configured to receive each GTMD and schedule a corresponding grid for execution on the SM 630 via the distributor, which serves to allocated threads as CTAs within the SM 630. The continuation state buffer discussed in FIG. 5 may be stored in continuation state buffer 642, residing within memory 640. In one, embodiment, scheduler 610 comprises task management unit 300 of FIG. 3A, distributor 620 comprises work distribution unit 340, SM 630 comprises SM 310 of FIG. 3B, and memory 640 comprises PP memory 204, system memory 104, or a combination thereof.
> 
> [0094] When a thread executing within SM 630 launches a child CTA, a new GTMD is generated for the child CTA and queued for execution within GTMD queue 652. Scheduler 610 is able to distinguish new application work 612 arriving from GTMD queue 650 from nested processing work arriving in GTMD queue 652 because each set of work is stored in separate queues. Scheduler 610 may assign different execution priority at different times to work stored in each GTMD queue using any technically feasible technique that guarantees forward execution progress.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2039.png)

> **[图片提取文字 (image.png)]:**
> [0095] When a grid executing within SM 630 is halted in response to calling cudaThreadSynchronize(), execution state is stored to continuation state buffer 642, and a scheduler kernel is queued for execution. In one embodiment, the scheduler kernel is queued for execution in a queue of task metadata descriptors QTMD 654. For example, task7 may comprise a descriptor for a scheduler kernel queued for execution within QTMD 654 by a CTA that previously exited SM 630, and taskP may comprise a most recent scheduler kernel queued for execution by a most recently exited CTA from SM **630**.
> 
> [0096] In one embodiment, a restoration kernel is executed to prepare resources within SM 630 for resumed execution of a CTA that previously exited by executing cudaThreadSynchronize(). In certain implementations, the restoration kernel is queued for execution within QTMD 654. In alternative implementations, the restoration kernel, such as taskQ, is queued for execution within a separate QTMD 656 for greater scheduling flexibility. While execution restoration of the CTA is described above in terms of a restoration kernel, any other
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2040.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2047.png)

nested depth：线程上面嵌套的父亲grid层数，CPU线程的深度是0。

Fig7是CPU线程组（main线程）动态创建的cudaStream，每个Stream动态分配1个TMDQ管理并存储在CPU内存中（TMU需要访问CPU内存），因此main可以自由分配TMDQ。Fig8是GPU线程组（device线程）动态创建Stream，但GPU不能动态分配内存，因此提前在VRAM中静态分配TMDQ的内存并映射（TMU直接访问GPU内存）。

queue task（Stream）包含若干GPU命令，Stream**分配一个TMDQ管理**，Stream中task串行执行。

Fig6中任务grid按CTA执行，动态生成的grid大小和Stream个数是基于CTA确定，CTA可能分支发散到不同线程组，因此Fig7-8将CTA扩展到线程组。

**执行图的不同深度上有各自的线程组（执行相同kernel的线程集合），作为线程组Ctx的若干TMDQ**（712-0/1/2等）。

每个TMDQ指向所管理task stream中的task。

task完成要求生成的儿子线程组也完成，如740-2需要等待740-1、780和790完成后启动。

不同深度的线程组表示不同时候（SM/host）生成的任务，深度0的线程组是CPU创建的，TMDQ分别对应Fig6中的GTMQ或QTMD。

> **[图片提取文字 (image.png)]:**
> [0098] FIG. 7 illustrates an exemplary hierarchical execution graph including associated task metadata queues and tasks, according to one embodiment of the present invention. As shown, the hierarchical execution graph includes thread group 710 at nesting depth 0, task metadata descriptor queues (TMDQs) 712, tasks 720 730 740, an execution graph 780 at nesting depth 1, and an execution graph 790 at nesting depth 2. [0099] The thread group 710 at nesting depth 0 includes threads created and managed by CPU 102. A thread group includes any set of threads, including a CTA, where all threads exist at the same nesting depth. The nesting depth of
> 
> threads exist at the same nesting depth. The nesting depth of a thread is the number of parent grids above the level of the thread. For example, a CPU thread has a nesting depth of 0, because there are no parent grids above a CPU thread. If that CPU thread launches a grid, then that grid is said to be at nesting depth 1. If a thread in the grid at nesting depth 1 launches a new grid, then the new grid is said to be at nesting depth 2, and so forth. Because the threads in thread group 710 are CPU threads, each of these threads are at nesting depth 0.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2041.png)

> **[图片提取文字 (image.png)]:**
> [0102] Execution graph 780 at nesting depth 1 is a thread group, plus the associated TMDQs and tasks, which has been launched by one of the tasks at nesting depth 0. Any task may launch one or more grids, where such grids are at a nesting depth that is one greater than the nesting depth associated with the task that launched the grid. As shown, task 740(1), existing at nesting depth 0, launched execution graph 780 sometime during the execution of task 740(1). Each task and TMDQ within execution graph 780 functions essentially the same as tasks and TMDQs at nesting depth 0. When each task within execution graph 780 completes, and all other commands in task 740(1) have completed, then task 740(2) may begin execution.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2042.png)

> **[图片提取文字 (image.png)]:**
> points to tasks belonging to one or more streams. TMDQ (0) 712(0) points to task 720(0) associated with a first stream. TMDQ (1) 712(1) points to tasks 730(0) and 730(1) associated with a second stream. TMDQ (2) 712(2) points to tasks 734(0), 740(1), and 740(2) associated with a third stream. Any number of TMDQs 712 may be defined where each TMDQ 712 includes an arbitrary number of tasks. [0101] Tasks 720 730 740 comprise data structures that include one or more commands to be executed by the GPU. Tasks launched onto a given TMDQ 712 execute in sequential order. Task 730(0) completes before task 730(1) begins execution. Likewise, Task 740(0) completes before task 740 (1) begins execution, which, in turn, completes before task 740(1) begins execution. A task at the front of a TMDQ 712 begins execution as soon as the task is launched. So, tasks 720(0), 730(0), and 740(0) execute as soon as those tasks are launched. Tasks in different TMDQs 712 have no sequential dependencies. For example, task 730(1) could execute either before, after, or concurrently with task 740(1).
> 
> [0100] TMDQs 712 include pointers to data structures for
> 
> pending tasks, as further described below. Each TMDQ 712
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2043.png)

> **[图片提取文字 (image.png)]:**
> [0103] Execution graph 790 at nesting depth 2 is a thread group plus associated TMDQs and tasks, that has been launched by one of the tasks at nesting depth 1. Each task and TMDQ within execution graph 790 functions essentially the same as tasks and TMDQs at lower nesting levels. When each task within execution graph 790 completes, then the launching task may complete once all other commands in the launching task have completed. In this manner, sequential execution is preserved within any grid, and grids may be nested to an arbitrary nesting depth while preserving sequential execution of tasks within a stream.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2044.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 7
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2045.png)

线程组一般是warp，SM内若干线程并发，每个线程可能生成stream。

**线程组用Ctx定义，Ctx是线程组负责的任务和分配的资源**，Ctx内线程共享Stream任务和TMDQ资源。CPU线程组的Ctx是CUDA Ctx的线程。GPU线程组的Ctx是CTA或一个深度内任意线程集合。

CTA在SM运行中生成儿子grid，grid大小是CTA中调用儿子kernel的线程数×儿子kernel的调用grid，自身被挂起作为父亲CTA，通过TMDQs管理。**GPU线程组的TMDQ资源数固定且按grid静态分配，grid内线程组共享TMDQ资源**，线程组动态生成的Stream超过资源上限会共享TMDQ。

> **[图片提取文字 (image.png)]:**
> [0104] Threads within a thread group are defined in terms of a context, where the context is the set of threads that have access to the same stream and TMDQ resources. Threads within the same context may create and share TMDQs, so long as the threads are at the same nesting depth and on the same device (GPU, or CPU 102). For CPU threads, the context is defined as the set of threads associated with the CUDA context. For GPU threads, the context may represent a Cooperative Thread Array (CTA) or any set of threads that exist at the same nesting depth.
> 
> [0105] When a new stream is created by a CPU thread, CPU 102 dynamically allocates memory to support management of the stream. When the stream is subsequently destroyed after the completion of the streams tasks, CPU 102 frees the memory previously allocated for the stream. The GPU typically is not able to dynamically allocated memory. Therefore, the GPU pre-allocates context data for each context that may simultaneously execute. As a result, a thread group associated with a GPU grid has a fixed number of TMDQs that may not change during the execution of the grid. A new stream within a GPU grid is created with the cudaStreamCreate() function call. The function call returns an integer index pointing to one of the pre-allocated TMDQs in the grid. No dynamic allocation of memory is needed to create the stream. Once all tasks within a GPU stream have completed, the stream is destroyed with a cudaStreamDestroy() function call. Because no memory was dynamically allocated for the GPU stream, cudaStreamDestroy() has no memory to place back into a memory allocation free pool and therefore simply returns back to the calling program.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2046.png)

> **[图片提取文字 (image.png)]:**
> [0108] As shown, each TMDQ 812 of thread group 810 has one or more pending tasks. In one example, task 820(0)associated with stream 870 could have been launched into TMDQ 812(0), but task 860(0) associated with stream 875would not yet have been launched. Tasks 830 associated with one stream could have been launched into TMDQ (1) 812(1). Likewise, tasks 840 associated with a second stream could have been launched into TMDQ (2) 812(2), tasks 850 associated with a third stream could have been launched into TMDQ (N) 812(N), and all intervening TMDQ 812 could also have one or more associated tasks. At such a time, a thread within thread group 810 could attempt to create a new stream 875. However, the thread group 810 has a nesting depth of 1, and is associated with the GPU. Because the GPU is not able to dynamically allocate memory, a new TMDQ could not be created to accommodate the new stream 875. In such a case, tasks 860 associated with the new stream 875 could be launched into TMDQ (0) currently being used by stream 870. Stream 875 could launch tasks 860(0) and 860(1)into TMDQ (0) 812(0). Stream 870 could then launch task 820(1) into TMDQ (0) 812(0). Stream 875 could then launch task 860(2) into TMDQ (0) 812(0). Note that this approach results in unneeded dependencies. Even though streams 870 and 875 are independent of each other, the sequential nature of TMDQs results in task 860(0) depending on completion of task 820(0), task 820(1) depending on completion of task 860(1), and so on. While performance could decrease as a result, sequential ordering of tasks 820 in stream 870 and tasks 860 in stream 875 is properly preserved.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 8
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2048.png)

## Fig9、10、11、12（多命令流并发的cross-stream依赖处理）

不同cudastream的task并发，但要求按照一定顺序执行，来满足应用定义的**cross-stream依赖**。

cudaEventCreate创造等待事件，cudaStreamWaitEvent将cudaStream绑定等待事件，cudaEventRecord将事件通知某cudaStream。

过去：CPU通过cuda函数，显式调度任务执行来满足cross-stream依赖，同时避免死锁。

现在：取消CPU的显示控制和中途干预，让线程使用cuda函数定义SE和PE事件，利用scheduler kernel和TMDQ调度任务，来满足cross-stream依赖。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2048.png)

> **[图片提取文字 (image.png)]:**
> [0110] In one embodiment, task dependencies, including cross stream dependencies, are specifies using CUDA function calls cudaEventCreate( ), cudaEventRecord( ), and cudaStreamWaitEvent( ). The function cudaEventCreate( ) creates an event object and assigns an identifier to the event object. The event object may be referenced by cudaStream-WaitEvent() as an event upon which to wait. An event may be recorded to a respective stream by cudaEventRecord(). These three function calls are implemented in prior art CUDA execution models that are managed by a GPU driver running within CPU **102**. In such prior art execution models, CPU **102** explicitly schedules task execution to avoid deadlock conditions that may occur as a consequence of cross stream dependencies. Executing tasks having cross stream dependencies without CPU 102 management involvement enables greater overall execution efficiency, but also requires the cuda Event-Create(), cudaEventRecord(), and cudaStreamWaitEvent() calls to have analogous implementations that can execute within parallel processing subsystem 112. In one embodiment, the hierarchical execution graph is transformed into an equivalent hierarchical execution graph to facilitate implementation of the three calls, as described in greater detail below.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2050.png)

> **[图片提取文字 (image.png)]:**
> tion graph 902 of tasks and events for enforcing execution order among dependent tasks in different streams, according to one embodiment of the present invention. Cross stream dependence is managed using two new constructs, a wait event (WE) and a signaling (firing) event (SE). Each WE blocks until all input conditions are met. Each firing event generates one or more events upon being triggered. As shown, cross stream dependence of task B 910(1) on task A 910(0) and task D 920(0) is represented using WE 0 950(0) and SE 0 952(0). Similarly, cross stream dependence of task G 930(1) on task B 910(1), task E 920(1), and task F 930(0) is represented using WE 2(1) 950(2) and WE 1 950(1). Persons skilled in the art will recognize that hierarchical execution graph 900 of FIG. 9A and equivalent hierarchical execution graph 902 implement identical task dependencies and will therefore enforce identical execution order. Equivalent hierarchical execution graph 902 does not require locks and may therefore be implemented efficiently within parallel processing subsystem 112 without intervention from CPU 102. [0112] A WE may be queued into a particular stream to enforce execution order. For example, WE 0 950(0) waits for task A 910(0) to complete and for SE 0 952(0) to fire to
> 
> complete. Because SE 0 952(0) depends on task D 920(0) to
> 
> complete, the cross stream dependence of task B 910(0) on
> 
> both task A 910(0) and task D 920(0) is properly enforced.
> 
> [0111] FIG. 9B illustrates an equivalent hierarchical execu-
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2051.png)

线程通过cudaEventCreate创造SE和WE，通过cudaEventRecord将SE加入cudaStream（TMDQ），通过cudaStreamWaitEvent将WE加入cudaStream来等待SE完成。

> **[图片提取文字 (image.png)]:**
> described below in FIG. 11, describes signaling event state. The data structure may be queued into a given stream. Each signaling event maintains a list of events that are waiting on the signaling event to fire. Initially the list is empty, as no waiting event has year been created to wait on the signaling event. When the signaling event fires (there is not stream work ahead of it), an associated list is traversed to mark the signaling event as having fired. Furthermore, waiting events are notified that the signaling event has fired. Notifying a waiting event satisfies one or potentially, more events needed by the signaling event to complete, allowing a corresponding stream to progress. [0115] A thread may create a cross stream dependency by calling cudaStreamWaitEvent() to queue a WE into a given
> 
> [0114] A thread may call cudaEventRecord() to queue an
> 
> SE into a stream. If the event has already been recorded, a new
> 
> event must be allocated and tracked as a most recent event
> 
> record. A data structure, such as a task status data structure,
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2052.png)

> **[图片提取文字 (image.png)]:**
> stream. The waiting event must be first allocated from the free pool of event structures. In one embodiment, the WE looks up the most recent call to cudaEventRecord() on the referenced signaling event. If there is nothing to wait for, then the WE may complete. Once the most recent signaling event is found, the WE is atomically added to a waiting list associated with a respective signaling event object. A WE added to the same stream as the stream the SE to be waited upon is treated semantically as a null operation.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> [0116] When a scheduler kernel is run upon completion of a grid, the scheduler should run the next task in an associated stream. If the next task another grid, then the scheduler kernel simply launches the grid. However, if the next task is an event (WE or SE), then the scheduler kernel needs to handle all ready to fire events (e.g., fire a signaling event, which unblocks one or more waiting events, which in turn may unblock one or more signaling events).
> 
> [0117] Each WE includes a data structure with a dependency count of how many items the WE is waiting on to complete. In one embodiment, the count may be zero, one or two. A count of zero indicates that the event has fired. A count of one indicates the WE is waiting on one task or event. A count of two indicates that the event is waiting on both a signaling event and another task or event in the same stream. This count is decremented atomically whenever a scheduler kernel determines that one dependency is satisfied. When the count is decremented to zero, a stream next (StreamNext) pointer within the data structure may be traversed and related dependency counts are decremented. All access to the dependency count should be atomic to guarantee that only one scheduler kernel attempts to schedule a next task in a stream, or schedule the dependent list of WE's.
> 
> [0118] When a signaling event fires, an entire related event-WaitingList must be traversed, decrementing every waiting event dependency count associated with the event. For events that have no more dependencies, each stream next pointer must be traversed. A stream next pointer associated with the signaling event must also be traversed because the signaling event is completing. To avoid the need to build a traversal state stack, the tree-walk algorithm is flattened. Any technically feasible non-recursive tree-walking technique may be implemented. On machines where limited thread-local stack space is less of an issue than current PPU implementations, any, technically feasible recursive tree-walking technique may be implemented.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2054.png)

**线程组定义为应用中协作的多个cudaStream**。在不考虑cudaStream，也不存在cross-stream时，线程组Ctx就是pushbuffer中命令流的计算TMD，因此将cudaStream的slot命名为PB（pushbuffer）。

**线程组Ctx**包含管理**协作cudaStream（PB）**的TMDQ中末尾任务的指针last task。

某个cudaStream中的线程组可能产生新cudaStream或grid，即动态产生新任务，改变线程Ctx在PB中定义的last task。

> **[图片提取文字 (image.png)]:**
> structure including parameters and context information associated with a thread group, according to one embodiment of the present invention. As shown, the thread group context 1020 includes a last task pointer 1040 for each TMDQ in the thread group, and a work counter 1050. [0120] The last task pointer 1040 is a pointer to the last task in the associated TMDQ. When a new task in launched into a TMDQ, the last task pointer 1040 is updated via an atomic operation to indicate the new task is now the last task in the TMDQ. Table 8, below, illustrates launching a new task in a TMDQ in an exemplary CUDA program.
> 
> [0119] FIG. 10 illustrates a thread group context 1020 data
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2055.png)

> **[图片提取文字 (image.png)]:**
> at address StreamEnd is overwritten with a pointer to NewTask, and the prior value in the last task pointer 1040 is returned as FormerStreamEnd. If FormerStreamEnd is nonzero (that is, FormerStreamEnd is a pointer to a task), then the StreamNext value associated with the task is updated to point to the newly launched tasks. If FormerStreadEnd is zero, then no tasks are pending in the TMDQ, and the new task may begin execution immediately. [0122] The example of Table 8 executes within a critical section of operation so as to avoid deadlock where a thread has posted a task into a stream, but then the thread has been swapped out before launching the new task. In such a case, deadlock may occur if the swapped out thread is not allowed to be swapped back until the new task has completed. However, the new task may not begin execution because the new task has not yet been launched.
> 
> [0121] In the example of Table 8, the last task pointer 1040
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2056.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE 8
> 
> ```
> formerStreamEnd = atomicExchange(&streamEnd, newTask);
> if (formerStreamEnd) {
>     formerStreamEnd->streamNext = newTask;
> } else {
>     LaunchTask(newTask);
> ```
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2057.png)

> **[图片提取文字 (image.png)]:**
> ## Thread Group Context 1020
> 
> | PB(0): Last Task  |
> |-------------------|
> | PB(1): Last Task  |
> | PB(2): Last Task  |
> | •                 |
> | PB(N)N: Last Task |
> | Work Counter      |
> |                   |
> 
> Figure 10
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2044.png)

cudaStream是任务负载，TMDQ是管理cudaStream的队列（线程组运行的Ctx资源）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> | 1140~ | Task ID                 |
> |-------|-------------------------|
> | 1142~ | Stream Next             |
> | 1144~ | Thread Group Context ID |
> |       | •                       |
> 
> Figure 11A
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2058.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 11B
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> | 1160~ | Event ID    |
> |-------|-------------|
> | 1162~ | Stream Next |
> | 1164~ | Dep. Count  |
> |       | •           |
> 
> Figure 11C
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2046.png)

> **[图片提取文字 (image.png)]:**
> [0123] When a task completes, a scheduler executing on parallel processing subsystem 112 reads the last stream pointer corresponding to the TMDQ associated with the completed task. If the last task pointer 1040 of the associated TMDQ does not point to the completed task, then the completed task is not the last task in the TMDQ. In such a case, the scheduler causes the next task in the TMDQ to begin execution, as described below in conjunction with FIG. 11. If the last task pointer 1040 of the associated TMDQ points to the completed task, then the completed task is the last task in the TMDQ. In such a case, the scheduler performs an atomic compare and swap to set the last task pointer 1040 to a null pointer and read the value currently stored in the last task point 1040. The scheduler performs a function call in the form of "currentEnd=atomicCAS(&StreamEnd, finishedTask, NULL)," where "StreamEnd" is the last task pointer 1040 of the associated TMDQ, "finishedTask" is a pointer to the completed task, and "NULL" is the null pointer. The function atomically returns the value stored in the last task pointer **1040**, as represented by "currentEnd" in the function call.
> 
> [0124] If the value of "currentEnd" is a pointer to the completed task, then all tasks in the TMDQ have completed, and no new task has been launched. The scheduler knows that all tasks in the stream have completed. If the value of "currentEnd" is not a pointer to the completed task, then a new task has been launched, and the thread group context 1020 has been updated to reflect the existence of the new task. In such a case, the scheduler reads the StreamNext pointer (described below) associated with the completed task. If the StreamNext pointer associated with the completed task is non-zero, then the scheduler causes the task at address StreamNext to begin execution. If the value of StreamNext is the null pointer, then a new task has been launched, but the task status has not yet been updated to reflect the existence of the new task. In such a case, the scheduler monitors StreamNext until the value changes from the null pointer to a pointer to the new task. The scheduler then causes the new task pointed to by StreamNext to begin execution.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2059.png)

> **[图片提取文字 (image.png)]:**
> including parameters associated with a computational task, according to one embodiment of the present invention. As shown, the task status 1120 includes a task identifier (task ID) 1140, a stream next pointer 1142, a thread group context identifier (thread group context ID) 1144, and other parameters associated with the task (not shown). [0126] The task ID 1140 is a unique identifier pointing to the task associated with the task status 1120. A task status
> 
> [0125] FIG. 11A illustrates a task status 1120 data structure
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2060.png)

> **[图片提取文字 (image.png)]:**
> launched on a TMDQ. The task ID enables the scheduler to find the task associated with a given task status 1120. [0127] The stream next pointer 1142 is a pointer to the next task in the TMDQ. When a task completes, the scheduler reads the next stream pointer to determine where to find the next task in the TMDQ that may begin execution. The scheduler then causes the task located at the address pointed to by the stream next pointer 1142 to begin execution. If the completed task is the last task in the TMDQ, then the stream next pointer 1142 is set to a null pointer. [0128] The thread group context ID 1120 is a unique identifier pointing to the thread group context 1020 associated
> 
> 1120 is created for each new task as tasks are created and
> 
> tifier pointing to the thread group context 1020 associated with the task status 1120. When a task completes, the scheduler reads the thread group context ID 1120 to find the thread group context 1020. The scheduler can then perform associated task completion steps, such as updating the work counter closing a TMDQ, and closing a context, as described above in association with FIG. 10.
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2061.png)

> **[图片提取文字 (image.png)]:**
> structure including parameters associated with a signal event, according to one embodiment of the present invention. Signal event status 1122 data structure comprises an event identifier (ID) 1150, a stream next pointer 1152 and an event next pointer 1154. Event ID 1150 uniquely identifies a particular event upon which different tasks may depend. Stream Next 1152 has substantially identical meaning and function relative to Stream Next 1142 of FIG. 11A. Event Next 1154 is a pointer to a next dependent event. [0130] FIG. 11C illustrates a wait event status data 1124 structure including parameters associated with a computational task, according to one embodiment of the present invention. Wait event status 1124 data structure comprises an event ID 1160, a stream next pointer 1162, and a dependency count 1164. Event ID 1160 is defined and operates substantially identically to Event ID 1150 of FIG. 11B. Stream Next 1162 is defined and operates substantially identically to Stream Next 1142 of FIG. 11A. [0131] Task status 1120 data structure of FIG. 11A, signal event status 1122 data structure of FIG. 11B, and wait event status data 1124 structure comprise items that may be queued within a TMDQ for processing. Tasks are queued to initiate a corresponding computation, while events enforce computation ordering among a plurality of queued tasks. Certain
> 
> applications have execution dependencies among different
> 
> tasks that need to be satisfied before a particular task is per-
> 
> mitted to execute.
> 
> [0129] FIG. 11B illustrates a signal event status 1122 data
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2062.png)

scheduler kernel从对应TMDQ的队头取item（TMD、WED、SED），按照规则执行操作，让task执行满足cross-stream依赖。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2063.png)

> **[图片提取文字 (image.png)]:**
> the systems of FIGS. 1-4 and 6, persons of ordinary skill in the art will understand that any system configured to perform the method steps, in any order, is within the scope of the present invention. [0133] In one embodiment, method 1200 is executed by a scheduler kernel, discussed previously in FIG. 5. The scheduler kernel executes after a task completes to determine what work should be scheduled next. Method 1200 processes an item at the front of stream task queue, such as a TMDO, described previously. [0134] The method begins in step 1210, where the scheduler kernel retrieves a next item from a corresponding TMDQ. In one embodiment, the next item may comprise a task metadata descriptor, a waiting event descriptor, or a
> 
> [0132] FIG. 12 is a flow diagram of method 1200 for deter-
> 
> mining that task execution dependencies have been satisfied,
> 
> according to one embodiment of the present invention.
> 
> Although the method steps are described in conjunction with
![image.png](nested%20grid%E3%80%81stream%E5%9C%A8GPU%E4%B8%8A%E6%89%A7%E8%A1%8C%EF%BC%88TG%E3%80%81CTA%E3%80%81nested%E3%80%81runtime%EF%BC%89/image%2064.png)