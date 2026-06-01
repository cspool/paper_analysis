# GPU并发机制TS、MPS、MIG、vGPU

## 提示词

**基础知识**

作为一个GPU架构专家，你知道如下基础知识：

作为一个GPU架构专家，请判断下面总结是否正确：

1、多应用在GPU上的执行。

1）不同应用的任务轮流在GPU执行，channels是GPU为应用卸载任务提供的通道，每个channel占用GPU Partition/vGPU一个时间片TS后或Ctx Drain后被新channel抢占。

2）主机接口轮流读取不同channel，则runlist是调度不同channel执行的队列。interleave freq越大的channel在runlist中出现频率更高，对应高优先级应用。

zzzzzzzzzzzzzz

3）前端从channel buffer中读取work指针，构建和维护应用定义的Stream（计算grid队列和图形命令队列）。

4）前端scheduler调度不同Stream中grid到GPC，可配置Stream优先级。

5）前端WDU按照SM-CTA/GPC-CTA/GPU-CTA粒度调度任务，并分别派发到SM/GPC/uGPU范围内的SM。

zzzzzzz

6）SM内warp内线程按SIMT pipeline并行，不同warp轮流发射指令。

2、上下文Ctx是channel中任务在GPU分配资源上的运行时镜像。

1）不同channel按照时间片TS轮流执行，需要切换不同的channel Ctx。

2）stream并发是channel Ctx内并发不同Stream的grid，不用切换channel Ctx。

3）SM内的warp并发，不用切换channel Ctx，并且SM内并发warp的Ctx同时存在，不用切换。

3、旧channel到达timeslice，新channel抢占GPU过程

1）Host提前在显存VRAM分配Ctx buffer空间。

2）主机接口从Runlist中选择新channel，抢占旧channel的Ctx占据的资源，前端和各层trap handler控制抢占过程。

3）GPU停止context处理、保存context到内存、重置Engine/SM、加载上下文、恢复任务启动。

4）GPU完成抢占后，前端向主机接口发送ACK，切换新channel执行。

4、GPU抢占的原理

1）暂停不同深度层次上资源（前端和GPC、SM）上的任务（CTA或指令）调度和发射。

2）等待抢占深度上资源空闲，即前端和GPC已调度CTA或SM已发射指令完成后释放资源。

3）资源空闲后，保存旧Ctx，加载新Ctx并且开始执行。

**GPU专利**

现在我给你一些资料，需要你回答下面问题。

1、对比上述基础知识，有哪些修正和新的认知？

2、有什么新的机制？回答时，请给出例子。

2、这些新机制的设计目的是什么？应用背景是什么？

**GPU sharing论文**

现在我给你一些关于GPU上多任务抢占的论文，需要你回答下面问题。

1、对比上述基础知识，有哪些修正和新的认知？

2、论文对比的baseline是什么？baseline在多任务场景中有什么问题？回答时，请给出例子。

2、论文相比baseline做出的修改是什么？回答时，请给出例子。

应用将任务卸载到GPU channel后，无法干预GPU对runlist内channel和channel内CTA的调度执行，因为GPU没有提供相关接口。

MIG、μGPU是物理隔离的GPU划分，看成算力和容量完全隔离的GPU Partition（资源）。相比GPU进行channel切换时，GPU所有资源执行stall-unload-load带来的抢占延迟和吞吐下降，partition内开销更少，partition之间互不影响。

**CTA是资源**，是SM并发能力的单位，warp是资源，是SM的SIMT并行能力单位。

vGPU、vGPC、vTPC等是关联物理资源的虚拟化封装，为应用的不同粒度任务（VM、block）提供更灵活的资源分配方式。Migration开销更小？抢占开销呢？

## **GPC、SM；TG、CTA/block、CGA、CG；channel、runlist**

**channel是GPU TMD指针的集合，应用包含若干channel。channel是TMU中的work group硬件模块。多channel对应Kepler中hyperQ。pushbuffer是Stream在memory中的存储区域。**

**channel**抢占通过清除channel ram entry的valid后写入preempt寄存器来unload Ctx，完成后切换runlist中下一个channel。

**runlist**抢占将新runlist写入runlist寄存器，找到有pending任务的第一个channel来load Ctx和执行。

**Timeslice Out**抢占，每个**channel**的时间片按照**Ctx中命令流stream的方法数（kernel call）定义**，dense stream的更长，sparse stream的更短。

> **[图片提取文字 (image.png)]:**
> configured to access the buffer in a system memory connected to the interconnect 302 via memory requests transmitted over the interconnect 302. In an embodiment, the host processor writes the command stream to the buffer and then transmits a pointer to the start of the command stream to the PPU 300. The front end unit 315 receives pointers to one or more command streams. The front end unit 315 manages the one or more streams, reading commands from the streams and forwarding commands to the various units of the PPU 300. [0064] The front end unit 315 is coupled to a scheduler unit 320 that configures the various GPCs 350 to process tasks defined by the one or more streams. The scheduler unit 320 is configured to track state information related to the various tasks managed by the scheduler unit 320. The state may indicate which GPC 350 a task is assigned to, whether the task is active or inactive, a priority level associated with the task, and so forth. The scheduler unit 320 manages the execution of a plurality of tasks on the one or more GPCs 350.
> 
> the PPU 300. For example, the I/O unit 305 may be
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image.png)

> **[图片提取文字 (image.png)]:**
> [0041] The first set of graphics tasks 106 and the second set of compute tasks 108 are scheduled for simultaneous execution on the processing unit 104 by a hardware scheduler 110 that schedules tasks from a graphics queue 114 of graphics work items and a compute queue 116 of compute work items in a memory 117. The graphics work items in the graphics queue 114 may be enqueued by one or more graphics pipelines, and the compute work items in the compute queue 116 may be enqueued by one or more compute pipelines. A "graphics pipeline" (an example is
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%201.png)

> **[图片提取文字 (image.png)]:**
> [0042] The hardware scheduler 110 is configured to be operating in either a "graphics-greedy mode" in which it repeatedly extracts work items from the graphics queue 114 and launches to the processing unit 104, or in a "computegreedy mode" in which it repeatedly extracts work items from the compute queue 116 and launches to the processing unit 104.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%202.png)

> **[图片提取文字 (image.png)]:**
> ## Runlist Interleave Frequency
> 
> The runlist is an ordered list of channels that the GPU HOST reads to find work for the downstream engines to complete. To enable the GPU HOST to schedule a given channel more often, include the channel multiple times on a runlist. For each priority level, the runlist interleave frequency must be set to match the priority.
> 
> For example, if a system has one high-priority application, one medium-priority application, and two low-priority applications, the GPU scheduler constructs the runlist as follows:
> 
> ![](_page_0_Picture_3.jpeg)
> 
> The scheduling latency for when a high-priority application will be able to run is governed by:
> 
> worst-case latency(high) = (h-1) × timeslice(high) + execution time(low) + channel reset
> 
> Where:
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## Program Execution And Preemption
> 
> Preemption may be used to time-slice a processor between multiple different applications so that the different applications are serialized and each execute for a short time-slice on the processor. Preemption may also be used to unload the currently executing context for other purposes. For example, the host interface 206 may preempt a context when the CPU 102 initiates a channel preempt or a runlist preempt, where a channel is a collection of pointers to processing work and an application may contain one or more channels. A channel preempt is performed by clearing a valid bit in a channel ram entry and writing a channel identifier of the channel to be preempted to a preempt register. The specified channel is then unloaded from the PPU 202 off both host and the engine.
> 
> A runlist preempt is performed by writing a pointer to the runlist register. The pointer may point to a new runlist or may point to the runlist that is currently active. Runlist preempt causes what is running in a PPU 202 to be unloaded. The host interface 206 then begins processing at the first entry on the runlist associated with the pointer and searches for the first valid entry with pending work. The first channel on the runlist which has pending work is loaded into the PPU 202
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%204.png)

> **[图片提取文字 (image.png)]:**
> The host interface 206 may also preempt a context that is executing before a time slice has expired when the context is out of methods (i.e. programs) and another context is waiting to execute. In one embodiment, the time slices are not equal amounts of time, but instead are based on each context's method stream, so that a context with a dense method stream is allocated a larger time slice compared with a different context having a sparse method stream. The host interface 206 is configured to indicate to the front end 212 when the host interface 206 does not have any methods for an executing context. However, the host interface 206 does not initiate a context switch for the executing context until either the time slice allocated to the context has expired or the processing pipeline is idle and there are no methods.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%205.png)

应用的**kernel调用或API调用**经过GPU驱动转为**任务task（TMD）**，应用可能存在多个**命令流协作（Streams）**，存储在内存的**pushbuffer区域**。（数据结构DS定义命令流，DS指针写入pushbuffer来初始化命令流）

channel是TMU管理Stream的硬件队列，应用只能占用1个channel（占用了GPU全部），因为HyperQ之前GPU只存在一个channel。

**runlist**定义了轮流读取不同pushbuffer的顺序，即**应用轮流抢占channel的顺序**，高优先级应用的channel出现/读取频率更高。

GPU从不同pushbuffer轮流读取命令流Stream，通过WMU中的channel管理任务。**命令流Stream就是cudaStream。命令流包含SM处理的命令发给host interface**、**Memory操作的命令发给memory cross bar**。

**SM处理的命令包含TMD指针**。计算任务编码成TMD，包含数据坐标、state参数、定义数据处理的命令。

TMD的调度优先级由channle决定，channel的优先级由管理的pushbuffer优先级确定。

**WMU**分别维护**不同优先级应用的task group（channel）并按优先级调度**。将pushbuffer存储Stream，扩展到TMDQs和GTMD queue并存在memory，管理动态生成的grid TMD或Stream。

**WDU**接受TMD指针，确保在GPC处于valid状态后，读取TMD并初始化TMD，将**grid TMD或queue TMD定义的若干CTA发到GPC**。grid TMD一般是计算任务，queue TMD一般是图形任务（可能动态生成子任务）。

> **[图片提取文字 (image.png)]:**
> other system components. In particular, CPU 102 issues commands that control the operation of PPUs 202. In some embodiments, CPU 102 writes a stream of commands for each PPU 202 to a data structure (not explicitly shown in either FIG. 1 or FIG. 2) that may be located in system memory 104, parallel processing memory 204, or another storage location accessible to both CPU 102 and PPU 202. A pointer to each data structure is written to a pushbuffer to initiate processing of the stream of commands in the data structure. The PPU 202 reads command streams from one or more pushbuffers and then executes commands asynchronously relative to the operation of CPU 102. Execution priorities may be specified for each pushbuffer to control scheduling of the different pushbuffers.
> 
> In operation, CPU 102 is the master processor of com-
> 
> puter system 100, controlling and coordinating operations of
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%206.png)

> **[图片提取文字 (image.png)]:**
> appropriate components of PPU 202. For example, commands related to processing tasks may be directed to a host interface 206, while commands related to memory operations (e.g., reading from or writing to parallel processing memory 204) may be directed to a memory crossbar unit 210. Host interface 206 reads each pushbuffer and outputs the command stream stored in the pushbuffer to a front end 212.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%207.png)

> **[图片提取文字 (image.png)]:**
> Each PPU 202 advantageously implements a highly parallel processing architecture. As shown in detail, PPU 202(0) includes a processing cluster array 230 that includes a number C of general processing clusters (GPCs) 208, where C≥1. Each GPC 208 is capable of executing a large number (e.g., hundreds or thousands) of threads concurrently, where each thread is an instance of a program. In various applications, different GPCs 208 may be allocated for processing different types of programs or for performing different types of computations. The allocation of GPCs 208 may vary dependent on the workload arising for each type of program or computation.
> 
> GPCs 208 receive processing tasks to be executed from a work distribution unit within a task/work unit 207. The work distribution unit receives pointers to compute processing tasks that are encoded as task metadata (TMD) and stored in memory. The pointers to TMDs are included in the command stream that is stored as a pushbuffer and received by the front end unit 212 from the host interface 206. Processing tasks that may be encoded as TMDs include indices of data to be processed, as well as state parameters and commands defining how the data is to be processed (e.g., what program is to be executed). The task/work unit 207 receives tasks from the front end **212** and ensures that GPCs 208 are configured to a valid state before the processing specified by each one of the TMDs is initiated. A priority may be specified for each TMD that is used to schedule execution of the processing task.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%208.png)

> **[图片提取文字 (image.png)]:**
> ing on the host processor to schedule operations for execution on the PPU 300. In an embodiment, multiple compute applications are simultaneously executed by the PPU 300 and the PPU 300 provides isolation, quality of service (QoS), and independent address spaces for the multiple compute applications. An application may generate instructions (i.e., API calls) that cause the driver kernel to generate one or more tasks for execution by the PPU 300. The driver kernel outputs tasks to one or more streams being processed by the PPU 300. Each task may comprise one or more groups of related threads, referred to herein as a warp. In an embodiment, a warp comprises 32 related threads that may be executed in parallel. Cooperating threads may refer to a plurality of threads including instructions to perform the task and that may exchange data through shared memory. Threads and cooperating threads are described in more detail in conjunction with FIG. 5A.
> 
> [0068] In an embodiment, a host processor executes a
> 
> driver kernel that implements an application programming
> 
> interface (API) that enables one or more applications execut-
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%209.png)

> **[图片提取文字 (image.png)]:**
> [0046] FIG. 3A is a block diagram of the task/work unit 207 of FIG. 2, according to one embodiment of the present invention. The task/work unit 207 includes a task management unit 300 and the work distribution unit 340. The task management unit 300 organizes tasks to be scheduled based on execution priority levels. For each priority level, the task management unit 300 stores a list of pointers to the TMDs 322 corresponding to the tasks in the scheduler table 321, where the list may be implemented as a linked list. The TMDs 322 may be stored in the PP memory 204 or system memory 104. The rate at which the task management unit 300 accepts tasks and stores the tasks in the scheduler table 321 is decoupled from the rate at which the task management unit 300 schedules tasks for execution. Therefore, the task management unit 300 may collect several tasks before scheduling the tasks. The collected tasks may then be scheduled based on priority information or using other techniques, such as round-robin scheduling.
> 
> The work distribution unit 340 includes a task table 345 with slots that may each be occupied by the TMD 322 for a task that is being executed. The task management unit 300 may schedule tasks for execution when there is a free slot in the task table 345. When there is not a free slot, a higher priority task that does not occupy a slot may evict a lower priority task that does occupy a slot. When a task is evicted, the task is stopped, and if execution of the task is not complete, then a pointer to the task is added to a list of task
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2010.png)

> **[图片提取文字 (image.png)]:**
> pointers to be scheduled so that execution of the task will resume at a later time. When a child processing task is generated, during execution of a task, a pointer to the child task is added to the list of task pointers to be scheduled. A child task may be generated by a TMD 322 executing in the processing cluster array 230.
> 
> [0048] Unlike a task that is received by the task/work unit 207 from the front end 212, child tasks are received from the
> 
> processing cluster array 230. Child tasks are not inserted into pushbuffers or transmitted to the front end. The CPU 102 is not notified when a child task is generated or data for the child task is stored in memory. Another difference between the tasks that are provided through pushbuffers and child tasks is that the tasks provided through the pushbuffers are defined by the application program whereas the child tasks are dynamically generated during execution of the tasks.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2011.png)

TMD分为grid和queue，grid TMD是CTA的三维阵列，queue TMD是图形pipeline或dataflow任务。

> **[图片提取文字 (image.png)]:**
> with different levels of accessibility. Special registers (not shown) are readable but not writeable by LSU 303 and are used to store parameters defining each thread's "position." In one embodiment, special registers include one register per thread (or per exec unit 302 within SM 310) that stores a thread ID; each thread ID register is accessible only by a respective one of the exec unit 302. Special registers may also include additional registers, readable by all threads that execute the same processing task represented by a TMD 322 (or by all LSUs 303) that store a CTA identifier, the CTA dimensions, the dimensions of a grid to which the CTA belongs (or queue position if the TMD 322 encodes a queue task instead of a grid task), and an identifier of the TMD 322 to which the CTA is assigned.
> 
> [0062] SM 310 provides on-chip (internal) data storage
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2012.png)

> **[图片提取文字 (image.png)]:**
> include a compute priority which represents the priority level to be set for compute work (e.g., this is the priority set by the driver inside each compute work item data structure (e.g., referred to sometimes as "QMD")), and a compute MPC resource reserve parameter representing the number of CTAs ("cooperative thread array" (also cooperative group above) or other quantity of compute work) to reserve in MPC when compute is granted by arbiter when in graphics greedy mode. The default is set at a small value such as, for example, 1 in order let compute just trickle in while in graphics greedy mode.
> 
> [0147] The compute-related policy parameters also
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2013.png)

> **[图片提取文字 (image.png)]:**
> [0063] If the TMD 322 is a grid TMD, execution of the TMD 322 causes a fixed number of CTAs to be launched and executed to process the fixed amount of data stored in the queue 525. The number of CTAs is specified as the product of the grid width, height, and depth. The fixed amount of data may be stored in the TMD 322 or the TMD 322 may store a pointer to the data that will be processed by the CTAs. The TMD 322 also stores a starting address of the program that is executed by the CTAs.
> 
> [0064] If the TMD 322 is a queue TMD, then a queue feature of the TMD 322 is used, meaning that the amount of data to be processed is not necessarily fixed. Queue entries store data for processing by the CTAs assigned to the TMD 322. The queue entries may also represent a child task that is generated by another TMD 322 during execution of a thread, thereby providing nested parallelism. Typically, execution of the thread, or CTA that includes the thread, is suspended until execution of the child task completes. The queue may be stored in the TMD 322 or separately from the TMD 322, in which case the TMD 322 stores a queue pointer to the queue. Advantageously, data generated by the child task may be written to the queue while the TMD 322 representing the child task is executing. The queue may be implemented as a circular queue so that the total amount of data is not limited to the size of the queue.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2014.png)

GPC管理多个TMD的CTA，**SM按照CTA粒度接取负载、分配和保留Ctx资源（SCG），按线程组warp执行**，pipeline manager可能给每个SM配置Controller来分别管理SM内CTA的运行情况，如哪些warp被抢占。

GPC是图形pipeline的图形处理单元/Tile，只考虑通用计算的情况下，GPC包含SM集合和相关机制的模块支持（复用图形模块的datapath）。

**线程组在表述中表示运行相同code的一组线程，大小是warp、block、CG都可。**

**warp是SM执行CTA的单位，和SM中lane数量成比例（SIMD-Cores多周期完成），分支发散时部分idle/masked。**

线程组/warp是任务划分，warp Ctx是资源单位。线程（模型）是逻辑上的负载，Ctx是负载实际执行的环境/资源（指令、数据、存储、隔离）。

**CTA是指SM内通过sharedMem协作的线程阵列，CTA大小是blockSz定义的线程数，不能超过SM的并发线程容量，通过编程时的blockSz定义。**

> **[图片提取文字 (image.png)]:**
> unit 340 passes the preempt command to the pipeline manager 305 in the GPCs 208. The pipeline manager 305 may include a controller for each SM 310. Upon receiving the preempt command, the SMs 310 stop issuing instructions and enter a trap handler. The SMs 310 also wait for all memory transactions associated with previously issued instructions to complete, i.e., for all outstanding memory requests to complete. Memory requests are considered to be outstanding when data for a read request has not been returned and when an acknowledgement has not been received from the MMU 328 for a write request for which an acknowledgement was explicitly requested. The pipeline managers 305 maintain information about CTAs and thread groups and track which thread groups are preempted per
> 
> CTA.
> 
> During instruction level preemption, the work distribution
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2015.png)

> **[图片提取文字 (image.png)]:**
> unit 340 passes the preempt command to the pipeline manager 305 in the GPCs 208. The pipeline manager 305 may include a controller for each SM 310. Upon receiving the preempt command, the SMs 310 stop issuing instructions and enter a trap handler. The SMs 310 also wait for all memory transactions associated with previously issued instructions to complete, i.e., for all outstanding memory requests to complete. Memory requests are considered to be outstanding when data for a read request has not been returned and when an acknowledgement has not been received from the MMU 328 for a write request for which an acknowledgement was explicitly requested. The pipeline managers 305 maintain information about CTAs and thread groups and track which thread groups are preempted per CTA.
> 
> During instruction level preemption, the work distribution
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2016.png)

> **[图片提取文字 (image.png)]:**
> when the preempt command was output by the work distribution unit 340. The state information indicates whether a thread group exited after completing execution or if the thread group was preempted. The state information is saved by the pipeline managers 305 and may be used by the pipeline managers 305 to restore only those thread groups that were preempted. When all of the threads in a thread group exit after the pipeline manager 305 receives the preempt command and before the trap handler is entered to store the state information, state information is not stored for the thread group and the thread group is not restored. After the GPCs 208 are idle, the GPCs may be reset to complete the third phase of the preemption process.
> 
> The pipeline manager 305 holds state information for
> 
> each thread group that was executing within the GPC 208
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2017.png)

> **[图片提取文字 (image.png)]:**
> 208 constitutes a thread, as previously defined herein, and the collection of a certain number of concurrently executing threads across the parallel processing engines (not shown) within an SM 310 is referred to herein as a "warp" or "thread group." As used herein, a "thread group" refers to a group of threads concurrently executing the same program on different input data, with one thread of the group being assigned
> 
> to a different processing engine within an SM 310. A thread
> 
> group may include fewer threads than the number of pro-
> 
> cessing engines within the SM 310, in which case some
> 
> processing engines will be idle during cycles when that
> 
> thread group is being processed. A thread group may also
> 
> The series of instructions transmitted to a particular GPC
> 
> include more threads than the number of processing engines within the SM 310, in which case processing will take place over consecutive clock cycles. Since each SM 310 can support up to G thread groups concurrently, it follows that up to G\*M thread groups can be executing in GPC 208 at any given time.
> 
> Additionally, a plurality of related thread groups may be active (in different phases of execution) at the same time within an SM 310. This collection of thread groups is referred to herein as a "cooperative thread array" ("CTA") or "thread array." The size of a particular CTA is equal to m\*k, where k is the number of concurrently executing threads in a thread group and is typically an integer multiple of the number of parallel processing engines within the SM 310, and m is the number of thread groups simultaneously active
> 
> within the SM 310. The size of a CTA is generally deter-
> 
> mined by the programmer and the amount of hardware
> 
> resources, such as memory or registers, available to the
> 
> CTA.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2018.png)

> **[图片提取文字 (image.png)]:**
> dispatches tasks for execution on the GPCs 350 of the PPU 300. The tasks are allocated to a particular DPC 420 within a GPC 350 and, if the task is associated with a shader program, the task may be allocated to an SM 440. The scheduler unit 510 receives the tasks from the work distribution unit 325 and manages instruction scheduling for one or more thread blocks assigned to the SM 440. The scheduler unit 510 schedules thread blocks for execution as warps of parallel threads, where each thread block is allocated at least one warp. In an embodiment, each warp executes 32 threads. The scheduler unit 510 may manage a plurality of different thread blocks, allocating the warps to the different thread blocks and then dispatching instructions from the plurality of different cooperative groups to the various functional units (i.e., cores 550, SFUs 552, and LSUs 554) during each clock cycle.
> 
> [0084] As described above, the work distribution unit 325
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2019.png)

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

**协作组CGA是应用定义的协作的线程集合，包含CTA但支持比CTA更大范围的协作。**

CG或CGA是CTA的扩展，**扩展了CTA在SM内协作和同步的模式，提供更大量线程在更大资源范围内的协作和同步。**

> **[图片提取文字 (image.png)]:**
> [0085] Cooperative Groups is a programming model for organizing groups of communicating threads that allows developers to express the granularity at which threads are communicating, enabling the expression of richer, more efficient parallel decompositions. Cooperative launch APIs support synchronization amongst thread blocks for the execution of parallel algorithms. Conventional programming models provide a single, simple construct for synchro-
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2022.png)

> **[图片提取文字 (image.png)]:**
> thread block (i.e., the syncthreads() function). However, programmers would often like to define groups of threads at smaller than thread block granularities and synchronize within the defined groups to enable greater performance, design flexibility, and software reuse in the form of collective group-wide function interfaces. [0086] Cooperative Groups enables programmers to define groups of threads explicitly at sub-block (i.e., as small as a single thread) and multi-block granularities, and to perform collective operations such as synchronization on the threads in a cooperative group. The programming model supports clean composition across software boundaries, so that libraries and utility functions can synchronize safely within their local context without having to make assumptions about convergence. Cooperative Groups primitives enable new patterns of cooperative parallelism, including producer-consumer parallelism, opportunistic parallelism, and global synchronization across an entire grid of thread blocks.
> 
> nizing cooperating threads: a barrier across all threads of a
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2023.png)

> **[图片提取文字 (image.png)]:**
> include a compute priority which represents the priority level to be set for compute work (e.g., this is the priority set by the driver inside each compute work item data structure (e.g., referred to sometimes as "QMD")), and a compute MPC resource reserve parameter representing the number of CTAs ("cooperative thread array" (also cooperative group above) or other quantity of compute work) to reserve in MPC when compute is granted by arbiter when in graphics greedy mode. The default is set at a small value such as, for example, 1 in order let compute just trickle in while in graphics greedy mode.
> 
> [0147] The compute-related policy parameters also
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2013.png)

> **[图片提取文字 (image.png)]:**
> [0026] To take advantage of increased parallelism offered by modern GPUs, NVIDIA in CUDA Version 9 introduced a software-based "Cooperative Groups" API for defining and synchronizing groups of threads in a CUDA program to allow kernels to dynamically organize groups of threads. See e.g., https://developer.nvidia.com/blog/cooperative-groups/ (retrieved 2021); https://developer.nvidia.com/blog/cuda-9features-revealed/(retrieved 2021); Bob Crovella et al, "Cooperative Groups" (Sep. 17, 2020), https://vimeo.com/ 461821629; US2020/0043123. [0027] Before Cooperative Groups API, both execution control (i.e., thread synchronization) and inter-thread communication were generally limited to the level of a thread block (also called a "cooperative thread array" or "CTA") executing on one SM. The Cooperative Groups API extended the CUDA programming model to describe synchronization patterns both within and across a grid (see FIG. 8 discussed below) or across multiple grids and thus potentially (depending on hardware platform) spanning across
> 
> devices or multiple devices.
> 
> [0025] Cooperative Groups API Software Implementation
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2024.png)

> **[图片提取文字 (image.png)]:**
> ing groups of threads—where "groups" are programmable and can extend across thread blocks. The Cooperative Groups API also provides host-side APIs to launch grids whose threads are all scheduled by software-based scheduling to be launched concurrently. These Cooperative Groups API primitives enable additional patterns of cooperative parallelism within CUDA, including producer-consumer parallelism and global synchronization across an entire thread grid or even across multiple GPUs, without requiring hardware changes to the underlying GPU platforms. For example, the Cooperative Groups API provides a grid-wide (and thus often device-wide) synchronization barrier ("grid.sync()") that can be used to prevent threads within the grid group from proceeding beyond the barrier until all threads in the defined grid group have reached that barrier. Such device-wide synchronization is based on the concept of a grid group ("grid\_group") defining a set of threads within the same grid, scheduled by software to be resident on the device and schedulable on that device in such a way that each thread in the grid group can make forward progress. Thread groups could range in size from a few threads (smaller than a warp) to a whole thread block, to all thread blocks in a grid launch, to grids spanning multiple GPUs. Newer GPU platforms such as NVIDIA Pascal and Volta GPUs enable grid-wide and multi-GPU synchronizing groups, and Volta's independent thread scheduling enables significantly more flexible selection and partitioning of thread groups at arbitrary cross-warp and sub-warp granularities.
> 
> [0028] The Cooperative Groups API provides CUDA
> 
> device code APIs for defining, partitioning, and synchroniz-
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2025.png)

## 应用在GPU的执行模型

[https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/embedded_software_components.html](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/embedded_software_components.html)

**channel、runlist、抢占设置**

**channel是TMU的硬件模块task group，包含TMD指针，应用可以分时占用多个channel（TSG）。**

**channel**抢占通过清除channel ram entry的valid后写入preempt寄存器来unload Ctx，完成后切换runlist中下一个应用占有的channel。

**runlist**抢占将新runlist写入runlist寄存器，找到有pending任务的第一个channel来load Ctx和执行。

**Timeslice Out**抢占，每个**channel**的时间片按照**Ctx中命令流stream的方法数（kernel call）定义**，dense stream的更长，sparse stream的更短。

> **[图片提取文字 (image.png)]:**
> ## Runlist Interleave Frequency
> 
> The runlist is an ordered list of channels that the GPU HOST reads to find work for the downstream engines to complete. To enable the GPU HOST to schedule a given channel more often, include the channel multiple times on a runlist. For each priority level, the runlist interleave frequency must be set to match the priority.
> 
> For example, if a system has one high-priority application, one medium-priority application, and two low-priority applications, the GPU scheduler constructs the runlist as follows:
> 
> ![](_page_0_Picture_3.jpeg)
> 
> The scheduling latency for when a high-priority application will be able to run is governed by:
> 
> worst-case latency(high) = (h-1) × timeslice(high) + execution time(low) + channel reset
> 
> Where:
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%203.png)

> **[图片提取文字 (image.png)]:**
> **Table 1.** Comparison of BERT inference latency against turn-around latency of different scheduling granularity for Whisper training on NVIDIA A100 GPU.
> 
> | Inference time | Turnaround latency (Whisper) |        |         |        |
> |----------------|------------------------------|--------|---------|--------|
> | (BERT)         | Iteration                    | Kernel | Block   | Thread |
> | 3.93ms         | ~ 3s                         | ~ 10ms | ~ 304µs | ~ 38µs |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2026.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3
![image.png](SCG-SM%EF%BC%9A%E5%8F%96%E6%B6%88%E5%9B%BE%E5%BD%A2Stream%E5%92%8C%E8%AE%A1%E7%AE%97Stream%E4%BA%A4%E6%9B%BF%E5%90%AF%E5%8A%A8%E4%BB%BB%E5%8A%A1%E7%9A%84Ctx%E5%88%87%E6%8D%A2%EF%BC%8C%E7%94%B1SM%E7%94%B3%E8%AF%B7%E4%BB%BB%E5%8A%A1%E8%B0%83%E5%BA%A6%E5%88%B0%E7%A9%BA/image%2024.png)

**主机接口**传递host抢占命令（指定channel或runlist切换）或计时器抢占命令（TS超时切换），并从新pushbuffer中读取TMD指针到runlist指定的新channel。

**前端**控制GPU的抢占过程，完成后将TMD指针发到TMU。

**TMU**对TMD指针进行任务管理、调度执行和派发资源。

任务管理：不同优先级分为不同TMD group，每个TMD group维护**多个队列，支持GPU程序的嵌套执行、同步和抢占恢复等机制**。

调度任务：按照队列内串行、队列间并发，和队列中任务内容（wait、signal、grid）定义的同步依赖和队列优先级调度任务。

派发资源：为grid task分配CTA，和CTA在grid中位置打包成**packet**，派发到有CTA余量的GPC。

SCG-SM-Sched&Arbitor：SM可能是调度任务的主控，按照优先级和资源情况决定申请计算任务或图形任务，图形任务直接调度，计算任务由TMU同意申请后调度。

**CGA机制**：按照SM-CTA/GPC-CTA/GPU-CTA粒度分配资源，并派发到SM/GPC/uGPU范围内的SM。

GPC的pipeline-manager调度CTA packet并派发CTA到有足够余量的SM（DPC）。SM对CTA按warp执行，直到CTA包含的所有warp完成。

SM是**SIMD pipeline**，执行单元包含LS指令的SIMD单元LSUs、算术指令的SIMD单元Cuda Cores、SF指令的SIMD单元SFUs和Tensor指令的SIMD单元Tensor Core。

**SIMD单元每次执行1个SIMD指令，执行1个warp的指令，使用warp内线程定义的数据。**

SIMD单元花费若干cycles完成（pipeline并行）1个SIMD指令，对应不同指令的不同延迟。

SIMD单元每次执行1个warp的1条指令，为了占满全部种类的SIMD单元（隐藏LD的长延迟），使用多个warp的多条指令来填充全部SIMD单元，因此**SM交错发射不同warp的指令**。

no OoO：1个warp的多条指令并行开销大（处理指令依赖interlock），SM通过SB机制阻塞存在前序依赖的后序指令发射，让没有依赖的后序指令发射。

LD长延迟：执行warp的LD指令会阻塞所有warp的LD指令发射，发射其他warp中的非LS指令来填充非LSU的SIMD单元。

## TS、Preemption（单实例）

[https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/embedded_software_components.html](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/embedded_software_components.html)

**channel是GPU TMD指针的集合，应用包含若干channel。**

**channel**抢占通过清除channel ram entry的valid后写入preempt寄存器来unload Ctx，完成后切换runlist中下一个channel。

**runlist**抢占将新runlist写入runlist寄存器，找到有pending任务的第一个channel来load Ctx和执行。

**Timeslice Out**抢占，每个**channel**的时间片按照**Ctx中命令流stream的方法数（kernel call）定义**，dense stream的更长，sparse stream的更短。

> **[图片提取文字 (image.png)]:**
> ## Program Execution And Preemption
> 
> Preemption may be used to time-slice a processor between multiple different applications so that the different applications are serialized and each execute for a short time-slice on the processor. Preemption may also be used to unload the currently executing context for other purposes. For example, the host interface 206 may preempt a context when the CPU 102 initiates a channel preempt or a runlist preempt, where a channel is a collection of pointers to processing work and an application may contain one or more channels. A channel preempt is performed by clearing a valid bit in a channel ram entry and writing a channel identifier of the channel to be preempted to a preempt register. The specified channel is then unloaded from the PPU 202 off both host and the engine.
> 
> A runlist preempt is performed by writing a pointer to the runlist register. The pointer may point to a new runlist or may point to the runlist that is currently active. Runlist preempt causes what is running in a PPU 202 to be unloaded. The host interface 206 then begins processing at the first entry on the runlist associated with the pointer and searches for the first valid entry with pending work. The first channel on the runlist which has pending work is loaded into the PPU 202
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%204.png)

> **[图片提取文字 (image.png)]:**
> ## Runlist Interleave Frequency
> 
> The runlist is an ordered list of channels that the GPU HOST reads to find work for the downstream engines to complete. To enable the GPU HOST to schedule a given channel more often, include the channel multiple times on a runlist. For each priority level, the runlist interleave frequency must be set to match the priority.
> 
> For example, if a system has one high-priority application, one medium-priority application, and two low-priority applications, the GPU scheduler constructs the runlist as follows:
> 
> ![](_page_0_Picture_3.jpeg)
> 
> The scheduling latency for when a high-priority application will be able to run is governed by:
> 
> worst-case latency(high) = (h-1) × timeslice(high) + execution time(low) + channel reset
> 
> Where:
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%203.png)

> **[图片提取文字 (image.png)]:**
> **Table 1.** Comparison of BERT inference latency against turn-around latency of different scheduling granularity for Whisper training on NVIDIA A100 GPU.
> 
> | Inference time | Turnaround latency (Whisper) |        |         |        |
> |----------------|------------------------------|--------|---------|--------|
> | (BERT)         | Iteration                    | Kernel | Block   | Thread |
> | 3.93ms         | ~ 3s                         | ~ 10ms | ~ 304µs | ~ 38µs |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2026.png)

> **[图片提取文字 (image.png)]:**
> The host interface 206 may also preempt a context that is executing before a time slice has expired when the context is out of methods (i.e. programs) and another context is waiting to execute. In one embodiment, the time slices are not equal amounts of time, but instead are based on each context's method stream, so that a context with a dense method stream is allocated a larger time slice compared with a different context having a sparse method stream. The host interface 206 is configured to indicate to the front end 212 when the host interface 206 does not have any methods for an executing context. However, the host interface 206 does not initiate a context switch for the executing context until either the time slice allocated to the context has expired or the processing pipeline is idle and there are no methods.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%205.png)

应用**Preemption**类型包括WFI、CLIP、GFXP，决定应用channel在时间片内是否被抢占。

（短时）高优先级任务设置**WFI（wait-for-idle）**，保证**时间片内不被抢占**，**同时设置较长的时间片，保证时间片内kernel完成**。

**辨析wait-for-idle**指令： **wait-for-idle**是应用卸载到GPU任务的指令流里的一种**常规同步指令**。要求让前端暂停调度该进程任务的后续任务，直到资源空闲。**wait-for-idle**常用于低优先级任务，让任务执行到wait-for-idle后**主动“休息”**，**不和高优先级任务争抢资源**。

中低优先级设置**CLIP或GFXP**，**时间片内允许指令级抢占**，设置较短时间片来及时释放资源。

高优先级任务的时间片TS设置不能超过最差情况下的调度延迟，因为TS超过阈值还没有完成会导致违背帧率的计算速度要求（16.6ms/channel）。

最差情况考虑：上一个高优先级channel完成后，低优先级抢先执行，但时间片到达后（lpt）没有正常切出，经过切换超时检测后（cst），强制清空channel后（crt），调度下一个高优先级channel。强制清空channel不会保存channel ctx。

图中没有显示H1、H2、H3任务的Ctx保存延迟，因为保证在TS内完成，不需要保存。

时间片开头是Ctx加载延迟，而时间片不会包含Ctx保存延迟。

> **[图片提取文字 (image.png)]:**
> ## Runlist Interleave Frequency
> 
> The runlist is an ordered list of channels that the GPU HOST reads to find work for the downstream engines to complete. To enable the GPU HOST to schedule a given channel more often, include the channel multiple times on a runlist. For each priority level, the runlist interleave frequency must be set to match the priority.
> 
> For example, if a system has one high-priority application, one medium-priority application, and two low-priority applications, the GPU scheduler constructs the runlist as follows:
> 
> ![](_page_0_Picture_3.jpeg)
> 
> The scheduling latency for when a high-priority application will be able to run is governed by:
> 
> worst-case latency(high) = (h-1) × timeslice(high) + execution time(low) + channel reset
> 
> Where:
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## ∨ High-Priority Applications ∂
> 
> For high priority applications, set the timeslice large enough so that all work can be completed within one timeslice.
> 
> The recommended upper bound for timeslice in single, high-priority applications is:
> 
> 16.6 ms - lpt - cst - crt = 11.6 ms
> 
> Where:
> 
> - *lpt* is the low-priority timeslice, set to 1.5 ms
> - cst is the context-switch timeout, equal to 2.0 ms
> - crt is the channel reset time, equal to 1.5 ms
> 
> For multiple, high-priority applications, use the timeslice for each high-priority application to determine a reasonable bound. The recommended combined workload of all high-priority applications must not exceed 50% of a display refresh cycle.
> 
> High-priority applications must avoid flushing work prematurely, whether by calling glFlush or glFinish or by other means. This ensures all rendering for a frame completes without any context switches.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## ✓ Medium-Priority Applications
> 
> For medium-priority applications, set the timeslice both:
> 
> - · Large enough that an application can make progress, but
> - Not so large that it affects the scheduling latency of high-priority applications.
> 
> The recommended upper bound for timeslice in medium-priority applications is 2 ms.
> 
> ## ∨ Low-Priority Applications ∂
> 
> For low-priority applications, set the timeslice both:
> 
> - · Large enough that an application can make progress, but
> - Not so large that it affects the scheduling latency of high- or medium-priority applications.
> 
> The recommended upper bound for timeslice in low-priority applications is 1.5 milliseconds (ms).
> 
> ## ∨ Reserve Time for Lower-Priority Applications ∂
> 
> To ensure lower-priority applications make reasonable progress, you must ensure that high- and medium-priority applications do not use 100% of the GPU by:
> 
> - Lowering your application frame rate targets and/or
> - Reducing complexity of rendered frames.
> 
> The proportion of time to reserve for low-priority applications depends on the number and nature of the applications.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2030.png)

> **[图片提取文字 (Screenshot from 2026-03-12 16-04-32.png)]:**
> ## Setting Parameters on Behalf of Other Applications
> 
> A privileged application can set scheduling parameters (timeslice and interleave) on behalf of other applications based on their PID.
> 
> The <a href="libnvrm\_gpusched">libnvrm\_gpusched</a> library provides a way to:
> 
> - Get a list of all TSGs
> - Get a list of recent TSGs (i.e., a list of TSGs opened since the last query)
> - Get a list of TSGs opened by a given process
> - Get notifications when a TSG is allocated
> - Get current scheduling parameters for a TSG
> - Set runlist interleave for a TSG
> - Set timeslice for a TSG
> - Lock control (i.e., prevent other applications from changing their own scheduling parameters such as timeslice and interleave)
> 
> Library API is defined in <a href="mailto:nvrm\_gpusched.h">nvrm\_gpusched.h</a> and sample code implements command line to control GPU scheduling parameters.
![Screenshot from 2026-03-12 16-04-32.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/Screenshot_from_2026-03-12_16-04-32.png)

> **[图片提取文字 (image.png)]:**
> ## Setting the Preemption Type
> 
> ## → High-Priority Applications ②
> 
> For high-priority applications, set the timeslice large enough that all work can complete. The recommended Compute-Instruction-Level-Preemption (CILP) setting for graphics and for compute is a preemption type of Wait-For-Idle (WFI). This ensures CILP will not be hit because NVIDIA® CUDA® kernels will have completed.
> 
> ## Medium-Priority Applications @
> 
> The recommended setting for medium-priority applications is a preemption type enabled for graphics (GFXP) and compute (CILP). For applications that can complete in their timeslice, context-switch overhead is minimal because the GPU is in an idled state.
> 
> ## ∨ Low-Priority Applications ∂
> 
> For low-priority applications, always enable graphics and compute preemption because workloads are unpredictable.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2031.png)

> **[图片提取文字 (Screenshot from 2026-03-12 16-03-33.png)]:**
> ## **Setting Parameters**
> 
> Use the following guidelines when setting scheduling parameters.
> 
> - Identify the relevant use cases before setting scheduling parameters.
> - Associate a priority with each application.
> - Determine appropriate timeslices.
> 
> Then using the environment variables listed below, specify the runlist interleave frequency, timeslice, and preemption type.
> 
> | Environment Variable                  | Role                                       | Values                                                       |
> |---------------------------------------|--------------------------------------------|--------------------------------------------------------------|
> | NVRM_GPU_CHANNEL_ INTERLEAVE          | Sets runlist<br>frequency for a<br>context | 1: LOW 2: MEDIUM 3: HIGH                                     |
> | NVRM_GPU_CHANNEL_ TIMESLICE           | Sets timeslice for a context               | Non-zero value in<br>microseconds (minimum<br>1000 for 1 ms) |
> | NVRM_GPU_NVGPU_FORCE_ GFX_ PREEMPTION | Enables GFXP                               | 0: off<br>1: on                                              |
> 
> ![](_page_0_Picture_7.jpeg)
> 
> The environment variables apply to  $\emph{all}$  contexts belonging to a process.
![Screenshot from 2026-03-12 16-03-33.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/Screenshot_from_2026-03-12_16-03-33.png)

**旧channel到达timeslice，新channel抢占GPU过程**

1、GPU驱动提前在显存VRAM分配**Ctx buffer**空间。

2、主机接口从Runlist中选择新channel，**抢占**旧channel的Ctx占据的资源，**前端和各层trap handler控制**抢占过程。

3、**GPU停止**context处理、**保存**context到内存、**重置**Engine/SM、**加载**上下文、**恢复**任务启动。

4、GPU完成抢占后，前端向主机接口发送**ACK**，切换新channel执行。

**抢占的原理**

暂停不同深度层次上（TMU和GPC、SM）上的任务调度和资源分配（CTA或warp）。

等待已分配资源被释放，即TMU和GPC已分配CTA或SM已发射指令完成后释放资源。

“暂停-等待空闲”是因为**GPU高并发，不能追踪和控制特定任务，以资源为handler**，等待资源上任务完成（wait-based/drain），或者放弃任务（flush-based）。

资源空闲后，保存旧Ctx，加载新Ctx并且开始执行。

**硬件控制**：恢复时，硬件必须将之前被抢占的CTA精确发射回**原来执行它的同一个SM内的同一个物理CTA槽位（physical CTA slot）**，甚至warp也要使用相同的物理warp ID，保证资源不超限。

**软件辅助**：抢占时，将被抢占CTA包装成最高优先级的**抢占TMD**。恢复时，这个TMD作为优先级极高的任务被前端调度器调度，随后执行`preemption-restore kernel`来恢复上下文。解除CTA、warp对底层资源的强绑定。

软件辅助指令级抢占的设计初衷，是为了在**复用CTA级抢占的硬件控制流**的基础上，通过软件（子任务生成和exit指令）跳过耗时的排空阶段，**不需要增加复杂的纯硬件指令级卸载逻辑**。

## Hyper-Q（Stream并发优化）

[https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/simpleHyperQ/doc/HyperQ.pdf](https://developer.download.nvidia.com/compute/DevZone/C/html_x64/6_Advanced/simpleHyperQ/doc/HyperQ.pdf)

hyper-Q：在WDU之前**增加GMU（即WMU），**GMU设置32个**硬件queue是task group（链表），也称为channel**，**移除单个channel管理多个cudaStream的False Dependence**。

多个cudaStream交错整合后进入一个task group（channel），WDU从单一channel中读取grid TMD并派发CTA，则不同Stream之间的并发存在隐式Dependence（**Stream A和Stream B共享task group**），**GMU**让接受不同Stream并行发射。

> **[图片提取文字 (image.png)]:**
> ## Background
> 
> Hyper-Q enables multiple CPU threads or processes to launch work on a single GPU simultaneously, thereby dramatically increasing GPU utilization and slashing CPU idle times. This feature increases the total number of "connections" between the host and GPU by allowing 32 simultaneous, hardware-managed connections, compared to the single connection available with GPUs without Hyper-Q (e.g. Fermi GPUs).
> 
> Hyper-Q is a flexible solution that allows connections for both CUDA streams and Message Passing Interface (MPI) processes, or even threads from within a process. Existing applications that were previously limited by false dependencies can see a dramatic performance increase without changing any existing code.
> 
> ## False Dependencies before Kepler
> 
> On Fermi, when a CPU thread dispatched work into a CUDA stream, the work was joined into a single pipeline to the Work Distributor. The Work Distributor takes work from the front of the pipeline, checks all dependencies are satisfied, and farms the work to the available SMs.
> 
> Consider three CUDA streams, each containing a sequence of kernels A, B, C as shown in the adjacent figure.
> 
> ```
> for (int i = 0; i < 3; i++)
> {
>      A<<<gdim,bdim,smem,streams[i]>>>();
>      B<<<gdim,bdim,smem,streams[i]>>>();
>      C<<<gdim,bdim,smem,streams[i]>>>();
> }
> ```
> 
> ![](_page_0_Figure_7.jpeg)
> 
> Using CUDA streams, we have declared the dependency chains  $A_0$ - $B_0$ - $C_0$  and  $A_1$ - $B_1$ - $C_1$  and  $A_2$ - $B_2$ - $C_2$ . Each of these chains is independent and therefore they could be executed at the same time (i.e. concurrently). With Fermi's single pipeline, however, this depth-first launch sequence will result in false dependencies:
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ## HYPER QUEUE
> 
> ## **Behind MPS**
> 
> Hyper-Q is introduced since Kepler GPU.
> 
> To enable multiple CPU threads or processes to launch work on a single GPU simultaneously.
> 
> Supported connection types:
> 
> Multiple CUDA streams;
> 
> Multiple CPU threads;
> 
> Multiple CPU processes;
> 
> ![](_page_0_Figure_8.jpeg)
> 
> ## Hyper-Q whitepaper:
> 
> https://developer.download.nvidia.com/compute/DevZone/C/html\_x64/6\_Advanced/simpleHyperQ/doc/HyperQ.pdf
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> As a result the hardware is only able to determine that it can execute the shaded pairs concurrently.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ## Grid Management Unit
> 
> Kepler GK110 introduces the Grid Management Unit, which creates multiple hardware work queues to reduce or eliminate false dependencies. With the GMU, streams can be kept as individual pipelines of work.
> 
> Also shown on the diagram is the feedback path from the SMXs to the Work Distributor, and the work creation path from the SMXs to the GMU. These components provide dynamic parallelism (see the CUDA Programming Guide for more information on dynamic parallelism).
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## HYPER QUEUE
> 
> ## Example: \$CUDA\_PATH/samples/6\_Advanced/simpleHyperQ
> 
> ```
> for (int i = 0; i < nstreams; ++i)
> {
> kernel_A<<<1,1,0,streams[i]>>>(&d_a[2*i], time_clocks);
> total_clocks += time_clocks;
> kernel_B<<<1,1,0,streams[i]>>>(&d_a[2*i+1], time_clocks);
> total_clocks += time_clocks;
> }
> ```
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2036.png)

## 多进程服务MPS（共享相同App Ctx，不同task group/channel）

**MPS：**架构基于Hyper-Q（设置多个work queue/channel）改进，由CPU的MPS进程将相同VM的不同请求打包成一个应用的若干命令流，发给GPU中WMU的不同work queue管理。

GPU视角中，只有一个应用的多个命令流在执行，因此**隔离性差**带来性能干扰，所有任务响应针对应用的抢占或异常处理。

> **[图片提取文字 (image.png)]:**
> ## MULTI-PROCESS SERVICE
> 
> ## What's MPS
> 
> An alternative, binary-compatible implementation of the CUDA Application Programming Interface (API).
> 
> ## Based on GPU Hyper-Q capability
> 
> - Enabling multiple CPU processes sharing one GPU context;
> - Allowing kernels and memcpy in different processes can be executed simultaneously on the same GPU, to utilize GPU better;
> 
> ## MPS includes
> 
> - Control Daemon Process The control daemon is responsible for starting and stopping the server, as well as coordinating connections between clients and servers.
> - Server Process The server is the clients' shared connection to the GPU and provides concurrency between clients.
> - Client Runtime The MPS client runtime is built into the CUDA Driver library and may be used transparently by any CUDA application.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ## **MULTI-PROCESS SERVICE**
> 
> ## Without MPS VS With MPS
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Without MPS With MPS
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2038.png)

**CPU进程cuda context server打包和混合同一个用户（VM）的user**的不同请求client，但阻塞其他用户的请求，用户user是OS level分类，请求client是进程level的分类；

> **[图片提取文字 (image.png)]:**
> ## **MULTI-PROCESS SERVICE**
> 
> ## **MPS Architecture**
> 
> ## System-wide provisioning with multiple users.
> 
> - Client A from User 1 request;
> - Daemon create MPS server for User 1 and Client A runs;
> - Client B from User 1 request and assigned to MPS server, and to run;
> - Client C from User 2 request, and pending;
> - Util all clients from User 1 running end and MPS server exit for User 1, Daemon create MPS server for User 2, and Client C begin to run;
> 
> ![](_page_0_Figure_8.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ## **MPS** Benefits
> 
> ## **GPU Utilization**
> 
> A single process may not utilize all the compute and memory-bandwidth capacity available on the GPU. MPS allows kernel and memcopy operations from different processes to overlap on the GPU, achieving higher utilization and shorter running times.
> 
> ## Reduced on-GPU Context Storage
> 
> The MPS server allocates one copy of GPU storage and scheduling resources shared by all its clients, thus reduces the resource storage.
> 
> ## Reduced on-GPU Context Switching
> 
> The MPS server shares one set of scheduling resources between all of its clients, eliminating the overhead of swapping when the GPU is scheduling between those clients.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2040.png)

**对进程数/线程数不足的LC应用，MPS不足以充分利用GPU资源；**

**Volta架构的MPS中每个进程的地址空间独立，而非共享全体地址空间；**

> **[图片提取文字 (image.png)]:**
> ## **MULTI-PROCESS SERVICE**
> 
> ## Potential Applications for MPS
> 
> - Application process does not generate enough work to saturate the GPU. Applications like this are identified by having a small number of blocks-per-grid.
> - Application shows a low GPU occupancy because of a small number of threads-per-grid.
> - In strong-scaling case, some MPI processes may underutilize the available compute capacity.
> - Especially for AI inference, with critical latency limitation, which not allowed batching for inference.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ## **MULTI-PROCESS SERVICE**
> 
> ## Volta MPS
> 
> Volta MPS provides a few key improvements, compared with pre-Volta:
> 
> - Volta MPS clients submit work directly to the GPU without passing through the MPS server.
> - Each Volta MPS client owns its own GPU address space instead of sharing GPU address space with all other MPS clients.
> - Volta MPS supports limited execution resource provisioning for Quality of Service (QoS).
> 
> ![](_page_0_Figure_6.jpeg)
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2042.png)

MPS使用

> **[图片提取文字 (image.png)]:**
> ## **MULTI-PROCESS SERVICE**
> 
> MPS Usage
> 
> Start MPS daemon process
> 
> nvidia-cuda-mps-control -d
> 
> Check MPS process
> 
> ps -ef | grep mps
> 
> Recommend to set compute mode to exclusive
> 
> sudo nvidia-smi -c EXCLUSIVE\_PROCESS
> 
> Quit MPS daemon
> 
> echo quit | nvidia-cuda-mps-control
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ## **MULTI-PROCESS SERVICE**
> 
> ## MPS Usage
> 
> nvidia-smi shows when running eight trtexec processes with MPS:
> 
> | N |                       |                            |                                      |                       | +-                                                   |           |          |     |          |                                               |                                      |
> |---|-----------------------|----------------------------|--------------------------------------|-----------------------|------------------------------------------------------|-----------|----------|-----|----------|-----------------------------------------------|--------------------------------------|
> | G | PU                    | Name                       |                                      | Persis                | tence-M                                              | Bus-Id    | Disp     | p.A | Volatile | Uncorr.                                       | ECC                                  |
> |   |                       | -                          |                                      |                       | -                                                    | M         | -        | -   |          | -                                             |                                      |
> |   |                       |                            |                                      |                       |                                                      | 00000000: |          |     |          |                                               | Off                                  |
> | N | /A                    | 46C                        | P0                                   | 140W                  | / 300W                                               | 7027MiB   | / 161601 | MiB | 100%     | Defa                                          | ault                                 |
> |   |                       | esses:                     | PID                                  | Type                  | Process                                              | name      |          |     |          | GPU Men<br>Usage                              | _                                    |
> |   | GPU                   |                            |                                      | ======                | ======                                               |           |          |     |          | Usage                                         |                                      |
> |   | GPU                   | 81                         | <br>1016                             | C                     | nvidia-                                              |           |          |     |          | Usage<br>==================================== | ====<br>9MiB                         |
> |   | GPU<br>               | 81                         | <br>1016<br>1074                     | с<br>с                | ======                                               |           |          |     |          | Usage<br>=======<br>29<br>873                 |                                      |
> |   | GPU<br>0<br>0         | 81<br>81                   | <br>1016<br>1074<br>1075             | с<br>с<br>с           | nvidia-c                                             |           |          |     | ======   | Usage<br>=======<br>29<br>873<br>873          | ====<br>9MiB<br>3MiB                 |
> |   | 0<br>0<br>0           | 81<br>81<br>81             | <br>1016<br>1074<br>1075             | с<br>с<br>с           | nvidia-c<br>trtexec<br>trtexec                       |           |          |     |          | Usage<br>29<br>873<br>873                     | ====<br>9MiB<br>3MiB<br>3MiB         |
> |   | 0<br>0<br>0<br>0      | 81<br>81<br>81<br>81<br>81 | 1016<br>1074<br>1075<br>1076         | с<br>с<br>с<br>с      | nvidia-o<br>trtexec<br>trtexec<br>trtexec            |           |          |     |          | Usage<br>29<br>873<br>873<br>873              | 9MiB<br>3MiB<br>3MiB<br>3MiB         |
> |   | 0<br>0<br>0<br>0<br>0 | 81<br>81<br>81<br>81<br>81 | 1016<br>1074<br>1075<br>1076<br>1077 | C<br>C<br>C<br>C<br>C | nvidia-c<br>trtexec<br>trtexec<br>trtexec<br>trtexec |           |          |     |          | Usage<br>29<br>873<br>873<br>873<br>873       | 9MiB<br>3MiB<br>3MiB<br>3MiB<br>3MiB |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2044.png)

**Test Case 1**

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 1
> 
> ## Simple Kernel with One Thread Running
> 
> Simple kernel code: (Ignore the computing content)
> 
> ```
> __global__ void testMaxFlopsKernel(float * pData, int nRepeats, float v1, float v2)
> {\nint tid = blockIdx.x* blockDim.x+ threadIdx.x;
> float s = pData[tid], s2 = 10.0f - s, s3 = 9.0f - s, s4 = 9.0f - s2;
> for(int i = 0; i < nRepeats; i++)
> {
> s=v1-s*v2;
> }
> pData[tid] = ((s+s2)+(s3+s4));
> }
> ```
> 
> To test: run four processes with and without MPS
> 
> To profile: profiling analysis the running characteristic
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 1
> 
> ## **Test Results**
> 
> Run multiple processes with mpirun, command like: mpirun -np \$NP ./testMPS
> 
> | Category | Average Wall Clock Time |             |             |
> |----------|-------------------------|-------------|-------------|
> |          | 1 Process               | 2 Processes | 4 Processes |
> | MPS OFF  | 2924 ms                 | 6013 ms     | 12002 ms    |
> | MPS ON   | 2924 ms                 | 2924 ms     | 2924 ms     |
> 
> Without MPS, the kernel running time increases linearly along with the number of processes.
> 
> With MPS, the kernel run time of multi processes is almost the same as one process.
> 
> This is the extreme case, but it's the best case to show MPS benefit.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 1
> 
> ## **Profiling Analysis**
> 
> ## Use nvprof to capture trace:
> 
> ```
> node1:~$ nvprof -o ./profile-test2-%p --profile-child-processes mpirun -np 2 ./testMPS
> ==56763== NVPROF is profiling process 56763, command: ./testMPS
> ==56768== NVPROF is profiling process 56768, command: ./testMPS
> Rank0: BlockSize(1, 1, 1), GirdSize(1, 1, 1)
> Ranko: Iteration: 1, Total Elapsed Time: 2918.924ms, Single kernel cost time: 2918.924ms
> Rank0: Performance: 0.685GFLOPS
> Rank1: BlockSize(1, 1, 1), GirdSize(1, 1, 1)
> Rank1: Iteration: 1, Total Elapsed Time: 2917.827ms, Single kernel cost time: 2917.827ms
> Rank1: Performance: 0.685GFLOPS
> ==56768== Generated result file: /home/dgx/src/testMPS/profile-test2-56768
> ==56763== Generated result file: /home/dgx/src/testMPS/profile-test2-56763
> ```
> 
> Then import into NVVP profiler tool for visual profiling analysis.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 1
> 
> ## Profiling Analysis: With MPS
> 
> With MPS, four processes.
> 
> Only one CUDA context to run these four processes.
> 
> The kernels from different processes are really running overlapped.
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2048.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 1
> 
> ## Profiling Analysis: Without MPS
> 
> Without MPS, four processes.
> 
> Four CUDA contexts on a V100 GPU.
> 
> Although it seems like that they are running concurrently, the execution time for each kernel is lengthened.
> 
> That is because that they are running under the GPU time slice rotation scheduling mechanism. These CUDA contexts need to be switched in each time slice which introduces extra time overhead.
> 
> ![](_page_0_Figure_6.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2049.png)

**Test Case 2**

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 2
> 
> ## ResNet-50 Inference in 7ms Budget
> 
> - This example is to run ResNet-50 inference with TensorRT engine.
> - We use NGC container "nvcr.io/nvidia/tensorrt:19.07-py3" on SXM2 V100 16GB.
> - We run and compare several scenarios in 7ms inference time budget:
> - Batching in single process;
> - No batching(batch size is 1) in multiple processes, without MPS;
> - No batching(batch size is 1) in multiple processes, with MPS;
> - Batching and multiple processes combination;
> 
> At the same time, we capture some utilization metrics with dcgmi, to quantify GPU usage.
> 
> dcgmi dmon -e 1001,1002,1004,1005,1009,1010,1011,1012
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2050.png)

> **[图片提取文字 (image.png)]:**
> # MPS TEST CASE 2
> 
> # Steps to Test
> 
> #### Start container
> 
> nvidia-docker run -it --name click-trt --privileged -v /home/click/models/:/click nvcr.io/nvidia/tensorrt:19.07-py3 bash
> 
> ### Build out ResNet-50 TRT engine (using caffemodel here)
> 
> ## Example, for batch size 1, 32, ...
> 
> trtexec --batch=1 --iterations=100 --workspace=1024 --deploy=/click/ResNet-50-deploy.prototxt --model=/click/ResNet-50-model.caffemodel --output=prob --fp16 --saveEngine=/workspace/rn50-bs1.engine
> 
> trtexec --batch=32 --iterations=100 --workspace=1024 --deploy=/click/ResNet-50-deploy.prototxt --model=/click/ResNet-50-model.caffemodel --output=prob --fp16 --saveEngine=/workspace/rn50-bs32.engine
> 
> #### Test in single process
> 
> trtexec --loadEngine=/workspace/rn50-bs1.engine --iterations=1000 --workspace=1024 --fp16
> 
> trtexec --loadEngine=/workspace/rn50-bs32.engine --iterations=10000 --workspace=1024 --fp16 --batch=32
> 
> ### Test in multi processes with MPI
> 
> mpirun -np 8 --allow-run-as-root trtexec --loadEngine=/workspace/rn50-bs1.engine --iterations=1000 --workspace=1024 --fp16 > trt-mps-mpi-8.log
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 2
> 
> ## **Test Results**
> 
> Batching is the recommended way to reach best throughput.
> 
> Without batching, i.e. BS=1 cases, MPS can bring ~3X throughput.
> 
> Batching and MPS can be combined, to improve throughput to some extent.
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2052.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 2
> 
> ## **GPU Utilization Metrics - MPS OFF**
> 
> GPU Utilization Metrics - Without Batching, Without MPS
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 2
> 
> ## **Profiling Analysis**
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> BS=1, NP=8, MPS OFF
> 
> BS=1, NP=8, MPS ON
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2054.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 2
> 
> ## **GPU Utilization Metrics - MPS ON**
> 
> GPU Utilization Metrics - Without Batching, With MPS
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2055.png)

**Test Case 3**

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 3
> 
> ## JPEG Resize
> 
> JPEG to JPEG resizing is an essential workload for many internet services, including training and inference for image classification, object detection, etc.
> 
> And for some service provider, to cut storage expense, they might just storage one image instead of several dozens in different resolutions.
> 
> <u>Fastvideo</u>, an NVIDIA Preferred Partner, developed an image processing SDK with CUDA acceleration (one of their customer was Flickr), since there're multi phases in the whole JPEG resize implementation pipeline, like copy from storage to CPU memory, then copy to GPU memory, JPEG decoding, resizing, sharp, JPEG encoding, copy to CPU memory, etc. They've done many optimizations across the whole pipeline, and one technical they adopted is NVIDIA MPS, to optimize the throughput of the GPU system.
> 
> We use Fastvideo SDK to perform this testing.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2056.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 3
> 
> ## Test Results
> 
> Resize JPEG from 1920x1080 to 480x270.
> 
> Up to 3.5x throughput improvement when MPS enabled.
> 
> | Processes Number | FPS - MPS OFF | FPS - MPS ON | Speedup |
> |------------------|---------------|--------------|---------|
> | 2                | 1152          | 1633         | 1.42    |
> | 4                | 1025          | 2319         | 2.26    |
> | 6                | 1016          | 2786         | 2.74    |
> | 8                | 1014          | 3024         | 2.98    |
> | 10               | 1011          | 3190         | 3.15    |
> | 12               | 1014          | 3301         | 3.25    |
> | 14               | 1154          | 3367         | 2.92    |
> | 16               | 1012          | 3458         | 3.42    |
> | 18               | 1009          | 3558         | 3.53    |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2057.png)

> **[图片提取文字 (image.png)]:**
> ## MPS TEST CASE 3
> 
> ## Test Results
> 
> Resize JPEG from 1280x720 to 320x180.
> 
> Up to 4.4x throughput improvement when MPS enabled.
> 
> | Processes Number | FPS - MPS OFF | FPS - MPS ON | Speedup |
> |------------------|---------------|--------------|---------|
> | 2                | 937           | 2007         | 2.14    |
> | 4                | 904           | 2910         | 3.22    |
> | 6                | 897           | 3451         | 3.85    |
> | 8                | 894           | 3813         | 4.26    |
> | 10               | 890           | 3848         | 4.32    |
> | 12               | 891           | 3878         | 4.35    |
> | 14               | 900           | 3860         | 4.29    |
> | 16               | 889           | 3921         | 4.41    |
> | 18               | 886           | 3942         | 4.45    |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2058.png)

## 多实例GPU MIG（空间划分、物理隔离）

> **[图片提取文字 (image.png)]:**
> ## GPU ARCHITECTURE AND CUDA
> 
> CUDA 8.0 CUDA 9.0 CUDA 10.0 CUDA 11.0
> 
> ![](_page_0_Picture_2.jpeg)
> 
> 2016
> 
> **PASCAL** 
> 
> ![](_page_0_Picture_5.jpeg)
> 
> 2017
> 
> **VOLTA** 
> 
> ![](_page_0_Picture_8.jpeg)
> 
> 2018
> 
> **TURING** 
> 
> ![](_page_0_Picture_11.jpeg)
> 
> 2020
> 
> **AMPERE** 
> 
> HBM, NVLINK, FP16
> 
> HBM, NVLINK, TENSOR CORES, MPS
> 
> TENSOR CORES, RT CORES
> 
> HBM, NVLINK, TENSOR CORES, PARTITIONING
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2059.png)

> **[图片提取文字 (image.png)]:**
> ## **A100 GPU**
> 
> ## Highest Performance, Efficiency and Utilization
> 
> ![](_page_0_Figure_2.jpeg)
> 
> | New Technology                                        | Benefit over Volta                                                                              |
> |-------------------------------------------------------|-------------------------------------------------------------------------------------------------|
> | Faster Tensor Core for AI, support FP16 & bfloat16    | >2x V100 RN50 & Transformer train<br>~3x Tensor Core FLOPS<br>Dramatically reduce time-to-soln. |
> | New Tensor Core for HPC                               | 2.5x FP64 FLOPS Accelerate core HPC kernels                                                     |
> | Wider + Faster Memory                                 | 1.7x memory bandwidth Up to 40GB per GPU Larger model & dataset                                 |
> | New NVLINK3 + PCIe Gen4                               | 2x NVLINK bandwidth<br>2x PCle bandwidth + SR-IOV                                               |
> | New Multi-Instance GPU, with Fault and Perf Isolation | Up to 7 concurrent GPUs Higher utilization Substantially lower entry cost                       |
> | New Hardware Engines                                  | JPEG HW decoder, 5 video NVDEC<br>Optical flow accelerator                                      |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2060.png)

**Mult-Instance GPU**将性能超强的GPU单卡“拆分”成多个“小”GPU实例，面向多用户应用，提高资源使用率；

MIG的多个子GPU运行时所用资源是**“硬件隔离”**；MPS的进程间通过资源分配进行隔离，但部分硬件资源共享；

> **[图片提取文字 (image.png)]:**
> ## NEW MULTI-INSTANCE GPU (MIG)
> 
> Optimize GPU Utilization, Expand Access to More Users with Guaranteed Quality of Service
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2061.png)

> **[图片提取文字 (image.png)]:**
> # MIG ISOLATION
> 
> ## Computational Isolation
> 
> - SM are not shared between MIGs
> - This provides high QoS for each MIG users
> 
> #### **DRAM Bandwidth Isolation**
> 
> - Slices of the L2 cache are physically associated with particular DRAM channels and memory
> - Isolating MIGs to non-overlapping sets of L2 cache slices does two things:
>   - Isolates BW
>   - Allocates DRAM memory between the MIGs
> 
> ## **Configuration Isolation**
> 
> Creating GPU Instances or Compute Instances do not disturb work running on existing instances
> 
> ### **Error Isolation**
> 
> Resources within the chip are separately resettable
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2062.png)

**MIG的子GPU配置和用途**

> **[图片提取文字 (image.png)]:**
> ## GPU INSTANCE PROFILES
> 
> For A100-SXM4-40GB
> 
> | GPU<br>Instance | Number of<br>Instances<br>Available | SMs | Memory | NVDECs | Target use-cases                                                       |                                                                    |
> |-----------------|-------------------------------------|-----|--------|--------|------------------------------------------------------------------------|--------------------------------------------------------------------|
> |                 |                                     |     |        |        | Training                                                               | Inference                                                          |
> | 1g.5gb          | 7                                   | 14  | 5 GB   | 0      | BERT Fine-tuning (e.g. SQuAD),<br>Multiple chatbots, Jupyter notebooks |                                                                    |
> | 2g.10gb         | 3                                   | 28  | 10 GB  | 1      |                                                                        | Multiple inference (e.g. TRITON);<br>ResNet-50, BERT, WnD networks |
> | 3g.20gb         | 2                                   | 42  | 20 GB  | 2      | Training on ResNet-50, BERT, WnD<br>networks                           |                                                                    |
> | 4g.20gb         | 1                                   | 56  | 20 GB  | 2      |                                                                        |                                                                    |
> | 7g.40gb         | 1                                   | 98  | 40 GB  | 5      |                                                                        |                                                                    |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ## FLEXIBLE MIG CONFIGURATIONS FOR DIFFERENT SCENARIOS
> 
> | Slice #1 | Slice #2 | Slice #3 | Slice #4 | Slice #5 | Slice #6 | Slice #7 |
> |----------|----------|----------|----------|----------|----------|----------|
> | 7        |          |          |          |          |          |          |
> | 4        |          |          |          |          | 2        | 1        |
> | 4        |          |          |          | 1        | 1        | 1        |
> | 2        |          | 2        |          | 3        |          |          |
> | 2        |          | 1        | 1        | 3        |          |          |
> | 1        | 1        | 2        | 2        | 3        |          |          |
> | 1        | 1        | 1        | 1        | 3        |          |          |
> | 3        |          |          |          | 3        |          |          |
> | 3        |          |          |          | 2 1      |          |          |
> | 3        |          |          |          | 1        | 1        | 1        |
> | 2        |          |          | 2        | 2 1      |          |          |
> | 2        |          | 2        |          | 1        | 1        | 1        |
> | 1        | 1        | 2        |          | 2 1      |          |          |
> | 1        | 1        | 2        |          | 1        | 1        | 1        |
> | 2        |          | 1        | 1        | 7        | 2        | 1        |
> | 2        |          | 1        | 1        | 1        | 1        | 1        |
> | 1        | 1        | 1        | 1        | 2 1      |          | 1        |
> | 1        | 1        | 1        | 1        | 1        | 1        | 1        |
> 
> - 18 possible configurations
> - NVML or NVIDIA-SMI to create and retire Instance
> - Config. can be dynamically updated when the GPU slices involved are idle
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2064.png)

**内存和计算的双重划分，支持容器部署。软件栈**

> **[图片提取文字 (image.png)]:**
> ## **EXAMPLE: TWO LEVEL PARTITIONING**
> 
> ## **GPU Instances and Compute Instances**
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2065.png)

> **[图片提取文字 (image.png)]:**
> ## ENABLEMENT ACROSS SOFTWARE STACK
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> - Support for bare-metal and containerized environments
>   - Interaction directly via NVML/nvidia-smi
>   - Kubernetes (device enumeration, resource type), Slurm
>   - Docker CLI
> - Monitoring and management (including device metrics association to MIG)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2066.png)

**用户工作流**

> **[图片提取文字 (image.png)]:**
> ## **USER WORKFLOW: MIG MANAGEMENT**
> 
> List/Create/Update/Destroy Instances via NVML and nvidia-smi
> 
> GPU reset required to enable/disable MIG mode (one-time operation)
> 
> Use NVML/nvidia-smi (even through containers) to manage MIG
> 
> Example: Create new instance with nvidia-smi
> 
> ![](_page_0_Figure_5.jpeg)
> 
> | nvid       | ia-smi mig<br>     | list-gp | u-instances               | 5<br>           |
> |------------|--------------------|---------|---------------------------|-----------------|
> | GPU<br>GPU | instances:<br>Name | Profile | Instance                  | <br>  Placement |
> |            |                    | ID      | Instance<br>ID<br>======= | Start:Size      |
> | 0<br>      | <br>1g.5gb<br>     | 19      | 9<br>                     | 2:1             |
> | 0          | 1g.5gb             | 19      | 10                        | 3:1             |
> | 0<br>0     | 1g.5gb             | 19      | 13                        | 6:1             |
> | 0          | 2g.10gb            | 14      | 3                         | 0:2  <br>       |
> | 0          | 2g.10gb            | 14      | 5                         | 4:2             |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2067.png)

> **[图片提取文字 (image.png)]:**
> ## MIG: RUNNING DOCKER CONTAINERS
> 
> ## **User Workflow**
> 
> - Run GPU containers with MIG using "--gpus" option in Docker 19.03
>   - Primarily for single node development and testing
> - Enabled via NVIDIA Container Toolkit (previously known as nvidia-docker2)
> - Users configure MIG partitions using NVML/nvidia-smi
> - Launching the container requires specifying the GPU instances to expose to the container
> 
> ```
> $ docker run \
>   --gpus '"device=0:0,0:1"' \
>  nvidia/cuda:11.0-base nvidia-smi -L
> GPU 0: A100-SXM4-40GB (UUID: GPU-2ceff3df-31b3-caf2-eace-a494b4b7926b)
>  MIG 3g.20gb Device 0: (UUID: MIG-GPU-2ceff3df-31b3-caf2-eace-
> a494b4b7926b/1/0)
>  MIG 3g.20gb Device 1: (UUID: MIG-GPU-2ceff3df-31b3-caf2-eace-
> a494b4b7926b/2/0)
> $ docker run \
>   --gpus '"device=MIG-GPU-2ceff3df-31b3-caf2-eace-a494b4b7926b/1/0"' \
>  nvidia/cuda:11.0-base nvidia-smi -L
> GPU 0: A100-SXM4-40GB (UUID: GPU-2ceff3df-31b3-caf2-eace-a494b4b7926b)
>  MIG 3g.20gb Device 0: (UUID: MIG-GPU-2ceff3df-31b3-caf2-eace-
> a494b4b7926b/1/0)
> ```
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2068.png)

> **[图片提取文字 (image.png)]:**
> ## MIG: RUNNING CONTAINERS USING K8S
> 
> ## **User Workflow**
> 
> - MIG configured on the node ahead of time
> - Expected to be transparent to the end user
> - Simple exposure model for homogenous nodes
> - Other exposure options still in discussion and not settled yet
> - User jobs will be able to only execute on a single Compute Instance
> 
> ```
> apiVersion: v1
> kind: Pod
> metadata:
>  name: gpu-example
> spec:
>   containers:
>     - name: gpu-example
>       image: nvidia/cuda:11.0-base
>       resources:
>         limits:
>           nvidia.com/gpu: 1
>  nodeSelector:
>     nvidia.com/gpu.product: A100-SXM4-40GB-MIG-1g.5gb
>     nvidia.com/cuda.runtime: 11.0
>     nvidia.com/cuda.driver: 450.28.0
> ```
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2069.png)

**Test Case**

> **[图片提取文字 (image.png)]:**
> ## MIG TEST CASE 1 - BERT LARGE INFERENCE
> 
> ## **Test Results**
> 
> Perf among 7 MIG 1g.5gb slice is very stable and consistent. MIG provides great perf isolation and QoS.
> 
> 2.1x throughput when MIG is enabled for this case and config.
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2070.png)

> **[图片提取文字 (image.png)]:**
> ## MIG TEST CASE 1 - BERT LARGE INFERENCE
> 
> ## **GPU Utilization Metrics**
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2071.png)

> **[图片提取文字 (image.png)]:**
> ## MIG TEST CASE 2 - JASPER INFERENCE
> 
> ## **Test Results**
> 
> Throughput: amount of audio seconds processed by GPU in one second
> 
> With MIG enabled, throughput up to 3.4x improvement.
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2072.png)

## 推理TRITON、vGPU（逻辑划分和虚拟划分资源，灵活Migration且不必Reset）

**推理系统**

> **[图片提取文字 (image.png)]:**
> ## INEFFICIENCY LIMITS INNOVATION
> 
> ## Difficulties with Deploying Data Center Inference
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> ## Custom Development
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Developers need to reinvent the plumbing for every application
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2073.png)

> **[图片提取文字 (image.png)]:**
> ## NVIDIA TRITON INFERENCE SERVER
> 
> ## Production Data Center Inference Server
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Maximize real-time inference performance of GPUs
> 
> Quickly deploy and manage multiple models per GPU per node
> 
> Easily scale to heterogeneous GPUs and multi GPU nodes
> 
> Integrates with orchestration systems and auto scalers via latency and health metrics
> 
> Now open source for thorough customization and integration
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2074.png)

> **[图片提取文字 (image.png)]:**
> ## DYNAMIC BATCHING
> 
> ## 2.5X Faster Inferences/Second at a 50ms End-to-End Server Latency Threshold
> 
> Triton Inference Server groups inference requests based on customer defined metrics for optimal performance
> 
> Customer defines
> 
> - 1) batch size (required)
> - 2) latency requirements (optional)
> 
> Example: No dynamic batching (batch size 1 & 8) vs dynamic batching
> 
> Static vs Dynamic Batching (V100 TRT Resnet50 FP16 Instance 1)
> 
> ![](_page_0_Figure_8.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2075.png)

**异构GPU虚拟化：面向PC、工作站、计算；**

vGPU是将MIG进行超卖，比如4个MIG分时轮转5个VM，等效提供了5个vGPU。

> **[图片提取文字 (image.png)]:**
> ## VGPU FOR GRAPHICS AND COMPUTING
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Virtualization Layer
> 
> Hardware
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2076.png)

> **[图片提取文字 (image.png)]:**
> ## VGPU FOR COMPUTING
> 
> ## vCS
> 
> - Hypervisor provides best security, isolation guarantee.
> - vCS provides a good option for cost sensitive customers and those new comers to GPU computing, or application of low-utilized GPU scenarios.
> - Flexible scheduler strategy: Best effort, fixed-share, equal-share.
> - Flexible scheduler time slice (1-20 ms controllable).
> - Perf is guaranteed even that it's time-round sharing for SM resources.
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2077.png)

## **不同cuda并发机制**

> **[图片提取文字 (image.png)]:**
> ## CUDA CONCURRENCY MECHANISMS
> 
> Triton, MPS, vGPU and MIG
> 
> |                       | Parallel<br>work | Address space isolation | SM performance isolation                    | Memory<br>performance<br>isolation | Error isolation |
> |-----------------------|------------------|-------------------------|---------------------------------------------|------------------------------------|-----------------|
> | TRITON (CUDA Streams) | Yes              | No                      | No                                          | No                                 | No              |
> | MPS                   | Yes              | Yes                     | Yes<br>(by percentage,<br>not partitioning) | No                                 | No              |
> | vGPU                  | Yes              | Yes (With hypervisor)   | Yes (Time-slicing)                          | Yes                                | Yes             |
> | MIG                   | Yes              | Yes                     | Yes                                         | Yes                                | Yes             |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2078.png)

> **[图片提取文字 (image.png)]:**
> ## **COMPARISON**
> 
> ## Part 1
> 
> | Simple Comparison Among MPS, vGPU, TRITON, MIG |                                                                                                                                      |                                                                                                                                                                                                                                           |                                                                                                                 |                                                                                                  |
> |------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
> |                                                | MPS                                                                                                                                  | vGPU                                                                                                                                                                                                                                      | TRITON                                                                                                          | MIG                                                                                              |
> | Intro Link                                     | MPS Whitepaper                                                                                                                       | Official Link                                                                                                                                                                                                                             | <u>Github</u>                                                                                                   | MIG Whitepaper-NDA                                                                               |
> | Open Source                                    | No                                                                                                                                   | No                                                                                                                                                                                                                                        | Yes                                                                                                             | No                                                                                               |
> | Free                                           | Yes                                                                                                                                  | No                                                                                                                                                                                                                                        | Yes                                                                                                             | Yes                                                                                              |
> | Main Positioning                               | Improve GPU utilization for applications that doesn't fully utilize GPU, by schedule multi-process, with limited execution resource. | Offer a consistent user experience for every virtual workflow and improve GPU utilization in some scenario, by split GPU into multiple vGPUs as memory size equal partition, by integrating with hypervisor (virtual machine technology). | Provide a cloud inferencing solution optimized for NV GPU, with an inference service via HTTP or gRPC endpoint. | Improve GPU utilization and serve more users with physical resource isolation and QoS guarantee. |
> | Target Applications                            | Applications that doesn't fully utilize GPU: HPC-MPI application, training, inference with small matrix size.                        | 3D Rendering, vGaming, training, inference.                                                                                                                                                                                               | Inference.                                                                                                      | Training, inference, HPC.                                                                        |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2079.png)

> **[图片提取文字 (image.png)]:**
> ## **COMPARISON**
> 
> ## Part 2
> 
> | Simple Comparison Among MPS, vGPU, TRITON, MIG |                                                   |                                                                   |                                                                                                  |                                                                                                 |
> |------------------------------------------------|---------------------------------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
> |                                                | MPS                                               | vGPU                                                              | TRITON                                                                                           | MIG                                                                                             |
> | Supported GPU                                  | GPU since Kepler                                  | P100, P40, P4, P6, V100, T4,<br>RTX8000, RTX6000, M10, M60        | All GPU                                                                                          | A100                                                                                            |
> | Supported OS                                   | Linux                                             | Linux, Windows                                                    | Linux                                                                                            | Linux                                                                                           |
> | Extra Software<br>Needed                       | No                                                | Hypervisor(KVM, Citrix, VMWare, etc)                              | No                                                                                               | No                                                                                              |
> | Benefits                                       | Improve GPU utilization, improve throughout       | Improve GPU utilization via time-sharing, improve user experience | Improve GPU utilization, improve throughout                                                      | Improve GPU utilization, improve throughput, serve more users, provide QoS and fault isolation. |
> | GPU Resource<br>Isolation                      | Context level isolation,<br>memory and SM sharing | GPU memory isolation, SM sharing by rotation.                     | TRTIS executes model(app) instance as Thread(CPU)- Stream(GPU). SM sharing is via multi- stream. | GPU memory isolation,<br>SM isolation, other<br>engines isolation(CEs,<br>NVDEC).               |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2080.png)

> **[图片提取文字 (image.png)]:**
> ## COMPARISON Part 3
> 
> ## Simple Comparison Among MPS, vGPU, TRITON, MIG
> 
> | Simple semperation / menty in style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style style |                                                                                                                                                                                                                                                                                                |                                                                         |                                                                                                                                           |                                                                                                      |
> |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
> |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | MPS                                                                                                                                                                                                                                                                                            | vGPU                                                                    | TRITON                                                                                                                                    | MIG                                                                                                  |
> | QoS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | No strong guarantee                                                                                                                                                                                                                                                                            | Guarantee in time-slicing sharing envelop                               | No strong guarantee                                                                                                                       | Strong, the best guarantee                                                                           |
> | Ease of Use                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Easy                                                                                                                                                                                                                                                                                           | Medium                                                                  | Easy                                                                                                                                      | Easy                                                                                                 |
> | Support                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Forum                                                                                                                                                                                                                                                                                          | Professional team                                                       | Github issue                                                                                                                              | Professional team                                                                                    |
> | Considerations/Limita<br>tions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | No fault tolerance. Really not suitable for arbitrary combination of multi-user applications, especially for public cloud scenario with full isolation requirements.                                                                                                                           | Not really sharing SM as this is a time-sharing/slicing implementation. | Mainly confined to inference type workloads. Multistreaming currently not effective to TF based models (limiting factor from TensorFlow). | Only for compute<br>workloads in MIG mode,<br>don't support P2P<br>between GPU compute<br>instances. |
> | Correlations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | MPS, vGPU, TRITON, MIG are not mutually exclusive solutions.  Example: you can run MPS or TRITON in vGPU environment.  Example: you can run MPS or vGPU in MIG-enabled A100 system.  Example: you can even run multi processes in TRITON with MPS enabled, under vGPU with MIG-enabled system. |                                                                         |                                                                                                                                           |                                                                                                      |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2081.png)

## GPU使用率metric和profile工具

[https://live.nvidia.cn/gtc-od/attachments/CNS20856.pdf](https://live.nvidia.cn/gtc-od/attachments/CNS20856.pdf)

> **[图片提取文字 (image.png)]:**
> ## **GPU UTILIZATION**
> 
> ## **Metrics and Tools**
> 
> - GPU utilization: reflect how busy different resources on GPU are, metrics including GPU core(CUDA core, integer, FP32, Tensor Core), frame buffer(capacity, bandwidth), PCIe RX and TX, NVLink RX and TX, encoder and decoder, etc.
> - Generally, when we talk about GPU utilization, we are mostly talking about GPU utilization of CUDA core.
> - GPU utilization reflects an impact on delivered application performance somehow, but not necessarily.
> 
> ## Monitor tools
> 
> nvidia-smi or NVML, installed with GPU driver;
> 
> <u>DCGM</u>: Data Center GPU Manager, standalone package, using NVML and advanced data center profiling metrics;
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2082.png)

> **[图片提取文字 (image.png)]:**
> ## **GPU UTILIZATION METRIC**
> 
> ## From nvidia-smi or NVML
> 
> "GPU Utilization" from nvidia-smi or NVML is a rough metric that reflects how busy GPU cores are utilized.
> 
> Defined by "Percent of time over the past sample period during which one or more kernels was executing on the GPU", from NVML API Guide.
> 
> Extreme case, the metric is 100% even there's only one thread launched to run kernel on GPU
> 
> during past sample period.
> 
> | NVID:      | IA-SMI             | 450.5        | 1.06 Di                | river | Version: | 450.51.06        | CUDA Vers: | ion: 11.0                             |
> |------------|--------------------|--------------|------------------------|-------|----------|------------------|------------|---------------------------------------|
> | GPU<br>Fan | Name<br>Temp       |              | Persister<br>Pwr:Usage |       |          | •                | •          | Uncorr. ECC<br>L Compute M.<br>MIG M. |
> | 0<br>N/A   |                    |              |                        |       |          | 00:06:00.0 O     |            | 0<br>Default<br>N/A                   |
> | Proce      | esses:<br>GI<br>ID | CI<br>ID     | PID                    | Тур   | e Proc   | ess name         |            | GPU Memory<br>Usage                   |
> | <br>0      | N/A                | =====<br>N/A | 10210                  |       | C pyth   | :=======<br>:on3 | ========   | =========<br>15693MiB                 |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2083.png)

> **[图片提取文字 (image.png)]:**
> ## **GPU UTILIZATION METRIC**
> 
> ## From DCGM
> 
> DCGM provides CLI dcgmi and API for C and Python language.
> 
> DCGM DCP(Data Center Profiling) provides lower level profiling metrics, which lists several utilization metrics in more accurate.
> 
> From these metrics, better reflect how well GPU resources are utilized to some extent.
> 
> Well, one GPU has many different resources (computing, memory, IO), it's highly recommended to capture several metrics to understand GPU utilization, not just one or two.
> 
> ![](_page_0_Picture_6.jpeg)
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2084.png)

> **[图片提取文字 (image.png)]:**
> ## GPU UTILIZATION METRIC
> 
> ## **DCGM DCP Metrics**
> 
> | Metric                      | Definition                                                                                                                                                       | DCGM Field ID                   |
> |-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
> | Graphics Engine<br>Activity | Ratio of time the graphics engine is active. The graphics engine is active if a graphics/compute context is bound and the graphics pipe or compute pipe is busy. | DCGM_FI_PROF_GR_ENGINE_ACTIVE   |
> | SM Activity                 | The ratio of cycles an SM has at least 1 warp assigned (computed from the number of cycles and elapsed cycles)                                                   | DCGM_FI_PROF_SM_ACTIVE          |
> | SM Occupancy                | The ratio of number of warps resident on an SM. (number of resident as a percentage of the theoretical maximum number of warps per elapsed cycle)                | DCGM_FI_PROF_SM_OCCUPANCY       |
> | Tensor<br>Utilization       | The ratio of cycles the tensor (HMMA) pipe is active (off the peak sustained elapsed cycles)                                                                     | DCGM_FI_PROF_PIPE_TENSOR_ACTIVE |
> | Memory BW<br>Utilization    | The ratio of cycles the device memory interface is active sending or receiving data.                                                                             | DCGM_FI_PROF_DRAM_ACTIVE        |
> | FLOP Counts                 | Ratio of cycles the fp64 /fp32 / fp16 / HMMA IMMA pipes are active.                                                                                              | DCGM_FI_PROF_PIPE_FPXY_ACTIVE   |
> | NVLink<br>Utilization       | The number of bytes of active NVLink rx or tx data including both header and payload.                                                                            | DCGM_FI_DEV_NVLINK_BANDWIDTH_L0 |
> | PCIe Utilization            | <pre>pcibytes_{rx, tx} - The number of bytes of active pcie rx or tx data\nincluding both header and payload.</pre>                                              | DCGM_FI_PROF_PCIE_[T R]X_BYTES  |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2085.png)

> **[图片提取文字 (image.png)]:**
> ## GPU UTILIZATION METRIC
> 
> ## Using dcgmi
> 
> Recommended monitor command with dcgmi
> 
> \$ dcgmi dmon -e 1001,1002,1004,1005,1009,1010,1011,1012,150,155,110,111
> 
> | # Entity GRACT SMACT TENSO DRAMA PCITX PCIRX NVLTX NVLTX TMPTR POWER SACLK MAC Id C W  GPU 0 0.931 0.777 0.175 0.496 175899291 1532954951 1547634958 1553333956 52 323.689 1410 1  GPU 1 0.948 0.780 0.173 0.496 172945598 1507859117 1522127704 1522127640 50 213.963 1410 1  GPU 2 0.952 0.778 0.175 0.493 178507418 1557783818 1577668487 1572504828 48 359.610 1410 1  GPU 3 0.962 0.793 0.178 0.503 164054321 1428701446 1327745638 1327396166 52 226.107 1410 1  GPU 4 0.960 0.786 0.179 0.499 163908021 1430858946 1288201051 1287639531 64 392.270 1410 1  GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1  GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1  GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 67 380.955 1410 1  GPU 0 0.950 0.793 0.179 0.505 162992146 1418772939 1429455794 1435422194 54 304.839 1410 1 |
> |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
> | GPU 0 0.931 0.777 0.175 0.496 175899291 1532954951 1547634958 1553333956 52 323.689 1410 1 GPU 1 0.948 0.780 0.173 0.496 172945598 1507859117 1522127704 15221276460 50 213.963 1410 1 GPU 2 0.952 0.778 0.175 0.493 178507418 1557783818 1577668487 1572504828 48 359.610 1410 1 GPU 3 0.962 0.793 0.178 0.503 164054321 1428701446 1327745638 1327396166 52 226.107 1410 1 GPU 4 0.960 0.786 0.179 0.499 163908021 1430858946 1288201051 1287639531 64 392.270 1410 1 GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1 GPU 6 0.966 0.817 0.200 0.508 132741767 148660264 128201051 1287637355 1127111684 64 258.063 1410 1 GPU 7 0.999 0.867 0.325 0.451 8908556 34245363 0 0 0 67 380.955 1410 1                                                                                                                                                                               |
> | GPU 1 0.948 0.780 0.173 0.496 172945598 1507859117 1522127704 1522126460 50 213.963 1410 1 GPU 2 0.952 0.783 0.175 0.493 178507418 1557783818 1572668487 1572504828 48 359.610 1410 1 GPU 3 0.962 0.793 0.178 0.593 164054321 1428701446 1327745638 1327396166 52 226.107 1410 1 GPU 4 0.960 0.786 0.179 0.499 163908021 1430858946 128201051 1287639531 64 392.270 1410 1 GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1 GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1 GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 0 67 380.955 1410 1                                                                                                                                                                                                                                                                                     |
> | GPU 2 0.952 0.778 0.175 0.493 178507418 1557783818 1572668487 1572504828 48 359.610 1410 1 GPU 3 0.962 0.793 0.178 0.503 164054321 1428701446 1327745638 1327396166 52 226.107 1410 1 GPU 4 0.960 0.786 0.179 0.499 163908021 1430858946 1288201051 1287639531 64 392.270 1410 1 GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1 GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1 GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 6 7 380.955 1410 1                                                                                                                                                                                                                                                                                                                                                                                |
> | GPU 3 0.962 0.793 0.178 0.503 164054321 1428701446 1327745638 1327396166 52 226.107 1410 1 GPU 4 0.960 0.786 0.179 0.499 163908021 1430858946 1288201051 1287639531 64 392.270 1410 1 GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1 GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1 GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 67 380.955 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
> | GPU 4 0.960 0.786 0.179 0.499 163908021 1430858946 1288201051 1287639531 64 392.270 1410 1 GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1 GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1 GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 67 380.955 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
> | GPU 5 0.952 0.797 0.182 0.506 182644334 1599554874 1235853101 1233988554 62 341.524 1410 1<br>GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1<br>GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 67 380.955 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
> | GPU 6 0.966 0.817 0.200 0.508 132741767 1148660264 1129637355 1127111684 64 258.063 1410 1<br>GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 67 380.955 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
> | GPU 7 0.999 0.867 0.325 0.451 8908656 34245363 0 0 67 380.955 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
> |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
> | GDU 0 0 050 0 702 0 170 0 505 162002146 1410772020 1420455704 1425422104 54 204 920 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
> | GPU 1 0.954 0.793 0.179 0.505 162944796 1418947251 1430185344 1430185344 52 201.105 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 2 0.959 0.795 0.179 0.505 162966713 1419469072 1430752928 1430665363 53 372.072 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 3 0.962 0.796 0.179 0.505 162992814 1418956709 1430003326 1429872315 56 195.564 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 4 0.960 0.792 0.179 0.505 162681409 1418483393 1431427800 1430779751 66 400.533 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 5 0.948 0.789 0.179 0.506 162794813 1419172557 1435095820 1431911846 65 355.586 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 6 0.957 0.794 0.179 0.505 162844371 1418843705 1439494242 1434260115 67 292.070 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 7 0.958 0.797 0.180 0.506 163341225 1422783028 1440327549 1443650284 70 384.132 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 0 0.949 0.793 0.179 0.505 163030005 1419242144 1431636763 1440810765 55 237.659 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 1 0.954 0.793 0.179 0.505 162773503 1418681427 1431965210 1431965210 53 184.241 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 2 0.957 0.795 0.179 0.506 162881890 1419208015 1432242919 1432242919 53 366.301 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 3 0.959 0.797 0.179 0.506 163018599 1419626682 1432350659 1432350659 56 225.281 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 4 0.957 0.792 0.180 0.506 162612068 1418432187 1433020167 1432423976 66 396.763 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 5 0.949 0.790 0.179 0.506 162527136 1417457784 1436326347 1431757595 65 278.087 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 6 0.956 0.794 0.179 0.505 162874341 1419286866 1446253398 1437670082 68 330.311 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 7 0.957 0.797 0.181 0.506 162804092 1419116905 1441752283 1446325537 71 402.675 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 0 0.951 0.796 0.180 0.507 163049364 1419855949 1431014322 1437445669 55 178.524 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 1 0.951 0.795 0.180 0.507 162991159 1420276463 1431002457 1431002457 52 237.659 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 2 0.958 0.797 0.180 0.507 162890127 1419226321 1430767359 1430767359 53 366.837 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 3 0.960 0.797 0.179 0.506 162951049 1419130718 1430934017 1430371626 56 320.427 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 4 0.957 0.794 0.181 0.506 162632607 1418105919 1431840408 1430961031 67 387.907 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 5 0.948 0.791 0.180 0.507 162692025 1418504737 1435486856 1431624416 66 202.787 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 6 0.958 0.795 0.180 0.507 162784709 1418531037 1441021899 1435473584 68 383.275 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 7 0.955 0.798 0.182 0.507 162806309 1418604281 1436151038 1440574380 71 408.494 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 0 0.954 0.795 0.180 0.506 162990387 1418928837 1429538441 1434605453 55 231.030 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 1 0.953 0.794 0.179 0.506 162726406 1418134181 1429561807 1429561807 53 327.149 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 2 0.957 0.795 0.179 0.506 162749917 1418119150 1429417688 1429417688 54 318.781 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 3 0.960 0.798 0.179 0.506 162980413 1418831034 1429747257 1429747257 56 366.668 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 4 0.958 0.794 0.181 0.507 162550382 1417626219 1430946788 1429692646 67 355.460 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 5 0.948 0.790 0.179 0.506 162619862 1417919747 1434448412 1431062748 65 225.026 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 6 0.958 0.796 0.180 0.506 162713169 1418220725 1438159351 1434345897 68 394.858 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | GPU 7 0.956 0.797 0.181 0.506 162676691 1417962711 1434673417 1438058602 72 403.272 1410 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2086.png)