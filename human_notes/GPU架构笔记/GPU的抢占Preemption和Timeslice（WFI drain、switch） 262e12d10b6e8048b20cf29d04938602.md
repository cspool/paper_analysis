# GPU的抢占Preemption和Timeslice（WFI/drain、switch）

# Program Execution And Preemption

ref：SOFTWARE-ASSISTED INSTRUCTION LEVEL, EXECUTION PREEMPTION

### 总结

[https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/embedded_software_components.html](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/embedded_software_components.html)

wait-for-idle抢占的time-slicing（**短时高优先级进程设置wait-for-idle禁止抢占**）：等待pipeline空闲，之后切换其他应用，要求pipeline能快速drain，不适合长延迟应用。

stall-switch抢占的time-slicing（CTA-level和instr-level）：阻塞不同深度的任务发射，保存不同深度的Ctx，切换新Ctx。

> **[图片提取文字 (image.png)]:**
> between multiple different applications. When multiple different applications need to use the processor simultaneously, one way to achieve forward progress on all the applications is to run each application for a short time-slice on the processor. Conventionally, time slicing requires that the processor pipeline be completely drained and when the processor is idle, a different application is switched in to be executed by the processor pipeline. This mechanism for time slicing has been referred to as "wait for idle" preemption and the mechanism does not work well when the processor takes a long time to drain the work that is running on the processor pipeline. For example, consider a very long running graphics shader program, or in the worst case, a shader program with an infinite loop. To be able to time slice between different applications, the amount of time needed to idle execution of each application should be limited so that long running applications do not effectively reduce the time slice available for other applications.
> 
> Preemption is a mechanism to time-slice a processor
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> Another mechanism that has been considered to implement preemption, is to stall or freeze the processor and then store the contents of all the registers and pipeline flip-flops within the processor and later restore the contents of all of the registers and pipeline flip-flops within the processor. Storing and restoring the contents of all of the registers and pipeline flip-flops typically results in a very large amount of state to be saved and restored. The time needed to store and restore the state reduces the time available for executing each of the applications during the time slices. Accordingly, what is needed in the art is a system and method for execution preemption that either does not require storing the entire state of an application when the application is preempted or does not require waiting for a processing pipeline to become idle to preempt the application.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%201.png)

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

> **[图片提取文字 (image.png)]:**
> **Table 1.** Comparison of BERT inference latency against turn-around latency of different scheduling granularity for Whisper training on NVIDIA A100 GPU.
> 
> | Inference time | Turnaround latency (Whisper) |        |         |        |
> |----------------|------------------------------|--------|---------|--------|
> | (BERT)         | Iteration                    | Kernel | Block   | Thread |
> | 3.93ms         | ~ 3s                         | ~ 10ms | ~ 304µs | ~ 38µs |
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2026.png)

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

**并发的上下文Ctx切换**

上下文Ctx是channel中任务在GPU分配资源上的运行时镜像。

不同channel按照时间片TS轮流执行，需要切换不同的channel Ctx。

stream并发是channel Ctx内同时派发不同Stream的CTA，不用切换channel Ctx。

SM内的warp并发，不用切换channel Ctx，并且SM内并发warp的Ctx同时存在，不用切换。

**旧channel到达timeslice，新channel抢占GPU过程**

1、GPU驱动提前在显存VRAM分配**Ctx buffer**空间。

2、主机接口从Runlist中选择新channel，**抢占**旧channel的Ctx占据的资源，**前端和各层trap handler控制**抢占过程。

3、**GPU停止**context处理、**保存**context到内存、**重置**Engine/SM、**加载**上下文、**恢复**任务启动。

4、GPU完成抢占后，前端向主机接口发送**ACK**，让主机接口切换新channel执行。

stall & switch**抢占的原理**

暂停不同深度层次上（TMU和GPC、SM）上的任务调度和资源分配（CTA或warp）。

等待对应深度上资源空闲，即TMU和GPC已调度CTA或SM已发射指令完成后释放资源。

“暂停-等待空闲”是因为**GPU高并发，不能追踪和控制特定任务，以资源为handler**，等待资源上任务完成，或者放弃任务。

资源空闲后，保存旧Ctx，加载新Ctx并且开始执行。

**硬件控制的CTA level或者指令level抢占**

**停止**context处理的行为

1、前端停止接收host卸载的任务、发出preemption信号。

2、各层scheduler收到preemption后停止调度任务，各层scheduler执行**trap handler**程序。

CTA level：前端停止为GPC和SM调度任务，尝试**等待GPC内CTA释放**（drain），**timer**超时后转为指令level抢占。

指令level：SM停止调度warp，**等待访存指令和其他已发射指令完成（drain）**。

等待资源idle而不打断的场合：中断和错误处理过程、快速的图形任务（属于WFI）。

3、前端等待Context停止后，发出**context freeze**信号，在save和restore过程中冻结SM。

**保存**context到内存，**重置**资源模块的行为。

1、**CTA level**的context是TMU中每个GPC运行的CTA级状态，因为下游GPC已经被排空（drained），**无需保存GPC/SM内部状态**。

2、**指令level**的context是任务从TMU调度到SM指令发射的**全栈**。context完成后trap handler退出所有活动线程，强制使SM和GPC进入**空闲**状态。

SM的GRF、SM内DSMEM（共享内存）、SM的调度器MPC。

GPC的pipeline manager、GPC的GRF，**将GPC level缓存（L1.5 cache）写回内存并失效**。

**加载**上下文，**恢复**任务启动的行为

1、CTA level的context-restore mode：由于没有底层状态需要加载，被抢占的CTAs会直接按照之前被抢占的顺序（preempted order）被重新发射执行。

2、指令level的context-restore mode

Pipeline manager配置SM进入**preemption-restore-begin模式**，使其处于暂停状态。

**精准原址恢复**：为确保资源不超限，之前被抢占的 CTA 会被精确发送回**同一个SM内的同一个物理CTA slot**，线程组也被恢复到**同一个物理线程组ID**。

SM 恢复所有寄存器、屏障（barriers）、程序计数器（PC）和活动掩码等，随后退出restore模式并恢复执行。

**软件辅助的指令level抢占**

和硬件控制的指令level抢占一样，都不需要等待SM内CTA空闲，但是实现方式不同，让SM中运行的CTA跳转到trap handler（PC强制）。

trap handler首先运行**preemption-save kernel**来保存指令level上下文。

基于上下文指针，生成最高优先级的**preemption TMD（被抢占任务的封装）插入前端队列**，之后执行**退出**，**“假装CTA已经完成”**。

恢复上下文时，preemption TMD最先调度到SM执行。

preemption TMD执行**preemption-restore kernel**，将硬件状态和寄存器恢复。

preemption TMD**恢复执行**CTA未完成的指令。

CTA正常运行结束时，preemption TMD调用**AtExit**，释放TMD和context占据的资源**。**

**指令level抢占的不同实现**

**硬件控制**：恢复时，硬件必须将之前被抢占的CTA精确发射回**原来执行它的同一个SM内的同一个物理CTA槽位（分配完全相同资源）**，甚至warp也要使用相同的物理warp ID，保证资源不超限。

**软件辅助**：抢占时，将被抢占CTA包装成最高优先级的**抢占TMD**。恢复时，这个TMD作为优先级极高的任务被前端调度器调度，随后执行`preemption-restore kernel`来恢复上下文。解除CTA、warp对底层资源的强绑定。

软件辅助指令级抢占的设计初衷，是为了在**复用CTA级抢占的硬件控制流**的基础上，通过软件（子任务生成和exit指令）跳过耗时的排空阶段，不需要增加极其复杂的纯硬件指令级卸载逻辑。

## Fig 系统图（TMU、GPC、SM、CTA、TG）

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3A
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3B
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%204.png)

**channel**是任务的指针集合，任务可能包含多个channel，**runlist**是channel的有序列表；

**CPU控制preemption，channel** preempt、**runlist** preempt将被抢占的任务指针写入channel reg和runlist reg；

不同任务的**timeslice**不同（**dense、sparse**），主机没有新任务会通知GPU front end；

当**timeslice**结束或任务完成时，**切换context**并切换任务；

> **[图片提取文字 (image.png)]:**
> time-slice on the processor. Preemption may also be used to unload the currently executing context for other purposes. For example, the host interface 206 may preempt a context when the CPU 102 initiates a channel preempt or a runlist preempt, where a channel is a collection of pointers to processing work and an application may contain one or more channels. A channel preempt is performed by clearing a valid bit in a channel ram entry and writing a channel identifier of the channel to be preempted to a preempt register. The specified channel is then unloaded from the PPU **202** off both host and the engine. A runlist preempt is performed by writing a pointer to the runlist register. The pointer may point to a new runlist or
> 
> Preemption may be used to time-slice a processor
> 
> between multiple different applications so that the different
> 
> applications are serialized and each execute for a short
> 
> preempt causes what is running in a PPU **202** to be unloaded. The host interface **206** then begins processing at the first entry on the runlist associated with the pointer and searches for the first valid entry with pending work. The first channel on the runlist which has pending work is loaded into the PPU **202**.
> 
> may point to the runlist that is currently active. Runlist
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> The host interface 206 may also preempt a context that is executing before a time slice has expired when the context is out of methods (i.e. programs) and another context is waiting to execute. In one embodiment, the time slices are not equal amounts of time, but instead are based on each context's method stream, so that a context with a dense method stream is allocated a larger time slice compared with a different context having a sparse method stream. The host interface 206 is configured to indicate to the front end 212 when the host interface 206 does not have any methods for an executing context. However, the host interface 206 does not initiate a context switch for the executing context until either the time slice allocated to the context has expired or the processing pipeline is idle and there are no methods.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%206.png)

## Fig 4 context切换过程

**host**选择新的context，指示**front end启动和控制context抢占过程**，包含五个phase：

**停止**当前context上的处理：**CTA level**停止为GPC、SM分配任务，**指令level**停止SM中指令调度和发射，若遇到中断和错误，则等待其完成；

>**保存**当前context到内存；

>**重置**engine；

>**加载**新的context；

>**启动**新context的处理；

完成context抢占后，front end向host发送**ACK**；

抢占过程中，**等待图形pipline完成而不抢占**，因为图形比计算快，并且图形切换开销大；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4
![image.png](GPU%E5%B9%B6%E5%8F%91%E6%9C%BA%E5%88%B6TS%E3%80%81MPS%E3%80%81MIG%E3%80%81vGPU/image%2028.png)

> **[图片提取文字 (image.png)]:**
> processing pipeline beginning with the task/work unit 207 through the GPCs 208, according to one embodiment of the invention. The preemption process has five phases that are controlled by the front end 212. A first phase (phase 1) stops the processing in the current context. For CTA level preemption this means stopping work at a CTA task boundary. For instruction level preemption this means stopping work at an SM 310 instruction boundary. If an interrupt or fault
> 
> FIG. 4 is a block diagram of the host interface 206 and the
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> occurs after preemption is initiated and during phase 1, the front end 212 waits for the pending interrupt or fault to be cleared before proceeding to phase 2.
> 
> Once the context is stopped (and any interrupts or faults are cleared), phase 2 saves the current context's state in memory. Phase 3 resets the engine before phase 4 loads a new context's state onto the machine. Phase 5 restarts the processing of any work that was preempted in a previous Phase 1. When preempting a context, the host interface 206 selects a new context from the runlist to execute and instructs the front end 212 to begin context preemption. The front end 212 configures the processing pipeline to execute the new context by completing the five phases of the preemption process. After the five phases of the preemption process are completed, the front end 212 sends an acknowledge (ACK) to the host interface 206. In one embodiment a separate graphics processing pipeline (not shown in FIG. 4) performs graphics-specific operations and the front end 212 also waits for the graphics processing pipeline to become idle. Typically, the graphics processing methods execute in shorter times compared with compute processing methods, so waiting for the graphics processing pipeline to become idle may be completed while the processing pipeline completes the first phase of the preemption process. Also, the amount of state information that is maintained in a graphics processing pipeline is typically much larger than the context state maintained in the (compute) processing pipeline. Waiting for the graphics processing pipeline to idle significantly reduces the storage needed to capture the context state.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%208.png)

抢占执行前，CPU程序为GPU的**CTA** level（不是thread block？似乎是channel）和**指令**level的context分配**context buffer**空间；

**phase 1**：front end**停止接收host**的kernel，对task/work unit发出**preemption命令**，处理单元/SM收到preemption后**停止**向下游单元输出任务，**等待**全部停止后，front end发出**context freeze**信号来确保处理单元不占用保存context的传输资源；

**wait-for-idle（WFI）**的任务被抢占时**要求front end等待pipeline idle之后**，一般是长时间片的高优先级任务；

**指令level**的context包括SM内warp/指令、GPC内的CTA和front end内任务的情况，即任务启动到指令发射的**全栈**；

**CTA level**的context包括front end内任务的情况，因为尝试等待GPC内的下游模块任务完成（drain），**减少需要保存的context**；

context 抢占过程中的**动态产生的任务**，包含在front end context中；

**task manager unit**（front end后的scheduler和distributor）将**任务组织成CTA构成的grid**（隐式的x，y，z三个维度），CTA是CUDA模型中block。

> **[图片提取文字 (image.png)]:**
> Before preemption is performed, a context buffer to store the CTA level (and instruction level) context state for a particular context is allocated by a program executed on the CPU 102. The size of the context buffer that is allocated may be based on the PPU 202 configuration and the number of SMs 310.
> 
> To complete the first phase of the preemption process, the front end 212 stops accepting new methods from the host interface 206 and outputs a preempt command to the task/
> 
> work unit 207. When the preempt command is received by
> 
> a processing unit, the processing unit stops outputting work
> 
> to a downstream unit. The front end 212 waits for all
> 
> downstream units to stop outputting work, and then asserts a context freeze signal to being the second phase of the preemption process. Assertion of the context freeze signal ensures that the processing pipeline does not perform any operation based on the transactions used to save the context state. The front end 212 also determines if a wait-for-idle command is being processed which requires the front end 212 to wait for the processing pipeline to become idle, and, if so, the front end 212 interrupts the wait-for-idle operation and saves context state information indicating that a wait-for-idle command was being executed for the context. When the context is resumed, the wait-for-idle execution will be restarted by the front end 212.
> 
> When the task/work unit 207 receives the preempt command, the task/work unit 207 stops launching new work.
> 
> two phases of the preemption process are complete and notifies the front end 212 that the processing pipeline is idle. The front end 212 will then store the context state maintained within the task/work unit 207 before resetting the processing pipeline to complete the third phase of the preemption process. When instruction level preemption is used, the context state maintained within the GPCs 208 is stored by the GPCs 208 themselves. When the CTA level preemption is used, the GPCs 208 are drained so that the amount of context state that is stored is reduced.
> 
> Eventually, the task/work unit 207 determines that the first
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> Even after the task/work unit 207 stops launching work, the task/work unit 207 may receive additional work that may be generated by the GPCs 208 during execution of previous instructions. The task/work unit 207 buffers the additional work to be stored by the front end 212 as part of the context state for the task/work unit 207.
> 
> When the preempt command is received, the work distribution unit 340 stops launching CTAs. When CTA level preemption is performed, the processing units in the processing pipeline that are downstream from the work distribution unit 340, e.g., GPCs 208, are drained so that no context state remains in those downstream processing units. Therefore, the amount of context state is reduced when CTA level preemption is performed compared with instruction level preemption because instruction level preemption does not require draining the downstream processing units.
> 
> The work distribution unit 340 determines which GPCs 208 will execute received work based on information generated by the task management unit 300. Because the GPCs 208 are pipelined, a single GPC 208 may execute multiple tasks concurrently. The task management unit 300 schedules each processing task for execution as either a grid or queue. The work distribution unit 340 associates each CTA with a specific grid or queue for concurrent execution of one or more tasks. CTAs that belong to a grid have implicit x, y, z parameters indicating the position of the respective CTA within the grid. The work distribution unit 340 tracks the available GPCs 208 and launches the CTAs as GPCs 208 are available.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2010.png)

指令level抢占时，**preemption命令**到达SM后，SM由**trap handler**控制preemption的过程，完成后向上层pipeline manager（GPC）传递信号；

> **[图片提取文字 (image.png)]:**
> manager 305 in the GPCs 208. The pipeline manager 305 may include a controller for each SM 310. Upon receiving the preempt command, the SMs 310 stop issuing instructions and enter a trap handler. The SMs 310 also wait for all memory transactions associated with previously issued instructions to complete, i.e., for all outstanding memory requests to complete. Memory requests are considered to be outstanding when data for a read request has not been returned and when an acknowledgement has not been received from the MMU 328 for a write request for which an acknowledgement was explicitly requested. The pipeline
> 
> managers 305 maintain information about CTAs and thread
> 
> groups and track which thread groups are preempted per
> 
> instructions and each SM 310 becomes idle, the trap handler
> 
> Once the SMs 310 in the GPCs 208 have stopped issuing
> 
> CTA.
> 
> During instruction level preemption, the work distribution
> 
> unit 340 passes the preempt command to the pipeline
> 
> unloads the context state for the CTAs running on the GPCs 208 and a combination of one or more of the trap handler, the pipeline manager 305, and the front end 212 stores the context state. The context state that is unloaded and stored includes registers within the SMs 310, registers within the pipeline manager 305, registers within the GPCs 208, shared memory, and the like, is saved to a predefined buffer in
> 
> graphics memory. Also, writes to memory from the caches
> 
> within the GPCs 208, e.g., L1.5 cache 335, are forced out to
> 
> memory and the caches are invalidated. Once all the context
> 
> state has been unloaded and stored, the trap handler will exit
> 
> all active threads, thereby idling the SMs **310** and the GPCs
> 
> The trap handler then controls a signal from the SMs 310 to the pipeline manager 305 indicating that the first two phases of the preemption process have been completed by the GPCs 208 and that the GPCs 208 are idle. The pipeline manager 305 reports to the work distribution unit 340, ACKing the preempt command to indicate that the first two
> 
> phases of the preemption process have been completed. This
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ACK is passed upstream from the work distribution unit **340** to the task management unit **300** and finally up to the front end **212**.
> 
> The pipeline manager 305 holds state information for each thread group that was executing within the GPC 208 when the preempt command was output by the work distribution unit 340. The state information indicates whether a thread group exited after completing execution or if the thread group was preempted. The state information is saved by the pipeline managers 305 and may be used by the pipeline managers 305 to restore only those thread groups that were preempted. When all of the threads in a thread group exit after the pipeline manager 305 receives the preempt command and before the trap handler is entered to store the state information, state information is not stored for the thread group and the thread group is not restored. After the GPCs 208 are idle, the GPCs may be reset to complete the third phase of the preemption process.
> 
> The front end 212 then completes the second phase of the preemption process by writing out the context state maintained by the front end 212. The front end 212 saves all registers and ramchains out into the context state buffer for the preempted context. To complete the third phase of the preemption process, the front end 212 asserts a context-reset signal that is received by the processing pipeline, e.g., the task/work unit 207, and the GPCs 208.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2012.png)

**phase4-5**：**加载**上下文，**恢复**被抢占的任务启动（SM的**preemption-restore** mode的开始和结束）、传递ready信号的顺序类似**“栈”**；

> **[图片提取文字 (image.png)]:**
> A context that has been preempted will be reloaded even when there are no methods left for the selected context because there may be work that was generated by the SMs 310 during execution of the methods and saved as part of the context state.
> 
> The front end 212 signals to the host interface 206 whether the context was idle when the host interface 206 initiated the preemption. If the context was idle, i.e., the processing pipeline was idle and there were no outstanding memory requests, the preempted context does not need to be reloaded before execution of the context resumes. If the
> 
> When a context is selected to be executed, the host
> 
> interface 206 needs to determine if the selected context is a
> 
> context that was previously preempted. A context reload
> 
> (ctx\_reload) flag indicating whether a context was pre-
> 
> empted is maintained by the host interface 206. When the
> 
> host interface 206 recognizes that the selected context was
> 
> preempted, the previously unloaded and stored context state
> 
> There is also the case where the processing pipeline is already idle when the front end 212 receives the preempt command from the host interface 206. When the processing pipeline is already idle, the front end 212 does not send a preempt command to the task/work unit 207, but rather continues with the second phase of the preemption process. Therefore, the idle state of the task/work unit 207 and GPCs 208 should enable those units to receive a new context state
> 
> context was not idle, the host interface 206 saves the context
> 
> reload state to be processed when the channel is reloaded.
> 
> pipeline managers 305 should restore only preempted thread groups or CTAs and should not restore thread groups that exited.
> 
> When the front end 212 completes the fourth phase of the preemption process, the selected context state is read from a context buffer and loaded into the registers and ramchains.
> 
> The context freeze signal is asserted by the front end 212
> 
> or restore a context state. For example, the task/work unit
> 
> 207 should be in a state such that no tasks are running. The
> 
> from the start of the second phase until the end of the fourth phase of the preemption process. Assertion of the context freeze signal ensures that the processing pipeline does not
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2013.png)

> **[图片提取文字 (image.png)]:**
> front end **212** to save and restore the context state.
> 
> The front end **212** initiates the fifth phase (phase **5**) of the preemption process by outputting a preempt restore command to the task/work unit **207**. After the task/work unit **207** 
> 
> receives the preempt restore command, the task/work unit 207 does not assert a ready signal to the front end 212 so that no new work can be passed from the front end 212 to the task/work unit 207 until the preemption process is completed. The work distribution unit 340 within the task/work unit 207 receives the preempt restore command and restores
> 
> perform any operation based on the transactions used by the
> 
> the selected context state, replaying the restored tasks into the GPCs 208, and restoring preempted CTAs and thread groups back into the pipeline managers 305 and the SMs 310, respectively.
> 
> For example, a pipeline manager 305 outputs the preempt restore command to configure a respective SM 310 to enter
> 
> "preemption-restore-begin" mode. Then the pipeline man-
> 
> ager 305 sends the preempted CTAs and thread groups to the
> 
> SM 310. After the pipeline manager 305 has restored all
> 
> preempted thread groups, the pipeline manager 305 outputs a command to the SM 310 indicating that the "preemption-restore-end" mode should be exited. When the CTA level preemption is used, the GPCs 308 do not have any stored context state to reload and there is no thread group state to restore.
> 
> When instruction level preemption is used to restore a
> 
> selected context, the GPCs 308 read the context state for the
> 
> selected context from a context buffer and load the registers and shared memory. Pipeline managers 305 restart all the CTAs that were preempted by sending the CTAs to the respective SM 310 which each CTA was executing on, in the order that the CTAs were reported preempted. This technique ensures that each CTA is launched in the same
> 
> physical CTA slot in an SM 310 as the CTA occupied when the context was preempted. Thread groups are launched in the same physical thread group ID. Restarting the thread groups in the same location after preemption is advantageous because the thread groups and CTAs are guaranteed to not exceed the memory and other resources available in the respective SM 310. Each SM 310 restores register values, barriers, a program counter, stack pointer, active mask for each thread group, and the like.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2014.png)

**ACK**

> **[图片提取文字 (image.png)]:**
> command to the host interface 206. The ACK indicates the preemption process is complete and execution of the selected context has been initiated. Any previously preempted CTAs have resumed execution in the Task/Work Unit 207 and the GPCs 208. When instruction level preemption is used, any previously preempted threads have resumed execution on the SMs 310. The Host interface 206
> 
> may now start sending new work into the graphics pipeline.
> 
> Finally, The front end 212 ACKs the original preemption
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> In one embodiment, the front end 212 ACKs the original preemption command after outputting the preempt restore command to the task/work unit 207 and the task/work unit **207** buffers any new work that is received after the preempt restore command until phase 5 is completed. The task/work unit 207 does not launch any new (unrestored) CTAs until the preemption process is completed. The front end 212 is therefore unaware of when the fifth phase is completed. If the task/work unit 207 cannot buffer all of the new work, the task/work unit 207 negates the ready signal to the front end 212. However, the front end 212 is not able to distinguish whether the ready signal is negated during or after comple-
> 
> tion of the preemption process.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2016.png)

## Fig 5A **指令level的context unload**

**指令level的context unload**：front end停止启动任务、scheduler/distributor停止派发任务给GPC、SM停止发调度和发射指令、等待内存传输指令完成、保存上下文、重置（trap handler）……

> **[图片提取文字 (image.png)]:**
> ## FIG. 5A illustrates an unload method 500 for unloading context state when a process is preempted at an instruction
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2017.png)

> **[图片提取文字 (image.png)]:**
> the art will understand that any system configured to perform the method steps, in any order, is within the scope of the inventions.
> 
> At step 505 the host interface 206 outputs an instruction level preemption command to the front end 212 and the unloading of the current context is initiated. At step 510 the
> 
> front end 212 determines if the processing pipeline is idle,
> 
> and, if so, then the front end 212 proceeds directly to step
> 
> level, according to one embodiment of the invention.
> 
> Although the method steps are described in conjunction with
> 
> the systems of FIGS. 1, 2, 3A, 3B, and 4, persons skilled in
> 
> 545 to store the context state that is maintained by the front end 212.
> 
> If, at step 510 the front end 212 determines that the processing pipeline is not idle, then at step 515 the front end 212 stops launching new work for the current context. At step 520 the front end 212 outputs a preempt command to the task/work unit 207. At step 525 the task management unit 300 within the task/work unit 207 stops issuing tasks to
> 
> the work distribution unit 340 and outputs the preempt
> 
> command to the work distribution unit 340. At step 525 the
> 
> work distribution unit 340 also stops launching CTAs and outputs the preempt command to the pipeline managers 305. The pipeline managers 305 output the instruction level preempt command to the SMs 310.
> 
> At step 525 the SMs 310 stop executing instructions and in step 530 the SMs 310 wait for any outstanding memory transactions to complete. Each SM 310 repeats step 530 until all of the memory transaction are completed. The SMs 310
> 
> group exited or was preempted. When all of the outstanding memory transactions are complete, at step 535 the context state maintained in the SMs 310 is stored into a context buffer and the context state maintained in the pipeline managers 305 is also stored into the context buffer.
> 
> At step 540 the pipeline managers 305 report to the work distribution unit 340 that the instruction level portion of the processing pipeline, e.g., the SMs 310 and the GPCs 208, are
> 
> indicate to the pipeline manager 305 whether each thread
> 
> idle and the work distribution unit 340 then saves the CTA level state that is maintained in the work distribution unit 340 for the current context. The work distribution unit 340 reports to the task management unit 300 that it has completed this phase of preemption. The task management unit 300 then saves the task level state maintained in the task management unit 300. The task management unit 300 reports to the front end 212 when the current state has been stored, and at step 545 the front end 212 stores the context state that is maintained for the current context by the front end 212 to the context buffer. At step 550 the front end 212
> 
> then stores an indication that the saved context state is for a
> 
> preempted context, and resets the processing pipeline.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 5A
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2019.png)

## Fig 5B 指令level的**context restore**

指令level的**context restore：context-restore mode；**

> **[图片提取文字 (image.png)]:**
> context state when a process that was preempted at the instruction level is restored, according to one embodiment of the invention. Although the method steps are described in conjunction with the systems of FIGS. 1, 2, 3A, 3B, and 4, persons skilled in the art will understand that any system configured to perform the method steps, in any order, is within the scope of the inventions. At step 565 the front end 212 initiates restoration of a saved context for a context selected by the host interface 206. At step 570 the front end 212 asserts the context freeze signal to ensure that the processing pipeline does not per-
> 
> form any operation based on the transactions used by the
> 
> front end 212 to restore the context state. At step 575 the
> 
> FIG. 5B illustrates a restore method 560 for restoring
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2020.png)

> **[图片提取文字 (image.png)]:**
> selected context state is read from a context buffer by the front end 212 and task/work unit 207, and restored at the task and CTA level.
> 
> At step 580 each pipeline manager 305 outputs a com-
> 
> mand down to configure the respective SM 310 to enter "preemption-restore-begin" mode, thereby configuring the SMs 310 into a paused state. At step 580 the pipeline manager 305 sends preempted CTAs and thread groups to the SMs 310 and the GPCs 208 restore the instruction level context state maintained in the SMs 310 for the selected context. After the CTA and instruction level state is restored, the pipeline managers 305 output a command to the respective SMs 310 indicating that the "preemption-restore-end" mode should be exited and, at step 582 the front end 212 negates the context freeze signal. Steps 580 and 582 may be performed simultaneously. At step 585 the CTAs are launched in the preempted order and at step 590 execution is resumed using the restored context state for the selected context. At step 590, the front end 212 also ACKs the host interface 206 to signal that the instruction level preemption command has completed execution. The Host interface 206 may now start sending more work from the pushbuffer to the front end 212. In one embodiment, the task/work unit 207 asserts and negates the context freeze and step 590 is performed (by the front end 212) after the context freeze is asserted in step 570. The task/work unit buffers the new work from the pushbuffer until the instruction level preemption command has completed execution. The new work is not output by the task/work unit until after the CTAs are launched in step **585**. As previously explained, the context state that is saved
> 
> and restored may be reduced at the expense of potentially longer latencies for stopping the running context by preempting at the CTA level instead of preempting at the instruction level. When a context is preempted at the CTA level the SMs 310 complete execution of any launched CTAs, so that there is not CTA state maintained within the pipeline managers 305 and GPCs 208 that needs to be stored.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Method
> 
> Figure 5B
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2022.png)

## Fig 6A **CTA level的context unload**

**CTA level的context unload：尝试等待**GPC下游中的任务完成，**timer超时**后转为指令level的context unload，即**中断/停止**SM内执行的指令（抢占SM）；

> **[图片提取文字 (image.png)]:**
> FIG. 6A illustrates an unload method 600 for unloading context state when a process is preempted at a CTA level, according to one embodiment of the invention. Although the method steps are described in conjunction with the systems of FIGS. 1, 2, 3A, 3B, and 4, persons skilled in the art will understand that any system configured to perform the method steps, in any order, is within the scope of the inventions.
> 
> At step 605 the host interface 206 outputs a CTA level preemption command to the front end 212 and the unloading of the current context is initiated. At step 610 the front end 212 determines if the processing pipeline is idle, and, if so, then the front end 212 proceeds directly to step 645 to store the context state that is maintained by the front end 212.
> 
> If, at step 610 the front end 212 determines that the processing pipeline is not idle, then at step 615 the front end 212 stops launching new work for the current context. At step 620 the front end 212 outputs a preempt command to the task/work unit 207. At step 625 the task management unit 300 within the task/work unit 207 stops issuing tasks to the work distribution unit 340 and outputs the preempt command to the work distribution unit 340. The work distribution unit 340 stops launching CTAs and, at step 630, the work distribution unit 340 waits for the GPCs 208 to become idle.
> 
> If, at step 630 the work distribution unit 340 determines that the GPCs 208 are not idle, then at step 635, the work distribution unit 340 determines if a timer has expired. The
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2023.png)

> **[图片提取文字 (image.png)]:**
> timer limits the number of clock cycles that the work distribution unit 340 will wait for the GPCs to become idle. The number of clock cycles may be a programmed value, and, in one embodiment, when the value is exceeded, the work distribution unit 340 performs preemption at the instruction level instead of at the CTA level. If, at step 635, the work distribution unit 340 determines that the timer has not expired, then the work distribution unit 340 returns to step 630. Otherwise, when the timer has expired, then the work distribution unit 340 proceeds to step 520 of FIG. 5A to perform preemption at the instruction level.
> 
> When at step 630 the GPCs 208 are idle, at step 640 the work distribution unit 340 saves the CTA level state that is maintained in the work distribution unit 340 for the current context. The work distribution unit 340 reports to the task management unit 300 the current state has been stored. The task management unit 300 then saves the task level state that is maintained in the task management unit 300. The task management unit 300 reports to the front end 212 when the current state has been stored, and at step 645 the front end 212 stores the context state that is maintained for the current context by the front end 212 to the context buffer. At step 650 the front end 212 then stores an indication that the saved context state is for a preempted context and resets the processing pipeline.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2024.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6A
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2025.png)

## Fig 6B **CTA level的context restore**

**CTA level的context restore**

> **[图片提取文字 (image.png)]:**
> At step 665 the front end 212 initiates restoration of a context that was previously preempted at the CTA level. At step 670 the front end 212 asserts the context freeze signal to ensure that the processing pipeline does not perform any operation based on the transactions used by the front end 212 to restore the context state. At step 675 the selected context state is read from a context buffer by the front end 212 and task/work unit 207, and restored at the task and CTA level. At step **682** the context freeze signal is deasserted. At step 685 the CTAs that were preempted the last time this context was running are relaunched by the task/work unit 207 into the GPCs 208. At step 690 the front end 212 ACKs the host interface 206 to signal that the CTA level preemption command has completed execution. The Host interface 206 may now start sending more work from the pushbuffer to the front end 212. In one embodiment, the
> 
> task/work unit 207 asserts and negates the context freeze and
> 
> step 690 is performed (by the front end 212) after the context
> 
> freeze is asserted in step 670. The task/work unit buffers the
> 
> new work from the pushbuffer until the instruction level
> 
> preemption command has completed execution. The new
> 
> work is not output by the task/work unit until after the CTAs
> 
> FIG. 6B illustrates a restore method 660 for restoring
> 
> context state when a process that was preempted at the CTA
> 
> level is restored, according to one embodiment of the
> 
> invention. Although the method steps are described in con-
> 
> junction with the systems of FIGS. 1, 2, 3A, 3B, and 4,
> 
> persons skilled in the art will understand that any system
> 
> configured to perform the method steps, in any order, is
> 
> within the scope of the inventions.
> 
> are relaunched in step 685.
> 
> stored.
> 
> The ability to preempt a context at either the instruction level or at the CTA level may be specified for each particular context. A long-running context may be preempted at the instruction level to avoid a long delay between when the preemption is initiated to when the preemption is completed. A context that is not necessarily long-running, but that maintains a large amount of state may be preempted at the
> 
> CTA level to minimize the amount of context state that is
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2026.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> ACK and resume execution 690
> 
> Figure 6B
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2027.png)

## Fig 3A task manager（软件辅助的指令level抢占）

Task MetaDada（TMD）是task manager调度任务的数据结构；

软件辅助的指令level抢占，是把**硬件控制**的SM内context的save/unload和restore/load过程，分别改为**preemption-save kernel和preemption-restore kernel的任务（软件辅助）**；

TMU将计算任务用scheduler table管理，scheduler table是很多TMD-group。

抢占时，执行save-kernel保存Ctx，save-krenl生成restore-kernel后插入**TMD-group**链表的head或tail，保证恢复被抢占的任务。

> **[图片提取文字 (image.png)]:**
> ## Referring back to FIG. 3A, the task management unit 300 manages compute tasks to be scheduled as an array of TMD
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2028.png)

> **[图片提取文字 (image.png)]:**
> group is a set of compute tasks with the same scheduling priority. The number of TMD groups, or priority levels, may be one or more. Within each TMD group, the compute tasks at the respective priority level are stored in a linked list. When compute tasks are received from the host interface 206 the task management unit 300 inserts the compute tasks into a TMD group. More specifically, a pointer to the TMD corresponding to the compute task is added to the tail of the linked list for that group unless a special TMD bit is set which causes the task to be added to the head of the linked list. Even though all tasks within a TMD group have the same scheduling priority level, the head of the TMD group linked list is the first compute task that is selected by the task management unit 300 and scheduled for execution. Thus, the compute task at the head of the linked list has a relatively higher priority compared with other compute tasks at the same priority level. Similarly, each successive compute task in the linked list at the same priority level as a lower priority relative to preceding compute tasks in the linked list. Therefore, the task management unit 300 is able to schedule the compute tasks within a TMD group in input order relative to one another (assuming none are specially marked to add to the head of the TMD group). Since the TMD group is specified as part of the TMD structure, the TMD group of a compute task cannot be changed while the compute task is being executed.
> 
> groups that are stored in the scheduler table 321. A TMD
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3A
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%203.png)

**硬件CTA** preemption后，GPC将save-kernel派发到SM执行，**SM强制跳转运行中CTA的PC到save-kernel**，保存SM的所有Ctx（save-kernel有最高权限），生成restore-kernel插入task group。

> **[图片提取文字 (image.png)]:**
> A software-assisted mechanism may be used in combination with the hardware implemented CTA level preemption to perform instruction level preemption. When CTA level preemption is performed instruction level context state that is maintained within the SMs 310 is not stored and the GPCs 208 and MPCs 415 are drained. The software-assisted instruction level preemption does store the instruction level context and allows for the preemption to occur quickly because the GPCs 208 and MPCs 415 are not drained. The software-assist mechanism exploits the capability of the GPCs 208 to generate child processing tasks that are output to the task/work unit 207 during execution of a processing task. A child processing task that is generated to perform the software-assisted instruction level preemption is encoded as preemption TMD and is added to the head of the linked list of the highest priority TMD group so that the preemption TMD will be scheduled and executed before any other TMDs. The preemption TMD includes a preemption-restore kernel and a pointer to a context buffer that stores the context state for the preempted context. A single preemption TMD is generated for all preempted CTAs that were launched to execute the same TMD. When CTAs for two different TMDs are executing, two preemption TMDs will be generated to preempt the two different contexts.
> 
> Before software-assisted instruction level preemption is performed, memory for the preemption TMD and a context buffer to store the instruction level context state are allocated by a program executed on the CPU 102. The purpose of the preemption TMD is to invoke a preemption-restore kernel for each CTA that was preempted. The size of the context buffer is known before preemption is initiated because the size is based on the processing pipeline, the number of SMs 310, and the size of a TMD that is executed for the particular context.
> 
> The software-assisted instruction level preemption is initiated by first performing CTA level preemption. When the preemption command is received by the GPCs **208**, the trap handler that includes the preemption-save kernel is executed for each CTA. When the preemption-save kernel is executed for a CTA, the context state maintained for execution of the
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2030.png)

> **[图片提取文字 (image.png)]:**
> necessary to restore the instruction level state for the preempted CTA is stored in the portion of the context state buffer. The preemption-save kernel may access the instruction level state by reading architectural registers, reading state from memory-mapped registers, execution of special instructions, or by a combination of instruction execution and memory accesses performed by an SM 310. The data necessary to restore the instruction level state for the context may include the starting program counter of the preemption restore program instructions, a pointer to the portion of the context state buffer where the context state for the particular CTA is stored, constant buffer values, a number of registers used by the CTA, an amount of shared memory used by the CTA, and the like.
> 
> CTA is stored in a portion of the context buffer. The data
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2031.png)

> **[图片提取文字 (image.png)]:**
> Once the preemption-save kernel has stored the instruction level context state, the preemption-save kernel notifies the task/work unit 207 that a child processing task, or, more specifically, a preemption TMD has been generated. The preemption TMD is not immediately scheduled by the task/work unit 207 because the task/work unit 207 has received the preempt command and is performing the preemption operations. However, the preemption TMD is stored in a buffer along with any other additional work transmitted to the task/work unit 207 after the preempt command was received. The preemption TMD is stored by
> 
> the front end 212 as part of the context state for the
> 
> instruction that causes the CTA to exit the SM 310, so that
> 
> Finally, the preemption-save kernel ends with an exit
> 
> task/work unit 207.
> 
> any other TMD.
> 
> the SM 310 becomes idle and appears to have drained as expected during normal CTA level preemption. The preemption-restore kernel performs restoration of the instruction level context state when a context that was preempted is resumed. When restoring the instruction level context state, the CTA level context state, including the preemption TMD, is first restored. Then, because the preemption TMD generated by the preemption-save kernel during the preemption process has a highest priority level, the preemption TMD is scheduled for execution by the task/work unit 207 before
> 
> TMD is executed by the SMs 310 and the instruction level state stored in the context buffer for each CTA is restored by the respective SM 310. The preemption-restore kernel also sets up the call stack such that an AtExit routine is invoked by the CTA when the CTA exits. After setting up the call stack, the preemption-restore kernel execution is complete and the CTA proceeds to execute user code at the point where the CTA was preempted. When the CTA runs to the normal exit point, The AtExit routine is invoked to free the preemption related resources, such as the memory allocated for storing the TMD and the portion of the context state buffer allocated for each CTA. After the AtExit routine is executed, the CTA exits.
> 
> The preemption-restore kernel included in the preemption
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2032.png)

## Fig 7A **preemption save kernel**

**preemption save kernel**完成指令level context unload，其中**生成preemption-restore kernel**作为最**高优先级**任务，插入到front end（TMDs）和其他任务一起**保存**；

> **[图片提取文字 (image.png)]:**
> according to one embodiment of the invention. Although the method steps are described in conjunction with the systems of FIGS. 1, 2, 3A, 3B, and 4, persons skilled in the art will understand that any system configured to perform the method steps, in any order, is within the scope of the inventions. As previously explained, when a context is initialized, a context buffer is allocated and when software-assisted instruction level preemption is enabled for the context, a
> 
> preemption TMD is allocated for use by the preemption-
> 
> FIG. 7A illustrates a software assisted unload method 700
> 
> for unloading context state when a process is preempted,
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2033.png)

> **[图片提取文字 (image.png)]:**
> front end 212 and the unloading of the current context is initiated. At step 710 the front end 212 determines if the processing pipeline is idle, and, if so, then the front end 212 proceeds directly to step 750 to store the context state that is maintained by the front end 212. If, at step 710 the front end 212 determines that the processing pipeline is not idle, then at step 715 the front end 212 stops launching new work for the current context. At step 720 the front end 212 outputs a preempt command to the task/work unit 207. At step 725 the task/work unit 207 stops issuing instructions to the CWD 410 and outputs the preempt command to the CWD 410. The CWD 410 stops launching CTAs and, at step 730, the CWD 410 waits for the GPCs 208 to become idle and the preemption-save kernel is invoked by the CTAs executing on the SMs 310. The preemption-save kernel is executed by each CTA. At step 735 the preemption-save kernel first allocates a portion of the context buffer for storing the instruction level context state that is maintained within the respective SM 310. At step 740 the preemption-save kernel saves the instruction level context state in the portion of the context state buffer. Steps 730, 735, and 740 are performed for each CTA that is being executed by an SM 310. At step 742 the preemption-save kernel sets the priority level of the preemption TMD to the highest priority level. At step 745 the preemption-save kernel notifies the task/work unit 307 that a preemption TMD was generated and the preemption-save kernel executes an exit instruction, causing each CTA to exit the SM 310 and idle the GPCs 208 and MPCs **415**. When the GPCs 208 are idle, at step 748 the CWD 410 saves the CTA level state that is maintained in the CWD 410
> 
> save and preemption-restore kernels. The preemption-save
> 
> and preemption-restore kernels and the AtExit routine are
> 
> stored in memory. A program is then executed by the
> 
> processing pipeline using the context. At step 705 the host
> 
> interface 206 outputs a CTA level preempt command to the
> 
> maintained for the current context by the front end 212 to the context buffer. At step 755 the front end 212 then stores an indication that the saved context state is for a preempted context and resets the processing pipeline.
> 
> for the current context. The CWD 410 reports to the front
> 
> end 212 when the current state has been stored, and at step
> 
> 750 the front end 212 stores the context state that is
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2035.png)

## Fig 7B **preemption-restore kernel**

**preemption-restore kernel**完成的context ****restore；

> **[图片提取文字 (image.png)]:**
> for restoring context state when a preempted process is restored, according to one embodiment of the invention. Although the method steps are described in conjunction with the systems of FIGS. 1, 2, 3A, 3B, and 4, persons skilled in the art will understand that any system configured to perform the method steps, in any order, is within the scope of the inventions.
> 
> At step 765 the front end 212 initiates restoration of a system that we are restoration as forces.
> 
> FIG. 7B illustrates a software-assisted restore method 760
> 
> context that was previously preempted using software-assisted instruction level preemption. At step 770 the front end 212 asserts the context freeze signal to ensure that the processing pipeline does not perform any operation based on the transactions used by the front end 212 to restore the context state. At step 775 the selected context state is read from a context buffer by the front end 212, task/work unit 207, and CWD 410 and restored at the CTA level.
> 
> At step 780 the CTAs are launched in the preempted order.
> 
> At step 785 the front end 212 negates the context freeze signal and execution is resumed using the restored CTA level context state for the selected context. The front end 212 also ACKs the host interface 206 to signal that the CTA level
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2036.png)

> **[图片提取文字 (image.png)]:**
> preemption command has completed execution. Note, that the instructions level context state is not yet restored.
> 
> At step 790 the task/work unit 207 schedules the preemption TMD for execution and each of the CTAs invokes the preemption-restore kernel. At step 795 the instruction level context state that was stored for each CTA is read and restored to the respective SM 310 for use by the CTA. At step 797 the execution stack is set up for the AtExit routine so that the memory allocated for the preemption TMD and context buffer will be deallocated (or freed). At step 797 the preemption-restore kernel also exits and the user code resumes execution using the restored context. When the user code reaches the normal exit point, the AtExit routine is invoked and the preemption TMD and context buffer are freed when the last CTA exits.
> 
> The ability to preempt a context at either the instruction level or at the CTA level may be specified for each particular context. A long-running context may be preempted at the instruction level to avoid a long delay between when the preemption is initiated to when the preemption is completed. A context that is not necessarily long-running, but that maintains a large amount of state may be preempted at the CTA level to minimize the amount of context state that is stored.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Method
> 
> 760
> 
> Figure 7B
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2038.png)

# 其他Preemption案例

ref：TECHNIQUES FOR MODIFYING EXECUTABLE GRAPHS TO PERFORM DIFFERENT WORKLOADS

> **[图片提取文字 (image.png)]:**
> [0120] In at least one embodiment, when host interface 1206 receives a command buffer via I/O unit 1204, host interface 1206 can direct work operations to perform those commands to a front end 1208. In at least one embodiment, front end 1208 couples with a scheduler 1210, which is configured to distribute commands or other work items to a processing array 1212. In at least one embodiment, scheduler 1210 ensures that processing array 1212 is properly configured and in a valid state before tasks are distributed to processing array 1212. In at least one embodiment, scheduler 1210 is implemented via firmware logic executing on a microcontroller. In at least one embodiment, microcontroller implemented scheduler 1210 is configurable to perform complex scheduling and work distribution operations at coarse and fine granularity, enabling rapid preemption and context switching of threads executing on processing array **1212**. In at least one embodiment, host software can prove workloads for scheduling on processing array 1212 via one of multiple graphics processing doorbells. In at least one embodiment, workloads can then be automatically distributed across processing array 1212 by scheduler 1210 logic within a microcontroller including scheduler 1210.
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 12A
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2040.png)

ref：PREEMPTION IN A MACHINE LEARNING HARDWARE ACCELERATOR

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2042.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2043.png)

中低优先级设置**CLIP或GFXP**，**时间片内允许抢占**，设置较短时间片来及时释放资源。

高优先级任务设置**WFI**，保证**时间片内不被抢占**，设置较长的时间片，保证一次kernel调用完成。

**辨析wait-for-idle**指令： **wait-for-idle**是应用卸载到GPU任务的指令流里的一种**常规同步指令**。要求让前端暂停调度该进程任务的后续任务，直到资源空闲。**wait-for-idle**常用于低优先级任务，让任务执行到wait-for-idle后**主动“休息”**，**不和高优先级任务争抢资源**。

应用Preemption类型包括WFI、CLIP、GFXP，决定channel在时间片内是否被抢占。

**抢占类型**

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
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 8
![image.png](GPU%E7%9A%84%E6%8A%A2%E5%8D%A0Preemption%E5%92%8CTimeslice%EF%BC%88WFI%20drain%E3%80%81switch%EF%BC%89/image%2048.png)

SM内CTA以warp为单位调度执行，不同warp轮流发射指令，warp内线程按SIMT-pipeline并行。

GPC的pipeline-manager将packet派发到有CTA并发余量的SM，或其他Engine。

**TMU**对TMD指针进行任务管理、调度执行和派发资源。

任务管理：不同优先级分为不同TMD group，每个TMD group维护**多个队列，支持GPU程序的嵌套执行、同步和抢占恢复等机制**。

调度任务：按照队列内串行、队列间并发，和队列中任务内容（wait、signal、grid）定义的同步依赖和队列优先级调度任务。

派发资源：为grid分配CTA，和CTA在grid中位置打包成**packet**，派发到有CTA余量的GPC。

空闲SM：允许SCG-SM申请资源正交的任务，图形任务直接调度，计算任务需要TMU调度。

**CGA机制**：按照SM-CTA/GPC-CTA/GPU-CTA粒度分配资源，并派发到SM/GPC/uGPU范围内的SM。

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

**前端**控制GPU的抢占过程，完成后将TMD指针发到TMU。

**主机接口**传递host抢占命令（指定channel或runlist切换）或计时器抢占命令（TS超时切换），并从pushbuffer中读取新channel中的TMD指针后传递给前端。

**runlist**是host定义的channel队列，runlist中channel在GPU中运行分配时间片TS后切换，**interleave freq**越大的channel在runlist中出现频率更高，对应**高优先级应用**。

**channel**是应用卸载任务TMDs的通道，是GPU执行任务的指针**集合**（kernel指令起始地址等），每个TMD可以指定执行优先级（不同kernel优先级），一个应用使用1个或多个channel来卸载任务。

**channel、runlist和TS抢占**

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