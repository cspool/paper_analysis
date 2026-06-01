# NV 聚合Barrier的分支执行

ref:[EXECUTION OF DIVERGENT THREADS USING A CONVERGENCE BARRIER]

**4中分支同步机制(运行栈和pop-syn)决定分支线程的静态调度和同步策略。warp内不同分支线程的切换以指令块为单位，即分支1的指令块完成后才能切换分支2。**

**5和4一致的策略是尽早同步分束线程,但将分支执行中最重要的同步通过聚合屏障实现,从而将分支执行的机制和线程调度策略解耦合。**

**vertice是图形的顶点,最基本的图元;**

**primitive是图形的几何图元(顶点、线段、三角形、圆等元素),图形的基本组成;**

**fragment是片元,光栅化的产物,是图元转变成的二维图像的像素点,每个像素包含颜色、深度、纹理;**

**pixel是屏幕上图像的像素,包含像素点填充的色彩信息(RGB).**

## 分束线程的同步

**聚合barrier将相同分支的线程动态分组,提高SIMD单元的利用率.**

**多线程并发的程序执行需要支持线程分束执行,提供YIELD指令让循环/不定指令数的线程挂起保证线程不会永远阻塞其余线程.**

**聚合barrier将分束的异步线程同步,比基于运行栈的静态优先级的异步线程调度和同步更加灵活.**

Convergence Barriers attempt to maintain high SIMD efficiency by keeping threads that take the same paths through a program grouped together for parallel execution of instructions. In the context of the following description, a thread refers to an agent that executes a task defined by a sequence of instructions. Concurrent tasks are often said to execute on "threads" to execute statements in sequence. Divergence is supported during execution of the program, and a YIELD mechanism provides a forward progress guarantee to threads such that no thread can indefinitely block the execution of any other thread. The barriers are termed "convergence barriers" rather than simply "barriers" to indicate that the convergence barriers are used to join divergent groups of threads back together (i.e., they are used to keep threads converged on SIMD hardware). Using convergence barriers to implement a flexible thread synchronization technique is substantially different than prior art stack-based and priority-based schemes.

**基于聚合Barrier的分支管理机制和线程调度机制解耦,两个机制独立工作.**

**分支管理是指分支切换,线程调度是指线程切换.**

**分支运行栈机制中,分束切换的实现机制决定了warp中线程的静态切换策略.即先执行分支A的代码块(分束线程A),阻塞后执行分支B的代码块(分束线程B),最后同步.**

**聚合Barrier机制中,只要求分束线程在聚合处同步,即分支A的指令(分束线程A)和分支B的指令(分束线程B)在到达Barrier之前可任意交错执行,在Barrier处同步.**

**聚合Barrier机制中,将warp和分束线程都视为线程组来调度,线程的切换策略和分支切换实现无关.**

In one embodiment, the divergence management mechanism that relies on the convergence barriers is decoupled from the thread scheduling mechanism. Therefore, the thread scheduling mechanism may be changed without changing the divergence management mechanism. Similarly, the divergence management mechanism may be changed without changing the thread scheduling mechanism.

**编译时识别出程序中单出入口的代码块,该代码块内部可能出现分束.**

**在代码块的单出入口指令处,插入聚合barrier让线程同步.即离开代码块前将分束的线程同步,以同步进入下一个代码块.**

**单出入口代码块的入口可能对应程序的有向控制流图中1个支配节点,出口可能对应程序的有向控制流图中1个尾支配节点.**

In one embodiment, a compiler is configured to analyze an application program, identify regions of the program having a single entry point and a single exit point, and insert convergence barrier instructions to synchronize threads that may diverge within each region. In the context of the following description, the single entry point may correspond to a dominator node of a directed control-flow graph of the program, and the single exit point may correspond to a post-dominator node of the directed control-flow graph of the program.

**执行程序中entry处指令的线程会参加聚合Barrier,SM中的scheduler单元记录聚合Barrier和参加线程.**

**该过程的实现机制是,编译时在entry处指令前插入ADD指令,ADD指令指明聚合barrier的名字,执行ADD指令的线程加入该命名的聚合Barrier.每个命名的聚合Barrier对应1个标记向量寄存器,标记参与Barrier的线程.**

**ADD对应WAIT指令,WAIT指令指明聚合barrier的名字,表示参与该命名聚合屏障的线程在此同步.**

FIG. 1 illustrates a flowchart of a method 100 for executing divergent threads using a convergence barrier, in accordance with one embodiment. At step 110, a plurality of threads execute a first instruction at an entry point in a program, where the first instruction, when executed by a particular thread, indicates to a scheduler unit that the thread participates in a convergence barrier. In other words, execution of the first instruction by a thread "adds" the thread to the convergence barrier. In one embodiment, an ADD instruction may be inserted into the program by a compiler. The ADD instruction may specify a convergence barrier name, and each thread that executes the ADD instruction participates in the named convergence barrier. In one embodiment, a multi-bit register may correspond to each convergence barrier name, and a bit is assigned for each thread that may participate in the convergence barrier. When a thread executes the instruction that specifies a convergence barrier, the bit assigned to the thread is set in the multi-bit register. The convergence barrier is represented by a WAIT instruction that may also be inserted into the program by the compiler. The WAIT instruction also specifies the name used by the ADD instruction. The WAIT instruction is usually a convergence point for various divergent code paths that synchronize on a specific barrier.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2033.png)

**程序中不同分束的线程指令分别在不同的SIMD单元上并行执行,而不是基于运行栈的调度串行.但资源使用率并没有提高,只是不同分支的线程调度更灵活,能让相同指令的线程更早同步.**

At step 115, a first path through the program is executed by a first divergent portion of the participating threads (e.g., threads that participate in the convergence barrier). The first path through the program may be different than one or more other paths taken by other threads of the participating threads. Each different path through the region of the program is a divergent path.

At step 120, a second path through the program is executed by a second divergent portion of the participating threads that are ready for execution. The second divergent portion of the participating threads is different than the first divergent portion of the participating threads. In one embodiment, **the first divergent portion of the participating threads is executed by a first core within a parallel processing unit, and the second divergent portion of the participating threads is simultaneously executed by a second core within the parallel processing unit.** Additional paths through the program may be executed by other divergent portions of the participating threads. In the extreme, each participating thread may execute a different path through the program so that for N threads, the execution is N-way divergent.

**分束的线程执行退出点指令,线程状态从ready变成blocked.**

At step 125, the first divergent portion of the participating threads executes a second instruction at an exit point in the program, where the second instruction, when executed by a particular thread, causes the particular thread to transition to a blocked state. In one embodiment, a state of the first divergent portion of the participating threads changes from ready (i.e., ready for execution) to blocked when the second instruction is executed. In one embodiment, the second instruction is a WAIT instruction that specifies the convergence barrier matching the name specified by the ADD instruction.

**分束的线程执行WAIT指令,并且WAIT指定的聚合Barrier和线程执行ADD时参与的Barrier名字相同,线程状态从ready变成blocked.**

If the second divergent portion of the participating threads executes the WAIT instruction that specifies the convergence barrier matching the name specified by the ADD instruction, execution is stopped and the state of the threads in the second divergent portion of the participating threads changes from ready to blocked. In one embodiment, the second divergent portion does not necessarily execute the second instruction and instead executes a third instruction. The third instruction may be either an OPT-OUT instruction or a YIELD instruction. In one embodiment, the OPT-OUT instruction is functionally equivalent to a Break instruction.

**SM中的scheduler通过命名Barrier的参与线程是否都到达同Barrier名的WAIT指令,即处于blocked状态,来检测聚合Barrier处线程是否达到同步.**

At step 130, a scheduler unit determines if the participating threads are synchronized at the convergence barrier. In the context of the following description, the scheduler unit may be implemented as circuitry and included within a multi-threaded execution unit, such as a streaming multiprocessor. In one embodiment, the scheduler unit determines that the participating threads are synchronized when all of the participating threads have reached the second instruction (e.g., WAIT instruction) that specifies the convergence barrier matching the name specified by the first instruction (e.g., ADD instruction) and determines that all of the participating threads are blocked.

**程序可能有多个退出点(break,return),即程序中指令块可能是单入口多出口的.**

**线程退出程序段后不再参与聚合Barrier,状态可能变成exited,可能不变.**

In one embodiment, one or more threads of the participating threads may have exited the program and are no longer considered to be included in the participating threads (i.e., exited threads are removed from the participating threads). In other words, in one embodiment, a region of the program may have a single entry and multiple exits. In one embodiment, threads may exit the region of the program when an OPT-OUT instruction is executed. The OPT-OUT instruction may be inserted into the program by the compiler. In one embodiment, a state of threads that execute an OPT-OUT instruction is changed from ready to exited. In another embodiment, the state of threads that execute an OPT-OUT instruction is unchanged and is maintained as ready.

**线程执行YIELD指令或线程满足YIELD条件(超时)时,状态变成yielded(检测同步时等效blocked).**

**YIELD指令可能被编译器插入,让达到聚合Barrier的线程休眠,从而增大调度未达到聚合Barrier线程的机会.**

In one embodiment, one or more threads of the participating threads may be disabled when the one or more threads execute a YIELD instruction. In one embodiment, one or more threads of the participating threads may be disabled when a YIELD condition is met even though a YIELD instruction has not been executed. The scheduler unit may determine a YIELD condition is met when a timeout event occurs or based on characteristics of execution, such as a certain number of backward branches being executed. The YIELD instruction may be inserted into the program by the compiler for threads that may not arrive at the convergence barrier. In one embodiment, participating threads that are disabled may be ignored for the synchronization determination. In other words, participating threads that are disabled may be treated as blocked at step 130. In one embodiment, a state of threads that execute a YIELD instruction is changed from ready to yielded.

**scheduler检测到聚合Barrier中全部线程达到同步,则清除Barrier,并唤醒Barrier处全部线程.聚合Barrier处的线程成组,在SIMD单元上同步执行.**

If, at step 130, the scheduler unit determines that the participating threads are synchronized at the convergence barrier, then at step 140, the convergence barrier is cleared and all threads that participated in the convergence barrier are released, i.e., unblocked. In one embodiment, participating threads that are in the yielded state are changed to the ready state. In other words, yielded threads are cleared (i.e., bits corresponding to yielded threads are cleared in the multi-bit register) when the convergence barrier is cleared. When the convergence barrier clears, all threads that were blocked on the convergence barrier will be grouped together and resume execution in SIMD fashion.

**scheduler检测到聚合Barrier处线程未达到同步,则将参与线程中的剩余分束调度执行.**

If, at step 130, the scheduler unit determines that the participating threads are not synchronized at the convergence barrier, then at step 135 the scheduler unit may execute an additional path (e.g., third, fourth... path) through the program by an additional divergent portion of the participating threads (e.g., third, fourth... Nth divergent portion of the participating threads), and return to step 130. The additional path may be different than either the first path or the second path. The additional divergent portion of the participating threads may be different than the first and second divergent portions of the participating threads.

## PPU/GPU的组成

**并行处理器PPU通过多线程并发来隐藏访存延迟,比如GPU.**

**GPU被设计成处理三维图形graphic数据的图形处理pipeline,来生成显示设备上的二维图像image数据.GPU可用于处理高并发的通用计算任务(GPGPU).**

FIG. 2 illustrates a parallel processing unit (PPU) 200, in accordance with one embodiment. In one embodiment, the PPU 200 is a multi-threaded processor that is implemented on one or more integrated circuit devices. The PPU 200 is a latency hiding architecture designed to process a large number of threads in parallel. A thread (i.e., a thread of execution) is an instantiation of a set of instructions configured to be executed by the PPU 200. In one embodiment, the PPU 200 is a graphics processing unit (GPU) configured to implement a graphics rendering pipeline for processing three-dimensional (3D) graphics data in order to generate two-dimensional (2D) image data for display on a display device such as a liquid crystal display (LCD) device. In other embodiments, the PPU 200 may be utilized for performing general-purpose computations. While one exemplary parallel processor is provided herein for illustrative purposes, it should be strongly noted that such processor is set forth for illustrative purposes only, and that any processor may be employed to supplement and/or substitute for the same.

**PPU包含I/O接口、host接口、前端单元、调度单元、任务分派单元、hub、互联单元Xbar、并行处理GPC、分区单元.**

**PPU的I/O接口通过系统总线和host或其他设备建立联系,通过分区单元连接存储设备(显存).**

As shown in FIG. 2, the PPU 200 includes an Input/Output (I/O) unit 205, a host interface unit 210, a front end unit 215, a scheduler unit 220, a work distribution unit 225, a hub 230, a crossbar (Xbar) 270, one or more general processing clusters (GPCs) 250, and one or more partition units 280. The PPU 200 may be connected to a host processor or other peripheral devices via a system bus 202. The PPU 200 may also be connected to a local memory comprising a number of memory devices 204. In one embodiment, the local memory may comprise a number of dynamic random access memory (DRAM) devices.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2034.png)

**I/O接口用于PPU和host或各种设备通过系统总线或其他设备进行通信.**

The I/O unit 205 is configured to transmit and receive communications (i.e., commands, data, etc.) from a host processor (not shown) over the system bus 202. The I/O unit 205 may communicate with the host processor directly via the system bus 202 or through one or more intermediate devices such as a memory bridge. In one embodiment, the I/O unit 205 implements a Peripheral Component Interconnect Express (PCIe) interface for communications over a PCIe bus. In alternative embodiments, the I/O unit 205 may implement other types of well-known interfaces for communicating with external devices.

**I/O接口将从host收到的数据包发给host接口译码,得到配置PPU的命令.host接口将host对PPU的配置命令传送到PPU的不同部分.**

The I/O unit 205 is coupled to a host interface unit 210 that decodes packets received via the system bus 202. In one embodiment, the packets represent commands configured to cause the PPU 200 to perform various operations. The host interface unit 210 transmits the decoded commands to various other units of the PPU 200 as the commands may specify. For example, some commands may be transmitted to the front end unit 215. Other commands may be transmitted to the hub 230 or other units of the PPU 200 such as one or more copy engines, a video encoder, a video decoder, a power management unit, etc. (not explicitly shown). In other words, the host interface unit 210 is configured to route communications between and among the various logical units of the PPU 200.

**host端程序将对PPU的任务负载编码成命令流(PPU指令和数据),存储在共享buffer供PPU执行负载.**

**流是一系列按顺序执行的命令,比如线程.不同的流可能无序或并发地执行命令,来提高资源利用率.**

**PPU中的前端根据host传递的命令指针去buffer中获取指令和数据,然后发送到PPU的不同部件.**

In one embodiment, a program executed by the host processor encodes a command stream in a buffer that provides workloads to the PPU 200 for processing. A workload may comprise a number of instructions and data to be processed by those instructions. The buffer is a region in a memory that is accessible (i.e., read/write) by both the host processor and the PPU 200. For example, the host interface unit 210 may be configured to access the buffer in a system memory connected to the system bus 202 via memory requests transmitted over the system bus 202 by the I/O unit 205. In one embodiment, the host processor writes the command stream to the buffer and then transmits a pointer to the start of the command stream to the PPU 200. The host interface unit 210 provides the front end unit 215 with pointers to one or more command streams. The front end unit 215 manages the one or more streams, reading commands from the streams and forwarding commands to the various units of the PPU 200.

**前端将命令流定义的任务负载发送到scheduler进行管理和调度,scheduler管理任务的状态(派发执行的GPC编号、任务的执行状态、调度优先级等).**

The front end unit 215 is coupled to a scheduler unit 220 that configures the various GPCs 250 to process tasks defined by the one or more streams. The scheduler unit 220 is configured to track state information related to the various tasks managed by the scheduler unit 220. The state may indicate which GPC 250 a task is assigned to, whether the task is active or inactive, a priority level associated with the task, and so forth. The scheduler unit 220 manages the execution of a plurality of tasks on the one or more GPCs 250.

**scheduler选择一堆任务启动并发,发送给任务分派单元distribution进行任务派发.**

**distribution单元为每个GPC设置1个pending池和1个active池来完成任务的并发,pending池记录ready的任务,active池running的任务.**

**scheduler调度、启动和管理线程的并发,distribution和GPC执行和完成线程的并发,scheduler和distribution是分层的控制设计.**

The scheduler unit 220 is coupled to a work distribution unit 225 that is configured to dispatch tasks for execution on the GPCs 250. The work distribution unit 225 may track a number of scheduled tasks received from the scheduler unit 220. In one embodiment, the work distribution unit 225 manages a pending task pool and an active task pool for each of the GPCs 250. The pending task pool may comprise a number of slots (e.g., 16 slots) that contain tasks assigned to be processed by a particular GPC 250. The active task pool may comprise a number of slots (e.g., 4 slots) for tasks that are actively being processed by the GPCs 250. As a GPC 250 finishes the execution of a task, that task is evicted from the active task pool for the GPC 250 and one of the other tasks from the pending task pool is selected and scheduled for execution on the GPC 250. If an active task has been idle on the GPC 250, such as while waiting for a data dependency to be resolved, then the active task may be evicted from the GPC 250 and returned to the pending task pool while another task in the pending task pool is selected and scheduled for execution on the GPC 250.

**Xbar是片上互联网络,完成PPU上不同部件的通信/路由.**

The work distribution unit 225 communicates with the one or more GPCs 250 via an XBar 270. The XBar 270 is an interconnect network that couples many of the units of the PPU 200 to other units of the PPU 200. For example, the XBar 270 may be configured to couple the work distribution unit 225 to a particular GPC 250. Although not shown explicitly, one or more other units of the PPU 200 are coupled to the host unit 210. The other units may also be connected to the XBar 270 via a hub 230.

**scheduler管理任务,distribution分派、切换任务到GPC执行以并发.**

**GPC的处理结果可能通过片上网络Xbar发送给其他GPC计算,也可能通过分区单元存到内存供后续计算.分区单元数量等于内存芯片数量.**

The tasks are managed by the scheduler unit 220 and dispatched to a GPC 250 by the work distribution unit 225. The GPC 250 is configured to process the task and generate results. The results may be consumed by other tasks within the GPC 250, routed to a different GPC 250 via the XBar 270, or stored in the memory 204. The results can be written to the memory 204 via the partition units 280, which implement a memory interface for reading and writing data to/from the memory 204. In one embodiment, the PPU 200 includes a number U of partition units 280 that is equal to the number of separate and distinct memory devices 204 coupled to the PPU 200.

## PPU/GPU的图形处理

**host处理器执行PPU的（图形）驱动内核,内核提供给host程序将操作卸载到PPU上执行的编程接口(API).**

**host程序通过API调用驱动，编译生成执行任务（图形计算）的PPU配置命令,并流式输出到PPU.**

**驱动内核配置GPU来执行图形任务，封装成接口，上层应用/OS调用驱动接口，完成图形加速。**

**驱动完成的任务类似cuda中的kernel program定义的任务，但抽象层次不同。驱动编写指令并提供执行图形计算的API；kernel基于runtime API在运行时执行kernel编译后的指令（编译时可能使用驱动API），编程不到指令级别。**

**cuda中kernel和host程序在编译时，host程序将对kernel函数的调用转为对runtime中API的调用，并将host、kernel、runtime编译（汇编）、链接后生成host的可执行文件。runtime负责运行时将host的请求转为对GPU的配置命令，如数据和指令（一般编译时生成）的加载、启动、结束等。**

**每个任务包含多个线程组,即多个warp.线程块是执行任务的若干线程组/warp.同组/warp的线程通过共享存储交换数据.每个warp一般包含32个线程.**

In one embodiment, a host processor executes a driver kernel that implements an application programming interface (API) that enables one or more applications executing on the host processor to schedule operations for execution on the PPU 200. An application may generate instructions (i.e., API calls) that cause the driver kernel to generate one or more tasks for execution by the PPU 200. The driver kernel outputs tasks to one or more streams being processed by the PPU 200. Each task may comprise one or more groups of related threads, referred to herein as a warp. A thread block may refer to a plurality of groups of threads including instructions to perform the task. Threads in the same group of threads may exchange data through shared memory. In one embodiment, a group of threads comprises 32 related threads.

**GPC包含pipeline管理单元、pre-raster操作单元(PROP)、Raster引擎、任务分派路由WDX、内存管理单元和若干Texture Processing Cluster.**

FIG. 3 illustrates a GPC 250 of the PPU 200 of FIG. 2, in accordance with one embodiment. As shown in FIG. 3, each GPC 250 includes a number of hardware units for processing tasks. In one embodiment, each GPC 250 includes a pipeline manager 310, a pre-raster operations unit (PROP) 315, a raster engine 325, a work distribution crossbar (WDX) 380, a memory management unit (MMU) 390, and one or more Texture Processing Clusters (TPCs) 320. It will be appreciated that the GPC 250 of FIG. 3 may include other hardware units in lieu of or in addition to the units shown in FIG. 3.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2035.png)

**GPC中pipeline控制器管理被分配的任务在TPC(SM、图元引擎)或其他模块(PROP、Raster Engine)的执行.**

In one embodiment, the operation of the GPC 250 is controlled by the pipeline manager 310. The pipeline manager 310 manages the configuration of the one or more TPCs 320 for processing tasks allocated to the GPC 250. In one embodiment, the pipeline manager 310 may configure at least one of the one or more TPCs 320 to implement at least a portion of a graphics rendering pipeline. For example, a TPC 320 may be configured to execute a vertex shader program on the programmable streaming multiprocessor (SM) 340. The pipeline manager 310 may also be configured to route packets received from the work distribution unit 225 to the appropriate logical units within the GPC 250. For example, some packets may be routed to fixed function hardware units in the PROP 315 and/or raster engine 325 while other packets may be routed to the TPCs 320 for processing by the primitive engine 335 or the SM 340.

**PROP将经过Raster引擎光栅化和TPC着色的数据进行优化(色彩融合、地址变换)后路由到分层模块的ROP单元.**

The PROP unit 315 is configured to route data generated by the raster engine 325 and the TPCs 320 to a Raster Operations (ROP) unit in the partition unit 280, described in more detail below. The PROP unit 315 may also be configured to perform optimizations for color blending, organize pixel data, perform address translations, and the like.

**Raster引擎执行光栅化操作(三维建模数据转二维屏幕数据),包含若干功能单元.**

**Raster引擎将转换后的顶点(vertices)数据进行一系列处理后,输出片元(fragment)到TPC中的fragment shader(Texture unit + SM).**

The raster engine 325 includes a number of fixed-function hardware units configured to perform various **raster operations**. In one embodiment, the raster engine 325 includes a setup engine, a coarse raster engine, a culling engine, a clipping engine, a fine raster engine, and a tile coalescing engine. The setup engine receives transformed vertices and generates plane equations associated with the geometric primitive defined by the vertices. The plane equations are transmitted to the coarse raster engine to generate coverage information (e.g., an x,y coverage mask for a tile) for the primitive. The output of the coarse raster engine may be transmitted to the culling engine where fragments associated with the primitive that fail a z-test are culled, and transmitted to a clipping engine where fragments lying outside a viewing frustum are clipped. Those fragments that survive clipping and culling may be passed to a fine raster engine to generate attributes for the pixel fragments based on the plane equations generated by the setup engine. The output of the raster engine 380 comprises fragments to be processed, for example, by a fragment shader implemented within a TPC 320.

**TPC包含图元引擎(primitive)、SM、Texture单元和多Pipe控制器MPC.**

**MPC将数据和操作分发到对应的执行单元,如将vertices数据路由到图元引擎得到相关属性,将需要着色(shader)的数据交给SM执行.**

Each TPC 320 included in the GPC 250 includes an M-Pipe Controller (MPC) 330, a primitive engine 335, an SM 340, and one or more texture units 345. The MPC 330 controls the operation of the TPC 320, routing packets received from the pipeline manager 310 to the appropriate units in the TPC 320. For example, packets associated with a vertex may be routed to the primitive engine 335, which is configured to fetch vertex attributes associated with the vertex from the memory 204. In contrast, packets associated with a shader program may be transmitted to the SM 340.

**SM中的texture单元加载texture map(纹素阵列)并取样,采样的texture数据在SM中进行着色.**

In one embodiment, the texture units 345 are configured to load texture maps (e.g., a 2D array of texels) from the memory 204 and sample the texture maps to produce sampled texture values for use in shader programs executed by the SM 340. The texture units 345 implement texture operations such as filtering operations using mip-maps (i.e., texture maps of varying levels of detail). In one embodiment, each TPC 320 includes four (4) texture units 345.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_2.jpeg)
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/496b34b0-2d74-4e84-9319-fd2c8b56eb30.png)

**SM由大量的可编程SP/pipeline组成,以SIMT的模式并发执行大量线程.**

The SM 340 comprises a programmable streaming processor that is configured to process tasks represented by a number of threads. Each SM 340 is multi-threaded and configured to execute a plurality of threads (e.g., 32 threads) from a particular group of threads concurrently. In one embodiment, the SM 340 implements a SIMD (Single-Instruction, Multiple-Data) architecture where each thread in a group of threads (i.e., a warp) is configured to process a different set of data based on the same set of instructions. All threads in the group of threads execute the same instructions. In another embodiment, the SM 340 implements a SIMT (Single-Instruction, Multiple Thread) architecture where each thread in a group of threads is configured to process a different set of data based on the same set of instructions, but where individual threads in the group of threads are allowed to diverge during execution. In other words, when an instruction for the group of threads is dispatched for execution, some threads in the group of threads may be active, thereby executing the instruction, while other threads in the group of threads may be inactive, thereby performing a no-operation (NOP) instead of executing the instruction. The SM 340 may be described in more detail below in conjunction with FIG. 4.

**MMU作为GPC和分区模块的接口,提供虚拟地址到物理地址变换、内存保护和裁决内存访问的功能.**

The MMU 390 provides an interface between the GPC 250 and the partition unit 280. The MMU 390 may provide translation of virtual addresses into physical addresses, memory protection, and arbitration of memory requests. In one embodiment, the MMU 390 provides one or more translation lookaside buffers (TLBs) for improving translation of virtual addresses into physical addresses in the memory 204.

## SM的组成

**SM包含指令cache、多个scheduler、GRF、若干处理Core、若干SFU、若干LD/ST单元、片上网络和共享内存/L1 Cache.**

FIG. 4A illustrates the streaming multiprocessor 340 of FIG. 3, in accordance with one embodiment. As shown in FIG. 4A, the SM 340 includes an instruction cache 405, one or more scheduler units 410, a register file 420, one or more processing cores 450, one or more special function units (SFUs) 452, one or more load/store units (LSUs) 454, an interconnect network 480, and a shared memory/L1 cache 470.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2037.png)

**任务由PPU顶层分派单元分派到GPC,被GPC分派到某个TPC执行,任务中着色程序分派到TPC中的SM执行.SM中多线程并发执行该任务,scheduler调度线程组/warp执行的指令,即调度不同线程组/warp并发执行任务.scheduler中dispatch单元每个周期将不同warp的指令发射到不同功能单元(Core、SFU、LD/ST).**

As described above, the work distribution unit 225 dispatches tasks for execution on the GPCs 250 of the PPU 200. The tasks are allocated to a particular TPC 320 within a GPC 250 and, if the task is associated with a shader program, the task may be allocated to an SM 340. The scheduler unit 410 receives the tasks from the work distribution unit 225 and manages instruction scheduling for one or more groups of threads (i.e., warps) assigned to the SM 340. The scheduler unit 410 schedules threads for execution in groups of parallel threads, where each group is called a warp. In one embodiment, each warp includes 32 threads. The scheduler unit 410 may manage a plurality of different warps, scheduling the warps for execution and then dispatching instructions from the plurality of different warps to the various functional units (i.e., cores 450, SFUs 452, and LSUs 454) during each clock cycle.

**若scheduler中配置两个dispatch单元,则每周期发射1个warp中2个指令.**

In one embodiment, each scheduler unit 410 includes one or more instruction dispatch units 415. Each dispatch unit 415 is configured to transmit instructions to one or more of the functional units. In the embodiment shown in FIG. 4A, the scheduler unit 410 includes two dispatch units 415 that enable two different instructions from the same warp to be dispatched during each clock cycle. In alternative embodiments, each scheduler unit 410 may include a single dispatch unit 415 or additional dispatch units 415.

**SM的GRF临时存储操作数.GRF可能按照执行单元划分,也可能按照warp划分.**

**GRF按照执行单元划分是指线程的不同操作数存储在不同Bank;按照warp划分是指每个warp/线程的存储映射到1个Bank.**

Each SM 340 includes a register file 420 that provides a set of registers for the functional units of the SM 340. In one embodiment, the register file 420 is divided between each of the functional units such that each functional unit is allocated a dedicated portion of the register file 420. In another embodiment, the register file 420 is divided between the different warps being executed by the SM 340. The register file 420 provides temporary storage for operands connected to the data paths of the functional units.

**SM可能包含192个Core、32个SFU、32个LSU.**

**Core可能包含单精度浮点的pipeline执行浮点运算和定点运算,双精度浮点的pipeline执行双精度浮点运算.SFU执行特殊函数计算.LSU执行从共享内存和寄存器之间的ld/st指令.**

Each SM 340 comprises L processing cores 450. In one embodiment, the SM 340 includes a large number (e.g., 192, etc.) of distinct processing cores 450. Each core 450 may include a fully-pipelined, single-precision processing unit that includes a floating point arithmetic logic unit and an integer arithmetic logic unit. The core 450 may also include a double-precision processing unit including a floating point arithmetic logic unit. In one embodiment, the floating point arithmetic logic units implement the IEEE 754-2008 standard for floating point arithmetic. Each SM 340 also comprises M SFUs 452 that perform special functions (e.g., pixel blending operations, and the like), and N LSUs 454 that implement load and store operations between the shared memory/L1 cache 470 and the register file 420. In one embodiment, the SM 340 includes 192 cores 450, 32 SFUs 452, and 32 LSUs 454.

**SM的片上网络将任意执行单元和任意寄存器/共享内存建立连接.**

Each SM 340 includes an interconnect network 480 that connects each of the functional units to the register file 420 and the shared memory/L1 cache 470. In one embodiment, the interconnect network 480 is a crossbar that can be configured to connect any of the functional units to any of the registers in the register file 420 or the memory locations in shared memory/L1 cache 470.

**共享内存/L1 Cache是片上存储阵列,可以分区成共享内存和L1 Cache,或者只作为共享内存或L1 Cache.**

The shared memory/L1 cache 470 is an array of on-chip memory that, in one embodiment, may be configured as either shared memory or an L1 cache, or a combination of both, as the application demands. For example, the shared memory/L1 cache 470 may comprise 64 kB of storage capacity. The shared memory/L1 cache 470 may be configured as 64 kB of either shared memory or L1 cache, or a combination of the two such as 16 kB of L1 cache and 48 kB of shared memory.

## 图形处理的过程

**PPU可以是GPU,GPU接收的命令一般是处理图形数据的着色程序.**

**图形数据是图元的集合,图元包含点、线、三角、方形、三角带(连着的三角形).**

**图元数据是组成图元的顶点(建模空间坐标系中)和图元中每个顶点的属性.**

**GPU处理图元数据,生成帧缓存(显示器的像素数据).**

In one embodiment, the PPU 200 comprises a graphics processing unit (GPU). The PPU 200 is configured to receive commands that specify shader programs for processing graphics data. Graphics data may be defined as a set of primitives such as points, lines, triangles, quads, triangle strips, and the like. Typically, a primitive includes data that specifies a number of vertices for the primitive (e.g., in a model-space coordinate system) as well as attributes associated with each vertex of the primitive. The PPU 200 can be configured to process the graphics primitives to generate a frame buffer (i.e., pixel data for each of the pixels of the display).

**应用将显示的建模数据(顶点和属性)写在存储中,建模数据定义了显示可见的对象.**

**应用调用驱动内核的API,发出将建模数据进行渲染和显示的请求**

**驱动内核读取建模数据,将处理建模数据所需操作以命令流输出.**

**命令可能引用不同的着色程序在SM上执行,包括vertex、hull、domain、geometry和pixel的着色.**

**不同SM可以并发执行不同的着色程序,组成pipeline完成建模数据到显示时帧缓存数据的转换和渲染.**

**第一组SM配置执行vertex的着色程序,将处理后的vertex数据写到L2 Cache或存储;**

**处理后的vertex数据经过Raster引擎光栅化(三维建模数据转二维屏幕数据),得到片元数据fragment;**

**第二组SM配置执行片元数据的着色程序,处理后的fragment进行混合blend后写入帧缓存.**

**帧缓存数据传送到显示设备显示图像.**

An application writes model data for a scene (i.e., a collection of vertices and attributes) to a memory such as a system memory or memory 204. The model data defines each of the objects that may be visible on a display. The application then makes an API call to the driver kernel that requests the model data to be rendered and displayed. The driver kernel reads the model data and writes commands to one or more streams to perform operations to process the model data. The commands may reference different shader programs to be implemented on the SMs 340 of the PPU 200 including one or more of a vertex shader, hull shader, domain shader, geometry shader, and a pixel shader. For example, one or more of the SMs 340 may be configured to execute a vertex shader program that processes a number of vertices defined by the model data. In one embodiment, the different SMs 340 may be configured to execute different shader programs concurrently. For example, a first subset of SMs 340 may be configured to execute a vertex shader program while a second subset of SMs 340 may be configured to execute a pixel shader program. The first subset of SMs 340 processes vertex data to produce processed vertex data and writes the processed vertex data to the L2 cache 360 and/or the memory 204. After the processed vertex data is rasterized (i.e., transformed from three-dimensional data into two-dimensional data in screen space) to produce fragment data, the second subset of SMs 340 executes a pixel shader to produce processed fragment data, which is then blended with other processed fragment data and written to the frame buffer in memory 204. The vertex shader program and pixel shader program may execute concurrently, processing different data from the same scene in a pipelined fashion until all of the model data for the scene has been rendered to the frame buffer. Then, the contents of the frame buffer are transmitted to a display controller for display on a display device.

**PPU可能存在于个人电脑、服务器、移动设备、电子相机、手持电子设备等.**

**PPU可能封装成独立芯片,也可能和CPU、MMU、DAC等封装成SoC芯片.**

The PPU 200 may be included in a desktop computer, a laptop computer, a tablet computer, a smartphone (e.g., a wireless, hand-held device), personal digital assistant (PDA), a digital camera, a hand-held electronic device, and the like. In one embodiment, the PPU 200 is embodied on a single semiconductor substrate. In another embodiment, the PPU 200 is included in a system-on-a-chip (SoC) along with one or more other logic units such as a reduced instruction set computer (RISC) CPU, a memory management unit (MMU), a digital-to-analog converter (DAC), and the like.

**PPU芯片可能和存储芯片(GDDR5)组装成显卡(板级封装),显卡使用PCIe接口和南桥或北桥建立连接.**

**PPU可能作为集成GPU,封装在主板上chipset中(北桥).**

**南桥和北桥都是芯片组chipset,分别负责低速信号和高速信号的处理和传输.**

In one embodiment, the PPU 200 may be included on a graphics card that includes one or more memory devices 204 such as GDDR5 SDRAM. The graphics card may be configured to interface with a PCIe slot on a motherboard of a desktop computer that includes, e.g., a northbridge chipset and a southbridge chipset. In yet another embodiment, the PPU 200 may be an integrated graphics processing unit (iGPU) included in the chipset (i.e., Northbridge) of the motherboard.

## 聚合Barrier机制

**通过线程屏障同步不同分支且未退出的线程,从而管理线程分束,提高SIMD core利用率.线程同步后,以SIMD模式逐指令执行.**

Thread divergence is managed using one or more convergence barriers to synchronize threads that have taken different paths during execution of the program and have not exited the program. Once threads have synchronized at a convergence barrier, the threads may begin executing instructions in lock-step SIMD manner again.

**聚合Barrier是体系结构管理线程分束的机制向上提供的编程抽象.**

**聚合Barrier代表指令执行时相关硬件的关联操作.相关硬件包含scheduler、命名聚合Barrier的标志寄存器,关联操作由编程指令和硬件机制控制执行.**

A convergence barrier, as the term is used herein, refers to a programming abstraction for managing thread divergence in a SIMD architecture. In one embodiment, the convergence barrier includes operations associated with instructions executed by the scheduling unit 410 and a set of hardware for performing the operations. The set of hardware may include a multi-bit register corresponding to each of one or more convergence barrier names implemented by the SIMD architecture and made available to programmers through the instructions.

**聚合Barrier一般有两个关联操作:ADD指令和WAIT指令.**

**ADD指令将当前激活的线程加入一个ADD指令命名的聚合Barrier,WAIT指令阻塞线程执行直到参与聚合Barrier的线程全部到达.**

**编译器在可能存在线程分束的控制流区间(代码块)的入口处插入ADD指令,在该区间的出口处插入WAIT指令,理想情况是控制流分区是单入口,线程在内部分束,在出口同步(后进入下一个块).**

In one embodiment, convergence barriers support two main operations: ADD and WAIT. ADD adds a set of active threads to participate in a specific convergence barrier that is named by the ADD instruction. WAIT suspends the execution of threads until all participating threads have arrived. The compiler is responsible for placing ADD instructions at the entry points to a control flow region (e.g., portion of a program) that may contain a divergent branch, and for placing WAIT instructions at the exit points to the region. Ideally, regions will have a single exit point, and threads that take divergent paths within the region will synchronize at the exit point.

**每个控制流分区被编译器分配唯一的聚合Barrier.不同warp的命名Barrier相互独立.**

**聚合Barrier的名字(ID)和状态信息需建立映射,可被不同机制实现(纯硬件映射、编译器分配、软件调度等).**

Conceptually, each control flow region in the program is assigned a unique convergence barrier (or other type of barrier) by the compiler. In one embodiment, N different names are available, and state information may be stored for each one of the N different convergence barriers. However, in some embodiments, a mechanism may be required to map logical barriers (the number of which is unbounded) to hardware resources such as registers (which are finite). There are multiple techniques for handling the mapping (e.g., with a memory-backed cache, with compiler-managed allocation, with software-managed scheduling, etc.).

**线程组/warp同步进入代码区间,并通过ADD指令配置聚合Barrier,ADD指令可能不在代码区间入口.**

**执行ADD指令的线程参与ADD指定的命名Barrier,代码区间内部线程分束执行,最终执行到WAIT指令.**

**WAIT指令阻塞参与命名Barrier的线程,直到Barrier被释放,scheduler调度其余线程组执行.**

**当所有参与命名Barrier的线程到达WAIT指令,即线程状态是Blocked时,命名Barrier被释放.**

**线程执行OPT-OUT指令,退出所在代码分区,则该线程退出该命名Barrier.**

**线程执行exit指令,生命周期结束,线程状态变为exited,该线程可能不会退出该命名Barrier,相对的warp scheduler检测是否释放Barrier时忽略exited的线程.**

Typically, a group of threads will enter a region synchronized and configure a convergence barrier based on an ADD instruction included at the entry point to the region. In one embodiment, threads may execute an ADD instruction that is not at the entry point to the region and participate in the convergence barrier. The ADD instruction indicates that a thread executing the ADD instruction participates in a convergence barrier named by the ADD instruction. Then, when executing instructions within the region, the threads may take different execution paths through the region. Eventually, each of the threads will reach a WAIT instruction at the end of the region. At the WAIT instruction, the threads are suspended until the convergence barrier releases, and the scheduler unit 410 selects another set of threads to execute. The convergence barrier releases when all threads that participated in the convergence barrier arrive at the WAIT instruction and have a status of blocked. However, one or more threads participating in the convergence barrier may exit the region and terminate participation in the convergence barrier by executing an OPT-OUT instruction included in a divergent path of the region. In one embodiment, threads that execute an EXIT instruction change from a ready state to an exited state but do not explicitly terminate participation in the convergence barrier. Instead, the scheduler unit 410 ignores threads that participate in a barrier and have a state of exited but have not arrived at the convergence barrier.

**参与Barrier的线程可能执行YILED指令,则检测是否释放Barrier时不考虑yielded状态的线程.**

**yielded的线程也不必要执行WAIT指令,当命名Barrier释放时,yielded线程也退出Barrier.**

**yielded线程在退出Barrier时可唤醒，或存在其他同步机制.**

One or more other threads participating in the convergence barrier may execute a YIELD instruction included in another divergent path of the region, and threads that have arrived at the convergence barrier need not wait for the yielding threads to release the convergence barrier. Likewise, the yielding threads need not necessarily execute the WAIT instruction. Regardless, once the convergence barrier is cleared, the yielding threads no longer participate in the convergence barrier.

**分离的线程子集能参加聚合Barrier,如果线程子集在任一方执行WAIT指令前执行ADD指令.**

**分离的线程子集需要在任一方执行WAIT指令之前,将自身加入聚合Barrier;**

**先到WAIT指令处的线程阻塞,直到两个线程子集都到达WAIT指令.**

**基于运行栈的分支执行和前述场景中执行ADD的线程都是同步的（成组的）,此处分离的线程子集是异步执行ADD指令.**

**实际场景是代码块内从不同入口进入的分束线程也能进行同步,运行栈的分支执行只对warp内线程进行同步,并且发生在父亲线程组直接分束的线程之间.**

Disjoint subsets of threads may participate in a convergence barrier if both subsets of threads execute an ADD instruction specifying the convergence barrier before either subset of threads executes the WAIT instruction. In this case, both subsets of threads will add themselves to the convergence barrier before either one of the subsets of threads reaches the convergence barrier. The subset of threads that reaches the WAIT instruction first will block on the convergence barrier until the other subset of threads arrives. **This type of re-convergence behavior is not possible when a stack-based divergence mechanism is used.**

## Scheduler的线程调度

**scheduler的目标是确保控制流的推进和执行分束线程以尽快聚合.**

**scheduler为每个聚合Barrier和warp中每个线程维护状态信息.**

**barrier participation mask表示warp中参与某个barrier的线程(如0-31).当线程执行ADD指令时,mask中对应位置1,当Barrier释放或某个线程执行OPT-OUT指令时,线程对应的位置0.**

**Barrier释放后,参与Barrier的线程的thread state从blocked或yielded变成ready,表示可被scheduler调度.**

FIG. 4B illustrates convergence barrier state, in accordance with one embodiment. The scheduler unit 410 functions to achieve two objectives: ensuring forward progress and executing divergent paths such that they converge as soon as possible. To support these objectives, state information is maintained for each convergence barrier and each thread in a warp. The state information may be stored in registers. A barrier participation mask 425 indicates the threads in a warp that participate in a convergence barrier. In one embodiment, a single bit is stored for each thread in the warp. Bits corresponding to threads that participate in the convergence barrier are set in the barrier participation mask 425 when an ADD instruction is executed. The bits corresponding to the threads that participate in the convergence barrier are cleared when the convergence barrier is cleared (or when a corresponding thread executes an OPT-OUT instruction). When the convergence barrier is cleared, the thread state for threads participating in the barrier is changed from either blocked or yielding to ready.

> **[图片提取文字 (image.png)]:**
> ## Barrier Participation Mask <u>425</u> Barrier State <u>430</u>
> 
> | Thread State<br>440-0      |  | <br>Thread State<br>440-31     |                             |  |
> |----------------------------|--|--------------------------------|-----------------------------|--|
> | Thread rPC<br><u>445-0</u> |  |                                | Thread rPC<br><u>445-31</u> |  |
> | Thread<br>Active<br>460-0  |  | <br>Thread<br>Active<br>460-31 |                             |  |
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2038.png)

**Barrier state是最近被scheduler调度执行的线程ID,以便scheduler实现线程时间片轮转.**

**Barrier state包含yield计数器来记录激活线程执行的YIELD指令.当计数器超过阈值,执行Yield操作,并且释放聚合Barrier.(降低YIELD频率,让yielded线程尽量同步成组)**

Barrier state 430 may also include an identifier of the last thread that was selected for execution by the scheduler unit 410 so that round-robin scheduling may be implemented by the scheduler unit 410. Barrier state 430 may also include a yield counter that counts the number of YIELD instructions that have been executed by the active threads. In one embodiment, when the yield counter crosses a threshold, the Yield action is performed to release the convergence barrier and the yield counter is reset.

**warp中每个线程被维护Thread state,包含Ready、Blocked、Yielded、Exited、Ready-、Blocked+.**

Thread state 440 is maintained for each thread in a warp. As shown in FIG. 4B, there are 32 threads in a warp, although in other embodiments, a different number of threads may be included in a warp. The thread state 440 indicates the current state of a thread. In one embodiment, threads in a warp are in exactly one of the following states:

- Ready state: thread ready for execution.
- Blocked state (1-N): Thread is not eligible for execution because it is blocked on a convergence barrier. The state specifies the barrier number that a thread is blocked on.
- Yielded state: Thread has voluntarily yielded its execution by executing the YIELD instruction. The scheduler unit 410 can move the yielded thread to ready state and the thread can be scheduled again. Also, convergence barriers should not wait for a yielded thread to be released.
- Exited state: thread has executed an EXIT instruction.
- Ready- state: thread has executed a NANOSLEEP instruction and remains in this state until the thread is scheduled again, or until the sleep timer expires.
- Blocked+ state: thread is blocked on a barrier and the scheduler unit 410 requires all threads participating in the barrier to be synchronized, **including yielded threads**. In one embodiment, this state is entered when a WAIT instruction is executed for threads participating in a specific number of convergence barrier (e.g., convergence barrier 15).

**Thread active表示该线程是否已经被scheduler选择执行,只有Ready状态的线程可能被选择.**

**Thread rPC在线程离开Active状态被写入,记录线程下次调度执行时的指令PC.**

**scheduler创建新的激活mask时使用rPC(动态warp?).**

Thread active 460 indicates whether the thread is active, i.e., selected by the scheduler unit 410 for execution. Only threads in the Ready state may be active. Thread rPC 445 stores a program counter that is written by the scheduler unit 410 when the thread leaves the active mask. When the thread is not active, the rPC stores the program counter of an instruction that the thread will execute when unblocked. The rPC is read when a new active mask is constructed by the scheduler unit 410.

**scheduler调度不同warp/线程组执行,保证任务进程的推进.SIMD的执行模式让warp内多个线程同时执行一条指令,因此直接调度不同线程执行非常复杂.**

**当warp内出现线程分束,采用周期切换不同分支的线程来推进任务,但会产生性能开销和利用率降低.因此SM的调度目标是尽快同步分束的线程.**

As previously explained, forward progress is guaranteed for the threads executing on the SM 340. Individual threads will eventually execute instructions when all other threads either synchronize or exit the program. Forward progress between warps is handled implicitly by the scheduler unit 410, which runs warps independently. Providing a forward progress guarantee to threads within a warp is more difficult because warps run in SIMD fashion, i.e., they can only execute threads at the same PC simultaneously. If threads are at different PCs, then it is necessary to periodically switch between threads to provide forward progress. However, threads that are not running synchronously at the same PC incur a performance overhead, so the SM 340 has a strong motivation to attempt to synchronize divergent threads as soon as possible.

**scheduler初始选择线程组/warp并发送到SIMD core执行.线程组被选择后同步顺序执行指令,直到遇到分支指令或者控制权回到scheduler(阻塞、cache miss).**

**线程执行分支指令并分束后,处于相同分支的线程继续执行,其余分支线程被scheduler阻塞(Ready).**

The scheduler unit 410 is responsible for picking a new set of threads at the same PC that are not waiting on a convergence barrier and loading them onto the SIMD datapath. Once the new set of threads is selected by the scheduler unit 410, the selected threads are run until they diverge or transfer control back to the scheduler unit 410. When the threads execute a divergent branch, the threads will no longer all be at the same PC and cannot continue to execute concurrently. At this time, some threads that took the same path will continue executing, and all other threads will be suspended by the scheduler unit 410.

**scheduler记录因分束被阻塞的线程的下一条指令PC(在rPC中),以便稍后恢复.**

**scheduler会周期性调度(RR)不被聚合Barrier阻塞的线程(Ready)去SIMD core执行.**

**scheduler的分束调度目标是尽快让分束线程同步,切推进任务过程.**

The scheduler unit 410 will remember the PC of the suspended threads so that execution of the suspended threads may be resumed later. In one embodiment, the scheduler unit 410 may also periodically switch the currently active threads with any other threads that are not waiting on a convergence barrier. The divergent path logic of the scheduler unit 410 attempts to simultaneously maximize opportunities for divergent threads to synchronize and not violate the forward progress guarantee.

**scheduler优先调度深层嵌套的分支,并且优先调度拥有规整控制流代码块的线程,所有规整的控制路径调度完后(active),再以RR的策略调度不规整的路径(出现OPT-OUT和YIELD指令).**

**因为不规整的路径对SIMD core的利用率一般较低，并且不会在聚合Barrier处阻塞其余线程,因此优先级较低.**

**深嵌套优先次序的机制是将因分束被阻塞的线程ID存入栈中,当scheduler调度新线程时,弹出栈顶的线程,调度该线程执行(以及相同PC的线程),栈为空则以RR策略调度其他线程.**

In one embodiment, the scheduler unit 410 implements a scheduling policy where structured control flow blocks are scheduled in depth-first order. Once all structured paths have been scheduled, unstructured paths (corresponding to OPT-OUT and YIELD) are scheduled in a round-robin manner. The depth-first order is determined by saving, in a stack, an identifier of a thread that is disabled on a divergent branch. When the scheduler unit 410 switches to another thread, the scheduler unit 410 pops the entry from the top of the stack and schedules the popped thread corresponding to the identifier (along with all other threads at the same PC). If the stack is empty, the scheduler unit 410 picks a new thread in round-robin order.

## 简单控制流

**控制流图中的node表示顺序指令的代码块,edge表示分支指令.**

**规整的控制流定义为任一诱导子图(“选点必选点间的所有边”)都是单入口和单出口的.**

FIG. 5A illustrates a control flow graph that represents a program with conditional flow, in accordance with one embodiment. Each node 501, 502, 503, and 504 in the control flow graph represents a block of one or more instructions that are executed in sequence without any branches. An edge connecting two nodes represents a branch. Structured control flow is defined as any induced subgraph of a program control flow graph with a single entry point and a single exit point.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5A
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2039.png)

**scheduler可能优先选择持有最少active线程的分支路径去执行,因为理论执行时间更短,其次选择直通(自增PC)的分支路径执行.**

In the node 501, threads are added to the convergence barrier B0 at the entry point to the region of the program. The barrier participation mask 425 is updated to indicate that the threads participate in the convergence barrier B0. The threads may diverge when the branch instruction (BRA) is executed. For example, while executing @P0 BRA, the threads that have the predicate P0 will take the branch, while the threads that do not have the predicate P0 set will take the fall-through path. A first divergent path is the node 503 and a second divergent path is the node 502. In one embodiment, the scheduler unit 410 selects one of the two divergent paths to execute ending in the WAIT instruction in the node 504, at which point the active threads will block on the convergence barrier B0 and the remaining threads will execute the other divergent path. **In one embodiment, the scheduler unit 410 selects the divergent path with the fewest number of active threads (according to thread active 460) to execute first.** When the number of active threads is equal, the scheduler unit 410 may select the fall-through path (e.g., node 502). After the selected divergent path is executed, the other divergent path will execute through the WAIT instruction, and the convergence barrier will be released. Finally, all of the original threads will resume execution after the WAIT instruction.

**控制流可能出现任意的嵌套逻辑,内层同步后的线程可以在外层和其他线程进行同步.**

Control flow regions may be nested arbitrarily, and a set of threads that have synchronized at an inner nesting level can subsequently synchronize with another set of threads in an outer nesting level. FIG. 5B illustrates a control flow graph that represents a program with nested control flow, in accordance with another embodiment. Each node 510, 511, 512, 514, 515, 516, and 518 in the control flow graph represents a block of one or more instructions that are executed in sequence without any branches.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5B
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2040.png)

In the node 510, threads are added to the convergence barrier B0 at the entry point to the region of the program. The barrier participation mask 425 for the convergence barrier B0 is updated to indicate that the threads participate in the convergence barrier B0. The convergence barrier B0 synchronizes threads at an outer if/else block at the exit point of the region of the program. The threads participating in the convergence barrier B0 may diverge when the branch instruction (BRA) is executed. A first divergent path is the node 512 and a second divergent path is the node 511. A first set of divergent threads that take the first divergent path may include zero threads or up to all threads that participate in the convergence barrier B0. A second set of divergent threads that take the second divergent path includes the remaining threads that participate in the convergence barrier B0.

In the node 512 of the second divergent path, the second set of divergent threads are added to a convergence barrier B1 corresponding to an inner if/else block within the region of the program. The barrier participation mask 425 for the convergence barrier B1 is updated to indicate that the threads participate in the convergence barrier B1. The second set of divergent threads that participate in the convergence barriers B0 and B1 may diverge when the branch instruction (BRA) in the node 512 is executed. A third divergent path is the node 515 and a fourth divergent path is the node 514. The convergence barrier B1 synchronizes the threads in the second set of divergent threads at the WAIT instruction in the node 516 of the region of the program.

The scheduler unit 410 selects either the first or second divergent path to execute ending at either the WAIT instruction in the node 518 or the WAIT instruction in the node 516, respectively. When the second set of divergent threads executing the third and fourth divergent paths have both executed to reach the convergence barrier B1, the convergence barrier B1 is released and the threads in the second set of divergent threads resume SIMD execution until they reach the convergence barrier B0 in node 518. The barrier participation mask 425 for the convergence barrier B1 is updated to indicate that the convergence barrier B1 is cleared. When all of the threads participating in the convergence barrier B0 have executed the WAIT instruction in the node 518, the convergence barrier B0 is released. The barrier participation mask 425 for the convergence barrier B0 is updated to indicate that the convergence barrier B0 is cleared. Finally, all of the threads in the first and second sets of divergent threads resume execution after the WAIT instruction in the node 518.

## 函数调用的同步

**函数调用的同步,在函数入口指令插入ADD指令,在return处插入WAIT指令.**

FIG. 5C illustrates an example of a control flow graph of a region of a program with function calls, in accordance with one embodiment. Each node 520, 521, 522, 525, and 526 in the control flow graph represents a block of one or more instructions that are executed in sequence without any branches. An edge connecting two nodes represents a branch. Function calls can be synchronized with ADD instruction at the entry point in node 520 and a WAIT instruction at the return point in node 526.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> *Fig.* 5*C*
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2041.png)

**内联函数调用等价分支.使用函数指针的函数调用通过branch指令实现,在代码块入口插入ADD指令,在代码块返回处同步.**

Inline function calls are equivalent to branches. Function calls that use function pointers are implemented using a branch instruction (BRX). For example, BRX R0 causes each thread to branch to a location that is determined by the per-thread register value R0. In node 520, threads are added to the convergence barrier B0 at the entry point to the region of the program. The barrier participation mask 425 for the convergence barrier B0 is updated to indicate the threads that participate in the convergence barrier B0. The convergence barrier B0 synchronizes threads after the function calls complete. The threads participating in the convergence barrier B0 may diverge when the branch instruction is executed.

**不同线程的函数调用的目标PC可能不同(寄存器的值不同),则不同分支在代码块返回处同步.**

A first divergent path is the node 522, and a second divergent path is the nodes 521 and 525. Each set of divergent threads that takes one of the divergent paths may include zero threads or up to all threads that participate in the convergence barrier B0. The threads participating in the convergence barrier each execute one of the divergent paths and wait for all of the participating threads to reach the convergence barrier at node 526. When all of the participating threads execute the WAIT instruction at node 526, the barrier participation mask 425 for the convergence barrier B0 is updated to indicate that the convergence barrier B0 is cleared.

## YIELD指令(循环)

**聚合Barrier能尽早同步线程,减少SIMD的空闲时间.**

**但WAIT指令会阻塞线程直到同步,如果未同步线程因为环形依赖无法到达,则会导致死锁(运行栈的例子).**

**比如执行循环的线程一直等待资源才能跳出循环,而资源被其他线程占用,而由于scheduler的深度优先策略（每次都处于栈顶）每次调度执行循环的线程(等待资源的线程).**

**循环分支的代码块中插入YIELD指令,以解决死锁.**

**执行YIELD指令将线程状态变成yielded,并且线程被阻塞.通过YIELD指令,能让执行循环等待资源的线程挂起,让其余线程调度执行后释放资源,解决死锁.**

**线程处于yielded状态表示线程参与的聚合Barrier在判定同步时不需要考虑自身,即只要参与Barrier的所有非yielded状态的线程达到同步,Barrier就可被释放.**

**Barrier被释放时,因为部分线程处于yielded状态,线程组内同步的线程数未占满,降低SIMD的使用率.**

FIG. 5D illustrates a control flow graph of a region of a program with divergent loop control flow, in accordance with one embodiment. Note that as described thus far, convergence barriers provide a way for scheduling threads that reduce SIMD idle time, but they do not necessarily provide a forward progress guarantee. This is because the WAIT operation will block some threads until other threads arrive at the WAIT instruction. If the other threads do not arrive for some reason (e.g., they are waiting for yet another set of threads in a way that forms a circular dependency), then the waiting threads block indefinitely. To address the indefinite blocking issue, in one embodiment, the YIELD instruction is inserted into the region of the program with divergent loop control flow. Threads that execute the YIELD instruction change their state from ready to yielded. The yielded state indicates that any convergence barrier waiting for the threads to arrive may be released when all non-yielding threads participating in the convergence barrier have reached the convergence barrier. Threads that execute the YIELD instruction will also be suspended by clearing the thread active 460 for each yielding thread. While releasing the convergence barrier does allow for forward progress, releasing the convergence barrier when all of the threads are not at the convergence barrier skips an opportunity for divergent threads to synchronize, reducing SIMD efficiency.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5D
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2042.png)

As shown in FIG. 5D, each node 530, 532, 534, 536, and 538 in the control flow graph represents a block of one or more instructions that are executed in sequence without any branches. An edge connecting two nodes represents a branch or a fall-through path. In node 530, threads are added to the convergence barrier B1 at the entry point to the region of the program. The threads may diverge when the branch instruction (BRA) is executed. A first divergent path branches around the loop to node 538 and a second divergent path is through nodes 532, 534, and 536. A first set of divergent threads that take the first divergent path may include zero threads or up to all threads that participate in the convergence barrier B1. A second set of divergent threads that take the second divergent path includes the remaining threads that participate in the convergence barrier B1.

Threads in the first set of divergent threads block on the WAIT instruction in node 538 and wait for the threads in the second set of divergent threads to execute through the loop when a YIELD instruction is not included in node 532. As each thread in the second set of divergent threads exits the loop, the exiting threads block on the WAIT instruction in node 538. When all of the threads in the second set of divergent threads have exited the loop, the threads in the first set of divergent threads and the threads in the second set of divergent threads are synchronized and resume execution being converged.

**执行YIELD指令的线程状态变成yielded,线程被阻塞来让其他线程执行,不会参与下一个聚合Barrier.**

Node 532 may include a YIELD instruction that allows threads to not synchronize at the convergence barrier, so that the synchronization is flexible. When a thread executes the YIELD instruction, execution of the thread is suspended and the thread is placed in the yielded state. Threads are suspended to give other threads a chance to execute and the yielding threads will not participate in the next convergence barrier. The convergence barrier B1 is released when all of the threads taking the first divergence path are blocked at the convergence barrier and when all of the threads taking the second divergence path are either blocked at the convergence barrier or are in the yielded state.

**scheduler保证每个分支的线程都有机会执行,因此可能使用RR策略(每个分支都属于规整代码块).**

**编译器通过插入YIELD指令让一个分支线程挂起,从而不会无限执行.**

**编程时显式在边界条件处挂起线程能减少不必要的YIELD指令.**

When there is a choice between multiple divergent paths to execute, the scheduler unit 410 is responsible for making sure that all paths eventually get to execute. In one embodiment, a round-robin technique is used to select ready threads for execution. The compiler is responsible for ensuring that a divergent path does not execute indefinitely by periodically yielding to the thread scheduler by inserting the YIELD instruction as needed (thereby implementing a form of cooperative multi-threading). The compiler may be assisted in ensuring that a divergent path does not execute indefinitely by suitable language specifications that place restrictions on infinite loops, e.g., as in C++, where specific side effects are expected to eventually happen. Unnecessary yields may be avoided by inserting yield instructions at the points the specific side effects may occur.

**编译器可能在无法在静态确定个数的指令中完成的执行路径中插入YIELD指令.**

**编译器可能在任何回跳的branch指令前插入YIELD指令,保证其余线程能够推进.**

**编译器可能在从内存中加载的指令前插入YIELD指令,因为延迟不定.**

In one embodiment, the compiler may be configured to insert a YIELD instruction along any control path that does not terminate in a statically determined number of instructions. Although the compiler may try to aggressively eliminate YIELD instructions where the compiler can prove that a YIELD instruction is not necessary, in one embodiment, YIELD instructions may be inserted before any branch instructions that may branch backwards (to a program counter with a lower value than the branch itself) to guarantee forward progress. A further optimization is to only insert YIELD instructions where there are loads from volatile memory locations.

**YIELD指令导致线程错过同步成组的机会.保守执行方式下,yielded线程最终会执行到聚合Barrier,此时Barrier内其余线程已经前行,该线程对SIMD core的使用率低.**

A YIELD instruction may result in a missed synchronization opportunity. For example, if a thread executes a YIELD instruction, then other threads waiting at the next convergence barrier are free to continue without the yielding thread. If the YIELD instruction was executed conservatively, and the yielded threads would have eventually reached the convergence barrier, then some performance may be lost when divergent paths are not executed simultaneously.

**减少yielded线程错过同步次数的优化思路有:**

**1、对于只被yielded线程阻塞释放的屏障,只有当scheduler选择参与该屏障的线程执行时,才释放屏障,即尽可能让线程成组.**

**2、释放只被yielded线程阻塞释放的屏障后,将yielded状态的线程变为ready,即原地成组.**

**3、软件做YIELD优化,设定YIELD指令的执行频率,即scheduler累计识别到YIELD指令到一定次数才执行YIELD指令.**

A straightforward optimization that minimizes the number of lost synchronization opportunities is to only release a convergence barrier that is waiting on yielded threads when the scheduler unit 410 selects a thread for execution that is waiting on the convergence barrier. Another straightforward optimization to reduce the number of lost synchronization opportunities is to clear the yielded state from threads when that state is used to release a convergence barrier. The yielded state is cleared by updating the thread state 440 for each yielded thread to a different state, such as ready. Yet another optimization would be to elide YIELD instructions with some probability (e.g., ignore YIELD instructions until a timer expires, until a number of cycles have elapsed, until a number of branch instructions have been executed, etc.). In one embodiment, software performs a YIELD optimization, for example, by annotating YIELD instructions with an expected frequency count, such that in a short loop, the YIELD instruction is elided by the scheduler unit 410 until a software-specified (e.g., programmable) counter value is exceeded.

## 不规整的控制流

FIG. 5E illustrates a control flow graph for a region of a program with short-circuit control flow, in accordance with one embodiment. Each node 540, 542, 544, 546, 548, 550, 552, 554, 556, and 560 in the control flow graph represents a block of one or more instructions that are executed in sequence without any branches. An edge connecting two nodes represents a branch.

**不规整的控制流包含多入口和(或)多出口.聚合Barrier通过OPT-OUT指令来掌控多出口的控制流.**

**OPT-OUT指令允许参加聚合Barrier的线程退出Barrier而线程不阻塞.**

**编译器将多出口中的一个设置为主出口并插入WAIT指令,其余出口插入OPT-OUT指令.**

All induced subgraphs of the program control flow graph that are not structured are unstructured. Unstructured control flow contains multiple entry points, multiple exit points, or both. In one embodiment, convergence barriers handle control flow regions with multiple exit points by introducing a third primitive operation (OPT-OUT). OPT-OUT allows a set of threads that were previously added to a convergence barrier to exit the convergence barrier without blocking. In one embodiment, the compiler may handle a control flow region with multiple exit points by designating one exit point the "primary exit point" and placing a WAIT instruction at that point. Other exit points are then assigned OPT-OUT operations. A short-circuit control flow, where control can opt-out of a convergence barrier B1 for an inner loop and proceed directly to an outer convergence barrier B0, is shown in FIG. 5E.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5E
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2043.png)

In node 540, threads are added to the convergence barrier B0 at the entry point to the region of the program. The threads may diverge when the branch instruction (BRA) is executed. A first divergent path branches around an outer loop to node 556 and a second divergent path continues through nodes 542, 544, and 546. A first set of divergent threads that take the first divergent path may include zero threads or up to all threads that participate in the convergence barrier B0. A second set of divergent threads that take the second divergent path includes the remaining threads that participate in the convergence barrier B0.

In node 544, the threads in the second set of divergent threads are added to the convergence barrier B1 at the entry point to an inner loop within the region of the program. The threads may diverge when the branch instruction (BRA) in node 544 is executed. A third divergent path branches around the inner loop to node 552 and a fourth divergent path continues to node 546. A third set of divergent threads that take the third divergent path may include zero threads or up to all threads that participate in the convergence barrier B1. A fourth set of divergent threads that take the fourth divergent path includes the remaining threads that participate in the convergence barrier B1.

Threads in the fourth set of divergent threads that take an early exit path are a fifth set of divergent threads that explicitly opt out of the inner loop on a fifth divergent path through node 560. The threads in the fifth set of divergent threads execute an OPT-OUT instruction in node 560 and exit the convergence barrier B1. **When the OPT-OUT instruction is executed, the thread state 440 remains unchanged and remains ready**. However, since the threads in the fifth set of divergent threads still need to synchronize on the convergence barrier B0 at node 556, the threads in the fifth set of divergent threads continue to node 554. The remaining threads in the fourth set of divergent threads that do not take the early exit path through node 560 eventually execute the WAIT instruction at node 552 and are blocked at the convergence barrier B1 or execute a YIELD instruction in node 546. When the remaining threads in the fourth set of divergent threads that are not yielding synchronize at the convergence barrier B1, the convergence barrier B1 is cleared and the remaining threads proceed to node 554.

**一个yielded的线程只能被一个聚合Barrier释放时变为ready,并退出Barrier.**

**In one embodiment, any of the remaining threads that were yielding when the convergence barrier B1 is cleared transition from the yielding state to the ready state.** A yielding thread only yields for one convergence barrier. As each thread reaches node 556 and executes the WAIT instruction at the convergence barrier B0, the thread is blocked until all of the threads have executed the WAIT instruction. When all of the threads participating in the convergence barrier B0 are blocked, the convergence barrier B0 is cleared and the threads execute subsequent instructions in a SIMD manner.

## Scheduler检验同步

**参与Barrier的线程到达对应名字Barrier的指令,并且参与线程状态是阻塞.**

**到达Barrier且被阻塞的线程的rPC相同并且是Barrier处指令的PC。**

FIG. 6 illustrates a flowchart of a method for testing for synchronization at a convergence barrier for step 130 of FIG. 1, in accordance with one embodiment. At step 610, the scheduler unit 410 determines if all of the participating threads are synchronized at the convergence barrier. In one embodiment, the scheduler unit 410 determines that all of the threads participating in the convergence barrier (indicated by the barrier participation mask 425) are synchronized when all of the participating threads have reached the convergence barrier instruction matching the name specified by the ADD instruction and the thread state 440 for each of the participating threads is blocked. The threads that are at the convergence barrier have the same program counter value stored in the thread rPC 445.

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 6
![image.png](NV%20GPU%E4%B8%93%E5%88%A9/image%2044.png)

**参加Barrier的线程执行opt-out指令后，从Barrier mask中清除，并且状态变为exited.**

If, at step 610, the scheduler unit 410 determines that all of the participating threads that are participating in the convergence barrier are blocked at the convergence barrier, then the scheduler unit 410 proceeds directly to step 140. Otherwise, at step 615, the scheduler unit 410 removes any of the participating threads that executed an OPT-OUT instruction from the participating threads. In one embodiment, bits in the barrier participation mask 425 corresponding to each thread that executed an OPT-OUT instruction are cleared. In one embodiment, participating threads that execute an OPT-OUT instruction are removed from the participating threads when the OPT-OUT instruction is executed and step 615 is omitted. The thread state 440 for each of the threads participating in the convergence barrier that execute an OPT-OUT instruction changes from ready to exited.

**参与Barrier但状态是yielded的线程在检查是否同步时被忽略.**

**当线程执行YIELD指令，状态变为yielded但仍参与Barrier.当大部分指令分束到存在YIELD指令的分支，YIELD指令不会执行.因为YIELD指令是防止小部分线程一直执行.**

At step 620, the scheduler unit 410 ignores any of the participating threads that executed a YIELD instruction from the participating threads. Unlike the threads that have executed the OPT-OUT instruction, the threads that are in the yielding state are not removed, but are instead ignored for the convergence barrier release analysis. In other words, the bits in the barrier participation mask 425 corresponding to each thread that executed a YIELD instruction are not cleared, so the yielding threads still participate in the convergence barrier. The thread state 440 for each of the threads participating in the convergence barrier that execute a YIELD instruction changes from ready to yielded. Note that when a portion of threads diverge and take the same path that includes a YIELD instruction, all of the threads in the divergent group do not necessarily execute a YIELD instruction. In other words, only a subset of the threads in the divergent group may execute the YIELD instruction and change from ready to yielded. The remaining threads in the divergent group may remain ready.

At step 625, the scheduler unit 410 ignores any participating threads that are in the yielded state (according to the thread state 440) and determines if all of the participating threads are synchronized at the convergence barrier.

**scheduler在Barrier指令处发现参与Barrier的非yielded线程都blocked，则判定达成同步，并改变参与Barrier的线程状态到ready（包括yielded）.**

If, at step 625, the scheduler unit 410 determines that all of the non-yielding threads participating in the convergence barrier are blocked at the convergence barrier, then the scheduler unit 410 proceeds to step 630. Otherwise, the scheduler unit 410 proceeds directly to step 135. At step 630, the scheduler unit 410 clears the yielded state for any of the participating threads that executed a YIELD instruction and then proceeds to step 140. In one embodiment, the yielded state in the thread state 440 is changed from yielded to ready.

**当线程执行的指令访问内存（长延迟操作），在其后插入nanosleep指令短暂阻塞该线程，让其余线程执行，能提高资源使用率.**

**scheduler跟踪nanosleep指定的时间，计时满后将线程状态恢复成ready.**

However, when at least one thread is checking on a volatile value (i.e., polling), the polling consumes execution cycles. A NANOSLEEP instruction improves execution efficiency for threads to check on a volatile value and then be suspended for a specified duration to allow other threads to execute. The NANOSLEEP instruction enables the expression of a back-off routine that prevents severe performance degradation when a thread is spinning in a loop waiting for synchronization to occur. The scheduler unit 410 tracks the specified duration and suspends any threads that have executed the NANOSLEEP instruction, changing the suspended threads back to the ready state when the specified duration has transpired.

We note that the convergence barrier mechanism has been described from the perspective of multiple threads mapped onto a single SIMD datapath. The divergent thread execution technique can also be applied directly to a multicore processor with multiple datapaths, or a system that implements simultaneous multithreading or barrel processing where multiple sets of SIMD threads are scheduled onto a single SIMD datapath in MIMD fashion.