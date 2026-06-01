# GPU Multi-Context Engine（MPS>MIG和SME轮转不同Stream，不同Stream是不同Ctx）

ref：TECHNIQUES FOR CONFIGURING A PROCESSOR TO FUNCTION AS MULTIPLE , SEPARATE PROCESSORS

Ctx定义了任务需要的资源和其所处的目标状态（共享、隔离）。

MPS是？？？父亲进程的很多子进程共享父进程Ctx，子进程消耗资源不可调度（GPU眼中只有一个父进程channel），任一子进程的异常处理影响全部子进程，即MPS互相性能和异常干扰。

**MIG和SME提供更好的隔离性。**

> **[图片提取文字 (image.png)]:**
> doing, the CPU process generates a processing context on the GPU that specifies a target state for the various GPU resources that are to be implemented to perform the processing task. Those GPU resources may include processing, graphics, and memory resources, among others. The CPU process then launches a set of threads on the GPU in accordance with the processing context, and the set of threads utilizes the various GPU resources to perform the processing task. In many of these types of implementations, the GPU is configured according to only one processing context at a time. However, in some situations, the CPU needs to offload more than one CPU process to the GPU during the same interval of time. In such situations, the CPU can dynamically change the processing context implemented on the GPU at different points in time in order to service those CPU processes serially across the interval of time. One drawback of this approach, however, is that the processing tasks offloaded by certain CPU processes do not fully utilize the resources of the GPU. Consequently, when one or more processing tasks associated with those CPU
> 
> [0003] In some implementations, a CPU process executing
> 
> on a CPU can offload a given processing task to a GPU in
> 
> order to have that processing task performed faster. In so
> 
> One approach to executing multiple CPU processes simultaneously on a GPU is to generate multiple different processing subcontexts within a given "parent" processing context and to assign each different processing subcontext to a different CPU process. Multiple CPU processes can then launch different sets of threads on the GPU simultaneously, where each set of threads utilizes specific GPU resources that are configured according to a specific processing subcontext. With this approach, the GPU can be more efficiently
> 
> processes are performed serially on the GPU, some GPU
> 
> resources can go unused, which reduces the overall GPU
> 
> performance and utilization.
> 
> utilized because more than one CPU process can offload processing tasks to the GPU at the same point in time, potentially avoiding situations where some GPU resources go unused.
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image.png)

> **[图片提取文字 (image.png)]:**
> [0005] One problem with the above approach is that CPU processes associated with different processing subcontexts can unfairly consume GPU resources that should be more evenly allocated or distributed across the different processing subcontexts. For example, a first CPU process could launch a first set of threads within a first processing subcontext that performs a large volume of read requests and consumes a large amount of available GPU memory bandwidth. A second CPU process could subsequently launch a second set of threads within a second processing subcontext that also performs a large volume of read requests. However, because much of the available GPU memory bandwidth is already being consumed by the first set of threads, the second set of threads could experience high latencies, which could cause the second CPU process to stall. [0006] Another problem with the above approach is that,
> 
> because processing subcontexts share a parent context, any
> 
> faults occurring when the threads associated with one processing subcontext execute can interfere with the execution of other threads associated with another processing subcontext sharing the same parent context. For example, a first CPU process could launch a first set of threads associated with a first processing subcontext to perform a first processing task. A second CPU process could launch a second set of threads associated with a second processing subcontext, and the second set of threads could subsequently experience a fault and fail. To recover from the failure, the GPU would have to reset the parent context, which would automatically reset both the first processing subcontext and the second processing subcontext. In such a scenario, the execution of the first set of threads would be disrupted even though the fault arose from the second set of threads, not the first set of threads.
> 
> [0007] As the foregoing illustrates, what is needed in the art are more effective techniques for configuring a GPU to execute processing tasks associated with multiple contexts.
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%201.png)

## Fig1、2、3、4

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 1
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 2
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%205.png)

## Fig5、6、7、8

timeslice ctx switch，DRAM Arch for MIG（SMC Engine） partition。SME的划分标准是SysPipe，接受一个应用的channels或一个TSG。

FECS：前端Ctx切换控制，GPU Slice的Channel Ctx切换。

GPC逻辑划分到SME（可扩展到vGPC便于迁移），SysPipe物理划分到SME，SysPipe派发CTA到GPC之间的路由**Route**由Crossbar和SMC Arbiter设置。

CPU通过特权寄存器Pri配置GPU，Pri Hub映射/划分全局Pri地址空间到不同SysPipe中Pri的地址空间，Pri地址空间管理Pri寄存器。

> **[图片提取文字 (image.png)]:**
> 520 and 522, a front-end context switch (FECS) 530, a compute (COMP) front end (FE) 540, a scheduler (SKED) 550, and a CUDA work distributor (CWD) 560. PBDMAs 520 and 522 are hardware memory controllers that manage communications between device driver 122 and PPU 200. FECS 530 is a hardware unit that manages context switches. Compute FE 540 is a hardware unit that prepares processing compute tasks for execution. SKED 550 is a hardware unit that schedules processing tasks for execution. CWD 560 is a hardware unit configured to queue and dispatch one or more grids of threads to one or more GPCs 242 to execute one or more processing tasks. In one embodiment, a given processing task may be specified in a CUDA program. Via the above components, sys pipes 230 can be configured to perform and/or manage general-purpose compute opera-
> 
> tions.
> 
> [0095] Each sys pipe 230 generally includes PBDMAs
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 500
> 
> FIGURE 5
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 6
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%208.png)

> **[图片提取文字 (image.png)]:**
> [0097] Control crossbar and SMC arbiter 510 facilitates communications between sys pipes 230 and GPCs 242. In some configurations, one or more specific GPCs 242 are programmably assigned to perform processing tasks on behalf of a specific sys pipe 230. In such configurations, control crossbar and SMC arbiter 510 is configured to route data between any given GPC(s) 242 and the corresponding sys pipe(s) 220. PRI hub 512 provides access, by the CPU 110 and/or PPU 200 units, to a set of privileged registers to control configuration of the PPU 200. The register address space with the PPU 200 can be configured by a PRI register, and, in so doing, PRI hub 212 is used to configure the mapping of PRI register addresses between a generic PRI address space and a PRI address space defined separately for each sys pipe 230. This PRI address space configuration provides for broadcasting to multiple PRI registers from SMC engines described below in conjunction with FIG. 7. GPCs 242 write data to and read data from L2 cache 400 via crossbar unit 250 in the manner described previously. In some configurations, each GPC 242 is allocated a separate set of L2 slices derived from L2 cache 400 and any given GPC 242 can perform write/read operations with the corresponding set of L2 slices.
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 7
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2010.png)

> **[图片提取文字 (image.png)]:**
> [0107] A given PPU partition 600 can be configured to simultaneously execute processing tasks associated with multiple processing contexts. The term "processing context" or "context" generally refers to the state of hardware, software, and/or memory resources during execution of one or more threads, and generally corresponds to one process on CPU 110. The multiple processing contexts associated with a given PPU partition 600 can be different processing contexts or different instances of the same processing context. When configured in this manner, specific PPU resources allocated to the given PPU partition 600 are logically grouped into separate "SMC engines" that execute separate processing tasks associated with separate processing contexts, as described in greater detail below in conjunction with FIG. 7. For example, a given processing context could include hardware settings, per-thread instructions, and/or register contents associated with threads, that executes within an SMC Engine 700. 1 01
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2013.png)

## Fig9、10、11、12、13

host分隔GPU到多块，插板法配置

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 9
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2014.png)

> **[图片提取文字 (image.png)]:**
> FIGURE 10
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 11
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 13
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 12
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2018.png)

## Fig14、15、16、17

多VM执行，地址划分，GPU/GPC/TPC迁移

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 14A
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 14B
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 15
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2022.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2023.png)

## Fig18、19、20、21、22、23、24

资源分隔、内存管理支持

> **[图片提取文字 (image.png)]:**
> [0089] In any of the above configurations, one or more PMs 360 monitor the performance of the various components of GPC 242 in order to provide performance data to users, and/or balance the utilization of compute, graphics, and/or memory resources across groups of threads, and/or balance the utilization of those resources with that of other GPCs 242. Further, in any of the above configurations, SM 332 and other components within GPC 242 may perform memory access operations with memory interface 260 via MMU 300. MMU 300 generally writes output data to various memory spaces and/or reads input data from various memory spaces on behalf GPC 242 and the components included therein. MMU 300 is configured to map virtual addresses into physical addresses via a set of page table entries (PTEs) and one or more optional address translation lookaside buffers (TLBs). MMU 300 can cache various data in L1.5 cache 350, including read data returned from memory interface 260. In the embodiment shown, MMU 300 is coupled externally to GPC 242 and may potentially be shared with other GPCs 242. In other embodiments, GPC 242 may include a dedicated instance of MMU 300 that provides access to one or more partition units 262 included in memory interface 260.
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2024.png)

> **[图片提取文字 (image.png)]:**
> [0108] FIG. 7 illustrates an example of how the hypervisor of FIG. 1 configures a set of PPU partitions to implement one or more simultaneous multiple context (SMC) engines, according to various embodiments. As shown, PPU partitions 600 include one or more SMC engines 700. In particular, PPU partition 600(0) includes SMC engines 700(0) and 700(2), PPU partition 600(4) includes SMC engine 700(4), PPU partition 600(6) includes SMC engine 700(6), and PPU partition 600(7) includes SMC engine 700(7). Each SMC engine 700 can be configured to execute one or more
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2025.png)

> **[图片提取文字 (image.png)]:**
> processing contexts and/or be configured to execute one or more processing tasks associated with a given processing context, in like fashion to PPU **200** as a whole.
> 
> [0109] A given SMC engine 700 generally includes compute and memory resources associated with at least one PPU slice 610. For example, SMC engines 700(6) and 700(7) include the compute and memory resources associated with PPU slices 610(6) and 610(7), respectively. Each SMC engine 700 also includes a set of virtual engine identifiers (VEIDs) 702 that locally reference one or more subcontexts, where a VEID is associated with, and may be identical to, a virtual address space identifier, used to select a virtual address space, where the pages of the virtual address spaces are described by page tables managed by the MMU 1600. A given SMC engine 700 can also include compute and memory resources associated with multiple PPU slices 610. For example, SMC engine 700(0) includes the compute resources associated with PPU slices 610(0) and 610(1), but does not utilize sys pipe 230(1). SMC engine 700(0)includes and utilizes the L2 slices in four PPU slices 610(0), **610(1)**, **610(2)**, and **610(3)**. In some embodiments, SMC engines 700 within the same PPU partition 600 share the L2 Slices within the PPU partition 600. In this configuration, the sys pipe 230(1) of PPU partition 600(1) is unused, as is shown, because an SMC engine generally runs one processing context at time, and only one sys pipe 230 is needed for one processing context. SMC engine 700(2) is configured in like fashion to SMC engine 700(0). The memory resources included within any particular PPU partition 600, which can be allocated to and/or distributed across any one or more SMC engines 700 within that particular PPU partition 700, are shown as PPU memory partitions 710.
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2026.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 19
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 20
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2029.png)

global virtual、sub virtual

> **[图片提取文字 (image.png)]:**
> [0189] FIG. 21 illustrates how the memory management unit of FIG. 16 provides access to different PPU memory partitions, according to various embodiments. As shown, MMU 1600 of FIG. 16 is coupled between DRAM 272 and 1D SPA space 850. 1D SPA space 850 is divided into top addresses 852 that correspond to top section 810, partitionable addresses 854 that correspond to partitionable section 820, and bottom addresses 856 that correspond to bottom section 830, as also shown in FIG. 8B. During partitioning, hypervisor 124 generates 1D SPA space 850 based on the configuration of DRAM 272.
> 
> [0190] MMU 1600 includes an address mapping unit (AMAP) 2110 that is configured to map top addresses 852, partitionable addresses 854, and bottom addresses 858 into raw addresses associated with top section 810, partitionable section 820, and bottom section 830, respectively. In this manner, MMU 1600 services memory access requests received from hypervisor 124 that target top section 810 and/or bottom section 830 as well as memory access requests received from SMC engines 700 that target partitionable section 820, as described in greater detail below in conjunction with FIG. 22.
> 
> [0191] FIG. 22 illustrates how the memory management unit of FIG. 16 performs various address translations, according to various embodiments. As shown, partitionable addresses 854 include address region 856(0) that includes
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2030.png)

> **[图片提取文字 (image.png)]:**
> addresses corresponding to PPU memory partition **710(0)**, as discussed above in conjunction with FIG. **8**B. MMU **1600** translates physical addresses included in address region **856(0)** into raw addresses associated with DRAM portion **822(0)** via AMAP **2110**. AMAP **2110** is configured to swizzle addresses from address region **856(0)** across L2 cache slices **800(0)** included in PPU memory partition **710(0)** in order to avoid situations where striding causes the same L2 cache slice **800(0)** to be accessed repeatedly (also known as "camping").
> 
> [0192] In one embodiment, AMAP 2110 may implement a "memory access" swizID that identifies a memory interleave factor for a given region of memory. A given memory access swizID determines a set of L2 cache slices that are interleaved across for various types of memory accesses, including video memory, system memory, and peer memory access. Different PPU partitions 600 generally implement different and non-overlapping memory regions 822 within the partitionable section 829 to minimize interference between concurrently executing jobs. Hypervisor 124 may use a memory access swizID of zero in order to balance memory access operations across L2 cache slices, which would generally access either top section 810 or bottom section 830
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2031.png)

> **[图片提取文字 (image.png)]:**
> [0195] MMU 1600 also provides support for translating virtual addresses associated with a virtual address space identifier 1500 into a system physical address in the 1D system physical address space 850. For example, suppose SMC engine 700(0) of FIG. 7 executes using PPU memory partition 710(0) and corresponding DRAM portion 822(0), and, in doing so, causes a memory fault. MMU 1600 would issue a fault, with a local fault identifier 1610. MMU 1600 would in turn translate local fault identifier 1610 into a global fault identifier 1620. Fault and errors can be reported to virtual functions according to the SR-IOV public specification.
> 
> [0196] MMU 1600 also facilitates subdividing address regions 856 and PPU memory partitions 710 to provide
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2032.png)

> **[图片提取文字 (image.png)]:**
> [0197] FIG. 23 illustrates how the memory management unit of FIG. 16 provides support for operations associated with multiple processing contexts simultaneously, according to various embodiments. As shown, address region 856(0)encompasses multiple virtual memory pages 2310 of varying sizes. For SMC Engine 700(0), a virtual memory space identifier 1500 is mapped to a global virtual memory space identifier 1510 that selects the page table for a particular virtual address space being used by a processing context on SMC Engine 700(0). Pages specified by a page table A select pages 2310(A) within DRAM portion 822(0). Simultaneously, SMC Engine 700(2) can use pages specified by a page table B that selects pages 2310(B) also within DRAM portion 822(0). By a page-based virtual memory management scheme, pages within DRAM portion 822(0) can be allocated to different subcontexts or to different processing contexts. Note that a processing context can use multiple virtual address space identifiers 1500 because it can execute many subcontexts.
> 
> [0198] Subdividing address region 856(0) and DRAM portion 822(0) corresponding to PPU memory partitions 710(0) in the manner shown provides different SMC engines 700 that execute within a corresponding PPU partition 600 with dedicated memory resources within PPU memory partition 822(0). Accordingly, multiple SMC engines 700 in different PPU 600 partitions can execute processing tasks within different processing contexts simultaneously without interfering with one another in terms of bandwidth.
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2033.png)

> **[图片提取文字 (image.png)]:**
> 858
> 
> FIGURE 21
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> FIGURE 23
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2035.png)

> **[图片提取文字 (image.png)]:**
> FIGURE 22
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ## DETERMINE MEMORY CONFIGURATION PARAMETERS
> 
> 2402
> 
> ACTIVATE FIRST SET OF BOUNDARY OPTIONS BASED ON MEMORY CONFIGURATION PARAMETERS TO DIVIDE L2 CACHE INTO SECTIONS
> 
> 2404
> 
> DETERMINE A CONFIGURATION SWIZID BASED ON PARTITIONING INPUT
> 
> - 2406
> 
> ACTIVATE SECOND SET OF BOUNDARY OPTIONS BASED ON CONFIGURATION SWIZID TO GENERATE PPU MEMORY PARTITIONS WITHIN PARTITIONABLE SECTION
> 
> - 2408
> 
> DETERMINE SET OF PARTITIONABLE ADDRESSES BASED ON MEMORY CONFIGURATION PARAMETERS
> 
> - 2410
> 
> SWIZZLE PARTITIONABLE ADDRESSES ACROSS A
> PPU MEMORY PARTITION BASED ON MEMORY
> ACCESS SWIZID ASSOCIATED WITH PPU PARTITION
> 
> - 2412
> 
> TRANSLATE LOCAL FAULT ID ASSOCIATED WITH PPU MEMORY PARTITION INTO GLOBAL FAULT ID ASSOCIATED WITH L2 CACHE
> 
> - 2414
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2037.png)

## Fig25、26、27（*）

MIG运行VM负载，VM Ctx包含内部所有的SME Ctx和Memory Ctx。

SME运行runlist负载，SME轮转TSG的顺序是runlist，TSG内应用可能占用多个channel，不同TSG分时共享SME Ctx（SME中所有channel）。

> **[图片提取文字 (image.png)]:**
> cessing contexts listed on a runlist, as managed by the PBDMA 520 and 522 of the SMC engine 700. In general, when switching between VMs, the runlists on all affected SMC engines 700 are replaced, so that a different set of processing contexts are time-sliced. If multiple SMC engines 700 are active, then the runlists replaced at the same time. This type of scheduling via runlist replacement is referred to herein as "software scheduling." Switching between processing contexts within the same VM similarly may also involve replacing runlists, and is very similar to switching VMs, except that the VM does not change as a result of the context switch. As a result, no additional hardware support is needed for this type of context switching. In some embodiments, VMs may have numerous distinct processing contexts of various sizes. In such embodiments, software scheduling may consider how to pack these processing contexts into SMC engines 700 for correct and efficient execution. Further, software scheduling may reconfigure the number of PPU partitions 600 and the number of SMC engines within each PPU partition 600 to correctly and
> 
> efficiently execute the processing contexts for the VM.
> 
> [0212] Each SMC engine 700 time-slices between pro-
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2038.png)

> **[图片提取文字 (image.png)]:**
> ing to various embodiments. As shown, guest OS 916 includes various processing contexts 1400 associated with different PPU partitions 600. Processing contexts 1400(0) and 1400(1) are associated with PPU partition 600(0) and can be launched on either SMC engine 700(0) or SMC engine 700(1). In some embodiments, once a processing context is assigned to an SMC engine 700, it remains on that smc engine 700 until completion. Processing context 1400 (4) is associated with PPU partition 600(4) and can be launched on SMC engine 700(4). Processing contexts 1400 (6) and 1400(6) are associated with PPU partitions 600(6) and 600(7), respectively, and can be launched on SMC engines 700(6) and 700(7), respectively.
> 
> simultaneously within one or more PPU partitions, accord-
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2039.png)

VM-level timeslicing（25、26）将**时间片分配给不同VM**，切换VM内所有runlist：

25中，t1时刻VM中断，**partition重置**（SME合并），支持1个Ctx更大的runlist，t4时刻VM中断，SME拆分，支持2个Ctx更小的runlist；

26中，t2时刻partition空闲，**partition重划分**（4、6、7作merge），支持更大Ctx的runlist。

SMC-level timeslice（27）将**时间片分配给关键runlists**，切换关键runlist：

27中，**时间片足够完成关键runlists**，合并后移除关键runlist（t1-t2，t3-t4，t5-t6），切换runlist或VM。

Fast ReCfg：划分内的FECS和GPC统一load配置，之后统一init。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 25
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2040.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 2700
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2042.png)

## Fig28、29、30（*）

VM的GPU迁移（28、29）：5个VM轮流在4个PPU迁移执行，负载均衡，每个VM平均使用80%的单个PPU时间。

time-slice VM流程（30）：PPU内ReCfg、PPU之间ReCfg、抢占目标PPU的目标VM的资源。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 28
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  |  |
> |----------|----------|----------|----------|----------|----------|----------|--|
> | 2910(C0) | 2910(D1) | 2910(E2) | 2910(A3) | 2910(B3) | 2910(C4) | 2910(D5) |  |
> 
> 2902(1)
> 
> | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  |  |
> |----------|----------|----------|----------|----------|----------|----------|--|
> | 2910(D0) | 2910(E1) | 2910(A2) | 2910(B2) | 2910(C3) | 2910(D4) | 2910(E5) |  |
> 
> 2902(2)
> 
> | Γ | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  |  |
> |---|----------|----------|----------|----------|----------|----------|----------|--|
> |   | 2910(E0) | 2910(A1) | 2910(B1) | 2910(C2) | 2910(D3) | 2910(E4) | 2910(A5) |  |
> 
> 2902(3)
> 
> | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  | CONTEXT  |
> |----------|----------|----------|----------|----------|----------|----------|----------|
> | 2910(A0) | 2910(B0) | 2910(C1) | 2910(D2) | 2910(E3) | 2910(A4) | 2910(B4) | 2910(C5) |
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2044.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> - 3000
> 
> FIGURE 30A
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2046.png)

## Fig31、32（Privileged Reg）

memory Mapping：Bar0对应权限Regs（31、32）

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIGURE 32
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2048.png)

## Fig33、34、35、36、37、38、39

性能监控PM（33、34、35、36、37）

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 3300 FIGURE 33
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2050.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2052.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIGURE 36
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ## GENERATE TRIGGER TO PM(S) AND CORRESPONDING PMA RECORD TO SAMPLE PERFORMANCE DATA
> 
> 3702
> 
> PM(S) DETERMINE WHETHER A RESPONSE TO THE TRIGGER IS WARRANTED
> 
> 3704
> 
> RECEIVE SAMPLED PERFORMANCE DATA FROM PM(S)
> 
> 3706
> 
> SORT RECEIVED PERFORMANCE DATA BY SMC ENGINE AND SHARED COMPONENTS
> 
> 3708
> 
> STORE PMA RECORDS AND/OR PMM RECORDS INTO A PMA RECORD BUFFER IN PPU MEMORY
> 
> 3710
> 
> TRANSMIT PMA RECORDS AND/OR PMM RECORDS
> TO PERFORMANCE ANALYSIS WORKSTATION
> 
> -3712
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2054.png)

功耗和时钟监控（38、39）

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2055.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Multi-Context%20Engine%EF%BC%88MPS%20MIG%E5%92%8CSME%E8%BD%AE%E8%BD%AC%E4%B8%8D%E5%90%8CStream%EF%BC%8C%E4%B8%8D%E5%90%8CS/image%2056.png)