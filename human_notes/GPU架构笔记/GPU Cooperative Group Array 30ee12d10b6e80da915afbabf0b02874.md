# GPU Cooperative Group Array

ref：Cooperative Group Array

CGA API和对应支持机制（DSMEM）

CGA扩展了CTA在SM内协作和同步的模式，提供更大量线程在更大资源范围内的协作和同步。**CGA同时启动，独占更大范围的资源，提高重负载的性能。**

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

## Fig1A、1B、1C

每个SM是一个PE，执行一个kernel。

weak scaling：每个PE加任务，增加PE并发负载。

strong scaling：每个PE任务相同，增加并行PE。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%201.png)

## Fig2A、2B、2C

CGA前是全硬件的SM内同步机制。

CGA引入软件辅助的**同步**机制，高效支持更大范围的grid同步，访问存储更灵活高效。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> PE = Polymorph Engine
> 
> FIG. 2A
![image.png](GPU%20Cooperative%20Group%20Array/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 2B
![image.png](GPU%20Cooperative%20Group%20Array/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%204.png)

## Fig3、4、5、5A

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3
> 
> Prior Art GPU Hardware Partitions
![image.png](GPU%20Cooperative%20Group%20Array/image%205.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5
> Prior Art uGPU Partitions
![image.png](GPU%20Cooperative%20Group%20Array/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Prior Art GPU Hardware With Graphics Processing Clusters
![image.png](GPU%20Cooperative%20Group%20Array/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5A
![image.png](GPU%20Cooperative%20Group%20Array/image%208.png)

## Fig6、7

CGA的存储架构下，SM之间的数据交换更容易。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Strong Scaling
![image.png](GPU%20Cooperative%20Group%20Array/image%2010.png)

## Fig8、9、10、11、12

GPU分别看成**不同层次Tile（SM、GPC、GPU）组成的三维阵列**，不同层次Tile有不同并发容量，分别调度不同层次的CGA（grid-SM/GPC/GPU-CGA），将**grid分层映射到不同层次的不同CGA**。

**grid-CTA/grid-SM-CGA（CTA in SM）**

GPU是SM组成的三维阵列，每个CTA在一个SM上执行，定义CTA不超过SM容量。grid拆成CTA的三维阵列，分别调度到SM。

跨SM/grid协作时协作效率低，CTA映射的SM容易“乱飞”让DSMEM传输拥塞，中介L2Cache慢。

**grid-GPC-CGA（CTA in SM、GPC-CGA in GPC）**

GPU是GPC组成的三维阵列，每个GPC-CGA在一个GPC上执行，定义GPC-CGA不超过GPC容量。grid拆成GPC-CGA的三维阵列，分别调度到GPC。每个GPC-CGA拆成CTA的三维阵列，分别调度到自身所在GPC的SM。

跨SM/grid协作时协作效率适中，CGA映射的SM集中，DSMEM传输高效。

**grid-GPU-CGA（CTA in SM、GPU-CGA in GPU）**

GPU（cluster）是**GPU**组成的三维阵列，每个GPU-CGA在一个**GPU**上执行，定义GPU-CGA不超过**GPU**容量。grid拆成GPU-CGA的三维阵列，分别调度到GPU。每个GPU-CGA拆成CTA的三维阵列，分别调度到自身所在GPU的SM。

跨SM/grid协作时协作效率适中，CGA映射的SM集中，DSMEM传输高效。

**grid-GPU-GPC-CGA（CTA in SM、GPC-CGA in GPC、GPU-CGA in GPU）**

在grid-GPU-CGA的基础上，每个GPU-CGA在一个**GPU**上执行，定义GPU-CGA不超过**GPU**容量。grid拆成GPU-CGA的三维阵列，分别调度到GPU。。每个GPU-CGA拆成GPC-CGA的三维阵列，分别调度到GPC，每个GPC-CGA拆成CTA的三维阵列，分别调度到所在GPC的SM。

尽管协作效率提升，但同步开销仍然随着协作容量提升而提升。

kernel（1个grid）设计将关系紧密的CTAs（1个blk）组织为CGA（1个blk cluster），映射到访存效率更高的资源组（***-CGA）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Example Prior Art CTA Hierarchy
![image.png](GPU%20Cooperative%20Group%20Array/image%2011.png)

> **[图片提取文字 (image.png)]:**
> GPU but may run on different SMs GRID0 GRID1 **GPU0** GPU1 All threads in a CTA run cooperatively on same SM CTA0 CTA1 SM0 SM1 GPU0 All threads in a warp run Thread0 Core 1 Core 2 Threadk concurrently on Thread0 same SM ... SM<sub>0</sub> Warp0 Warpn Warp Scheduler(s)
> 
> All CTAs in a grid run on same
> 
> FIG. 9
> Prior Art Hierarchy
> Mapping Onto GPU Hardware Partitions
![image.png](GPU%20Cooperative%20Group%20Array/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 10A Prior Art Hierarchy
> 
> ![](_page_0_Picture_2.jpeg)
> 
> FIG. 10B New Hierarchy
![image.png](GPU%20Cooperative%20Group%20Array/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> **FIG. 11B**
![image.png](GPU%20Cooperative%20Group%20Array/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 12 New Grid Types
> 
> 12(3). GPU\_CGAs
> 
> 12(4). GPU\_CGAs of GPC\_CGAs
![image.png](GPU%20Cooperative%20Group%20Array/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ## FIG. 12A
> 
> Example: CTA C (X, Y, Z) = (7, 3, 0) CgaCtald = 4
![image.png](GPU%20Cooperative%20Group%20Array/image%2016.png)

## Fig13、14、15、16、17、18、19

DSMEM：分布式SM存储，用于SM之间数据交换。

> **[图片提取文字 (image.png)]:**
> hardware statement machine. Its functionality is expanded/enhanced to provide a speculative CGA launch capability to confirm that resources are available to launch all CTAs in a CGA. If all CTAs of a CGA cannot be launched at the same time, then the CWD 420 does not launch any of the CTAs of the CGA, but instead waits until sufficient resources of the relevant GPU hardware domain become available so that all CTAs of the CGA can be launched so they run concurrently. In example embodiments, the CWD 420 supports nesting of multiple levels of CGAs (e.g., multiple GPC-CGAs within a GPU-CGA) using a multi-level work distribution architecture to provide CGA launch on associated hardware affinity/domain.
> 
> [0158] In more detail, CWD 420 shown in FIG. 15A
> 
> and MPC for scheduling work. In an embodiment herein, the
> 
> CWD 420 comprises registers, combinatorial logic and a
> 
> speculative execution technique, that all CTAs of the CGA can fit on the hardware resources available in the specified hardware domain. In this way, CWD 420 in one example mode makes sure there are enough resources across all GPCs or other relevant hardware domain for all CTAs of the CGA before launching any. In one embodiment, the algorithm to launch CTAs of a CGA can borrow some techniques from legacy (non CGA) grid launch while first confirming that all CTAs of a CGA can be launched in a way that ensures they will run simultaneously.
> 
> [0159] FIG. 15A shows the basic architecture of CWD
> 
> launches the CTAs in a CGA after determining, using a
> 
> (TRTs) 425(0), 425(1), . . . 425(N-1), a TPC enable table 430, a local memory (LMEM) block index table 432, credit counters 434, a task table 436, and a priority-sorted task table 438. Each of the TRTs 425(0), 425(1), . . . 425(N-1) communicates with a corresponding TPC 340(0), 340(1), . . . 340(N-1). For more detail concerning legacy operation of these structures, see e.g., U.S. Pat. No. 10,817,338;
> 
> 420, which includes a load balancer 422, resource trackers
> 
> these structures, see e.g., U.S. Pat. No. 10,817,338; US20200043123; US20150178879; and U.S. Pat. No. 10,217,183. In example embodiments, functionality of these and other structures is enhanced in example embodiments along the following lines:
![image.png](GPU%20Cooperative%20Group%20Array/image%2017.png)

> **[图片提取文字 (image.png)]:**
> | Function/Operation                                                                                                                                                              | Units Enhanced                                                                                                                 |
> |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
> | Distributed CTA rasterization New launch packets for legacy grids/queues and CGAs Wider Bundles in compute pipe & new OMD format                                                | M-Pipe Controllers (MPCs) instead of CWD<br>Compute Work Distributor (CWD), GPM, MPC,<br>SM<br>Compute Pipe, CWD, GPM, MPC, SM |
> | Parallel load balancer for CGAs CTA complete bandwidth\nimprovements                                                                                                            | CWD<br>GPM, SMCARB                                                                                                             |
> | CGA tracking and barriers CGA completions and DSMEM flush New S2R registers Error handling for SM2SM traffic New GPC/TPC numbering Compute Instruction Level Preemption changes | CWD, GPM, MPC, SM<br>GPM, MPC, SM<br>SM<br>SM<br>CWD, GPM, MPC, SM, CTXSW<br>MPC, SM, Trap handler                             |
![image.png](GPU%20Cooperative%20Group%20Array/image%2018.png)

> **[图片提取文字 (image.png)]:**
> the CPU 212 can issues such tasks. For example, the CPU 212 can execute one or more programs stored in nontransitory memory such as global memory to generate CGA launch commands that command the GPU to launch CGA grids. [0161] In operation, CPU 212 executes a driver program (see FIG. 15C-2) that generates "grid launch" (and other) commands for the GPU. The grid launch command has associated state parameters that define a grid of CGAs to be executed by the GPU. In one embodiment, the state parameters include size parameters that specify the number of CGAs in the grid, the number of CTAs in each CGA, and the number of threads in each CTA (see FIG. 15C-2, block 552, 554). [0162] If the thread identifiers are multidimensional (e.g., 2-D, or 3-D), the size of the CTA in each dimension is specified; thus, the size might be specified as n0 for a CTA with 1-D thread IDs or as n0=d0\*d1\*d2 for a CTA with 3-D thread IDs. Similarly, if the CTA or CGA identifiers are multidimensional, the size of the grid in each dimension is specified. The state parameters also identify the CTA program to be executed by each thread, a source location (e.g., an array) in global memory (see FIG. 15) for input data for
> 
> accommodated). Each process or application executing on
> 
> the grid and a destination location (e.g., an array) in global memory for output data produced by the grid. See for example U.S. Pat. Nos. 7,937,567; 9,513,975; and 9,928,109 for background on how the CPU can launch grids using for example a thread-oriented programming environment such as the CUDA<sup>TM</sup> programming environment from NVIDIA<sup>TM</sup>. The CPU **212** also arranges for the threads to be executed by the SMs to be stored e.g., in global memory such that direct memory access hardware of the GPU can retrieve the threads through the system's memory management unit (MMU) for the SMs to execute (see FIG. **15**)
![image.png](GPU%20Cooperative%20Group%20Array/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 14A** 
> 
> Veid63
> 
> 1008..1023
> 
> CGA linear
> 
> memory slot
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Veid1
> 
> Veid0
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Veid2
![image.png](GPU%20Cooperative%20Group%20Array/image%2021.png)

CGA启动

> **[图片提取文字 (image.png)]:**
> [0163] Example CGA Launch Command [0164] In example embodiments, a launch command from the CPU 212 to CWD 420 may specify a CGA grid, which includes an enumeration of the various dimensions of the composite thread blocks and CGAs. As one example, a CGA grid launch command could specify to run 10240 CGAs, where each CGA is 8 CTAs, where each CTA has 256 threads, where each thread has (needs) 64 registers, and where each CTA allocates 128 KB of shared memory, etc. These numbers are encoded into a launch command like {10240, 8, 256, 64, 128}, and that is the information which the hardware work distributor CWD 420 processes when
![image.png](GPU%20Cooperative%20Group%20Array/image%2022.png)

> **[图片提取文字 (image.png)]:**
> (FIG. 15C-2, block 558). In another embodiment, the SMs can issue these commands to CWD 420, i.e., tasks described as being performed by CPU 212 can also be done by the SMs. [0165] Using the above technique, the application program can launch many small CGAs in a GPC or other hardware partition but the number diminishes as the size of the CGA grows. At a certain point (depending on the hardware platform), no CGA can fit in the GPC or other hardware partition anymore, which may compromise code portability. If one assumes that every platform has at least one GPC with 4 TPCs, the maximum CGA size that guarantees compatibility across future architectures is 8 CTAs. A given application program could dynamically adjust CGA size based on querying the platform to determine the number of CGAs that can run concurrently in the GPU as a function of 1) CTA resource requirements and 2) number of CTAs per CGA.
> 
> launching threads or CTAs on SMs. The CPU 212 sends
> 
> such launch commands to a scheduler 410 within the GPU
![image.png](GPU%20Cooperative%20Group%20Array/image%2023.png)

> **[图片提取文字 (image.png)]:**
> [0166] GPU CGA Scheduling & Launch [0167] In example embodiments, a scheduler 410 within the GPU receives tasks from the CPU 212 and sends them to the CWD 420 (FIG. 15C-1, blocks 502, 504). The CWD 420 queries and launches CTAs from multiple CGAs. In one embodiment, it works on one CGA at a time. For each CGA, CWD 420 speculatively launches all of the CTAs in the CGA, incrementing the "launch" registers to store the speculative launch. If all free slots in SMs or other processors in the hardware domain are exhausted before all CTAs of the CGA are speculatively launched, the CWD 420 terminates the launch and may try again later. If, in contrast, there are sufficient free slots for all CTAs in the CGA, the CWD 420 generates sm\_masks from the "launch" registers accumulated in the speculative launch process (this sm\_masks data structure stores reservation information for the number of CTAs to be run on each SM in the relevant hardware domain for the CGA launch), and moves on to a next CGA. The hardware allocates a CGA sequential number and attaches it to each sm\_mask. It also attaches an end of CGA bit to the last one to prevent interleaving of sm\_masks from different CGAs. [0168]Example CGA Launch Packet Based on a successful speculative launch, CWD **420** sends launch packets such as the following to the GPCs
> 
> (SMs). Such launch packets may for example include the
> 
> following fields:
![image.png](GPU%20Cooperative%20Group%20Array/image%2024.png)

> **[图片提取文字 (image.png)]:**
> | cwd2pdb_grid_cga_launch_sm_mask<br>task_id | // launch GPC CGA CTAs<br>// task identifier |
> |--------------------------------------------|----------------------------------------------|
> | vgpc_id                                    | // virtual GPC ID, which may be              |
> |                                            | based on the number of non-                  |
> |                                            | floorswept TPCs that exist per               |
> |                                            | GPC                                          |
> | sm_mask                                    | // bitmask of SMs; each bit                  |
> |                                            | signifies a CTA launch to a                  |
> |                                            | corresponding SM; also includes a            |
> |                                            | GPU CGA sequential number as                 |
> |                                            | discussed above                              |
> | sm_mask1,                                  | // further SM bitmask(s) if more             |
> | sm_mask2,                                  | than one CTA of the specified                |
> |                                            | CGA is assigned to run on the                |
> |                                            | same SM                                      |
> | num_ctas                                   | // # of 1's in sm_mask i.e. CTAs             |
> |                                            | first least significant bit with "1"         |
> |                                            | in sm_mask corresponds to first              |
![image.png](GPU%20Cooperative%20Group%20Array/image%2025.png)

> **[图片提取文字 (image.png)]:**
> CTA of GPC CGA and most significant bit with "1" in
> 
> sm\_mask corresponds to last CTA of GPC CGA // last packet of GPU CGA
> 
> -continued
> 
> last\_gpu\_cga
![image.png](GPU%20Cooperative%20Group%20Array/image%2026.png)

> **[图片提取文字 (image.png)]:**
> launch packet is used to broadcast them (with the associated CGA ID) to all SM work schedulers of the GPU. In one embodiment, the CPU 212 attaches a GPU CGA sequential number to the launch command it sends to the GPU. This sequential number is prepended to the sm\_masks generated for each GPC CGA and is used to map an sm\_mask of every GPC CGA to the GPU CGA (it may also be used by any reorder unit before sending masks to the M-Pipe Controllers (MPCs) within individual SMs). [0171] Broadcasting the launch packets to all SMs allows all MPCs within SMs to observe the entire sequence of CGA/CTA launches. By observing the stream of CGAs and CTAs, every SM's MPC (to which the grid is currently assigned) is able to carry out rasterization redundantly and independently. Also broadcast are lmem\_blk\_idx packets which carry lmem\_blk\_idx (see LMEM block index table
> 
> **432** of FIG. **15**A) from CWD **420** to the SMs.
> 
> [0170] CWD 420 may provide multiple iterative waves of
> 
> sm\_masks to map all CTAs in the CGA to SMs such that the
> 
> CGA can launch. Once the SM masks are ready, the above
![image.png](GPU%20Cooperative%20Group%20Array/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 15A
![image.png](GPU%20Cooperative%20Group%20Array/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig 15C-1
![image.png](GPU%20Cooperative%20Group%20Array/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2031.png)

地址空间

> **[图片提取文字 (image.png)]:**
> linear\_mem\_base. The buffer size in bytes (S) is called the CGA\_linear\_mem\_size. These values are both used by the shader code to calculate the virtual address of its shared memory region when executing. These values may be passed to the shader using constants, or compiled directly into the shader code. Here is an example equation the shader code could implement: Linear\_mem\_base\_for\_this\_ CGA=CGA\_linear\_mem\_base+(CGA\_linear\_memory\_ size\*CGA\_slot). [0236] Software is expected to allocate a buffer in video (global) memory per any arbitrary group of grids to serve as the CGA linear shared memory region for a given context. Conceptually this buffer is broken into N equal sized segments of S bytes as FIG. 14A shows. Each CGA that executes is given access to one of the N segments based on its CGA\_linear\_memory\_slot. S may be set based on the size of the shared memory a GPC-sized CGA needs, and N may be set based on the number of GPC-sized CGAs that are allowed to concurrently execute on the hardware domain. The total buffer size would then be N\*S bytes. As mentioned above, software can have multiple pools per context [0237] Assume that to run, each CGA in one example requires a 1-MB buffer in CGA linear memory. If the grid has 10,000 CGAs, then to run the entire grid would require 10,000 1-MB buffers. However, most platforms will not be able to run 10,000 CGAs all at the same time. Example embodiments take advantage of this to reduce memory footprint, by providing a pool of CGA linear shared memory, and having the CGAs declare how many can run at the same time. For example, if the CGAs in the grid declare that a maximum of N CGAs (N<10,000) can run on the platform at the same time, then a maximum of only N 1-MB buffers (not 10,000) need to be allocated. The hardware throttling tracks how many buffers have been allocated, and prevents the N+1 CGA from launching until a previously-launched CGA completes and frees its allocated buffer. In this way, software can limit the total number of CGAs that can execute concurrently and thereby limit the size of the overall
> 
> shared memory buffer.
> 
> [0235] The virtual address of the CGA linear shared
> 
> memory buffer in one embodiment is called the CGA\_
![image.png](GPU%20Cooperative%20Group%20Array/image%2032.png)

> **[图片提取文字 (image.png)]:**
> [0242] Example Overall Memory Map of "Generic" Memory Including Private, Shared & Global Memory [0243] FIG. 18 is a memory map of "generic" memory of one example embodiment. This memory map in one embodiment defines the memory address space of an SM and includes both virtual and physical addresses. In this example, the generic memory map thus defines a "generic" memory address space that includes global memory (including the CGA shared linear memory described above) and has a window into which local (DSMEM) memory is mapped.
![image.png](GPU%20Cooperative%20Group%20Array/image%2033.png)

> **[图片提取文字 (image.png)]:**
> Some of this local memory is private to an SM or to a task running on an SM, but other local memory in embodiments herein is shared or shareable with other tasks running on other SMs—namely other CTAs within the same CGA. [0244] In one embodiment, most of generic address space that is not shared memory is mapped to global memory. Such generic address space is thus made up of global memory and DSMEM in one embodiment. Other embodiments may include other special regions for other memory spaces such as thread-private stack memory for example. In this example, global memory is DRAM that is backed by an L2 cache. This global memory is thus the "main memory" of the system that an SM can access through the GPU's main memory management unit (MMU) to read and write data. Global memory may include for example frame buffer memory used to display images; program storage; data storage; texture storage; ray tracing BVH storage; and many other kinds of data including CGA linear shared memory. [0245] The FIG. 18 memory map further shows a DSMEM address memory block which is broken out on the right-hand side to show discontinuous blocks of shared
> 
> memory block was much smaller (e.g., 16 MB) and was mapped to "shared memory" that was logically part of each SM and could be shared between all threads executing on the SM but could not be shared by other processes running on a different SM. In other words, this "shared memory" was private to an SM and enabled an SM to access its own local memory and was termed "shared memory" because different threads or thread groups executing on the SM were able to share the memory and use it to exchange data. No capability was provided to enable one SM to share or access the shared memory of another SM. Different SMs would each see the same e.g., 16 KB of "shared memory" but that mapping enabled the particular SM to access only its own local shared
> 
> memory that was shared between thread groups or CTAs
> 
> running on that particular SM.
> 
> distributed memory. In prior architectures, this DSMEM
![image.png](GPU%20Cooperative%20Group%20Array/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 18
![image.png](GPU%20Cooperative%20Group%20Array/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 17C
![image.png](GPU%20Cooperative%20Group%20Array/image%2038.png)

地址空间

> **[图片提取文字 (image.png)]:**
> CIAs within the CGA. In one non-limiting example shown in FIG. 18, 256 such mappings can be activated to accommodate up to 32, 64, 128, 256 or any other number of CTAs in a GPC-CTA. Of course, particular hardware platforms may support fewer, more, or different numbers of CTAs per GPC-CTA as needed or desired. [0247] In one embodiment, the number of such regions the hardware allocates at any time is dependent on the actual number of CTAs in a CGA. Thus, if a CGA comprises 32 CTAs, then the hardware will allocate and enable 32 shared memory windows—one for each activated CTA in the CGA. Similarly, if a CGA includes only 23 CTAs, then the hardware will allocate and enable 23 such shared memory windows. The hardware could dynamically allocate/deallocate such shared memory windows as additional CTA launch/complete consistent with the concurrent execution guarantee discussed above. [0248] The load, store, and atomic instructions an SM executes can be indexed by the CTA as certain bits in the
> 
> address to select the shared memory region, and additional
> 
> [0246] In example embodiments herein, this "shared
> 
> memory" window has now been expanded to include a
> 
> mapping for other (and in one embodiment, every) CTA in
> 
> a GPC-CGA. In other words, the local memory window of
> 
> "shared memory" has been expanded to allow access to
> 
> portions of the local memories of all other SMs running (or
> 
> in some embodiments, which could run) thread groups or
![image.png](GPU%20Cooperative%20Group%20Array/image%2039.png)

> **[图片提取文字 (image.png)]:**
> (e.g., lower order) address bits that specific a particular location within that particular CTA's shared memory region. An example shared memory address could thus look like the following for LDS and STS instructions accessing shared memory:
> 
> | [0249] Such an addressing arrangement can provide back- |
> |---------------------------------------------------------|
> | wards compatibility to CGA-unaware code by setting the  |
> | "CTA ID within CGA" to zero (the CTA can read an S-to-R |
> 
> SMEM Offset
> 
> CTA ID within CGA
> 
> hardware register to determine which CTA ID is assigned to it) to thereby allow a CTA running on an SM to address the CTA's own shared memory local to that particular SM. The legacy usage is thus supported and is expanded to permit CTAs to access distributed shared memory of other CTAs within the CGA they are all grouped within.
> 
> [0250] Meanwhile, the following format may be used to permit an SM to issue LD, ST, and atomic instructions to access shared memory:
> 
> | 0 DSMEM/SMEM CTA ID within SMEM Offs<br>Aperture CGA | et |
> |------------------------------------------------------|----|
> |------------------------------------------------------|----|
![image.png](GPU%20Cooperative%20Group%20Array/image%2040.png)

> **[图片提取文字 (image.png)]:**
> [0264] DSMEM Mapping Tables FIG. 19 shows an example DSMEM mapping table arrangement maintained by each SM. In one embodiment, the SM determines the target based on the segmented address and then choses the correct packet type to let the interconnect know this is a SM2SM transaction, and provides the physical SM id based on lookup in the routing table as shown in FIG. 19. In one embodiment, the SM maps the logical CTA ID within the GPC\_CGA to the physical SM on which the CTA is running, and that CTA's physical shared memory on the SM. Each time a CTA launches, all of the SMs on the GPC may need to know about it because any one of those SMs might be executing a CTA that is part of the same CGA. In one embodiment, MPC informs (broadcasts a message to) all of the SMs each time a new CTA is launched. In response, each SM updates the mapping table it maintains. In one embodiment, a CAM structure is used for this mapping to allow DSMEM addressing from remote (other) SMs. As FIG. 19 shows, the CAM structure is stored in RAM as an SM-to-SM mapping table 5004 that is indexed by a SMCGAslot value. Mapping table 5004 identifies to the SM which other SMs the other CTAs in the CGA are executing on. Pseudocode defining the 5004 example table is shown below:
> 
> ```
> CTA_ID -> CGA_ID, SM_ID, and TPC_ID
> 
> // Directory to find SM ID in GPC from CTA ID in CGA
> 
> // Source SM looks up this directory to find destination SM
> 
> struct {
> 
>             U008 gpc_local_cga_id;
> ```
![image.png](GPU%20Cooperative%20Group%20Array/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 19
![image.png](GPU%20Cooperative%20Group%20Array/image%2042.png)

## Fig20、21、22、23

SM在DSMEM上的数据交换机制。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 21A
![image.png](GPU%20Cooperative%20Group%20Array/image%2044.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Cooperative%20Group%20Array/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 21C
![image.png](GPU%20Cooperative%20Group%20Array/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 21D
![image.png](GPU%20Cooperative%20Group%20Array/image%2048.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 22
![image.png](GPU%20Cooperative%20Group%20Array/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 23A**
![image.png](GPU%20Cooperative%20Group%20Array/image%2050.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## FIG. 23B
![image.png](GPU%20Cooperative%20Group%20Array/image%2051.png)