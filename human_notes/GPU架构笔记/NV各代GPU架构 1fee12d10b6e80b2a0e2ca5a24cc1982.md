# NV各代GPU架构

## 目录

**架构对比、CUDA模型、统一内存管理**

NVIDIA GPU ARCHITECTURE:FROM TURING TO BLACKWELL

Evolution of Nvidia GPU from microarchitectures Pascal to Ampere

**fermi**

NVIDIA’s Fermi: The First Complete GPU Computing Architecture

NVIDIA’s Next Generation CUDA Compute Architecture: Fermi

FERMI GF100, A GRAPHICS PROCESSING UNIT (GPU) ARCHITECTURE FOR COMPUTE, TESSELLATION, PHYSICS, AND COMPUTATIONAL GRAPHICS

Whitepaper NVIDIA GF100 World’s Fastest GPU Delivering Great Gaming Performance with True Geometric Realism

**kepler**

Technology Overview NVIDIA GeForce GTX 680 The fastest, most efficient GPU ever built.

Whitepaper NVIDIA’s Next Generation CUDA Compute Architecture Kepler GK110/210

**maxwell**

Whitepaper NVIDIA GeForce GTX 980 Featuring Maxwell, The Most Advanced GPU Ever Made.

Whitepaper NVIDIA GeForce GTX 750 Ti Featuring First-Generation Maxwell GPU Technology, Designed for Extreme Performance per Watt

**pascal**

NVIDIA Tesla P100 *The Most Advanced Datacenter Accelerator Ever Built Featuring Pascal GP100, the World’s Fastest GPU*

Whitepaper NVIDIA GeForce GTX 1080

**Volta**

NVIDIA TESLA V100 GPU ARCHITECTURE *THE WORLD’S MOST ADVANCED DATA CENTER GPU*

**Turing**

NVIDIA TURING GPU ARCHITECTURE *Graphics Reinvented*

RTX ON – THE NVIDIA TURING GPU

**Ampere**

NVIDIA A100 Tensor Core GPU Architecture UNPRECEDENTED ACCELERATION AT EVERY SCALE

NVIDIA AMPERE GA102 GPU ARCHITECTURE *Second-Generation RTX*

**Hopper**

NVIDIA GH200 Grace Hopper Superchip Architecture

NVIDIA H100 Tensor Core GPU Architecture EXCEPTIONAL PERFORMANCE, SCALABILITY, AND SECURITY FOR THE DATA CENTER

**Blackwell**

NVIDIA Blackwell Architecture Technical Brief Built for the Age of AI Reasoning

NVIDIA RTX BLACKWELL GPU ARCHITECTURE Built for Neural Rendering

[https://medium.com/@kvnagesh/nvidia-blackwell-architecture-a-deep-dive-into-the-next-generation-of-ai-computing-79c2b1ce3c1b](https://medium.com/@kvnagesh/nvidia-blackwell-architecture-a-deep-dive-into-the-next-generation-of-ai-computing-79c2b1ce3c1b)

**Rubin**

GTC25

## CUDA、统一内存、NVLink

CUDA是使用GPU的SIMT执行模型来执行并行计算的多线程编程模型。

统一地址空间的目的是使用统一地址识别数据位于的层次和device。

统一内存的目的是让CPU和GPU的内存构成内存池，使用统一地址“自动地”访问不同device的内存数据，缺页时由runtime自动传输数据page，而不需要应用手动管理内存一致性。

NVLink的目的是让GPU之间、CPU和GPU之间的内存传输更快。

统一地址空间，不等于虚拟地址空间。虚拟是说不同应用有独立的地址空间，经过MMU映射到物理页。统一地址空间是把不同物理资源整合成统一资源池的物理页。

## 图形加速器和Fermi（CUDA、统一地址空间）

GeForce3是**3D图形pipeline加速器**，pipeline不同阶段是不同的处理核心群。

GeForce8800：最早支持**统一编程框架CUDA**的GPU，使用统一核心和基于C语言的GPU编程支持图形pipeline的所有阶段。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3. The GeForce 7800 had three kinds of programmable engines for different stages of the 3D pipeline plus several additional stages of configurable and fixed-function logic. (Source: NVIDIA)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4. The GeForce 8800 introduced a unified shader architecture with just one kind of programmable processing element that could be used for multiple purposes. Some simple graphics operations still used special-purpose logic. (Source: NVIDIA)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%201.png)

**Fermi（GF100）**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2. GF100 Architectural Overview
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%202.png)

> **[图片提取文字 (image.png)]:**
> At any one time, the entire Fermi device is dedicated to a single application. As mentioned above, an application may include multiple kernels. Fermi supports simultaneous execution of multiple kernels from the same application, each kernel being distributed to one or more SMs on the device. This capability avoids the situation where a kernel is only able to use part of the device and the rest goes unused.
> 
> Switching from one application to another is about 20 times faster on Fermi (just 25 microseconds) than on previous-generation GPUs. This time is short enough that a Fermi GPU can still maintain high utilization even when running multiple applications, like a mix of compute code and graphics code. Efficient multitasking is important for consumers (e.g., for video games using physics-based effects) and professional users (who often need to run computationally intensive simulations and simultaneously visualize the results).
> 
> This switching is managed by the chip-level GigaThread hardware thread scheduler, which manages 1,536 simultaneously active threads for each streaming multiprocessor across 16 kernels.
> 
> This centralized scheduler is another point of departure from conventional CPU design. In a multicore or multiprocessor server, no one CPU is "in charge". All tasks, including the operating system's kernel itself, may be run on any available CPU. This approach allows each operating system to follow a different philosophy in kernel design, from large monolithic kernels like Linux's to the microkernel design of QNX and hybrid designs like Windows 7. But the generality of this approach is also its weakness, because it requires complex CPUs to spend time and energy performing functions that could also be handled by much simpler hardware.
> 
> With Fermi, the intended applications, principles of stream processing, and the kernel and thread model, were all known in advance so that a more efficient scheduling method could be implemented in the GigaThread engine.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%203.png)

CUDA app、grid、block、thd，Fermi的设计初心包含**加速Tessellation的新pipeline**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## Thread Block per-Block Shared Memory
> 
> ![](_page_0_Figure_2.jpeg)
> 
> CUDA Hierarchy of threads, blocks, and grids, with corresponding per-thread private, per-block shared, and per-application global memory spaces.
> 
> A thread block is a set of concurrently executing threads that can cooperate among themselves through barrier synchronization and shared memory. A thread block has a block ID within its grid.
> 
> A grid is an array of thread blocks that execute the same kernel, read inputs from global memory, write results to global memory, and synchronize between dependent kernel calls. In the CUDA parallel programming model, each thread has a per-thread private memory space used for register spills, function calls, and C automatic array variables. Each thread block has a per-Block shared memory space used for inter-thread communication, data sharing, and result sharing in parallel algorithms. Grids of thread blocks share results in Global Memory space after kernel-wide global
> 
> synchronization.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6 Tessellation in DX11 showing new pipeline stages
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%205.png)

面向VM的执行模型，设计ISA（PTX）和GPU架构支持，便于迭代，VM类似软工中的“接口”。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 7. A total of 32 instructions from one or two warps can be dispatched in each cycle to any two of the four execution blocks within a Fermi SM: two blocks of 16 cores each, one block of four Special Function Units, and one block of 16 load/store units. This figure shows how instructions are issued to the execution blocks. (Source: NVIDIA)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%206.png)

> **[图片提取文字 (image.png)]:**
> ## ISA improvements
> 
> Fermi debuts the Parallel Thread eXecution (PTX) 2.0 instruction-set architecture (ISA). PTX 2.0 defines an instruction set and a new virtual machine architecture that amounts to an idealized processor designed for parallel thread operation.
> 
> Because this virtual machine model doesn't literally model the Fermi hardware, it can be portable from one generation to the next. NVIDIA intends PTX 2.0 to span multiple generations of GPU hardware and multiple GPU sizes within each generation, just as PTX 1.0 did.
> 
> Compilers supporting NVIDIA GPUs provide PTX-compliant binaries that act as a hardware-neutral distribution format for GPU computing applications and middleware. When applications are installed on a target machine, the GPU driver translates the PTX binaries into the low-level machine instructions that are directly executed by the hardware. (PTX 1.0 binaries can also be translated by Fermi GPU drivers into native instructions.)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## **Third Generation Streaming Multiprocessor**
> 
> The third generation SM introduces several architectural innovations that make it not only the most powerful SM yet built, but also the most programmable and efficient.
> 
> ## **512 High Performance CUDA cores**
> 
> Each SM features 32 CUDA processors—a fourfold increase over prior SM designs. Each CUDA processor has a fully pipelined integer arithmetic logic unit (ALU) and floating
> 
> ![](_page_0_Picture_4.jpeg)
> 
> point unit (FPU). Prior GPUs used IEEE 754-1985 floating point arithmetic. The Fermi architecture implements the new IEEE 754-2008 floating-point standard, providing the fused multiply-add (FMA) instruction for both single and double precision arithmetic. FMA improves over a multiply-add (MAD) instruction by doing the multiplication and addition with a single final rounding step, with no loss of precision in the addition. FMA is more accurate than performing the operations separately. GT200 implemented double precision FMA.
> 
> ![](_page_0_Picture_6.jpeg)
> 
> Fermi Streaming Multiprocessor (SM)
> 
> In GT200, the integer ALU was limited to 24-bit precision for multiply operations; as a result, multi-instruction emulation sequences were required for integer arithmetic. In Fermi, the newly designed integer ALU supports full 32-bit precision for all instructions, consistent with standard programming language requirements. The integer ALU is also optimized to efficiently support 64-bit and extended precision operations. Various instructions are supported, including Boolean, shift, move, compare, convert, bit-field extract, bit-reverse insert, and population count.
> 
> ## 16 Load/Store Units
> 
> Each SM has 16 load/store units, allowing source and destination addresses to be calculated for sixteen threads per clock. Supporting units load and store the data at each address to cache or DRAM.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%208.png)

> **[图片提取文字 (image.png)]:**
> ## **Four Special Function Units**
> 
> Special Function Units (SFUs) execute transcendental instructions such as sin, cosine, reciprocal, and square root. Each SFU executes one instruction per thread, per clock; a warp executes over eight clocks. The SFU pipeline is decoupled from the dispatch unit, allowing the dispatch unit to issue to other execution units while the SFU is occupied.
> 
> ## **Designed for Double Precision**
> 
> Double precision arithmetic is at the heart of HPC applications such as linear algebra, numerical simulation, and quantum chemistry. The Fermi architecture has been specifically designed to offer unprecedented performance in double precision; up to 16 double precision fused multiply-add operations can be performed per SM, per clock, a dramatic improvement over the GT200 architecture.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%209.png)

PTX2用**统一地址空间address**来统一管理不同存储层次Memory。

> **[图片提取文字 (image.png)]:**
> ## **Summary Table**
> 
> | GPU                              | G80         | GT200              | Fermi                 |
> |----------------------------------|-------------|--------------------|-----------------------|
> | Transistors                      | 681 million | 1.4 billion        | 3.0 billion           |
> | CUDA Cores                       | 128         | 240                | 512                   |
> | <b>Double Precision Floating</b> | None        | 30 FMA ops / clock | 256 FMA ops /clock    |
> | Point Capability                 |             |                    |                       |
> | Single Precision Floating        | 128 MAD     | 240 MAD ops /      | 512 FMA ops /clock    |
> | Point Capability                 | ops/clock   | clock              |                       |
> | Special Function Units           | 2           | 2                  | 4                     |
> | (SFUs) / SM                      |             |                    |                       |
> | Warp schedulers (per SM)         | 1           | 1                  | 2                     |
> | Shared Memory (per SM)           | 16 KB       | 16 KB              | Configurable 48 KB or |
> |                                  |             |                    | 16 KB                 |
> | L1 Cache (per SM)                | None        | None               | Configurable 16 KB or |
> |                                  |             |                    | 48 KB                 |
> | L2 Cache                         | None        | None               | 768 KB                |
> | <b>ECC Memory Support</b>        | No          | No                 | Yes                   |
> | Concurrent Kernels               | No          | No                 | Up to 16              |
> | Load/Store Address Width         | 32-bit      | 32-bit             | 64-bit                |
> 
> ## Second Generation Parallel Thread Execution ISA
> 
> Fermi is the first architecture to support the new Parallel Thread eXecution (PTX) 2.0 instruction set. PTX is a low level virtual machine and ISA designed to support the operations of a parallel thread processor. At program install time, PTX instructions are translated to machine instructions by the GPU driver.
> 
> ## The primary goals of PTX are:
> 
> - Provide a stable ISA that spans multiple GPU generations
> - Achieve full GPU performance in compiled applications
> - □ Provide a machine-independent ISA for C, C++, Fortran, and other compiler targets.
> - Provide a code distribution ISA for application and middleware developers
> - □ Provide a common ISA for optimizing code generators and translators, which map PTX to specific target machines.
> - Facilitate hand-coding of libraries and performance kernels
> - Provide a scalable programming model that spans GPU sizes from a few cores to many parallel cores
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## **Unified Address Space enables Full C++ Support**
> 
> Fermi and the PTX 2.0 ISA implement a unified address space that unifies the three separate address spaces (thread private local, block shared, and global) for load and store operations. In PTX 1.0, load/store instructions were specific to one of the three address spaces; programs could load or store values in a specific target address space known at compile time. It was difficult to fully implement C and C++ pointers since a pointer's target address space may not be known at compile time, and may only be determined dynamically at run time.
> 
> With PTX 2.0, a unified address space unifies all three address spaces into a single, continuous address space. A single set of unified load/store instructions operate on this address space, augmenting the three separate sets of load/store instructions for local, shared, and global memory. The 40-bit unified address space supports a Terabyte of addressable memory, and the load/store ISA supports 64-bit addressing for future growth.
> 
> ## **Separate Address Spaces**
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2011.png)

同类型指针，通过指针地址所在范围自动识别设备和层次，而不需要编程时额外指定。

[https://cloud.tencent.com/developer/inventory/8372/article/1151503](https://cloud.tencent.com/developer/inventory/8372/article/1151503)

> **[图片提取文字 (Screenshot from 2026-03-26 20-57-09.png)]:**
> ## 3.2.7. Unified Virtual Address Space【统一虚拟地址空间】
> 
> When the application is run as a 64-bit process, a single address space is used for the host and all the devices of compute capability 2.0 and higher. All host memory allocations made via CUDA API® calls and all device memory allocations on supported devices are within this virtual address range. As a consequence:
> 
> - The location of any memory on the host allocated through CUDA, or on any of the devices which use the unified address space, can be determined from the value of the pointer usingcudaPointerGetAttributes().
> - · When copying to or from the memory of any device which uses the unified address space, the cudaMemcpyKind parameter of cudaMemcpy\*() can be set to cudaMemcpyDefault to determine locations from the pointers. This also works for host pointers not allocated through CUDA, as long as the current device uses unified addressing.
> - · Allocations via cudaHostAlloc() are automatically portable (see <u>Portable Memory</u>) across all the devices for which the unified address space is used, and pointers returned bycudaHostAlloc() can be used directly from within kernels running on these devices (i.e., there is no need to obtain a device pointer via cudaHostGetDevicePointer() as described in <u>Mapped Memory</u>.
> 
> Applications may query if the unified address space is used for a particular device by checking that the unifiedAddressing device property (see <u>Device Enumeration</u>) is equal to 1.
![Screenshot from 2026-03-26 20-57-09.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/Screenshot_from_2026-03-26_20-57-09.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-59-22.png)]:**
> Unified Virtual Address Space 统一的虚拟地址空间,这个空间包含:进程的传统Host虚拟地址空间,所有卡的虚拟 地址空间。也就是CPU + GPU (多个)。用人话说就是,将你分配的普通malloc(), 每个卡上的cudaMalloc()出来 的,这些得到的分配出来的缓冲区地址,都在同一个64-bit的进程虚拟地址空间内。可以直接使用一个普通的指针Type \*p指向,而不是每个分配的指针只在每个设备上才有意义。以前的我们会往往遇到这种情况: 我在CPU上分配到地址int \*p是0x12345678, 然后在GPU上分配到的地址也是0x12345678, 用户必须明确的知道这个地址是在哪里有效的, 才能 用它。(因为以前不是统一编址的,大家各自为战)所以你会看到以前cudaMemcpy之类的函数,指定了目标地址,源 地址,传输大小等信息后, 却需要额外的添加一个类似cudaMemcpyHostToDevice这种参数告诉CUDA Runtime,源 地址是从Host来的,目标地址是在设备(卡)上的。现在统一编址后,不需要用户维护这个信息了,直接CUDA就能知 道,哦,这个地址是卡1上的,这个地址是卡2上的,这个地址是卡3的, 这个地址是内存,这个地址是自动管理的 (unified memory) ...类似这种,方便了很多。也为以后实现很多功能打下了基础。这个是个老特性,从Fermi开始 的,但是有了这个基础,我们现在用Pascal,跨卡P2P Access(你还记得这个是什么吗,昨天才讲过?), 直接卡1 上的kernel,能够使用一个指针p,而p指向的内容却在卡2上,没有这个基础,P2P Access无法实现。类似的,这还为 其他特性,例如现在的unified memory,假设你有一个链表,非常巨大, CPU想负责一部分适合它处理的里面的节点 数据,GPU想处理一部分它想处理的,以前的写法只能是每个节点标记一下,例如: 本节点标记为是内存上,必须用 CPU处理,本节点链接到的下一个节点是在GPU上,这个下一个节点的指向的指针必须GPU有效,CPU不能处理,云 云的。 现在统一编址后,可以直接获取某个节点在哪里,甚至通过UVA + Unified memory,程序员偶尔不小心用CPU 处理了某个应当GPU处理的节点(或者反过来),也不要紧, Runtime/Driver自动给你迁移了位置,处理起来很方便。 再比如,以前很多显卡没有显存,(很多笔记本的集成的N卡,虽然支持CUDA。但没有显存), 用户以前都用zerocopy, 但是zero-copy以前有个问题, 同样的一段缓冲区, 例如100MB, 它在CPU上的地址, 和在GPU上的地址是不 同的,用户必须同时保存两份指针信息,一个指针是host上有效的,一个指针是GPU上有效的。用错了,程序就挂 了。当Fermi开始,引入了UVA后, 这两个地址变成了同样的值,用户知道int \*p可以在host上用,也可以直接在GPU 上用,不仅仅简单了很多,还减少了很大的出错可能。很是方便的。 这个是一个巨大的基础改进。当年Fermi引入的和 UVA同样的改进还有一部分,叫Generic Addressing。UVA是全局的(卡,CPU,多卡),Generic Addressing是卡 内部的,pre-fermi的时候,卡内部的地址也不是统一的,local memory, shared memory, global memory是分裂的, 一个指针必须需要在编译时刻知道指向哪里,否则不能使用。 就像DOS时代的segment一样难用。fermi起,将卡内, 卡间(系统内)都统一了。一个指针可以打天下了。相当方便和给力。 这其实主要是为了易用性,对性能其实无提升
> 
> 的。
![Screenshot from 2026-03-26 20-59-22.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/Screenshot_from_2026-03-26_20-59-22.png)

### 应用抢占/独占、MPS

不同应用对shared memory和cache的需求偏好不同。

WDU派发CTA到GPC/SM，SM内派发warp到SIMD Cores。

**CUDA Ctx**是**应用在CPU侧进程的运行时，相同CUDA Ctx的kernel表示相同应用的kernel call**。相同App内不同**kernel** grid并发，因为TMU和WDU相对于SM执行是异步发送。不同App内不同**kernel** grid可以并发，但是存在性能干扰和可能的控制干扰（一挂全挂）。

不同App分时间片独占GPU时，需要切换**App Ctx（从前端开始的Ctx）**，因为需要抢占TMU中的所有的task group（channel）。

> **[图片提取文字 (image.png)]:**
> ## **Memory Subsystem Innovations**
> 
> ## NVIDIA Parallel DataCache™ with Configurable L1 and Unified L2 Cache
> 
> Working with hundreds of GPU computing applications from various industries, we learned that while Shared memory benefits many problems, it is not appropriate for all problems. Some algorithms map naturally to Shared memory, others require a cache, while others require a combination of both. The optimal memory hierarchy should offer the benefits of both Shared memory and cache, and allow the programmer a choice over its partitioning. The Fermi memory hierarchy adapts to both types of program behavior.
> 
> Adding a true cache hierarchy for load / store operations presented significant challenges. Traditional GPU architectures support a read-only "load" path for texture operations and a write-only "export" path for pixel data output. However, this approach is poorly suited to executing general purpose C or C++ thread programs that expect reads and writes to be ordered. As one example: spilling a register operand to memory and then reading it back creates a read after write hazard; if the
> 
> Fermi Memory Hierarchy Thread **Shared Memory** L1 Cache L2 Cache DRAM
> 
> read and write paths are separate, it may be necessary to explicitly flush the entire write / "export" path before it is safe to issue the read, and any caches on the read path would not be coherent with respect to the write data.
> 
> The Fermi architecture addresses this challenge by implementing a single unified memory request path for loads and stores, with an L1 cache per SM multiprocessor and unified L2 cache that services all operations (load, store and texture). The per-SM L1 cache is configurable to support both shared memory and caching of local and global memory operations. The 64 KB memory can be configured as either 48 KB of Shared memory with 16 KB of L1 cache, or 16 KB of Shared memory with 48 KB of L1 cache. When configured with 48 KB of shared memory, programs that make extensive use of shared memory (such as electrodynamic simulations) can perform up to three times faster. For programs whose memory accesses are not known beforehand, the 48 KB L1 cache configuration offers greatly improved performance over direct access to DRAM.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ## GigaThread<sup>™</sup> Thread Scheduler
> 
> One of the most important technologies of the Fermi architecture is its two-level, distributed thread scheduler. At the chip level, a global work distribution engine schedules thread blocks to various SMs, while at the SM level, each warp scheduler distributes warps of 32 threads to its execution units. The first generation GigaThread engine introduced in G80 managed up to 12,288 threads in realtime. The Fermi architecture improves on this foundation by providing not only greater thread throughput, but dramatically faster context switching, concurrent kernel execution, and improved thread block scheduling.
> 
> ## 10x Faster Application Context Switching
> 
> Like CPUs, GPUs support multitasking through the use of context switching, where each program receives a time slice of the processor's resources. The Fermi pipeline is optimized to reduce the cost of an application context switch to below 25 microseconds, a significant improvement over last generation GPUs. Besides improved performance, this allows developers to create applications that take greater advantage of frequent kernel-to-kernel communication, such as fine-grained interoperation between graphics and PhysX applications.
> 
> ## **Concurrent Kernel Execution**
> 
> Fermi supports concurrent kernel execution, where different kernels of the same application context can execute on the GPU at the same time. Concurrent kernel execution allows programs that execute a number of small kernels to utilize the whole GPU. For example, a PhysX program may invoke a fluids solver and a rigid body solver which, if executed sequentially, would use only half of the available thread processors. On the Fermi architecture, different kernels of the same CUDA context can execute concurrently, allowing maximum utilization of GPU resources. Kernels from different application contexts can still run sequentially with great efficiency thanks to the improved context switching performance.
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
> 
> **Serial Kernel Execution**
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2013.png)

## Kepler（动态并行、Hyper-Q）

Kepler的1个SMX等价于Fermi中2个SM，面积效率和能耗更好。

Kepler相比Maxwell，双精度性能更高。

> **[图片提取文字 (image.png)]:**
> ## Kepler Architecture In-Depth (GeForce GTX 680)
> 
> Like Fermi, Kepler GPUs are composed of different configurations of Graphics Processing Clusters (GPCs), Streaming Multiprocessors (SMs), and memory controllers. The GeForce GTX 680 GPU consists of four GPCs, eight next-generation Streaming Multiprocessors (SMX), and four memory controllers.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 1: GeForce GTX 680 Block Diagram
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2014.png)

> **[图片提取文字 (image.png)]:**
> | GPU                        | GT200 (Tesla)       | GF110 (Fermi)       | GK104 (Kepler)       |
> |----------------------------|---------------------|---------------------|----------------------|
> | Transistors                | 1.4 billion         | 3.0 billion         | 3.54 billion         |
> | CUDA Cores                 | 240                 | 512                 | 1536                 |
> | <b>Graphics Core Clock</b> | 648MHz              | 772MHz              | 1006MHz              |
> | Shader Core Clock          | 1476MHz             | 1544MHz             | n/a                  |
> | GFLOPs                     | 1063                | 1581                | 3090                 |
> | Texture Units              | 80                  | 64                  | 128                  |
> | Texel fill-rate            | 51.8 Gigatexels/sec | 49.4 Gigatexels/sec | 128.8 Gigatexels/sec |
> | Memory Clock               | 2484 MHz            | 4008 MHz            | 6008MHz              |
> | Memory Bandwidth           | 159 GB/sec          | 192.4 GB/sec        | 192.26 GB/sec        |
> | Max # of Active Displays   | 2                   | 2                   | 4                    |
> | TDP                        | 183W                | 244W                | 195W                 |
> 
> The overall configuration of GTX 680 was chosen to provide a large increase in shader and texture horsepower vs. the GTX 580, while maintaining per clock operand throughputs for most other metrics (which also benefit from the increased core clock frequency).
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2015.png)

> **[图片提取文字 (image.png)]:**
> GK110 and GK210 are both designed to provide fast double precision computing performance to accelerate professional HPC compute workloads; this is a key difference from the NVIDIA Maxwell GPU architecture, which is designed primarily for fast graphics performance and single precision consumer compute tasks. While the Maxwell architecture performs double precision calculations at rate of 1/32 that of single precision calculations, the GK110 and GK210 Kepler-based GPUs are capable of performing double precision calculations at a rate of up to 1/3 of single precision compute performance.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2016.png)

> **[图片提取文字 (image.png)]:**
> To understand SMX performance, it helps to start by comparing the chip level unit counts for GeForce GTX 580 (containing 16 SMs) to GeForce GTX 680 (containing 8 SMXs):
> 
> | GPU                             | GF110<br>(Fermi) | GK104<br>(Kepler) | Ratio | Ratio<br>(w/ clk freq) |
> |---------------------------------|------------------|-------------------|-------|------------------------|
> | Total unit counts :             |                  |                   |       |                        |
> | CUDA Cores                      | 512              | 1536              | 3.0x  |                        |
> | SFU                             | 64               | 256               | 4.0x  |                        |
> | LD/ST                           | 256              | 256               | 1.0x  |                        |
> | Tex                             | 64               | 128               | 2.0x  |                        |
> | Polymorph                       | 16               | 8                 | 0.5x  |                        |
> | Warp schedulers                 | 32               | 32                | 1.0x  |                        |
> | Throughput per graphics clock : |                  |                   |       |                        |
> | FMA32                           | 1024             | 1536              | 1.5x  | 2.0x                   |
> | SFU                             | 128              | 256               | 2.0x  | 2.6x                   |
> | LD/ST (64b operations)          | 256              | 256               | 1.0x  | 1.3x                   |
> | Tex                             | 64               | 128               | 2.0x  | 2.6x                   |
> | Polygon/clk                     | 4                | 4                 | 1.0x  | 1.3x                   |
> | Inst/clk                        | 32*32            | 64*32             | 2.0x  | 2.6x                   |
> 
> At the chip level, the per-clock throughput for key graphics operations (FMA32, SFU operations, and texture operations) have all been increased substantially, while other operations retain per-clock throughput equal to GeForce GTX 580. GeForce GTX 680's substantially higher clock frequency provides a further throughput boost for all operations.
> 
> For GeForce GTX 680, we chose—for area efficiency reasons—to divide the aggregate horsepower into 8 total SMX units (rather than dividing the aggregate horsepower into 16 SM units as we did in GeForce GTX 580). Considering this and the other factors above, the per-SMX unit count and throughput can be compared as follows:
> 
> | GPU                  | GF110<br>(Fermi) | GK104<br>(Kepler) | Ratio | Ratio<br>(w/ clk freq) |
> |----------------------|------------------|-------------------|-------|------------------------|
> | Per SM unit counts : |                  |                   |       |                        |
> | CUDA Cores           | 32               | 192               | 6.0x  |                        |
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2017.png)

> **[图片提取文字 (image.png)]:**
> | SFU                            | 4    | 32   | 8.0x |      |
> |--------------------------------|------|------|------|------|
> | LD/ST                          | 16   | 32   | 2.0x |      |
> | Tex                            | 4    | 16   | 4.0x |      |
> | Polymorph                      | 1    | 1    | 1.0x |      |
> | Warp schedulers                | 2    | 4    | 2.0x |      |
> | Throughput per graphics clock: |      |      |      |      |
> | FMA32                          | 64   | 192  | 3.0x | 3.9x |
> | SFU                            | 8    | 32   | 4.0x | 5.2x |
> | LD/ST (64b operations)         | 16   | 32   | 2.0x | 2.6x |
> | Tex                            | 4    | 16   | 4.0x | 5.2x |
> | Polygon/clk                    | 0.25 | 0.5  | 2.0x | 2.6x |
> | Inst/clk                       | 32*2 | 32*8 | 4.0x | 5.2x |
> 
> See below for the block diagram illustration of the functional units in SMX.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2018.png)

**算术指令执行延迟确定**，编译来确定SMX中指令发射顺序（打包在指令中）。使用简单硬件模块提取发射延迟，在warp之间**调度发射**时剔除（mask out）特定warps（需要等待预定延迟后发射）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: GeForce GTX 680 SMX
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ## **Next Generation SM (SMX) Architectural Details**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> To feed the execution resources of SMX, each unit contains four warp schedulers, and each warp scheduler is capable of dispatching two instructions per warp every clock.
> 
> More importantly, the scheduling functions have been redesigned with a focus on power efficiency. For example: Both Kepler and Fermi schedulers contain similar hardware units to handle scheduling functions, including, (a) register scoreboarding for long latency operations (texture and load), (b) interwarp scheduling decisions (e.g., pick the best warp to go next among eligible candidates), and (c) thread block level scheduling (e.g., the GigaThread engine); however, Fermi's scheduler *also* contains a complex hardware stage to prevent data hazards in the math datapath itself. A multi-port register scoreboard keeps track of any registers that are not yet ready with valid data, and a dependency checker block analyzes register usage across a multitude of fully decoded warp instructions against the scoreboard, to determine which are eligible to issue.
> 
> For Kepler, we realized that since this information is deterministic (the math pipeline latencies are not variable), it is possible for the compiler to determine up front when instructions will be ready to issue, and provide this information in the instruction itself. This allowed us to replace several complex and power-expensive blocks with a simple hardware block that extracts the pre-determined latency information and uses it to mask out warps from eligibility at the inter-warp scheduler stage.
> 
> We also developed a new design for the processor execution core, again with a focus on best performance per watt. Each processing unit was scrubbed to maximize clock gating efficiency and minimize wiring and retiming overheads.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2020.png)

取消shader clock（部分单元用更高时钟），执行单元使用面积换吞吐，提高TOPS/W。PolyMorph Engine（图形）优化。

> **[图片提取文字 (image.png)]:**
> The biggest visible change for the processor core is the elimination of shader clock. Shader clock was introduced in the Tesla architecture as an area optimization. Running execution units at a higher clock rate allows a chip to achieve a given target throughput with fewer copies of the execution unit.
> 
> However, the higher clock rate also implies more power, especially clock power. Doubling the clock frequency implies twice as many pipeline stages, each running at twice the clock rate—so 4x power per unit. Even with half as many units required for a given throughput target, a 2x power penalty for the retiming stage units remains.
> 
> For Kepler, our priority was perf/W. While we made many optimizations that benefitted both area and power, this was an example of a case where we chose to optimize for power even at the expense of added area.
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ## **PolyMorph Engine 2.0**
> 
> The final SMX unit to receive significant modifications in Kepler is the PolyMorph Engine. The PolyMorph Engine is the key unit responsible for Fermi's extraordinary performance on DX11 tessellation workloads. It is designed to ensure that even as tessellation is increased to very high expansion factors (i.e., ratio of output polygons emitted per input patch), the impact on rendering performance is minimized.
> 
> GeForce GTX 680 contains 8 PolyMorph Engines, compared to 16 for GeForce GTX 580; however, the Kepler PolyMorph engine was redesigned to deliver roughly double the per-clock performance of the Fermi version. GeForce GTX 680's 30% higher shipping clock speed ensures a significant overall improvement in tessellation workloads.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2022.png)

### Hyper-Q，动态并行

线程Ctx寄存器容量扩大（最大并发线程数和最大RF/线程Ctx匹配）。

Shuffle指令允许warp内线程通过寄存器交换数据。

> **[图片提取文字 (image.png)]:**
> Kepler GK110 supports the new CUDA Compute Capability 3.5. (For a brief overview of CUDA see
> 
> **Appendix A - Quick Refresher on CUDA**). The following table compares parameters of different Compute Capabilities for Fermi and Kepler GPU architectures:
> 
> ## **Compute Capability of Fermi and Kepler GPUs**
> 
> |                                    | FERMI<br>GF100 | FERMI<br>GF104 | KEPLER<br>GK104 | KEPLER<br>GK110 | KEPLER<br>GK210 |
> |------------------------------------|----------------|----------------|-----------------|-----------------|-----------------|
> | Compute Capability                 | 2.0            | 2.1            | 3.0             | 3.5             | 3.7             |
> | Threads / Warp                     | 32             |                |                 |                 |                 |
> | Max Threads / Thread Block         | 1024           |                |                 |                 |                 |
> | Max Warps / Multiprocessor         | 48             |                | 64              |                 |                 |
> | Max Threads / Multiprocessor       | 1536           |                | 2048            |                 |                 |
> | Max Thread Blocks / Multiprocessor | 8              |                | 16              |                 |                 |
> | 32-bit Registers / Multiprocessor  | 32768          |                | 65536           |                 | 131072          |
> | Max Registers / Thread Block       | 32768          |                | 65536           |                 | 65536           |
> | Max Registers / Thread             | 63             |                |                 | 255             |                 |
> | Max Shared Memory / Multiprocessor | 48K            |                |                 | 112K            |                 |
> | Max Shared Memory / Thread Block   | 48K            |                |                 |                 |                 |
> | Max X Grid Dimension               | 2^16-1 2^32-1  |                |                 | 2^32-1          |                 |
> | Hyper-Q                            | No             |                | Yes             |                 |                 |
> | Dynamic Parallelism                | No             |                |                 | Yes             |                 |
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ## **New ISA Encoding: 255 Registers per Thread**
> 
> The number of registers that can be accessed by a thread has been quadrupled in GK110, allowing each thread access to up to 255 registers. Codes that exhibit high register pressure or spilling behavior in Fermi may see substantial speedups as a result of the increased available per-thread register count. A compelling example can be seen in the QUDA library for performing lattice QCD (quantum chromodynamics) calculations using CUDA. QUDA fp64-based algorithms see performance increases up to 5.3x due to the ability to use many more registers per thread and experiencing fewer spills to local memory.
> 
> GK210 further improves this, doubling the overall register file capacity per SMX as compared to GK110. In doing so, it allows applications to more readily make use of higher numbers of registers per thread without sacrificing the number of threads that can fit concurrently per SMX. For example, a CUDA kernel using 128 registers thread on GK110 is limited to 512 out of a possible 2048 concurrent threads per SMX, limiting the available parallelism. GK210 doubles the concurrency automatically in this case, which can help to cover arithmetic and memory latencies, improving overall efficiency.
> 
> ## **Shuffle Instruction**
> 
> within a warp to share data. Previously, sharing data between threads within a warp required separate store and load operations to pass the data through shared memory. With the Shuffle instruction, threads within a warp can read values from other threads in the warp in just about any imaginable permutation. Shuffle supports arbitrary indexed references – i.e. any thread reads from any other thread. Useful shuffle subsets including next-thread (offset up or down by a fixed amount) and XOR "butterfly" style permutations among the threads in a warp, are also available as CUDA intrinsics.
> 
> Shuffle offers a performance advantage over shared memory, in that a store-and-load operation is carried out in a single step. Shuffle also can reduce the amount of shared memory needed per thread block, since data exchanged at the warp level never needs to be placed in shared memory. In the case of FFT, which requires data sharing within a warp, a 6% performance gain can be seen just by using Shuffle.
> 
> ![](_page_0_Figure_6.jpeg)
> 
> This example shows some of the variations possible using the new Shuffle instruction in Kepler.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2024.png)

**动态并行**允许**device线程动态生成grid、Stream和event**，提高数值仿真效率。

channel是TMU中的task group（链表实现），pushbuffer存储cudaStream（TMD指针），cudaStream通过channel传递，**Fermi允许连接多个pushbuffer来读取命令流，但只提供1个channel**，不同Stream竞争channel引入False依赖。

Hyper-Q：增加更多硬件队列（物理连接数）。

> **[图片提取文字 (image.png)]:**
> In Kepler GK110/210 any kernel can launch another kernel, and can create the necessary streams, events and manage the dependencies needed to process additional work without the need for host CPU interaction. This architectural innovation makes it easier for developers to create and optimize recursive and data-dependent execution patterns, and allows more of a program to be run directly on GPU. The system CPU can then be freed up for additional tasks, or the system could be configured with a less powerful CPU to carry out the same workload.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Dynamic Parallelism allows more parallel code in an application to be launched directly by the GPU onto itself (right side of image) rather than requiring CPU intervention (left side of image).
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ## **Hyper-Q**
> 
> One of the challenges in the past has been keeping the GPU supplied with an optimally scheduled load of work from multiple streams. The Fermi architecture supported 16-way concurrency of kernel launches from separate streams, but ultimately the streams were all multiplexed into the same hardware work queue. This allowed for false intra-stream dependencies, requiring dependent kernels within one stream to complete before additional kernels in a separate stream could be executed. While this could be alleviated to some extent through the use of a breadth-first launch order, as program complexity increases, this can become more and more difficult to manage efficiently.
> 
> Kepler GK110/210 improve on this functionality with their Hyper-Q feature. Hyper-Q increases the total number of connections (work queues) between the host and the CUDA Work Distributor (CWD) logic in the GPU by allowing 32 simultaneous, hardware-managed connections (compared to the single connection available with Fermi). Hyper-Q is a flexible solution that allows connections from multiple CUDA streams, from multiple Message Passing Interface (MPI) processes, or even from multiple threads within a process. Applications that previously encountered false serialization across tasks, thereby limiting GPU utilization, can see up to a 32x performance increase without changing any existing code.
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Hyper-Q permits more simultaneous connections between CPU and GPU.
> 
> Each CUDA stream is managed within its own hardware work queue, inter-stream dependencies are optimized, and operations in one stream will no longer block other streams, enabling streams to execute concurrently without needing to specifically tailor the launch order to eliminate possible false dependencies.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2026.png)

channel Ctx的切换是什么？抢占TMU的work group（channel）。设置GMU管理host传递和SM生成的grid（GTMQ queue、QTMD）或cudaStream（TMDQs）。

> **[图片提取文字 (image.png)]:**
> Hyper-Q offers significant benefits for use in MPI-based parallel computer systems. Legacy MPI-based algorithms were often created to run on multi-core CPU systems, with the amount of work assigned to each MPI process scaled accordingly. This can lead to a single MPI process having insufficient work to fully occupy the GPU. While it has always been possible for multiple MPI processes to share a GPU, these processes could become bottlenecked by false dependencies. Hyper-Q removes those false dependencies, dramatically increasing the efficiency of GPU sharing across MPI processes.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Each stream receives its own work queue
> 
> Hyper-Q working with CUDA Streams: In the Fermi model shown on the left, only (C,P) & (R,X) can run concurrently due to intra-stream dependencies caused by the single hardware work queue. The Kepler Hyper-Q model allows all streams to run concurrently using separate work queues.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ## **Grid Management Unit - Efficiently Keeping the GPU Utilized**
> 
> New features introduced with Kepler GK110, such as the ability for CUDA kernels to launch work directly on the GPU with Dynamic Parallelism, required that the CPU-to-GPU workflow in Kepler offer increased functionality over the Fermi design. On Fermi, a grid of thread blocks would be launched by the CPU and would always run to completion, creating a simple unidirectional flow of work from the host to the SMs via the CUDA Work Distributor (CWD) unit. Kepler GK110/210 improve the CPU-to-GPU workflow by allowing the GPU to efficiently manage both CPU- and CUDA-created workloads.
> 
> We discussed the ability of Kepler GK110 GPU to allow kernels to launch work directly on the GPU, and it's important to understand the changes made in the Kepler GK110 architecture to facilitate these new functions. In Kepler GK110/210, a grid can be launched from the CPU just as was the case with Fermi, however new grids can also be created programmatically by CUDA within the Kepler SMX unit. To manage both CUDA-created and host-originated grids, a new Grid Management Unit (GMU) was introduced in Kepler GK110. This control unit manages and prioritizes grids that are passed into the CWD to be sent to the SMX units for execution.
> 
> The CWD in Kepler holds grids that are ready to dispatch, and it is able to dispatch 32 active grids, which is double the capacity of the Fermi CWD. The Kepler CWD communicates with the GMU via a bidirectional link that allows the GMU to pause the dispatch of new grids and to hold pending and suspended grids until needed. The GMU also has a direct connection to the Kepler SMX units to permit grids that launch additional work on the GPU via Dynamic Parallelism to send the new work back to GMU to be prioritized and dispatched. If the kernel that dispatched the additional workload pauses, the GMU will hold it inactive until the dependent work has completed.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> The redesigned Kepler HOST to GPU workflow shows the new Grid Management Unit, which allows it to manage the actively dispatching grids, pause dispatch, and hold pending and suspended grids.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2028.png)

## Maxwell（Tiled Texture>动态分辨率）

> **[图片提取文字 (image.png)]:**
> ## **GM204** Hardware Architecture In-Depth
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Figure 2: GM204 Full-chip block diagram
> 
> Like Fermi and Kepler, GM204 is composed of an array of Graphics Processing Clusters (GPCs), Streaming Multiprocessors (SMs), and memory controllers. GM204 consists of four GPCs, 16 Maxwell SMs (SMM), and four memory controllers. GeForce GTX 980 uses the full complement of these architectural components (if you are not well versed in these structures, we suggest you first read the <u>Kepler</u> and <u>Fermi</u> whitepapers).
> 
> Another version of the chip, with 13 SMs, will ship concurrently and be called GeForce GTX 970. In the future we plan to offer additional products based on GM204 that will ship with different combinations of GPCs, SMs, and memory controllers to address various segments of the graphics market.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## **Maxwell Streaming Multiprocessor**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> The SM is the heart of our GPUs. Almost every operation flows through the SM at some point in the rendering pipeline.
> 
> Maxwell GPUs feature a new SM that's been designed to provide dramatically improved performance per watt than prior GeForce GPUs.
> 
> Compared to GPUs based on our Kepler architecture, Maxwell's new SMM design has been reconfigured to improve efficiency. Each SMM contains four warp schedulers, and each warp scheduler is capable of dispatching two instructions per warp every clock. Compared to Kepler's scheduling logic, we've integrated a number of improvements in the scheduler to further reduce redundant recomputation of scheduling decisions, improving energy efficiency. We've also integrated a completely new datapath organization. Whereas Kepler's SM shipped with 192 CUDA Cores—a non-power-of-two organization—the Maxwell SMM is partitioned into four distinct 32-CUDA core processing blocks (128 CUDA cores total per SM), each with its own dedicated resources for scheduling and instruction buffering. This new configuration in Maxwell aligns with warp size, making it easier to utilize efficiently and saving area
> 
> Figure 3: GM204 SMM Diagram (GM204 also features 4 DP units per SMM, which are not depicted on this diagram)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2030.png)

Tiled Texture支持不同区域的动态分辨率。

> **[图片提取文字 (image.png)]:**
> ## **Tiled Resources**
> 
> DirectX 11.2 introduced a feature called <u>Tiled Resources</u> that could be accelerated with an NVIDIA Kepler and Maxwell hardware feature called **Sparse Texture**. With Tiled Resources, only the portions of the textures required for rendering are stored in the GPU's memory. Tiled Resources works by breaking textures down into tiles (pages), and the application determines which tiles might be needed and loads them into video memory. It is also possible to use the same texture tile in multiple textures without any additional texture memory cost; this is referred to as aliasing. In the implementation of voxel grids, aliasing can be used to avoid redundant storage of voxel data, saving significant amounts of memory. You can read more about Tiled Resources at this link.
> 
> One interesting application of Tiled Resources is multi resolution shadow maps. In the following *Figure 13*, the image on the left shows the result of determining shadow information from a fixed resolution shadow map. In the foreground, the shadow map resolution is not adequate, and blocky artifacts are clearly visible. One solution would be to use a much higher resolution shadow map for the whole scene, but this would be expensive in memory footprint and rendering time. Alternatively, with Tiled Resources it is possible to render multiple copies of the shadow map at different resolutions, each populated only where that level of resolution detail is needed based on the scene. In the image on the right, each resolution of shadow map is illustrated with a different color. The highest resolution shadow map (in red) is only used in the foreground when that high resolution is required.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ## FIXED RESOLUTION SHADOW MAP MULTI RESOLUTION SHADOW MAP
> 
> Figure 13: Fixed resolution vs multi resolution shadow map quality
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2032.png)

## Pascal（NVLink、HBM2、统一内存、计算抢占）

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2033.png)

> **[图片提取文字 (image.png)]:**
> Table 1. Tesla P100 Compared to Prior Generation Tesla products
> 
> | Tesla Products                                          | Tesla K40           | Tesla M40           | Tesla P100          |
> |---------------------------------------------------------|---------------------|---------------------|---------------------|
> | GPU                                                     | GK110 (Kepler)      | GM200 (Maxwell)     | GP100 (Pascal)      |
> | SMs                                                     | 15                  | 24                  | 56                  |
> | TPCs                                                    | 15                  | 24                  | 28                  |
> | FP32 CUDA Cores / SM                                    | 192                 | 128                 | 64                  |
> | FP32 CUDA Cores / GPU                                   | 2880                | 3072                | 3584                |
> | FP64 CUDA Cores / SM                                    | 64                  | 4                   | 32                  |
> | FP64 CUDA Cores / GPU                                   | 960                 | 96                  | 1792                |
> | Base Clock                                              | 745 MHz             | 948 MHz             | 1328 MHz            |
> | GPU Boost Clock                                         | 810/875 MHz         | 1114 MHz            | 1480 MHz            |
> | Peak FP32 GFLOPs <sup>1</sup>                           | 5040                | 6840                | 10600               |
> | Peak FP64 GFLOPs <sup>1</sup>                           | 1680                | 210                 | 5300                |
> | <b>Texture Units</b>                                    | 240                 | 192                 | 224                 |
> | Memory Interface                                        | 384-bit GDDR5       | 384-bit GDDR5       | 4096-bit HBM2       |
> | Memory Size                                             | Up to 12 GB         | Up to 24 GB         | 16 GB               |
> | L2 Cache Size                                           | 1536 KB             | 3072 KB             | 4096 KB             |
> | Register File Size / SM                                 | 256 KB              | 256 KB              | 256 KB              |
> | Register File Size / GPU                                | 3840 KB             | 6144 KB             | 14336 KB            |
> | TDP                                                     | 235 Watts           | 250 Watts           | 300 Watts           |
> | Transistors                                             | 7.1 billion         | 8 billion           | 15.3 billion        |
> | GPU Die Size                                            | 551 mm <sup>2</sup> | 601 mm <sup>2</sup> | 610 mm <sup>2</sup> |
> | Manufacturing Process                                   | 28-nm               | 28-nm               | 16-nm FinFET        |
> | Manufacturing Process  1 The CELORS in this chart are k |                     |                     | 16-nm FinFET        |
> 
> <sup>&</sup>lt;sup>1</sup> The GFLOPS in this chart are based on GPU Boost Clocks.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2034.png)

更多的双精度DP Cores用于训练（Kepler是3:1，Pascal是2:1），FP16支持更好性能的DropOut（防止过拟合）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## Support for FP16 Arithmetic Speeds Up Deep Learning
> 
> Deep learning is one of the fastest growing fields of computing. It is a critical ingredient in many important applications, including real-time language translation, highly accurate image recognition, automatic image captioning, autonomous driving object recognition, optimal path calculations, collision avoidance, and others. Deep learning is a two-step process.
> 
> - First, a neural network must be trained.
> - Second, the network is deployed in the field to run inference computations, where it uses the results of previous training to classify, recognize, and generally process unknown inputs.
> 
> Compared to CPUs, GPUs can provide tremendous performance speedups for Deep Learning training and inference.
> 
> Unlike other technical computing applications that require high-precision floating-point computation, deep neural network architectures have a natural resilience to errors due to the backpropagation algorithm used in their training. In fact, to avoid overfitting a network to a training dataset, approaches such as dropout aim at ensuring a trained network generalizes well and is not overly reliant on the accuracy of (or errors in) any given unit's computation.
> 
> Storing FP16 data compared to higher precision FP32 or FP64 reduces memory usage of the neural network and thus allows training and deploying of larger networks. Using FP16 computation improves performance up to 2x compared to FP32 arithmetic, and similarly FP16 data transfers take less time than FP32 or FP64 transfers.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2036.png)

显存升级到HBM2，支持NvLink、HBM2的GPU芯片封装。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2038.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2040.png)

NVLink：CPU和GPU，GPU和GPU之间数据传输。

> **[图片提取文字 (image.png)]:**
> ## **GPU-to-GPU NVLink Connectivity**
> 
> Figure 14 shows an 8-GPU Hybrid Cube Mesh that includes two fully NVLink-connected quads of GPUs, with NVLink connections between the quads, and GPUs within each quad connected to their respective CPUs directly through PCIe. By using separate NVLink connections to span the gap between the two quads, it relieves pressure on the PCIe uplink to each CPU, and likewise avoids routing transfers through system memory and over an inter-CPU link.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 14. Eight GPU Hybrid Cube Mesh Architecture
> 
> Note that each half of the 8-GPU Hybrid Cube Mesh can operate as a shared memory multiprocessor, while the remote nodes can also share memory with DMA through peers. With all GPU-to-GPU traffic flowing over NVLINK, PCIe is now entirely available for either connection to a NIC (not shown) or for accessing system memory traffic. This configuration will be commonly recommended for general-purpose Deep Learning applications and is implemented in NVIDIA's new DGX-1 server.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ## NVLink Interface to the Tesla P100
> 
> As described in the Tesla P100 Design section, NVLink interconnections are included on the P100 accelerator. The P100 includes two 400-pin high speed connectors. One of these connectors is used for the NVLink signals on/off the module; the other is used to supply power, control signals and PCIe I/O.
> 
> The Tesla P100 accelerator can be installed into a larger GPU carrier or system board. The GPU carrier makes the appropriate connections to other P100 accelerators or PCIE controllers. Because of the smaller size of the P100 accelerator compared to traditional GPU boards, customers can easily build servers that are packed with more GPUs than ever before. With the added bandwidth provided by NVLink, GPU-to-GPU communications will not be bottlenecked by the limitations of PCIe bandwidth, enabling previously unavailable opportunities for GPU clustering.
> 
> At the level of the GPU architectural interface, the NVLink controller communicates with the GPU internals through another new block called the High-Speed Hub (HSHUB). The HSHUB has direct access to the GPU-wide crossbar and other system elements, such as the High-Speed Copy Engines (HSCE). which can be used to move data into and out of the GPU at peak NVLink rates. Figure 18 shows how NVLink relates to HSHUB and some of the higher level blocks in a GP100 GPU.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> Figure 18. NVLink relationship to other major blocks in GP100
> 
> For more details, see Appendix A: NVLink Signaling and Protocol Technology.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2042.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Figure 16. NVLink GPU-to-CPU Interconnect
> 
> Figure 17 shows a system with two NVLinks from the CPU to each GPU. The remaining two links on each GPU are used for peer-to-peer communication.
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Figure 17. Two GPUs and a CPU Connected with 80 GB/sec Bidirectional Bandwidth NVLink Interfaces
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2043.png)

### 统一内存（GPU访问CPU内存）

CPU和GPU统一内存让CPU和GPU Cache“隐式自动”从CPU内存和GPU内存中读取数据。

CUDA4是统一虚拟地址空间来管理不同内存。

CUDA6的统一内存创造CPU和GPU的内存池，但是CPU只能在GPU执行kernel之前或等待完成后访问统一内存，CPU和GPU不能同时访问内存，统一地址空间不超过VRAM大小。

> **[图片提取文字 (image.png)]:**
> ## **Unified Memory History**
> 
> The NVIDIA Fermi GPU architecture, introduced in 2009, implemented a unified GPU address space spanning the three main GPU memory spaces (thread private local memory, thread block shared memory, and global memory). This unified address space only applied to GPU memory addressing, and mainly resulted in simpler compilation by enabling a single load/store instruction and pointer address to access any of the GPU memory spaces (global, local, or shared memory), rather than different instructions and pointers for each. This also enabled full C and C++ pointer support, which was a significant advancement at the time.
> 
> In 2011, CUDA 4 introduced Unified Virtual Addressing (UVA) to provide a single virtual memory address space for both CPU and GPU memory and enable pointers to be accessed from GPU code no matter where in the system they reside, whether in GPU memory (on the same or a different GPU), CPU memory, or on-chip shared memory. UVA enables *Zero-Copy* memory, which is pinned CPU memory accessible by GPU code directly, over PCIe, without a memcpy. Zero-Copy provides some of the convenience of Unified Memory, but none of the performance, because it is always accessed by the GPU with PCIe's low bandwidth and high latency.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2044.png)

> **[图片提取文字 (image.png)]:**
> Figure 20 shows an example of how Unified Memory in CUDA 6 simplifies porting of code to the GPU by providing a single pointer to data, making explicit CPU-GPU memory copies an optimization rather than a requirement.
> 
> ## CPU Code
> 
> ```
> void sortfile(FILE *fp, int N) {
>   char *data;
>   data = (char *)malloc(N);
> 
>   fread(data, 1, N, fp);
> 
>   qsort(data, N, 1, compare);
> 
>   use_data(data);
> 
>   free(data);
> }
> ```
> 
> ## **CUDA 6 Code with Unified Memory**
> 
> ```
> void sortfile(FILE *fp, int N) {
>   char *data;
>   cudaMallocManaged(&data, N);
>   fread(data, 1, N, fp);
>   qsort<<<...>>>(data,N,1,compare);
>   cudaDeviceSynchronize();
>   use_data(data);
>   cudaFree(data);
> ```
> 
> Figure 20. CUDA 6 Unified Memory Simplifies Porting Code to the GPU (This is done by providing a new *managed* memory allocator that returns a pointer to data that can be accessed from either CPU or GPU code.)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2045.png)

> **[图片提取文字 (image.png)]:**
> CUDA 6 introduced Unified Memory, which creates a pool of managed memory that is shared between the CPU and GPU, bridging the CPU-GPU divide. Managed memory is accessible to both the CPU and GPU using a single pointer. The CUDA system software automatically migrates data allocated in Unified Memory between GPU and CPU, so that it looks like CPU memory to code running on the CPU, and like GPU memory to code running on the GPU. But CUDA 6 Unified Memory was limited by the features of the Kepler and Maxwell GPU architectures: All managed memory touched by the CPU had to be synchronized with the GPU before any kernel launch. The CPU and GPU could not simultaneously access a managed memory allocation and the Unified Memory address space was limited to the size of the GPU physical memory.
> 
> ## **CUDA 6 Unified Memory**
> 
> ![](_page_0_Picture_2.jpeg)
> 
> (Limited to Device Memory Size)
> 
> Figure 19. CUDA 6 Unified Memory
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2046.png)

Pascal中CPU和GPU统一内存，虚拟地址空间覆盖系统内存池，支持CPU和GPU之间的缺页自动传输（CPU/GPU使用映射到其他U内存的虚拟地址区域时可触发缺页中断，从其他U内存中调页）。

> **[图片提取文字 (image.png)]:**
> ## Pascal GP100 Unified Memory
> 
> Expanding on the benefits of CUDA 6 Unified Memory, Pascal GP100 adds features to further simplify programming and sharing of memory between CPU and GPU, and allowing easier porting of CPU parallel compute applications to use GPUs for tremendous speedups. Two main hardware features enable these improvements: support for large address spaces and page faulting capability.
> 
> GP100 extends GPU addressing capabilities to enable 49-bit virtual addressing. This is large enough to cover the 48-bit virtual address spaces of modern CPUs, as well as the GPU's own memory. This allows GP100 Unified Memory programs to access the full address spaces of all CPUs and GPUs in the system as a single virtual address space, unlimited by the physical memory size of any one processor (see Figure 21).
> 
> Memory page faulting support in GP100 is a crucial new feature that provides more seamless Unified Memory functionality. Combined with the system-wide virtual address space, page faulting provides several benefits. First, page faulting means that the CUDA system software does not need to synchronize all managed memory allocations to the GPU before each kernel launch. If a kernel running on the GPU accesses a page that is not resident in its memory, it faults, allowing the page to be automatically migrated to the GPU memory on-demand. Alternatively, the page may be mapped into the GPU address space for access over the PCIe or NVLink interconnects (mapping on access can sometimes be faster than migration). Note that Unified Memory is system-wide: GPUs (and CPUs) can fault and migrate memory pages either from CPU memory or from the memory of other GPUs in the system.
> 
> ![](_page_0_Picture_4.jpeg)
> 
> Figure 21. Pascal GP100 Unified Memory is not Limited by the Physical Size of GPU Memory.
> 
> With the new page fault mechanism, global data coherency is guaranteed with Unified Memory. This means that with GP100, the CPUs and GPUs can access Unified Memory allocations without any programmer synchronization. This was illegal on Kepler and Maxwell GPUs because coherency could not be guaranteed if the CPU accessed a Unified Memory allocation while a GPU kernel was active.
> 
> ![](_page_0_Picture_7.jpeg)
> 
> **Note:** As with any parallel application, developers need to ensure correct synchronization to avoid data hazards between processors.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2047.png)

> **[图片提取文字 (image.png)]:**
> Finally, on supporting operating system platforms, memory allocated with the default OS allocator (for example, malloc or new) can be accessed from both GPU code and CPU code using the same pointer (see Figure 22). On these systems, Unified Memory can be the default: there is no need to use a special allocator or for the creation of a special managed memory pool. Moreover, GP100's large virtual address space and page faulting capability enable applications to access the entire system virtual memory. This means that applications are permitted to oversubscribe the memory system: in other words they can allocate, access, and share arrays larger than the total physical capacity of the system, enabling out-of-core processing of very large datasets.
> 
> Certain operating system modifications are required to enable Unified Memory with the system allocator. NVIDIA is collaborating with Red Hat and working within the Linux community to enable this powerful functionality.
> 
> ## **CPU Code**
> 
> ```
> void sortfile(FILE *fp, int N) {
>   char *data;
>   data = (char *)malloc(N);
> 
>   fread(data, 1, N, fp);
> 
>   qsort(data, N, 1, compare);
> 
>   use_data(data);
> 
>   free(data);
> }
> ```
> 
> ## Pascal Unified Memory\*
> 
> ```
> void sortfile(FILE *fp, int N) {
>   char *data;
>   data = (char *)malloc(N);
> 
>   fread(data, 1, N, fp);
> 
>   qsort<<<...>>>(data,N,1,compare);
>   cudaDeviceSynchronize();
> 
>   use_data(data);
> 
>   free(data);
> }
> 
>   *with operating system support
> ```
> 
> Figure 22. With Operating System Support, Pascal is Capable of Supporting Unified Memory with the Default System Allocator.
> 
> (Here, malloc is all that is needed to allocate memory accessible from any CPU or GPU in the system.)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2048.png)

**计算任务的指令级抢占（接口）**，允许切换其他应用Ctx。Kepler的抢占接口只支持CTA级抢占，等待CTA完成后才能切换Ctx。

支持指令级抢占接口之前，需要将长任务切分成时间片执行，以便CTA尽快结束（drain）后被抢占和切换Ctx，也不必被OS强制结束（重置GPU，导致任务reset）。

> **[图片提取文字 (image.png)]:**
> The new Pascal GP100 Compute Preemption feature allows compute tasks running on the GPU to be interrupted at instruction-level granularity, and their context swapped to GPU DRAM. This permits other applications to be swapped in and run, followed by the original task's context being swapped back in to continue execution where it left off.
> 
> Compute Preemption solves the important problem of long-running or ill-behaved applications that can monopolize a system, causing the system to become unresponsive while it waits for the task to complete, possibly resulting in the task timing out and/or being killed by the OS or CUDA driver. Before Pascal, on systems where compute and display tasks were run on the same GPU, long-running compute kernels could cause the OS and other visual applications to become unresponsive and non-interactive until the kernel timed out. Because of this, programmers had to either install a dedicated compute-only GPU or carefully code their applications around the limitations of prior GPUs, breaking up their workloads into smaller execution timeslices so they would not time out or be killed by the OS.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2049.png)

> **[图片提取文字 (image.png)]:**
> Indeed, many applications do require long-running processes, and with Compute Preemption in GP100, those applications can now run as long as they need when processing large datasets or waiting for specific conditions to occur, while visual applications remain smooth and interactive—but not at the expense of the programmer struggling to get code to run in small timeslices.
> 
> Compute Preemption also permits interactive debugging of compute kernels on single-GPU systems. This is an important capability for developer productivity. In contrast, the Kepler GPU architecture only provided coarser-grained preemption at the level of a block of threads in a compute kernel. This blocklevel preemption required that all threads of a thread block complete before the hardware can context switch to a different context. However when using a debugger and a GPU breakpoint was hit on an instruction within the thread block, the thread block was not complete, preventing block-level preemption. While Kepler and Maxwell were still able to provide the core functionality of a debugger by adding instrumentation during the compilation process, GP100 is able to support a more robust and lightweight debugger implementation.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2050.png)

### 图形pipeline的Fast Sync

**图形pipeline的V-SYNC和Fast SYNC**

开启V-SYNC要求pipeline降速到和display刷新率相同（pipeline速度被display“后端阻压”），让Engine发任务、pipeline处理和display帧率一致，防止帧抖动tearing，但引入较高的输入延迟（Engine发任务慢）。

关闭V-SYNC让pipeline全速处理任务，不必要和display刷新率同步，导致帧抖动，但输入延迟低。

> **[图片提取文字 (image.png)]:**
> ## Fast Sync
> 
> Fast Sync is a latency-conscious alternative to traditional Vertical Sync (V-SYNC) that eliminates tearing, while allowing the GPU to render unrestrained by the refresh rate to reduce input latency.
> 
> ## Rendered Frames – Traditional Method
> 
> This is a rough outline of how frame rendering works through the NVIDIA graphics pipeline:
> 
> ![](_page_0_Picture_4.jpeg)
> 
> The game engine is responsible for generating the frames that are sent to DirectX. The game engine also calculates animation time; the encoding inside the frame that eventually gets rendered. The draw calls and information are communicated forward, the NVIDIA driver and GPU converts them into actual rendering, and then spits out a rendered frame to the GPU frame buffer. The last step is to scan the frame to the display.
> 
> We are doing something different now with Pascal.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ## **HIGH FPS Games**
> 
> High FPS games like *Counter-Strike: Global Offensive* are running at many hundreds of frames per second today on Pascal. The question is: what good is that? Today, there are two choices on how to display the game; with V-SYNC ON or with V-SYNC OFF.
> 
> |               | V-SYNC ON    | V-SYNC OFF |
> |---------------|--------------|------------|
> | Flow Control  | Backpressure | None       |
> | Input Latency | High         | Low        |
> | Frame Tearing | None         | Tearing    |
> 
> If you use V-SYNC ON, the pipeline gets back-pressured all the way to the game engine, and the entire pipeline slows down to the refresh rate of the display. With V-SYNC ON, the display is essentially telling the game engine to slow down, because only one frame can be effectively generated for every display refresh interval. The upside of V-SYNC ON is the elimination of frame tearing, but the downside is high input latency.
> 
> When using V-SYNC OFF, the pipeline is told to ignore the display refresh rate, and to deliver game frames as fast as possible. The upside of V-SYNC OFF is low input latency (as there is no backpressure), but the downside is frame tearing.
> 
> These are the choices that gamers face today, and the vast majority of eSports gamers are playing with V-SYNC OFF to leverage its lower input latencies, lending them a competitive edge. Unfortunately, tearing at high FPS causes a vast amount of jittering, which can hamper their gameplay.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2052.png)

架构改变：增加更大GPU frame buffer匹配高速GPU的pipeline和低速Display的Scan，将pipeline中的GPU→Display和Game Engine→GPU两个阶段解耦，打破耦合pipeline同步导致高输入延迟，或者耦合pipeline异步导致帧抖动。

新架构支持的pipeline的新执行模式Fast Sync中，Engine、GPU和display不同步，但frame buffer按照刷新率输出以同步Display。

> **[图片提取文字 (image.png)]:**
> ## **Decoupled Render and Display**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2053.png)

> **[图片提取文字 (image.png)]:**
> NVIDIA has taken another look at how the traditional process works, and for the first time, rendering and display are being decoupled from the pipeline. This allows the rendering stage to continually generate new frames from data sent by the game engine and driver at full speed, and those frames can can be temporarily stored in the GPU frame buffer.
> 
> ## **Rendered Frames - FAST SYNC**
> 
> NVIDIA has decoupled the front end of the render pipeline from the backend display hardware. This allows different ways to manipulate the display that can deliver new benefits to gamers. Fast Sync is one of the first applications of this new approach.
> 
> With Fast Sync, there is no flow control. The game engine works as if V-SYNC is OFF. And because there is no backpressure, input latency is almost as low as with V-SYNC OFF. Best of all, there is no tearing because FAST SYNC chooses which of the rendered frames to scan to the display. FAST SYNC allows the front of the pipeline to run as fast as it can, and it determines which frames to scan out to the display, while simultaneously preserving entire frames so they are displayed without tearing.
> 
> |               | V-SYNC ON    | V-SYNC OFF | FAST SYNC |
> |---------------|--------------|------------|-----------|
> | Flow Control  | Backpressure | None       | None      |
> | Input Latency | High         | Low        | Low       |
> | Frame Tearing | None         | Tearing    | None      |
> 
> The experience that FAST SYNC delivers, depending on frame rate, is roughly equal to the clarity of V-SYNC ON combined with the low latency of V-SYNC OFF.
> 
> ## **Decoupled Buffers**
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2054.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> One way to think about Fast Sync is to imagine that three areas in the frame buffer have been allocated in three different ways. The first two buffers are very similar to double-buffered VSYNC in classic GPU pipelines. The Front Buffer (FB) is the buffer scanned out to the display. It is a fully rendered surface. The Back Buffer (BB) is the buffer that is currently being rendered to and it can't be scanned out until it is completed. Using traditional VSYNC In high render-rate games is not good for latency, since the game must wait for the display refresh interval to flip the back buffer to become the front buffer before another frame can be rendered into the back buffer. This slows down the entire process...and adding additional back buffers just adds latency, since they could all fill up at high rendering rates, causing similar stalls of the game engine.
> 
> Fast Sync introduces a third buffer called the Last Rendered Buffer (LRB) which is used to hold all newly rendered frames just completed in the back buffer – in effect having a copy of the most recently rendered back buffer - until the front buffer has finished scanning, at which point the Last Rendered Buffer is copied to the Front buffer and the process continues. Actual buffer copies would be inefficient, so instead the buffers are just renamed. The buffer being scanned to the display is the FB, the buffer being actively rendered to is the BB and the buffer holding the most recently rendered frame is the LRB. New flip logic in the Pascal architecture controls the entire process.
> 
> A typical process would look like this:
> 
> - Scan from FB
> - Render to BB
> - When Render completes
>   - BB becomes LRB
>   - LRB becomes BB and render continues
> - When Render completes
>   - BB becomes LRB
>   - LRB becomes BB and render continues
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2055.png)

> **[图片提取文字 (image.png)]:**
> LRB becomes BB and render continues
> When scan completes
> 
> BB becomes LRB
> 
> When Render completes
> 
> LRB becomes FB
> Start scanning from the new FB
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2056.png)

**游戏中异步的多计算负载**：**物理计算和媒体处理的GPU并发overlap**；渲染帧的后处理；**VR中异步timewarp（ATW）在刷新间隔内，基于头的实时位置修改当前帧**。

图形GPU命令包含多个draw call，每个draw call包含很多三角形，每个三角形包含很多pixel的渲染计算，命令长延时且一般不被抢占。

Pascal支持**Pixel Level Preemption**，图形命令的执行拆分成Pixel Level执行，添加抢占接口和对应架构支持。

> **[图片提取文字 (image.png)]:**
> ## **Asynchronous Compute**
> 
> Modern gaming workloads are increasingly complex, with multiple independent, or "asynchronous," workloads that ultimately work together to contribute to the final rendered image. Some examples of asynchronous compute workloads include:
> 
> - GPU-based physics and audio processing
> - Postprocessing of rendered frames
> - Asynchronous timewarp, a technique used in VR to regenerate a final frame based on head position just before display scanout, interrupting the rendering of the next frame to do so
> 
> These asynchronous workloads create two new scenarios for the GPU architecture to consider.
> 
> The first scenario involves overlapping workloads. Certain types of workloads do not fill the GPU completely by themselves. In these cases there is a performance opportunity to run two workloads at the same time, sharing the GPU and running more efficiently—for example a PhysX workload running concurrently with graphics rendering.
> 
> For overlapping workloads, Pascal introduces support for "dynamic load balancing." In Maxwell generation GPUs, overlapping workloads were implemented with static partitioning of the GPU into a subset that runs graphics, and a subset that runs compute. This is efficient provided that the balance of work between the two loads roughly matches the partitioning ratio. However, if the compute workload takes longer than the graphics workload, and both need to complete before new work can be done, and the portion of the GPU configured to run graphics will go idle. This can cause reduced performance that may exceed any performance benefit that would have been provided from running the workloads
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2057.png)

> **[图片提取文字 (image.png)]:**
> overlapped. Hardware dynamic load balancing addresses this issue by allowing either workload to fill the rest of the machine if idle resources are available.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 10: Pascal's Dynamic Load Balancing Reduces GPU Idle Time When Graphics Work Finishes Early,
> 
> Allowing the GPU to Quickly Switch to Compute
> 
> Time critical workloads are the second important asynchronous compute scenario. For example, an asynchronous timewarp operation must complete before scanout starts or a frame will be dropped. In this scenario, the GPU needs to support very fast and low latency preemption to move the less critical workload off of the GPU so that the more critical workload can run as soon as possible.
> 
> As a single rendering command from a game engine can potentially contain hundreds of draw calls, with each draw call containing hundreds of triangles, and each triangle containing hundreds of pixels that have to be shaded and rendered. A traditional GPU implementation that implements preemption at a high level in the graphics pipeline would have to complete all of this work before switching tasks, resulting in a potentially very long delay.
> 
> To address this issue, Pascal is the first GPU architecture to implement Pixel Level Preemption. The graphics units of Pascal have been enhanced to keep track of their intermediate progress on rendering work, so that when preemption is requested, they can stop where they are, save off context information about where to start up again later, and preempt quickly. The illustration below shows a preemption request being executed.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2058.png)

### Pixel&Thd Level抢占

命令流存储在pushbuffer，每个命令包含若干三角形queue TMD或grid TMD，queue TMD按照pixel为单位执行，grid TMD按照CTA为单位执行。

不同level抢占的共性是等待不同level负载完成后抢占，支持pixel level、CTA level和instr level抢占。

instr抢占不必等待CTA完成而只需等待指令完成，因此抢占延迟确定且更短，抢占指令不必过早发出就能保证ATW能及时开始和完成，减少GPU抢占开销。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Figure 11: Pascal Supports Pixel- Level Graphics Preemption, Allowing the GPU to Switch Workloads Mid-Triangle
> 
> In the command pushbuffer, three draw calls have been executed, one is in process and two are waiting. The current draw call has six triangles, three have been processed, one is being rasterized and two are waiting. The triangle being rasterized is about halfway through. When a preemption request is received, the rasterizer, triangle shading and command pushbuffer processor will all stop and save off their current position. Pixels that have already been rasterized will finish pixel shading and then the GPU is ready to take on the new high priority workload. The entire process of switching to a new workload can complete in less than 100 microseconds (µs) after the pixel shading work is finished.
> 
> Pascal also has enhanced preemption support for compute workloads. The illustration below shows the execution of a compute workload.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> Figure 12: Pascal Supports Compute Preemption at the Thread Level for DX12 Graphics
> 
> Thread Level Preemption for compute operates similarly to Pixel Level Preemption for graphics.
> 
> Compute workloads are composed of multiple grids of thread blocks, each grid containing many threads.
> 
> When a preemption request is received, the threads that are currently running on the SMs are completed. Other units save their current position to be ready to pick up where they left off later, and then the GPU is ready to switch tasks. The entire process of switching tasks can complete in less than 100 µs after the currently running threads finish.
> 
> For gaming workloads, the combination of pixel level graphics preemption and thread level compute preemption gives Pascal the ability to switch workloads extremely quickly with minimal preemption overhead.
> 
> For CUDA compute tasks, Pascal is also capable of preempting at the finest granularity possible—instruction level.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2059.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 13: Pascal GPUs Support Instruction-Level Compute
> Preemption when Running CUDA Apps
> 
> In this mode of operation, when a preemption request is received, all thread processing stops at the current instruction and state is switched out immediately. This mode of operation involves substantially more state information, because all the registers of every running thread must be saved, but this is the most robust approach for general GPU compute workloads that may have substantial per-thread runtimes.
> 
> One example application of preemption in gaming is asynchronous timewarp. The left side of the illustration below shows an asynchronous timewarp operation with traditional GPU preemption. The ATW process runs as late as possible before the display refresh interval. However the ATW work has to be given to the GPU several milliseconds in advance, because without fine grained preemption, there is variability in the time it will take to preempt and start execution of the ATW process. On the right image, with fine-grained preemption (pixel level graphics plus thread level compute preemption), the preemption time is much faster and more deterministic, so the ATW work can be submitted much later, while still being assured of completion before the display refresh deadline.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2060.png)

> **[图片提取文字 (image.png)]:**
> ## PREEMPTION IN VR
> 
> ## **IMPROVED LATENCY & SPEED**
> 
> Traditional GPU
> 
> Conservative Preemption Request **GRAPHICS OR COMPUTE ATW** 11 ms (90 Hz) Display Display Refresh Refresh
> 
> Pascal with Fine-grained Preemption Support
> 
> ![](_page_0_Figure_5.jpeg)
> 
> NVIDIA GEFORCE GTX
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2061.png)

### SMP几何引擎

同时多投影引擎SMP Engine能对2个投影中心分别完成16次投影计算，提供dataflow满足几何pipeline（几何命令Stream）的多投影需求，节省几何负载繁重场景下使用原有dataflow存在的重复计算。

> **[图片提取文字 (image.png)]:**
> ## Simultaneous Multi-Projection Engine
> 
> The Simultaneous Multi-Projection block is a new hardware unit, which is located inside the PolyMorph Engine at the end of the geometry pipeline and right in front of the Raster Unit. As its name implies, the Simultaneous Multi-Projection (SMP) unit is responsible for generating multiple projections of a single geometry stream, as it enters the SMP engine from upstream shader stages.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2062.png)

> **[图片提取文字 (image.png)]:**
> The Simultaneous Multi-Projection Engine is capable of processing geometry through up to 16 preconfigured projections, sharing the center of projection (the viewpoint), and with up to 2 different projection centers, offset along the X axis. Projections can be independently tilted or rotated around an axis. Since each primitive may show up in multiple projections simultaneously, the SMP engine provides multi-cast functionality, allowing the application to instruct the GPU to replicate geometry up to 32 times (16 projections x 2 projection centers) without additional application overhead as the geometry flows through the pipe.
> 
> In all scenarios, the processing is hardware-accelerated, and the stream of data never leaves the chip. Since the multi-projection expansion happens after the geometry pipeline, the application saves all the work that would otherwise need to be performed in upstream shader stages. The savings are particularly important in geometry-heavy scenarios, such as tessellation, where running the geometry processing pipeline multiple times (once for each projection) would be prohibitively expensive. In extreme cases, the SMP engine can reduce the amount of required geometry work by up to 32x!
> 
> One example application of SMP is optimal support for surround displays. The correct way to render to a surround display is with a different projection for each of the three displays, matching the display angle. This is supported directly in a single pass by Pascal SMP, by specifying three separate projections, each corresponding to the appropriately tilted monitor. Now, the user has the flexibility to choose the desired tilt for their side displays and will see their graphics rendered with geometrically correct perspectives, at a much wider field of view (FOV). Note that an application using SMP to generate surround display images must support wide FOV settings, and also use SMP API calls to enable the wider FOV.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 15: GeForce GTX 1080 Features a New PolyMorph Engine that Supports Simultaneous Multi-Projection
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2064.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 16: Surround setups with SMP perspective correction (left) and without SMP perspective correction (right). Note how in the right image the rendering frustum used by the application is inconsistent with display placement, resulting in geometric distortions on the side displays
> 
> In cases where the projection surface can't be exactly represented with finite number of planar projections, the SMP engine can still provide substantial efficiency gains by generating a much closer approximation to the desired projection surface.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2065.png)

## Volta（合并L1和SharedMem、Tensor Core、FP32+INT32并行、独立线程调度、MPS、CG）

### Tensor Core

合并SM中的L1 Cache和SharedMem，提高L1性能，简化SharedMem的编程管理开销，之前SharedMem没有设置Cache，需要**精细管理（作为Buffer）**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2066.png)

> **[图片提取文字 (image.png)]:**
> ## **Enhanced L1 Data Cache and Shared Memory**
> 
> The new combined L1 data cache and shared memory subsystem of the Volta SM significantly improves performance while also simplifying programming and reducing the tuning required to attain at or near-peak application performance.
> 
> Combining data cache and shared memory functionality into a single memory block provides the best overall performance for both types of memory accesses. The combined capacity is 128 KB/SM, more than seven times larger than the GP100 data cache, and all of it is usable as a cache by programs that do not use shared memory. Texture units also use the cache. For example, if shared memory is configured to 64 KB, texture and load/store operations can use the remaining 64 KB of L1.
> 
> Integration within the shared memory block ensures the Volta GV100 L1 cache has much lower latency and higher bandwidth than the L1 caches in past NVIDIA GPUs. The L1 In Volta functions as a high-throughput conduit for streaming data while simultaneously providing high-bandwidth and low-latency access to frequently reused data—the best of both worlds. This combination is unique to Volta and delivers more accessible performance than in the past.
> 
> A key reason to merge the L1 data cache with shared memory in GV100 is to allow L1 cache operations to attain the benefits of shared memory performance. Shared memory provides high bandwidth, low latency, and consistent performance (no cache misses), but the CUDA programmer needs to explicitly manage this memory. Volta narrows the gap between applications that explicitly manage shared memory and those that access data in device memory directly. To demonstrate this, we modified a suite of programs by replacing shared memory arrays with device memory arrays so that accesses would go through L1 cache. As Figure 11 shows, on Volta these codes saw only a 7% performance loss running without using shared memory, compared to a 30% performance loss on Pascal. While shared memory remains the best choice for maximum performance, the new Volta L1 design enables programmers to get excellent performance quickly, with less programming effort.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2067.png)

> **[图片提取文字 (image.png)]:**
> ## **L1 Instruction Cache**
> 
> ![](_page_0_Figure_2.jpeg)
> 
> | Warp Scheduler (32 thread/clk)  |           |           |           |           |           |           |           |                |
> |---------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|----------------|
> | Dispatch Unit (32 thread/clk)   |           |           |           |           |           |           |           |                |
> | Register File (16,384 x 32-bit) |           |           |           |           |           |           |           |                |
> | FP                              | 64        | INT       | INT       | FP32      | FP32      |           |           |                |
> | FP64                            |           | INT       | INT       | FP32      | FP32      |           |           | TENSOR<br>CORE |
> | FP64                            |           | INT       | INT       | FP32      | FP32      |           |           |                |
> | FP64                            |           | INT       | INT       | FP32      | FP32      | TENSOR    |           |                |
> | FP64                            |           | INT       | INT       | FP32      | FP32      | CORE      |           |                |
> | FP64                            |           | INT       | INT       | FP32      | FP32      |           |           |                |
> | FP64                            |           | INT       | INT       | FP32      | FP32      |           |           |                |
> | FP64                            |           | INT       | INT       | FP32      | FP32      |           |           |                |
> | LD/<br>ST                       | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | SFU            |
> 
> **L0 Instruction Cache** 
> 
> Warp Scheduler (32 thread/clk)
> 
> Dispatch Unit (32 thread/clk)
> 
> **L0 Instruction Cache** 
> 
> ## **L0 Instruction Cache** Warp Scheduler (32 thread/clk) Dispatch Unit (32 thread/clk) Register File (16,384 x 32-bit) FP32 FP32 INT INT FP64 FP32 FP32 FP64 INT INT INT FP32 FP32 INT FP64 FP64 FP32 FP32 INT INT **TENSOR TENSOR** CORE CORE FP32 FP32 FP64 INT INT FP64 INT INT FP32 FP32 FP32 FP32 FP64 INT INT FP64 INT INT FP32 FP32 LD/ LD/ LD/ LD/ LD/ LD/ LD/ LD/ **SFU** ST ST ST ST ST ST ST ST
> 
> ## Register File (16,384 x 32-bit) FP32 FP32 INT INT FP64 FP32 FP32 INT INT FP64 FP32 FP32 INT INT FP64 FP32 FP32 FP64 INT INT **TENSOR TENSOR** CORE CORE FP32 FP32 FP64 INT INT FP32 FP32 INT FP64 INT FP32 FP32 FP64 INT INT FP32 FP32 FP64 INT INT LD/ LD/ LD/ LD/ LD/ LD/ LD/ LD/ SFU ST ST ST ST ST ST ST ST
> 
> ## 128KB L1 Data Cache / Shared Memory
> 
> Tex Tex Tex Tex
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2068.png)

**Tensor Core**是**矩阵乘法的特化datapath**，比Cuda Core的矩阵乘更快。warp-level矩阵乘法操作。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 7. cuBLAS Mixed Precision (FP16 Input, FP32 Compute)
> 
> Tensor Cores and their associated data paths are custom-designed to dramatically increase floating-point compute throughput with high energy efficiency.
> 
> Each Tensor Core operates on a 4x4 matrix and performs the following operation:
> 
> $$D = A \times B + C$$
> 
> where **A**, **B**, **C**, and **D** are 4x4 matrices (Figure 8). The matrix multiply inputs **A** and **B** are FP16 matrices, while the accumulation matrices **C** and **D** may be FP16 or FP32 matrices (see Figure 8).
> 
> ![](_page_0_Figure_6.jpeg)
> 
> Figure 8. Tensor Core 4x4 Matrix Multiply and Accumulate
> 
> Tensor Cores operate on FP16 input data with FP32 accumulation. The FP16 multiply results in a full precision product that is then accumulated using FP32 addition with the other intermediate products for a 4x4x4 matrix multiply (see Figure 9). In practice, Tensor Cores are used to perform much larger 2D or higher dimensional matrix operations, built up from these smaller elements.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2069.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 9. Mixed Precision Multiply and Accumulate in Tensor Core
> 
> Figure 10 shows the 4x4 matrix multiplication (using the two source 4x4 matrices outside the cube) requiring 64 operations (represented by the cube) to generate a 4x4 output matrix (shown below the cube). The Volta-based V100 accelerator with Tensor Cores can perform such calculations at 12x faster rate than Pascal-based Tesla P100.
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Figure 10. Pascal and Volta 4x4 Matrix Multiplication
> 
> The Volta tensor cores are accessible and exposed as Warp-Level Matrix Operations in the CUDA 9 C++ API. The API exposes specialized matrix load, matrix multiply and accumulate, and matrix store operations to efficiently use Tensor Cores from a CUDA-C++ program. At the CUDA level, the warp-level interface assumes 16x16 size matrices spanning all 32 threads of the warp.
> 
> In addition to CUDA-C++ interfaces to program Tensor Cores directly, cuBLAS and cuDNN libraries have been updated to provide new library interfaces to make use of Tensor Cores for deep learning applications and frameworks. NVIDIA has worked with many popular deep learning frameworks such as Caffe2 and MXNet to enable use of Tensor Cores for deep learning research on Volta GPU based systems. NVIDIA is working to add support for Tensor Cores in other frameworks as well.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2070.png)

### 独立线程调度 for 分支并行和Starvation-Free（解决死锁）

独立线程调度给每个线程设置独立的PC和call stack空间，线程可以脱离warp单独“前进”，而不必束缚于warp，不存在warp内active线程对inactive线程的“PC结构”阻塞。

之前的SIMT中warp内所有线程共享PC，warp内线程一旦脱离warp，就无法正确执行，因为线程没有单独的PC和call stack管理指令执行，导致warp内active线程阻塞inactive线程的执行。

> **[图片提取文字 (image.png)]:**
> ## INDEPENDENT THREAD SCHEDULING
> 
> The Volta architecture is designed to be significantly easier to program than prior GPUs, enabling users to work productively on more complex and diverse applications. Volta GV100 is the first GPU to support independent thread scheduling, which enables finer-grain synchronization and cooperation between parallel threads in a program. One of the major design goals for Volta was to reduce the effort required to get programs running on the GPU, and to enable greater flexibility in thread cooperation, leading to higher efficiency for fine-grained parallel algorithms.
> 
> ## Prior NVIDIA GPU SIMT Models
> 
> Pascal and earlier NVIDIA GPUs execute groups of 32 threads (known as warps) in SIMT (Single Instruction, Multiple Thread) fashion. The Pascal warp uses a single program counter shared amongst all 32 threads, combined with an active mask that specifies which threads of the warp are active at any given time. This means that divergent execution paths leave some threads inactive, serializing execution for different portions of the warp as shown in Figure 20. The original mask is stored until the warp reconverges, typically at the end of the divergent section, at which point the mask is restored and the threads run together once again.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> Thread scheduling under the SIMT warp execution model of Pascal and earlier NVIDIA GPUs. Capital letters represent statements in the program pseudocode. Divergent branches within a warp are serialized so that all statements in one side of the branch are executed together to completion before any statements in the other side are executed. After the else statement, the threads of the warp will typically reconverge.
> 
> Figure 20. SIMT Warp Execution Model of Pascal and Earlier GPUs
> 
> The Pascal SIMT execution model maximizes efficiency by reducing the quantity of resources required to track thread state and by aggressively reconverging threads to maximize parallelism. Tracking thread state in aggregate for the whole warp, however, means that when the execution pathway diverges, the threads which take different branches lose concurrency until they reconverge. This loss of concurrency means that threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data. This presents an inconsistency in which threads from different warps continue to run concurrently, but diverged threads from the same warp run sequentially until they reconverge. This means, for example, that algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from. Therefore, on Pascal and earlier GPUs, programmers need to avoid fine-grained synchronization or rely on lock-free or warp-aware algorithms.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2071.png)

> **[图片提取文字 (image.png)]:**
> ## Volta SIMT Model
> 
> Volta transforms this picture by enabling equal concurrency between all threads, regardless of warp. It does this by maintaining execution state per thread, including a program counter and call stack, as shown in Figure 21.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> 32 thread warp with independent scheduling
> 
> Volta (bottom) independent thread scheduling architecture block diagram compared to Pascal and earlier architectures (top). Volta maintains per-thread scheduling resources such as program counter (PC) and call stack (S), while earlier architectures maintained these resources per warp.
> 
> ## Figure 21. Volta Warp with Per-Thread Program Counter and Call Stack
> 
> Volta's independent thread scheduling allows the GPU to yield execution of any thread, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. To maximize parallel efficiency, Volta includes a schedule optimizer which determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity, while the convergence optimizer in Volta will still group together threads which are executing the same code and run them in parallel for maximum efficiency
> 
> Execution of the code example from Figure 20 looks somewhat different on Volta. Statements from the *if* and *else* branches in the program can now be interleaved in time as shown in Figure 22. Note that execution is still SIMT: at any given clock cycle, CUDA cores execute the same instruction for all active threads in a warp just as before, retaining the execution efficiency of previous architectures. Importantly, Volta's ability to independently schedule threads within a warp makes it possible to implement complex, fine-grained algorithms and data structures in a more natural way. While the scheduler supports independent execution of threads, it optimizes non-synchronizing code to maintain as much convergence as possible for maximum SIMT efficiency.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2072.png)

正确的**多线程程序**，让线程总能等待并获得mutex，而不需要考虑结构性阻塞，因为不存在warp内active线程对inactive线程的“PC结构”阻塞。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Volta independent thread scheduling enables interleaved execution of statements from divergent branches. This enables execution of fine-grain parallel algorithms where threads within a warp may synchronize and communicate.
> 
> ## Figure 22. Volta Independent Thread Scheduling
> 
> It is interesting to note that Figure 22 does not show execution of statement **Z** by all threads in the warp at the same time. This is because the scheduler must conservatively assume that **Z** may produce data required by other divergent branches of execution, in which case it would be unsafe to automatically enforce reconvergence. In the common case where **A**, **B**, **X**, and **Y** do not consist of synchronizing operations, the scheduler can identify that it is safe for the warp to naturally reconverge on **Z**, as in prior architectures.
> 
> Programs can call the new CUDA 9 warp synchronization function \_\_syncwarp() to force reconvergence, as shown in Figure 23. In this case, the divergent portions of the warp might not execute Z together, but all execution pathways from threads within a warp will complete before any thread reaches the statement after the \_\_syncwarp(). Similarly, placing the call to \_\_syncwarp() before the execution of Z would force reconvergence before executing Z, potentially enabling greater SIMT efficiency if the developer knows that this is safe for their application.
> 
> ```
> if (threadIdx.x < 4) {
>     A;
>     B;
> } else {
>     X;
>     Y;
>     Z;
>     A;
>     B;
> } z;
> </pre>
> 
> A;
> B;
> Time
> ```
> 
> Figure 23. Programs use Explicit Synchronization to Reconverge Threads in a Warp
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2073.png)

> **[图片提取文字 (image.png)]:**
> ## Starvation-Free Algorithms
> 
> Starvation-free algorithms are a key pattern enabled by independent thread scheduling. These are concurrent computing algorithms that are guaranteed to execute correctly so long as the system ensures that all threads have adequate access to a contended resource. For example, a mutex (or lock) may be used in a starvation-free algorithm if a thread attempting to acquire the mutex is guaranteed eventually to succeed. In a system that does not support starvation-freedom, one or more threads may repeatedly acquire and release a mutex while starving another thread from ever successfully acquiring the mutex.
> 
> Consider a simplified example that Volta's independent thread scheduling enables: inserting nodes into a doubly linked list in a multithreaded application.
> 
> ```
> __device__ void insert_after(Node *a, Node *b)
> {
>     Node *c;
>     lock(a); lock(a->next);
>     c = a->next;
> 
>     a->next = b;
>     b->prev = a;
> 
>     b->next = c;
>     c->prev = b;
> 
>     unlock(c); unlock(a);
> }
> ```
> 
> In this example, each element of a doubly linked list has at least three components: a *next pointer*, a *previous pointer*, and a *lock* providing the owner exclusive access to update the node. Figure 24 shows the insertion of node **B** after node **A** with updates to the *next* and *previous* pointers of **A** and **C**.
> 
> ![](_page_0_Figure_5.jpeg)
> 
> Per-node locks are acquired (left) before inserting node B into the list (right).
> 
> Figure 24. Doubly Linked List with Fine-Grained Locks
> 
> Independent thread scheduling in Volta ensures that even if a thread TO currently holds the lock for node A, another thread T1 in the same warp can successfully wait for the lock to become available without impeding the progress of thread TO. Note, however, that because active threads in a warp execute together, threads spinning on a lock may degrade the performance of the thread holding the lock.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2074.png)

### MPS（单用户的多应用）、统一内存和地址翻译ATS

CPU的**MPS进程把相同用户的多个应用的任务打包成相同应用的命令流**，**共享地址空间和资源**，GPU将其当作单应用，只拥有应用内的多任务隔离性，承受单应用的性能干扰和异常处理打扰。

Kepler将多个协作App Ctx**打包**成一个App Ctx，多个App共享GPU资源。

Pascal设置**中介**CPU进程将协作App的任务作为自己的任务提交给GPU。

Volta允许多个App的任务**直接提交到GPU的channel**，并发多个App的kernel。

Volta的不同任务可以设置访问资源，限制任务间性能干扰。

**内存设置Access Counter**让GPU内存缓存其他处理器内存的物理页，减少跨U传输。

**地址翻译ATS**允许GPU直接查询CPU的页表，GPU的MMU缺页时，可向CPU请求访问页表（减少CPU在缺页中断中参与的深度）。此外，ATS允许GPU从CPU申请内存malloc。

> **[图片提取文字 (image.png)]:**
> ## **VOLTA MULTI-PROCESS SERVICE**
> 
> Volta Multi-Process Service (MPS) is a new feature of the Volta GV100 architecture enabling improved performance and isolation for multiple compute applications sharing the GPU. Typical execution of multiple applications sharing the GPU is implemented with time-slicing, that is, each application gets exclusive access for a period of time before access is granted to another application. Volta MPS improves aggregate GPU utilization by allowing multiple applications to simultaneously share GPU execution resources when these applications individually under-utilize the GPU execution resources.
> 
> Starting with Kepler GK110 GPUs, NVIDIA introduced a software-based Multi-Process Service (MPS) and MPS Server that allowed multiple different CPU processes (application contexts) to be combined into a single application context and run on the GPU, attaining higher GPU resource utilization.
> 
> Volta MPS provides hardware acceleration of critical components of the MPS server for improved performance and isolation, while increasing the maximum number of MPS clients from 16 on Pascal up to 48 on Volta (see Figure 25). Volta Multi-Process service is designed for sharing the GPU amongst applications from a single user and is not for multi-user or multi-tenant use cases.
> 
> For Pascal, CUDA Multi-Process Service is a CPU process which acts on behalf of GPU applications that have requested to simultaneously share execution resources with other GPU applications.
> 
> This process acts as the intermediary to submit work to the work queues inside the GPU for concurrent kernel execution.
> 
> The Volta Multi-Process Service provides hardware acceleration of CUDA MPS which enables MPS clients to submit work directly to the work queues within the GPU. This acceleration significantly decreases submission latency and increases aggregate throughput. For Volta, the CPU MPS control process remains for configuration and opt-in to the MPS.
> 
> Volta MPS improves isolation amongst MPS clients on two critical metrics: Quality of Service (QoS) and independent address spaces. In Volta, the work from different MPS clients **A**, **B**, and **C** get address isolation as shown in Figure 25, in addition to QoS. Volta MPS, as with CUDA MPS on prior NVIDIA GPUs, does not provide fatal fault isolation between clients.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2075.png)

> **[图片提取文字 (image.png)]:**
> ## Efficient inference deployment without batching system
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 26. Volta MPS for Inference
> 
> A major feature for Volta MPS includes support for the roadmap of Unified Memory features with Linux support (for example, malloc memory access from the GPU). CUDA MPS clients in prior NVIDIA GPU architectures run under a single address space during execution on the GPU, which is incompatible with accessing independent CPU process memories.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2076.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 25. Software-based MPS Service in Pascal vs Hardware-Accelerated MPS Service in Volta
> 
> Quality of Service is how quickly the GPU execution resources will be available to process work for a client upon submission of the work. Volta MPS provides control for MPS clients to specify what fraction of the GPU is necessary for execution. This control to restrict each client to only a fraction of the GPU execution resources reduces or eliminates head-of-line blocking where work from one MPS client may overwhelm GPU execution resources, preventing other clients from making progress until prior work from another MPS client completes. This QoS improvement reduces average latency and jitter in a system, which is critical for both MPI/HPC use cases as well as for deep learning inference use cases.
> 
> Volta provides very high throughput and low latency for deep learning inference particular when there is a batching system in place to aggregate images to submit to the GPU simultaneously to maximize performance. Without such a batching system, individual inference jobs do not fully utilize execution resources of a GPU. Volta MPS provides an easy option to improve throughput while satisfying latency targets, by permitting many individual inference jobs to be submitted concurrently to the GPU and improving overall GPU utilization.
![image.png](../../meeting-26%2003%2024/image%204.png)

> **[图片提取文字 (image.png)]:**
> ## UNIFIED MEMORY AND ADDRESS TRANSLATION SERVICES
> 
> A limited form of Unified Memory was introduced with CUDA 6 in our Kepler and Maxwell GPUs, and it was improved with hardware page faulting and a larger address space in the Pascal GP100 GPU. Unified memory allows a single unified virtual address space for CPU and GPU memory, greatly simplifying GPU programming and porting of applications to GPUs. Programmers no longer need to worry about managing data sharing between GPU and CPU virtual memory systems. Unified Memory in Pascal GP100 provided transparent migration of data between the full virtual address spaces of both the GPU and CPU. (For a detailed explanation of Pascal Unified Memory technology, please see our *Pascal Architecture Whitepaper*.)
> 
> Although Unified Memory in Pascal GP100 improved CUDA programming in many ways, Volta GV100 further improves efficiency and performance of Unified Memory. A new Access Counter feature keeps track of the frequency of access that a GPU makes to memory located on other processors. Access Counters help ensure memory pages are moved to the physical memory of the processor that is accessing the pages most frequently. The Access Counters feature can work in either NVLink- or PCle-connected GPU-CPU or GPU-GPU architectures, and can work with different types of CPUs including Power 9, x86, and others.
> 
> Volta also supports Address Translation Services (ATS) over NVLink. ATS allows the GPU to directly access the CPU's page tables. A miss in the GPU MMU will result in an Address Translation Request (ATR) to the CPU. The CPU looks in its page tables for the virtual-to-physical mapping for
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2077.png)

> **[图片提取文字 (image.png)]:**
> that address and supplies the translation back to the GPU. ATS provides the GPU full access to CPU memory, for example to memory allocated directly with 'malloc'.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2078.png)

### CG API（自定义大小的线程组同步）

CG前静态定义blockSz，只能定义warp和block范围内的线程同步。

CG允许**编程时定义运行时sub-warp、warp、blk、multi-blk、grid、multi-grid范围的线程同步**。

**架构定义执行模型是**SM按block为单位接受负载和分配资源，SIMD Cores按warp为单位执行指令，因此自定义范围的CG运行时按照block为单位调度到SM后启动，按warp为单位发射指令。

**架构定义传输模型是**block内线程可通过SMEM共享数据，warp内线程通过寄存器交换数据，因此自定义范围的CG中线程快速共享数据的方式仍然是block内走SMEM和warp内走寄存器。

> **[图片提取文字 (image.png)]:**
> ## **COOPERATIVE GROUPS**
> 
> In parallel algorithms, threads often need to cooperate to perform collective computations. Building these cooperative codes requires grouping and synchronizing the cooperating threads. CUDA 9 introduces Cooperative Groups, a new programming model for organizing groups of threads.
> 
> Historically, the CUDA programming model has provided a single, simple construct for synchronizing cooperating threads: a barrier across all threads of a thread block, as implemented with the \_\_syncthreads() function. However, programmers would often like to define groups of threads at smaller than thread block granularities and synchronize within them to enable greater performance, design flexibility, and software reuse in the form of "collective" group-wide function interfaces.
> 
> Cooperative Groups introduces the ability to define groups of threads explicitly at sub-block and multiblock granularities, and to perform collective operations such as synchronization on them. This programming model supports clean composition across software boundaries, so that libraries and utility functions can synchronize safely within their local context without having to make assumptions about convergence. It lets developers optimize for the hardware fast path—for example the GPU warp size—using flexible synchronization in a safe, supportable way that makes programmer intent explicit. Cooperative Groups primitives enable new patterns of cooperative parallelism within CUDA, including producer-consumer parallelism, opportunistic parallelism, and global synchronization across the entire grid.
> 
> Cooperative Groups also provides an abstraction by which developers can write flexible, scalable code that will work safely across different GPU architectures, including scaling to future GPU capabilities. Thread groups may range in size from a few threads (smaller than a warp) to a whole thread block, to all thread blocks in a grid launch, to grids spanning multiple GPUs.
> 
> While Cooperative Groups works on all GPU architectures, certain functionality is inevitably architecture-dependent as GPU capabilities have evolved. Basic functionality, such as synchronizing groups smaller than a thread block down to warp granularity, is supported on all architectures, while Pascal and Volta GPUs enable new grid-wide and multi-GPU synchronizing groups. In addition, Volta's independent thread scheduling enables significantly more flexible selection and partitioning of thread groups at arbitrary cross-warp and sub-warp granularities. Volta synchronization is truly per thread: threads in a warp can synchronize from divergent code paths.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2079.png)

> **[图片提取文字 (image.png)]:**
> The Cooperative Groups programming model consists of the following elements:
> 
> - ▶ New mixed-precision FP16/FP32 Tensor Cores purpose-built for deep learning matrix arithmetic
> - ▶ Data types for representing groups of cooperating threads;
> - ▶ Default groups defined by the CUDA launch API (e.g., thread blocks and grids);
> - Operations for partitioning existing groups into new groups;
> - ► A barrier operation to synchronize all threads within a group;
> - ▶ Operations to inspect the group properties as well as group-specific collectives.
> 
> Some basic Cooperative Groups operations are illustrated in the following simple example.
> 
> ```
> __global__ void cooperative_kernel(...)
> {
> 
>     // obtain default "current thread block" group
>     thread_group my_block = this_thread_block();
> 
>     // subdivide into 32-thread, tiled subgroups
>     // Tiled subgroups evenly partition a parent group into
>     // adjacent sets of threads - in this case each one warp in size
>     thread_group my_tile = tiled_partition(my_block, 32);
> 
>     // This operation will be performed by only the
>     // first 32-thread tile of each block
>     if (my_block.thread_rank() < 32) {
>         ...
>         my_tile.sync();
>     }
> }</pre>
> ```
> 
> Cooperative Groups uses C++ templates to provide types and API overloads to represent groups whose size is statically determined for even greater efficiency. The language-level interface is supported by a set of PTX assembly extensions that provide the substrate for the CUDA C++ implementation. These PTX extensions are also available to any programming system that wants to provide similar functionality. Finally, the race detection tool in cuda-memcheck and the CUDA debugger are compatible with the more flexible synchronization patterns permitted by Cooperative Groups, to make it easier to find subtle parallel synchronization bugs such as Read After Write (RAW) hazards.
> 
> Cooperative Groups allows programmers to express synchronization patterns that they were previously unable to express. When the granularity of synchronization corresponds to natural architectural granularities (warps and thread blocks), the overhead of this flexibility is negligible. Libraries of collective primitives written using Cooperative Groups often require less complex code to achieve high performance.
> 
> Consider a particle simulation, where we have two main computation phases in each step of the simulation. First, integrate the position and velocity of each particle forward in time. Second, build a regular grid spatial data structure to accelerate finding collisions between particles. Figure 27 shows the two phases.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2080.png)

例子：grid-group、multi-grid-group。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Phase 1: Integration
> 
> Phase 2: Collision Detection
> 
> Two phases of a particle simulation, with numbered arrows representing the mapping of parallel threads to particles. Note that after integration and construction of the regular grid data structure, the ordering of particles in memory and mapping to threads changes, requiring a synchronization between phases
> 
> ## Figure 27. Two Phases of a Particle Simulation
> 
> Before Cooperative Groups, implementing such a simulation required multiple kernel launches, because the mapping of threads changes from phase 1 to phase 2. The process of building the regular grid acceleration structure reorders particles in memory, necessitating a new mapping of threads to particles. Such a remapping requires synchronization among threads. The implicit synchronization between back-to-back kernel launches satisfies this requirement, as the following CUDA pseudocode shows.
> 
> ```
> // threads update particles in parallel\nintegrate<<<blooks, threads, 0, s>>>(particles);
> // Note: implicit sync between kernel launches
> // Collide each particle with others in neighborhood
> collide<<<blooks, threads, 0, s>>>(particles);
> ```
> 
> Cooperative Groups provides flexible and scalable thread group types and synchronization primitives enable parallelism remapping in situations like the above example within a single kernel launch. The following CUDA kernel provides a sketch of how the particle system update could be updated in a single kernel. The use of this\_grid() defines a thread group comprising all threads of the kernel launch, which is then synchronized between the two phases.
> 
> ```
> __global__ void particleSim(Particle *p, int N) {
> 
> grid_group g = this_grid();
> 
> // phase 1
> 
> for (i = g.thread_rank(); i < N; i += g.size())
>     integrate(p[i]);
> 
> g.sync() // Sync whole grid
> 
> // phase 2
> 
> for (i = g.thread_rank(); i < N; i += g.size())
>     collide(p[i], p, N);
> </pre>
> ```
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2081.png)

> **[图片提取文字 (image.png)]:**
> This kernel is written in such a way that it is trivial to extend the simulation to multiple GPUs. The Cooperative Groups function this\_multi\_grid() returns a thread group spanning all threads of a kernel launch across multiple GPUs. Calling sync() on this group synchronizes all threads running the kernel on multiple GPUs. Note that in both cases, the thread\_rank() method provides a linear index of the current thread within the thread group, which the kernel uses to iterate over the particles in parallel in case there are more particles than threads.
> 
> ```
> __global__ void particleSim(Particle *p, int N) {
> 
> multi_grid_group g = this_multi_grid();
> 
> // phase 1
> 
> for (i = g.thread_rank(); i < N; i += g.size())
>     integrate(p[i]);
> 
> g.sync() // Sync whole grid
> 
> // phase 2
> 
> for (i = g.thread_rank(); i < N; i += g.size())
>     collide(p[i], p, N);
> }</pre>
> ```
> 
> To use groups that span multiple thread blocks or multiple GPUs, applications must use the cudaLaunchCooperativeKernel() or cudaLaunchCooperativeKernelMultiDevice() API, respectively. Synchronization requires that all thread blocks are simultaneously resident, so the application must also ensure that the resource usage (registers and shared memory) of the thread blocks launched does not exceed the total resources of the GPU(s).
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2082.png)

## Turing（细节优化）

FP32和INT32的比例按照workload中Op比例设置。更高性能的TensorCore。

> **[图片提取文字 (image.png)]:**
> ## GigaThread Engine
> 
> ![](_page_0_Figure_2.jpeg)
> 
> **NVLink – Two x8 Links**
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2083.png)

> **[图片提取文字 (image.png)]:**
> ## CONCURRENT EXECUTION
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Per 100 FP instructions, average 36 INT PIPE instructions (ie iadd, select, fp min/max, compare etc)
> 
> ![](_page_0_Picture_3.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2084.png)

> **[图片提取文字 (image.png)]:**
> ## PASCAL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> TURING TENSOR CORE FP16
> 
> ![](_page_0_Picture_3.jpeg)
> 
> TURING TENSOR CORE INT 8
> 
> ![](_page_0_Picture_5.jpeg)
> 
> TURING TENSOR CORE INT 4
> 
> ![](_page_0_Picture_7.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2085.png)

> **[图片提取文字 (image.png)]:**
> ## SM Warp Scheduler + Dispatch (32 thread/clk) Warp Scheduler + Dispatch (32 thread/clk) Register File (16,384 x 32-bit) Register File (16,384 x 32-bit) **TENSOR TENSOR** INT32 **FP32** INT32 **FP32 CORES CORES** LD/ST **SFU** SFU LD/ST LD/ST LD/ST LD/ST LD/ST LD/ST LD/ST Warp Scheduler + Dispatch (32 thread/clk) Warp Scheduler + Dispatch (32 thread/clk) Register File (16,384 x 32-bit) Register File (16,384 x 32-bit) **TENSOR TENSOR** INT32 **FP32 INT32 FP32 CORES CORES** LD/ST SFU SFU LD/ST LD/ST LD/ST LD/ST LD/ST LD/ST LD/ST 96KB L1 Data Cache / Shared Memory Tex Tex Tex Tex
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2086.png)

Cache优化、独立线程调度下优化重复数据计算。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## TURING 5M MICROARCHITECTURE
> 
> **Evolved for Efficiency** 
> 
> Built on foundation of Volta SM (V100: HPC/Datacenter solution between Pascal and Turing Architectures: see HotChips2017 talk)
> 
> Compared to Pascal, Turing provides:
> 
> - Twice the schedulers
> - Simplified issue logic
> - Large, fast L1 cache unified with TEX \$ and Shared Memory
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2087.png)

> **[图片提取文字 (image.png)]:**
> ## **NEW CACHE & SHARED MEM ARCHITECTURE**
> 
> Evolved for Efficiency
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ![](_page_0_Figure_6.jpeg)
> 
> Compared to Pascal: 2x L1 Bandwidth Lower L1 Hit Latency Up to 2.7x L1 Capacity 2x L2 Capacity
> 
> ![](_page_0_Figure_8.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2088.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## TURING SM MICROARCHITECTURE
> 
> Evolved for Efficiency
> 
> ## Compared to Pascal:
> 
> - Twice the register file capacity
> - Improved SIMT model & branch unit
> - Concurrent FP and INT execution
> - New Uniform registers and datapath
> - **New Tensor Core**
> 
> 16x8x8 FP16 tensor/8 clk 8x8x16 INT8 tensor/4 clk 8x8x32 INT4 tensor/4 clk
> 
> Fast FP16 math
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2089.png)

> **[图片提取文字 (image.png)]:**
> ## TENSOR CORE
> 
> ## Breakthrough Acceleration for Computation of Matrix Multiplies
> 
> | A <sub>0,0</sub> | A <sub>0,1</sub> | A <sub>0,2</sub> | A <sub>0,3</sub> | ` |
> |------------------|------------------|------------------|------------------|---|
> | A <sub>1,0</sub> | A <sub>1,1</sub> | A <sub>1,2</sub> | A <sub>1,3</sub> |   |
> | A <sub>2,0</sub> | A <sub>2,1</sub> | A <sub>2,2</sub> | A <sub>2,3</sub> |   |
> | A <sub>3,0</sub> | A <sub>3,1</sub> | A <sub>3,2</sub> | A <sub>3,3</sub> |   |
> 
> | B <sub>0,0</sub> | B <sub>0,1</sub> | B <sub>0,2</sub> | B <sub>0,3</sub> |
> |------------------|------------------|------------------|------------------|
> | B <sub>1,0</sub> | B <sub>1,1</sub> | B <sub>1,2</sub> | B <sub>1,3</sub> |
> | B <sub>2,0</sub> | B <sub>2,1</sub> | B <sub>2,2</sub> | B <sub>2,3</sub> |
> | B <sub>3,0</sub> | B <sub>3,1</sub> | B <sub>3,2</sub> | B <sub>3,3</sub> |
> 
> ![](_page_0_Picture_4.jpeg)
> 
> |             | C <sub>0,0</sub> | C <sub>0,1</sub> | C <sub>0,2</sub> | C <sub>0,3</sub> |
> |-------------|------------------|------------------|------------------|------------------|
> |             | C <sub>1,0</sub> | C <sub>1,1</sub> | C <sub>1,2</sub> | C <sub>1,3</sub> |
> |             | C <sub>2,0</sub> | C <sub>2,1</sub> | C <sub>2,2</sub> | C <sub>2,3</sub> |
> | $\setminus$ | C <sub>3,0</sub> | C <sub>3,1</sub> | C <sub>3,2</sub> | C <sub>3,3</sub> |
> 
> ## **PASCAL**
> 
> ![](_page_0_Picture_7.jpeg)
> 
> ## **TURING TENSOR CORES**
> 
> ![](_page_0_Picture_9.jpeg)
> 
> 114 TFLOPS FP16
> 
> **228 TOPS INT8** 
> 
> **455 TOPS INT4** 
> 
> \*GTX 2080 Ti
> 
> ![](_page_0_Figure_14.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2090.png)

> **[图片提取文字 (image.png)]:**
> ## UNIFORM DATAPATH & REGISTER FILE
> 
> Goal: Exploit redundant computation & data across multiple threads while preserving our Independent Thread Scheduling model
> 
> Automatically promote ops/data when warp-uniform data is detected
> 
> - Compiler + hardware assist
> - Executed by an independent datapath
> - 'Reverse vectorization'
> 
> Example: Enabling DX12 bindless constants with URF/UDP on Forza MS7 yielded +12.7% performance
> 
> ```
> UIADD3
>                UR13, UR9, 0x300001, URZ
> ULDC.64
>                UR20, [UR6 + 0x18], [UP7]
>                UR6, UR8, UR10, URZ
> UIADD3
>                     UR9. 0x300002, URZ
> UIADD3
> FSETP.NEU.FTZ.AND P1, PT, R15, cx[UR20][0x64], PT
>                UR12, UR13, Oxffffff, URZ, OxcO, !UP7
> ULOP3.LUT
> ```
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2091.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## TENSOR CORE
> 
> Breakthrough Acceleration for Computation of Matrix Multiplies
> 
> Multi-thread collaborative matrix math operation
> 
> Sharing operands across threads saves RF and shared memory BW
> 
> Fine-grained integration inside SM
> 
> - Provides maximum algorithmic flexibility
>   - Different activation functions, Batch norm variants, etc.
> - Leverages huge storage capacity and BW provided by RF and shared mem/L1\$
> 
> 8b & 4b integer support with 32b accumulation for maximum inference performance
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2092.png)

## Ampere（TF32、TC支持更多精度和结构性稀疏、MIG、NVSwitch）

### Tensor Core

![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2093.png)

> **[图片提取文字 (image.png)]:**
> Figure 8 compares V100 and A100 FP16 Tensor Core operations, and also compares V100 FP32, FP64, and INT8 standard operations to respective A100 TF32, FP64, and INT8 Tensor Core operations. Throughputs are aggregate per GPU, with A100 using sparse Tensor Core operations for FP16, TF32, and INT8. Note the upper left diagram shows two V100 FP16 Tensor Cores, since a V100 SM has two Tensor Cores per SM partition, while an A100 SM one.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> ![](_page_0_Figure_4.jpeg)
> 
> A100 Tensor Core operations compared to V100 Tensor Core and standard operations for different data types.
> 
> Figure 8. A100 vs V100 Tensor Core Operations
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2094.png)

> **[图片提取文字 (image.png)]:**
> ## **L1 Instruction Cache**
> 
> | L0 Instruction Cache            |           |           |           |              |           |           |           |        |
> |---------------------------------|-----------|-----------|-----------|--------------|-----------|-----------|-----------|--------|
> | Warp Scheduler (32 thread/clk)  |           |           |           |              |           |           |           |        |
> | Dispatch Unit (32 thread/clk)   |           |           |           |              |           |           |           |        |
> | Register File (16,384 x 32-bit) |           |           |           |              |           |           |           |        |
> | INT32                           | INT32     | FP32      | FP32      | FP64         |           |           |           |        |
> | INT32                           | INT32     | FP32      | FP32      | FP64<br>FP64 |           |           |           |        |
> | INT32                           | INT32     | FP32      | FP32      |              |           |           |           |        |
> | INT32                           | INT32     | FP32      | FP32      | FP64         |           | TE        | ENSO      | R CORE |
> | INT32                           | INT32     | FP32      | FP32      | FP64         |           | 1 1/2     |           |        |
> | INT32                           | INT32     | FP32      | FP32      | FP64         |           |           |           |        |
> | INT32                           | INT32     | FP32      | FP32      | FP64         |           |           |           |        |
> | INT32                           | INT32     | FP32      | FP32      | FP64         |           |           |           |        |
> | LD/<br>ST                       | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST    | LD/<br>ST | LD/<br>ST | LD/<br>ST | SFU    |
> 
> |           | Dispatch Unit (32 thread/clk)   |           |           |           |           |           |           |        |  |
> |-----------|---------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|--------|--|
> |           | Register File (16,384 x 32-bit) |           |           |           |           |           |           |        |  |
> | INT32     | INT32                           | FP32      | FP32      | FP        | 64        |           |           |        |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           |           |        |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           |           |        |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           | ENICO     | D CODE |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           | ENSO      | R CORE |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           |           |        |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           |           |        |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |           |           |        |  |
> | LD/<br>ST | LD/<br>ST                       | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | SFU    |  |
> 
> **L0 Instruction Cache** 
> 
> Warp Scheduler (32 thread/clk)
> 
> **L0 Instruction Cache** 
> 
> Warp Scheduler (32 thread/clk)
> 
> | L0 Instruction Cache            |                                |                  |                  |  |  |  |  |  |
> |---------------------------------|--------------------------------|------------------|------------------|--|--|--|--|--|
> |                                 | Warp Scheduler (32 thread/clk) |                  |                  |  |  |  |  |  |
> | Dispatch Unit (32 thread/clk)   |                                |                  |                  |  |  |  |  |  |
> | Register File (16,384 x 32-bit) |                                |                  |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             | TENSOR CORE      |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | INT32 INT32                     | FP32 FP32                      | FP64             |                  |  |  |  |  |  |
> | LD/<br>ST ST                    | LD/ LD/<br>ST ST               | LD/ LD/<br>ST ST | LD/<br>ST ST SFU |  |  |  |  |  |
> 
> |           | Dispatch Unit (32 thread/clk)   |           |           |           |           |                   |  |  |
> |-----------|---------------------------------|-----------|-----------|-----------|-----------|-------------------|--|--|
> |           | Register File (16,384 x 32-bit) |           |           |           |           |                   |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP        | 64        |                   |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |                   |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP        | 64        |                   |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           | TENSOR CORE       |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           | TENSON CORE       |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP        | 64        |                   |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |                   |  |  |
> | INT32     | INT32                           | FP32      | FP32      | FP64      |           |                   |  |  |
> | LD/<br>ST | LD/<br>ST                       | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST | LD/<br>ST SFU SFU |  |  |
> 
> ## 192KB L1 Data Cache / Shared Memory
> 
> Tex Tex Tex
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2095.png)

TF32

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2096.png)

> **[图片提取文字 (image.png)]:**
> |      | INPU      | T OPERANDS                                   | AC    | CUMULATOR                               | TOPS | X-factor<br>vs. FFMA | SPARSE<br>TOPS | SPARSE<br>X-factor<br>vs. FFMA |
> |------|-----------|----------------------------------------------|-------|-----------------------------------------|------|----------------------|----------------|--------------------------------|
> | V100 | FP32      | ((mm))                                       | FP32  | ((mm))                                  | 15.7 | 1x                   | -              | -                              |
> |      | FP16      | •                                            | FP32  | ((((((((((((((((((((((((((((((((((((((( | 125  | 8x                   | -              | -                              |
> | A100 | FP32      | ((mm))                                       | FP32  | ( <del>(111111)</del> )                 | 19.5 | 1x                   | -              | -                              |
> |      | TF32      |                                              | FP32  |                                         | 156  | 8x                   | 312            | 16x                            |
> |      | FP16      | (COLUMN 111111111111111111111111111111111111 | FP32  | ((((((((((((((((((((((((((((((((((((((( | 312  | 16x                  | 624            | 32x                            |
> |      | BF16      | •                                            | FP32  |                                         | 312  | 16x                  | 624            | 32x                            |
> |      | FP16      | •                                            | FP16  | ((((((((((((((((((((((((((((((((((((((( | 312  | 16x                  | 624            | 32x                            |
> |      | INT8      |                                              | INT32 |                                         | 624  | 32x                  | 1248           | 64x                            |
> |      | INT4      | <b></b>                                      | INT32 |                                         | 1248 | 64x                  | 2496           | 128x                           |
> |      | BINARY    | 0                                            | INT32 |                                         | 4992 | 256x                 | -              | -                              |
> |      | IEEE FP64 |                                              |       |                                         | 19.5 | 1x                   | -              | -                              |
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2097.png)

结构化稀疏TC

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Dense trained weights
> 
> 2:4 sparsity: 2 nonzero out of 4 entries
> 
> Fine-grained
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2098.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> N cycles on Dense MMA
> 
> N/2 cycles on Sparse MMA
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%2099.png)

### MIG

MIG将资源partition成多个GPU Slice，每个GPU Slice可以作为独立的GPU instance（MIG），提供多VM（用户）的资源隔离。

> **[图片提取文字 (image.png)]:**
> ## **CSP Multi-User Node Today**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20100.png)

> **[图片提取文字 (image.png)]:**
> ## **CSP Multi-Instance GPU (MIG)**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20101.png)

1个图形-计算前端Sys Pipe和之前一致，额外6个计算前端。

每个MIG Slice可以独立承载一个应用Ctx或VM Ctx，每个Slice拥有独立的channel资源（Sys Pipe）。

若干MIG slice可以组成不同大小的GPU instance（MIG），MIG可以分时超卖来提供更多的vGPU，vGPU间可迁移VM负载。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20102.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20103.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ## GPU MEMORY PARTITIONING
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ## COMPUTE PARTITIONING
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20104.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> MIG Memory Partitions -
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20105.png)

> **[图片提取文字 (image.png)]:**
> ## MIG Migration
> 
> An important MIG feature to manage, tune, service, and load-balance vGPU (virtual GPU) virtual machine (VM) configurations is the ability to migrate vGPUs between GPU Instances on a single GPU, and more frequently between different GPUs in a cluster. The migration process is conceptually straightforward. State information of the GPU slices of a vGPU within a GPU Instance are saved and then restored onto another GPU Instance with the same number of GPU slices.
> 
> When various GPUs in a cluster are only partially utilized, MIG migration allows moving and packing jobs onto fewer GPUs, reducing fragmentation, and often reducing the number of physical GPUs necessary to support a given number of vGPUs. This can free up certain GPUs to run larger jobs, or have the unused GPUs placed in power-saving mode to reduce data center costs. MIG migration also allows GPUs to be deloaded for servicing, without killing the jobs.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20106.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20107.png)

### 异步指令、Cache、任务抽象、CG level指令

FP32计算数值、INT32计算地址。TC支持更大的数据共享。L2 Cache维持部分数据不刷新。

异步Copy指令允许线程的传输指令和计算指令Overlap，因为VRAM到SMEM的异步Copy过程不占用线程资源。

异步Barrier将Arrive-Wait融合的Primitive拆成Arrive和Wait Primitive，先Arrive（Produce）的线程继续执行（Steal），直到遇到Wait才等待（Consume），设计成**Produce-Steal-Consume pipeline来减少线程阻塞**。

> **[图片提取文字 (image.png)]:**
> ## Simultaneous Execution of FP32 and INT32 Operations
> 
> Similar to Tesla V100 and Turing GPUs, the A100 SM also includes separate FP32 and INT32 cores, allowing simultaneous execution of FP32 and INT32 operations at full throughput, while also increasing instruction issue throughput. Many applications have inner loops that perform pointer arithmetic (integer memory address calculations) combined with floating-point computations that will benefit from simultaneous execution of FP32 and INT32 instructions. Each iteration of a pipelined loop can update addresses (INT32 pointer arithmetic) and load data for the next iteration while simultaneously processing the current iteration in FP32.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20108.png)

> **[图片提取文字 (image.png)]:**
> ## A100 Tensor core: 2x throughput vs. V100, >2x efficiency
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ## V100 TC Instruction (1024 MACs, 8 cycles)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> ## A100 TC Instruction
> 
> (2048 MACs, 8 cycles)
> 
> ![](_page_0_Figure_6.jpeg)
> 
> | 16x16x16 matrix multiply     | FFMA | V100 TC | A100 TC | A100 vs.<br>V100<br>(improvement) | A100 vs.<br>FFMA<br>(improvement) |
> |------------------------------|------|---------|---------|-----------------------------------|-----------------------------------|
> | Thread sharing               | 1    | 8       | 32      | 4x                                | 32x                               |
> | Hardware instructions        | 128  | 16      | 2       | 8x                                | 64x                               |
> | Register reads+writes (warp) | 512  | 80      | 28      | 2.9x                              | 18x                               |
> | Cycles                       | 256  | 32      | 16      | 2x                                | 16x                               |
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20109.png)

> **[图片提取文字 (image.png)]:**
> For example, for DL inferencing workloads, ping-pong buffers can be persistently cached in the L2 for faster data access, while also avoiding writebacks to DRAM. For producer-consumer chains, such as those found in DL training, L2 cache controls can optimize caching across the write-to-read data dependencies. In LSTM networks, recurrent weights that are shared across multiple GEMM operations can be preferentially cached and reused in L2.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> A100 L2 cache residency controls help applications reduce DRAM bandwidth. This example shows different data buffers highlighted with colors to indicate data that has been marked for persistent caching in L2.
> 
> Figure 16. A100 L2 cache residency controls
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20110.png)

> **[图片提取文字 (image.png)]:**
> ## A100 SM Data Movement Efficiency 3x SMEM/L1 bandwidth, 2x in-flight capacity
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20111.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20112.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Single-Stage barriers combine back-to-back arrive & wait
> 
> Asynchronous barriers enable pipelined processing
> 
> An asynchronous barrier allows a thread to indicate that its data is ready and then continue to work on independent operations, postponing the wait so that idle time is reduced. This is a form of asynchronous processing known as *pipelining*, and is commonly used to hide high latency operations such as memory loads (see "async copy", above).
> 
> ## Figure 33. A100 Asynchronous Barriers
> 
> The new Asynchronous Barriers also provide a significant advancement in synchronization granularity compared with barriers on previous architectures, by allowing hardware-accelerated synchronization of any subset of CUDA threads within the block. Previous architectures only accelerate synchronization at a whole-warp or whole-block level. Barriers can be used to overlap asynchronous copies from global memory into shared memory (described in the previous section) by having the copy operation signal ("arrive on") the barrier when it is complete. This allows overlap of the copy with other execution in the SM, hiding the latency of the copy and increasing efficiency.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20113.png)

**task graph**（和grid、stream同功能的任务容器）和延迟breakdown，task graph将GPU的树形架构作类似**dataflow**配置来启动。

> **[图片提取文字 (image.png)]:**
> ## **CUDA Task Graph Basics**
> 
> Many GPU-intensive applications such as deep neural network training and scientific simulations have an iterative structure where the same workflow is executed repeatedly. Using CUDA Streams for such workflows requires that the work be resubmitted to the GPU by the CPU with every iteration, which consumes both time and CPU resources. CUDA Task Graphs were introduced as part of the CUDA 10 release in 2018, and provide a more efficient model for submitting work to the GPU. A task graph consists of a series of operations, such as memory copies and kernel launches, connected by dependencies, and is defined separately from its execution. Task graphs enable a define-once/run-repeatedly execution flow. A predefined task graph allows launch of any number of kernels in one single operation, greatly improving application efficiency and performance.
> 
> Execution of work on the GPU breaks down into three stages: launch, grid initialization, and kernel execution. For GPU kernels with short runtimes in particular, these overheads can be a significant fraction of the overall end-to-end execution time.
> 
> Separating out the definition of a task graph from its execution (where the task graph is executed repeatedly) reduces CPU kernel launch costs significantly. Task graphs also enable the CUDA driver to perform a number of optimizations because the whole workflow is visible to
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20114.png)

> **[图片提取文字 (image.png)]:**
> ## EXECUTION BREAKDOWN FOR SEQUENTIAL $2\mu$ s KERNELS
> 
> Breakdown of time spent during execution
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20115.png)

> **[图片提取文字 (image.png)]:**
> ## CPU Launch Speedup Using Graphs
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20116.png)

> **[图片提取文字 (image.png)]:**
> ## Grid-to-Grid Latency Speedup Using Graphs
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20117.png)

**CG级指令**：硬件加速warpSz的reduction指令

> **[图片提取文字 (image.png)]:**
> ## Cooperative Groups
> 
> Cooperative Groups extends its programming model (originally introduced in CUDA 9) to encapsulate the asynchronous memory copy in a group-wide collective. This utilizes A100's hardware acceleration for the non-blocking memory copy from global to shared memory as well as providing (blocking) software fallbacks in the other direction and on earlier architectures.
> 
> Cooperative Groups uses the threads named in the group to distribute the workload automatically and as efficiently as possible, deducing the correct alignment and data transfer size per thread. While the default operation behaves as a single-stage pipeline, overloads are
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20118.png)

> **[图片提取文字 (image.png)]:**
> provided to extend this to multi-stage pipelines in cooperation with the new memory pipeline object provided by CUDA.
> 
> Once the transfers are kicked off, the data can be read following a call to wait, which signals that the pipeline is empty or that the corresponding stage has completed moving data to shared memory.
> 
> Using A100's powerful new warp reduce instruction, Cooperative Groups expands its set of collectives with a reduce API. This performs a reduction operation on the data provided by each thread named in the group passed in. The hardware can accelerate arithmetic ADD, MIN, or MAX operations and the logical AND, OR, or XOR. Additional types and operations are implemented in software as well as fallbacks on older generation hardware.
> 
> Cooperative launches continue to provide interesting benefits to CUDA developers, and we've been able to reduce grid synchronization overhead by up to 30%, and remove the need for separate compilation when writing cooperative kernels and taking advantage of grid groups.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20119.png)

> **[图片提取文字 (image.png)]:**
> ## WARP-WIDE REDUCTION IN A SINGLE STEP
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ```
> 32
> ```
> 
> ```
> __device__ int reduce(int value) {
> value += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
> ```
> 
> ```
> int total = __reduce_add_sync(0xffffffff, value);
> ```
> 
> ```
> thread_block_tile<32> tile32 =
> ```
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20120.png)

## Hopper和Grace-H（FP8、CGA&DSMEM、TMA、扩展内存池）

### FP8、DPX（加速DP算法）、TC

![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20121.png)

> **[图片提取文字 (image.png)]:**
> A100 FP16 H100 FP16
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20122.png)

> **[图片提取文字 (image.png)]:**
> A100 FP16 H100 FP8
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20123.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20124.png)

> **[图片提取文字 (image.png)]:**
> # SM
> 
> ### **L1 Instruction Cache**
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> #### **L0 Instruction Cache** Warp Scheduler (32 thread/clk) Dispatch Unit (32 thread/clk) Register File (16,384 x 32-bit) INT32 FP32 FP64 FP32 FP32 FP32 FP64 INT32 FP32 INT32 FP32 FP64 FP32 FP32 INT32 FP64 FP32 FP32 FP64 INT32 FP32 FP32 INT32 FP64 FP32 FP32 INT32 FP64 **TENSOR CORE** FP32 FP32 FP64 INT32 4th GENERATION FP32 INT32 FP32 FP64 INT32 FP32 FP32 FP64 INT32 FP32 FP32 FP64 INT32 FP32 FP32 FP64 INT32 FP32 FP32 FP64 INT32 FP32 FP32 FP64 FP32 FP32 INT32 FP64 FP32 FP32 FP64 INT32 LD/ LD/ LD/ LD/ LD/ LD/ LD/ LD/ SFU ST ST ST ST ST ST ST ST
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ### **Tensor Memory Accelerator**
> 
> ## 256 KB L1 Data Cache / Shared Memory
> 
> Tex Tex Tex
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20125.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Allocate 1 bit to either range or precision
> 
> Support for multiple accumulator and output types
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20126.png)

> **[图片提取文字 (image.png)]:**
> ## DYNAMIC PROGRAMING
> 
> Exponential to polynomial time problem solving
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ## A BROAD RANGE OF USE CASES
> 
> from genomics to routing optimization
> 
> ![](_page_0_Figure_5.jpeg)
> 
> REAL-TIME PERFORMANCE 40X speedup
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20127.png)

### 新执行模型block cluster & DSMEM

Volta CG API的自定义范围的线程组同步，是硬件独立线程调度和软件Barrier实现，但架构面向block和warp来提供数据共享和任务调度功能，即架构的执行模型和传输模型是block/CTA和warp。

Hopper架构在**GPC内的每个SM引入DSMEM**，面向**block cluster**提供数据共享和任务调度，是新的执行模型和传输模型（grid、cluster、block、warp）。

GPC按cluster为单位接受负载，cluster通过GPC内的所有DSMEM共享数据。cluster内线程将cluster的所有block在DSMEM的地址段映射到自己的地址空间。

> **[图片提取文字 (image.png)]:**
> ## **Thread Block Clusters**
> 
> The CUDA programming model has long relied on a GPU compute architecture that uses Grids containing multiple Thread Blocks to leverage locality in a program. A Thread Block contains multiple threads that run concurrently on a single SM, where the threads can synchronize with fast barriers and exchange data using the SM's shared memory. However, as GPUs grow beyond 100 SMs, and compute programs become more complex, the Thread Block as the only unit of locality expressed in the programming model is insufficient to maximize execution efficiency.
> 
> H100 introduces a new Thread Block Cluster architecture that exposes control of locality at a granularity larger than a single Thread Block on a single SM. Thread Block Clusters extend the CUDA programming model and add another level to the GPU's physical programming hierarchy to now include Threads, Thread Blocks, Thread Block Clusters, and Grids. A Cluster is a group of Thread Blocks that are guaranteed to be concurrently scheduled onto a group of SMs, where the goal is to enable efficient cooperation of threads across multiple SMs.
> 
> The Clusters in H100 run concurrently across SMs within a GPC. A GPC is a group of SMs in the hardware hierarchy that are always physically close together. Clusters have hardware-accelerated barriers and new memory access collaboration capabilities discussed in the following sections. A dedicated SM-to-SM network for SMs in a GPC provides fast data sharing between threads in a Cluster. In CUDA, Thread Blocks in a Grid can optionally be grouped at kernel launch into Clusters as shown in Figure 14, and cluster capabilities can be leveraged from the CUDA cooperative groups API.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20128.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ![](_page_0_Picture_1.jpeg)
> 
> A Grid is composed of Thread Blocks in the legacy CUDA programming model as in A100, shown in the left half of the above diagram. The Hopper architecture adds an optional Cluster hierarchy, shown in the right half of the diagram.
> 
> Figure 14. Thread Block Clusters and Grids with Clusters
> 
> ## **Distributed Shared Memory**
> 
> With Clusters, it is possible for all the threads to directly access other SM's shared memory with load, store, and atomic operations. This feature is called Distributed Shared Memory (DSMEM) because the shared memory's virtual address space is logically distributed across all the Blocks in the Cluster. DSMEM enables more efficient data exchange between SMs, where data no longer needs to be written to and read from global memory to pass the data. The dedicated SM-to-SM network for Clusters ensures fast, low latency access to remote DSMEM. Compared to using global memory, DSMEM accelerates data exchange between Thread Blocks by about 7x.
> 
> ![](_page_0_Figure_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
> 
> Figure 15. Thread Block to Thread Block data exchange (A100 vs H100 with Clusters)
> 
> At the CUDA level, all the DSMEM segments from all Thread Blocks in the Cluster are mapped into the generic address space of each thread, such that all of DSMEM can be referenced directly with simple pointers. CUDA users can leverage the cooperative\_groups API to construct generic pointers to any Thread Block in the cluster. DSMEM transfers can also be expressed as asynchronous copy operations synchronized with shared memory-based barriers for tracking completion.
> 
> Figure 16 below shows the performance advantage of using Clusters on different algorithms. Clusters improve the performance by allowing the programmer to directly control a larger portion
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20129.png)

### 异步TMA、异步传输Barrier

把之前的grid中同步等待的目标指令（如Copy），设置额外硬件处理，让同步等待的对象脱离原有datapath，异步进行和其他指令overlap，为了完成原有同步，设置对应的屏障类型和接口。

A100中设置特定线程计算细粒度地址、发起细粒度传输和等待所有传输完成，H100的TMA硬件打包细粒度地址的计算过程，不需要特定线程进行计算-传输-等待的循环。

> **[图片提取文字 (image.png)]:**
> ## **Asynchronous Execution**
> 
> Each new generation of NVIDIA GPUs includes numerous architectural enhancements to improve performance, programmability, power efficiency, GPU utilization, and many other factors. Recent NVIDIA GPU generations have included asynchronous execution capabilities to allow more overlap of data movement, computation, and synchronization. The Hopper architecture provides new features that improve asynchronous execution and allow further overlap of memory copies with computation and other independent work, while also minimizing synchronization points.
> 
> A new async memory copy unit called the Tensor Memory Accelerator (TMA) and a new Async Transaction Barrier are described below.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20130.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> | CUDA Programming Model Exposure | A100                 | New for H100                            |
> |---------------------------------|----------------------|-----------------------------------------|
> | Barrier.arrive(),               | Asynchronous Barrier | Asynchronous Transaction Barrier        |
> | Barrier.wait()                  | Waiter spins in SMEM | Waiter sleeps until all threads arrive  |
> | Memcpy_async()                  | Direct copy to SMEM  | Asynchronous mem copy unit (called TMA) |
> 
> Programmatic overlap of data movement, computation, and synchronization. Asynchronous concurrency and minimizing synchronization points are keys to performance.
> 
> Figure 17. Asynchronous Execution Concurrency and Enhancements in Hopper
> 
> ## Tensor Memory Accelerator (TMA)
> 
> features.
> 
> To help feed the powerful new H100 Tensor Cores, data fetch efficiency is improved with a new Tensor Memory Accelerator (TMA) that can transfer large blocks of data and multi-dimensional tensors from global memory to shared memory and vice-versa.
> 
> TMA operations are launched using a copy descriptor which specifies data transfers using tensor dimensions and block coordinates instead of per-element addressing (see Figure 18 below). Large blocks of data (up to the shared memory capacity) can be specified and loaded from global memory into shared memory or stored from shared memory back to global memory. TMA significantly reduces addressing overhead and improves efficiency with support for different tensor layouts (1D-5D tensors), different memory access modes, reductions, and other
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20131.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 18. TMA Address Generation via Copy Descriptor
> 
> The TMA operation is asynchronous and leverages the shared memory-based asynchronous barriers introduced in A100. Additionally, the TMA programming model is single-threaded, where a single thread in a warp is elected to issue an asynchronous TMA operation (<a href="mailto:cuda::memcpy\_async">cuda::memcpy\_async</a>) to copy a tensor, and subsequently multiple threads can wait on a <a href="cuda::barrier">cuda::barrier</a> for completion of the data transfer. To further improve performance, the H100 SM adds hardware to accelerate these asynchronous barrier wait operations.
> 
> A key advantage of TMA is it frees the threads to execute other independent work. On A100, in the left part of Figure 19, asynchronous memory copies were executed using a special LoadGlobalStoreShared instruction, so the threads were responsible for generating all addresses and looping across the whole copy region.
> 
> On Hopper, TMA takes care of everything. A single thread creates a copy descriptor before launching the TMA, and from then on address generation and data movement are handled in hardware. TMA provides a much simpler programming model because it takes over the task of computing stride, offset, and boundary calculations when copying segments of a tensor.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20132.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 19. Asynchronous Memory Copy with TMA on H100 vs LDGSTS on A100
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20133.png)

异步屏障将barrier拆分为Arrive和Wait，让先Arrive的线程可以继续，直到遇到Wait等待，计数线程Arrive计数足够后Wait完成。

异步传输屏障的Wait的完成，既需要Arrive计数足够，也需要传输量满足Wait定义。因为原来负责传输的线程启动TMA后，不需要等待TMA传输完成就能到达Wait，需要数据量要求来强制等待TMA完成传输。

> **[图片提取文字 (image.png)]:**
> ## **Asynchronous Transaction Barrier**
> 
> Asynchronous Barriers were originally introduced in the Ampere GPU architecture. See the left part of Figure 20. Consider an example where a set of threads are producing data that they will all consume after a barrier. Asynchronous barriers split the synchronization process into two steps. First, threads signal "Arrive" when they are done producing their portion of the shared data. This "Arrive" is non-blocking so the threads are free to execute other independent work. Eventually the threads need the data produced by all the other threads. At this point they do a "Wait" which blocks them until every thread has signaled "Arrive".
> 
> The advantage of Asynchronous Barriers is they allow threads that arrive early to execute independent work while waiting. This overlap is the source of extra performance. If there is enough independent work for all threads, the barrier effectively becomes "free" because the Wait instruction can retire immediately, since all threads have already Arrived.
> 
> New for Hopper is the ability for "Waiting" threads to sleep until all other threads arrive. On previous chips, Waiting threads would spin on the barrier object in shared memory.
> 
> While Asynchronous Barriers are still part of the Hopper programming model, Hopper adds a new form of barrier called an Asynchronous Transaction Barrier. The asynchronous transaction barrier is very similar to an Asynchronous Barrier. See the right part of Figure 20. It too is a split barrier, but instead of counting just thread arrivals, it also counts transactions. Hopper includes a new command for writing Shared Memory that passes both the data to be written and a transaction count. The transaction count is essentially a byte count. The asynchronous transaction barrier will block threads at the Wait command until all the producer threads have
> 
> performed an Arrive, and the sum of all the transaction counts reaches an expected value.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20134.png)

> **[图片提取文字 (image.png)]:**
> Asynchronous Transaction Barriers are a powerful new primitive for async mem copies or data exchanges. As mentioned earlier, Clusters can do Thread Block-to-Thread Block communication for a data exchange with implied synchronization, and that Cluster capability is built on top of Asynchronous Transaction barriers.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 20. Asynchronous Barrier in A100 vs Asynchronous Transaction Barrier in H100
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20135.png)

### 安全MIG、Transformer Engine

单GPU（不同MIG）运行不同VM、单GPU单VM、多GPU单VM时，保证隔离性和传输加密。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 23. Secure MIG Example in Multi-Tenant Single GPU Configuration
> 
> Hopper architecture also now permits dedicated image and video decoders for each GPU Instance to deliver secure, high throughput intelligent video analytics (IVA) on shared infrastructure. Each MIG GPU Instance can receive at least one NVDEC and NVJPG unit.
> 
> In addition, H100 MIG Instances now include their own sets of performance monitors that work with NVIDIA developer tools. With Hopper's concurrent profiling, administrators can monitor right-sized GPU acceleration and optimally allocate resources among users seamlessly.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20136.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 28. Confidential Computing for Different Use Cases
> 
> The Confidential Computing capability of Hopper architecture further amplifies and accelerates security for collaborative multi-party computing use-cases like Federated Learning. Federated Learning enables multiple organizations to work together to train or evaluate AI models without having to share each group's proprietary datasets. Confidential Federated Learning with H100 ensures that data and AI models are protected from unauthorized access by external or internal threats, at each participating site, and each site can understand and attest the software running at their peers. This increases confidence in secure collaboration and drives advancement of medical research, expedites drug development, mitigates insurance and financial fraud, and a host of other applications - while maintaining security, privacy, and regulatory compliance.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20137.png)

**TF Engine是对Transformer结构的软硬件协同加速方案**，统计每层输出，每层使用不同的精度计算，混合精度推理。

> **[图片提取文字 (image.png)]:**
> ## **Transformer Engine**
> 
> Transformer models are the backbone of language models used widely today from BERT to GPT-3 and require enormous compute resources. Initially developed for natural language processing (NLP) Transformers are increasingly applied across diverse fields such as computer vision, drug discovery, and more. Their size continues to increase exponentially, now reaching trillions of parameters and causing their training times to stretch into months, which is impractical for business needs due to the large compute requirements. For example, Megatron Turing NLG (MT-NLG) requires 2048 NVIDIA A100 GPUs running for eight weeks to train. Overall, transformer models have been growing much faster than most other AI models at the
> 
> rate of 275x every two years for the past five years (see Figure 24).
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20138.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 24. Transformers Model Sizes Increasing Exponentially with Different Use Cases
> 
> H100 includes a new **Transformer Engine** that is a custom Hopper Tensor Core technology to dramatically accelerate the AI calculations for Transformers.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20139.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 25. Transformer Engine Conceptual Operation.
> 
> The goal of mixed precision is to intelligently manage the precision to maintain accuracy, while still gaining the performance of smaller, faster numerical formats. At each layer of a Transformer model, the Transformer Engine analyzes the statistics of the output values produced by the Tensor Core. With knowledge about which type of neural network layer comes next and what precision it requires, the Transformer Engine also decides which target format to convert the tensor to before storing it to memory. FP8 has a more limited range than other numerical formats. To optimally use the available range, the Transformer Engine also dynamically scales tensor data into the representable range using scaling factors computed from the tensor statistics. Therefore, every layer operates with exactly the range it requires and is accelerated in an optimal manner.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20140.png)

### Grace CPU扩展统一内存池

Hopper GPU使用96GB HBM3或141GB HBM3e，Grace CPU管理扩展的480GB DDR5，CPU和GPU之间通过NVLink C2C传输。

> **[图片提取文字 (image.png)]:**
> ## Inside NVIDIA's First GPU-CPU Superchip
> 
> ![](_page_0_Picture_1.jpeg)
> 
> The NVIDIA® GH200 Grace Hopper architecture brings together the groundbreaking performance of the NVIDIA Hopper GPU with the versatility of the NVIDIA Grace™ CPU, connected with a high bandwidth and memory coherent NVIDIA NVLink Chip-2-Chip (C2C)® interconnect in a single Superchip, and support for the new NVIDIA NVLink Switch System.
> 
> NVIDIA NVLink-C2C is NVIDIA's memory coherent, high-bandwidth, and low-latency interconnect for superchips. It is the heart of the Grace Hopper Superchip and delivers up to 900GB/s total bandwidth. This is 7x higher bandwidth than x16 PCIe Gen5 lanes commonly used in accelerated systems.
> 
> NVLink-C2C memory coherency increases developer productivity, performance, and the amount of GPU-accessible memory. CPU and GPU threads can now concurrently and transparently access both CPU and GPU resident memory, allowing developers to focus on algorithms instead of explicit memory management. Memory coherency allows developers to only transfer the data they need, and not migrate entire pages to and from the GPU. It also enables lightweight synchronization primitives across GPU and CPU threads by enabling native atomics from both the CPU and GPU. NVLink-C2C with Address Translation Services (ATS) leverages NVIDIA Hopper DMA engines for accelerating bulk transfers of pageable memory across host and device.
> 
> NVLink-C2C enables applications to oversubscribe the GPU's memory and directly utilize NVIDIA Grace CPU's memory at high bandwidth. With up to 480GB of LPDDR5X CPU memory per Grace Hopper Superchip, the GPU has direct high-bandwidth access to 6x more memory than available with PCIe. Combined with NVIDIA NVLink Switch System, all GPU threads running on up to 256 NVLink-connected GPUs on DGX GH200 can now access up to 144TB of memory at high bandwidth. Fourth generation NVLink allows
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20141.png)

> **[图片提取文字 (image.png)]:**
> accessing peer memory using direct loads, stores, and atomic operations, enabling accelerated applications to solve larger problems more easily than ever.
> 
> NVIDIA GH200 Grace Hopper Superchip with HBM3 uses 96GB of HBM3 memory, delivering 4TB/s of memory bandwidth. The next generation NVIDIA GH200 Grace Hopper Superchip with HBM3e is the world's first processor to utilize HBM3e memory technology and has 141GB of HBM3e delivering over 4.8TB/s, 1.5X more bandwidth than an H100 80GB SXM. The HBM in NVIDIA Grace Hopper is combined with the CPU memory over NVLink-C2C to provide up to 621GB of fast-access memory to the GPU to deliver the memory capacity and bandwidth required to handle the world's most complex accelerated computing and generative AI workloads.
> 
> The NVIDIA GH200 Grace Hopper Superchip is the first true heterogeneous accelerated platform for <a href="https://doi.org/10.2007/nj.com/high-performance.computing">https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing">https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing">https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a> (HPC) and <a href="https://doi.org/10.2007/nj.com/high-performance.computing</a>
> 
> This whitepaper highlights NVIDIA Grace Hopper's key features, its programming model, and the performance improvements they deliver to the most demanding HPC and AI applications.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20142.png)

> **[图片提取文字 (image.png)]:**
> Figure 1 shows the logical overview of the NVIDIA GH200 Grace Hopper Superchip and Table 1 lists its key features.
> 
> ## NVIDIA GH200 Grace Hopper Superchip
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20143.png)

InfiniBand网络传输组成服务器集群，水平扩展Scale-Out。高算力、高容量存储、高带宽的通信和传输适合有硬性的超大算力和存储需求的负载（strong-scaling）。

多个Grace-Hopper通过NVLink互联组成服务器，GPU是核心计算节点，Grace管理巨大内存池。

> **[图片提取文字 (image.png)]:**
> The configuration in Figure 3 simplifies cluster management. It is designed for workloads that can leverage the strong compute capabilities of NVIDIA Grace Hopper and are also not bottlenecked by the network communication overhead of InfiniBand, which is one of the fastest network interconnects available, but is still a traditional RDMA-accelerated network.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 3. NVIDIA MGX Grace Hopper Superchip system with InfiniBand networking for scale-out ML and HPC workloads
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20144.png)

> **[图片提取文字 (image.png)]:**
> using direct loads, stores, and atomic operations, makes this system configuration ideal for strong-scaling machine learning and HPC workloads, and training giant AI models.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 4. NVIDIA DGX GH200 with NVLink Switch System for strongscaling giant ML workloads
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20145.png)

> **[图片提取文字 (image.png)]:**
> ## Accelerating Applications with Extended GPU Memory
> 
> The NVIDIA GH200 is designed to accelerate applications with exceptionally large memory footprints, larger than the capacity of the HBM3 / HBM3e and LPDDR5X memory of a single superchip (see the NVIDIA GH200 Accelerated Applications section below).
> 
> The Extended GPU Memory (EGM) feature over the high-bandwidth NVLink-C2C enables GPUs to access all the system memory efficiently. EGM provides up to 144TBs system memory in a multi-node NVSwitch-connected system. With EGM, physical memory in the system can be allocated to be accessible from any GPU thread. All GPUs can access EGM at the minimum of GPU-GPU NVLink or NVLink-C2C speed.
> 
> Memory accesses within a Grace Hopper Superchip configuration go through the local high-bandwidth NVLink-C2C at 900GB/s total. Remote memory accesses are performed via GPU NVLink, and depending on the memory being accessed, also NVLink-C2C as shown in Figure 5. With EGM, GPU threads can now access all memory resources available over the NVSwitch fabric, both LPDDR5X and HBM3 or HBM3e, at 450GB/s.
> 
> ![](_page_0_Figure_4.jpeg)
> 
> Figure 5. Memory Accesses across NVLink-connected Grace Hopper Superchips
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20146.png)

硬件加速处理内存一致性。

> **[图片提取文字 (image.png)]:**
> ## Hardware Accelerated Memory Coherency
> 
> In PCIe-connected x86+Hopper systems, the CPU and the GPU have independent perprocess page tables, and system-allocated memory is not directly accessible from the GPU (Figure 7). When a program allocates memory with the system allocator on the host the page entry of the allocation is not available in the GPU's page table and accessing it from GPU threads fails<sup>1</sup>.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 7. NVIDIA Hopper System with Disjoint Page Tables
> 
> In NVIDIA Grace Hopper Superchip-based systems, Address Translation Service (ATS) enables the CPU and GPU to share a single per-process page table, enabling all CPU and GPU threads to access all system-allocated memory (Figure 8), which can reside on physical CPU or GPU memory. The CPU heap, CPU thread stack, global variables,
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20147.png)

> **[图片提取文字 (image.png)]:**
> memory-mapped files, and inter-process memory are accessible to all CPU and GPU threads.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 8. ATS in an NVIDIA Grace Hopper Superchip System
> 
> NVIDIA NVLink-C2C hardware-coherency enables the Grace CPU to cache GPU memory at cache-line granularity and for the GPU and CPU to access each other's memory without page-migrations. NVLink-C2C also accelerates all atomic operations supported by the CPU and GPU on system-allocated memory. Scoped atomic operations are fully supported and enable fine-grained and scalable synchronization across all threads in the system.
> 
> The runtime backs system-allocated memory with physical memory on first touch, either on LPDDR5X or HBM3 / HBM3e, depending on whether a CPU or a GPU thread accesses it first. From an OS perspective, the Grace CPU and Hopper GPU are just two separate NUMA nodes.
> 
> System-allocated memory is migratable, i.e., the runtime can change its physical memory backing to improve application performance (Figure 9) or deal with memory pressure. Hardware access counters allow delayed migrations over a page-fault-based method so that only hot pages are migrated.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20148.png)

Access Counter缓存频繁的物理页。NVLink Switch中的内存访问。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 9. Access-Frequency-Based Automatic Memory Migration
> 
> Network and storage devices connected via non-coherent PCIe lanes have several methods for performing Direct Memory Access (DMA) and Remote DMA (RDMA) on system-allocated memory. On-Demand Paging (ODP) is an RDMA extension supported by NVIDIA InfiniBand Networking products like BlueField-3 and ConnectX-7 that allows devices to track pages being migrated. It enables communication and storage libraries such as MPI, HPC-X, NCCL, NVSHMEM, UCX, MAGNUM IO, and GPUDirect Storage to perform efficient zero-copy I/O operations on system-allocated memory without having to stage transfers through separate buffers.
> 
> CUDA-specific memory APIs provide users with guarantees about where the memory resides, which threads can access it, whether it is migratable, and many other features that enable users to extract all the performance the hardware has to offer. Applications can hint the system about their memory access patterns, for example, using <u>CUDA</u> and/or <u>NUMA\_APIs</u>, to enable the users to perform application-specific optimizations. NUMA memory hints enable applications to inform the runtime about their memory access patterns.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20149.png)

> **[图片提取文字 (image.png)]:**
> ## Memory Access in NVLink Switch System
> 
> On Grace Hopper Superchips connected with NVLink Switch Systems, GPU threads can address peer HBM3 / HBM3e and LPDDR5X memory from other Grace Hopper Superchips in the NVLink network via an NVLink page table (Figure 10). CUDA APIs allow applications to map memory from remote nodes into the current process and then perform load, stores, atomics, and as well as bulk memory transfers to directly access the memory.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20150.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 10. GPU threads address peer memory from other superchips in the NVLink Switch network
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20151.png)

## Blackwell（FP4、FP32/INT32、NR、TMEM）和Grace-B（Attn、RLLM）

### TC（FP4、TMEM）

> **[图片提取文字 (image.png)]:**
> ## **Blackwell GB202 GPU**
> 
> The **full GB202 GPU** includes 12 Graphics Processing Clusters (GPCs), 96 Texture Processing Clusters (TPCs), 192 Streaming Multiprocessors (SMs), and a 512-bit memory interface with sixteen 32-bit memory controllers.
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Figure 3. GB202 GPU block diagram (full chip).
> 
> **Note:** The GB202 GPU also includes 384 FP64 Cores (two per SM) which are not depicted in the above diagram. The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations. The small number of FP64 Cores are included to ensure any programs with FP64 code operate correctly. Similarly, a very minimal number of FP64 Tensor Cores are included for program correctness.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20152.png)

> **[图片提取文字 (image.png)]:**
> The full GB202 GPU includes:
> 
> - 24576 CUDA Cores
> - 192 RT Cores
> - 768 Tensor Cores
> - 768 Texture Units
> 
> ![](_page_0_Figure_5.jpeg)
> 
> Figure 4. The Blackwell GPC with Raster Engine, 8 TPCs, 16 SMs, and 16 ROPs.
> 
> The GPC is the dominant high-level hardware block within all GB20x Blackwell family GPUs, with all of the key graphics processing units residing within a GPC. Each GPC includes a dedicated Raster Engine, two Raster Operations (ROPs) partitions, with each partition containing eight individual ROP units, and eight TPCs. Each TPC includes one PolyMorph Engine and two SMs.
> 
> The full GB202 GPU includes 128 MB of L2 cache, while the RTX 5090 specifically includes 96 MB of L2. All applications benefit from having such a large pool of fast cache memory available, and complex operations such as ray tracing (particularly path tracing) will yield great benefit.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20153.png)

FP4的TC。INT32和FP32的统一CUDA Core。

> **[图片提取文字 (image.png)]:**
> INT operation update added in v1.1 of this whitepaper >> Note that the number of possible integer operations in Blackwell GB20x GPUs are doubled for many integer instructions compared to Ada, by fully unifying the INT32 cores with the FP32 cores, as depicted in Figure 6 below. However, the unified cores can only operate as either FP32 or INT32 cores in any given clock cycle. While many common INT operations can run at up to 2x throughput, not all INT operations can attain 2x speedups. For more details, please refer to the NVIDIA CUDA Programming Guide.
> 
> Figure 6 below shows how the SM architecture evolved between Ada and Blackwell.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Ada SM was designed & optimized for standard shaders. Blackwell SM was designed & optimized for neural shaders.
> 
> Figure 6. Ada SM vs Blackwell
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20154.png)

> **[图片提取文字 (image.png)]:**
> ## **FP4 Support**
> 
> Generative AI models have improved in capabilities since the first ones released in 2022. But the improvements have often come with an increase in parameters and size. As models grow in both compute and memory requirements, it can be difficult to run such models even on the latest hardware.
> 
> The GeForce RTX 50 Series includes support for the FP4 data format in its new Tensor Cores to help address this issue. FP4 provides a lower quantization method, similar to file compression, which decreases model sizes. Compared with FP16 precision — the default method that most models publish with — FP4 requires less than half of the memory, and 50 Series GPUs provide over 2x performance compared to the previous generation. FP4 allows virtually no loss in quality with advanced quantization methods offered by the NVIDIA TensorRT Model Optimizer.
> 
> For example, Black Forest Labs' FLUX.dev model at FP16 requires over 23GB of VRAM, meaning it can only be supported by the GeForce RTX 4090, RTX 5090, and our professional GPUs. With FP4, FLUX.dev requires less than 10GB, so it can run locally on more GeForce RTX GPUs.
> 
> With a GeForce RTX 4090 with FP16, the FLUX.dev model can generate images in 15 seconds with 30 steps. With a GeForce RTX 5090 with FP4, images can be generated in just over five seconds.
> 
> ![](_page_0_Figure_5.jpeg)
> 
> Figure 8. Blackwell 5th Generation Tensor Cores with FP4, double throughput of FP8
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20155.png)

> **[图片提取文字 (image.png)]:**
> ## SM
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ## 128 KB L1 Data Cache / Shared Memory
> 
> Tex Tex Tex Tex
> 
> ## 4TH GENERATION RT CORE
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> Box Intersection Engine
> 
> Triangle Cluster Intersection Engine
> 
> Linear Swept Spheres
> 
> ![](_page_0_Picture_14.jpeg)
> 
> ![](_page_0_Picture_15.jpeg)
> 
> ![](_page_0_Picture_16.jpeg)
> 
> Triangle Cluster Compression Engine
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20156.png)

> **[图片提取文字 (image.png)]:**
> ## NVIDIA Blackwell Ultra GPU
> 
> ![](_page_0_Figure_1.jpeg)
> 
> NVLink-C2C 900GB/s Coherent CPU-GPU Interface
> 
> Confidential Computing TEE-I/O Capable
> 
> NVLink v5
> 
> 288GB HBM3E Memory (12 Stacks, Up to 8 TB/s)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20157.png)

TMEM：Tensor Core专属的Memory，而不需要占用通用寄存器资源。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20158.png)

> **[图片提取文字 (Screenshot from 2026-03-23 21-57-38.png)]:**
> Tensor Memory (TMEM) Innovation
> 
> Tensor Memory is arguably the most transformative addition to Blackwell's SM architecture. TMEM is organized as a 2D structure with 128 rows (lanes) and 512 columns of 4-byte cells, totaling 256 KB per SM — matching the size of the register file itself.
> 
> ## Why TMEM Matters:
> 
> In previous architectures including Hopper, tensor operations suffered from extreme register pressure. Matrix multiply-accumulate (MMA) operations required loading matrix fragments from shared memory into thread registers, executing the operation, and storing results back. This created several bottlenecks:
> 
> - 1. Register File Contention: Large matrix operations consumed massive amounts of register space, limiting occupancy and the number of in-flight operations
> - 2. Memory Traffic: Constant data movement between shared memory and registers created bandwidth bottlenecks
> - 3. Synchronization Overhead: Multiple threads had to synchronize at every step of the computation
![Screenshot from 2026-03-23 21-57-38.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/Screenshot_from_2026-03-23_21-57-38.png)

> **[图片提取文字 (Screenshot from 2026-03-23 21-59-05.png)]:**
> TMEM solves these issues by providing a dedicated staging area for tensor operations that sits closer to the Tensor Cores than traditional shared memory. Key advantages include:
> 
> - Direct Tensor Core Access: TMEM is tightly coupled with the Tensor Cores, reducing latency and eliminating register file pressure
> - Higher Throughput: Dedicated pathways allow sustained tensor throughput
> - without competing for general-purpose resources
> - Dual-Thread-Block MMA: Paired SMs can cooperate on single MMA operations, sharing operands through TMEM and reducing redundant memory traffic
![Screenshot from 2026-03-23 21-59-05.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/Screenshot_from_2026-03-23_21-59-05.png)

### AMP设置RV Core作为GPU前端

由AMP负责调度GPU Ctx，而不需要系统CPU参与，之前channel preemption是CPU控制或TO，将**CUDA Ctx放在GPU侧**，减少CPU的参与。

AMP提供类似SCG的功能，动态改变计算和图形的策略。

> **[图片提取文字 (image.png)]:**
> ## Al Management Processor (AMP)
> 
> The AI Management Processor (AMP) is a fully programmable context scheduler on the GPU designed to offload scheduling of GPU contexts from the system CPU. AMP enhances the scheduling of GPU contexts in Windows to more efficiently manage different workloads running on the GPU. A GPU context encapsulates all the state information the GPU needs to execute one or more tasks. Multiple contexts can be used for better task isolation when running multiple tasks, and ensuring that multiple applications can share the GPU simultaneously without conflicts. An example could be coordinating and scheduling asynchronous AI model workloads like NVIDIA Avatar Cloud Engine (ACE) with its speech, translation, vision, animation, and behavior models, and G-Assist, running simultaneously with other graphics workloads on the GPU.
> 
> The AI Management Processor is implemented using a dedicated RISC-V processor located at the front of the GPU pipeline, and it provides faster scheduling of GPU contexts with lower latency than prior CPU-driven methods. The Blackwell AMP scheduling architecture matches the Microsoft architectural model that describes a configurable scheduling core on the GPU through Windows Hardware-Accelerated GPU Scheduling (HAGS), introduced in Windows 10 (May 2020 Update). HAGS allows the GPU to handle its own memory management more efficiently, reducing latency and potentially improving performance in games and other graphics-intensive applications.
> 
> The role of AMP is to take over the responsibility of the CPU's scheduling of GPU tasks, reducing dependency on the system CPU, which is often a bottleneck for game performance. In fact, allowing the GPU to manage its own task queue can lead to lower latency because of less backand-forth communication between the GPU and CPU. This allows smoother frame rates in games, and better multitasking in Windows because the CPU is less burdened.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20159.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 17. Al Management Processor (AMP) Schedules Al / Graphics Workloads
> 
> Essentially, AMP is used to coordinate, schedule fairly, and ensure a smoother gaming experience without performance drops. With LLMs, it does this by reducing the time to first response, and with games, it prioritizes work with the game engine to prevent stuttering. By delivering work at more predictable times, AMP can significantly improve quality of service depending on workloads.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20160.png)

### Neural Render

> **[图片提取文字 (image.png)]:**
> ## **Neural Shaders**
> 
> Blackwell was designed to jumpstart the future, where neural shaders become the predominant form of shader technology for developing games. Many architectural improvements to Blackwell were made specifically to increase the performance and efficiency of neural shaders and this section describes those optimizations.
> 
> A shader is a program that runs on the GPU to control how graphics are rendered, varying in complexity depending on the visual effects and processing required. Newer shading techniques have added new levels of realism. In its most basic form, shaders calculate the levels of light, darkness, and color used when rendering a scene in 3D space in a game in a process known as shading. They run on the GPU and as part of the rendering pipeline.
> 
> Graphics were first processed on the GPU using non-programmable shaders, also known as fixed-function pipeline, where operations in the graphics pipeline were predefined and configurable, but not programmable. This was because they were limited by the hardware design of the GPU which was specifically built to execute a predefined set of operations.
> 
> GeForce 3 introduced the first bit of programmable shading with vertex shaders. Soon after, the high-level shading language HLSL allowed pixel shading so everything on the screen could be customized. DX10 introduced Geometry shaders. DX11 introduced compute shaders and then an update to DX12 delivered DirectX ray tracing using an acceleration structure of BVH (Bounding Volume Hierarchy) that allowed any ray to intersect with the scene geometry and then be able to spawn a cascade of different shading operations.
> 
> With the Blackwell launch we introduce the era of developer-created neural shaders, some of which will also run on prior generation GPUs.
> 
> Neural Shaders are the next evolutionary step in
> 
> The evolution of shaders in GPUs has been marked by significant advancements in graphics programming and rendering capabilities.
> 
> Here's a brief overview of key milestones:
> 
> - 1. Fixed-Function Pipeline (*Pre-2000s*)
>   Graphics processed using a fixed-function pipeline where operations are predefined and configurable, but not programmable and with limited control over rendering of simple effects like lighting and texturing.
> - 2. Vertex Shaders (DirectX 8.0 / OpenGL 1.4, Early 2000s)
> 
>   Drogrammable vertex shaders gave developers
> 
> Programmable vertex shaders gave developers access to vertex data including transformations and lighting calculations, enabling more complex effects.
> 
> - 3. Fragment Shaders (Pixel Shaders) (DirectX 9.0 / OpenGL 2.0, Early 2000s)
>   Enabled developers to write custom code for operations at the pixel level, allowing for dynamic lighting and texturing, expanding rendering flexibility with Shader Model 2.0.
> - 4. Unified Shader Architecture (DirectX 10.0 / OpenGL 3.3, 2006)
> 
> The unification of geometry, vertex, and fragment shaders, allowing for better utilization of resources and great efficiency. Introduced Shader Model 4.0 supporting more advanced techniques and performance optimizations.
> 
> 5. Geometry Shaders (DirectX 10 / OpenGL 3.2, 2006)
> 
> Geometry shaders expanded to allow for the creation and manipulation of primitives like triangles in the shader pipeline. New effects include dynamic tessellation and particle systems.
> 
> 6. Tessellation and Compute Shaders (DirectX 11 / OpenGL 4.0, 2009)
> 
> Redefined geometry with higher surface detail and smoother curves in 3D models. Shader Model 5.0 added more features for real-time
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20161.png)

> **[图片提取文字 (image.png)]:**
> programmable shading. Rather than writing complex shader code to describe these functions, developers train AI models to approximate the result that the shader code would have computed. Neural shaders are set to become the predominant form of shaders in games, and in the future, all gaming will use AI technology for rendering.
> 
> Up until this point, NVIDIA has been using neural shading for DLSS, using CUDA to harness the Tensor Cores. With the new Cooperative Vectors API for DX12 and Vulkan, Tensor Cores can be accessed through any type of shader, including pixel and ray tracing, in a graphics application allowing for a host of neural technologies. NVIDIA has worked with Microsoft to create the new Cooperative Vectors API. When combined with differentiable shading language features in Slang, Cooperative Vectors unlock the ability for game developers to use neural techniques in their games including neural texture compression, that provides up to seven-to-one VRAM compression over block compressed formats, and other techniques such as RTX Neural Materials, Neural Radiance Cache, RTX Skin, and RTX Neural Faces.
> 
> - rendering techniques. Compute Shaders added parallel processing and complex simulations.
> - 7. Primitive and Mesh Shaders (DirectX 12 Ultimate / Vulkan extension, 2018–2020)
> 
>   Expanded the capabilities and performance of the geometry pipeline by incorporating the features of vertex and geometry shaders into a single shader stage. Mesh shaders allowed the GPU to handle more complex algorithms by offloading more work from the CPU to the GPU.
> - 8. RTX (NVIDIA Turing Architecture / DirectX Raytracing, 2018)
>   Added real-time ray tracing capabilities (RTX) directly to the SM in the GPU, enabling realistic lighting, shadows, and reflections. Introduced dedicated RT cores in hardware that are optimized for ray tracing by accelerating tree traversal and geometry intersection.
> - 9. Blackwell Neural Shaders (Unified AI and Traditional Shaders) (NVIDIA Blackwell Architecture, 2025)
>   Al is embedded into parts of the traditional rendering pipeline, paving the path towards full neural shading. Enhanced Tensor Cores that are now accessible to graphics shaders combined with scheduling optimizations in SER 2.0 (Shader Execution Reordering) so that AI graphics with neural filtering features and AI models including generative AI can be run concurrently in next-generation games.
> 
> ![](_page_0_Figure_6.jpeg)
> 
> Figure 28. Neural Acceleration in Graphics
> 
> Neural shaders allow us to train neural networks to learn efficient approximations of complex algorithms that calculate how light interacts with surfaces, efficiently decompress textures that
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20162.png)

Grace-Blackwell

> **[图片提取文字 (image.png)]:**
> ## NVIDIA Blackwell and Blackwell Ultra Overview
> 
> The NVIDIA Blackwell and Blackwell Ultra products are designed to address the needs of ever-increasing AI complexity, including larger model sizes and AI reasoning, with a long list of new innovations.
> 
> With NVIDIA Blackwell and Blackwell Ultra products, every enterprise can use and deploy state-of-the-art LLMs with affordable economics, optimizing their business with the benefits of reasoning AI. At the same time, NVIDIA Blackwell and Blackwell Ultra products
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20163.png)

> **[图片提取文字 (image.png)]:**
> enable the next era of AI models, supporting high throughput with real-time performance, something unattainable without Blackwell's architectural innovations.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Figure 1. NVIDIA Grace Blackwell Ultra Superchip with ConnectX-8 SuperNICs
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20164.png)

> **[图片提取文字 (image.png)]:**
> ## **Second-Generation Transformer Engine**
> 
> Blackwell introduces the new second-generation Transformer Engine. The second-generation Transformer Engine uses custom Blackwell Tensor Core technology combined with <a href="NVIDIA Dynamo">NVIDIA Dynamo</a>, <a href="TensorRT-LLM">TensorRT-LLM</a> and <a href="Memory Nemo Framework">Nemo Framework</a> innovations to accelerate inference and training for LLMs, Al reasoning, and Mixture-of-Experts (MoE) models.
> 
> The Blackwell Transformer Engine utilizes advanced dynamic range management algorithms and fine-grain scaling techniques, called micro-tensor scaling, to optimize inference performance, accuracy, and enable FP4 AI. This doubles the performance of Blackwell's FP4 Tensor Core, doubles the parameter bandwidth to the HBM memory, and doubles the size of models supported per GPU.
> 
> Innovations in Dynamo and TensorRT-LLM, including quantization to 4-bit precision, custom kernels with expert parallelism mapping, and disaggregation are democratizing today's MoE models for real-time inference, using less hardware and less energy, with less cost.
> 
> For training, the second-generation Transformer Engine works with Nemo Framework and Megatron-Core innovations in new expert parallelism techniques that combine with other parallelism techniques and fifth-generation NVLink for unprecedented model performance. Lower precision formats open possibilities for further acceleration of large-scale training.
> 
> With the Blackwell second-generation Transformer Engine, enterprises can use and deploy state-of-the-art AI reasoning models with affordable economics, optimizing their business with the benefits of generative AI. NVIDIA Blackwell makes the next era of AI reasoning models possible—supporting both training and real-time inference.
> 
> ## Attention Layer Acceleration
> 
> The Blackwell Ultra GPU provides a 2X speedup over Blackwell GPUs for attention layer compute with new instructions to improve the performance of long input sequences. Doubling the attention operations using the Blackwell Ultra GPU architecture enhances Al performance by reducing latency and enabling faster and more intelligent decision-making for Al reasoning models. This acceleration also helps lower compute costs by reducing processing time, leading to energy and infrastructure savings. Enterprises can scale more efficiently, handling larger workloads with the same resources, ultimately driving greater efficiency, cost savings, and a competitive advantage in Al-driven business operations.
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20165.png)

### Chip & Cheese（模块在芯片的布局）

> **[图片提取文字 (image.png)]:**
> ## **GPU Die**
> 
> ## **Streaming Multiprocessor**
> 
> ![](_page_0_Figure_2.jpeg)
> 
> ## **Memory Hierarchy**
> 
> | Process Node | Transistors    | Die Size      | TDP   | FP32        | TFLOPS      | Bandwidth |
> |--------------|----------------|---------------|-------|-------------|-------------|-----------|
> | НВМ3е        | НВМ3е          | HBMB          | 1X5   | TFLOPS      | TFLOPS      | 19150     |
> | L2 Cache     | 128MB HBM3e    | 64MB L2 Cache | 25.80 | TFLOPS      | TFLOPS      | 14750     |
> | L2 Cache     | 66MB           | L2 Cache      | 31    | TFLOPS      | 195 TFOPS   | 4077      |
> | L1 Cache     | 128KB L1 Cache | L1 Cache      | 2120  | FP32 TFLOPS | 1998 TFLOPS | 1090,450  |
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20166.png)

> **[图片提取文字 (image.png)]:**
> #### **GB202**
> 
> ### **Streaming Multiprocessor**
> 
> ![](_page_0_Figure_2.jpeg)
> 
> #### **Tensor Memory**
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> # **Memory Hierarchy**
> 
> # HBM3e 16GB Capacity 1.15TB/s Bandwidth 256b Interface 8192b Bus
> 
> ## **Arch Comp**
> 
> |      | GB100 | GB202  |
> |------|-------|--------|
> | FP64 | 128   | 1701   |
> | FP32 | 255   | 952    |
> | FP16 | 646   | INT8   |
> | INT8 | 8X    | Tensor |
> | FP8  | 1220  | PP8    |
> | BF16 | 1437  | BF16   |
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20167.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20168.png)

> **[图片提取文字 (image.png)]:**
> # Chips &nd Cheese
> 
> ## Blackwell vs Architectura)
> 
> #### Dual-die
> 
> ![](_page_0_Figure_3.jpeg)
> 
> | Spec                     | Hopper  | Ampere | Turing |
> |--------------------------|---------|--------|--------|
> | Transistor               | 177 m/n | 15 min | 25 m/n |
> | Process Node             | 9.4%    | 3.4%   | 9.0%   |
> | Die size                 | 8.5%    | 12.2%  | 12.2%  |
> | Peak FP32<br>Performance | 5.76    | 6.16   | 5.10   |
> | Memory                   | 4.6%    | 1.6%   | 10.0%  |
> | Bandwidth                | 2.6%    | 3.9%   | 21.2%  |
> | Powe FEH                 | 11.7%   | 10.970 | 19.7%  |
> 
> **NVLink** 
> 
> #### Interposer
> 
> ![](_page_0_Figure_6.jpeg)
> 
> ![](_page_0_Figure_7.jpeg)
> 
> Streaming Multiprocesor
> 
> RT Cores
> 
> **CUDA Cores** 
> 
> Shared Memory (SM)
> 
> **CUDA Cores** 
> 
> Tores
> 
> ## Contitecture Compartenl(Bbit)
> 
> | Spec         | Hopper  | Ampere  | Turing   |
> |--------------|---------|---------|----------|
> | Transistor   | 2.20    | 2.60    | 2.06     |
> | Process Node | 152 mmi | 166 mmi | 153 mimi |
> | Die size     | 16.8%   | 15,5%   | 16.4%    |
> | Peak FP32    | 1.60    | 1.00    | 1.07     |
> | Memory       | 257 mg  | 251 mg  | 259 mg   |
> | Bandwidth    | 14 mn   | 17 mn   | 10.90    |
> | Powe FEF1    | 2.56    | 2.40    | 2.20     |
> 
> ## **Memory Hierarchy**
> 
> ![](_page_0_Figure_13.jpeg)
> 
> ### Compattions
> 
> ![](_page_0_Figure_15.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20169.png)

## Rubin

infer at scale。PD workload。Reasoning的token暴增。

> **[图片提取文字 (image.png)]:**
> ## **Announcing NVIDIA Dynamo**
> 
> Distributed Inference Serving Library
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Disaggregated Inference
> 
> **GPU Resource Allocation** 
> 
> **KV Cache Routing** 
> 
> Communication Library (NIXL)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20170.png)

> **[图片提取文字 (image.png)]:**
> ## **NVIDIA Blackwell System**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20171.png)

> **[图片提取文字 (image.png)]:**
> I need to seat 7 people around a table at my wedding reception, but my parents and in-laws should not sit next to each other. Also, my wife insists we look better in pictures when she's on my left, but I need to sit next to my best man. How do I seat us on a round table? But then, what happens if we invited our pastor to sit with us?
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## **Traditional LLM Model**
> 
> Tokens: 439
> 
> ## **Reasoning Model**
> 
> Tokens: **8,559** 
> 
> Reasoning On
> 
> ![](_page_0_Figure_6.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20172.png)

数据

> **[图片提取文字 (image.png)]:**
> ## Inference At-Scale is Extreme Computing
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Tokens per Second For One User
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20173.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20174.png)

> **[图片提取文字 (image.png)]:**
> Blackwell 25X Hopper
> 
> FP4, NVL72, Dynamo, and TRT-LLM Continuous Optimization 1K ISL / 2K OSL
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20175.png)

> **[图片提取文字 (image.png)]:**
> Blackwell 40X Hopper
> 
> FP4, NVL72, Dynamo, and TRT-LLM Continuous Optimization 32K ISL / 8K OSL
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20176.png)

产品

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ## Grace
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ## Blackwell Ultra NVL72
> 
> Second Half 2025
> 
> 1.1 EF Dense FP4 Inference 0.36 EF FP8 Training 1.5X GB200 NVL72
> 
> New Attention Instructions 2X
> 
> 20 TB HBM | 40 TB Fast Memory 1.5X
> 
> 14.4 TB/s CX8 2X
> 
> ## Blackwell Ultra
> 
> ![](_page_0_Picture_10.jpeg)
> 
> 2 Reticle-Sized GPUs 15PF Dense FP4 | 288GB HBM3e
> 
> Oberon Rack Liquid Cooled
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20177.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ## Vera Rubin NVL144
> 
> Second Half 2026
> 
> 3.6 EF FP4 Inference 1.2 EF FP8 Training 3.3X GB300 NVL72
> 
> 13 TB/s HBM4 75 TB Fast Memory 1.6X
> 
> 260 TB/s NVLink6 2X
> 
> > 28.8 TB/s CX9 2X
> 
> ## Vera
> 
> ![](_page_0_Picture_8.jpeg)
> 
> 88 Custom Arm Cores 176 Threads 1.8 TB/s NVLink-C2C
> 
> ## Rubin
> 
> ![](_page_0_Picture_11.jpeg)
> 
> 2 Reticle-Sized GPUs 50PF FP4 | 288GB HBM4
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20178.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ## Rubin Ultra NVL576
> 
> Second Half 2027
> 
> 15 EF FP4 Inference 5 EF FP8 Training 14X GB300 NVL72
> 
> 4.6 PB/s HBM4e 365 TB Fast Memory 8X
> 
> > 1.5 PBs NVLink7 12X
> 
> 115.2 TB/s CX9 **8X** 
> 
> ## Vera
> 
> ![](_page_0_Picture_8.jpeg)
> 
> 88 Custom Arm Cores 176 Threads 1.8 TB/s NVLink-C2C
> 
> ## Rubin Ultra
> 
> ![](_page_0_Picture_11.jpeg)
> 
> 4 Reticle-Sized GPUs 100PF FP4 | 1TB HBM4e
> 
> Kyber Rack Liquid Cooled
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20179.png)

> **[图片提取文字 (image.png)]:**
> ## **NVIDIA Blackwell System**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> **Grace Blackwell NVLink72** 
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20180.png)

> **[图片提取文字 (image.png)]:**
> ## **NVIDIA Rubin System**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20181.png)

> **[图片提取文字 (image.png)]:**
> ## **NVIDIA** Paves Road to Gigawatt AI Factories
> 
> One-Year Rhythm | Full-Stack | One Architecture | CUDA Everywhere
> 
> ![](_page_0_Figure_2.jpeg)
> 
> 2025 2026 2027 2028
![image.png](NV%E5%90%84%E4%BB%A3GPU%E6%9E%B6%E6%9E%84/image%20182.png)