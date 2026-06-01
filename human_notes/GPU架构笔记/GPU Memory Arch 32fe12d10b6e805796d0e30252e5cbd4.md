# GPU Memory Arch

[https://ajdillhoff.github.io/notes/cuda_memory_architecture/](https://ajdillhoff.github.io/notes/cuda_memory_architecture/)

[https://uplatz.com/blog/the-cuda-memory-hierarchy-architectural-analysis-performance-characteristics-and-optimization-strategies/](https://uplatz.com/blog/the-cuda-memory-hierarchy-architectural-analysis-performance-characteristics-and-optimization-strategies/)

[https://www.tutorialspoint.com/cuda/cuda_memories.htm](https://www.tutorialspoint.com/cuda/cuda_memories.htm)

[https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

MEMORY SHARING VIA A UNIFIED MEMORY ARCHITECTURE

## GPU Mem

统一地址空间是索引所有设备内存的虚拟地址空间，不同设备划分不同子地址空间，不同应用从所在设备的虚拟地址空间中划分子空间。

> **[图片提取文字 (image.png)]:**
> of memory may be stale, while the LLC 114 includes the most recent data. Furthermore, in embodiments, the CPU and GPU can employ any mutually accessible storage location to perform shared virtual memory. Any mutually accessible storage location may include, but is not limited to, any area of the memory device 104, any area of the storage 120, a networked storage location, a thumbdrive, or any combination thereof. The storage 120 includes a surface 122 as well as any number of applications 124 that are configured to run on the computing device 100. The surface 122 is a designated portion of physical memory that is allocated by the device driver 110. The surface may be updated based on processing
> 
> performed on the contents of the physical memory within
> 
> Additionally, in embodiments, the CPU and GPU can
> 
> access any level of memory. However, data from other levels
![image.png](GPU%20Memory%20Arch/image.png)

> **[图片提取文字 (image.png)]:**
> is executed by CPU 104, the application 124 may request that a surface be allocated by the device driver 110. Furthermore, the applications 124 running on the CPU 102 may configure the surface 122 depending on the memory allocation called for by the applications 124 by specifying the desired size and characteristics of the surface 122. Additionally, surface allocation may be performed, for example, in response to input from the CPU 102 of the computing device 100. Furthermore, in embodiments, the surface is marked as LLC cacheable. By designated the surface 122 as LLC cacheable, the data cached from locations within the surface 122 may be cached to the LLC 114, and thereby accessible in the LLC by both the CPU 102 and the GPU
> 
> 104.
> 
> the surface 122. In embodiments, when an application 124
![image.png](GPU%20Memory%20Arch/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 2A
![image.png](GPU%20Memory%20Arch/image%202.png)

> **[图片提取文字 (image.png)]:**
> A memory management unit (MMU) 126 may be used to manage access to data that is stored within the surface 122. The MMU 126 can divide the virtual address space of the CPU 102 and the GPU 104 into various pages of address space. The CPU 102 and the GPU 104 each have their own virtual address spaces. The virtual address space allows for protection of the data contained within the surface 122 by isolating the various applications 124 executing within a computing system to a particular subset of virtual addresses. Through the use of virtual address spaces, one application 124 will not access the data of another application 124. Accordingly, the MMU 126 includes a CPU page table 128 and a GPU page table 130. The CPU page table contains the virtual addresses of the CPU mapped to a physical address location within the surface 122. Similarly, the GPU page table contains the virtual addresses of the GPU mapped to a physical address location within the surface 122. In the memory sharing procedure described herein, the CPU page table 128 may include a mapping of the CPU virtual address space to a physical address space. The physical address space corresponds to physical locations within the surface 122. Likewise, the GPU page table 130 may include a mapping of the GPU virtual address space to the same.
> 
> In various embodiments, the virtual memory addresses from the CPU page table 128 and the graphics virtual memory addresses from the GPU page table 130 are mapped to the physical memory pages of the surface 122 via a translation procedure. The translation procedure may be used to convert any of the virtual memory addresses to a corresponding physical address. For example, the translation procedure may be performed via a page table walk, which may be performed based on a specific translation table for converting virtual memory addresses within a page table to physical memory addresses within the page table. Additionally, in embodiments, a translation look-aside buffer may be used to translate the virtual addresses of the CPU and the GPU into physical address spaces within their respective page tables.
![image.png](GPU%20Memory%20Arch/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 2000 FIG. 2B
![image.png](GPU%20Memory%20Arch/image%204.png)

kernel定义Var，是分配memory资源的索引。统一地址空间是对CPU和GPU内存使用统一索引标识。统一内存是建立CPU和GPU内存的统一操作接口，底层驱动管理PTE和缺页处理。

**生命周期Lifetime**是Var存活时段（启动和销毁层），决定**物理资源分配和回收，由负载的生命周期决定**，生命周期内已分配资源只属于分配目标（抢占目标）。

**作用域Scope**是Var可见范围（运行时层），是负载的**虚拟地址空间**，作用域内的线程通过Var来使用资源、影响作用域内线程，不同作用域的资源操作隔离。

Channel和内存partition分配给应用/MPS（lifetime），应用按grid或queue TMD执行（scope），grid执行操作Channel和内存（资源）。

SMEM（SPM）分配给grid（lifetime），grid按block执行（scope），block操作SMEM和RF（资源）。

WDU解析grid TMD得到，需要执行gridSz个block，派发执行block需要分配SMEM空间，每个block需要执行blockSz个线程，每个线程需要分配RF空间，派发执行block需要分配RF空间。

grid从若干SM中申请“多份”SMEM空间和RF空间后，配置SMEM地址和RF地址给block后发到对应SM，对应数量的block并行。

每个block使用SMEM地址访问SMEM资源执行，使用RF地址和线程Id访问RF空间，执行完毕后等待下一个block到来。

gridSz个block派发且执行完毕后，申请的多个SM中的SMEM空间和RF空间释放，生命周期结束。

> **[图片提取文字 (Screenshot from 2026-03-26 19-21-42.png)]:**
> The \_\_constant\_\_ keyword can be used to store a variable in constant memory. They are always declared as global variables.
> 
> Registers and shared-memory are on-chip memories. Variables that are stored in these memories are accessed at a very high speed in a highly parallel manner. A thread is allocated a set of registers, and it cannot access registers that are not parts of that set. A kernel generally stores frequently used variables that are private to each thread in registers. The cost of accessing variables from registers is less than that required to access variables from the global memory.
> 
> SM 2.0 GPUs support up to 63 registers per thread. If this limit is exceeded, the values will be spilled from local memory, supported by the cache hierarchy. SM 3.5 GPUs expand this to up to 255 registers per thread.
> 
> ## **Shared Memory**
> 
> All threads of a block can access its shared memory. Shared memory can be used for inter-thread communication. Each block has its own shared-memory. Just like registers, shared memory is also on-chip, but they differ significantly in functionality and the respective access cost.
> 
> While accessing data from the shared memory, the processor needs to do a memory load operation, just like accessing data from the global memory. This makes them slower than registers, in which the LOAD operation is not required. Since it resides on-chip, shared memory has shorter latency and higher bandwidth than global memory. Shared memory is also called scratchpad memory in computer architecture parlance.
> 
> ## Variable Lifetime
> 
> Lifetime of a variable tells the portion of the programs execution duration when it is available for use. If a variables lifetime is within the kernel, then it will be available for use only by the kernel code. An important point to note here is that multiple invocations of the kernel do not maintain the value of the variable across them.
> 
> ## **Automatic Variables**
> 
> Automatic variables are those variables for which a copy exists for each thread. In the matrix multiplication example, row and col are automatic variables. A private copy of row and col exists for each thread, and once the thread finishes execution, its automatic variables are destroyed.
> 
> The following table summarizes the lifetime, scope and memory of different types of CUDA variables –
![Screenshot from 2026-03-26 19-21-42.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_19-21-42.png)

> **[图片提取文字 (Screenshot from 2026-03-26 19-21-57.png)]:**
> | Variable declaration                  | Memory   | Scope  | Lifetime    |
> |---------------------------------------|----------|--------|-------------|
> | Automatic variables other than arrays | Register | Thread | Kernel      |
> | Automatic array variables             | Local    | Thread | Kernel      |
> | deviceshared int sharedVar            | Shared   | Block  | Kernel      |
> | device int globalVar                  | Global   | Grid   | Application |
> | deviceconstant int constVar           | Constant | Grid   | Application |
> 
> Constant variables are stored in the global memory (constant memory), but are cached for efficient access. They can be accessed in a highly-parallel manner at high-speeds. As their lifetime equals the lifetime of the application, and they are visible to all the threads, declaration of constant variables must be done outside any function.
> 
> ## Memory as a Bottleneck
> 
> Although shared memory and registers are high-speed memories with huge bandwidth, they are available in limited amounts in a CUDA device. A programmer should be careful not to overuse these limited resources. The limited amount of these resources also caps the number of threads that can actually execute in parallel in a SM for a given application. The more resources a thread requires, the less the number of threads that can simultaneously reside in the SM. It is simply because there is a dearth of resources.
> 
> Let us suppose that each SM can accommodate upto 1536 threads and has 16,384 registers. To accommodate 1536 threads, each thread can use no more than 16,384/1536 = 10 registers. If each threads requires 12 registers, the number of threads that can simultaneously reside in the SM is reduced. Such reduction is done per block. If each block contains 128 threads, the reduction of threads will be done by reducing 128 threads at a time.
> 
> Shared memory usage can also limit the number of threads assigned to each SM. Suppose that a CUDA GPU has 16k/SM of shared memory. Suppose that each SM can support upto 8 blocks. To reach the maximum, each block must use no more than 2k of shared memory. If each block uses 5k of shared memory, then no more than 3 blocks can live in a SM.
![Screenshot from 2026-03-26 19-21-57.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_19-21-57.png)

Scope内所有线程共享资源，共享虚拟索引空间（global、SMEM、RF），线程通过指令（grid-base）+blockId+thdId访问虚拟地址进行数据交换。

资源分配给Lifetime，不同Scope的资源独立且隔离，Scope有独立的虚拟索引，内层Scope的线程可以访问外层Scope资源（不同Scope通信）。

内存分配给应用Lifetime，不同grid有不同的子虚拟空间，处于隔离的物理内存空间。

虚拟空间中，多个线程的邻接虚拟地址访问“尽量”合并成一个地址字（Cache Line）进行访问，将数据加载到SMEM中分配的空间，虚拟地址按物理page不连续，邻接虚拟地址的物理地址一定连续。

> **[图片提取文字 (Screenshot from 2026-03-26 17-47-36.png)]:**
> ## 2.2 Memory Coalescing: The Critical Optimization
> 
> The memory controller does not interact with DRAM at the granularity of individual bytes or floats. It operates on **transactions** (cache lines), typically 32 bytes, 64 bytes, or 128 bytes in size.3 When a warp executes a global memory load instruction, the hardware inspects the addresses requested by the 32 threads.
> 
> ## The Coalescing Mechanism:
> 
> If the addresses are contiguous and aligned—for example, Thread \$k\$ accesses address \$Base + k \times 4\$—the hardware coalesces these 32 requests into a single or minimum number of transactions. For 32-bit words (4 bytes), a full warp requests \$32 \times 4 = 128\$ bytes. If aligned, this results in exactly one 128-byte transaction, achieving 100% bus utilization efficiency.3
> 
> The Penalty of Uncoalesced Access:
> 
> If threads access memory with a stride—for example, Thread \$k\$ accesses \$Base + k \times 8\$—the requested addresses span 256 bytes. The memory controller must issue two 128-byte transactions (or more, depending on alignment) to fetch the data. However, only half of the data in those transactions is actually used by the threads. This reduces effective bandwidth by 50%. In random access patterns (e.g., pointer chasing or indirect indexing A[i]]), the efficiency can drop to 3-4%, as a full 128-byte line is fetched to satisfy a request for a single 4-byte value.3
![Screenshot from 2026-03-26 17-47-36.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_17-47-36.png)

> **[图片提取文字 (Screenshot from 2026-03-26 17-34-57.png)]:**
> | Access Pattern           | Description                           | Transactions per Warp (approx) | Bus<br>Efficiency |
> |--------------------------|---------------------------------------|--------------------------------|-------------------|
> | Sequential<br>Aligned    | \$Address = Base + tid\$              | 1 (128 bytes)                  | 100%              |
> | Sequential<br>Misaligned | \$Address = Base + tid + Offset\$     | 2 (128 bytes)                  | ~50-80%           |
> | Strided (Stride 2)       | \$Address = Base + tid<br>\times 2\$  | 2 (128 bytes)                  | 50%               |
> | Strided (Large)          | \$Address = Base + tid<br>\times 32\$ | 32 (32 bytes each)             | ~12.5%            |
> | Random                   | Indirect access                       | up to 32                       | < 10%             |
![Screenshot from 2026-03-26 17-34-57.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_17-34-57.png)

> **[图片提取文字 (Screenshot from 2026-03-26 19-55-55.png)]:**
> | Memory<br>Type | Scope  | Lifetime | Physical<br>Location | Cached?        | Latency<br>(Cycles) | Bandwidth                     | Optimal<br>Access<br>Pattern   |
> |----------------|--------|----------|----------------------|----------------|---------------------|-------------------------------|--------------------------------|
> | Register       | Thread | Thread   | On-Chip<br>(SM)      | N/A            | ~0                  | ~8-10 TB/s<br>(Aggregate)     | N/A                            |
> | Shared         | Block  | Block    | On-Chip<br>(SM)      | N/A            | 20-50               | ~10-15<br>TB/s<br>(Aggregate) | Conflict-<br>Free<br>(Padding) |
> | L1<br>Cache    | N/A    | N/A      | On-Chip<br>(SM)      | N/A            | 30-50               | High                          | Spatial<br>Locality            |
> | L2<br>Cache    | Device | Арр      | On-Chip<br>(Shared)  | N/A            | 200                 | ~3-5 TB/s                     | Spatial<br>Locality            |
> | Global         | Grid   | Арр      | Off-Chip<br>DRAM     | Yes<br>(L1/L2) | 400-800             | 1-3 TB/s                      | Coalesced<br>(Sequential)      |
> | Local          | Thread | Thread   | Off-Chip<br>DRAM     | Yes<br>(L1/L2) | 400-800             | 1-3 TB/s                      | Coalesced<br>(per-thread)      |
> | Constant       | Grid   | Арр      | Off-Chip<br>DRAM     | Yes<br>(Const) | varies              | High<br>(Broadcast)           | Uniform<br>(Broadcast)         |
> | Texture        | Grid   | Арр      | Off-Chip<br>DRAM     | Yes (Tex)      | 100+                | High                          | 2D/3D<br>Spatial<br>Locality   |
![Screenshot from 2026-03-26 19-55-55.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_19-55-55.png)

## GPU MM kernel

Tiling：外层Memory容量大但访问慢，将数据分块加载到内层Memory。

> **[图片提取文字 (Screenshot from 2026-03-26 15-37-00.png)]:**
> These memory types serve as tools that we can use to increase efficiency. The first pattern discussed is **tiling**. Throughout the rest of the course, we will add many more patterns to our repertoire. Tiling is a well-described technique that has a fitting analogy. If a wall needs to be tiled, it is more efficient to use many small tiles that are lighter and easier to handle. In GPU programming, the wall represents the entire global memory space. The individual tiles are local memory that is allocated to each thread block.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Figure 1: Global memory access pattern (source: NVIDIA DLI).
![Screenshot from 2026-03-26 15-37-00.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-37-00.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-37-20.png)]:**
> The kernels we have seen so far have used a *global memory access pattern*. In this pattern, all threads have access to every data point from the input. Using a *tiling pattern*, we can optimize memory accesses by moving shared resources to local memory that is faster to access.
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Figure 2: Tiling pattern (source: NVIDIA DLI).
![Screenshot from 2026-03-26 15-37-20.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-37-20.png)

Tiled MM：线程block计算矩阵乘法的输出矩阵，SIMT要求线程block执行相同指令，统一成ld-com-str阶段，每个阶段是相同指令的不同地址或数据。

> **[图片提取文字 (Screenshot from 2026-03-26 15-39-37.png)]:**
> The tool itself is quite simple in concept, but the challenge will be identifying when the tool can be properly applied. Consider matrix multiplication. The naive kernel we explored previously uses each thread to compute one value of the output matrix. This kernel uses a global memory access pattern, and we can identify that many of the computations require the same input. They key to introducing tiling for matrix multiplication will be identifying which data are reused.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 3: Memory accesses for matrix multiplication (source: NVIDIA DLI).
![Screenshot from 2026-03-26 15-39-37.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-39-37.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-40-10.png)]:**
> In the figure above, the block size is  $2 \times 2$ . Each row of the block relies on the same input row from the matrix on the left. That is,  $P_{0,0}$  and  $P_{0,1}$  will use the same data from the first row of M. In our original kernel, this requires 8 global memory accesses. If we placed this row in shared memory, each output thread could access the values much quicker. We can see a similar pattern for the column values in N.
> 
> Since we are using tiling with a block size of B, we will consider working with 2B values from the input at a time. If the number of values we need to compute an output entry exceeds 2B, then we can synchronize the threads before moving to the next section.
> 
> ![](_page_0_Figure_2.jpeg)
![Screenshot from 2026-03-26 15-40-10.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-40-10.png)

每个线程block有4个线程，并行计算2*2的输出矩阵，使用4个线程依次加载2*2的A和B到SMEM，并行计算2*2个内积，存储到SMEM。

kernel编程：每个线程依次加载A和B的对应元素，计算1个内积，存储到SMEM。

> **[图片提取文字 (Screenshot from 2026-03-26 15-40-25.png)]:**
> The concept of tiled matrix multiplication is this: load a subset of data from M and N into shared memory before using that data to perform the dot product. We have a few limitations to think about here. First, the amount of shared memory is much smaller than global memory; we cannot fit all the data at once. Second, the block size will limit how many elements can be loaded into shared memory at once. As suggested by tiling, we are only working with a small chunk at a time.
> 
> Using a  $2 \times 2$  block gives us 4 threads to work with. Overlaying that block on the input only allows us to grab 2 values from the first 2 rows in M and 2 values from the first 2 columns in M. For each tile, the subset of data will be loaded in followed by adding the dot product of the subset to the current value.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 5: Loading the first tile (source: NVIDIA DLI).
![Screenshot from 2026-03-26 15-40-25.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-40-25.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-40-58.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6: Computing the dot product of the first subset (source: NVIDIA DLI).
> 
> In this example, the block will move to the next subset of data to finish computing the first block of the output matrix. This process can be arbitrarily scaled up to support larger matrices without necessarily increasing the block size. Although, we would want to increase the block size to take advantage of the additional threads. The figure below shows a table of the computations required for each phase.
![Screenshot from 2026-03-26 15-40-58.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-40-58.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-41-09.png)]:**
> |                       | Phase 0                                              |                                                                                                        |                                                                                                                                       | Phase 1                                                 |                                                                                      |                                                                                                                                       |
> |-----------------------|------------------------------------------------------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
> | thread <sub>0,0</sub> | $\mathbf{M}_{0,0}$ $\downarrow$ $\mathbf{M}ds_{0,0}$ | $N_{0,0}$ $\downarrow$ $Nds_{0,0}$                                                                     | $\begin{array}{l} \text{PValue}_{0,0} += \\ \text{Mds}_{0,0} * \text{Nds}_{0,0} + \\ \text{Mds}_{0,1} * \text{Nds}_{1,0} \end{array}$ | $\mathbf{M_{0,2}}$ $\downarrow$ $\mathbf{Mds_{0,0}}$    | $\begin{vmatrix} \mathbf{N}_{2,0} \\ \downarrow \\ \mathbf{Nds}_{0,0} \end{vmatrix}$ | $\begin{array}{l} \text{PValue}_{0,0} += \\ \text{Mds}_{0,0} * \text{Nds}_{0,0} + \\ \text{Mds}_{0,1} * \text{Nds}_{1,0} \end{array}$ |
> | thread <sub>0,1</sub> | $M_{0,1}$ $\downarrow$ $Mds_{0,1}$                   | $N_{0,1}$ $\downarrow$ $Nds_{1,0}$                                                                     | $\begin{array}{l} PValue_{0,1} += \\ Mds_{0,0}*Nds_{0,1} + \\ Mds_{0,1}*Nds_{1,1} \end{array}$                                        | $\mathbf{M}_{0,3}$ $\downarrow$ $\mathbf{M}ds_{0,1}$    | $N_{2,1}$ $\downarrow$ $Nds_{0,1}$                                                   | $PValue_{0,1} += \\ Mds_{0,0}*Nds_{0,1} + \\ Mds_{0,1}*Nds_{1,1}$                                                                     |
> | thread <sub>1,0</sub> | $\mathbf{M}_{1,0}$ $\downarrow$ $\mathbf{M}_{1,0}$   | $N_{1,0}$ $\downarrow$ $Nds_{1,0}$                                                                     | $\begin{array}{l} PValue_{1,0} += \\ Mds_{1,0}*Nds_{0,0} + \\ Mds_{1,1}*Nds_{1,0} \end{array}$                                        | $\mathbf{M}_{1,2}$ $\downarrow$ $\mathbf{M}ds_{1,0}$    | $N_{3,0}$ $\downarrow$ $Nds_{1,0}$                                                   | $PValue_{1,0} += \\ Mds_{1,0}*Nds_{0,0} + \\ Mds_{1,1}*Nds_{1,0}$                                                                     |
> | thread <sub>1,1</sub> | $M_{1,1}$ $\downarrow$ $Mds_{1,1}$                   | $\begin{matrix} \mathbf{N}_{1,1} \\ \downarrow \\ \mathbf{N} \mathbf{d} \mathbf{s}_{1,1} \end{matrix}$ | $\begin{array}{c} PValue_{1,1} += \\ Mds_{1,0}*Nds_{0,1} + \\ Mds_{1,1}*Nds_{1,1} \end{array}$                                        | $\mathbf{M}_{1,3}$ $\downarrow$ $\mathbf{M}_{ds_{1,1}}$ | $N_{3,1}$ $\downarrow$ $Nds_{1,1}$                                                   | $\begin{array}{c} PValue_{1,1} += \\ Mds_{1,0}*Nds_{0,1} + \\ Mds_{1,1}*Nds_{1,1} \end{array}$                                        |
> | time                  |                                                      |                                                                                                        |                                                                                                                                       |                                                         |                                                                                      |                                                                                                                                       |
> 
> Figure 7: Tiled matrix multiplication computations (source: NVIDIA DLI).
> 
> ## Check your understanding
> 
> By using tiling with a block size of  $B \times B$ , what is the total reduction in global memory traffic?
![Screenshot from 2026-03-26 15-41-09.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-41-09.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-42-02.png)]:**
> The row and column of the output computed by the current thread is calculated using the block and thread indices. Of course, this is simply (0,0) for the first thread. It gets slightly more complicated when computing the input subset in the loop. The input needs to be transferred to shared memory. The loop will skip over a tile at a time. At this point, we already know which row of M and column of N we need to access. We need to compute the column index for M and the row index for N.
> 
> For M, we start with Row \* Width. This needs to be offset by the tile offset index ph of the main loop, yielding Row \* Width + ph \* TILE\_WIDTH. Finally, we need to add the thread index tx to get the final index Row \* Width + ph \* TILE\_WIDTH + tx. The same process is applied to N. Note that this only transfers a single value from each matrix to shared memory, but our computation relies on 2 values from each matrix. Each thread in the block is collaboratively loading the data into shared memory. This is why the call to \_\_syncthreads() is necessary.
> 
> Specifically, the thread for  $P_{0,0}$  copies  $M_{0,0}$  and  $N_{0,0}$  to shared memory. The thread for  $P_{0,1}$  copies  $M_{0,1}$  and  $N_{1,0}$  to shared memory. The thread for  $P_{1,0}$  copies  $M_{1,0}$  and  $N_{0,1}$  to shared memory. Finally, the thread for  $P_{1,1}$  copies  $M_{1,1}$  and  $N_{1,1}$  to shared memory.
> 
> The next step is to compute the dot product of the subset. Again, we see a call to \_\_syncthreads(). Without this synchronization, the loop may be allowed to continue and overwrite the data in shared memory before a thread has finished. Once the final value is computed, each thread can freely write it back to global memory. Since each thread is computing a different value, there is no need to synchronize the threads before writing to global memory.
> 
> $$\begin{aligned} P_{0,0}+&=2\times2+1\times1\ P_{0,1}+&=2\times1+1\times0\ P_{1,0}+&=1\times2+0\times1\ P_{1,1}+&=1\times1+0\times0 \end{aligned}$$
![Screenshot from 2026-03-26 15-42-02.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-42-02.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-41-36.png)]:**
> ```
> _global__ void MatMulKernel(float* M, float* N, float* P, int Width) {
> // Block index
> int bx = blockIdx.x;
> int by = blockIdx.y;
> // Thread index
> int tx = threadIdx.x;
> int ty = threadIdx.y;
> __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
> __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
> // Identify the row and column of the P element to work on
> int Row = by * TILE_WIDTH + ty;
> int Col = bx * TILE_WIDTH + tx;
> float Pvalue = 0;
> for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {</pre>
>     // Collaborative loading of M and N tiles into shared memory
>     Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
>     Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
>     __syncthreads();
>     for (int k = 0; k < TILE_WIDTH; ++k) {</pre>
>         Pvalue += Mds[ty][k] * Nds[k][tx];
>     __syncthreads();
> P[Row * Width + Col] = Pvalue;
> ```
![Screenshot from 2026-03-26 15-41-36.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-41-36.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-42-40.png)]:**
> The previous implementation assumed that the width of the matrices was a multiple of the tile width and that the input would always be square matrices. Consider changing our  $2 \times 2$  block to a  $3 \times 3$  block using the same input sizes.
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 11: Using a 3x3 block (source: NVIDIA DLI).
![Screenshot from 2026-03-26 15-42-40.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-42-40.png)

边界处理、Memory占用、动态决定使用SMEM的大小来支持分块大小。

SMEM的分配是地址段的分配（不涉及MMU，类似地址索引的Buffer），因此允许block启动时指定block需要的SMEM大小。

> **[图片提取文字 (Screenshot from 2026-03-26 15-42-50.png)]:**
> Our implementation would follow the same process for the first subset of pattern. An issue arises when computing the second tile offset since the block exceeds the boundaries of our input and output. One solution would be to check the boundary condition on both the input, when transferring the data to shared memory, and the output, when reading the data from shared memory. This would require a conditional statement in the inner loop. This is not ideal since the conditional statement would be executed for every thread in the block.
> 
> Another solution is to pad the input with zeros. If the index is outside our boundary, adding a 0 will not affect the result of the dot product. This allows for a simpler implementation while still being flexible enough to handle matrices of any size. The relevant portion of the kernel is shown below.
> 
> ```
> float Pvalue = 0;
> for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ph++) {</pre>
>     // Collaborative loading of M and N tiles into shared memory
>     if (Row < Width && ph * TILE_WIDTH + tx < Width) {</pre>
>         Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
>     } else {
>         Mds[ty][tx] = 0.0;
>     if (ph * TILE_WIDTH + ty < Width && Col < Width) {</pre>
>         Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
>     } else {
>         Nds[ty][tx] = 0.0;
>      _syncthreads();
> ```
> 
> The rest of the kernel remains the same. In Lab 2, you will implement this and adapt it to work with non square matrices as well.
![Screenshot from 2026-03-26 15-42-50.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-42-50.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-44-02.png)]:**
> ## Memory Use and Occupancy
> 
> Just like exceeding the number of registers per thread can negatively affect occupancy, so can over allocating shared memory. The H100 can have up to 228 KB per SM. If we are maximizing the 2048 threads available per SM, each block cannot exceed 228 KB / 2048 threads = 112 B/thread.
> 
> How much shared memory is used by each block? Each block has 2 arrays of size  $TILE\_WIDTH \times TILE\_WIDTH$  of type float. This gives us a total of  $2 \times TILE\_WIDTH \times TILE\_WIDTH \times 4 = 8(TILE\_WIDTH)^2$  B. Each block uses  $TILE\_WIDTH^2$  threads, resulting in 8 B/thread. This is well below the limit of 112 B/thread.
![Screenshot from 2026-03-26 15-44-02.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-44-02.png)

> **[图片提取文字 (Screenshot from 2026-03-26 15-44-48.png)]:**
> The solution presented above uses a constant to determine the tile size. What if this tile size was not optimal for a given hardware configuration? We would surely want to adjust this dynamically to maximize performance. In CUDA, we can support this by using the extern keyword. First, we need to define our shared memory as one array: extern \_\_shared\_\_ float Mds\_Nds[];. This is a 1D array that represents the shared memory for both input matrices.
> 
> When launching this kernel, we need some way to inform it of the tile size. First, we would query the device properties and determine the optimal tile size based on the hardware. This size can be used as a third launch configuration input, as shown below. Additionally, the size of the shared memory for each input matrix is provided as two additional arguments to the kernel.
> 
> ```
> size_t size = compute_optimal_size(); // Determine optimal tile size
> MatMulKernel<<<dimGrid, dimBlock, size>>>(M_d, N_d, P_d, Width, size/2, size/2);
> ```
> 
> The kernel will need to be modified to use the new shared memory array. The first step is to determine the offset for each matrix. This is done by multiplying the tile size by the thread index. The second step is to use the offset to access the correct value in the shared memory array. The kernel is shown below.
> 
> ```
> __global__ void MatMulKernel(float* M, float* N, float* P, int Width, int Mds_offset, int Nds_offset) {
>     extern __shared__ float Mds_Nds[];
> 
>     float *Mds = (float *)Mds_Nds;
>     float *Nds = (float *)Mds_Nds + Mds_offset;
> 
>     // Rest of the kernel
> }
> ```
> 
> Completing this modification would require us to use linear indexing for Mds and Nds.
![Screenshot from 2026-03-26 15-44-48.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_15-44-48.png)

## GPU Conv

[https://ajdillhoff.github.io/notes/gpu_pattern_convolution/](https://ajdillhoff.github.io/notes/gpu_pattern_convolution/)

[https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/](https://eunomia.dev/others/cuda-tutorial/06-cnn-convolution/)

基础Conv。优化后，权重存储Constant Mem，被Cache而快速访问。

> **[图片提取文字 (Screenshot from 2026-03-26 18-11-01.png)]:**
> It is straightforward to write the convolution operation in CUDA C++. Each thread will compute the value for a single output pixel using the filter. We already implemented something very similar with the blurring kernel. The kernel itself should accept the following arguments:
> 
> - The input image
> - The output image
> - The kernel
> - The radius of the kernel
> - The width of the output image
> - The height of the output image
> 
> A more robust implementation would consider things like padding, stride, dilation, and whether or not a valid or full convolution is desired. For now, we will focus on the simplest case: a valid convolution with a stride of 1 and no padding or dilation. First, let's review the initial naive solution from *Programming Massively Parallel Processors* (Hwu, Kirk, and El Hajj 2022).
> 
> ```
> __global___ void conv2D(float *input, float *filter, float *output,
> ```
![Screenshot from 2026-03-26 18-11-01.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_18-11-01.png)

> **[图片提取文字 (Screenshot from 2026-03-26 18-11-47.png)]:**
> ## Constant Memory and Caching
> 
> There is a much larger issue present in both versions of this kernel in terms of memory bandwidth. Similar to the matrix multiplication kernel, this kernel can benefit from tiling. However, there is a new problem that arises specifically with convolution. The same filter is accessed by every single thread. This filter does not change for the entire duration of the kernel. This means that we are wasting memory bandwidth by having every thread access the same filter.
> 
> Given its relatively small size, this kernel is a perfect candidate for constant memory. This is a special type of memory that is cached on the GPU. It is read-only and has a limited size, but it is much faster than global memory. We can write to the devices constant memory from the host code.
> 
> ```
> #define FILTER_RADIUS 1
> __constant__ float kFilter_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
> ```
> 
> This informs the compiler to allocate a 2D array of floats in constant memory. The size of the array is determined by the constant `FILTER\_RADIUS`. We can then copy the filter to the device using the `cudaMemcpyToSymbol` function.
> 
> ```
> cudaMemcpyToSymbol(kFilter_d, filter_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
> ```
> 
> The line above assumes there is some data on the host in the array `filter\_h`. This array is copied to the device. A small note on naming convention, Google's C++ style guide recommends naming constant variables with a k prefix. I have adopted this convention here.
> 
> At this point, kFilter\_d is accessible from the kernel as a global variable. There is no need to pass it as an argument. The kernel can be modified to use this constant memory as follows.
![Screenshot from 2026-03-26 18-11-47.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_18-11-47.png)

> **[图片提取文字 (Screenshot from 2026-03-26 18-12-21.png)]:**
> ```
> __global__ void conv2D(float *input, float *output,
>                    int r, int width, int height) {
> int outCol = blockIdx.x * blockDim.x + threadIdx.x;
> int outRow = blockIdx.y * blockDim.y + threadIdx.y;
> float sum = 0.0f;
> for (int row = 0; row < 2*r+1; row++) {
>     for (int col = 0; col < 2*r+1; col++) {
>         int inRow = outRow + row;
>         int inCol = outCol + col;
>         sum += input[inRow * width + inCol] * F_d[row][col];
> output[outRow * width + outCol] = sum;
> ```
> 
> If you organize your files such that the kernel is in a separate file from the host code, you will need to declare the constant variable in the kernel file as well.
> 
> Constant memory variables are stored in DRAM with global memory. The CUDA runtime will cache them since it knows they will not be modified. Processors use caches to reduce the latency of memory accesses by keeping frequently used data in a small, fast memory that is often located directly on the chip. This type of *constant cache* is preferable to one that would support high-throughput writes in terms of chip design. It would require specialized hardware to support both which would increase the cost of the chip.
![Screenshot from 2026-03-26 18-12-21.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_18-12-21.png)

Tiled Conv用多个线程读取Tile输入到SPM，直接Conv的访存和内积，没有提前im2col后GEMM。

Cache the Halo，光环线程不参与当前Tile结果计算，参与后续Tile计算而直接从SPM读取。

> **[图片提取文字 (Screenshot from 2026-03-26 18-14-23.png)]:**
> There are a few things to consider here. The first phase of this kernel collaboratively loads data into a shared memory space, similar to what we have seen before. This kernel assumes a convenient indexing scheme where the row and column will always be >= 0. We could adopt a scheme that centers the convolution on the center point of the kernel by allowing for negative indices. In this case, it would be necessary to check if the row and column are less than 0. This implementation only needs to verify that the row and column are within the given image size.
> 
> When it comes to computing the output, not every thread will contribute. This is depicted by the lightly shaded areas in the figure below. You should also note which threads are active for output computation per block. In this simple example, a  $3 \times 3$  filter is used. The input tile dimension is  $4 \times 4$  which means the output tile will be  $2 \times 2$ . Only the threads corresponding to the darker blue on the left contribute to the output calculation. Since this one block computes 4 output values, the next block should start 2 units to the right of this one.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 7: The active threads for computing the output tile.
![Screenshot from 2026-03-26 18-14-23.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_18-14-23.png)

> **[图片提取文字 (Screenshot from 2026-03-26 18-14-05.png)]:**
> ```
> #define FILTER_RADIUS 1
> #define IN_TILE_DIM 4
> #define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
> __constant__ float kFilter_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
>  _global__ void conv2DTiledConstKernel(float *input, float *output,
>                                         int width, int height) {
>     __shared__ float inputTile[IN_TILE_DIM][IN_TILE_DIM];
>     // Input tile coordinates
>     int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
>     int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
>     if (row < height && col < width) {</pre>
>         inputTile[threadIdx.y][threadIdx.x] = input[row * width + col];
>     } else {
>         inputTile[threadIdx.y][threadIdx.x] = 0.0f;
>     __syncthreads();
>     // Output tile coordinates
>     int tileCol = threadIdx.x - FILTER_RADIUS;
>     int tileRow = threadIdx.y - FILTER_RADIUS;
>     // In a valid convolution, the output is smaller than the input
>     row -= FILTER_RADIUS;
>     col -= FILTER_RADIUS;
>     if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
>         float sum = 0.0f;
>         for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {</pre>
>             for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {</pre>
>                 sum += inputTile[tileRow + fRow][tileCol + fCol] * kFilter_d[fRow][fCol];
>         output[row * (width - 2 * FILTER_RADIUS) + col] = sum;
> ```
![Screenshot from 2026-03-26 18-14-05.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_18-14-05.png)

> **[图片提取文字 (Screenshot from 2026-03-26 19-41-40.png)]:**
> ## Caching the Halo Cells
> 
> In the previous example, the size of the input tile compared to the output tile means that there were some threads that did not contribute to the output computation. These are the threads managing the lightly shaded cells in the figure above. We will refer to these as *halo cells*.
> 
> This implementation is going to take advantage of the caching behavior in the chip itself. **Values that have been recently used are more likely to already be in L2 cache.** This is a safe assumption since the neighboring blocks will have loaded these values into shared memory. This means that the input and output tile sizes can be the same; there is no need to waste any threads in the block. The full kernel is given below.
![Screenshot from 2026-03-26 19-41-40.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_19-41-40.png)

> **[图片提取文字 (Screenshot from 2026-03-26 18-16-12.png)]:**
> ```
> _global__ void conv2DTiledCachedConstKernel(float *input, float *output,
>                                               int width, int height) {
>     __shared__ float inputTile[IN_TILE_DIM][IN_TILE_DIM];
>     // Input tile coordinates
>     int col = blockIdx.x * IN_TILE_DIM + threadIdx.x;
>     int row = blockIdx.y * IN_TILE_DIM + threadIdx.y;
>     if (row < height && col < width) {</pre>
>         inputTile[threadIdx.y][threadIdx.x] = input[row * width + col];
>     } else {
>         inputTile[threadIdx.y][threadIdx.x] = 0.0f;
>     __syncthreads();
>     if (row < FILTER_RADIUS || col < FILTER_RADIUS || col >= (width - FILTER_RADIUS) || row >= (height -
> FILTER_RADIUS)) return;
>     // Output tile coordinates
>     row -= FILTER_RADIUS;
>     col -= FILTER_RADIUS;
>     int tileCol = threadIdx.x - FILTER_RADIUS;
>     int tileRow = threadIdx.y - FILTER_RADIUS;
>     float sum = 0.0f;
>     for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {</pre>
>         for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {</pre>
>             // If this value is in shared memory, access it there
>             if (tileCol + fCol >= 0 &&
>                 tileCol + fCol < IN_TILE_DIM &&
>                 tileRow + fRow >= 0 &&
>                 tileRow + fRow < IN_TILE_DIM) {</pre>
>                 sum += inputTile[tileRow + fRow][tileCol + fCol] * kFilter_d[fRow][fCol];
>             } else {
>                 // Otherwise, access it from global memory
>                 sum += input[(row + fRow) * width + (col + fCol)] * kFilter_d[fRow][fCol];
>     output[row * (width - 2 * FILTER_RADIUS) + col] = sum;
> ```
![Screenshot from 2026-03-26 18-16-12.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_18-16-12.png)

## GPU Reduction

[https://ajdillhoff.github.io/notes/gpu_pattern_reduction/](https://ajdillhoff.github.io/notes/gpu_pattern_reduction/)

reduction：多输入单输出的规约，如内积。

> **[图片提取文字 (Screenshot from 2026-03-26 20-12-17.png)]:**
> ## Introduction
> 
> Given a set of values, a **reduction** produces a single output. It is an important part of many parallel algorithms including MapReduce. Other patterns that we have studied can also be viewed as reductions, such as GPU Pattern: Parallel Histogram. Implementing this parallel pattern requires careful consideration of thread communication, and will be the focus of these notes.
> 
> Many of the operations you rely on are examples of reductions. For example, the `sum` function is a reduction, as is the `max` function. A reduction can be viewed as a linear combination of the input values, or transformed values, and is often used to compute a summary statistic. If  $\phi(\cdot)$  is a binary operator, then a reduction computes the following:
> 
> $$v=\phi(v,x_i) \text{ for } i=1,2,\ldots,n,$$
> 
> where v is the accumulated value and  $x_i$  are the input values. The operator  $\phi(\cdot)$  can be any associative and commutative operation, such as addition or multiplication. Each operator has a corresponding identity element, such as 0 for addition or 1 for multiplication. The identity element is used to initialize the reduction and can be represented as  $v=v_0$  in the equation above.
> 
> ## **Reduction Trees**
> 
> Reductions of any kind are well represented using trees. The first level of reduction maximizes the amount of parallelism. As the input is gradually reduced, fewer threads are needed.
![Screenshot from 2026-03-26 20-12-17.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-12-17.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-11-47.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Sum reduce as a reduction tree.
> 
> In order to implement a parallel reduction, the chosen operator must be associative. For example, a + (b + c) = (a + b) + c. The operator must also be commutative, such that a + b = b + a.
> 
> Reduction trees reveal the logarithmic nature of parallel reductions. Just like divide and conquer algorithms, the number of threads is halved at each level of the tree. The number of levels in the tree is  $\log_2(n)$ , where n is the number of input values. Given an input size of n=1024, the number of threads required is  $\log_2(1024)=10$ . This is a significant reduction from the original input size. The sequential version of this reduction would require 1023 operations.
![Screenshot from 2026-03-26 20-11-47.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-11-47.png)

多输出对应多线程时，线程同步开销、线程分束（warp效率）和内存访问效率（Stride地址）低。

> **[图片提取文字 (Screenshot from 2026-03-26 20-12-38.png)]:**
> ## A Simple Kernel
> 
> As mentioned above, reduction requires communication between threads. Since only the threads within a single block can communicate, we will focus on a block-level reduction. For now, each block can work with a total of 2048 input values based on the limitation of 1024 threads per block.
> 
> ```
> _global___ void sumReduceKernel(float *input, float *output) {
> unsigned int i = 2 * threadIdx.x;
>     for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {</pre>
>         // Only threads in even positions participate
>         if (threadIdx.x % stride == 0) {
>              input[i] += input[i + stride];
>          __syncthreads();
>     }
>     if (threadIdx.x == 0) {
>          *output = input[0];
> ```
> 
> Each thread is assigned to a single write location 2 \* threadIdx.x. The stride is doubled after each iteration of the loop, effectively halving the number of active threads. The stride also determines the second value that is added to the first. By the last iteration, only one thread is active to perform that last reduction.
![Screenshot from 2026-03-26 20-12-38.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-12-38.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-12-46.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Execution of kernel reduction (Source: NVIDIA DLI).
> 
> You can see that the kernel is simple, but it is also inefficient. There is a great deal of control divergence that will be addressed in the next section.
![Screenshot from 2026-03-26 20-12-46.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-12-46.png)

重新排列线程和输出的映射，提高warp小丑，降低分束开支。

> **[图片提取文字 (Screenshot from 2026-03-26 20-17-01.png)]:**
> ## Minimizing Control Divergence
> 
> As we just saw, the key to optimizing a reduction kernel is to minimize control divergence and make sure as many threads stay active as possible. A warp of 32 threads would consume the execution resources even if half of them are inactive. As each stage of the reduction tree is completed, the amount of wasted resources increases. Depending on the input size, entire warps could be launched and then immediately become inactive.
> 
> The number of execution resources consumed is proportional to the number of active warps across all iterations. We can compute the number of resources consumed as follows:
> 
> $$(\frac{5N}{64} + \frac{N}{128} + \frac{N}{256} + \dots + 1) * 32$$
> 
> where N is the number of input values. Each thread operates on 2 values, so  $\frac{N}{2}$  are launched in total. Since every warp has 32 threads, a total of  $\frac{N}{64}$  warps are launched. For the first 5 iterations, all warps will be active. The 5th iteration only has 1 active thread in each warp. On the 6th iteration, the number of active warps is halved, and so on.
> 
> For an input of size N=1024, the number of resources consumed is (80+8+4+2+1)\*32=3040. The total number of results committed by the active threads is equal to the number of operations performed, which is N-1=1023. The efficiency of the kernel is then  $\frac{1023}{3040}=0.34$ . Only around 34% of the resources are used to perform the reduction.
> 
> ## Rearranging the Threads
> 
> A simple rearrangement of where the active results are stored can improve the efficiency of the kernel by reducing control divergence. The idea is to keep the threads that own the results of the reduction close together. Instead of increasing the stride, it should be decreased. The figure below shows the rearrangement of the threads.
![Screenshot from 2026-03-26 20-17-01.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-17-01.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-19-43.png)]:**
> ```
> __global__ void sumReduceKernel(float *input, float *output) {
>     unsigned int i = threadIdx.x;
> 
>     for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
>         if (i < stride) {
>             input[i] += input[i + stride];
>         }
>         __syncthreads();
>     }
> 
>     if (threadIdx.x == 0) {
>         *output = input[0];
>     }
> }</pre>
> ```
> 
> The kernel itself is effectively the same, but the rearrangement of the threads ensures that each warp has less control divergence. Additionally, warps that drop off after each iteration are no longer consuming execution resources. For an input of 256, the first 4 warps are fully utilized (barring the last thread of the last warp). After the first iteration, the number of active warps is halved. Warps 3 and 4 are now fully inactive, leaving warps 1 and 2 to perform the reduction operation on all threads. We can compute the number of resources consumed under this new arrangement as follows:
> 
> $$(\frac{N}{64} + \frac{N}{128} + \frac{N}{256} + \dots + 1 + 5) * 32$$
> 
> At each iteration, half of warps become inactive and no longer consume resources. The last warp will consume execution resources for all 32 threads, even though the number of active threads is less than 32. For our input of size N=1024, the number of resources consumed is (16+8+4+2+1+5)\*32=1152, resulting in an efficiency of  $\frac{1023}{1152}=0.89$ . This is a significant improvement over the original kernel. This will increase based on the input size.
![Screenshot from 2026-03-26 20-19-43.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-19-43.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Memory%20Arch/image%205.png)

内存访问优化

> **[图片提取文字 (Screenshot from 2026-03-26 20-20-11.png)]:**
> ## Memory Divergence of Reduction
> 
> Does this kernel take advantage of memory coalescing? Each thread reads and writes from and to its *assigned* location. It also makes a read from a location that is a stride away. These locations are certainly not adjacent and will not be coalesced.
> 
> Adjacent threads do not access adjacent locations. The warp itself is unable to coalesce the thread requests into a single global memory request. Each data element is 4 bytes. Since each of the 32 threads in a warp are accessing their assigned locations with a separation of stride, the 64 \* 4 bytes will require two 128 byte memory requests to access the data. With each iteration, the assigned locations will always be separated such that two 128 byte memory requests will need to be made. Only on the last iteration, where only a single thread accesses a single assigned location, will a single memory request be made.
> 
> The convergent kernel from the last section takes advantage of memory coalescing, leading to fewer memory requests.
> 
> ## Reducing the number of global memory requests
> 
> As we saw with tiling in GPU Performance Basics, we can reduce the number of global memory requests by using shared memory. Threads write their results to global memory, which is read again in the next iteration. By keeping the intermediate results in shared memory, we can reduce the number of global memory requests. If implemented correctly, only the original input values will need to be read from global memory.
![Screenshot from 2026-03-26 20-20-11.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-20-11.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-21-05.png)]:**
> ```
> <u>__global__</u>    void sumReduceSharedKernel(float *input, float *output) {
> __shared__ float input_s[BLOCK_DIM];
> unsigned int i = threadIdx.x;
> input_s[i] = input[i] + input[i + BLOCK_DIM];
>  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
>      __syncthreads();
>      if (i < stride) {</pre>
>          input_s[i] += input_s[i + stride];
> if (i == 0) {
>      *output = input_s[0];
> ```
> 
> At the very top of this kernel, the necessary input is loaded from global memory, added, and written to shared memory. This is the only time global memory is accessed, with the exception of the final write to the output. The call to syncthreads() moves to the top so that the shared memory is guaranteed before the next update.
> 
> This approach not only requires fewer global memory requests, but the original input is left unmodified.
![Screenshot from 2026-03-26 20-21-05.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-21-05.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-21-13.png)]:**
> ## Hierarchical Reduction
> 
> One major assumption that has been made in each of these kernels is that they are running on a single block. Thread synchronization is critical to the success of the reduction. If we want to reduce a larger number of input across multiple blocks, the kernel should allow for independent execution. This is achieved by segmenting the input and performing a reduction on each segment. The final reduction is then performed on the results of the segment reductions.
> 
> ```
> __global__ void sumReduceHierarchicalKernel(float *input, float *output) {
> __shared__ float input_s[BLOCK_DIM];
> unsigned int segment = 2 * blockDim.x * blockIdx.x;
> unsigned int i = segment + threadIdx.x;
> unsigned int t = threadIdx.x;
> input_s[t] = input[i] + input[i + BLOCK_DIM];
>  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
>     __syncthreads();
>     if (t < stride) {</pre>
>          input_s[t] += input_s[t + stride];
> if (t == 0) {
>      atomicAdd(output, input_s[0]);
> ```
> 
> Each block has its own shared memory and can independently perform the reduction. Depending on the completion order, an atomic operation to add the local result is necessary.
![Screenshot from 2026-03-26 20-21-13.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-21-13.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-21-27.png)]:**
> ## Thread Coarsening - Back Again
> 
> Thread coarsening was first analyzed in the context of matrix multiplication in GPU Performance Basics. Whenever the device does not have enough resources to execute the number of threads requested, it is forced to serialize the execution. In this case, we can serialize the work done by each thread so that no extra overhead is incurred. Another benefit to thread coarsening is improved data locality.
> 
> Successive iterations increase the amount of inactive warps. For reduction, thread coarsening can be applied by increasing the number of elements that each one processes. If the time to perform the arithmetic is much faster than the time to load the data, then thread coarsening can be beneficial. We could further analyze our program to determine the optimal coarsening factor.
> 
> ```
> _global__ coarsenedSumReductionKernel(float *input, float *output) {
> __shared__ float input_s[BLOCK_DIM];
> uint segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
> uint i = segment + threadIdx.x;
> uint t = threadIdx.x;
> float sum = input[i];
> for (uint tile = 1; tile < COARSE_FACTOR * 2; tile++) {</pre>
>     sum += input[i + tile * BLOCK_DIM];
> input_s[t] = sum;
> for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2) {
>     __syncthreads();
>     if (t < stride) {</pre>
>         input_s[t] += input_s[t + stride];
> if (t == 0)
>     atomicAdd(output, input_s[0]);
> ```
> 
> In the coarsened version, less thread communication is required since the first several steps are computed in a single thread.
![Screenshot from 2026-03-26 20-21-27.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-21-27.png)

## GPU SpMM

[https://ajdillhoff.github.io/notes/sparse_matrix_computation/](https://ajdillhoff.github.io/notes/sparse_matrix_computation/)

> **[图片提取文字 (Screenshot from 2026-03-26 20-29-32.png)]:**
> ## Coordinate List Format (COO)
> 
> This format stores non-zero elements in a 1D array of values. It also requires two 1D arrays to store the row and column indices, incurrent an overhead of 2N. The values in each array are contiguous, which is good for memory access.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 1: COO Format (Hwu, Kirk, and El Hajj 2022).
> 
> ## Kernel Implementation
> 
> ```
> __global__ void spmv_coo_kernel(COOMatrix cooMatrix, float *x, float *y) {
>     int i = blockIdx.x * blockDim.x + threadIdx.x;
>     if (i < cooMatrix.numNonzeros) {
>         int row = cooMatrix.rowIdx[i];
>         int col = cooMatrix.colIdx[i];
>         float val = cooMatrix.values[i];
>         atomicAdd(&y[row], val * x[col]);
>     }
> }</pre>
> ```
![Screenshot from 2026-03-26 20-29-32.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-29-32.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-29-39.png)]:**
> ## Evaluation
> 
> - 1. **Compaction**: Compared to representing the matrices in dense format, the COO format is very compact. However, it is not as compact as some other sparse matrix formats. It requires an additional over head of 2N elements to store the row and column indices.
> - 2. **Flexibility:** Indices and values can be easily modified in this format. This is good for applications that require frequent modifications.
> - 3. Accessibility: It is easy to access nonzero elements. It is **not** easy to access the original 0s in each row.
> - 4. **Memory access efficiency:** The values in this format are contiguous, resulting in coalesced memory access.
> - 5. **Load balance:** The data is uniformly distributed across threads, resulting in good load balance.
> 
> One major drawback, as seen in the code above, is the use of atomic operations.
![Screenshot from 2026-03-26 20-29-39.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-29-39.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-29-46.png)]:**
> ## Compressed Sparse Row Format (CSR)
> 
> The key idea of this format is that each thread is responsible for all nonzeros in a row.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 2: CSR Format (Hwu, Kirk, and El Hajj 2022).
> 
> ## Kernel Implementation
> 
> ```
> __global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float *x, float *y) {
>     int row = blockIdx.x * blockDim.x + threadIdx.x;
>     if (row < csrMatrix.numRows) {
>         float sum = 0.0f;
>         for (int j = csrMatrix.rowPtr[row]; j < csrMatrix.rowPtr[row + 1]; j++) {
>             sum += csrMatrix.values[j] * x[csrMatrix.colIdx[j]];
>         }
>         y[i] = sum;
>     }
> }</pre>
> ```
> 
> The rows are mapped to a single pointer index, so it only needs m entries to store them. The columns are not required to be in order. If the columns are in order, the data is represented in row-major order without the zero elements.
![Screenshot from 2026-03-26 20-29-46.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-29-46.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-29-54.png)]:**
> ## Evaluation
> 
> - 1. **Compaction**: The CSR format is more compact than the COO format since it only requires m entries to store the row pointers.
> - 2. **Flexibility:** The CSR format is not as flexible as the COO format. It is not easy to modify the values or indices.
> - 3. Accessibility: There is less parallelization than COO due to the row sizes.
> - 4. **Memory access efficiency:** The memory access pattern is poor since the data is separated over columns.
> - 5. **Load balance:** The load is not balanced across threads. Some threads will have more work than others, leading to control divergence.
![Screenshot from 2026-03-26 20-29-54.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-29-54.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-30-13.png)]:**
> ## **ELL Format**
> 
> ELL fixes the non-coalesced memory accesses of CSR via data padding and transposition. This is visualized below:
> 
> - 1. Start with CSR format
> - 2. Pad rows to equal size
> - 3. Store in column-major order
> 
> ![](_page_0_Figure_5.jpeg)
> 
> Figure 3: ELL Format (Hwu, Kirk, and El Hajj 2022).
> 
> ## Kernel Implementation
> 
> ```
> __global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float *x, float *y) {
>     int row = blockIdx.x * blockDim.x + threadIdx.x;
>     if (row < ellMatrix.numRows) {
>         float sum = 0.0f;
>         for (int j = 0; j < ellMatrix.nnzPerRow[row]; j++) {
>             int col = ellMatrix.colIdx[j * ellMatrix.numRows + row];
>             sum += ellMatrix.values[j * ellMatrix.numRows + row] * x[col];
>         }
>         y[row] = sum;
>     }
> }</pre>
> ```
![Screenshot from 2026-03-26 20-30-13.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-30-13.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-30-21.png)]:**
> ## Evaluation
> 
> - 1. **Compaction:** Padding the rows means this is less space efficient than CSR.
> - 2. **Flexibility:** More flexible than CSR; adding nonzeros in CSR requires a shift of values. This format can replaced a padded element if necessary.
> - 3. **Accessibility:** ELL can return the row given the index of a nonzero element as well as the nonzero of a row given that index.
> - 4. **Memory access efficiency:** Consecutive threads access consecutive memory locations.
> - 5. **Load balance:** Shares the same control divergence issues as CSR.
![Screenshot from 2026-03-26 20-30-21.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-30-21.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-30-29.png)]:**
> ## **ELL-COO Format**
> 
> ELL-COO combines the two formats to improve space efficiency and control divergence.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 4: ELL-COO Format (Hwu, Kirk, and El Hajj 2022).
> 
> ## Evaluation
> 
> - 1. **Compaction:** ELL-COO has the same compaction as ELL.
> - 2. **Flexibility:** ELL-COO is more flexible than ELL thanks to inclusion of the COO format.
> - 3. **Accessibility:** It is not always possible to access all nonzeros given a row index.
> - 4. **Memory access efficiency:** The memory access pattern is coalesced.
> - 5. **Load balance:** COO reduces the control divergence seen in ELL alone.
![Screenshot from 2026-03-26 20-30-29.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-30-29.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-30-36.png)]:**
> ## Jagged Diagonal Storage Format (JDS)
> 
> The last format we will consider is the Jagged Diagonal Storage format. This format reduces divergence and improves memory coalescing without padding. The main idea is to sort the rows by length from longest to shortest.
> 
> - 1. Group nonzeros by row
> - 2. Sort rows by length while preserving their original row indices
> - 3. Store in column-major order
> 
> ![](_page_0_Figure_5.jpeg)
> 
> Figure 5: JDS Format (Hwu, Kirk, and El Hajj 2022).
![Screenshot from 2026-03-26 20-30-36.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-30-36.png)

> **[图片提取文字 (Screenshot from 2026-03-26 20-30-45.png)]:**
> ## Evaluation
> 
> - 1. **Compaction:** Avoid paddding, so it is more space efficient than ELL.
> - 2. **Flexibility:** Less flexible than ELL since it requires sorting when adding new elements.
> - 3. Accessibility: Cannot access a row and column given the index of a nonzero element.
> - 4. Memory access efficiency: Without padding, the starting location of memory accesses in each iteration can vary.
> - 5. **Load balance:** Since the rows are sorted, threads of the same warp are likely to iterate over rows of similar length.
![Screenshot from 2026-03-26 20-30-45.png](GPU%20Memory%20Arch/Screenshot_from_2026-03-26_20-30-45.png)