# GPU Memory Manage、M1 Memory、Copy

## UVM Manage

MEMORY SYSTEM HAVING COMBINED HIGH DENSITY , LOW BANDWIDTH AND LOW DENSITY , HIGH BANDWIDTH MEMORIES：Apple内存芯片架构

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8
> 
> ![](_page_0_Figure_2.jpeg)
> 
> <u>Fig. 9</u>
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 14
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%201.png)

UNIFIED MEMORY SYSTEMS AND METHODS

cudaMallocManaged分配的统一内存，由CUDA运行时自动管理缺页和复制。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG 2
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG 3
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%203.png)

> **[图片提取文字 (image.png)]:**
> allocation call in accordance with one embodiment. In one embodiment, when an API managed pointer memory allocation call is encountered, addresses or locations within the reserved VA range are returned and a chunk 1522 from the reserved range 1511 in the GPU VA space 1510 is allocated to the managed pointer Ptr. Pages or addresses "A" from the GPU PA 1530 are allocated and mapped to GPU VA 1522 in GPU page table 1520 map entry 1521. A GPU kernel mode driver is also notified of the new managed allocation. Now the GPU side mapping is set up and a GPU kernel which accesses the allocation can use the physical memory mapped under it.
> 
> [0031] FIG. 3 is a block diagram of exemplary memory
> 
> spaces associated with an API managed pointer memory
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%204.png)

> **[图片提取文字 (image.png)]:**
> [0027] In one embodiment, a novel API managed memory allocation call triggers an automated unified memory management method. The API managed memory allocation call can instruct a driver (e.g., GPU driver, etc.) to automatically manage the memory. In one exemplary implementation, the novel API call includes a GPU cudaMallocManaged call. In one embodiment, a cudaMallocManaged call returns pointers within a reserved VA range associated with managed
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%205.png)

> **[图片提取文字 (image.png)]:**
> memory. Reserving a certain VA range for use by a pointer in multiple VA spaces ensures the pointer can be used in multiple VA spaces (e.g., CPU and GPU memory spaces, etc.). FIGS. 2 through 5 are block diagrams of exemplary memory spaces associated with an automated unified memory management process in accordance with one embodiment.
> 
> [0028] In one embodiment, regions from a CPU's virtual
> 
> address space are reserved and a similar set of regions are also reserved in a CPUs virtual address space. FIG. 2 is a block diagram of exemplary memory space reservation n accordance with one embodiment. Managed memory chunks or addresses 1511 in GPU VA 1510 and corresponding managed memory chunks or addresses 1591 in CPU VA 1590 are reserved. In one embodiment, the reserved managed memory chunks or addresses 1511 and 1591 are the same size. A pointer managed by a particular driver can be used and accessed by multiple processors because accesses to the reserved managed memory space by other "nonmanaged" pointers (e.g., pointers not managed by the particular device) is prevented. In one exemplary implementation, if code includes an allocation call associated with a non-managed pointer, (e.g., if GPU code calls cudaMalloc, if CPU code calls Malloc, etc.) the system will use or allocate a part of the VA space that has not been reserved for managed pointer memory (e.g., an address outside the reserved range is returned for allocation to the "non-managed" pointer, etc.).
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%206.png)

CPU和GPU的虚拟地址共享数据时，访问主体device或host在本地内存复制数据后进行地址映射和修改，修改完成后数据修改返回并作用于源数据。不同主体共享数据时，对共享数据的修改需要同步排队，即轮流获取写权限、修改、释放权限、更新。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG 4
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG 5
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%208.png)

> **[图片提取文字 (image.png)]:**
> spaces associated with an access call from a different entity in accordance with one embodiment. When there is an access to the same pointer Ptr from the CPU, initially there is not a CPU virtual address that maps to the pointer and a page fault is triggered. The kernel mode driver which was previously notified of the allocation handles the page fault. A physical page or address "B" is allocated from the CPU
> 
> [0032] FIG. 4 is a block diagram of exemplary memory
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%209.png)

> **[图片提取文字 (image.png)]:**
> PA 1570. The driver copies the data contents of the corresponding GPU physical page or address "A" into the CPU physical page or address "B". The CPU virtual page or address 1592 is mapped to the physical page "B" by the mapping 1581 in the CPU page table 1580. Control returns to the user code on the CPU which triggered the fault. The virtual address 1592 is now valid, and the access which faulted is retried and operations are directed to the CPU physical memory page or address "B".
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2010.png)

> **[图片提取文字 (image.png)]:**
> memory, and the CPU's virtual address mappings may be unmapped. In one exemplary implementation, data' is flushed back from CPU PA 1570 to GPU PA 1530 and the 1592 previously mapped to B in map 1581 (shown in FIG. 4) is unmapped (in FIG. 5). Data' may be the same as the data copied or moved to the CPU in FIG. 4 or data' may be the result of modification of the data by the CPU. After this point, the CPU needs to synchronize on the pending GPU work before it can access the same data from the CPU again. Otherwise the application could be accessing the same data from both the CPU and the GPU, violating the programming model and possibly resulting in data corruption. One way the page fault handler can prevent such coherency violations is by throwing a segmentation fault on CPU access to data that is potentially being used by the GPU. However, the programming model doesn't require this, and this is meant as a convenience to the developer to know when a concurrency violation occurred. There are other ways in which coherency violations can be prevented that may be part of the driver implementation.
> 
> [0034] FIG. 5 is a block diagram of exemplary memory
> 
> space associated with a launch in accordance with one
> 
> embodiment. When work is launched on the GPU, any pages
> 
> that were migrated to CPU memory are flushed back to GPU
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2011.png)

独立地址空间，显式Copy内存。统一虚拟内存的动态分配和静态分配。

> **[图片提取文字 (image.png)]:**
> [0035] The following is one exemplary utilization of two pointers and an explicit copy instruction:
> 
> ```
> global___ k(int *ptr){
>     //use ptr
> void ( ){
>     int *d_ptr, *h_ptr;
>     size_t size=100;
>     cudaMalloc (& d_ptr, size);
>     k <<<1,1>>>(d_ptr);
>     h_ptr=malloc(size);
> ```
> 
> cudaMemcpy (h\_ptr, d\_ptr, size);
> 
> //verify h\_ptr on CPU
> 
> printf("%d\n", h\_ptr[0]);
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2012.png)

> **[图片提取文字 (image.png)]:**
> [0038] The following is one exemplary utilization of a single unified pointer:
> 
> ```
> ____global____ k (int*ptr) {
> 
> //use ptr
> }
> 
> void main() {
> 
>     int *ptr;
> 
>     size_t size =100;
> 
>     cudaMallocManaged (&ptr, size);
> 
>     k<<<1,1>>>(ptr);
> 
>     cudaDeviceSynchronize ();
> 
>     printf("%d\n", ptr[0]);
> }
> ```
> 
> [0039] Alternatively, the above code can use a qualified variable rather than a dynamic allocation:
> 
> ```
> ___device___ __managed___ int foo[100];
> ___global___ k ( ){
> //use foo
> }
> void main( ) {
>     k<<<1,1>>>( );
>     cudaDeviceSynchronize ( );
>     printf ("%d\n", foo[0]);
> }
> ```
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2013.png)

## UVM一致性访问

UNIFIED VIRTUAL MEMORY MANAGEMENT IN HETE CS

UVM对共享数据的同步排队修改，严重影响处理速度，为了维护统一内存的一致性。

Fig2A对应上面的Fig4-5。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 1
> 
> ![](_page_0_Figure_3.jpeg)
> 
> FIG. 2A
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 2C
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3B
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2016.png)

相关维护一致性的统一内存访问算法。

> **[图片提取文字 (image.png)]:**
> Receive indication that data stored at a first physical memory address, to which a virtual memory address is mapped, is discardable
> 
> 402
> 
> Delete mapping between the first physical memory address and the virtual memory address
> 
> 404
> 
> Receive instruction to map the virtual memory address to a second physical memory address
> 
> 406
> 
> Map the virtual memory address to the second physical memory address, without transferring the data from the first physical memory address to the second physical memory address
> 
> 408
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 6
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2019.png)

> **[图片提取文字 (image.png)]:**
> Map the virtual memory address to
> 
> Yes
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2020.png)

## On-demand、Page-Fault、Prefetch

Unified Memory: GPGPU-Sim/UVM Smart Integration

CPU多进程，则能够缺页时切换进程。因为GPU**缺少管理多进程MMU的后台运行时**，因此缺页时不能切换channel Ctx，否则补齐的物理页没有MMU管理，无法完成缺页处理。

CPU是OS统一负责处理缺页，而GPU只有应用进程和辅助应用的运行时。

> **[图片提取文字 (image.png)]:**
> Graphics processing units (GPUs) have become more general purpose and are increasingly used for a wide range of applications. As an accelerator device, however, a conventional discrete GPU only allows access to its own device memory, which can force developers to make tradeoffs in problem size and performance to ensure that kernels fit in the device memory. This makes it very challenging and costly to run large-scale applications with hundreds of GBs of memory footprint, such as Graph Computing workloads, because it requires careful data and algorithm partitioning in addition to purchasing more GPUs just for memory capacity. To address this issue, recent GPUs support Unified Virtual Memory (UVM) [3]. UVM provides a coherent view of a single virtual address space between CPUs and GPUs with automatic data migration via demand paging. This allows GPUs to access a page that resides in the CPU memory as if it were in the GPU memory, thereby allowing GPU applications to run without worrying about the device memory capacity limit. As such, UVM frees programmers from tuning an application for an individual GPU and allows the application to run on a variety of GPUs with different physical memory sizes without any source code changes. This is good for programmability and portability.
> 
> While the feature sounds promising, in reality the benefit comes with a non-negligible performance cost. Virtual memory support requires address translation for every memory request, and its performance impact is more substantial than in CPUs because GPUs can issue a significantly larger number of memory requests in a short period of time. In addition, transferring GPU pages requires large communication overhead between the CPU and GPU over an interconnect such as PCIe and an interrupt handler invocation. Prior work reports that page fault handling latency ranges from 20µs to 50µs [4]. Unfortunately, this page-fault latency cannot be easily hidden even with thread-level parallelism (TLP) in GPUs.
> 
> Recently, Debashis explored various hardware prefetchers in the context of FPU's unified memory management [2]. His results show prefetching larger chunks of memory improves PCIe utilization and reduces transfer latency. Further, prefetched pages reduce the number of page-faults and the overhead to resolve them. To explore this design space, he developed a simulation framework, GPGPU-Sim UVM Smart [2], which provides both functional and timing simulation support for UVM.
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ## 2.1. On-Demand GPU Memory
> 
> On-demand paged GPU memory can improve performance over up-front bulk memory transfer by overlapping concurrent GPU execution with memory transfers. However, fine-grain migration of memory pages to the GPU might cause significant overheads to be incurred on each transfer rather than amortized across many pages in an efficient bulk transfer.
> 
> CPUs are able to hide the long-latency of page-faults by context switching. However, GPUs do not support context switching to operating system service routines. Thus page-faults that can be resolved by migrating a physical page from the host to the device cannot be handled in-line by the GPU compute units. Instead, the GPU's MMU (GMMU) must handle this outside of the compute unit, returning either a successful page translation request or a fatal exception. Because the GMMU handling of this page-fault actually invokes a software runtime on the host CPU, the latency of completing this handling is both long (10s of  $\mu$ s) and non-deterministic. As such, GPUs may choose to implement page-fault handling by having the GMMU stop the GPU TLB from taking new translation requests until the SW runtime has performed the page migration and the GMMU can successfully return a page translation. Under such a scenario, each individual CU could be blocked for many microseconds while its page-fault is handled, but other non-faulting compute units can continue making progress, enabling some overlap between GPU kernel execution and on-demand memory migration.
> 
> UVM Smart [2] explores two techniques that are able to hide on-demand GPU page-fault latencies rather than trying to reduce them. First, page-fault latency can potentially be hidden by not only decoupling GPU CUs from each other under page-faults, but by allowing each CU to continue executing in the presence of a page-fault. GPUs are efficient because their pipelines are drastically simplified and do not typically support restartable instructions, precise exceptions, nor the machinery required to replay a faulting instruction without side effects. While replayable instructions are a common technique for supporting long latency paging operations on CPUs, this would be an exceptionally invasive modification to current GPU designs. Instead, UVM Smart explores the option of augmenting the GPU memory system, which already supports long latency memory operations, to gracefully handle occasional ultra-long latency memory operations. Second, in addition to improving CU execution and memory transfer overlap, aggressive page-prefetching can build upon this concurrent execution model and eliminate the latency penalty associated with the first touch to a physical page.
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2022.png)

缺页处理。Prefetch减少缺页开销。

> **[图片提取文字 (image.png)]:**
> ## 2.2. **GPU Page-Fault Handling**
> 
> The previous section explained that allowing GPU compute units to execute independently and stalling execution only on their own page-faults, was insufficient to hide the effects of long latency page-fault handling. Due to the fact that the GPU compute units are not capable of resolving these page-faults locally, the GMMU must interface with a software driver executing on the CPU to resolve these faults. The architectural support for this augmentation was proposed in [4], as shown in Figure 2-1. Since this fault handling occurs outside the GPU CU, they are oblivious that a page-fault is even occurring. To prevent overflowing the GMMU with requests while a page-fault is being resolved, the GMMU may choose to pause the CU TLB from accepting any new memory requests, effectively blocking the CU. Alternatively, to enable the CU to continue executing in the presence of a page-fault, both the CU TLB and GMMU structures need to be extended with new capabilities to track and replay page translation requests once they have been handled by the software runtime, a capability refered to as "replayable faults".
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 2-1. Architectural View of GPU MMU and TLBs Implementing CU Transparent Far Page-faults
> 
> Figure 2-1 shows a simplified architecture of a GPU that supports 'replayable' page-faults. ① Upon first access to a page that is not present in GPU memory, a TLB miss will occur in the CU's local TLB structure. ② This translation miss will be forwarded to the GMMU which performs a local page table lookup. Once discovering that this page is not physically present, the GMMU would normally return an exception to the CU or block the TLB from issuing additional requests. To enable the CU to continue computation under a page-fault, the GPU's GMMU employs a book-keeping structure called 'far-fault MSHRs' to track potentially multiple outstanding page migration requests to the CPU. ③ Upon discovery that a translation request has transitioned into a far-fault, the GMMU inserts an entry into the far-fault MSHR table. ④ Additionally,the GMMU also sends a new 'Nack-Replayable' message to CU's requesting TLB. This Nack response tells the CU's TLB that this particular fault may need to be re-issued to the GMMU for translation at a later time. ⑤ Once this Nack-Replayable message has been sent, the GMMU initiates the SW handling routine for page-fault servicing by putting its page translation request in memory and interrupting the CPU to initiate fault servicing. ⑥ Once the page is migrated to the GPU, the corresponding entry
> 
> in the far-fault MSHRs is used to notify the appropriate TLBs to replay their translation request for
> 
> this page. This translation will then be handled locally a second time, successfully translated, and
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2023.png)

> **[图片提取文字 (image.png)]:**
> returned to the TLB as though the original TLB translation request had taken tens of microseconds to complete.
> 
> ## 2.3. Hardware Prefetchers
> 
> it is still difficult to completely hide the page-fault latency. Thus the total kernel execution time increases dramatically as it includes far-fault handling latency and memory copy time. cudaMem-PrefetchAsync, is an asynchronous construct in CUDA 8.0, that allows programmers to specify an address range to migrate in parallel to the kernel execution. Prefetching later referenced pages helps reduce the number of page-faults and also ensures overlap between data migration and kernel execution. However, the responsibility of what to prefetch and when to prefetch still belongs to the programmer. Zheng *et al.* [4] are the first to propose programmer-agnostic hardware prefetchers to overlap kernel execution and data migration. They introduced (i) random, (ii) sequential, and (iii) locality-aware hardware prefetchers. Debashis *et al.* [2] explore and verify a tree-based hardware prefetcher, called (iv) tree-based neighborhood prefetcher, that is implemented by NVIDIA. Hardware prefetchers take away the burden from the programmer by automatically deciding what and
> 
> when to prefetch. These hardware prefetchers are incorporated in UVM Smart.
> 
> As described in previous sections, while the CU is able to continue executing upon a page-fault,
> 
> ## 2.3.1. Random Prefetcher
> 
> A random prefetcher prefetches a random 4KB page along with the 4KB page for which the far-fault occurred in the current cycle. The prefetch candidate is selected randomly from the 2MB large page boundary to which the faulty page belongs. This not only helps CUDA workloads with random access pattern, but also selecting from 2MB large page boundary instead of the whole virtual address space helps in cases of locality of memory accesses.
> 
> ## 2.3.2. Sequential-local Prefetcher
> 
> pages from the lowest to the highest order of virtual address irrespective of page access pattern or far-faults. Their locality aware prefetcher migrates consecutive 128 4KB pages (or total 512KB memory chunk) starting from the faulty-page. Debashis *et al.* [2] propose a different variation called sequential-local hardware prefetcher. Each cudaMallocManaged allocation is logically split into multiple 64KB basic blocks. GMMU upon discovering the pages corresponding to the coalesced memory requests are invalid in the GPU page table, first calculates the base addresses of the 64KB logical chunks to which these faulty 4KB pages belong. Thus, GMMU identifies these 64KB basic blocks as prefetch candidates. Further, it divides these candidate basic blocks into prefetch groups and page-fault groups based on the position of the faulty page in the current basic block and then schedules them for sequential transfers by the PCIe interconnect. Prefetching 64KB
> 
> basic blocks ensures contiguous 16 4KB pages local to the current faulty pages. The position of a
> 
> faulty page can be anywhere within the corresponding 64KB basic block. Further, multiple faulty
> 
> Zheng et al. [4] describe their sequential prefetcher as the process of bringing a sequence of 4KB
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2024.png)

> **[图片提取文字 (image.png)]:**
> pages are taken in consideration while choosing a basic block for prefetching and can be grouped within the same 64KB boundary.
> 
> ## 2.3.3. Tree-based Neighborhood Prefetcher
> 
> The semantics of TBNp demands that every cudaMallocManaged allocation is first logically divided into 2MB large pages. Then, these 2MB large pages are further divided into logical 64KB basic blocks to create a full binary tree per large page boundary. By the definition of a full binary tree, every node has exactly 2 children nodes. The root node of each binary tree corresponds to the virtual address of a 2MB large page and the leaf-level nodes correspond to the virtual addresses of the 64KB basic blocks. If the user-specified size of an allocation is not a perfect multiple of 2MB, then the remainder allocation is rounded up to the next  $2^i * 64KB$  and another full binary tree is created.
> 
> The maximum memory capacity of a node in the full binary tree can be calculated as  $2^h * 64KB$ ,
> 
> where h is the height of a node and h = 0 at the leaf level. On every far-fault, the GMMU first identifies the 64KB basic block corresponding to the faulty page being requested. With the understanding that upon migrating, 16 pages in the basic block will be validated in the GPU page table, GMMU then recalculates the to-be valid size of its parent and grandparent up to the root node of the tree. Here and henceforth, valid size is the size of all valid pages corresponding to the leaf-nodes belonging to a given node. At any point, if GMMU discovers the to-be valid size of a node is strictly greater than 50% of the maximum memory capacity at this level, it tries to balance the valid sizes between the two children of that node. This balancing process is recursively pushed down to the children which have not reached the maximum valid size quota. This balancing act identifies basic blocks for prefetching. This process continues till no more basic blocks at leaf level can be identified as prefetch candidates and the to-be valid size of any non-leaf node including root is not more than 50% of maximum size capacity at its level.
> 
> In Figure 2-2, Tree-based Neighborhood Prefetcher is demonstrated by two examples. Both of
> 
> these examples explain the semantics on 512KB memory chunk for simplicity. These examples use  $N_h^I$  to denote a node in the full binary tree, where h is the height of the node and i is the numeric position of the node in that particular level. These examples assume initially all pages in this 512KB allocation are invalid with valid bit not set in the GPU's page table and thus every first access to a page causes a far-fault. In the first example, for the first four far-faults, GMMU identifies the corresponding basic blocks  $N_0^1$ ,  $N_0^3$ ,  $N_0^5$ , and  $N_0^7$  for migration. As the first byte of every basic block is accessed, the basic blocks are split into 4KB page-fault groups and 60KB prefetch groups. All memory transfers are serialized in time. After these first four accesses, each of nodes  $N_0^1$ ,  $N_0^3$ ,  $N_0^5$ , and  $N_0^7$  has 64KB valid pages. Then, GMMU traverses the full tree to update the valid page size for all the parent nodes and thus each node at h = 1 ( $N_0^1$ ,  $N_1^1$ ,  $N_1^2$ , and  $N_1^3$ ) has 64KB valid pages. When the fifth access occurs, GMMU discovers that  $N_1^0$  and  $N_2^0$ will have 128KB and 192KB valid pages respectively. For  $N_2^0$ , the to-be valid size is greater than 50% of the maximum valid size of 256KB. Hence, the right child  $N_1^1$  is identified for prefetching. This decision is then pushed down to the children. This process identifies the basic block  $N_0^2$  as a prefetch candidate. Further, GMMU discovers that after prefetching  $N_0^2$ ,  $N_3^0$  will have 320KB of valid pages which is more than 50% of the maximum valid size of 512KB. Then, node  $N_3^0$  pushes
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2-2. Demonstration of TBNp on 512 KB memory chunk for two different page access patterns.
> 
> prefetch request to the node  $N_2^1$  which in turn pushes it to its children. This process identifies basic blocks  $N_0^4$  and  $N_0^6$  for further prefetching.
> 
> In the second example, the first two far-faults cause migration of basic blocks  $N_0^1$  and  $N_0^3$ . GMMU traverses the tree to update the valid size of nodes  $N_1^0$  and  $N_1^1$  as 64KB each. At the third far-fault, as basic block  $N_0^0$  is migrated, the estimated valid sizes for nodes  $N_1^0$ , and  $N_2^0$  are updated as 128KB and 192KB respectively. As the valid size of  $N_2^0$  is more than 50% of the maximum valid size of 256KB,  $N_0^2$  is identified for prefetching. After this point, the  $N_2^0$  is fully balanced and both  $N_2^0$  and  $N_3^0$  have exactly 256KB of valid pages. On fourth access, GMMU discovers that the valid size of  $N_3^0$  will be 320KB which is more than 50% of the maximum memory size it can hold. This imbalance causes prefetching of nodes  $N_0^5$ ,  $N_0^6$ , and  $N_0^7$ . Note at this point as GMMU finds four consecutive basic blocks, it groups them together to take advantage of higher bandwidth. Then, based on the page-fault, it splits this 256KB into two transfers: 4KB and 252KB. An interesting point to observe here is that for a full binary tree of 2MB size, TBNp can prefetch at most 1020KB at once in a scenario similar to the second example.
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2026.png)

## Copy Engine

NVIDIA QUADRO DUAL COPY ENGINES

COPY ENGINE AND A METHOD FOR DATA MOVEMENT

CPU将应用Space的数据Copy到driver space并转换，之后下载到GPU VRAM组成overlap。

> **[图片提取文字 (image.png)]:**
> ## **CURRENT STREAMING APPROACHES**
> 
> A typical download-process-readback pipeline can be broken down into the following:
> 
> - ► **Copy** involves CPU cycles in data conversions if any to native GPU formats and memcpy from the application memory space to the driver space.
> - ▶ **Download** the time for the actual data transfer on PCI Express from host to device.
> - ▶ **Process** GPU cycles for rendering and compute.
> - ▶ **Readback** time for the data transfer from device back to host.
> 
> To achieve maximum end-to-end throughput on the GPU, maximum overlap is required between these various components in the pipeline.
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ## Synchronous Downloads
> 
> The straightforward download method for textures is to call <code>glTexSubImage</code> which involves and blocks the CPU while copying data from user space to the driver space and subsequent data transfer on the bus to the GPU. Figure 2 illustrates the inefficiency of this method as the GPU is idle while the CPU is busy with the <code>memcpy</code>.
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Figure 2. Synchronous Downloads With No Overlap
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ## CPU Asynchronous Downloads with PBOs
> 
> The OpenGL PBO [2] mechanism provides for transfers that are asynchronous on the CPU. If an application can schedule enough work between initiating the transfer and actually using the data, CPU asynchronous transfers are possible. In this case, the glTexSubImage call operates with little CPU intervention. PBOs allow direct read/write into GPU driver memory eliminating need for additional memcpys. The CPU after the copy operation does not stall while the transfer takes place and can continue on to process the next frame. However, downloads and uploads still involve GPU context switch and cannot be done in parallel with the GPU processing or drawing. Multiple PBOs can potentially speed up the transfers. A ping pong version is shown in Figure 3.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 3. CPU Asynchronous Downloads with Ping Pong PBOs
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## GPU ASYNCHRONOUS TRANSFERS WITH QUADRO DUAL COPY ENGINES
> 
> The copy engine featured in Quadro solutions provides real GPU-asynchronous texture downloads. Texture data can be downloaded or uploaded in parallel with 3D rendering. As shown in Figure 4, supported Quadro solutions add an additional DMA engine making it now possible to overlap download, processing, and readback. To take advantage of this, one thread (channel) is used for rendering, one is used for download and the third is used for upload, and all transfers are done via PBOs. When partitioned this way, the render thread will run on the graphics engine and the transfer threads on the copy engines in parallel and completely asynchronous. These are fully functional GL contexts so that non-DMA commands can be issued in the transfer threads but will time slice with the rendering thread. Copy engines can also handle format conversions and swizzling for same data types without CPU intervention, in contrast to previous hardware constraints where the input data formats had to be GPU native. Figure 5 shows the end-to-end frame time amortized over 3 frames for a time sequence. It is seen how the current frame download (t1) is overlapped with render of previous frame (t0) and CPU memcpy of next frame (t2).
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 4. Quadro Dual Copy Engine Block Diagram and Application Layout
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2030.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 5. GPU Asynchronous Transfers with Dual Copy Engines
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2031.png)

> **[图片提取文字 (image.png)]:**
> FIG. 1
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2032.png)

> **[图片提取文字 (image.png)]:**
> FIG. 2A
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](GPU%20Memory%20Manage%E3%80%81M1%20Memory%E3%80%81Copy/image%2033.png)