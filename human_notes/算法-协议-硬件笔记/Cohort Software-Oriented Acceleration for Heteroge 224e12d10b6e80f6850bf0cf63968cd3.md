# Cohort: Software-Oriented Acceleration for Heterogeneous SoCs

异构SoC视为多核SoC的扩展，核扩展到CPU和各种加速器，Cache一致性扩展到**CPU Cache和queue的一致性**；

利用SPSC queue作为软件和硬件的**公共模型**，软件接口push、pop进行数据流端到端的输入和输出，register注册queue来定义硬件数据流；

将SPSC queue的多核一致性扩展到加速核，利用invalidate完成硬件数据流的数据传输；

**一致性**是每个核对系统造成的影响，能够被其他核感知，让系统内所有核对系统的认知一致（对相同事物的感知一致）；

# Abstract

异构设计：**硬件最大化性能，软件弥补其余**；

软件编程缺少数据格式（layout？）、手动存储、一致性管理等？

软件使用共享内存queue的抽象和硬件交互，硬件运行时重配置；

> **[图片提取文字 (image.png)]:**
> Philosophically, our approaches to acceleration focus on the extreme. We must optimise accelerators to the maximum, leaving software to fix any hardware-software mismatches. Today's software abstractions for programming accelerators leak hardware details, requiring changes to data formats and manual memory and coherence management, among other issues. This harms generality and requires deep hardware knowledge to efficiently program accelerators, a state which we consider hardware-oriented.
> 
> This paper proposes Software-Oriented Acceleration (SOA), where
> 
> software uses existing abstractions, like software shared-memory queues, to interact with accelerators. We introduce the Cohort engine which exploits these queues' standard semantics to efficiently connect producers and consumers in software with accelerators with minimal application changes. Accelerators are even usable in chains which can be runtime reconfigured by software. Cohort significantly reduces the burden to add new accelerators while maintaining system-level guarantees. We implement a Cohort FPGA prototype which supports SOA applications running on multicore Linux. Our evaluation shows speedups for Cohort over traditional approaches ranging from 1.83× to 8.38× over MMIO, and from 1.69× to 11.24× for DMA baselines. Our software-oriented batching optimisations within Cohort also improve performance from 2.32× to 8.10×, demonstrating the power of SOA.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image.png)

# intro

异构SoC中，加速器为CPU服务，CPU管理内存分配、调度和通信；

但是，SoC的需求由加速器完成，**加速器有特殊需求和定制的编程范式**，缺乏统一性的加速器需求，需要为每个加速器编写定制代码；

并且，加速器的组合需要特权软件层（**CPU**）来做中间通信；

> **[图片提取文字 (image.png)]:**
> To meet users' efficiency and performance demands, systems-onchip (SoCs) feature many efficient, specialised accelerators alongside general-purpose cores [16, 21, 37, 39, 62, 81]. The architecture of these SoCs suggests the accelerators are subservient to the generalpurpose cores, which manage their memory allocation, scheduling, and communication. One could thus view this as a softwareoriented SoC rather than an accelerator- or hardware-oriented one. However, we argue that the system is driven by the demands of accelerators, which have special needs and frustratingly bespoke programming paradigms, lacking commonality [4, 23, 31, 46, 52, 61, 73] and requiring more code with each additional accelerator. Further, the hardware accelerators cannot be easily composed together since layers of privileged software must mediate communication. If heterogeneity is to further grow, we must find more efficient means for management of and communication with hardware accelerators.
> 
> Due to accelerators' piecemeal adoption, their software and hard-ware interfaces are routinely afterthoughts. There is little consideration for hardware or software approaches which are common across accelerators and smooth adoption. An <u>ideal approach would</u> feature:
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%201.png)

理想异构SoC需要：

**通用、灵活的硬件接口**，完成软件和硬件之间的通信；

使用**现有硬件抽象**的编程范式，新增加速器时的**最小化软件修改（底软和编译）**；

> **[图片提取文字 (image.png)]:**
> across accelerators and smooth adoption. An <u>ideal approach would</u> <u>feature:</u>
> 
> • A common, flexible hardware interface to efficiently com-
> 
> municate between software and hardware elements.
> A software paradigm using existing abstractions to minimise software modification to add new accelerators.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%202.png)

尽可能复用**现有底层软件**环境，**不希望新ISA和硬件接口**来增大底软和编译环境的设计难度；

## Fig 1

**Cohort：共享内存queue**的抽象、Core和Acc的点对点通信 ；

面向使用SBIO（stream、buffer）通信模式的加速器，DMA、MMIO属于SBIO；

[内存访问、地址空间、PCIe](%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE%E3%80%81%E5%9C%B0%E5%9D%80%E7%A9%BA%E9%97%B4%E3%80%81PCIe%20264e12d10b6e80138167c60d97b8e20d.md)

软件产生数据**入队**，被硬件、软件（没有特殊内存分配和cache一致性管理）**使用**，加速器无需修改而通过**AXI**接口接入；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Proposed Cohort System Design
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%203.png)

> **[图片提取文字 (image.png)]:**
> We introduce an approach, termed **Software-Oriented Acceleration**, to emphasise that accelerator communication, isolation, and memory management respect existing software paradigms. Our chosen software abstraction is shared-memory queues, with which we reduce hardware and software overhead and ease accelerator adoption. Our system design (Figure 1) has cores and accelerators communicating arbitrarily as peers, forming a **Cohort**. This enables accelerator composition without software imposition, a pattern which has been underutilised to date.
> 
> The Cohort SOA model focuses on accelerators which follow a stream/buffer in and stream/buffer out (SBIO) communication pattern. Many accelerators follow this model, using memory-mapped I/O (MMIO) or direct memory access (DMA) to consume and produce buffers or streams of data [26, 43, 47, 52, 75]. For accelerators with broader communication patterns than SBIO (e.g., GPGPUs or sparse graph and neural network accelerators [7, 39, 59, 62, 72]), Cohort can enable more efficient communication for the use cases which are SBIO.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%204.png)

> **[图片提取文字 (image.png)]:**
> Cohort provides a common, standard queue interface both to software and to hardware, enabling reuse both of existing multithreaded software and of existing hardware accelerators with little to no modification. To bridge between softwarefriendly shared-memory queues and hardware-friendly interfaces, we created the Cohort engine. With it, existing software produces data into queues for consumption by hardware or software without special memory allocation routines, cache coherence management, etc. Likewise, accelerators are connected unmodified with existing, high-performance, latency-insensitive [12, 13, 71] hardware interfaces including AXI-Stream [3].
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%205.png)

## Fig 2

Cohort engine在SoC上实现，基于**原生一致性协议**（CXL等）；

FPGA上异构原型（core、加速器、Cohort engine互联）

**consistency、Coherence**：多核对相同内存读写时，保证行为正确性需要的协议；

> **[图片提取文字 (image.png)]:**
> We implement Cohort on an SoC using the native coherence protocol to enable efficient operation. Cohort's primitives could also be implemented atop emerging cache coherent interconnects beyond the SoC scale, e.g. CXL, CCIX, and CAPI [15, 70, 74]. We have implemented a software-oriented SoC prototype on FPGA with RISC-V cores booting Linux and accelerators connected via Cohort engines (shown in Figure 2. Our evaluation tests lightly modified software running on Linux, communicating between accelerators and software threads with no consideration for the nature of the producer or consumer on the other side of the queues.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%206.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Cohort FPGA prototype with AES and SHA accelerators
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%207.png)

# motivation

软件编程问题

加速器的**软件接口**：编译pass、DSL、特权kernel call、自定义库；

加速器要求特殊**memory分配、cache flush**，因为需求IOMMU、一致性、编址、对齐；

MMIO的ld/st操作**语义开销大**，需要映射到kernel虚拟内存，损伤性能；

MMIO执行**无法预测（kernel执行）**，MMIO期间通常**无法处理中断**，MMIO一般**阻塞CPU**/同步；

编程者使用MMIO会担忧编译优化或重排指令引发的问题，**MMIO的性能和可编程性弱**，因此**只作为备选**传输方法；

> **[图片提取文字 (image.png)]:**
> Accelerator vendors provide varying software interfaces [29, 30, 36, 45, 49]. Some provide compiler passes or domain specific languages, some call into privileged kernel modules, and others adopt vendor-specific libraries, making for a fragmented software environment. To exploit multiple vendors' accelerators becomes difficult, requiring a variety of bespoke expertise. This results in fragile software, as vendors' requirements conflict, if composition is at all possible.
> 
> We have also been challenged by memory allocation and management for accelerators. It is routine for new accelerators to require special memory allocations or the addition of manual cache flushing [52, 59, 74]. These requirements come for a variety of reasons, including accelerators' use of IOMMUs, differing coherence models, and special addressing and alignment constraints. As a result, the task of complexity management falls to software.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%208.png)

> **[图片提取文字 (image.png)]:**
> In existing SoCs which use MMIO to configure and interact with accelerators and I/O devices, MMIO loads and stores have special semantics which hurt performance. MMIO operations have side effects, meaning that they cannot be performed speculatively and that interrupts generally should not be taken while they are outstanding in the memory system. Many cores, particularly slim in-order ones, will simply stall on these until a load or store returns from the accelerator or device. This behaviour is often relied on for synchronisation, with the return guaranteeing that an operation has completed. In terms of programmability, performing these MMIO operations from user mode is fraught with problems to ensure that operations are not optimised out or reordered by the compiler. We argue that MMIO is undesirable for performance and programmability and believe this is backed up by other recent approaches in, for example, high-performance networking, where shared-memory queues have replaced such operations [57, 80].
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%209.png)

硬件接口问题

加速器、core的**数据传输**：MMIO读写寄存器、DMA读写内存；

加速器的**一致性模型**：部分一致性、不支持、全一致性均有，全一致性的代价是性能；

桌面、服务级的基于PCIe的CXL、CCIX、CAPI：基于**PCIe传输有开销、延迟**（90%），不利于LC；

**CCIX**将一致性管理卸载到加速器负责，对加速器管理的内存区域的**访问绑定特定的PCIe拓扑**；

移动SoC很少使用PCIe（功耗、面积高），**加速器使用原生/共享memory**；

> **[图片提取文字 (image.png)]:**
> It is now essential to optimise the data-movement interface, both between cores and accelerators and among accelerators. Coarsegrained accelerators in SoCs today are connected in various ways [14, 39, 59]. Some act as MMIO devices, with private registers or memories that are filled prior to invocation, and read out upon completion. Others use DMA engines, programmed prior to invocation, to fetch data from and store data to memory.
> 
> Accelerators also vary in their coherence models, often operating partially- or non-coherently, which complicates software. While fully-coherent accelerators are convenient, they have costs from participating in cache-coherence. Balancing these models is sufficiently complex that research has even been done on machine learning techniques to choose between them [83].
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2010.png)

> **[图片提取文字 (image.png)]:**
> In desktop- and server-class settings with PCIe, protocols like CCIX, CXL, and CAPI [15, 70, 74] enable accelerator cache coherence. However, building on PCIe brings overhead and latency, meaning applications with strict latency requirements usually perform poorly. Neugebauer et al. [58] characterised end host networking and found that for a 128B payload, 90% of the observed 1000ns round-trip latency comes from PCIe itself. In addition, coherence systems like CCIX can offload coherence management to accelerators, meaning coherent access to accelerator-managed memory regions is PCIe topology dependent, which further complicates the system [15]. Many mobile-class SoCs limit use of PCIe due to its power and area, relying instead on native (shared) memory for many of their accelerators and peripherals [54, 55, 60, 65, 66].
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2011.png)

# Cohort方法

软件中心的架构SOA，不需要采用特殊IO操作、内存和cache管理、加速器定制驱动；

SOA以user为中心，向app和runtime提供灵活、隔离、安全；

SOA使用**kernel interposition？**的机制来支持system level policy、profiling，**不支持损伤性能的context switch、kernel interaction？**；

## Q-A

**kernel interposition（介入）**是软件提供起始输入、接收最终输出、配置硬件数据流而中间无交互；

**kernel interaction（交互）**是软件调用kernel-硬件执行kernel的多次交互；

> **[图片提取文字 (image.png)]:**
> munication and application design. Software is routinely optimised in unforeseen ways to drive performance and efficiency. Using programming languages' and libraries' existing abstractions, we enable new avenues for optimisations across many hardware accelerators. We need not adopt special I/O operations, memory and cache management, or complex accelerator-specific drivers, unlike existing approaches [52, 59].
> 
> At a high level, SOA reuses existing software mechanisms for com-
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2012.png)

> **[图片提取文字 (image.png)]:**
> SOA is user mode-oriented to give applications and language runtimes greater flexibility, isolation, and security. Mechanisms which provide kernel interposition are preferred as they enable enforcing system-level policy, profiling, etc. However, encouraging user-level operation should not fundamentally require context switches and kernel interaction for operation, which hurt performance.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2013.png)

## Fig 3

queue Coherence：通用抽象，**拆解线程kernel**，增加并行和tuning；

> **[图片提取文字 (image.png)]:**
> our heterogeneous SoC, as shown in Figure 1. In high-performance parallel software, shared-memory queues enable the decoupling of threads to gain parallelism [53, 76]. We work to provide a common producer-consumer communication abstraction in order to improve programmability and composition, and to enable new performance tuning opportunities. Cohort users replace existing
> 
> In Cohort, we adopt shared-memory queues as the lingua franca of
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2014.png)

> **[图片提取文字 (image.png)]:**
> queue-decoupled software threads with accelerators while maintaining their existing code and its chains of queues connecting producers and consumers.
> 
> Many parallelisation techniques rely on the queue-based producer
> 
> consumer design pattern for communication and decoupling [33, 37, 38, 51, 61, 69, 78]. Producers and consumers maintain a common queue for communication of data and synchronisation. When available, the producer hands data to the consumer by pushing it into the queue. When a consumer is able to process more data, it pops it from the queue. The producer-consumer design pattern is used in both shared-memory multithreaded applications and for inter-process communication by sharing a small memory region between the processes for the queue. Figure 3 shows a simple example code snippet following the generic queue API shown in Table 1.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ```
> fifo_t *fifo = fifo_init(...);
> pid_t pid = fork();
> 3 if (pid == 0) { // producer
> push(element, fifo);
> 5 | } else { // consumer
> element = pop();
> printf("element: %d\n", element);
> ```
> 
> Figure 3: Generic producer-consumer code snippet
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2016.png)

## Tab 1

lock-free queue lib：减小一致性开销；

**SPSC queue** **coherence**：producer数据入队，consumer看到写指针改变时，queue中数据也改变；

基于cache coherence system的**cohort engine**实现queue coherence；

> **[图片提取文字 (image.png)]:**
> Much effort has gone into developing high-performance lock-free queue libraries. These libraries provide performance by minimising coherence effects and taking advantage of the memory-level parallelism of modern cores. With their widespread use, these queues' semantics are well understood and ready to be exploited in Cohort. We focus here on lock-free single producer-single consumer (SPSC) queues, which are supported in widely-adopted software libraries [11] and further boost performance.
> 
> We introduce the term Queue Coherence to refer to the semantic behaviour of SPSC queues. In existing libraries, there is an agreed meaning to enqueue/push and dequeue/pop with respect to both cache coherence and memory consistency. When the producer completes an enqueue and the consumer observes a change in the queue's write pointer, there is a guarantee that the consumer will also observe the updated data items in the queue. Cohort exploits queue coherence by implementing queue operations in an engine connected to the cache coherence system. This enables software to operate as designed while feeding accelerators with the highest performance and efficiency that general-purpose cores offer.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ## **Table 1: Cohort API Listing**
> 
> ```
> Existing generic SPSC queue API calls\nint fifo_init(int element_size, int queue_length);
> void push(int element, fifo_t *q);\nint pop(fifo_t *q);\nint fifo_deinit(fifo_t *q);
> ```
> 
> ## Additional Cohort-specific API calls
> 
> int cohort\_register(int acc\_id, fifo\_t \*acc\_in, fifo\_t \*acc\_out);\nint cohort\_unregister(int acc\_id, fifo\_t \*acc\_in, fifo\_t \*acc\_out);
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2018.png)

## Fig 4

软件、硬件的拆解

Cohort接口的加速器**输入连接软件queue，输出连接软件queue**；

软件只看到数据在queue中的移动，硬件在shared memory中读写数据；

> **[图片提取文字 (image.png)]:**
> Cohort enables replacement of a software thread with a Cohort engine to enable transparent acceleration, as Figure 4 shows. This "accelerator thread" communicates with our software threads in the same way that another software thread would. Instead of leaning on complex accelerator management code and driver support, offloading computation to a Cohort-enabled accelerator is as transparent as pushing data into the software queue connected to the accelerator's input. To receive results back, the software thread simply pops data from another software queue connected to the accelerator's output.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/f1c7b00b-2a96-4635-aff8-b036c51cb3d5.png)

> **[图片提取文字 (image.png)]:**
> ```
> fifo_t compute_fifo = fifo_init(...);
> fifo_t result_fifo = fifo_init(...);
> cohort_register(acc, compute_fifo, result_fifo);
> push(acc_in_elem, compute_fifo);\nint acc_out_elem = pop(result_fifo);
> printf("element is %d\n", acc_out_elem);
> ```
> 
> Figure 4: Producer-consumer code snippet for Cohort with a single accelerator
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2019.png)

## Fig 5

**多硬件协作**完成算子是queue的读写数据流，**软件只考虑数据在queue之间的移动**，不关心硬件；

编程 ease：增加queue registration routine（不改底层内存分配），软件queue位于片上Cohort；

通过**软件queue的运行时配置**，完成多硬件协作的运行时重配置；

> **[图片提取文字 (image.png)]:**
> In addition, the interoperability brings the benefit of transparent accelerator chaining, enabling chaining of a series of computations through several accelerators, whilst being transparent to software. If we have a hashing and an encryption accelerator, then to perform an encryption followed by a hash, we only need the code snippet
> 
> in Figure 5.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/197bcda4-d5b6-4e89-8276-d22b40f2f322.png)

> **[图片提取文字 (image.png)]:**
> Our approach, introducing Cohort, provides this exact ease of programming. With a minor addition in the form of the queue registration routine (which does not modify the underlying memory allocation and could be incorporated into the queue library), software queues are connected to Cohort units on-chip, ready for production and consumption by accelerators. This enables runtime reconfiguration of the hardware, with accelerator chains created dynamically by software. The programmer can maintain their highlevel producer-consumer abstraction while incrementally moving functionality to hardware accelerators.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ```
> fifo_t encrypt_fifo = fifo_init(...);
> fifo_t hash_fifo = fifo_init(...);
> fifo_t result_fifo = fifo_init(...);
> cohort_register(encrypt_acc, encrypt_fifo, hash_fifo);
> 5 | cohort_register(hash_acc, hash_fifo, result_fifo);
> push(data, encrypt_fifo);\nint chain_result = pop(result_fifo);
> ```
> 
> ## Figure 5: Producer-consumer code snippet for Cohort accelerator chaining
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2021.png)

# Cohort实现

启动**SMP**（对称多处理器）

> **[图片提取文字 (image.png)]:**
> This section explains the implementation of Cohort and its integration into the SoC as well as the library and operating system support we developed to enable booting SMP Linux on the Cohort SoC on FPGA.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2022.png)

[UMA、NUMA、MPP、CMP；AMP、SMP、BMP辨析](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90%2026fe12d10b6e801ba9bcfa11b4a7198c.md)

编程模型支持：Cohort driver、Cohort MMU、虚存上连续-串行访问、多核性能优化（大页）

**queue 描述符**

**注册**queue：data structure（fifo_t）配置Cohort engine；

**queue属性**：

fifo-base-adr：fifo在内存中的基地址；

fifo-element-size：每个数据元素的大小（block）；

fifo-length：fifo的长度；

read/write pointers：用于读写的指针（队列头尾）；

> **[图片提取文字 (image.png)]:**
> Cohort is designed for use from user mode with minimal operating system involvement. There is a single Cohort driver to support all Cohort-enabled accelerators, regardless of the variety of accelerators available. This is in contrast to many existing accelerator drivers which grow the size and complexity of our trusted software over time.
> 
> To support maximally transparent use of SPSC queues, the Co-
> 
> hort engine features an MMU compatible with the cores' ISA's MMU. In a modern SoC, such an MMU is low cost and its benefits are significant, as we show in Section 6. Queues are virtually contiguous and are accessed sequentially, which significantly improves TLB hit rates. Beyond this, Cohort benefits from the very same optimisations as software programmers adopt for multicore performance. E.g. If the programmer adopts huge pages in their queue library, the Cohort MMU will take advantage of them and see performance improvements just as cores would.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2023.png)

> **[图片提取文字 (image.png)]:**
> 4.1.1 Queue Descriptors. A common goal for high-performance software is to choose a queue structure that minimises coherence traffic. Software queues are thus organised in a variety of ways that Cohort must support. To register a queue with Cohort, its structure must be described to properly configure the Cohort engine. For Cohort to achieve broad compatibility, desirable features of SPSC queues include configurable element size, configurable queue size, and use of read and write indices versus pointers.
> 
> To support a range of queue formats, we develop a queue descriptor struct which the queue library developer uses to describe their queue. The descriptor also contains (virtually addressed) pointers to the queue elements in question, such as the read or write index. At present, we support the following attributes for the queue.
> 
> - write\_pointer/index
> - read\_pointer/index
> - fifo\_base\_address
> - fifo\_element\_size
> - fifo\_length
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2024.png)

## Fig 7

用户态API

**fifo_t**结构初始化queue，**cohort_register**将queue连接到加速器acc，push、pop对queue操作，**cohort_unregister**结束acc对queue的使用；

Cohort kernel **driver**：cohort_register、cohort_unregister；

> **[图片提取文字 (image.png)]:**
> ing the Cohort user-mode programming interface. The programmer starts by allocating two cohort-enabled queues with queue descriptors in their fifo\_t structures from their queue library.
> 
> The programmer then registers the two queues connected to a
> 
> 4.1.2 *User-Mode API.* Figure 7 shows a code snippet demonstrat-
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2025.png)

> **[图片提取文字 (image.png)]:**
> Cohort-enhanced accelerator by calling the cohort\_register system call with the ID of the accelerator to be used plus its input and output queues. From that point, the standard push and pop functions from their queue library are used to feed data to and from their accelerator. At the end of the application, they then call cohort\_unregister to end their use of the queues with the accelerator by again providing its ID and input and output queues.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2026.png)

> **[图片提取文字 (image.png)]:**
> ```
> fifo_t *cohort_to_sw = fifo_init(...);
> cohort_register(acc_id, sw_to_cohort, cohort_to_sw);
> push(0xcafedeed, sw_to_cohort);\nuint64_t result = pop(cohort_to_sw);
> cohort_unregister(acc_id, sw_to_cohort, cohort_to_sw);
>                               Figure 7: Cohort API usage example
> ```
> 
> fifo\_t \*sw\_to\_cohort = fifo\_init(...);
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2027.png)

> **[图片提取文字 (image.png)]:**
> From the programmer's perspective, the programming experience is extremely close to programming for an everyday multithreaded environment. Unlike existing approaches which either expose complexity to the programmer or try to hide it in allocation routines and syscalls, Cohort-enabled library code continues nearly unmodified. The queue library developer simply describes their queue via a Cohort queue descriptor to make the queue usable by the programmer, with no need for specialised memory allocation or deallocation routines. The only addition are the cohort\_register and cohort\_unregister syscalls provided by the Cohort kernel driver. A generic representation of the existing queue APIs on which Cohort can be built is shown in Table 1 alongside the two new API calls required for Cohort.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2028.png)

**Cohort queue library**：基于现有SPSC queue lib的C实现，也能基于C++ Boost Lockfree库；

> **[图片提取文字 (image.png)]:**
> Queue Library Support. Thanks to Cohort's flexible queue descriptors, it is reusable in a variety of queue settings. As noted in Section 3.2, SPSC queues are commonly adopted in software libraries. We designed Cohort's hardware based on established queue implementations in software and thus have seen remarkably straightforward development and integration. While our evaluation uses a hand-rolled C implementation, we have also demonstrated Cohort's cohesive integration with a high-level software library by implementing support in the C++ Boost Lockfree library to communicate with Cohort-enabled accelerators [11].
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2029.png)

**锁自由lock-free**是多线程函数的**属性**/特点，**无锁lock-less**是多线程函数的**实现**规则/约束；

[https://zhuanlan.zhihu.com/p/342921323](https://zhuanlan.zhihu.com/p/342921323)

> **[图片提取文字 (image.png)]:**
> lock-free 和 无锁 两个概念,前者是对代码性质的描述,后者 是说代码如何实现。那么,什么是 lock-free ——
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2030.png)

> **[图片提取文字 (image.png)]:**
> 对于一个并发实现,无论当前处于什么状态,只要运行足够长 的时间,至少有一个 process 能取得进展或完成其操作,则 其实现称之为 lock-free (这里的 process 代表并发操作中一 条独立的逻辑流,可以是线程,也可以是进程)。这个描述并
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/d443ff31-8b58-48b3-bcd5-fb85f23445f9.png)

> **[图片提取文字 (image.png)]:**
> MakeProgress
> 
> MakeNoProgress
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2031.png)

> **[图片提取文字 (image.png)]:**
> 在这个基础上,很容易理解所有基于『锁』的并发实现,都不 是 lock-free 的,因为它们都会遇到同样的问题 —— 如果我 们永久暂停当前占有锁的某一个线程 / 进程的执行,将会阻塞 其他线程 / 进程的执行。而对于 lock-free 实现,允许部分 process 饿死但保证整体逻辑的持续前进。
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2032.png)

> **[图片提取文字 (image.png)]:**
> MakeProgress MakeNoProgress
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2033.png)

[https://zhuanlan.zhihu.com/p/53012280](https://zhuanlan.zhihu.com/p/53012280)

lock-free一般通过**原子操作**实现（如CAS），存在**ABA**问题

> **[图片提取文字 (image.png)]:**
> ## 3 一个 stack 的 lock-free C++ 实现以及 ABA 问题<sup>↑</sup>
> 
> 一般情况下,实现一个 lock-free 算法需要系统提供一个 atomic RMW (read-modify-write) 操作。常用 RMW 操作 包括 test-and-set, fetch-and-add, compare-and-swap (CAS<sup>+</sup>) 以及更进一步的 LL / SC , 在 C++ 11 中的 atomic 库 中有许多类似的操作。大多 lock-free 实现都是使 用 CAS 来实现
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2034.png)

> **[图片提取文字 (image.png)]:**
> 但是上面代码还存在一个使用 CAS 操作时非常经典的问题, 就是 ABA 问题 —— 我们常常使用 CAS 时会假设,如果 CAS 成功则代表**事物没有任何变化**。但是有时候这种假设是错的, 比如线程 1 到达值的 CAS 操作时,线程 2 开始执行了一系列 操作 1) 修改值 2) 执行其他操作 3) 将值修改为原值, 线程 1继续执行时发现值没有变换,然后 CAS 成功。这里虽然 CAS 成功,但是实际上线程 2 已经做了很多事情,如果我们 做了没有变化的假设,那么将会发生非预期行为,比如说对于 上述 stack 实现
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2035.png)

## Fig 6

基于**queue一致性**，Cohort Engine将软件线程安全queue操作转为对硬件queue的FIFO/AXI操作；

Cohort Engine属于**cache一致性**系统，处理一致性细节，对SBIO的加速器**透明**；

**架构**：P-Mesh（一致性基系统）、Cohort Engine（Register、MTE、EndPoint）、Accelerator；

**Uncached Register**用于CPU通过MMIO配置；

Producer、Consumer **EndPoint**生成queue操作，endpoint将软件数据结构转为加速器数据流；

**MTE（MMU）**将EndPoint的高层内存操作转为一致性物理地址；

> **[图片提取文字 (image.png)]:**
> Thanks to queue coherence, the Cohort engine bridges from software level thread-safe queue operations to simple hardware queues with latency insensitive interfaces, including standard valid/ready FIFOs and AXI Stream [3, 12, 13, 71]. The designer chooses where to place the Cohort engine in their cache coherence system, how to optimise it for their specific needs, and uses it to hide coherence system details from the accelerators. For SBIO accelerators, Cohort does not require the accelerator to have any knowledge of cache coherence. Instead, the accelerator connects to two latency-insensitive endpoints provided by the Cohort engine, for consumption of input and production of results. When data is available, it is provided by the Cohort engine and otherwise the accelerator avoids polling. As shown in Figure 6, Cohort is separated into several functional
> 
> units, each handling a specific task. Our prototype Cohort engine
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2036.png)

> **[图片提取文字 (image.png)]:**
> is integrated into the P-Mesh cache coherence system of Open-Piton [8, 10], which is an open-source tile-based SoC framework. The components of P-Mesh that Cohort directly interacts with are shown in orange at the top of the figure. Cohort itself is rendered in blue in the centre of the figure, while the connected accelerator
> 
> is shown at the bottom in green. Cohort is integrated on its own
> 
> tile within the SoC: a zoomed out view of a complete SoC is shown
> 
> in Figure 2.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/9f8bab06-d12d-4d8a-9379-4f7907ed5fbc.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6: Cohort Engine Architecture
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2037.png)

> **[图片提取文字 (image.png)]:**
> The Cohort Engine has a few main components: uncached configuration registers, the Memory Transaction Engine (MTE) which contains the MMU, a Consumer endpoint, and a Producer endpoint. CPU cores may configure Cohort through its uncached configuration registers, which are the only MMIO component of Cohort. The producer and consumer endpoints are responsible for generating queue operations. When the endpoints need to communicate with memory, the MTE translates from higher level memory operations and invalidations to physically addressed coherence operations. The endpoints effectively translate the software queue structure into a data stream for the connected accelerator to handle, with a
> 
> hardware-friendly interface.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/97984766-d172-43bb-b64c-5b63f8734db1.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Cohort FPGA prototype with AES and SHA accelerators
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%207.png)

**consumer**将数据输入acc：queue注册后，consumer追踪queue的**读写指针**，queue写指针被producer改变后，consumer收到**invalidation**后为acc读取数据；

consumer实现时**优化**，收到**invalidation后等待period（cache flush）**以降低一致性开销；

endpoint一旦有数据（写指针改变**blksz**），就拉高**valid**来通知acc并等待ready，acc一旦接收数据，consumer就更新queue读指针（读指针改变**blksz**），被**producer捕获**更新；

**producer**执行consumer的关联操作：acc产生数据**valid**且queue not full，acc将**数据写入queue后更新写指针**，更新操作保证一致性，被**consumer捕获**更新；

> **[图片提取文字 (image.png)]:**
> 4.2.1 Consumer Endpoint. The consumer endpoint is the point of ingress for data into the accelerator. Following registration of a queue with the Cohort engine, the consumer endpoint performs the coherence operations needed to track the read and write pointers of the SPSC queue. When the write pointer is updated by the queue's producer (from software or hardware), the consumer endpoint receives an invalidation, which is its signal to fetch new data for the accelerator. To reduce coherence effects, our implementation is optimised to wait a configurable period. Once the endpoint has the data, it sets its valid signal to the accelerator high and waits for the accelerator to be ready. Once the accelerator receives the data, the consumer endpoint updates the queue's read pointer accordingly, which is communicated to the producer via the coherence system.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2038.png)

> **[图片提取文字 (image.png)]:**
> 4.2.2 Producer Endpoint. The producer endpoint performs the reciprocal operations of the consumer endpoint. When the accelerator produces data (its valid signal goes high) and the queue is not full, the accelerator will take the data, store it into the queue, and then update the write pointer. Both operations are performed with appro-
> 
> priate consistency guarantees and the updates are communicated
> 
> to the consumer via the coherence system.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Proposed Cohort System Design
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%203.png)

**一致性**和代码语义

## Q-A

lock-free queue和Cohort多线程的底层关联是什么？一致性如何体现？

一个队列的生产者是加速器输出过程或软件线程，消费者是加速器输入过程或软件线程，对应**多线程共享队列**，线程对队列的操作满足一致性；

push、pop函数是**数据流**暴露给软件的IO接口，**register**函数是对加速器、软件间**内部数据流**的定义，数据流中数据传输由每个endpoint对队列的操作完成，endpoint读取queue是**利用Cache的一致性更新**（invalidate&flush），即producer入队、触发consumer更新cache并输入acc；

**代码视角**，endpoint修改queue指针，意味着数据出入队，**SoC视角**中，内存中queue改变，相关cache line在另一侧Cache中**invalidate**，通知endpoint进行cache更新；

**RCM**监控uncached register传递的地址（fifo-base-adr）是否invalidate，某地址invalidate后进入backoff，backoff结束后，RCM发起读操作，取回最新数据的副本，维护cache一致性；

lock-free **SPSC queue lib**：writer通过**fence、store语义**，在改变写指针之前完成数据写入，保证一致性；

类似queue lib，**WCM**排序写操作，确保reader先看到写指针更新，后看到数据更新；

> **[图片提取文字 (image.png)]:**
> We exploit this behaviour in two ways for Cohort to transparently observe the changes made by software. The Reader Coherency Manager (RCM) monitors an address passed from the uncached registers for incoming invalidations. Whenever an invalidation matches the address the RCM monitors, it enters the backoff state. The length of the backoff is controlled by the backoff unit. After the
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2040.png)

> **[图片提取文字 (image.png)]:**
> backoff ends, the RCM issues a coherent read to bring the most up to date copy of the data. Consistency is guaranteed by the writer ordering the data write before the pointer write using appropriate fences or store semantics, which is the standard behaviour in lock-free SPSC queue libraries. Just as the software library does, the Write Coherency Manager (WCM) carefully orders its write operations to ensure that a reader will see the update to the write pointer before seeing the data update.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ```
> fifo_t *cohort_to_sw = fifo_init(...);
> cohort_register(acc_id, sw_to_cohort, cohort_to_sw);
> push(0xcafedeed, sw_to_cohort);\nuint64_t result = pop(cohort_to_sw);
> cohort_unregister(acc_id, sw_to_cohort, cohort_to_sw);
>                               Figure 7: Cohort API usage example
> ```
> 
> fifo\_t \*sw\_to\_cohort = fifo\_init(...);
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2027.png)

**cache invalidation**：**多核共享内存**时，每个cache都有相同数据的副本，当某个核修改数据时只维护更新自身cache数据，此时其他cache的数据需要**invalidate**；

> **[图片提取文字 (image.png)]:**
> **Cache invalidation** is a state where we push away the data from the <u>cache</u> memory. When the data present in cache is outdated. We perform this operation of pushing back or flushing the data from the cache. Otherwise, it will cause data inconsistency.
> 
> When cached data gets stale or inaccurate, cache invalidation is the process of removing or updating it. When the original data changes, the process of invalidating a cache involves deleting or updating cached data. It's crucial because programs that rely on cached data may experience issues if it becomes outdated or erroneous over time.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2042.png)

[https://www.geeksforgeeks.org/system-design/cache-invalidation-and-the-methods-to-invalidate-cache/](https://www.geeksforgeeks.org/system-design/cache-invalidation-and-the-methods-to-invalidate-cache/)

Cohort **MMU**：基于原有MMU，扩展queue的VA，简化对加速器的编程，TLB、page table walker分页管理；

> **[图片提取文字 (image.png)]:**
> a result of the ISA-native MMU, queues are allocatable with malloc
> 
> and the Cohort endpoints seamlessly translate VAs to PAs.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2043.png)

> **[图片提取文字 (image.png)]:**
> The Cohort MMU features a TLB and page table walker to maximise its independence from the cores in the SoC. The operating system manages the Cohort MMU coherently with the others in the SoC, as we describe in Section 4.4. Each MMU has the page base pointer pointing to the root address of the page table of the corresponding process. Upon a TLB miss, the page table walker traverses the page table and refills the entry transparently. When there is a page fault due to a miss in the TLB and an unsuccessful page table walk, the Cohort MMU raises an interrupt to a core to resolve the page fault. Once resolved, the core writes to one of two MMU registers: the first simply resolves the fault and requires the page table walker to complete its own page table walk, while the second enables the core to write the page table entry directly into the TLB. Besides these registers, the Cohort MMU's TLB is flushed via a write to another register, in order to maintain TLB coherence.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2044.png)

加速器**接口**：valid-ready握手、AXI-Stream；

consumer从queue中取数给acc时，传递**valid**给acc，等待加速器可消耗数据的**ready**；

acc传递**valid**给producer，表示acc产生数据到producer endpoint，endpoint更新queue；

SBIO模式的加速器直接使用Cohort Engine，其余可将LSU替换成Cohort Engine；

acc产生数据的发送/消耗数据的获取，以可设定大小的**data block**被Cohort**封装/解包**，以block size更新指针，降低一致性开销；

在queue注册、初始化执行时，用户可通过CSR Bank**配置**加速器；

> **[图片提取文字 (image.png)]:**
> Figure 6 shows the accelerator connecting to the consumer and producer endpoints. Our prototype supports both simple valid-ready handshakes [12, 13, 71] and AXI-Stream [3] as latency insensitive interfaces to and from the accelerator. As the consumer endpoint retrieves data from the queue feeding into the accelerator, it sets its valid signal to the accelerator and waits for the accelerator to be ready to consume the data. Similarly, the accelerator sets its valid signal high to produce data to the producer endpoint, which makes the corresponding updates to the queue feeding out of the accelerator. Many SBIO accelerators are usable unmodified with the Cohort accelerator interface, while others could have their LSUs straightforwardly replaced with the Cohort engine.
> 
> Accelerators need not produce data with the same interface width as it consumes data. Cohort produces and consumes data to/from the accelerator using blocks of parameterised size, with appropriate ratchet logic to resize data to the accelerator's required sizes. As an optimisation, the producer and consumer endpoints reduce coherence traffic commensurate with the accelerator's data block input or output size, updating the read or write pointers by the data block size.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2045.png)

> **[图片提取文字 (image.png)]:**
> Upon queue registration to initiate execution, the user is also able to provide a data buffer akin to a control and status register (CSR) bank to configure the accelerator. The user simply points Cohort to the virtually contiguous data block, which the programmer formats
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2046.png)

> **[图片提取文字 (image.png)]:**
> as an accelerator-specific struct. The Cohort-enhanced accelerator directly receives this configuration data at registration time before data is passed. For accelerators with CSRs configured by AXI-Lite, Cohort also supports writing the data over AXI-Lite.
> 
> We have so far connected four accelerators into our Cohort SoC.
> 
> Section 5.2 describes the integration of three of these: SHA-256, AES, and an H264 encoder; we have also connected a short-time Fourier transform accelerator which we do not describe here. Each accelerator uses valid-ready handshakes for input and output data. We have demonstrated AXI-Stream functionality using an AXI-
> 
> Stream FIFO as a "null" accelerator, which is easily replaceable with a more complete accelerator.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6: Cohort Engine Architecture
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2037.png)

OS支持

**Cohort kernel driver**建立数据流，抽象和封装硬件细节，user API只需要**queue Cohort语义**，通过system call传递底层细节到kernel；

**运行时流程**：

kernel boot时，Cohort driver**申请**中断和MMU notifier**资源**（resource）；

app启动时，cohort_register**注册**queue，**push/pop**到注册的queue；

app结束时，cohort_unregister**释放**内存和MMU notifier**服务**（service）；

**driver feature**：

**MMU notifier**用于按照多核TLBs来flush MMU，维持TLB一致性，加速器是新的core；

Cohort driver申请并注册**Irq handler**，处理Cohort MMU触发的缺页中断；

**cohort_register**引发driver将Cohort配置注册Bank**映射到应用虚拟空间**，**注册**MMU notifier和Irq handler，之后driver传递**queue描述符**给Cohort；

**cohort_unregister**释放并**取消映射**上述资源；

> **[图片提取文字 (image.png)]:**
> Cohort provides a kernel driver to establish a complete and safe data flow and to abstract hardware details from the user. It shrinks the user space API to Queue Coherent semantics only and passes the low-level details to kernel via two simple system calls. Unlike many embedded accelerator environments, user space may not touch Cohort's configuration registers and need not be aware of physical addressing. This, in turn, justifies the usage of Cohort further, as functionality is solely, but safely implemented via the Cohort driver, without the need of wrappers. This also simplifies the usage of multiple Cohort engines using established OS abstractions. Our driver supports the following features:
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2048.png)

> **[图片提取文字 (image.png)]:**
> The driver is first probed at kernel boot time to request interrupts and MMU notifier resources. When applications launch, they call cohort\_register to register queues. The applications then push and pop data to and from Cohort via the registered queues. User space applications invoke cohort\_unregister at exit to clear the allocated memory and MMU notifier service.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2049.png)

> **[图片提取文字 (image.png)]:**
> - *MMU notifier for TLB flushes*. To maintain TLB coherence, Cohort's ISA-native MMU is also flushed alongside cores' TLBs. To enable this, MMU notifiers in Linux, normally used by platforms with IOMMUs, unified virtual memory, or hypervisors. The Cohort kernel driver registers its TLB flush function with an MMU notifier for processes at queue registration.
> - *Page fault resolution via interrupt.* As noted in Section 4.2.4, when the Cohort MMU sees a page fault, it triggers an interrupt and invokes a handler registered by the Cohort kernel driver.
> - Registering queues with Cohort. When a user mode application calls cohort\_register, the driver virtually maps the Cohort engine's configuration register bank and registers the MMU notifier and page fault handler. Then, the driver writes the queue descriptor information to Cohort. After this simple syscall, the requested Cohort engine is ready to use.
> - Unregistering queues from Cohort. Calling cohort\_unregister deallocates and unmaps the prior resources.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2050.png)

Cohort**能力和扩展**

queue抽象用于**线程/进程间通信**：分配一次queue，Cohort从一个process接收数据，产生数据到另一个process；

> **[图片提取文字 (image.png)]:**
> Inter-thread and Inter-process Communication. Cohort is able to accelerate inter process communication with the same queue abstraction, just as two software threads can communicate via a
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2051.png)

> **[图片提取文字 (image.png)]:**
> shared-memory queue. Such communication is commonly done by allocating the queue once and sharing its memory across two processes. This enables Cohort to receive input data from one process and produce output to another using the same push and pop methods as within a single process.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2052.png)

Cohort组成**加速器链**处理多负载，SoC设计者可将现有解耦的软件线程替换成加速器，用于线程通信的queue使用Cohort注册，加速器链可**运行时重配置**（Cohort数据流）；

> **[图片提取文字 (image.png)]:**
> Accelerator Chaining. As more workloads necessitate hardware acceleration, accelerator chaining is a perfect way to accelerate complex computations. Cohort supports chaining together multiple accelerators using SPSC queues, bringing acceleration to more diverse workloads with minimal overhead and great flexibility. As an SoC designer moves to add new accelerators to their system, they can take existing decoupled software threads and directly replace them with Cohort-enhanced accelerators. The queues that are already used to communicate with the software thread are maintained, but registered with the relevant Cohort engine instead. This accelerator chaining capability also enables runtime reconfiguration of accelerator chains based on applications' needs.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2053.png)

Cohort基于SPSC模型，限制应用的数据流，**多生产者/消费者是未来扩展**；

> **[图片提取文字 (image.png)]:**
> Multi Producer/Consumer. Cohort sticks with the SPSC model, essentially a restricted dataflow where queues are not split or merged. Enabling queues supporting multiple producers or multiple consumers would provide value for a broader set of multithreaded use cases and for multiple accelerators to process data to/from a single queue. Generally these queues require atomic memory operations to guarantee correct operation, for which there is somewhat less standardisation of queue organisation. As a result, we leave support for these queues and design of their queue descriptors to future work.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2054.png)

# 实验方法

环境：**OpenPiton**多核研究框架集成Cohort，在Alveo U200 FPGA实现，Linux；

4-tile设计，2个64位R-Core、2个加速器；

> **[图片提取文字 (image.png)]:**
> This section details our methodology, accelerator configurations, and benchmarks. We integrate Cohort into OpenPiton [8–10] and boot Linux (v5.6-rc4) on a Xilinx Alveo U200 FPGA running at 100 MHz. We use a four tile design with two 64-bit 6-stage Ariane RISC-V RV64GC cores and two accelerators (shown in Figure 2). We use the default OpenPiton configuration of 8KiB L1D, 16KiB L1I, 8KiB L1.5, and 64KiB 4-way L2 caches. The Cohort TLB has 16 entries and the producer and consumer endpoint accelerator interfaces are 64-bit wide. We use a minimal user-mode driver for our experiments.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2055.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Cohort FPGA prototype with AES and SHA accelerators
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%207.png)

> **[图片提取文字 (image.png)]:**
> OpenPiton is the world's first open source, general purpose, multithreaded manycore processor. It is a tiled manycore framework scalable from one to 1/2 billion cores. It is a 64-bit architecture using SPARC v9 ISA with a distributed directory-based cache coherence protocol across on-chip networks. It is highly configurable in both core and uncore components. OpenPiton has been verified in both ASIC and multiple Xilinx FPGA prototypes running full-stack Debian linux. We have released both the Verilog RTL code as well as synthesis and back-end flow. We believe OpenPiton is a great framework for researchers in computer architecture, OS, compilers, EDA, security and more.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2056.png)

[https://www.mouser.com/pdfDocs/u280userguide.pdf](https://www.mouser.com/pdfDocs/u280userguide.pdf)

Baseline：**MAPLE**的核外高内存并行的LSU;

**MAPLE:slim-in-order**多核间的高效数据传输，侧重**非直接内存访问（如指针）**应用（图计算）的访问-执行解耦的多线程和prefetch model；

基于MAPLE，改成**MMIO-based和DMA-based的多核应用加速**；

> **[图片提取文字 (image.png)]:**
> connect with accelerators. In its original setting, MAPLE is an outof-core, highly memory parallel load-store unit. It is designed to enable efficient data movement in manycores with slim in-order cores. Its particular focus is decoupled access-execute multithreading and prefetching models for applications with many indirect memory accesses, like graph processing. We modified MAPLE to instead host accelerators and provide MMIO-based and coherent DMA-based invocation (two common approaches). Performance counter data comes from each Cohort Engine, Ariane core or MAPLE unit.
> 
> For our baselines, we repurposed a MAPLE decoupling unit [61] to
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2057.png)

MMIO baseline，MMIO传输数据字是不可预测的往返，没有一致性；

> **[图片提取文字 (image.png)]:**
> *MMIO Baseline.* For some workloads, an MMIO-based invocation can make sense, as queue contents are not saved in cache and are hence disconnected from the coherence system. However, as
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2058.png)

> **[图片提取文字 (image.png)]:**
> noted in Section 3.2, MMIO often requires non-speculative roundtrips from core to accelerator for each data word, a fact that was highlighted in the original presentation of MAPLE [61]. We use MAPLE's MMIO queue interface to provide data to and collect data from the accelerator with no coherence effects.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2059.png)

Coherent DMA baseline（RISC-V MMU）

> **[图片提取文字 (image.png)]:**
> tween memory and the device without host interference. The DMA must be enabled and programmed by a core, and accelerators must still wait for sufficient data to operate. We use MAPLE's coherent LLC data prefetching feature to provide data to the accelerator and use the P-Mesh TRI [9] to coherently store results. Note that MAPLE's coherent DMA is more software efficient than typical DMA approaches as it uses a RISC-V MMU rather than requiring an IOMMU.
> 
> Coherent DMA Baseline. DMA enables bulk data movement be-
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2060.png)

加速器：SHA256、AES128、H264

> **[图片提取文字 (image.png)]:**
> SHA-256. The first accelerator is an open-source SHA-256 cryptographic core [24], which is usable from Linux. Our accelerator accepts input in 512 bit data blocks and hashes them to a 256 bit irrevertible hash digest [27]. The accelerator accepts incoming 64 bit data blocks from the consumer endpoint and uses a ratchet to build a 512 bit input block. The hash digest is fed back using a
> 
> ratchet to the producer unit in four chunks of 64 bits.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2061.png)

> **[图片提取文字 (image.png)]:**
> ric encryption algorithm [28]. In our prototype, we connected an open-source accelerator for AES encryption [2] which generates a ciphertext in blocks of 128 bits with a key of the same width. With Cohort we added a ratchet to consume 128 bits of data from 64 bit chunks from the consumer endpoint and the reverse for sending encrypted ciphertext to the producer endpoint. The encryption key
> 
> is passed via a coherent CSR struct as described in Section 4.3.
> 
> AES-128. Advanced Encryption Standard (AES) is a symmet-
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2062.png)

> **[图片提取文字 (image.png)]:**
> *H264 Encoder.* H264 is a video compression standard widely used in the video industry. It provides good video quality at significantly reduced bitrates. We integrated an H264 encoder from Zexia [79] (using Context-adaptive variable-length coding or CAVLC) with the Cohort engine and confirmed its correct operation. This example also illustrates how Cohort can handle variable input size. The existing instance of the accelerator that we adopted accepts the number of frames at the start of its input to enable variable input length. The integration code includes a ratchet to prepare each frame for H264 in a similar manner to AES and SHA.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2063.png)

## Tab 2

BenchMark in Cohort、Baseline

> **[图片提取文字 (image.png)]:**
> and test the entire system. Benchmark parameters are illustrated in Table 2. Here batch size refers to an optimisation that updates the read and write pointers in batches instead of incrementally. This helps to reduce the coherency traffic in the system and improve performance.
> 
> To characterise the Cohort Engine, we run benchmark applications
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2064.png)

> **[图片提取文字 (image.png)]:**
> This is an optimisation that would be applied to multithreaded software but is also exploitable by Cohort, demonstrating Software-Oriented Acceleration in action. Below, we briefly describe our benchmark applications.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2065.png)

> **[图片提取文字 (image.png)]:**
> ## **Table 2: Benchmark Tuning Parameters.**
> 
> | AES, SHA          |
> |-------------------|
> | Cohort, MMIO, DMA |
> | 64/8192 elements  |
> | 2/64 elements     |
> | 256 Bytes         |
> |                   |
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2066.png)

> **[图片提取文字 (image.png)]:**
> Benchmark Implementation in Cohort. Benchmarks in Cohort initialise the SPSC queues then push and pop the data in sequence. To hash 1 block of text we push 64 bits of data 8 times and fetch the corresponding hash with 4 pops. For AES, there are 2 pushes and 2 pops. As mentioned earlier, we encapsulate these movements
> 
> into batches and run applications until queue size is reached.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2067.png)

> **[图片提取文字 (image.png)]:**
> Benchmark Implementation in Baselines. Benchmarks for our baselines are logically similar to Cohort's, but more complicated in their implementation. With MMIO, the core cannot achieve memory-level parallelism and so must receive the accelerator's output word by word before passing the next input word. This affects performance versus Cohort which can interleave input/output and expose MLP within batches. With Coherent DMA, special API functions (also containing MMIO writes) are called for each data block to be copied to/from the modified MAPLE unit, which matches common DMA programming mechanisms.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2068.png)

# 实验结果

> **[图片提取文字 (image.png)]:**
> The goal of our evaluation is to demonstrate Cohort in a realistic heterogeneous SoC with multicore Linux support while providing a combination of programmability and performance improvements over the state-of-the-art (SOTA).
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2069.png)

## Fig 8-9、Tab 3

> **[图片提取文字 (image.png)]:**
> ## Program Latency with SHA accelerator
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2070.png)

> **[图片提取文字 (image.png)]:**
> Table 3: Summary of peak speedup for Cohort with AES and SHA (Cohort batch=64). Cohort shows consistent speedup over both baselines.
> 
> | Queue size  | 4    | 128  | 256  | 512   | 1024  | 2048  | 4096  | 8192 |
> |-------------|------|------|------|-------|-------|-------|-------|------|
> | SHA Speedup |      |      |      |       |       |       |       |      |
> | Vs MMIO     | 5.44 | 6.05 | 6.75 | 7.22  | 7.62  | 8.30  | 8.38  | 7.16 |
> | Vs DMA      | 7.27 | 7.94 | 8.85 | 11.24 | 10.70 | 10.83 | 10.62 | 8.97 |
> | W/ Batching | 2.32 | 2.45 | 2.65 | 2.79  | 2.96  | 3.01  | 3.33  | 2.81 |
> | AES Speedup |      |      |      |       |       |       |       |      |
> | Vs MMIO     | 2.0  | 1.89 | 1.84 | 1.83  | 2.07  | 2.03  | 2.03  | 1.86 |
> | Vs DMA      | 1.9  | 1.83 | 1.74 | 1.71  | 1.75  | 2.03  | 1.94  | 1.69 |
> | W/ Batching | 5.3  | 6.05 | 7.11 | 7.16  | 8.02  | 7.99  | 8.10  | 7.42 |
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2071.png)

> **[图片提取文字 (image.png)]:**
> ## Program Latency with AES accelerator
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 9: Latency in kilocycles to execute the AES benchmark. X axis shows total number of queue elements, Y axis shows time in kilocycles (lower is better). Cohort batches queue elements in groups of 2 to 64.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2072.png)

延迟

> **[图片提取文字 (image.png)]:**
> Figure 8 illustrates the latency to queue size relationship for our SHA benchmark running with different batch sizes and communication APIs. "Cohort batch=N" indicates the Cohort application optimisation of batching its write/read pointer updates to occur only after N elements have been copied into the queue. The same goes for AES in Figure 9.
> 
> Across the board, Cohort outperforms the MMIO and Coherent DMA baselines for the SHA application. Over all queue sizes tested, Cohort performs the best with larger batches and the batching optimisation works robustly. Cohort starts at a batch size of 8 elements to reflect one SHA input of 512 bits. Table 3 shows the speedup for Cohort AES and SHA with 64 element batches over the baselines as well as the improvement within Cohort brought by batching. The speedup brought by Cohort on SHA versus MMIO and DMA ranges from  $5.44 \times$  to  $8.38 \times$  for MMIO and from  $7.27 \times$ to 11.24× for DMA. For SHA, batching increases performance by  $2.32 \times \text{ to } 3.33 \times .$
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2073.png)

> **[图片提取文字 (image.png)]:**
> For AES, Cohort improves over the baselines as batch size increases. Batch sizes larger than 16 elements always perform equal or better than both MMIO and DMA baselines. Table 3 shows a range in speedup from Cohort on AES versus MMIO and DMA ranging from  $1.83 \times$  to  $2.07 \times$  for MMIO and from  $1.69 \times$  to  $2.03 \times$  for DMA. For AES, batching increases performance by  $5.30 \times$  to  $8.10 \times$ . The lower performance for AES comes from two factors: the first is its symmetric data movement where AES produces output of the same size as the input, which increases false sharing on the read and write pointers with the Ariane core as it waits to acquire the output data. The second factor is the accelerator's lower latency of 41 cycles versus SHA's latency of 66 cycles.
> 
> These numbers demonstrate the importance of how, with Cohort, optimisations applied in software can bring valuable speedups at the accelerator interface. These results come in line with our expectations in Section 3 and prove the importance of Software Oriented Acceleration and the benefits of the SPSC queue abstraction exploited by Cohort.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2074.png)

## Fig 10-11

> **[图片提取文字 (image.png)]:**
> ## IPC Performance with SHA accelerator
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Figure 10: Instructions Per Cycle (IPC) speedup of Cohort over the baselines achieved while executing the SHA benchmark. X axis shows total number of queue elements, Y axis shows IPC speedup (higher is better). Cohort uses a batching factor of 64.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2075.png)

> **[图片提取文字 (image.png)]:**
> IPC Performance with AES accelerator
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ■ Speedup over MMIO ■ Speedup over Coherent DMA
> 
> Figure 11: Instructions Per Cycle (IPC) speedup of Cohort over the baselines achieved while executing the AES benchmark. X axis shows total number of queue elements, Y axis shows IPC speedup (higher is better). Cohort uses a batching factor of 64.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2076.png)

**IPC**

> **[图片提取文字 (image.png)]:**
> Beyond seeing an overall improvement in latency, we argued that Cohort would make more efficient use of the core as it provides data to and reads data from accelerators. Figure 10 and Figure 11
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2077.png)

> **[图片提取文字 (image.png)]:**
> show significant improvements in IPC for Cohort over the MMIO and DMA approaches, where cores must make MMIO round trips to communicate with the accelerators. Cohort provides a peak IPC speedup for Cohort SHA over MMIO and DMA of 4.42× and 2.11×. For AES, Cohort achieves a peak speedup in IPC over MMIO and DMA of  $2.83 \times$  and  $1.77 \times$ , respectively. These numbers validate that Cohort better utilises the core while data transfer is occurring.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2078.png)

## Tab 4

FPGA实现的面积开销

> **[图片提取文字 (image.png)]:**
> We perform FPGA implementation with Xilinx Vivado 2022.1 and report post-synthesis resource utilisation in Table 4. The empty Cohort engine comprises around 10% of the LUTs and 20% of the registers of a Cohort tile, or less than 4% of the LUTs and 10% of the registers of an Ariane tile. A tile with an empty Cohort Engine is about 39% and 46.6% of the Ariane tile by LUTs and registers (both tiles feature OpenPiton's NoC routers and L1.5 and
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2079.png)

> **[图片提取文字 (image.png)]:**
> ## L2 caches). Despite this small area, Cohort packs a punch in terms of the functionality and features it provides.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2080.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Cohort FPGA prototype with AES and SHA accelerators
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%207.png)

> **[图片提取文字 (image.png)]:**
> to the large memory needed for AES which would consume considerable SRAM in an ASIC process (note that the AES BRAM is larger than that of an Ariane tile with its previously stated cache configuration totalling around 100KiB). The Cohort engine consumes roughly 27% more LUTs and 57% more registers than the relatively small SHA accelerator. Compared to the H264 encoder, the empty Cohort engine consumes around 37% of the LUTs and 71% of the registers. H264 consumes 4 BRAM slices (tens of kilobytes of memory) and unlike Cohort or the other accelerators, it also consumes 6 DSP slices. Full tiles including Cohort plus AES, SHA, or H264 are significantly smaller than the size of an Ariane tile. The MMU itself is area efficient, consuming 1081 LUTs, 1206 registers, and no BRAM. Of that, the TLB makes up 911 LUTs and 1029 registers, while the page table walker makes up 168 LUTs and 109 registers. This small MMU leads to savings in memory (no need for separate I/O page tables), runtime (less page table management in software), and software/OS porting. While our demonstrated accelerators are relatively small, the same MMU and Cohort engine
> 
> are usable with larger accelerators.
> 
> The Cohort engine has roughly 68% the LUTs and 45% the regis-
> 
> ters of the AES accelerator while consuming no BRAM, compared
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2081.png)

> **[图片提取文字 (image.png)]:**
> Table 4: FPGA Synthesis results for resource utilisation: Cohort with or compared to Ariane, MAPLE, and accelerators
> 
> | Resource  | Ariane | Empty  | <b>Empty</b> | Cohort | Cohort | MAPLE | AES  | SHA  | H264 |
> |-----------|--------|--------|--------------|--------|--------|-------|------|------|------|
> | Type      | Tile   | Cohort | Cohort       | + AES  | + SHA  | + AES | Only | Only | Only |
> |           | 1      | Tile   | Engine       | 1      | 1      | + SHA |      |      |      |
> | LUTs      | 67083  | 26390  | 2594         | 6679   | 4524   | 21066 | 3837 | 2041 | 6851 |
> | Registers | 39879  | 18591  | 3799         | 12176  | 6064   | 28276 | 8531 | 2420 | 5341 |
> | BRAM      | 41.5   | 9.5    | 0            | 47.5   | 0      | 47.5  | 47.5 | 0    | 4    |
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2082.png)

# 相关工作

异构、DSA for **功耗性能比**，使用加速器通过编程配置或DSL；

> **[图片提取文字 (image.png)]:**
> As Moore's Law has slowed and Dennard Scaling has ended, system designs are increasingly exploiting heterogeneous parallelism and domain-specific accelerators to scale performance at acceptable power [23, 64]. Using accelerators often relies on the programmer configuring them manually [32, 63] or through a domain specific language, for the class of applications that are being executed [44,
> 
> 82].
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2083.png)

**数据访问和计算的分离**

[访存-计算解耦的处理器架构](%E8%AE%BF%E5%AD%98-%E8%AE%A1%E7%AE%97%E8%A7%A3%E8%80%A6%E7%9A%84%E5%A4%84%E7%90%86%E5%99%A8%E6%9E%B6%E6%9E%84%20275e12d10b6e8042ae35f27917277c44.md)

**隐藏向processor提供数据的延迟**,**访问core**访问数据后投喂给**计算core**进行计算;

相关体系结构设计是**通过queue**传输数据来隐藏传输延迟,作为设计**超标量处理器**的更简单替代；超标量是同时获取、发射、执行多个指令来**增大IPC并且隐藏访存指令的延迟**；多个指令之间可能存在冒险(控制冒险、数据冒险),需要暂停发射某些指令和转发结果到已发射指令,同时将没有冒险的后序指令提前发射和执行,以进一步增大IPC和隐藏长延迟指令的延迟,即**乱序执行**；

**DeSC**的解耦设计设置supply core（**ReOrder Buffer的OoO pipeline**）和compute core（**OoO core**或**加速器**），软件接口是新ISA（需要底软和编译适配），并且supply和compute的元素有严格**一对一关系（硬件queue和Buffer的entry粒度）**且在**tapeout**（指流片）时确定；

> **[图片提取文字 (image.png)]:**
> Decoupling of data-access and compute was originally proposed by J. Smith [69] to mitigate latency of data-supply to processor cores, by having a core accessing the data and feeding it to another core, which handles the computation. After that, several hardware implementations have been proposed, where data communication
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2084.png)

> **[图片提取文字 (image.png)]:**
> occurs through architectural queues [33, 51, 78]. These works aim to hide memory latency as a simpler alternative to superscalar processors. DeSC [37, 38] repurposed decoupling to supply computationexclusive units like accelerators. DeSC's programming model offers a producer-consumer relationship between heterogeneous processing units. However, DeSC uses architectural queues that are exposed to software through new ISA instructions, which requires software
> 
> changes. Furthermore, there is a strict one-to-one relation between
> 
> supply and compute elements, determined at tapeout time.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2085.png)

> **[图片提取文字 (image.png)]:**
> 所谓"超标量"是指 CPU 在一个时钟周期内**获取、执行和提交多条指令**,这个概念和"标量"对 应,"标量"指 CPU 在一个时钟周期内获取、执行和提交一条指令;
> 
> "乱序"和"顺序"对应,"顺序"的意思是"顺序发射、顺序执行",是指 CPU 按照指令原始顺序逐条发射、逐条执行,而"乱序"就是指"乱序发射、乱序执行"。"超标量"一般和"乱序"搭配, "标量"一般和"顺序"搭配,前两者是现代高速微处理器所广泛使用的技术。
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2086.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1. Block diagram of a decoupled architecture.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2087.png)

> **[图片提取文字 (image.png)]:**
> Fig. 3 shows DeSC's hardware implementation. Grey boxes represent an abstracted view of the hardware modules that either calculate the memory addresses or compute the output values. Here, SuppD is a nearly-generalpurpose core— an out-of-order pipeline with ROB, Reg-File, and a number of integer functional units for calculating memory addresses—but sizing choices are tailored to its role and no floating point functional units are needed. Likewise, CompD can be another out-of-order core or a specialized hardware accelerator for a particular application. Either way, CompD is tailored to its role by removing memory hierarchy access; the SuppD supplies it with data as needed. For data supply, a Communication Queue (CommQ)
> 
> interconnects SuppD to CompD, and feeds into a Com-
> 
> munication Buffer (CommBuf) from which value lookup
> 
> can be performed. The SuppD also includes a Store
> 
> Buffer for updating the memory hierarchy when a com-
> 
> puted value is returned. Finally, Table 3 lists the added
> 
> instructions on either side to support DeSC.
> 
> programmable processor cores with different roles, our
> 
> work is open to more specialization. That is, both SuppD
> 
> and CompD could be either processors or accelerators,
> 
> or (as we discuss here) processors with tailoring to each
> 
> of their roles.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2088.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: Hardware implementation of DeSC.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2089.png)

> **[图片提取文字 (image.png)]:**
> | Supplier Device                   | Computation Device |
> |-----------------------------------|--------------------|
> | PRODUCE(Reg)                      | Reg=CONSUME()      |
> | LOAD_PRODUCE(Addr)                |                    |
> | STORE_ADDR(Addr)                  | ST0RE_VAL(Reg)     |
> | Table 3: ISA extensions for DeSC. |                    |
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2090.png)

**数据流模式**

MAD和HWPE的数据流Engine优化数据访问，但**集成在硬件中会约束ISA或需要新ISA**，Cohort的数据访问对ISA透明；

MAPLE将数据访问解耦到memory-fetching engine，但对所有访问引入不小的**保底开销**；Cohort利用一致性协议进行完成SPSC传输，开销低；  

> **[图片提取文字 (image.png)]:**
> Data streaming proposals also have this restriction. MAD [41] and HWPEs [19, 22] have a data-access engine optimised for dataflow computation, which is integrated with cores or accelerators to perform the memory-access portion of programs. Cohort offers similar producer-consumer relationships without restricting them in hardware, and making them transparent to the ISA. In Cohort, accelerators can be configured at runtime to
> 
> the ISA. In Cohort, accelerators can be configured at runtime to be consuming from or producing to any core or accelerator onchip. This creates a more rich space of possible heterogeneous communication patterns.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2091.png)

> **[图片提取文字 (image.png)]:**
> Although MAPLE [61] offers a flexible decoupling mechanism where cores communicate via a network-connected memory-fetching engine, it still requires software changes with explicit produce and consume operations. Moreover, MAPLE's communication scheme resorts to side-effectful, non-idempotent memory-mapped I/O instructions, which sets a significant minimal latency threshold between devices. Cohort communicates using a simple software queue, which enables greater speculation and parallelism of produce and consume operations.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2092.png)

松耦合加速器有独立存储和控制，紧耦合加速器作为CPU的附庸执行特定功能的加速（可能共享CPU Cache）；

> **[图片提取文字 (image.png)]:**
> Loosely coupled and tightly coupled accelerators. Accelerators come in different flavors and sizes, but we classify them regarding their hardware integration into tighly-coupled and looselycoupled [20]. Often, the classification is in terms of where the accelerator sits in the architecture, namely, close to the CPU or not.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2093.png)

> **[图片提取文字 (image.png)]:**
> functional (computing) unit physically attached to the CPU pipeline (e.g. FPU, or inline accelerators [73]), and accelerators that share the private cache with the CPU (e.g. RocketChip [4], Hwacha [48], and Gemmini [31]). In the loosely-coupled category, we find accelerators placed on the main chip interconnect [14, 39, 59], or placed on a separate chip which communicates to the chip interconnect through
> 
> an I/O interface (e.g. PCIe).
> 
> In the tightly-coupled category, we can find accelerators as a
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2094.png)

互联

Cohort也可基于CXL、UCIe一致性协议通信；

Intel的Alder Lake处理器的PCIe链路占2.4mm2，CPU core（L1）占5.37mm2，PCIe开销不小；

> **[图片提取文字 (image.png)]:**
> Comparison with interconnects CXL and UCIe are emerging interconnect specifications that tackle host-to-host and chiplet-tochiplet connections respectively. Although CXL enables inter-host coherent communication at a low latency and high throughput, it has a large area and power overhead associated with underlying PCIe protocol stack. By contrast, our implementation of Cohort
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2095.png)

> **[图片提取文字 (image.png)]:**
> hort would also be implementable on top of the CXL coherency system with the correct semantics. UCIe takes a layered approach to accommodate challenges in die-to-die communication for emerging chiplets. Currently the protocol layer supports four standards: PCIe 6.0, CXL3.0, CXL2.0 and raw data streaming. We envision Cohort would be implemented on top of emerging coherency protocols in new UCIe protocol layers, or atop CXL as previously described. However, the control and physical layers of UCIe are out of scope for this paper.
> 
> tackles efficient coherent communication within a single SoC. Co-
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2096.png)

> **[图片提取文字 (image.png)]:**
> Area-wise, in a state of the art design such as Intel Alder Lake fabricated with 7nm process, an 8 lane PCIe 5 PHY takes 2.4mm<sup>2</sup> [50]. By contrast, a high performance CPU core on the same chip with L1 cache but without L2 cache or power gating logic takes around  $5.37mm^2$ , only a little over twice the PCIe PHY block size. Note that this does not take into account the area of the PCIe controller
> 
> which would also be required.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2097.png)

系统级设计

松耦合加速器不改变CPU和ISA，通过互联**即插即用**，不需要紧耦合的验证开销，符合**chiplet**思想；

紧耦合能利用CPU处理**VM地址翻译**；

松耦合（driver-triggered）加速器需要**预填充特定TLB和pin mem（没有缺页处理）**；

NVDLA使用**reserved、no-cacheable** memory，OS不可访问，只有DLA和driver可访问；

Cohort没有紧耦合所需的集成验证，但利用**全功能MMU**提供的**缺页处理**和**TLB shootdown**（TLB一致性的机制）；

> **[图片提取文字 (image.png)]:**
> System-level considerations. The advantage of loosely-coupled accelerators is that they do not modify the CPU or the ISA. They can be integrated in a relatively plug-and-play manner, through the interconnect, overcoming the verification overhead of tightlycoupled integration. This is aligned with trends in fast chip prototyping [1, 52, 77], where an SoC is made of reusable, third-party IP blocks. However, tight-coupled accelerators have the advantage of having the CPU dealing with VM translation. Loosely-coupled, driver-triggered accelerators like those of ESP [52] have the strategy to pre-fill a specialised TLB and pin memory so that no page-faults can occur, while NVDLA [59] uses a reserved, non-cacheable memory region, that the OS cannot use, and only the accelerator and its device driver have access to it. Cohort enjoys the advantages of loosely-coupled accelerators while supporting VM in a more flexible manner by incorporating a fully-capable MMU that handles page faults and TLB shootdowns.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2098.png)

可编程性

松耦合Acc一般通过中断通知cpu表示完成，紧耦合Acc一般通过ISA指令读写寄存器；

松耦合Acc的中断开销较大，开销优化方法侧重IO加速器，Cohort是片上加速器的通信模型；

许多工作将硬件语义融入OS机制（垂直直通）来获得简单高效可扩展的异构通信机制，但引入新的OS kernel和硬件；Cohort使用现有OS和多线程queue构造；

DTU是个硬件queue？baseline？

> **[图片提取文字 (image.png)]:**
> Programmability. While tightly-coupled accelerators are triggered by ISA instructions, and thus the result goes to the register file [4, 73], loosely-coupled ones often rely on interrupts or spinpolling for the CPU to know that they finished. Several works have enhanced the I/O software stacks, by reducing interrupt overheads [42, 67], or combining them with spin-polling as a hybrid notification mechanism [25]. Recently, HyperPlane [57] proposed a hardware mechanism to accelerate these notifications. However, these techniques focus on I/O accelerators, while Cohort leverages the coherence protocol to get notified that another processing element has produced or consumed data, thus also supporting on-chip accelerators. Cong et al. [17, 18] proposed an allocation protocol to avoid OS overhead in scheduling tasks to on-chip accelerators.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%2099.png)

> **[图片提取文字 (image.png)]:**
> However, this still requires large workloads to be efficient, while Cohort is also suited for finer-grain tasks.
> 
> M3 [6], M3X [5], Rackscale Microkernel [40], Solros [56] bake hardware semantics into OS mechanisms for tight integration. All target simple, efficient, and scalable heterogeneous communication, but implement brand new (micro)kernels (and new DTU hardware [6, 40]). Cohort exploits existing OS and multi-threaded application best practices, via well known queue semantics and enables accelerators and cores to share queues in coherent memory for communication. We improve performance over the baselines with low cost and a more intuitive API. Similarly to our MMIO baseline, DTU queues are either kept in (limited-size) scratchpads and modified by MMIO or, similarly to our DMA baseline, are kept in DRAM with pointers updated by MMIO and NoC messages. Both DTU mechanisms require software changes, unlike Cohort.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%20100.png)

加速器标准

许多加速器编程模型暴露硬件细节用于编程，是面向硬件的fashion；

Cohort将软件语义嵌入加速器集成环境；让加速器集成后适配现有os？但加速器设计仍然是面向硬件？

软件语义？硬件语义？

面向硬件是底软和编译适配硬件接口和硬件ISA；

面向软件是硬件系统适配现有底软、编译和OS；

> **[图片提取文字 (image.png)]:**
> Acceleration standards A number of accelerator standardisation efforts are taking place, particularly to standardise on programming models for GPGPUs and similar programmable accelerators. Vulkan [35] and SPIR-V [34], with features like pipelined barriers, expose deeper hardware details in a traditional, hardware-oriented fashion, implying a deep asymmetry between the accelerators and cores. Cohort on the other hand effectively embeds common software semantics into the accelerator integration environment (note: not the accelerator itself). Cohort-enabled accelerators can be decoupled from memory operations and act as peers to cores in a software-oriented manner. By leveraging software semantic awareness to provide a high performance API that is implementation agnostic, we see Cohort as additional, rather than a competitor to these solutions. For accelerators designed or programmed with Vulkan/SPIR-V, any communication of a SBIO nature (whether through explicit SPSC queues or other data movement behaviours) could target a specialised Cohort engine instead of the usual LSU for a potential uplift in performance.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%20101.png)

Queue Lib和语言支持

Cohort软件库能使用linux io kernel

> **[图片提取文字 (image.png)]:**
> Queue Libraries and Language Support Beyond adoption for Boost, we are investigating using Cohort with Unix pipes. Of further interest for high-performance, asynchronous acceleration use cases is the Linux io\_uring subsystem [68]. The io\_uring API has enabled a variety of new high-performance I/O use cases in Linux, particularly for networking and file I/O. Integrating Cohort with io\_uring would enable a rich runtime for managing accelerators. More intriguingly, Cohort accelerators could also use io\_uring to request services from the kernel via their native queue interfaces. With simple runtime support (comparable to the library support we added for Boost), languages could automatically retarget their queues to make use of Cohort. We leave automatic identification of
> 
> queues to future work.
![image.png](Cohort%20Software-Oriented%20Acceleration%20for%20Heteroge/image%20102.png)