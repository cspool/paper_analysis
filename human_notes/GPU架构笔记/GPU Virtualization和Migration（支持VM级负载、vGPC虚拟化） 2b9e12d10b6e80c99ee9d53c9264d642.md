# GPU Virtualization和Migration（支持VM级负载、vGPC虚拟化）

ref：GPU VIRTUALISATION

ref：VIRTUALIZING HARDWARE PROCESSING RESOURCES IN A PROCESSOR

# GPU Virtualization（VM）

虚拟化GPU，让GPU支持多个VM的请求，不同VM的请求可以忽略、等待和抢占，我或许能参考其接口，我的设计着重抢占设计。

## ABS & BG

虚拟机相比多进程是隔离程度更高的虚拟化方法；

> **[图片提取文字 (image.png)]:**
> A method of GPU virtualization comprises allocating each virtual machine (or operating system running on a VM) an identifier by the hypervisor and then this identifier is used to tag every transaction deriving from a GPU workload operating within a given VM context (i.e. every GPU transaction on the system bus which interconnects the CPU, GPU and other peripherals). Additionally, dedicated portions of a memory resource (which may be GPU registers or RAM) are provided for each VM and whilst each VM can only see their allocated portion of the memory, a microprocessor within the GPU can see all of the memory. Access control is achieved using root memory management units which are configured by the hypervisor and which map guest physical addresses to actual memory addresses based on the identifier associated with the transaction.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%201.png)

**hypervisor**（VMM）控制VM访问硬件，一般VM访问硬件会经过VMM；

虚拟化包含：VM被VMM分配唯一标识用于传输，内存分配，Root MMU控制访问， GPU内处理器管理不同VM负载；

> **[图片提取文字 (image.png)]:**
> virtual machines (VMs) which each run an operating system (e.g. where the operating systems running on different VMs may be the same or different). The VMs are created and supervised by software which is called a hypervisor (or virtual machine monitor, VMM) and which controls each VM's access to the hardware within a computing system. Consequently, communications between the operating systems (running on the VMs) and the GPU often go through the hypervisor which is responsible for enforcing security.
> 
> [0002] In many situations, the CPU may run a number of
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> VM) an identifier by the hypervisor and then this identifier is used to tag every transaction deriving from a GPU workload operating within a given VM context (i.e. every GPU transaction on the system bus which interconnects the CPU, GPU and other peripherals). Additionally, dedicated portions of a memory resource (which may be GPU registers or RAM) are provided for each VM and whilst each VM can only see their allocated portion of the memory, a microprocessor within the GPU can see all of the memory. Access control is achieved using root memory management units which are configured by the hypervisor and which map guest physical addresses to actual memory addresses based on the identifier associated with the transaction. Software running in the microprocessor within the GPU is involved in managing the workloads for the different VMs (e.g. instead of relying upon hypervisor software running in the CPU to control and manage the workloads).
> 
> [0005] A method of GPU virtualization comprises allocat-
> 
> ing each virtual machine (or operating system running on a
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%203.png)

虚拟化：CPU上调度不同VM和GPU上调度不同VM负载的**地址空间隔离**，所以设计Root MMU翻译**guest物理地址**，CPU程序和GPU kernel的**多进程/多kernel**是运行基于虚拟地址空间，所以设计MMU翻译**虚拟地址**；

虚拟化的**架构**：CPU（hypervisor+VMs）、CPU MMU（CPU虚拟地址和guest物理地址）、CPU Root MMU（guest物理地址和内存物理地址）、GPU（多任务管理+SMs）、GPU MMU（GPU虚拟地址和guest物理地址）、GPU Root MMU（guest物理地址和显存物理地址）；

虚拟化的**算法**：内存分配、Root MMU的地址翻译、CPU标记的VM传输；

虚拟化的**IC实现**：量产方法？

> **[图片提取文字 (image.png)]:**
> [0006] A first aspect provides a computing system comprising: a central processing unit arranged to run a plurality of virtual machines under the control of a hypervisor and wherein transactions output by the central processing unit are assigned an identifier for the virtual machine to which it relates; a CPU memory management unit arranged to translate between virtual memory addresses used by the virtual machines and guest physical addresses; a first root memory management unit configured by the hypervisor and arranged to translate between guest physical addresses used by the
> 
> CPU memory management unit and actual physical memory
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%204.png)

> **[图片提取文字 (image.png)]:**
> addresses based on the identifier for the virtual machine assigned to the transaction; a graphics processor unit comprising a microprocessor and wherein transactions output by the graphics processor unit are assigned an identifier for a virtual machine which is inherited from a triggering transaction received from the central processing unit; a GPU memory management unit arranged to translate between virtual memory addresses used by the graphics processor unit and guest physical addresses; and a second root memory management unit configured by the hypervisor and arranged to translate between guest physical addresses used by the GPU memory management unit and actual physical memory addresses based on the identifier for the virtual machine assigned to the transaction; and wherein access to one or more portions of a memory resource is controlled as a consequence of the address translations performed by the root memory management units.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> and an identifier to each of a plurality of virtual machines running on a central processing unit; configuring translation tables in a first and a second root memory management unit, wherein the translation tables define translations between guest physical addresses and actual physical memory addresses based on an identifier for the virtual machine, the first root memory management unit being associated with the central processing unit and the second root memory management unit being associated with a graphics processor unit; and during execution of one or more of the virtual machines, tagging transactions output by the central processing unit with the identifier for the virtual machine to which the transaction relates and, within the root memory management units, translating addresses in a transaction based in the identifier with which the transaction is tagged. [0008] Further aspects provide a method of manufacturing, at an integrated circuit manufacturing system, a computing system as described herein, an integrated circuit definition dataset that, when processed in an integrated circuit manufacturing system, configures the system to manufacture a computing system as described herein and a computer readable storage medium having stored thereon an integrated circuit definition dataset that, when processed in an integrated circuit manufacturing system, configures the system to manufacture a computing system as described
> 
> herein.
> 
> [0007] A second aspect provides a method comprising:
> 
> allocating, by a hypervisor, a portion of a memory resource
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%206.png)

GPU支持多VM的简单方法：

1、每个VM分配一个GPU，消耗过多GPU，VM数少于GPU数时资源浪费；

2、每个VM分配一个GPU context，让**CPU管理GPU的context switch**，消耗CPU性能，GPU driver和CPU**耦合**而很难设计；

专利设计：hypervisor为VM分配唯一OS-ID，GPU为每个VM分配专属register，**减少VM交互硬件（transaction）过程中hypervisor的参与**；

> **[图片提取文字 (image.png)]:**
> result of additional calls to and from the hypervisor and/or context switches (e.g. as derived by the hypervisor). Existing solutions to this either provide a dedicated GPU for each VM (i.e. by duplicating the entire GPU hardware) or dedicate one or more contexts to each VM (e.g. where the GPU supports multiple independent contexts). Providing multiple GPUs increases the size (and hence cost) of the system and may be inefficient (e.g. where 8 GPUs are provided and less than 8 VMs are running). Dedicating one or more contexts to each VM (a technique known as 'mediated pass-through') still involves the CPU in scheduling the work for the GPU across the VMs (which introduces latency of the round-trip to the CPU and uses CPU resources) and may require changes to the GPU drivers on the VMs.
> 
> hypervisor. This can introduce a performance penalty as a
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> is allocated an identifier (OS\_ID) by the hypervisor (e.g. on set-up) and then this OS\_ID is used to tag every transaction between a VM and the GPU (i.e. every GPU transaction on the system bus which interconnects the CPU and the GPU). Additionally, dedicated GPU registers are provided for each VM and whilst each VM can only see (i.e. access) their own GPU registers, the GPU can see all the GPU registers. The combination of the OS\_IDs and dedicated GPU registers, enables different jobs (which may be graphics or computing jobs) from different VMs (and hence different operating systems) to run on the GPU concurrently (e.g. in different GPU pipelines) and eliminates the need for every communication from a VM to the GPU to involve the hypervisor. This results in a performance improvement.
> 
> [0019] Described herein is a method of GPU virtualization
> 
> in which each VM (or operating system running on a VM)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%208.png)

## FIG.1 CPU和GPU计算系统

CPU R-MMU映射GPU Regs，让CPU能访问GPU Regs，VMM控制VM对GPU Regs的可见性；

CPU和GPU的R-MMU（guest addr→phy addr）都由CPU的VMM设置和控制。

> **[图片提取文字 (image.png)]:**
> [0020] FIG. 1 is a schematic diagram of a computing system 50 comprising a CPU 52 and a GPU 54. The diagram only shows hardware and so the VMs and hypervisor which run on the CPU are not shown. The GPU 54 comprises a GPU core **56** which comprises the hardware which performs the GPU tasks (e.g. data masters, shading clusters, texture pipelines and data post-processing modules). The GPU 54 also comprises an embedded microprocessor 58 which controls the operation of the GPU and a memory management unit (MMU) 15 within the GPU (and which may be referred to as the GPU guest MMU or GPU G-MMU to distinguish
> 
> it from the root and guest MMUs external to the GPU). The
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> GPU registers 114. [0021] The GPU registers 114 are also accessible to the CPU **52** via the system bus **116** and SOCIF **112** because they are mapped in a MMU (e.g. the CPU root MMU 120) but as described in more detail below, access to different registers is controlled so that each VM (denoted  $VM_0-VM_n$ ) can see a separate dedicated set of registers 126 and the hypervisor can see another set of registers 128 which is not accessible by the VMs. In contrast, the microprocessor 58 within the GPU **54** can see all of the GPU registers **114** (via the SOCIF 112). [0022] The system 50 further comprises three further MMUs: the CPU guest MMU (CPU G-MMU) 118 and two root MMUs (R-MMUs) 120, 122. The two R-MMUs 120, 122 (which may also be referred to as 'system MMUs', 'IP MMUs', 'IO MMUs' or 'stage 2 MMUs') are set up by (and hence may be described as being controlled by) the hypervisor running on the CPU 52. Each MMU 15, 118, 120, 122 performs translation of memory addresses and may also perform other memory management functions. In a more
> 
> GPU **52** further comprises a SoC (system on chip) interface
> 
> (SOCIF) 112 via which the microprocessor 58 can access
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 1
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2011.png)

**MMU和Memory**

CPU G-MMU由VM上运行的OS控制，VMM控制的R-MMU处于更底层管理多个VM；

Memory是提供driver给VM firmware（类似BIOS的底软）的存储设备（如RAM），也包含CPU或GPU私有的内存（硬件保留内存、texture buffer、render buffer等）。

CPU R-MMU生成的地址可能指向**GPU寄存器**（只占据Mapping空间而不占据Memory空间）或**某段内存（Memory）**。

> **[图片提取文字 (image.png)]:**
> the system bus 116 by failing to map the guest physical address to an actual physical address in the CPU R-MMU **120**. [0024] Although FIG. 1 shows the R-MMUs 120, 122 as being an external component to the adjacent CPU/GPU, in other examples, the R-MMUs 120, 122 may be implemented as an integral part of the associated processor (e.g. CPU R-MMU 120 may be implemented as an integral part of the CPU 52 and/or GPU R-MMU 122 may be implemented as an integral part of the GPU 54, e.g. as an integral part of the GPU core 56). [0025] Each of the sets of GPU registers 126 allocated to a VM (and which may be referred to as a 'register block') may, for example, consume 64 kB of space in the system physical address space map. This granularity may be selected to match the granularity of the R-MMUs. It will be
> 
> appreciated that these addresses do not overlap RAM and
> 
> [0023] By controlling the translations performed within
> 
> the R-MMUs 120, 122, the hypervisor controls what trans-
> 
> actions have access to the system bus 116. Although the
> 
> mapping from a virtual address to a guest physical address
> 
> by the CPU G-MMU 118 is managed by the operating
> 
> system running on a VM; the hypervisor can block access to
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2012.png)

> **[图片提取文字 (image.png)]:**
> (just space in the physical address map). [0026] The physical addresses which are generated by the CPU R-MMU 120 may relate to the GPU registers 114 (in which case calls are received by the SOCIF 112, which monitors the range of addresses corresponding to the GPU registers (or a plurality of addresses where the addresses are not contiguous), via the system bus 116) or to memory 124 within the computing system 50, where this memory 124 provides the driver to firmware per VM control interfaces and may comprise RAM. The physical addresses which are generated by the GPU R-MMU 122 may relate to the VM driver/firmware interface memory 124 within the computing system 50. [0027] It will be appreciated that a system 50 will comprise other memory aside from the VM driver/firmware interface memory 124, e.g. other CPU-only memory and CPU/GPU memory containing other GPU setup, textures and render buffers etc. and the address space occupied by the
> 
> memory 124 need not be contiguous (e.g. other memory
> 
> may exist in gaps between adjacent blocks 132).
> 
> hence these 64 kB regions do not actually consume memory
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2013.png)

## FIG.2 虚拟化系统的运行

初始化phase为VM分配内存、GPU Regs和OS-ID，设置R-MMU的page-table，运行phase的VM交互基于OS-ID控制；

> **[图片提取文字 (image.png)]:**
> with reference to the flow diagram shown in FIG. 2 which shows an initialization phase 202 and an operational phase 204. During the initialization phase 202, which may be performed on system set-up or when a new VM is provisioned, the hypervisor allocates memory (e.g. from memory 124 and/or GPU registers 114) to a VM and also allocates an identifier (the OS\_ID) to the VM (block 206). The OS\_IDs may each comprise m bits and in various examples, the OS\_IDs each comprise 3 bits (m=3), enabling each OS\_ID to identify one of a maximum of 8 VMs (although as described below, in various examples one of the OS\_IDs is allocated to the hypervisor). The hypervisor also sets up the translation tables (also referred to as 'page tables') within the R-MMUs 120, 122 (block 208) where the translations between a guest physical address and a real physical address are dependent upon on the OS\_ID and so the translation tables may be indexed or selected by a R-MMU 120, 122 according to the OS\_ID. Having allocated the memory and OS\_ID (in block 206) and set up the R-MMUs (in block 208), the VMs can be loaded (block 25) and can start to execute (block 212).
> 
> [0029] The operation of the system 50 can be described
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2014.png)

> **[图片提取文字 (image.png)]:**
> [0030] During the operational (or execution) phase 204, the VMs execute and each transaction on the system bus 116 is identified by the OS\_ID of the VM to which it relates (block 214). As described below, the hypervisor may also be allocated its own OS ID so that it can be used to determine which parts of the memory the hypervisor can access. As each transaction is tagged with the OS\_ID, OS\_IDs effectively provide m more bits of address space. Transactions which do not originate from the VM inherit their OS\_ID from the triggering transaction, such that output from the
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> GPU inherits the OS ID of the VM that submitted the GPU task that generated the output.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2017.png)

OS-ID和R-MMU的协作，翻译和隔离不同VM的内存访问，减少VMM作为交互中介的负担。

VM分配的GPU Regs中设置**kick register**能接受VM访问，并且触发GPU中微处理器；

Memory中**特定区域**设置为**circular Buffer**，记为**host driver/firmware interface（HD/FI）**。**HD/FI**是device提供给host的驱动程序或firmware接口，或者传递host命令的kernel command Buffer。

kick Regs和circular Buffer在初始化时分配给VM。VMM初始化CPU和GPU的R-MMU：kick reg的虚拟地址（**host进程接口**）和guest物理地址（**VMM管理信息**）的**mapping**，host驱动接口和其物理地址的mapping。

> **[图片提取文字 (image.png)]:**
> read from and/or write to). In this way, each VM sees the same set of registers in the same location (i.e. the virtual addresses of these registers may be the same and these may be mapped to the same guest physical addresses by the CPU G-MMU), but cannot see registers in pages mapped to other VMs by the R-MMUs. [0032] The use of the OS\_IDs in combination with the R-MMUs which have been set-up by the hypervisor provides an efficient way to switch between different VMs (and hence different operating systems) as it is not necessary for the hypervisor to mediate every transaction in order to identify the VM to which it relates and to enforce security measures to protect rogue memory accesses (e.g. accesses by one VM to memory written by other VMs or other parts of the system). As described above, access to the system bus 116 can be blocked by failing to map a guest physical address to an actual physical address (in a R-MMU) and this is configured during the initialization phase 52 rather than needing to be performed on a transaction by transaction basis involving the hypervisor each time.
> 
> [0031] The OS\_IDs are used by the R-MMUs 120, 122 to
> 
> select the appropriate pages (block 216) and this controls
> 
> which parts of the memory each VM can access (e.g. can
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> [0033] In order that individual VMs can independently submit GPU tasks to the GPU 54 without needing to involve the hypervisor, one of the registers in each dedicated set of registers 126 may be a special register 130 (referred to herein as a 'kick register') which triggers an event in the embedded microprocessor **58** within the GPU **54**. In various examples there may only be a single register in the 64 kB address range which corresponds to the set of registers 126. Additionally dedicated portions 132 of the memory 124 operate as circular buffers (which may be referred to as host driver/firmware interfaces and comprise kernel command circular buffers, CCBs) for each VM (denoted HD/FI<sub>0</sub>-HD/ FI<sub>n</sub>) and these operate as a command interface to the GPU with each VM only having access to its own host driver/ firmware interface 132. As with the GPU registers 114, the microprocessor 58 in the GPU 54 can see all of the host driver/firmware interfaces 132, although the access mechanisms are different (the memory 124 is accessed via the GPU R-MMU 122 and the GPU registers 114 are accessed via the SOCIF 112). The operation of these host driver/firmware interfaces 132 and kick registers 130 can be described with reference to the flow diagram shown in FIG. 3. [0034] The kick registers and the host driver/firmware
> 
> [0034] The kick registers and the host driver/firmware interfaces are allocated to a VM on initialization, e.g. in block 206 in FIG. 2. The mapping between the virtual and guest physical addresses for the kick registers and host driver/firmware interfaces and their actual physical addresses is also set up within the R-MMUs by the hypervisor as part of the initialization phase 202 (e.g. in block 208).
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2019.png)

## FIG.3 VM使用kick Regs和Circular Buffer

VM提交命令到GPU前的**准备**：

VM在内存中设置控制流。

VM通过HD驱动/FI接口的VA写命令，VA经过CPU G-MMU翻译成GPA。

CPU R-MMU将HD/FI的GPA翻译成实际物理地址PA。

命令保存到HD/FI中。

VM**提交命令**到GPU：

VM写kick reg。

SOCIF监控相关地址区间，包括kick Reg的写入。

GPU微控制器生成事件event（信号触发）和任务task（执行）。

读取kick reg中VM的OS-ID，基于OS-ID解读VM所用的HD/FI地址。

微控制器从HD/FI中读取命令，设置GPU负载workload。

GPU执行负载workload，完成后给出中断信号。

当GPU监控到Kick Reg，但是当前资源被占用时，要么**ignored**不执行，要么**stored**等待执行，要么**interrupt**抢占资源执行（workload包含host命令kick，和被中断workload的恢复）。

> **[图片提取文字 (image.png)]:**
> [0035] When a VM (e.g. a graphical application within the VM) wishes to trigger a GPU task, the VM sets up the control streams in memory (block 302) and this may be done in a conventional manner, e.g. as if the VM was a native CPU with an associated GPU. The VM then writes a command to a generic command circular buffer (block 304), i.e. each VM running on the CPU can write a command using the same virtual address and guest physical address, and the guest physical address is then mapped to the VM
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2020.png)

> **[图片提取文字 (image.png)]:**
> specific physical address for the HD/FI by the CPU R-MMU 120 (block 306) so that the command can be stored (block 308). This setting up of the control stream (in block 302) and storing the command (in blocks 304-308) can be performed ahead of when the VM wants to submit the GPU task to the GPU.
> 
> [0036] To submit the GPU task to the GPU, the VM writes
> 
> to a generic kick register (block 35) which may, in various examples, be the same for all VMs running on the CPU, i.e. each VM running on the CPU may, in various examples, use the same virtual and guest physical addresses for the kick register. As with the HD/FI address, the kick register address is mapped to a physical address for the kick register for the correct VM by the CPU R-MMU 120 (block 312) and this may be based on the OS\_ID which is received by the CPU R-MMU 120 along with the write command or may use another mechanism which is present in the CPU to select the right mappings based on the current VM that is executing. As described above, the SOCIF 112 monitors the address range corresponding to the GPU registers 114 and so detects the write (block 314). As a result of the write to the VM specific kick register 130, an event is generated in the microprocessor 58 within the GPU 54 (block 316) and this triggers the running of a task by the microprocessor 58 (block 318). The OS\_ID for the calling VM (which is derived directly from the kick register written to and which may be stored in a register internal to the microprocessor 58) is passed to the task which is running in the microprocessor (block **320**), e.g. as side band information, and the OS\_ID is then used to perform a translation and identify the address of the HD/FI for that particular VM (block 322), i.e. the OS\_ID is used to calculate the right address for the VM, e.g. by offsetting the accesses to the right VM interface memory (as described in more detail below). This means that the GPU knows which VM triggered the event without needing to trust the VM to correctly identify itself or ask the hypervisor since each VM only has visibility to its own kick register by means of the CPU R-MMU. Having identified the correct HD/FI (in block 322), all necessary information to perform the GPU task can be read from memory (e.g. the control streams written in block 302 and the command written in blocks 304-308). The microprocessor 58 can read the previously written command from the HD/FI (block 324) and then based on the command, access all necessary data.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2022.png)

> **[图片提取文字 (image.png)]:**
> 324) and if the GPU has capacity (e.g. if the particular pipeline within the GPU core that is required is not already executing another GPU task), the workload can be set up in the GPU by the microprocessor (block 326) and then the GPU can start the workload (block 328) without further involvement of the microprocessor. [0038] If the GPU (e.g. the required GPU pipeline) is already busy when the kick is received (e.g. when the command is read in block 324), then it may be ignored by the microprocessor 58, in which case the workload will not be set up and the task will not be performed. In some example implementations, however, a record of the kicks received but not actioned may be stored so that when capacity within the GPU core is available a decision can be made (e.g. by a scheduler within the microprocessor) as to which GPU task to action next. In various implementations, the completion of a GPU task may trigger an interrupt to the microprocessor (block 320) e.g. so that resources used in the task can be freed up for use by another GPU task and this
> 
> [0037] Based on the command which is read (in block
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2023.png)

> **[图片提取文字 (image.png)]:**
> interrupt on completion may in turn trigger the start of a next GPU workload, e.g. such that the GPU tasks are triggered not by the kicks but by the completion of a previous GPU task (e.g. a previous GPU task utilizing the same resources as the triggered GPU task).
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2024.png)

## FIG.4 Host内存地址空间（Guests Mem）

每个host VM在地址空间有一个VMi部分来映射对应的GPU Regs，占据地址空间而不占内存，CPU R-MMU将VMi部分的地址访问转发到GPU Regs。

对于host而言，每个VM中的device驱动程序和命令接口是**不可信软件**。在host内存中，每个VM中不可信软件的**可运行地址空间是MEMi**，**HD/FIi（灰色部分）**存放host VM和GPU交互的驱动指令和命令。

OS-ID被VM共享时，**VM执行VMM call或者GPU微控制器执行firmware code来搜索OS-ID**，**HD/FI中命令**包含MEMi中指针或者offset。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2025.png)

> **[图片提取文字 (image.png)]:**
> [0041] In a further example, a hybrid of the two approaches (dedicated kick register per VM and single kick register) may be used. In such an example, where there are more VMs than OS\_IDs, one of the kick registers may be used by all VMs exceeding the OS\_ID limit (e.g. for all those VMs which do not have their own unique OS\_ID). In such an implementation, these VMs do make a hypervisor call or the firmware code running on the microprocessor 58 knows to scan N VM queues for this special case OS\_ID.
> 
> [0042] The command which is written into the HD/FI (in blocks 304-306) may include a reference to the memory address at which the data has been stored in memory (in
> 
> address at which the data has been stored in memory (in block 302). The reference to the memory address may be in the form of a pointer. In other examples, however, an offset may be specified instead of a pointer in order to provide a more secure interface, as can be described with reference to FIG. 4.
> 
> [0043] FIG. 4 shows a schematic diagram of the memory address space, which includes the addresses corresponding
> 
> to the memory 124 in more detail. In this example, the memory addresses 402 to which the GPU registers 114 are mapped (in the CPU R-MMU 120) are shown along with the HD/FIs 132; however, it will be appreciated that whilst these GPU registers 114 occupy memory address space they do not occupy physical memory (i.e. they are not stored in
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2026.png)

> **[图片提取文字 (image.png)]:**
> RAM). Additionally portions of memory 404 which are reserved for each VM (denoted MEM<sub>0</sub>-MEM<sub>n</sub>) are shown in FIG. 4 and the HD/FIs 132 (and the CCBs within the HD/FIs) are sub-structures within the portions of memory 404. These portions of memory 404 are contiguous in virtual address space, so they appear contiguous to the microprocessor 58 and so instead of providing a pointer to the start of any data, an offset can be provided and the microprocessor 58 can calculate the actual memory address using:
> 
> Address=base address+(OS\_ID×portion size)+offset
> 
> and where the base address, B, is the starting address for the portion of memory and the portion size, C, is the size of a portion, as shown in FIG. 4. In this way, non-trusted software can never generate pointers outside of their own secure container ( $MEM_0$ - $MEM_n$ ). The caller only supplies the offset which internally has the secure container base address applied (as above) as well as a range check to ensure that the offset does not overflow the secure container range from its base.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2027.png)

**CPU OS-ID，GPU OS-ID**：guest地址空间的不同区域映射不同OS-ID，被不同角色（VMM、VM、GPU）访问。

**kick Regs**允许VM直接命令GPU，相互隔离而不需要通过VMM中转，OS-ID让GPU**标记**每个任务而并行执行。

GPU虚拟化中，**每个VM有自己的驱动栈**，因此每个VM独占一个vGPU，节省SoC中实例化多个GPU的开销。因为虚拟化，**VM切换GPU**时，没有软件开销。

> **[图片提取文字 (image.png)]:**
> [0047] In various examples, the OS\_ID which is passed by the CPU (and corresponds to calling VM) may be denoted CPU OS\_ID and the OS\_ID which is passed by the GPU (and corresponds to the VM being called) may be denoted GPU OS\_ID. The hypervisor running on the CPU may have its own allocated OS\_ID and these different OS\_IDs may be used to control access to different portions of the memory 124 (or memory other than memory 124, as described above). For example, portions which are only accessible to the hypervisor and the GPU may only be mapped from/to guest physical addresses accompanied by a pre-defined CPU OS\_ID or GPU OS\_ID which corresponds to the hypervisor, portions which are available to the GPU and trusted software running on the CPU (i.e. not untrusted VMs) may only be mapped from/to guest physical addresses accompanied by a pre-defined CPU OS\_ID or GPU OS\_ID which corresponds
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2028.png)

> **[图片提取文字 (image.png)]:**
> to a trusted execution environment and portions which are only accessible to the CPU (and not the GPU) may only be mapped from/to guest physical addresses accompanied by a pre-defined CPU OS\_ID which corresponds to the trusted execution environment (and not a corresponding GPU OS\_ID).
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2029.png)

> **[图片提取文字 (image.png)]:**
> [0048] The 'special' registers described above (e.g. register sets 126 and/or kick registers 130) enable untrusted software (e.g. the VMs or software running on the VMs) to talk directly to the GPU. Although the GPU does not trust any caller identity information provided by the VMs themselves (because the VMs are untrusted), the use of a single kick register or a dedicated kick registers for each VM (or a hybrid of these two approaches, as described above) provides a mechanism that cannot be spoofed by a rogue VM and does not require the hypervisor call to perform the 'kick' operation. This therefore enables each VM (or each OS running on a VM) to independently queue work on the GPU without hypervisor intervention and the use of the OS\_ID to tag all transactions enables GPU tasks from different VMs to be run in parallel on different GPU pipelines.
> 
> [0049] As described above, although there is a single GPU shared between multiple VMs, each VM has its own driver stack (i.e. a full standard driver stack as if it was a single native machine) so to each VM it appears to have a dedicated virtualized GPU. This represents a cost saving (e.g. in terms of chip area and efficiency) compared to a SoC with multiple GPUs, e.g. there is less redundant GPU hardware with one virtualized GPU rather than N dedicated GPUs. Furthermore, using the methods described above there is zero software overhead in switching the GPU across VMs.
> 
> [0050] Using the methods and apparatus described above,
> 
> GPU (e.g. the firmware runs inside the GPU) and so this is transparent to the CPU and to the VMs running on the CPU. This is in contrast to mediated pass-through where the mediation between the 'virtual GPUs' happens in the host
> 
> CPU.
> 
> the virtualization of the GPU is mediated inside within the
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2030.png)

## FIG.5：IC量产

IC dataset的形式是HDL如RTL，RTL设计综合出门级电路gate level，门级电路确定位置后得到circuit definition，生成mask（光刻）后生成集成电路IC。

> **[图片提取文字 (image.png)]:**
> one location, e.g. by one party. Alternatively, the IC manufacturing system 502 may be a distributed system such that some of the processes may be performed at different locations, and may be performed by different parties. For example, some of the stages of: (i) synthesising RTL code representing the IC definition dataset to form a gate level representation of a circuit to be generated, (ii) generating a circuit layout based on the gate level representation, (iii) forming a mask in accordance with the circuit layout, and (iv) fabricating an integrated circuit using the mask, may be performed in different locations and/or by different parties.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2032.png)

# GPU Virtualization Hardware（vGPC）

任务在GPU的GPC之间**灵活迁移Migration**。

设置floor-swept/ disabled/ nonfuctional（低配版芯片）或者全算力（满血版芯片）。

**动态资源关闭**。

使用**vTPC**而非物理TPC来分配资源，通过Migration来让不同物理分布TPC支持统一的SKU定义。

每个物理GPU，将不同MIG划分时需要“换号”Migration的TPC设置为**Singleton**单例，物理设置其支持metal和virtual双模，剩余TPC只支持metal单模（直接1to1迁移）。

> **[图片提取文字 (image.png)]:**
> [0012] The technology herein relates to integrated circuit design, and more particularly to solving problems relating to manufacturing defects in complex chips including but not limited to graphics processing units (GPUs). The technology further relates to defining virtual GPU processing clusters that are abstractions of logical or physical circuits to provide compatibility between differently structured chips; flexible migration between GPU processing clusters and processing components thereof; taking into account balance of floorswept/disabled/nonfunctional versus fully functional hardware across an integrated circuit substrate; and dynamic processing resource disablement that allows hardware to be selectively turned off when not needed.
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2033.png)

## ABS & BG

## FIG.1、2、3、4、5

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> FIG. 1
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 3
> Prior Art GPU Hardware Partitions
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Prior Art GPU Hardware With Graphics Processing Clusters
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2038.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5A
> Prior Art μGPU Partitions
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2040.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 5B
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2041.png)

> **[图片提取文字 (image.png)]:**
> All CTAs in a grid run on same GPU but may run on different SMs
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 5C
> 
> Prior Art Hierarchy
> Mapping Onto GPU Hardware Partitions
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2042.png)

## FIG.6、7、8、9

> **[图片提取文字 (image.png)]:**
> ## **Example Product using 3 Configurations**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Defect-free TPC
> 
> Defective TPC
> 
> used in the product SKU
> 
> not used in the product SKU
> 
> Single Product SKU
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2043.png)

> **[图片提取文字 (Picture7.png)]:**
> ## **Singletons**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Single Product SKU
![Picture7.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture7.png)

> **[图片提取文字 (Picture8.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 8
![Picture8.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture8.png)

> **[图片提取文字 (Picture9.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 9
![Picture9.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture9.png)

## FIG.10、11、12、13、14

> **[图片提取文字 (Picture10.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 10B (Prior Art)
![Picture10.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture10.png)

> **[图片提取文字 (Picture11.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 11
![Picture11.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture11.png)

> **[图片提取文字 (Picture12.png)]:**
> | 6/8/8/8/9/9/9/9/1x2  | 6/7/8/8/9/9/9/9/1x3                                                                                                                                                  | 5/8/8/8/9/9/9/9/1x3                                                                                                                                                                                                                                                                        | 6/8/8/8/8/8/9/9/1x2                                                                                                                                                                                                                                                                                                                                                                                           | 6/6/8/8/8/8/9/9/1x4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 5/8/8/8/8/8/9/9/1x3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 5/6/8/8/8/8/9/9/1x5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
> |----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
> | 6/6/8/8/8/8/8/8/1x4  | 5/8/8/8/8/8/8/8/1x3                                                                                                                                                  | 5/5/8/8/8/8/8/8/1x6                                                                                                                                                                                                                                                                        | 6/6/6/7/7/8/8/8/1x6                                                                                                                                                                                                                                                                                                                                                                                           | 6/7/7/7/7/8/8/8/1x4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 6/6/8/8/8/8/8/8/1x2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 6/6/7/7/7/8/8/8/1×5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
> | 5/5/8/8/8/8/8/8/1x4  | 5/5/7/7/7/8/8/8/1x7                                                                                                                                                  | 6/6/6/6/7/7/8/8/1x6                                                                                                                                                                                                                                                                        | 6/7/7/7/7/8/8/1x3                                                                                                                                                                                                                                                                                                                                                                                             | 6/6/7/7/7/7/8/8/1×4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 5/7/7/7/7/8/8/1x4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 5/5/8/8/8/8/8/8/8/1x2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
> | 6/6/6/6/7/7/7/8/1x5  | 6/7/7/7/7/7/8/1x2                                                                                                                                                    | 6/6/7/7/7/7/8/1x3                                                                                                                                                                                                                                                                          | 5/7/7/7/7/7/8/1x3                                                                                                                                                                                                                                                                                                                                                                                             | 5/5/7/7/7/7/8/1x5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 6/6/6/6/6/7/7/7/1xS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 6/6/7/7/7/7/7/7/1x2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
> | 5/5/7/7/7/7/7/1x4    | 0/6/8/8/8/8/8/8/1×8                                                                                                                                                  | 0/6/7/7/7/8/8/8/1x11                                                                                                                                                                                                                                                                       | 0/5/8/8/8/8/8/8/8/1x9                                                                                                                                                                                                                                                                                                                                                                                         | 0/5/7/7/7/8/8/8/1x12                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0/6/6/8/8/8/8/8/1x10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 0/6/6/7/7/8/8/8/1x12                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
> | 0/5/6/7/7/8/8/8/1x13 | 0/6/8/8/8/8/8/8/1x6                                                                                                                                                  | 0/6/7/7/7/7/8/8/1x10                                                                                                                                                                                                                                                                       | 0/5/8/8/8/8/8/8/1x7                                                                                                                                                                                                                                                                                                                                                                                           | 0/5/7/7/7/8/8/1x11                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 0/6/6/8/8/8/8/8/1x8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0/6/6/7/7/7/8/8/1x11                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
> | 0/5/5/7/7/7/8/8/1x13 | 0/6/6/6/7/7/8/8/1x12                                                                                                                                                 | 0/5/5/6/7/7/8/8/1x14                                                                                                                                                                                                                                                                       | 0/6/8/8/8/8/8/8/1×4                                                                                                                                                                                                                                                                                                                                                                                           | 0/6/7/7/7/7/8/1x9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 0/5/8/8/8/8/8/8/1x5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0/5/7/7/7/7/8/1x10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
> | 0/6/6/7/7/7/7/8/1x10 | 0/5/5/8/8/8/8/8/1x8                                                                                                                                                  | 0/5/5/7/7/7/8/1x12                                                                                                                                                                                                                                                                         | 0/6/6/6/7/7/7/8/1x11                                                                                                                                                                                                                                                                                                                                                                                          | 0/5/5/6/7/7/7/8/1x13                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0/6/8/8/8/8/8/8/1x2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0/6/7/7/7/7/7/1×8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
> | 0/5/7/7/7/7/7/1x9    | 0/6/6/8/8/8/8/8/8/1x4                                                                                                                                                | 0/6/6/7/7/7/7/7/1x9                                                                                                                                                                                                                                                                        | 0/5/5/8/8/8/8/8/1×6                                                                                                                                                                                                                                                                                                                                                                                           | 0/5/5/7/7/7/7/1x11                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 0/6/6/6/6/7/7/7/1x11                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 0/5/5/5/6/7/7/7/1x14                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
> |                      | 6/6/8/8/8/8/8/8/8/1x4<br>5/5/8/8/8/8/8/8/8/1x4<br>6/6/6/6/7/7/7/8/1x5<br>5/5/7/7/7/7/7/7/1x4<br>0/5/6/7/7/8/8/8/1x13<br>0/5/5/7/7/7/8/8/1x13<br>0/6/6/7/7/7/8/8/1x10 | 6/6/8/8/8/8/8/8/8/1x4 5/8/8/8/8/8/8/8/1x3 5/5/8/8/8/8/8/8/8/1x4 5/5/7/7/7/8/8/8/1x7 6/6/6/6/7/7/7/8/1x5 6/7/7/7/7/7/8/1x2 5/5/7/7/7/7/7/1x4 0/6/8/8/8/8/8/8/1x8 0/5/6/7/7/8/8/8/1x13 0/6/8/8/8/8/8/8/1x6 0/5/5/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/6/6/7/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 | 6/6/8/8/8/8/8/1x4 5/8/8/8/8/8/1x3 5/5/8/8/8/8/8/1x6 5/5/8/8/8/8/8/1x4 5/5/7/7/8/8/8/1x7 6/6/6/6/7/7/8/8/1x6 6/6/6/6/7/7/7/8/1x5 6/7/7/7/7/8/1x2 6/6/7/7/7/7/8/1x3 5/5/7/7/7/7/7/1x4 0/6/8/8/8/8/8/1x8 0/6/7/7/7/8/8/8/1x11 0/5/6/7/7/8/8/8/1x13 0/6/8/8/8/8/8/8/1x6 0/6/7/7/7/8/8/1x10 0/5/5/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/8/1x14 0/6/6/7/7/7/8/8/1x10 0/5/5/8/8/8/8/1x8 0/5/5/7/7/8/8/1x14 | 6/6/8/8/8/8/8/1x4 5/8/8/8/8/8/1x3 5/5/8/8/8/8/8/1x6 6/6/6/7/7/8/8/8/1x6 5/5/8/8/8/8/8/1x4 5/5/7/7/8/8/8/1x7 6/6/6/6/7/7/8/8/1x6 6/7/7/7/7/8/8/1x3 6/6/6/6/7/7/7/8/1x5 6/7/7/7/7/7/8/1x2 6/6/7/7/7/7/8/1x3 5/7/7/7/7/7/8/1x3 5/5/7/7/7/7/1x4 0/6/8/8/8/8/8/1x8 0/6/7/7/7/8/8/8/1x11 0/5/8/8/8/8/8/8/1x9 0/5/6/7/7/8/8/8/1x13 0/6/8/8/8/8/8/1x6 0/6/7/7/7/8/8/1x10 0/5/8/8/8/8/8/8/1x7 0/5/5/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/8/1x14 0/6/8/8/8/8/8/1x4 0/6/6/7/7/7/8/8/1x10 0/5/8/8/8/8/8/8/1x4 | 6/6/8/8/8/8/8/8/1x4 5/8/8/8/8/8/1x3 5/5/8/8/8/8/8/1x6 6/6/6/7/7/8/8/8/1x6 6/7/7/7/8/8/8/1x4 5/5/8/8/8/8/8/1x7 6/6/6/6/7/7/8/8/1x6 6/7/7/7/7/8/8/1x3 6/6/7/7/7/8/8/1x4 6/6/6/6/7/7/7/8/1x5 6/7/7/7/7/8/1x2 6/6/7/7/7/7/8/1x3 5/5/7/7/7/8/1x3 5/5/7/7/7/7/8/1x5 5/5/7/7/7/7/7/1x4 0/6/8/8/8/8/8/1x8 0/6/7/7/7/8/8/8/1x11 0/5/8/8/8/8/1x9 0/5/7/7/7/8/8/8/1x12 0/5/6/7/7/8/8/8/1x13 0/6/8/8/8/8/8/1x6 0/6/7/7/7/8/8/1x10 0/5/8/8/8/8/8/1x7 0/5/7/7/7/8/8/1x11 0/5/5/7/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/8/1x14 0/6/8/8/8/8/8/1x4 0/6/7/7/7/8/1x9 0/6/6/6/7/7/7/8/8/1x10 0/5/5/8/8/8/8/8/1x4 0/6/7/7/7/8/1x13 | 6/6/8/8/8/8/8/1x4 5/8/8/8/8/8/1x3 5/5/8/8/8/8/8/1x6 6/6/6/7/7/8/8/8/1x6 6/7/7/7/8/8/8/1x4 6/6/8/8/8/8/8/1x2 5/5/8/8/8/8/8/1x4 5/5/7/7/8/8/8/1x7 6/6/6/6/7/7/8/8/1x6 6/7/7/7/8/8/1x3 6/6/7/7/7/8/8/1x4 5/7/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/7/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/7/7/7/7/8/8/1x4 5/5/7/7/7/8/8/1x4 5/7/7/7/7/8/8/1x4 5/7/7/7/8/8/1x4 5/7/7/7/8/8/1x4 5/7/7/7/8/8/1x4 6/6/6/6/6/7/7/7/8/8/1x3 5/5/7/7/7/8/1x5 6/6/6/6/6/7/7/1x5 5/5/7/7/7/7/8/1x5 6/6/6/6/6/7/7/1x5 6/6/6/6/6/7/7/1x5 5/5/7/7/7/1x4 0/6/8/8/8/8/8/1x8 0/6/7/7/8/8/8/1x11 0/5/8/8/8/8/8/1x9 0/5/7/7/8/8/8/1x12 0/6/6/8/8/8/8/8/1x10 0/5/8/8/8/8/8/1x3 0/6/8/8/8/8/8/1x6 0/6/7/7/7/8/8/1x10 0/5/8/8/8/8/8/8/1x7 0/5/7/7/7/8/8/1x11 0/6/6/8/8/8/8/8/1x8 0/5/5/7/7/8/8/1x13 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/8/1x14 0/6/8/8/8/8/8/8/1x4 0/6/7/7/7/7/8/1x13 0/5/8/8/8/8/8/1x5 0/6/6/7/7/7/8/1x10 0/5/5/8/8/8/8/8/1x3 0/6/6/6/7/7/8/8/1x12 0/5/5/6/7/7/8/1x11 0/5/5/6/7/7/8/1x13 0/6/8/8/8/8/8/1x2 |
> 
> FIG. 12
> 
> | 6/8/8/8/9/9/9/9/1x2  | 3 Configurations  |
> |----------------------|-------------------|
> | 6/6/8/8/8/8/9/9/1x4  | 6 Configurations  |
> | 5/6/8/8/8/8/9/9/1x5  | 8 Configurations  |
> | 6/6/6/7/7/8/8/8/1x6  | 12 Configurations |
> | 6/6/7/7/8/8/8/0/1x12 | 24 Configurations |
> | 6/6/6/7/7/8/8/0/1x12 | 32 Configurations |
> | 5/5/5/6/7/7/7/0/1x14 | 63 Configurations |
> 
> FIG. 13
![Picture12.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture12.png)

> **[图片提取文字 (Picture13.png)]:**
> ## Various Skylines in Various SKUs
> 
> ![](_page_0_Figure_1.jpeg)
> 
> **FIG. 13A**
![Picture13.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture13.png)

> **[图片提取文字 (Picture14.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 14**
![Picture14.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture14.png)

> **[图片提取文字 (Picture15.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 14A**
![Picture15.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture15.png)

> **[图片提取文字 (Picture16.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 14B**
![Picture16.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture16.png)

> **[图片提取文字 (Picture17.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 14C
![Picture17.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture17.png)

## FIG.15、16、17、18

> **[图片提取文字 (Picture18.png)]:**
> ## Physical CSM map in CWD
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 15
![Picture18.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture18.png)

> **[图片提取文字 (Picture19.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 16
![Picture19.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture19.png)

[https://devblogs.microsoft.com/pix/hardware-counters-in-gpu-captures/](https://devblogs.microsoft.com/pix/hardware-counters-in-gpu-captures/)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2044.png)

> **[图片提取文字 (Picture20.png)]:**
> | 3 | 6 | 9         | 11 | 12 13 14 15 |
> |---|---|-----------|----|-------------|
> | 2 | 5 | 8         | 10 |             |
> | 1 | 4 | 7         |    |             |
> | 0 |   | المشمشمين |    |             |
> 
> FIG. 17
> 
> ![](_page_0_Picture_3.jpeg)
> 
> FIG. 18
![Picture20.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture20.png)

> **[图片提取文字 (Picture21.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 18A** 
> 
> Good TPC
> 
> in the Multi-TPC CGA Region
> 
> encompasses TPCs included
> 
> Good TPC used as a Singleton
> 
> GPU Instance
![Picture21.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture21.png)

## FIG.19、20、21、22、23

> **[图片提取文字 (Picture22.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 19
![Picture22.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture22.png)

> **[图片提取文字 (Picture23.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 20
![Picture23.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture23.png)

CSM：配置Singleton和Multiple TPC。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 21A**
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2046.png)

Scheduler（TMU）和CWD（WDU）。

> **[图片提取文字 (image.png)]:**
> [0250] Example Improved CWD Circuit Implementation In an embodiment herein, the CWD 420 comprises registers, combinatorial logic and a hardware state machine. See for example 20200043123 and in particular FIG. 7 of that patent publication and associated description for more information on an example GPU CWD and MPC for scheduling work. Its functionality is expanded/enhanced to provide a shadow state simulated CGA launch capability to confirm that resources are available to launch all CTAs in a CGA. If all CTAs of a CGA cannot be launched at the same time, then the CWD 420 does not launch any of the CTAs of the CGA, but instead waits until sufficient resources of the relevant GPU hardware domain become available so that all CTAs of the CGA can be launched so they run concurrently. In example embodiments, the CWD 420 supports nesting of multiple levels of CGAs (e.g., multiple GPC-CGAs within a GPU-CGA) using a multi-level work distribution architecture to provide CGA launch on associated hardware affinity/domain.
> 
> [0252] In more detail, CWD 420 shown in FIG. 21A launches the CTAs in a CGA after determining, using a simulation technique, that all CTAs of the CGA can fit on the hardware resources available in the specified hardware domain. In this way, CWD 420 in one example mode makes sure there are enough resources across all GPCs or other relevant hardware domain for all CTAs of the CGA before
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2047.png)

> **[图片提取文字 (image.png)]:**
> In example embodiments, a scheduler 410 within the GPU receives tasks from the CPU 212 and sends them to the CWD 420 (FIG. 21C-1, blocks 502, 504). The CWD 420 queries and launches CTAs from multiple CGAs. In one embodiment, it works on one CGA at a time. For each CGA, CWD 420 simulates launching of all of the CTAs in the CGA, incrementing the "launch" registers to store the simulated launch. If all free slots in SMs or other processors in the hardware domain are exhausted before all CTAs of the CGA are launched in the simulation, the CWD 420 terminates the launch and may try again later. If, in contrast, there are sufficient free slots for all CTAs in the CGA, the CWD 420 generates sm\_masks from the "launch" registers accumulated in the simulated launch process (this sm\_masks data structure stores reservation information for the number of CTAs to be run on each SM in the relevant hardware domain for the CGA launch), and moves on to a next CGA. The hardware allocates a CGA sequential number and attaches it to each sm mask. It also attaches an end of CGA bit to the last one to prevent interleaving of sm\_masks from different CGAs.
> 
> [0259] GPU CGA Scheduling & Launch
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2048.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 21C-1
![image.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/image%2049.png)

> **[图片提取文字 (Picture24.png)]:**
> ![](_page_0_Figure_0.jpeg)
![Picture24.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture24.png)

> **[图片提取文字 (Picture25.png)]:**
> | ,        |           |            |           |           |      |           |            |
> |----------|-----------|------------|-----------|-----------|------|-----------|------------|
> | 8/11     | 8/<br>/14 | 8/<br>/17  | 8/<br>/19 | 8/<br>/20 | 8/21 | 8/<br>/22 | 8 /<br>/23 |
> | 7<br>/10 | 7/<br>/13 | 7<br>/16   | 7<br>/18  | 7         | 7    | 7         | 7          |
> | 6/9      | 6/<br>/12 | 6 /<br>/15 | 6         | 6         | 6    | 6         | 6          |
> | 5/8      | 5         | 5          | 5         | 5         | 5    | 5         | 5          |
> | 4        | 4         | 4          | 4         | 4         | 4    | 4         | 4          |
> | 3        | 3         | 3          | 3         | 3         | 3    | 3         | 3          |
> | 2        | 2         | 2          | 2         | 2         | 2    | 2         | 2          |
> | 1        | 1         | 1          | 1         | 1         | 1    | 1         | 1          |
> | 0        | 0         | 0          | 0         | 0         | 0    | 0         | 0          |
> | vGPC     | vGPC      | vGPC       | VGPC      | vGPC      | vGPC | vGPC      | vGPC       |
> | 0        | 1         | 2          | 3         | 4         | 5    | 6         | 7          |
> | 1/2      |           |            |           | full      |      |           |            |
> 
> ![](_page_0_Picture_1.jpeg)
> 
> This CSM has selectable personality.
> It is either migratable TPC id "m" within its vGPC 0 to 7,
> Or it is singleton vGPC "v" within range 8 to 23.
> 
> m
> 
> This CSM is always migratable TPC id "m" within its vGPC 0 to 7
> 
> FIG. 22
![Picture25.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture25.png)

> **[图片提取文字 (Picture26.png)]:**
> ![](_page_0_Figure_0.jpeg)
![Picture26.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture26.png)

## FIG.24、25、26、27、28、29

> **[图片提取文字 (Picture27.png)]:**
> ## Flexible TPC Migration
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 24
![Picture27.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture27.png)

> **[图片提取文字 (Picture28.png)]:**
> ## Multiple-Instanced GPU
> 
> ![](_page_0_Figure_1.jpeg)
> 
> **FIG. 25A** 
> 
> **FIG. 25B** 
> 
> FIG. 25C
![Picture28.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture28.png)

> **[图片提取文字 (Picture29.png)]:**
> ## Multiple-Instanced GPU with Migration
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 26
![Picture29.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture29.png)

> **[图片提取文字 (Picture30.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 26A**
![Picture30.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture30.png)

> **[图片提取文字 (Picture31.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 26B**
![Picture31.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture31.png)

> **[图片提取文字 (Picture32.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> FIG. 26C
> 
> | Defect-free TPC used in the product SKU   |
> |-------------------------------------------|
> | Defective TPC not used in the product SKU |
> | Defect-free TPC used as a Singleton       |
> | GPU Instance encompasses TPCs included    |
> | Graphics TPU                              |
![Picture32.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture32.png)

逻辑编号不能解除任务分配和GPC和TPC之间物理关系的依赖，虚拟层能解除。

> **[图片提取文字 (Picture33.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 26D** 
> 
> Defect-free TPC
> 
> Defective TPC
> 
> Defect-free TPC
> 
> **GPU** Instance
> 
> Graphics TPU
> 
> used as a Singleton
> 
> used in the product SKU
> 
> not used in the product SKU
> 
> encompasses TPCs included
![Picture33.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture33.png)

> **[图片提取文字 (Picture34.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **FIG. 27**
![Picture34.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture34.png)

> **[图片提取文字 (Picture35.png)]:**
> ## **Barrier Table Organization**
> 
> ![](_page_0_Figure_1.jpeg)
![Picture35.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture35.png)

> **[图片提取文字 (Picture36.png)]:**
> ## MULTI-INSTANCE GPU ("MIG")
> 
> ![](_page_0_Figure_1.jpeg)
> 
> FIG. 29
![Picture36.png](GPU%20Virtualization%E5%92%8CMigration%EF%BC%88%E6%94%AF%E6%8C%81VM%E7%BA%A7%E8%B4%9F%E8%BD%BD%E3%80%81vGPC%E8%99%9A%E6%8B%9F%E5%8C%96%EF%BC%89/Picture36.png)

# Docker Virtualization