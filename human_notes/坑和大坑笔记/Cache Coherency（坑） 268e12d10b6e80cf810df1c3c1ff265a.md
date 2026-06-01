# Cache Coherency（坑）

[https://01.me/2023/08/cache-coherency/](https://01.me/2023/08/cache-coherency/)

我设计的重排方法能否应用在CC中，改进CC性能之类？多任务？

# 多核CC

[https://zhuanlan.zhihu.com/p/115114220](https://zhuanlan.zhihu.com/p/115114220)

**Cache不一致**

> **[图片提取文字 (image.png)]:**
> ## 问题背景
> 
> 首先我们假设2个CPU的系统,并且L1 Cache的cache line大小是64 Bytes。两个CPU都读取0x40地址数据,导致0x40开始的64 Bytes内容分别加载到CPU0和CPU1的私有的cache line。
> 
> ![](_page_0_Picture_2.jpeg)
> 
> CPU0执行写操作,写入值0x01。CPU0私有的L1 Cache更新cache line的值。然后,CPU1读取0x40数据,CPU1发现命中cache,然后返回0x00值,并不是CPU0写入的0x01。这就造成了CPU0和CPU1私有L1 Cache数据不一致现象。
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> 按照正确的处理流程,我们应该需要以下方法保证多核Cache 一致性:
> 
> - CPU0修改0x40的时候,除了更新CPU0的Cache之外,还应该通知CPU1的Cache更新0x40的数据。
> - CPU0修改0x40的时候,除了更新CPU0的Cache之外,还 可以通知CPU1的Cache将0x40地址所在cache line置成 invalid。保证CPU1读取数据时不会命中自己的Cache。不 命中自己的cache之后,我们有两种选择保证读取到最新的 数据。a) 从CPU0的私有cache中返回0x40的数据给 CPU1; b) CPU0发出invalid信号后,将写入0x40的数据写 回主存,CPU1从主存读取最新的数据。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%201.png)

**Bus Snooping**

> **[图片提取文字 (image.png)]:**
> ## **Bus Snooping Protocol**<sup>†</sup>
> 
> 继续以上面的例子说明bus snooping的工作机制。当CPU0修 改自己私有的Cache时,硬件就会广播通知到总线上其他所有 的CPU。对于每个CPU来说会有特殊的硬件监听广播事件,并 检查是否有相同的数据被缓存在自己的CPU,这里是指 CPU1。如果CPU1私有Cache已经缓存即将修改的数据,那么 CPU1的私有Cache也需要更新对应的cache line。这个过程 就称作bus snooping。如下图所示,我们只考虑L1 dCache 之间的一致性。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%202.png)

> **[图片提取文字 (image.png)]:**
> 这种bus snooping方法简单,但要需要每时每刻监听总线上 的一切活动。我们需要明白的一个问题是不管别的CPU私有 Cache是否缓存相同的数据,都需要发出一次广播事件。这在 一定程度上加重了总线负载,也增加了读写延迟。针对该问 题,提出了一种状态机机制降低带宽压力。这就是MESI protocol (协议)。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%204.png)

**MESI协议**

> **[图片提取文字 (image.png)]:**
> ## **MESI Protocol**<sup>†</sup>
> 
> MESI是现在一种使用广泛的协议,用来维护多核Cache一致性。我们可以将MESI看做是状态机。我们将每一个cache line标记状态,并且维护状态的切换。cache line的状态可以像tag, modify等类似存储。继续以上面的例子说明问题。
> 
> - 1. 当CPU0读取0x40数据,数据被缓存到CPU0私有Cache, 此时CPU1没有缓存0x40数据,所以我们标记cache line状态为Exclusive。Exclusive代表cache line对应的数据仅在数据只在一个CPU的私有Cache中缓存,并且其在缓存中的内容与主存的内容一致。
> - 2. 然后CPU1读取0x40数据,发送消息给其他CPU,发现数据被缓存到CPU0私有Cache,数据从CPU0 Cache返回给CPU1。此时CPU0和CPU1同时缓存0x40数据,此时cache line状态从Exclusive切换到Shared状态。Shared代表cache line对应的数据在"多"个CPU私有Cache中被缓存,并且其在缓存中的内容与主存的内容一致。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%205.png)

> **[图片提取文字 (image.png)]:**
> - 3. 继续CPU0修改0x40地址数据,发现0x40内容所在cache line状态是Shared。CPU0发出invalid消息传递到其他 CPU,这里是CPU1。CPU1接收到invalid消息。将0x40所在的cache line置为Invalid状态。Invalid状态表示表明当前cache line无效。然后CPU0收到CPU1已经invalid的消息,修改0x40所在的cache line中数据。并更新cache line状态为Modified。Modified表明cache line对应的数据仅在一个CPU私有Cache中被缓存,并且其在缓存中的内容与主存的内容不一致,代表数据被修改。
> - 4. 如果CPU0继续修改0x40数据,此时发现其对应的cache line的状态是Modified。因此CPU0不需要向其他CPU发送消息,直接更新数据即可。
> - 5. 如果0x40所在的cache line需要替换,发现cache line状态是Modified。所以数据应该先写回主存。
> 
> 以上是cache line状态改变的举例。我们可以知道cache line 具有4中状态,分别是Modified、Exclusive、Shared和 Invalid。取其首字母简称MESI。当cache line状态是 Modified或者Exclusive状态时,修改其数据不需要发送消息给其他CPU,这在一定程度上减轻了带宽压力。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%206.png)

**MESI消息**

read invalidate对应**自增**指令；

> **[图片提取文字 (image.png)]:**
> ## **MESI Protocol Messages**
> 
> Cache之间数据和状态同步沟通,是通过发送message同步和沟通。MESI主要涉及一下几种message。
> 
> - Read: 如果CPU需要读取某个地址的数据。
> - Read Response: 答复一个读消息,并且返回需要读取的数据。
> - Invalidate: 请求其他CPU invalid地址对应的cache line。
> - Invalidate Acknowledge: 回复invalidate消息,表明对应的cache line已经被invalidate。
> - Read Invalidate: Read + Invalidate消息的组合。
> - Writeback: 该消息包含要回写到内存的地址和数据。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> 继续以上的例子,我们有5个步骤。现在加上这些message, 看看消息是怎么传递的。
> 
> - 1. CPU0发出Read消息。主存返回Read Response消息,消息包含地址0x40的数据。
> - 2. CPU1发出Read消息,CPU0返回Read Response消息,消息包含地址0x40数据。
> - 3. CPU0发出Invalidate消息,CPU1接到消息后,返回 Invalidate Acknowledge消息。
> - 4. 不需要发送任何消息。
> - 5. 发送Writeback消息。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%208.png)

# CXL CC

[https://zhuanlan.zhihu.com/p/14268243479](https://zhuanlan.zhihu.com/p/14268243479)

[CXL分解内存、一致性协议](CXL%E5%88%86%E8%A7%A3%E5%86%85%E5%AD%98%E3%80%81%E4%B8%80%E8%87%B4%E6%80%A7%E5%8D%8F%E8%AE%AE%2026be12d10b6e80f08478c9eeb7b9d31b.md)

# host CPU和device的CC

不使用host的Cache和内存，使用PCIe连接的**自定义内存**，功能一致但**缺少Cache**的加速；

> **[图片提取文字 (image.png)]:**
> ## 主机内 CPU 和 device 之间的 CC
> 
> 我认为主机内 CPU 和 device 之间的 CC 是非常必要的。2017 年我在微软实习的时候,用 FPGA 做了一块内存挂到 PCle<sup>+</sup> 的 bar 空间上,真能在这块 bar 空间上跑起来一个 Linux 系统,但是本来只要 3 秒的启动流程花了 30 分钟,比 host memory 慢了 600 倍。这就是因为 PCle 不支持 CC,CPU 直接访问 device memory 只能是 uncacheable 的,每次访存都要通过 PCle 去 FPGA 转一圈,效率低得不行。
> 
> 因此目前 PCIe bar 空间只能用来让 CPU 给 device 下发 MMIO 命令,数据传输必须通过 device DMA 来进行。因此现在不管是 NVMe 盘还是 RDMA 网卡,都必须走 doorbell-WQE/command-DMA 这一套复杂的流程,如下图所示。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> ## CXL Load/Store: 同步远端内存访问
> 
> • RDMA是异步远端内存访问,每次访问需要多次PCle交互,时延最低也需要1.6 us
> 
> ![](_page_0_Figure_2.jpeg)
> 
> • CXL Load/Store是**同步**远端内存访问,CPU直出网络,指令直接访问远端内存,无需经过PCIe,无需WQE、CQE、doorbell开销,时延<0.5 us
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> - 如上图所示,在 RDMA 中如果想发送一个数据,那么:
> - 1. 软件首先会生成一个 WQE (work queue element) ,就是工作队列里边的一个工作任务。
> - 2. 然后这个任务再下发一个 doorbell, 就是按一个门铃到网卡告诉说我有事情要做了。
> - 3. 接着,网卡在收到这个门铃之后会从内存里面把这个工作任务取到网卡里面。
> - 4. 然后再根据工作任务当中的地址,访问内存中的数据,把它 DMA 到网卡。
> - 5. 再接下来,网卡会把这个数据封装成一个网络报文,从本地发送到远端。
> - 6. 然后,接收端的网卡在收到了这个数据之后,再把它写到远端的内存。
> - 7. 接着,接收端的网卡返回一个完成消息说我干完了。
> - 8. 发起端的网卡收到了这个完成消息之后,它就在本地内存中生成一个CQE。
> - 9. 最后,应用需要去 poll 这个CQE,也就是说它要获取这个完成队列里的完成事件才能够完成整个过程。
> 
> 我们可以看到,整个过程非常复杂。相比 RDMA 这种比较复杂的异步的远端内存访问,CXL 和 NVLink \* 这种 Load/Store 就是一种更简单的同步内存访问方式。为什么它会更简单呢?
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2011.png)

**RDMA**是异步远端内存访问，基本流程是：CPU和网卡的任务配置**doorbell**-**WQE**，网卡的任务/数据获取WQE/**DATA**-DMA的流程，网卡和远端设备的**通信**，远端设备的**执行**，网卡任务完成**CQE**；

**RDMA**中CPU和网卡间的交互通过**PCIe**，**doorbell、WQE、CQE**是使用PCIe所需的步骤，产生多次内存和网卡间的**DMA**，是完成任务的**PCIe开销**；

**CXL**是同步远端内存访问，在CPU（**CXL**）/GPU（**NVLink**）**设置网络模块**，支持**CXL指令直接通过网络收发**来访问远端内存，**不需要通过PCIe到达网卡**来发送和接收；

CXL CC仅要求网络节点内的CPU core和device之间的缓存一致性，**不要求跨节点的CC**；

> **[图片提取文字 (image.png)]:**
> 因为它的 Load/Store 是一个同步的内存访问指令,也就是说 CPU (对 CXL 而言) 或者 GPU (对 NVLink 而言) 有一个硬件模块能够直接访问网络单元。那么这个指令就可以直接去访问远程的内存,而不需要经过 PCIe, 这样就不需要 WQE、CQE 还有 doorbell 的这些开销,整个的时延可以降低到 0.5 us 以下。整个过程实际上只需要 4 步:
> 
> - 1. 应用发一个 Load/Store 指令;
> - 2. CPU 中的网络模块发起一个 Load 或 Store 网络报文, 在网络上面获取或者传送数据;
> - 3. 对方的网络模块会做一个 DMA, 把对应的数据从内存里面拿出来;
> - 4. 通过网络回馈给发起的网络模块,然后CPU的这条指令就宣告完成,可以继续进行后续的指令了。
> 
> **注意这里的 Load/Store 并不一定需要跨主机的 CC**,仅仅是要求主机内部的 CPU core 和 device 之间做到 CC。就像 RDMA Read/Write 一样,**RDMA Read/Write 是远端内存和本地内存之间 的数据搬移,Load/Store 是远端内存和寄存器之间的数据搬移**。远端内存或者本地寄存器的数据 修改了,并不需要同步到对端。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/79b4295b-d4ef-44a8-a7eb-1ba0b8694ccf.png)

> **[图片提取文字 (image.png)]:**
> ## 同步和异步远端内存访问的比较
> 
> |               | 同步远端内存访问                                          | 异步远端内存访问                                   |
> |---------------|---------------------------------------------------|--------------------------------------------|
> | 粒度            | Cache line                                        | 用户指定的消息大小                                  |
> | 时延            | 低                                                 | 高                                          |
> | 大块数据的访问<br>效率 | 低                                                 | 高                                          |
> | 应用透明性         | 应用无感,可用于扩展本地内存,<br>实现应用无感的内存池化                    | 应用需要显式访问远端内存,如果用于<br>内存扩展,应用需要修改           |
> | 对硬件的要求        | 高,需要网卡与CPU紧密配合                                    | 低,网卡可以是分离式的形态                              |
> | 可靠性           | 爆炸半径大,一个节点故障会影响<br>使用了该节点远端内存的所有节点,<br>异步指令错误难以捕获 | 容易通过应用捕获异步远端访问异常,将爆炸半径缩小到受影响的应用            |
> | 缓存一致性         | 取决于硬件是否支持,但在大规模下硬件支持缓存一致性的开销高                     | 不支持,软件显式在远端和本地内存之间拷贝,在有共享的情况下需要配合分布式锁保证一致性 |
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2012.png)

同步远端内存访问的优劣

> **[图片提取文字 (image.png)]:**
> 总的来说,同步和异步远程内存访问各有优缺点。同步远程内存访问,如 CXL 和 NVLink,使用简单的 Load 和 Store 操作,可以在不需太多复杂步骤的情况下实现远程内存访问。这种方式相较于异步远程内存访问,如 RDMA,更加简单,但也有一定的局限性。
> 
> 同步远程内存访问的优势在于:
> 
> - 2. 对应用程序来说是透明的,可以用来扩展本地内存,而不需要修改应用程序。
> - 2. 对应用性净未说定透明的,可以用未扩展中地内针,则个需要修以应用性净。
> - 3. 在访问较小数据量时,效率可能更高。
> - 4. 在硬件支持的情况下,可能支持缓存一致性。
> 
> 1.过程简单,交互流程简洁,使得访问延迟较低。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2013.png)

> **[图片提取文字 (image.png)]:**
> 同步远程内存访问的劣势包括:
> 
> - 1. 对硬件要求较高,需要网卡与 CPU 紧密配合。
> - 2. 每次访问的数据量相对较小(通常是一个缓存行,如 64 字节),因此在访问大数据量时,效率可能不如异步远程内存访问。
> - 3. 同步远程内存访问的可靠性可能较差,因为一个节点故障可能会影响到使用了该节点所贡献的远端内存的所有节点。有一个所谓"爆炸半径"的概念,远端内存如果发生故障了,影响的不只是自己这个节点,这就会导致爆炸半径增大。
> - 4. 大规模下的缓存一致性开销很高。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2014.png)

异步远端内存访问的优劣

> **[图片提取文字 (image.png)]:**
> 异步远程内存访问的优势在干: 1. 用户可以指定访问的数据量大小,从而在访问大数据量时,效率可能更高。 2. 对硬件要求相对较低,网卡可以采用分离式形态,如 PCIe 接口的网卡。
> 
> 3. 可以通过应用捕获异常,从而将影响范围缩小到受影响的应用。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2015.png)

> **[图片提取文字 (image.png)]:**
> 根据实际应用场景和需求,开发者可以选择适合的内存访问方式。对于需要访问较小数据量且对延 :迟要求较高的场景,同步远程内存访问可能更合适: 而对于需要访问大数据量且对延迟要求不高的: 场景,异步远程内存访问可能更高效。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2016.png)

> **[图片提取文字 (image.png)]:**
> 异步远程内存访问的劣势包括:
> 
> - 1. 过程较复杂,涉及到与网卡的复杂交互,导致访问延迟相对较高。
> - 2. 对应用程序来说不是透明的,需要显式访问远程内存,因此如果用于扩展内存,需要修改应用程序。
> - 3. 不支持缓存一致性,需要靠软件在远端和本地内存之间进行拷贝,并在共享内存情况下配合分布式锁来保证一致性。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2017.png)

**NVLink**

> **[图片提取文字 (image.png)]:**
> NVLink 比较有意思的是,大量数据传输也敢走 Load/Store,因为 GPU 的核多,而且 NVLink 的 时延低。最近也有一些研究指出 GPU Load/Store 的效率比较低,占用了大量的 GPU 核,而且造 成了 GPU 缓存的污染。例如 MSRA 在 NSDI '23 上的研究: ARK: GPU-driven Code Execution for Distributed Deep Learning
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: Comparison between CPU-controlled and GPU-controlled communication – the latter has two different approaches, which leverage (b) MMIO (like NCCL) or (c) directly initiated DMA (this work). DEV refers to any kind of devices in a can implement our DMA engine.
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2019.png)

Apple的cpu和gpu**统一内存**

> **[图片提取文字 (image.png)]:**
> 苹果的 Unified Memory 也是在单机内实现支持 CC 的共享内存的一个不错的设计,CPU 和 GPU 共享内存后,一方面解决了显存不够的问题,另一方面使得 CPU 和 GPU 的协同非常高效。
> 
> 比如我前几天在 MacBook Pro 上跑了 LLaMA 2<sup>+</sup> 的 4-bit 量化版本,笔记本就能跑 70B 的模型,还是非常 exciting 的。视频在此: Meta 发布开源可商用模型 Llama 2,实际体验效果如何?这得多亏有 96 GB 的 Unified Memory(其实跑起来只占了 50 GB),如果是 CPU memory 和 GPU memory 隔离开来分别做 96 GB,这成本就高了。
> 
> 有趣的是如果使用 llama.cpp,在有缓存的情况下,MacBook Pro 上加载 70B 的模型都用不了 1秒 (非首次加载),而 NVIDIA A100<sup>↑</sup> 服务器都需要 10 秒左右的时间通过 PCIe 加载模型。这就是因为 CPU 和 GPU 共享内存,减少了数据搬移。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2020.png)

# 跨主机CC

内存池化

> **[图片提取文字 (image.png)]:**
> 跨主机的 CC 是争议比较大的。一个原因是大规模分布式一致性难以实现,是学术界几十年的open problem。另一个原因是应用场景很多人没想清楚。
> 
> 比如内存池化,很多人在讲的故事是借用其他机器的空闲内存,提高集群内存使用率,这就不需要跨主机的 CC,只需要 Load/Store 和主机内的 CC。因为借来的内存也只有一台机器用,出借方不需要访问,其他机器也不需要访问。
> 
> 终极版的内存池化就是很多台机器共享内存,支持跨主机的 CC。优点很多,简化编程,减少拷贝,提高内存利用率。但是理想很美好,**现实中怎么存储数量巨大的 sharer list (共享者列表)?** 
> 
> Cache invalidation 的开销太高怎么办? 学术界提出了很多 mitigations,包括:
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2021.png)

> **[图片提取文字 (image.png)]:**
> - 1. 把**缓存粒度(granularity)**从 cache line 扩大到 block, page 甚至 object, 以降低存储共享者列表的开销,但是也增加了 false sharing 带来的 invalidation 开销;
> - 2. 把 **sharer list 的数据结构**从 bitmap 改成链表,或者采用分布式存储,把共享同一个 cache line 的机器组成一个 hierarchy;
> - 3. 控制共享者数量,比如 NVIDIA 之前就是用 page fault 来搞 CC,只允许单个共享者 exclusive access (但是 NVIDIA 毕竟还是对 CPU 和 OS 缺少控制,要是我做 page-fault-based CC,肯定把解决 page fault 的流程搞成全硬化的,通常情况下不让 CPU 和 OS 参与);学术界也有控制最多 3 个(for example)共享者的,多了就把原来的共享者踢出去;
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2022.png)

**lease**

> **[图片提取文字 (image.png)]:**
> 另一条路是**使用 lease 的概念取代共享者列表**,到了过期时间缓存自动失效,写操作的同步延迟最 大可能跟租期一样长。Lease 的方法有个 trade-off,如果租期太短,那么读操作需要反复获取更新, 的值,读效率变低;如果租期太长,写操作的同步延迟又太高。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2023.png)

面向应用的CC

> **[图片提取文字 (image.png)]:**
> 其实我感觉更靠谱的一种方法是**把 CC 跟业务相结合**,因为业务最清楚什么时候数据该同步,业务 一般也比较清楚数据有哪些共享者。比如在分布式系统里面,一般都是需要先获取对象的读写锁, 然后访问数据,最后再释放锁。访问数据的过程中可能有读有写,但这个中间过程可能不一定需要 实时同步到其他节点,事实上有很多场景是根本不想让中间结果为别人所知(分布式事务的原子) 性)。获取锁的时候把数据从源端同步到本地,释放锁的时候再把修改过的数据从本地同步到源
> 
> 端。既然业务里面都实现读写锁了,那肯定得在内存里面存储共享者列表吧?共享者列表没地方放的问题自然也解决了。这样,如果我做一个**硬件加速的读写锁 + 对象同步语义,作为 RDMA 语义** 
> 
> **的扩展**,是不是更实用?一个 RTT 不仅搞定了读写锁,又搞定了对象数据的同步,还不浪费数据所
> 
> 在主机的 CPU, 岂不美哉?
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2024.png)

> **[图片提取文字 (image.png)]:**
> 如果感觉大多数业务里没有用锁,上面的读写锁不够实用,那也可以做一个乞丐版的跨主机 CC,只支持用户按需同步,不支持实时同步,这样在很多场景下也够用了,只是需要软件上搞定触发同步的时机,也就是靠软件解决 cache invalidation 问题,编程比较麻烦。
> 
> GPU 和以 TPU 为代表的一大堆 DSA 就是典型的例子,这些 DSA 在 ResNet 的时代一个比一个牛,同等工艺算力提升几倍,但是遇到 Transformer 有效算力基本上都不行了。更不用说 DSA 的生态问题了,算子开发是要成本和时间的,DSA 算子的开发成本一般比 CUDA 更高。如果 DSA 在 Transformer 时代靠谱的话,A100/H100 就不会像现在这样卖断货了。
> 
> 简化编程有时候比追求那点性能更重要。编程简单了,很可能意味着架构更通用。比如 NVIDIA 的
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2025.png)

应用前景

> **[图片提取文字 (image.png)]:**
> 跨主机的 CC 我觉得主要还是用在 Web service、大数据、存储之类的场景。目前我还没想到在 Al 和 HPC 领域能有什么应用,AI 和 HPC 一般都是 collective operations(集合通信), embedding 也是有逻辑上中心化的 parameter server 来存储,对多机共享内存数据的需求似乎不 大。如果我说错了,欢迎指正。
![image.png](Cache%20Coherency%EF%BC%88%E5%9D%91%EF%BC%89/image%2026.png)

# 历史和演进

[https://zhuanlan.zhihu.com/p/162099300](https://zhuanlan.zhihu.com/p/162099300)