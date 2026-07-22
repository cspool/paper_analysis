# UMA、NUMA、MPP、CMP；AMP、SMP、BMP辨析

[https://www.cnblogs.com/yubo/archive/2010/04/23/1718810.html](https://www.cnblogs.com/yubo/archive/2010/04/23/1718810.html)

# 多核硬件系统

## SMP/UMA：多核平等共享内存

> **[图片提取文字 (image.png)]:**
> SMP (Symmetric Multi Processing),对称多处理系统内有许多紧耦合多处理器,在这样的系统中,所有的CPU共享全部资源,如总线,内存和I/O系统等,操作系统或管理数据库的复本只有一个,这种系 统有一个最大的特点就是共享所有资源。多个CPU之间没有区别,平等地访问内存、外设、一个操作系统。操作系统管理着一个队列,每个处理器依次处理队列中的进程。如果两个处理器同时请求访问一 个资源(例如同一段内存地址),由硬件、软件的锁机制去解决资源争用问题。Access to RAM is serialized; this and <u>cache coherency</u> issues causes performance to lag slightly behind the number of additional processors in the system.
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image.png)

> **[图片提取文字 (image.png)]:**
> 所谓对称多处理器结构,是指服务器中多个 CPU 对称工作,无主次或从属关系。各 CPU 共享相同的物理内存,每个 CPU 访问内存中的任何地址所需时间是相同的,因此 SMP 也被称为一致存储器访问 结构 (UMA: Uniform Memory Access) 。对 SMP 服务器进行扩展的方式包括增加内存、使用更快的 CPU 、增加 CPU 、扩充 I/O( 槽口数与总线数 ) 以及添加更多的外部设备 ( 通常是磁盘存储 ) 。 SMP 服务器的主要特征是共享,系统中所有资源 (CPU 、内存、 I/O 等 ) 都是共享的。也正是由于这种特征,导致了 SMP 服务器的主要问题,那就是它的扩展能力非常有限。对于 SMP 服务器而言,每 一个共享的环节都可能造成 SMP 服务器扩展时的瓶颈,而最受限制的则是内存。由于每个 CPU 必须通过相同的内存总线访问相同的内存资源,因此随着 CPU 数量的增加,内存访问冲突将迅速增加,最 终会造成 CPU 资源的浪费,使 CPU 性能的有效性大大降低。实验证明, SMP 服务器 CPU 利用率最好的情况是 2 至 4 个 CPU。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%202.png)

> **[图片提取文字 (image.png)]:**
> ## CPU利用率最好的情况是2至4个CPU
> 
> ![](_page_0_Figure_1.jpeg)
> 
> CPU个数
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%203.png)

## NUMA：多核不平等共享内存

> **[图片提取文字 (image.png)]:**
> 连接和信息交互,因此每个 CPU 可以访问整个系统的内存 (这是 NUMA 系统与 MPP 系统的重要差别 )。显然,访问本地内存的速度将远远高于访问远地内存 (系统内其它节点的内存 )的速度,这也是 非一致存储访问 NUMA 的由来。由于这个特点,为了更好地发挥系统性能,开发应用程序时需要尽量减少不同 CPU 模块之间的信息交互。 利用 NUMA 技术,可以较好地解决原来 SMP 系统的扩展问题,在一个物理服务器内可以支持上百个 CPU 。比较典型的 NUMA 服务器的例子包括 HP 的 Superdome 、 SUN15K 、 IBMp690 等。 但 NUMA 技术同样有一定缺陷,由于访问远地内存的延时远远超过本地内存,因此当 CPU 数量增加时,系统性能无法线性增加。如 HP 公司发布 Superdome 服务器时,曾公布了它与 HP 其它
> 
> UNIX 服务器的相对性能值,结果发现, 64 路 CPU 的 Superdome (NUMA 结构 ) 的相对性能值是 20 ,而 8 路 N4000( 共享的 SMP 结构 ) 的相对性能值是 6.3 。从这个结果可以看到, 8 倍数量的
> 
> NUMA 服务器的基本特征是具有多个 CPU 模块,每个 CPU 模块由多个 CPU( 如 4 个 ) 组成,并且具有独立的本地内存、 I/O 槽口等。由于其节点之间可以通过互联模块 ( 如称为 Crossbar Switch) 进行
> 
> CPU 换来的只是 3 倍性能的提升。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%205.png)

> **[图片提取文字 (image.png)]:**
> ## NUMA与MPP之比较 NUMA服务器 1/0 1/0 内存控制器 内存控制器 本地成功程 CPU n CPU 内存 NUMA内部 互联模块 1/0 内存控制器 内存控制器 本地成塔程 水油灰放射 CPU n MPP服务器 内存拉斯器 内存控制器 MPP节点 本地成战程 CPU 互联网络 CPU n 内存 ----1/0 内存控制器 水地成岩板 CPU n
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%206.png)

## MPP：SMP组网

> **[图片提取文字 (image.png)]:**
> 的信息交互是通过节点互联网络实现的,这个过程一般称为数据重分配 (Data Redistribution)。
> 但是 MPP 服务器需要一种复杂的机制来调度和平衡各个节点的负载和并行处理过程。目前一些基于 MPP 技术的服务器往往通过系统级软件 (如数据库)来屏蔽这种复杂性。举例来说, NCR 的
> Teradata 就是基于 MPP 技术的一个关系数据库软件,基于此数据库来开发应用时,不管后台服务器由多少个节点组成,开发人员所面对的都是同一个数据库系统,而不需要考虑如何调度其中某几个节点的负载。
> 
> MPP (Massively Parallel Processing),大规模并行处理系统,这样的系统是由许多松耦合的处理单元组成的,要注意的是这里指的是处理单元而不是处理器。每个单元内的CPU都有自己私有的资源,如
> 
> 在 MPP 系统中,每个 SMP 节点也可以运行自己的操作系统、数据库等。但和 NUMA 不同的是,它不存在异地内存访问的问题。换言之,每个节点内的 CPU 不能访问另一个节点的内存。节点之间,
> 
> 和 NUMA 不同, MPP 提供了另外一种进行系统扩展的方式,它由多个 SMP 服务器通过一定的节点互联网络进行连接,协同工作,完成相同的任务,从用户的角度来看是一个服务器系统。其基本特
> 
> 征是由多个 SMP 服务器 ( 每个 SMP 服务器称节点 ) 通过节点互联网络连接而成,每个节点只访问自己的本地资源 ( 内存、存储等 ) ,是一种完全无共享 (Share Nothing) 结构,因而扩展能力最好,理论
> 
> 上其扩展无限制,目前的技术可实现 512 个节点互联,数千个 CPU 。目前业界对节点互联网络暂无标准,如 NCR 的 Bynet , IBM 的 SPSwitch ,它们都采用了不同的内部实现机制。但节点互联网仅供
> 
> MPP 服务器内部使用,对用户而言是透明的。
> 
> 总线,内存,硬盘等。在每个单元内都有操作系统和管理数据库的实例复本。这种结构最大的特点在于不共享资源。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%208.png)

![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%209.png)

## CMP：片上多核

SMP、NUMA和MPP是**多核的硬件架构**，CMP是将SMP架构实现在**单个芯片**上；

CMP的提出是因为工艺进入**180nm后，线延迟超过门延迟**，多核集成能降低线延迟，提高性能；

> **[图片提取文字 (image.png)]:**
> 单芯片多处理器(Chip multiprocessors, 简称CMP), 也指多核心。CMP是由美国斯坦福大学提出的, 其思想是将大规 模并行处理器中的SMP(对称多处理器)集成到同一芯片内,各个处理器并行执行不同的进程。与CMP比较, SMT处理 器结构的灵活性比较突出。但是,当半导体工艺进入0.18微米以后,线延时已经超过了门延迟,要求微处理器的设计。 通过划分许多规模更小、局部性更好的基本单元结构来进行。相比之下,由于CMP结构已经被划分成多个处理器核来 设计,每个核都比较简单,有利于优化设计,因此更有发展前途。目前,IBM 的Power 4芯片和Sun的 MAJC5200芯片 都采用了CMP结构。多核处理器可以在处理器内部共享缓存,提高缓存利用率,同时简化多处理器系统设计的复杂 度。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2010.png)

同构CMP和异构CMP

> **[图片提取文字 (image.png)]:**
> 最初,商业化处理器都致力于单核处理器的发展,其性能已经 发挥到极致,仅仅提高单核芯片的速度会产生过多热量且无法 带来相应性能改善,但CPU性能需求大于CPU发展速度。
> 
> 尽管通过增加流水线可以提高CPU的频率,但是由于缓存的增加与漏电流控制不力的因素,导致功率大幅增加,性能反而不如之前低频率的CPU。由于CPU的功率增加,导致CPU的散热问题也就更加严重,风冷已经不能解决问题了。
> 
> 那么,此使新的技术就出现了:**多核处理器**。早在1996年就有第一款多核CPU原型Hydra<sup>+</sup>。2001年IBM推出第一个商用多核处理器POWER4<sup>+</sup>,2005年Intal和AMD多核处理器大规模应用。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2011.png)

> **[图片提取文字 (image.png)]:**
> - 同构多核架构: 系统中的处理器在架构上是相同的
> - 异构多核架构: 系统中的处理器在架构上是不同的
> 
> 同构多核架构在硬件与软件设计上比较简单,通用性高。
> 
> 异构多核处理器有: TI的达芬奇平台 DM6000系列<sup>†</sup> (ARM9+DSP)、Xilinx的 Zynq7000系列<sup>†</sup> (双核 Cortex-A9+FPGA)、Cell处理器<sup>†</sup> (1个64位 POWERPC+8个32位 协处理器)等等。
> 
> 同构多核处理器有: Exynos4412<sup>+</sup>, freescale i.mx6 dual<sup>+</sup>和 quad系列<sup>+</sup>、TI的OMAP4460<sup>+</sup>等, Intel的Core Duo、Core2 Duo等。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2012.png)

# 多核系统

[https://zhuanlan.zhihu.com/p/455322207](https://zhuanlan.zhihu.com/p/455322207)

## AMP：非对称多处理

**核之间独立运行**不同任务，**裸机任务独占核而没有**切换开销，核间需要**通信和资源共享机制**；

> **[图片提取文字 (image.png)]:**
> AMP是指,多个核相对独立的运行不同的任务,每个核之间相互隔离,可以运行不同的操作系统或裸机程序。
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## AMP运行模式
> 
> AMP的运行模式基本不会存在开销问题,尤其是在运行裸机程序时,甚至没有开销,这种模式比较适合实时性高的应用。但是两个核心之间的通信与资源共享需要有一套优秀的处理机制。
> 
> 虽然多个核心可以运行不同的系统,但是**需要有一个主要的核心**,需要使用该核心来控制整个系统以及其他的核心。例如:一个核心运行运行实时性较高的任务,另一个核心运行UI界面。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2014.png)

## SMP：对称多处理

多个核作为**1个OS**管理的运行资源；

> **[图片提取文字 (image.png)]:**
> SMP是指多个核心运行一个操作系统,该操作系统同等的管 理多个内核,这种运行模式就是简单提高运行性能。目前支持 该运行模式的操作系统有: Linux, Windows, Vxworks。 目前,我们的PC机使用的就是这种运行模式,一般适用于功 能复杂,对实时性要求不高的系统。
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ## 任务
> 
> 操作系统
> 
> CPU0 共享内存 CPU1
![image.png](UMA%E3%80%81NUMA%E3%80%81MPP%E3%80%81CMP%EF%BC%9BAMP%E3%80%81SMP%E3%80%81BMP%E8%BE%A8%E6%9E%90/image%2016.png)

## BMP：Bound 多处理

1个OS管理所有核，但任务能指定运行核，不会分配到其他核；（指定的核可能多任务吗？）