# Intel和AMD的总线和Chiplet

# Intel和AMD的总线

Intel：FSB-QPI-DMI-FDI

AMD：FSB-HT-IF

[https://www.cnblogs.com/aozhejin/p/16100778.html](https://www.cnblogs.com/aozhejin/p/16100778.html)

[https://www.bilibili.com/opus/922442938068238371](https://www.bilibili.com/opus/922442938068238371)

[https://www.cnblogs.com/iancloud/p/15014662.html](https://www.cnblogs.com/iancloud/p/15014662.html)

**芯片组chipset**中，**北桥MCH**负责内存、GPU和CPU的高速通信，**南桥IOH**负责低速IO和CPU的通信；

**FSB**是CPU和北桥芯片之间的通信总线，北桥芯片包含内存控制器和各种IO控制器，**内存通道和IO通道共享FSB**，由**Intel在P4处理器**中使用；

**PCIe**是PCIe控制器和高速外设（GPU）之间的传输总线；

> **[图片提取文字 (image.png)]:**
> ## FSB总线
> 
> Front Side BUS,前端总线,是将**CPU连接到北桥芯片**的系统总线,是CPU和外界交换数据的通道。
> 
> 注:历史上前端总线会被误认为是外频的另一个名称,外频指的是CPU与主板连接的速度,是建立在数字脉冲信号震荡速度基础上。
> 
> 前端总线指的是数据传输速度,传输速度的最大带宽=总线频率\*数据位宽/8
> 
> 前端总线频率越大,代表CPU与内存间的数据传输量越大,但随着不断提高提升的内存频率、CPU性能,前端总线的瓶颈越来越明显。
> 
> 如: 64位、1333MHz的FSB所提供的内存带宽是1333MHz \* 64bit/8 = 10667MB/s = 10.67GB/s,与双通道的DDR2-667内存刚好匹配,如果使用双通道的DDR2-800、DDR2-1066的内存,这时FSB的带宽就小于内存的带宽,更不用说更高端的内存了。
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%201.png)

**AMD特有架构**

[https://zhuanlan.zhihu.com/p/658349278](https://zhuanlan.zhihu.com/p/658349278)

[https://zhuanlan.zhihu.com/p/276021843](https://zhuanlan.zhihu.com/p/276021843)

[https://www.notebookcheck.net/AMD-Zen-3-Cezanne-Ryzen-5000-Mobile-architecture-deep-dive-Improvements-across-SoC-cache-topology-and-power-efficiency-promise-a-great-mobile-experience.516890.0.html](https://www.notebookcheck.net/AMD-Zen-3-Cezanne-Ryzen-5000-Mobile-architecture-deep-dive-Improvements-across-SoC-cache-topology-and-power-efficiency-promise-a-great-mobile-experience.516890.0.html)

[https://post.smzdm.com/p/a3gv3wlr/](https://post.smzdm.com/p/a3gv3wlr/)

[https://www.zhihu.com/question/488273142](https://www.zhihu.com/question/488273142)

**HT**是AMD将内存控制器集成进CPU后，提出的多CPU和**IO控制器芯片（如北桥）间**的**点对点连接**；

技术上将QPI和HyperTransport并**不能称之为总线而只是点对点连接**。**总线**是指允许多个部件同时连接的一组导线。**点对点连接**指的是仅仅连接两个部件。

> **[图片提取文字 (image.png)]:**
> ## HT总线
> 
> Hyper-Transport,是AMD为K8平台设计的**高速串行总线**。
> 
> HT本质是是一种为主板上的集成电路互连而设计 的端到端总线技术,目的是加快芯片间的数据传输速度。
> 
> HT在AMD平台上使用,是指AMD CPU到主板芯片间的连接总线(主板芯片组是南北桥架构,则指CPU到北桥芯片)。
> 
> HT规格有HT1.0/2.0/3.0/4.0。
> 
> HT总线带宽计划公式:
> 
> HT总线带宽=处理器外频\*HT倍频 \*处理器通道位宽/8 (将Bit转换为Byte) \*2 (时钟上下沿均能传输) \*2 (上下行双向全双工)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%203.png)

**Infinite Fabric/CCX总线**是AMD基于HT提出的采用**模块化die/chiplet**设计的die间高速总线，chiplet广泛用于**Zen2**之后的处理器；

> **[图片提取文字 (image.png)]:**
> 其实第一次听说这个新总线的时候,新闻上把它叫做GMI总 线,而正式定名则是在AMD的ZEN处理器发布的PPT上,命名 为Infinity Fabric,而我们更多的时候叫它CCX总线。其实 Infinity Fabric并不是什么深奥的东西,它由HT总线衍生而 来,但是相比HT总线技术对外开放,Infinity Fabric总线则是 AMD的专利技术, 你想用, 先交授权费。Infinity Fabric可以 说是AMD这个时代的基石,它的传速速率从30GB/s到 512GB/s,并且不和HT总线兼容。Infinity Fabric分为SCF和 SDF。SDF负责数据传输,而SCF则负责控制传输命令。SDF 部分就是HT总线衍生的产物了。而Infinity Fabric和HT总线一 样,也不仅仅限制于CPU上进行使用,包括CPU,GPU,APU 这些都可以使用,只不过它们的SDF层是不一样的。不过在最 新的APU上,CPU和GPU之间仍旧使用的PCI-E总线互联,并 没有见到CCX总线,也许这一代APU仅仅只是AMD赶工的产 物,希望下一代可以看到完全体的APU。
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/eba4a080-a667-454d-88fd-98b045c74301.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_1.jpeg)
> 
> ## INFINITY FABRIC
> 
> CUSTOMIZABLE DNA CONNECTS AMD CORES TO VIRTUALLY ANY IP
> 
> ## SCALABLE CONTROL FABRIC
> 
> - Power management
> - Security and encryption
> - Test and initialization
> - Quality-of-Service
> - 3<sup>rd</sup> party IP
> 
> ![](_page_0_Figure_10.jpeg)
> 
> ## SCALABLE DATA FABRIC
> 
> - Coherent
>   HyperTransport"plus
>   enhancements
> - Low latency
> - Standardized interfaces
> 
> ![](_page_0_Picture_15.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%204.png)

> **[图片提取文字 (image.png)]:**
> ## DIEO
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ## DIE1
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%205.png)

Zen2：CCX之间、CCD之间通过IF互联

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%206.png)

三代锐龙Ryzen，Zen2，io-die

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%207.png)

Zen2：CCD和IOD之间通过IF互联

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%208.png)

二代霄龙EPYC，Zen2，io-die

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> 2X L3 Cache Directly Accessible Per Core Accelerates Core and Cache Communication for Gaming Reduction in Effective
> Memory Latency
> 知乎 ②驱动之家
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%209.png)

Zen3：取消CCX划分，合并L3 Cache

**Intel特有架构**

[https://semianalysis.com/2022/05/26/meteor-lake-die-shot-and-architecture/](https://semianalysis.com/2022/05/26/meteor-lake-die-shot-and-architecture/)

[https://zhuanlan.zhihu.com/p/564034272](https://zhuanlan.zhihu.com/p/564034272)

[https://www.cnblogs.com/iancloud/p/15014662.html](https://www.cnblogs.com/iancloud/p/15014662.html)

**QPI**是**Intel**将北桥中的内存控制器集成到CPU后提出，CPU和北桥之间的**点对点连接**，**高速IO通道独占QPI，内存通道直接连接CPU（内存控制器）**；

QPI按**数据包**传输，因为CPU数据宽度是64bit，但QPI总线位宽是16bit，因此QPI连接的CPU和北桥两侧应设置**并串转换**；

> **[图片提取文字 (image.png)]:**
> ## QPI总线
> 
> QPI又名CSI(Common System Interface),快速通道互联,是一处可以实现 芯片间直接互联的架构,矛头直指AMD的HT总线,无论是速度、带宽、每个针 脚的带宽、功耗等一切规格都 要超越HT总线。
> 
> QPI是在处理器中集成内存控制器的体系架构,主要用于处理器之间和系统组件之间的互联通信(诸如I/O)
> 
> QPI是一种基于包传输的串行式高速点对点连接协议,采用差分信号与专门的时钟进行传输。
> 
> 在延迟方面,QPI与FSB几乎相同,却可以提高访问带宽。
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/014a865c-63e8-4e12-b7bb-c60f2a99b805.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%201.png)

> **[图片提取文字 (image.png)]:**
> 一组QPI具有20条数据传输线,以及发送 (TX)和接收方 (RX)的时钟信号。
> 
> 一个QPI数据包包含80位,两个时钟周期或四次传输完成整个数据包的传送。在每次传输的20bit数据中,有16bit是真实有效的数据,其余四位用于循环冗余校验,以提高系统的可靠性。由于QPI是双向的,在发送的同时也可以接收另一个端传输来的数据。这样,每个QPI总线总带宽=每秒传输次数(即QPI频率)\*每次传输的有效数据(即16bit/8=2Byte)\*双向。
> 
> 所以频率为4.8GT/s的QPI的总带宽= 4.8GT/s \*2Byte\*2=19.2GB/s
> 
> 频率为6.4GT/s的QPI的总带宽= 6.4GT/s \*2Byte\*2=25.6GB/s
> 
> 注: bit-位, Byte-字节, 1Byte=8bit
> 
> QPI另一亮点是支持多条系统总线连接,系统总线将会被我发到你多条连接,并且频率不再是单一固定的,也无需经过FSB进行连接,根据系统各个子系统对数据吞吐量的需求,每条系统总线连接的速度也可不同。
> 
> 支持多处理器的平台。
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/be962c3c-a17f-4248-a801-33511e50c239.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2010.png)

**DMI**是**Intel**将北桥芯片（高速IO控制、内存控制）集成到CPU后，提出的CPU和南桥（低速外设）之间的**点对点连接**；

> **[图片提取文字 (image.png)]:**
> ## DMI总线
> 
> Direct Media Interface,直接媒体接口,是Intel公司开发用于连接主板南北桥的总线,取代了Hub-Link总线。
> 
> DMI采用点对点的连接方式,具有PCI-E总线的优势,DMI实现了上行与下行各1GB/s的数据传输率,总带宽达到2GB/s。
> 
> Intel处理器集成了内存控制器、PCIE控制器等,也就是将整个北桥都 集成到了CPU内部。QPI主要用于CPU内部数据传输,而在外部接口设备进行连接的时候,需要一条简洁快速的通道,就是DMI总线。这样,两个总线的传输任务就分工明确了,QPI主管内,DMI主管外。
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## Intel G45
> 
> ## CPU Central Processor Unit Video **FSB** PCIe\* (multiple possible) (optional) **MCH** Memory Controller Hub DMI Networking (optional) **ICH** VO Controller Hub PCIe\* (multiple possible)
> 
> ## Intel H55
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2012.png)

**FDI**是Intel为Metoer Lake架构（chiplet）的tile间互联设计的**点对点连接**；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **IOE Tile**
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ## Die to die的互连
> 
> 英特尔将他们的 die to die 链接称为 "Foveros Die Interconnect<sup>†</sup>" (FDI)。其更高的 IO 密度应允许在空间受 限的超极本板上使用更小的封装。此外,功耗在电池供电设计 中至关重要,毫无疑问,与 AMD 的普通封装互连相比,FDI 更适合打入移动领域。AMD 表示, Zen 1 的 die-to-die Infinity Fabric<sup>†</sup> 链接的功耗为 2 pJ/bit,他们的 Zen 2 发布 幻灯片表明 Infinity Fabric 可以以每比特低 27% 的功耗传输 数据。显然,我们不知道与其他逻辑相比,优化 cross die link有多少功耗降低,但可以合理地假设 AMD 的 die-to-die 传输贡多与 Intel Haswell 的 OPIO 互联一样多(甚至更
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> 1pJ/bit
> 
> 2
> 
> ![](_page_0_Picture_1.jpeg)
> 
> OPIO (On Package IO) Interface
> 
> FDI (Foveros Die Interconnect)
> 
> Speed 2-8GT/s
> 
> 2GT/s
> 
> 10/mm^2
> 
> 10X (36 µm)
> 
> 1X (110 um)
> 
> ![](_page_0_Picture_13.jpeg)
> 
> Latency 10-20 ns
> 
> 0.2 - 0.3 pJ/bit
> 
> Power Number of Tiles
> 
> ![](_page_0_Picture_17.jpeg)
> 
> ![](_page_0_Picture_18.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ## Meteor Lake
> 
> ## Interconnect
> 
> | Link           | Mainband width | Mainband Protocol     |
> |----------------|----------------|-----------------------|
> | CPU-SoC        | ~2K            | 2x IDI                |
> | Graphics - SoC | ~2K            | 2xiCXL                |
> | SoC - IOE      | ~1K            | IOSF, 4x Display Port |
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2016.png)

# **Intel和AMD的chiplet**

**Intel Meteor Lake**架构将SoC划分成**chiplet**，并通过base die互联和封装，每个die称为tile，**tile之间通过FDI进行连接**；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
> 
> Edge-to-edge area of the top dies: 173.37 mm² Molding layer: 135.43 mm²
> 
> D151A779 QDF4
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2018.png)

**计算tile**：2 P-core，8 E-core

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2019.png)

**GPU tile**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2020.png)

> **[图片提取文字 (image.png)]:**
> |                 | Package Physical Specifications |               |              |               |
> |-----------------|---------------------------------|---------------|--------------|---------------|
> |                 | Alder Lake M                    | Meteor Lake M | Alder Lake P | Meteor Lake P |
> |                 | BGA Type 4                      | BGA Type 4    | BGA Type 3   | BGA Type 3    |
> | Dimensions (mm) | 28.5 x 19                       | 23 x 19       | 50 x 25      | 50 x 25       |
> | Pad Count       | 1781                            | 2593          | 1744         | 1940          |
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2021.png)

**chiplet封装**工艺

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2022.png)

Alder Lake的P-core，Meteor Lake的**P-core**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2023.png)

Alder Lake的E-core，Meteor-Lake的**E-core**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> PCH: 54.05 mm<sup>2</sup>
> 
> Alder Lake P die size: 207.62 mm<sup>2</sup>
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2024.png)

> **[图片提取文字 (image.png)]:**
> We believe the SOC tile is a combination of IP that is on the existing CPU die as well as the PCH. With Meteor Lake, there is no PCH/chipset. Currently PCH's are built on a 14nm class process node as a way to reduce cost for additional IP. The PCH on Alder Lake mobile is 54mm2 and contains IP such as the IO needed for more PCIe lanes, USB ports, SATA, Intel Management Engine, and the digital logic needed for Wi-Fi. We believe all of this will also be included on the SOC tile. Furthermore, there is a variety of other logic currently on the CPU that could be moved there. The whole uncore area on the left side on Alder Lake P (TB4, Display PHYs, PCIe PHY, digital control logic, Image Processing Unit, GNA AI Accelerator, System Agent and Memory Controller) takes 55.9mm<sup>2</sup>. The majority of this IP will be moved to the SOC tile, with some IP
> 
> being moved to the 10mm<sup>2</sup> IO tile.
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2025.png)

**SoC tile和IO tile**：充当PCH、IO控制等；

AMD-Zen架构的移动端和桌面端中CCD的设计和制造一致（7nm），移动端因**面积和功耗限制**将CCD和IO组件**制造成SoC**，桌面端和服务端**追求核心数**而将多个CCD和IOD**在封装上通过IF互联**；

**AMD服务端**

> **[图片提取文字 (image.png)]:**
> Table 1: The multi-die architecture has enabled significant improvements for each processor generation since the beginning
> 
> AMD EPYC 7002
> 
> AMD EPYC 7003
> 
> 4 TB DDR4-3200
> 
> AMD EPYC 9004, 8004
> 
> 6 TB DDR5-4800
> 
> AMD EPYC 7001
> 
> 2 TB DDR3-2400/2666
> 
> Max Memory Capacity
> 
> |                                          | 'NAPLES'        | 'ROME'                  | 'MILAN'         | 'GENOA', 'SIENA'                                     |
> |------------------------------------------|-----------------|-------------------------|-----------------|------------------------------------------------------|
> |                                          |                 |                         |                 |                                                      |
> | Core Architecture                        | 'Zen'           | 'Zen 2'                 | 'Zen 3'         | 'Zen 4' and 'Zen 4c'                                 |
> | Cores                                    | 8 to 32         | 8 to 64                 | 8 to 64         | 8 to 128                                             |
> | IPC Improvement Over<br>Prior Generation | N/A             | ~24% <sup>ROM-236</sup> | ~19% MLN-003    | ~14%EPYC-038                                         |
> | Max L3 Cache                             | Up to 64 MB     | Up to 256 MB            | Up to 256 MB    | Up to 384 MB (EPYC 9004)<br>Up to 128 MB (EPYC 8004) |
> | Max L3 Cache with 3D V-Cache™ technology |                 |                         | 768 MB          | Up to 1152 MB                                        |
> | PCIe® Lanes                              | Up to 128 Gen 3 | Up to 128 Gen 3         | Up to 128 Gen 4 | Up to 128 Gen 5<br>8 bonus lanes Gen 3               |
> | CPU Process Technology                   | 14nm            | 7nm                     | 7nm             | Snm                                                  |
> | I/O Die Process Technology               | N/A             | 14nm                    | 14nm            | 6nm                                                  |
> | Power (Configurable TDP [cTDP])          | 120-200W        | 120-280W                | 155-280VIIII    | <b>美國新北加察</b>                                        |
> 
> 4 TB DDR4-3200
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2026.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2027.png)

锐龙3000 Desktop，Zen2，7nm CPU-12nm IO

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2028.png)

锐龙4000 Mobile，Zen2，**单芯片**设计，7nm

> **[图片提取文字 (image.png)]:**
> Memory PHY
> 
> Memory Controllers
> 
> Infinity Fabrie
> 
> **CCX**0
> 
> CCX1
> 
> NCCUS
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## 备注,上图不同颜色区域之说明:
> 
> - 黄色: Zen2架构的8核心处理器模块,中间的8颗横置"田字格"为8MB L3缓存;
> - 红色: Renior雷诺阿7nm Vega核显,自上而下横置8组Compute Unit,合计512SP。另外,本代Vega核显的3D运算处理模块为旧款Vega架构(同Vega64、Radeon VII),而显示特性、媒体编解码引擎部分则是最新的Navi架构;
> - 深紫色: Infinity Fabric总线模块;
> - 棕橙色: 内存控制器;
> - 绿色: Soc I/O输入输出模块;
> - 浅蓝色: 内存物理层界面
> 
> 和桌面级Zen2处理器相比,移动版的SoC一体化设计优缺点同样明显:
> 
> - 无需Chiplets的额外总线连接,带宽大增,核内延迟显著降低;
> - 较前代12nm移动版CPU相比,在同样156mm²的内核面积上,容纳了98亿只晶体管(密度翻倍);
> - 最大只能塞进8颗锐龙CPU核心,无法像桌面级一样额外多装一颗CCX芯片;
> - 一颗芯片上还需为Vega核显留出空间,导致L3缓存容量以及核显CU单元规模有所减少。
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2030.png)

**AMD Zen3 Mobile**

> **[图片提取文字 (image.png)]:**
> ## AMD RYZEN™ 5000 SERIES MOBILE
> 
> ## BRINGING "ZEN 3" INTO THE MOBILE SPACE
> 
> ![](_page_0_Picture_3.jpeg)
> 
> Tech: 7nm TSMC Transistors: 10.7B Die size: 180mm<sup>2</sup>
> 
> - Modular fabric architecture and physical design process enabled fast turn to "Zen 3"
> - Tape-out occurred just months after the introduction of "Zen 2"
> - Fast integration of "Zen 3" the product of a deliberate, multi-year roadmap
> - Common pinout with Ryzen 4000 Series allows OEM reuse of existing board designs for fast integration
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ## AMD RYZEN™ 5000 SERIES MOBILE
> 
> "CEZANNE" SOC TOPOLOGY
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2032.png)

**AMD Zen3 Desktop**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_1.jpeg)
> 
> ## AMD RYZEN" 5000 SERIES
> 
> TOPOLOGY WITH 2X CCD + clOD
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_1.jpeg)
> 
> ## THE SECRET OF CHIPLETS
> 
> ## SUSTAINING AMD SOCKET AM4
> 
> ![](_page_0_Picture_4.jpeg)
> 
> - Chiplet transition enabled scalability up to 16 cores in AM4
> - Allowed migration to "Zen 3" CCDs without disrupting the platform
> - Enables in-place upgrades to "Zen 3" for AMD Socket AM4
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> ![](_page_0_Picture_1.jpeg)
![image.png](Intel%E5%92%8CAMD%E7%9A%84%E6%80%BB%E7%BA%BF%E5%92%8CChiplet/image%2035.png)