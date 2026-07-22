# meeting-26-01-09（实验平台、length-adaptive TF、多任务、idea、LPU、RDU、3DIC项目）

## idea

架构优化技术很常见，重点在于用于实现不同的应用，达到加速或者降内存的效果（别人用什么平台，我就用什么）。

目前：给一个conv算子，加载-计算-存储和固定tiling全在片上，没有ld、st指令，img2col fused在片上。

场景：多任务，VAE（conv）、Diffusion（conv）、VAE在LDM（VAE+Dif）中时间不长。

修改：我实现的conv是一个编程时算子，其中细粒度指令如果可拆分，那对应了指令级抢占？和软件解耦？上下文开销如果能重新计算（不对，传输麻烦？）

希望Acc低开销的抢占、较灵活tiling（或者多输入进入？多输入对应Batch？）。灵活tiling是为了利用率，那不如改为被其他输入抢占？（带宽压力？）

tiling限定单layer，多任务侧重不同layer。

img2col利于降低抢占开销，两种tiling，抢占时，中间结果传出来，进行其他任务，之后再恢复。这其中软件如何参与？

软件发信号，硬件自动抢占，决定tile完成（边算边启动传输）或者立刻停止（直接保留中间结果、启动传输），根据抢占延迟最小自动决策。

GPU Preemption文中的钩子kernel，来加载抢占任务，对应Acc提供抢占接口？；小timeslice是GPU自动按时间分块，对应Acc自动分块？

Acc尽可能用满资源，和小timeslice不兼容，按照硬件space来查看额外输入？（动态性）？

硬件调度和软件控制硬件的区别是？？？

## 实验平台

Diffusion

每个t的动态剪枝、映射到体系架构。

VAE+BackBone * N：是否类似first token+next token？

emulater：？

CPU、GPU simulater：模拟指令集的行为，交互系统。

AI simulator：CPU模拟器发指令，AI-core接收和执行，CPU接收。

profiler：软件描述的硬件行为，得到每个event的时间。

event-level：系统模拟器。

cycle-level：张量模拟器。

模拟器**描述硬件行为**。

multi-core架构：RV、AI core、SRAM。

MP：工艺难、精度固定。

PIM：HBM的logic die作计算（3d），或者logic die隔壁作AI core（2.5d）。

### PISA建模

**事件驱动的系统模拟器：基于PISA性能、XPU性能，计算端到端延迟。**

PISA模拟器、XPU模拟器；

**Ramulator**：建模HBM3-based PIM计算单元，主要建模内存；

> **[图片提取文字 (image.png)]:**
> ## Ramulator: A DRAM Simulator
> 
> Ramulator is a fast and cycle-accurate DRAM simulator [1, 2] that supports a wide array of commercial, as well as academic, DRAM standards:
> 
> - DDR3 (2007), DDR4 (2012)
> - LPDDR3 (2012), LPDDR4 (2014)
> - 21 001(3 (2012), 21 001(4 (2014)
> - GDDR5 (2009)
> - WIO (2011), WIO2 (2014)
> - · · · · · · · · · · · · · · · · · · ·
> - HBM (2013)
> - SALP [3]
> - TL-DRAM [4]
> - IL-DRAM [
> - RowClone [5]DSARP [6]
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image.png)

> **[图片提取文字 (image.png)]:**
> ## Introduction
> 
> Ramulator 2.0 is a modern, modular, and extensible cycle-accurate DRAM simulator. It is the successor of Ramulator 1.0 [Kim+, CAL'16], achieving both fast simulation speed and ease of extension. The goal of Ramulator 2.0 is to enable rapid and agile implementation and evaluation of design changes in the memory controller and DRAM to meet the increasing research effort in improving the performance, security, and reliability of memory systems. Ramulator 2.0 abstracts and models key components in a DRAM-based memory system and their interactions into shared interfaces and independent implementations, enabling easy modification and extension of the modeled functions of the memory controller and DRAM.
> 
> This Github repository contains the public version of Ramulator 2.0. From time to time, we will synchronize improvements of the code framework, additional functionalities, bug fixes, etc. from our internal version. Ramulator 2.0 is in its early stage and welcomes your contribution as well as new ideas and implementations in the memory system!
> 
> Currently, Ramulator 2.0 provides the DRAM models for the following standards:
> 
> - DDR3, DDR4, DDR5
> - LPDDR5
> - GDDR6
> - HBM(2), HBM3
> 
> Ramulator 2.0 also provides implementations for the following RowHammer mitigation techniques:
> 
> - PARA [Kim+, ISCA'14]
> - TWiCe [Lee+, ISCA'19]
> - Graphene [Park+, MICRO'20]
> - BlockHammer [Yağlıkçı+, HPCA'21]
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%201.png)

**Chipyard提供一个SoC的模拟（包含CPU、NPU、。。。），**生成PIM Controller、RV-core和dispatcher，**Verilator**模拟时序；

[https://github.com/ucb-bar/chipyard](https://github.com/ucb-bar/chipyard)

[https://chipyard.readthedocs.io/en/latest/](https://chipyard.readthedocs.io/en/latest/)

> **[图片提取文字 (image.png)]:**
> ## Chipyard Framework Chipyard-ci-process passing
> 
> ## **Quick Links**
> 
> - Latest Documentation: https://chipyard.readthedocs.io/
> - User Question Forum: https://groups.google.com/forum/#!forum/chipyard
> - Bugs and Feature Requests: <a href="https://github.com/ucb-bar/chipyard/issues">https://github.com/ucb-bar/chipyard/issues</a>
> 
> ## **Using Chipyard**
> 
> To get started using Chipyard, see the documentation on the Chipyard documentation site: https://chipyard.readthedocs.io/
> 
> ## What is Chipyard
> 
> Chipyard is an open source framework for agile development of Chisel-based systems-on-chip. It will allow you to leverage the Chisel HDL, Rocket Chip SoC generator, and other Berkeley projects to produce a RISC-V SoC with everything from MMIO-mapped peripherals to custom accelerators. Chipyard contains processor cores (Rocket, BOOM, CVA6 (Ariane)), vector units (Saturn, Ara), accelerators (Gemmini, NVDLA), memory systems, and additional peripherals and tooling to help create a full featured SoC. Chipyard supports multiple concurrent flows of agile hardware development, including software RTL simulation, FPGA-accelerated simulation (FireSim), automated VLSI flows (Hammer), and software workload generation for bare-metal and Linux-based systems (FireMarshal). Chipyard is actively developed in the Berkeley Architecture Research Group in the Electrical Engineering and Computer Sciences Department at the University of California, Berkeley.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%202.png)

**FireSim**

[https://fires.im/](https://fires.im/)

> **[图片提取文字 (image.png)]:**
> ## What can I simulate with FireSim?
> 
> FireSim can simulate arbitrary hardware designs written in Chisel or Verilog. With FireSim, users can write their own RTL (processors, accelerators, etc.) and run it at near-FPGA-prototype speeds on cloud or on-prem FPGAs, while obtaining performance results that match an ASIC implementation of the same design. Depending on the hardware design and the simulation scale, FireSim simulations run at 10s to 100s of MHz. Users can also integrate custom software models for components that they don't need or want to write as RTL. To help construct a closed and deterministic simulated environment around a design, FireSim includes validated and high-performance HW/SW models for I/Os like DRAM, Ethernet, Disks, UART, and more. The User Publications page links to a selection of papers written by FireSim users.
> 
> FireSim was originally developed to simulate datacenters by combining open RTL for RISC-V processors with a custom cycle-accurate network simulation. By default, FireSim provides all the RTL and models necessary to cycle-exactly simulate from one to thousands of multi-core compute nodes, derived directly from silicon-proven and open target-RTL (RISC-V Rocket Chip, BOOM, and Chipyard), with an optional cycle-accurate network simulation tying them together. FireSim also provides a Linux distribution that is compatible with the RISC-V systems it simulates and automates the process of including new workloads into this Linux distribution. These simulations run fast enough to interact with Linux on the simulated system at the command line, like a real computer. Users can even SSH into simulated systems in FireSim and access the Internet from within them.
> 
> Head to the FireSim Website to learn more.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%203.png)

### XPU建模

A100：基于profile建模，[**NVIDIA Nsight Compute**](https://developer.nvidia.com/nsight-compute)实机测试LLM kernel的结果，将结果加入系统模拟器。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%204.png)

> **[图片提取文字 (image.png)]:**
> | 73    |                                                                    |        |        |             |      | 22 00000000 00000(50 | COC 923, (81+8+8) 325)    | 853 | 1,3762 |  |
> |-------|--------------------------------------------------------------------|--------|--------|-------------|------|----------------------|---------------------------|-----|--------|--|
> | 74    |                                                                    | 4033   | 450    | 21,641/200  |      | 23 80080089 8000cc68 | 14003 13, 13, 6130, 128   | 832 | 13712  |  |
> | 75    |                                                                    |        |        |             |      | 21 0000000 00000070  | RET. ASS. NODEC 120 4 128 |     | 1,3762 |  |
> | 26    | if ((sortDir 66 (data[mosdos - 1] <= wal))    ((sortDir 66 (data[r | 34(58) | 19335  | 190,007,515 | - C2 | 25 00000000 00000080 |                           | 144 |        |  |
> | 77    |                                                                    |        |        |             |      | 25 80880885 8000cc98 |                           |     |        |  |
> | 78    |                                                                    | 2,948  | 1,600  | 12,515,12   |      | 27 00000000 0000cca9 |                           |     |        |  |
> | 79    |                                                                    |        |        |             |      | 21 00000000 00000000 |                           |     |        |  |
> | 10.00 |                                                                    |        |        |             |      | 27 #00#00#b #000ccc# |                           |     |        |  |
> | 83    |                                                                    |        |        |             |      | 30 00000000 00000000 |                           |     |        |  |
> | 12    |                                                                    | 1000   | 225525 | 14417.90    |      | 31 00000000 0000cce9 |                           |     |        |  |
> 
> 1366
> 
> 500
> 
> 29,649
> 
> 12.511.99
> 
> Sampling Data
> 
> (Not issued)
> 
> W K
> 
> 12 ecception ecceptife
> 
> 1 accounces activities
> 
> 2 00000000 000v1210
> 
> 3 80880089 80841228
> 
> 5 00000000 0001249
> 
> 6 80680685 800x1258
> 
> 7 00000000 000v1240
> 
> E 00000000 000x1270
> 
> 9 80680885 80043288
> 
> 10 00000000 00041290
> 
> 11 00000000 00041249
> 
> 12 00000000 00041250
> 
> 11 00000000 00041240
> 
> 14 00000000 00041269
> 
> 15 00000000 00041249
> 
> 16 00000000 00041250
> 
> 17 00000000 00043300
> 
> 18 00000000 00041310
> 
> 19 00000000 00041320
> 
> 20 00000000 00043330
> 
> 21 00000000 00041100
> 
> 22 00000000 00043550
> 
> 21 00000000 000v1160
> 
> 24 00000000 00041170
> 
> 5 80880885 805v1238 808
> 
> 100 blearySearchExclusive
> 
> \$28 EG. SE LECKSON
> 
> INCTP. GC. UN2. AND BE
> 
> STL ESS\*8483: 837
> 
> BMOV. 32. CLEAR #22.
> 
> BROV. SELECUTAR RES.
> 
> KOV HI, HI
> 
> MOV 85, 85
> 
> KOV 87, 87
> 
> NOV RG, RG
> 
> MOV RIS, RIS
> 
> NOV RIS, RIS
> 
> MOV 861 86
> 
> BOT, TRAP 812
> 
> K5530
> 
> 65536
> 
> 65530
> 
> 45534
> 
> 6553%
> 
> 65536
> 
> 65536
> 
> 65536
> 
> 65536
> 
> 65530
> 
> 65536
> 
> 65530
> 
> 65530
> 
> 65536
> 
> 65530
> 
> 65536
> 
> 65536
> 
> 655368
> 
> 45534
> 
> 65530
> 
> 6553%
> 
> 65536
> 
> 65530
> 
> tee
> 
> Line
> 
> tee
> 
> 100
> 
> tee
> 
> tee
> 
> Line
> 
> tee
> 
> 100
> 
> **Like** 
> 
> 160
> 
> 220
> 
> 23%
> 
> 285
> 
> 351
> 
> 443
> 
> 523
> 
> 623
> 
> 612
> 
> 593
> 
> 550
> 
> 514
> 
> 150
> 
> 148
> 
> 1430
> 
> 154
> 
> 1800 14417.90 180 1 2031636 Total Sample Count \$200 TARGET 0.06% No Instructions (5)
> 
> 0.09% Mo Throttle (T)
> 
> 0.48% Muth Pipe Throttle (39)
> 
> 1.65% Short Scoreboard (135)
> 
> 15.72% Long Scowboard (1299)
> 
> 499
> 
> 1,507
> 
> Samoling
> 
> **Outs (All)** 
> 
> 0.50% Branch Servicion (SG)
> 
> 1.54% Not Selected (126)
> 
> 24.15% to Twome (7560) 1076 51.88% Wat (4254)
> 
> 3.65% Selected (299)
> 
> 423
> 
> 2,890
> 
> Registers no sample count
> 
> udet one with
> 
> for (; stride > 0; stride >>> 1)
> 
> pas = needes;
> 
> wint newlos = unin(pos + stride, t);
> 
> if ((sorthir 66 (data[noshes - 1] < val)) || ((sorthir 66 (data[noshes - 1] < val)) ||
> 
> Inline Function
> 
> A55966
> 
> 11 3
> 
> 86 6
> 
> Inline Functions
> 
> # Source
> 
> 85 templateraint sertbire static inline ...device... wint binarylearchizelasine
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%205.png)

## **Length Adaptive Transformer Acc**

**DAC22：A Length Adaptive Algorithm-Hardware Co-design of Transformer on FPGA Through Sparse Attention and Dynamic Pipelining**

attention近似加速

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
>  $\begin{array}{cccccccccccccccccccccccccccccccccccc$
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%206.png)

> **[图片提取文字 (image.png)]:**
> value of any specific attention score, that matters.
>  We propose to quantize Q and K from the full-precision representation (usually 32-bit floating-point) into a low-precision integer
> 
> value relativity of all the attention scores, as opposed to the absolute
> 
> representation. Because both quantization and exponential operations used in softmax are monotonically increasing operators, the quantized results maintain the order of attention scores. We use our fast quantized matrix multiplication to extract dominant attention
> 
> fast quantized matrix multiplication to extract dominant attention values. Afterward, we perform accurate attention computation only for dominant attention scores. The design is depicted in Fig. 3. Particularly, we first find out the suitable scaling factor M for the given tensor to quantization, then perform  $x' = round(\frac{2^3-1}{|M|}x)$ ,
> 
> which casts all the floating point values into a desired integer. For example, the scaling factor M of  $\mathbf{K}$  in Figure 3 is 0.77, so each element is be multiplied with  $\frac{2^3-1}{0.77}$  and rounded to the nearest integer. We follow a similar procedure to quantize q into q'. Subsequently, we again use a look-up table to perform the multiplication. For
> 
> needs 256 entries. We can easily estimate the multiplied value. At the end of step ②, we derive the  $Q' \cdot K'^T$ . As the examples indicate, the quantized results keep the same rank and distribution compared with their full-precision counterpart.
> 
> We conduct Top-k sort and select the Top-k ranked attention
> 
> instance, if we multiply two 4-bit integers, the look-up table only
> 
> We conduct Top-k sort and select the Top-k ranked attention scores for exact matrix multiplication, which derives more accurate softmax values. This is faster than the original design because we only need to compute Top-k attention scores. In step  $\P$  in Fig. 3,
> 
> only need to compute Top-k attention scores. In step ② in Fig. 3, we select Top-2 element  $k_1$  and  $k_3$  to perform matrix multiplication and softmax, which is used as an approximation of the result of self-attention. Subsequently, we will perform full-precision  $\mathbf{Q} \cdot \mathbf{K}^T$  for the selected attention scores at ⑤ and final softmax at step ⑥.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%207.png)

把Encoder layer作成硬件pipeline，不同长度输入填充pipeline，但只限于Transformer layer。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: (a) Sparse attention on FPGA; (b) State machine.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%208.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Encoder layer 2
> 
> Encoder layer 1
> 
> Attention
> 
> Figure 5: Length-aware coarse-grained dynamic pipeline algorithm example: (a) timing diagram; (b) hardware utilization.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%209.png)

## 多任务抢占

[https://www.usenix.org/conference/osdi25/presentation/shen-weihang](https://www.usenix.org/conference/osdi25/presentation/shen-weihang) 

XPU Preemption：XPU（统一调度框架能**避免重复开发**）、多任务、调度需求

> **[图片提取文字 (image.png)]:**
> ## XPUs are Widely Adopted
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Typical scenarios:
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Autonomous Vehicles
> 
> ![](_page_0_Picture_7.jpeg)
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
> 
> Cloud Services
> 
> Edge Devices
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## XPU Multitasking is Common
> 
> Cloud: improve utilization, save cost
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Edge: limited XPU resources
> 
> 30+ tasks are deployed on  $1\sim2$  Orin GPU
> 
> One XPU can serve tens of job instances
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Picture_7.jpeg)
> 
> Perception
> 
> Planning & Control
> 
> Fatigue Detection
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## Diverse Scheduling Requirements of Apps
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ## Diverse Scheduling Requirements of Apps
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Task preemption is a crucial mechanism to meet diverse application needs
> 
> Tail HC33 / THI Oughput
> 
> ![](_page_0_Picture_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2013.png)

### 硬件调度、host+XPU调度

NPU执行AI会议：语音转录任务低频长持续（一秒转录一句话，每次800ms）、虚拟背景任务高频短持续（一秒25帧，40ms一次，每次几ms）。**需要抢占机制**，但XPU不支持。

> **[图片提取文字 (image.png)]:**
> ## Issues of Native Hardware Scheduling
> 
> ## #1 Most XPUs lack preemption support
> 
> Non-preemptive FCFS policy is naturally embedded in the HW, FW, and drivers.
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2014.png)

> **[图片提取文字 (image.png)]:**
> ## Example: Al Conference on Intel NPU
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2015.png)

硬件实现简单的**静态调度**，没有**高级调度策略**。高级调度策略（如MLFQ）在XPU上的运行和硬实现开销高。

不能利用运行时信息（统计QoS、ddl）**灵活切换调度策略**（FCFS+优先调度）。

> **[图片提取文字 (image.png)]:**
> ## Issues of Native Hardware Scheduling
> 
> ## #2 Hardware schedulers are not flexible
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ## Issues of Native Hardware Scheduling
> 
> #2 Hardware schedulers are not flexible
> 
> ## XPU hardware scheduling mismatches application requirements
> 
> ![](_page_0_Figure_3.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2017.png)

调度方式应该是**CPU调度+XPU执行**，但是现有框架不通用：

1、**抢占式调度面向GPU设计**，其他XPU不支持。> **通用的硬件抢占设计**？支持异构硬件？

2、无法利用硬件新特性，因为**特性绑定特定软件栈（运行时）**才可使用。> ？？？

3、不同XPU的**算子抽象和约束不同**，**难以统一调度**异构算力。> ？？？

**host+XPU现有框架的实现**：

GPU可编程性实现**低开销抢占**（设置timeslice、设计**钩子kernel** ），利用嵌入式GPU的**ioctl接口**。

XPU的**软件栈复杂**，调度框架的调度策略受**特定XPU的细节限制**（TimeGraph绑定DRI驱动、REEF绑定AMD的运行时和驱动）。

> **[图片提取文字 (image.png)]:**
> ## Approach: Host-Managed Scheduling
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ## Existing Solutions are not GENERAL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> **CFS** 
> 
> SJF
> 
> STCF
> 
> **MLFQ** 
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ## #3 Not Uniform
> 
> hinder hardware-agnostic policies & cooperative scheduling
> 
> ![](_page_0_Picture_9.jpeg)
> 
> Existing Solutions, e.g., Effisha [PPoPP' 17], FLEP [ASPLOS' 17], REEF [OSD1'22], ...
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> ![](_page_0_Picture_13.jpeg)
> 
> ![](_page_0_Picture_14.jpeg)
> 
> ![](_page_0_Picture_15.jpeg)
> 
> ## #1 Not Portable
> 
> designed for certain GPU, while new XPUs emerging
> 
> ![](_page_0_Picture_18.jpeg)
> 
> ## #2 Not Evolvable
> 
> for new / deprecated HW features, while XPUs keep advancing
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2019.png)

host+XPU虚拟化的实现挑战：每个XPU有**自己的垂直软件栈**，**XPU能力不同（可编程性、timeslice、抢占）**。

> **[图片提取文字 (image.png)]:**
> ## Existing Solutions are not GENERAL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> STCF
> 
> ![](_page_0_Picture_5.jpeg)
> 
> ![](_page_0_Picture_6.jpeg)
> 
> #3 Not Uniform
> 
> hinder hardware-agnostic policies & cooperative scheduling
> 
> ## Our Goal: **GENERAL** & **PREEMPTIVE**Host-managed XPU Scheduling
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> ![](_page_0_Picture_13.jpeg)
> 
> # I Not Portable designed for certain GPU, while new XPUs emerging
> 
> ![](_page_0_Picture_15.jpeg)
> 
> ## #2 Not Evolvable
> 
> for new / deprecated HW features, while XPUs keep advancing
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ## Challenges for General XPU Scheduling
> 
> scheduling could be tightly coupled with XPU details **CUDA Libs ROCm Libs CUDA RT** HIP RT oneAPI Libs **CUDA RT** User-space Libs ZE GPU ZE NPU **CUDA** Driver HIP Driver cuDLA VPI Kernel Modules **NV KMD** AMD KMD i915 **iVPU** NV L4T **XPUs** 
> 
> #2 Diverse HW Capabilities
> 
> #1 Complex SW Stack
> 
> scheduling could rely on specific functionalities e.g., programmability, CU reset, timeslice control
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2021.png)

### XQueue模型（XPU的线程模型）

CPU（进程）提交命令（指令）到**hwQueue**，XPU基于hwQueue串行执行命令，CPU命令提交后无法干涉命令执行，CPU进程**等待**XPU完成命令而**同步**。

> **[图片提取文字 (image.png)]:**
> ## Opportunity: XPU Programming Paradigm
> 
> ```
> def infer_task():
> memcpyH2D(...) # memcpy cmd
>          # kernel cmds
> conv(...)
> relu(...)
> softmax(...)
> memcpyD2H(...) # memcpy cmd
> sync(...) # wait for completion
> ```
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2022.png)

> **[图片提取文字 (image.png)]:**
> ## Opportunity: XPU Programming Paradigm
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## An XPU task can be abstracted as a sequence of commands executed on a command queue
> 
> memcpyD2H(...) # memcpy cmd
> sync(...) # wait for completion
> 
> Tensor Ops Algorithms
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Levelzero Command Queue
> 
> OpenCL Command Queue
> 
> . . .
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2023.png)

**XQueue**≈thread是进程的多任务调度模型，便于实现不同调度策略。

XQueue是可被CPU抢占的hwQueue，CPU提交命令到XQueue A后，可以暂停XQueue A的命令执行，恢复XQueue B的命令执行，并将XPU的hwQueue指针引导到XQueue B。

> **[图片提取文字 (image.png)]:**
> ## XQueue Abstraction: XPU Command Queue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2024.png)

> **[图片提取文字 (image.png)]:**
> ## XQueue: Preemptible XPU Command Queue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2025.png)

> **[图片提取文字 (image.png)]:**
> Table 1: Interfaces of the XQueue abstraction.
> 
> XQueue Interface Description
> 
> submit (xq, cmd) Submit a command (cmd) to XQueue (xq)
> 
> wait (xq, cmd) Wait for a given *cmd* in *xq* to complete
> 
> suspend (xq) Suspend xq to pause task execution resume (xq) Resume xq to continue task execution
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2026.png)

**异构XPUs的抢占调度**

调度模型和XPU之间插入**硬件模型层**，**面向XPU模型进行调度**，**不直接面对不同XPU的个性化接口**：XPU+软件栈支持硬件模型定义的接口，调度模型利用硬件模型提供的接口进行调度。

不同硬件抢占能力不同，调度假设单一的硬件抢占能力（模型）不合适：假设统一低的抢占能力，会损失高抢占能力硬件的性能；假设统一高的抢占能力，会不适配能力不足的硬件。

多层次硬件抽象，不同抢占能力的硬件划分到不同抽象模型：GPU可编程因此调度能力最强（timeslice，priority，线程blk or instr level Preemption），NPU部分可编程，ASIC和FPGA几乎不可编程（没有合适的接口）。

> **[图片提取文字 (image.png)]:**
> ## XQueue: Preemptible XPU Command Queue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ## Implementing XQueue on Diverse XPUs
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ## Dilemma: Compatibility or Performance?
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2029.png)

> **[图片提取文字 (image.png)]:**
> ## Our Solution: Multi-level Hardware Model
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Categorize preemption-related capabilities into three incremental levels (LvI—Lv3)
> - ➤ Higher levels posses all capabilities of lower ones and introduce advanced capabilities
> - ➤ The lowest level (LvI) is supported by ALL XPUs
> - ➤ The highest level (Lv3) unlocks full potentials of advanced XPUs
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2030.png)

hwQueue（待提交pending > 已提交in-flight > 执行中running）、XPU的launch+sync接口

lv1模型：抢占pending state的命令，即进程停止提交当前**XQueue的下一条**指令，XPU都有该接口。

lv2模型：抢占in-flight state的命令，即XPU μ-controller支持停止派发指令，类似GPU的线程块level Preemption，停止SM的dispatch Unit派发指令。

lv3模型：抢占running state的命令（GPU的指令level Preemption，保存或恢复上下文）。

> **[图片提取文字 (image.png)]:**
> ## Basic Assumption: hwQueue
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2031.png)

> **[图片提取文字 (image.png)]:**
> ## From the Perspective of Command States
> 
> ![](_page_0_Picture_1.jpeg)
> 
> - Pending commandsCreated and waiting on the host side
> - ➤ In-flight commands
> 
>   Launched to a hwQueue, escapes host control
> - Running commandsBeing executed on the XPU
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ## Level-I: Preempt Pending Command
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ## Level-2: Preempt In-flight Command
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Deactivate an hwQueue so that its commands will not be executed further
> - Significantly reduced preemption latency
> - Approach I: instruct the  $\mu$ -controllers in modern XPUs, e.g., Intel NPU, NV & AMD GPU to stall command dispatching
> - Approach 2: leverage command programmability to abort in-flight commands
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2034.png)

> **[图片提取文字 (image.png)]:**
> ## Level-3: Preempt Running Command
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - **Interrupt** the running command
> - Ultra-low & stable preemption latency
> - Supported in modern GPUs
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2035.png)

> **[图片提取文字 (image.png)]:**
> ## XSched: A Preemptive XPU Scheduling Framework
> 
> based-on the XQueue Abstraction and the Multi-level Hardware Model
> 
> ![](_page_0_Figure_2.jpeg)
> 
> tasks [3]). This may lead to unfairness and priority inver-
> 
> sions [60, 61, 100, 103, 113], which manifest as service-level
> 
> objective (SLO) violations in data centers or missed deadlines
> 
> in latency-critical autonomous systems. For example, in a
> 
> video conferencing application (see §8), the tail latency of a
> 
> with flexible policies. XSched provides unified interfaces
> 
> for scheduling XPU tasks through a preemptible command
> 
> queue abstraction (XQueue). The key challenge in imple-
> 
> menting the abstraction is adapting to XPUs with diverse and
> 
> evolving hardware capabilities and software stacks. XSched
> 
> 27
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2036.png)

30, 44, 46, 55, 72] 

### XSched Lib、Xsched Deamon**实现**

**（涉及GPU、NPU、XPU提供的抢占机制）**

利用硬件MCU提供的接口实现抢占，没有抢占相关接口的XPU用软件机制Flushing-based抢占。

。。。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4: Architecture and workflow of XSched.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2037.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8: Binary instrumentation for flushing-based preemption.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2038.png)

> **[图片提取文字 (image.png)]:**
> ## Supported & Evaluated XPUs
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> NV K40m GPU
> 
> ![](_page_0_Picture_5.jpeg)
> 
> AMD MI50 GPU
> 
> ![](_page_0_Picture_7.jpeg)
> 
> Ascend 910b NPU
> 
> ![](_page_0_Picture_9.jpeg)
> 
> Xilinx VU9P FPGA \*integrated with AWS-F1
> 
> ![](_page_0_Picture_11.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
> 
> NV DLA (NPU) + OFA (ASIC) + PVA (ASIC)
> \*integrated with Jetson AGX Orin
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2039.png)

> **[图片提取文字 (image.png)]:**
> ## Level-I: General Design for Effortless Integration
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
> 
> NV GV100 GPU
> 
> AMD MI50 GPU
> 
> Ascend 910b NPU
> 
> Xilinx VU9P FPGA \*integrated with AWS-F1
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Figure_11.jpeg)
> 
> ![](_page_0_Figure_12.jpeg)
> 
> ![](_page_0_Figure_13.jpeg)
> 
> NV DLA (NPU) + OFA (ASIC) + PVA (ASIC) \*integrated with Jetson AGX Orin
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2040.png)

> **[图片提取文字 (image.png)]:**
> ## Level-2 Preemption Support
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Pure software approach by modifying GPU kernel binary
> 
> ![](_page_0_Picture_3.jpeg)
> 
> NV GVI00 GPU
> 
> AMD MI50 GPU
> 
> Ascend 910b NPU
> 
> inx VU9P **FPGA** 
> 
> INVIDIA TESLA
> 
> Hardware-assisted approach by modifying driver to leverage the  $\mu$ -controller
> 
> Intel NPU3720 + Arc iGPU
> 
> \*integrated with Core Ultra 185H
> 
> ![](_page_0_Picture_13.jpeg)
> 
> NV DLA (NPU) + OFA (ASIC) + PVA (ASIC)
> \*integrated with Jetson AGX Orin
> 
> NV K40m GPU
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2041.png)

> **[图片提取文字 (image.png)]:**
> ## Level-3 Preemption Support
> 
> ![](_page_0_Picture_1.jpeg)
> 
> Modify GPU trap handler and proactively trigger GPU interrupts
> 
> ![](_page_0_Picture_3.jpeg)
> 
> NV GV100 GPU
> 
> ![](_page_0_Picture_8.jpeg)
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> ![](_page_0_Picture_12.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2042.png)

实验，开销很低

> **[图片提取文字 (image.png)]:**
> ## Is XSched Effective?
> 
> - Policy: Fixed-priority policy
> - Result: With XSched, latency close to standalone, reduce the P99 latency by up to 2.11 x compared with native
> 
> ## Latency CDF of the high-priority task
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2043.png)

> **[图片提取文字 (image.png)]:**
> ## Is XSched Effective?
> 
> - ➤ Policy: Bandwidth partition policy, partition ratio = 75% : 25%
> - > Result: Guarantee desired throughput ratio with 1.5% overhead
> 
> ## Normalized throughput of the two process
> 
> ![](_page_0_Figure_4.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2044.png)

> **[图片提取文字 (image.png)]:**
> ## Application: Al Video Conference with XSched
> 
> FPS: 24.18
> 
> ig Systems
> 
> **Native** 
> 
> fundamental Al Video Conference on Intel NPU w/ XSched 1000 FPS: 24.04 Frame Latency (ms) 19th USENIX Symposium g Systems Design and Implementation Test preemption is a critical and fundamental
> 
> Al Video Conference on Intel NPU
> 
> 19th USENIX Symposiur
> 
> Design and Implemental
> 
> Test preemption is a critical and
> 
> ![](_page_0_Figure_3.jpeg)
> 
> Frame Latency Monitor
> 
> **XSched** Laxity-based policy
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2045.png)

> **[图片提取文字 (image.png)]:**
> ## Application: Inference Serving with XSched
> 
> - ➤ Integrated with Triton
> - > 1.41 × reduction on P99
> 
> ➤ Comparable with ad-hoc solutions, Paella [SOSP'23]
> 
> Latency CDF of the high-priority requests.
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ![](_page_0_Figure_6.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2046.png)

> **[图片提取文字 (image.png)]:**
> ## Conclusion
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ![](_page_0_Picture_2.jpeg)
> 
> ![](_page_0_Picture_3.jpeg)
> 
> XSched — A preemptive scheduling framework that supports diverse XPUs
> 
> ## Challenges
> 
> - Complex XPU software stacks
> - Diverse and evolving hardware capabilities
> 
> ## Key Idea
> 
> ![](_page_0_Picture_9.jpeg)
> 
> ![](_page_0_Picture_10.jpeg)
> 
> • Multi-level hardware model (optimal scheduling performance and compatibility)
> 
> ![](_page_0_Picture_12.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2047.png)

> **[图片提取文字 (image.png)]:**
> ## **XSched**: Preemptive Scheduling for Diverse XPUs
> 
> Weihang Shen Mingcong Han Jialong Liu Rong Chen Haibo Chen
> 
> IPADS, Shanghai Jiao Tong University
> 
> ![](_page_0_Picture_3.jpeg)
> 
> ![](_page_0_Picture_4.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2048.png)

## HW Ascend

### AI Core和多线程并行

Cube、Vector（SP）的AI Core，DSA的多层编译。

> **[图片提取文字 (image.png)]:**
> ## **DaVinci Core**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Cube: 4096(16^3) FP16 MACs + 8192 INT8 MACs
> - Vector: 2048bit INT8/FP16/FP32 vector with special functions (activation functions, NMS- Non Minimum Suppression, ROI, SORT)
> - Explicit memory hierarchy design, managed by MTE
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ## **Overview of the DSA Developer Stack**
> 
> Level 3 Library (written by novice programmer)
> 
> Level 3 Compiler (mathematical programming model)
> 
> Level 2 Library (written by skilled programmer)
> 
> Level 2 Compiler (parallel/kernel programming model)
> 
> Level 1 Library (written by expert)
> 
> Low Level 1 Compiler (Intrinsic C) (Architecture defined programming)
> 
> TBE LIB
> 
> TVM/XLA
> 
> TBE
> 
> CudaNN/ CuBLAS
> 
> TIK LIB
> 
> Cuda/OpenCL
> 
> TIK
> 
> CCE Lib
> 
> CCE C
> 
> Instruction Set Architecture
> 
> **GPU** 
> 
> NPU
> 
> Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on January 11,2026 at 08:31:15 UTC from IEEE Xplore. Restrictions apply.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2050.png)

硬件实现多线程并行，共享资源的线程之间串行（可**静态调度**），没有共享资源的线程之间同样静态调度，因此没必要增加硬件开销。

> **[图片提取文字 (image.png)]:**
> ## Challenge 1: How to Enable Parallelism with Single Thread
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - Programmer is comfortable with the sequential code
> - Davinc's C like programming interface (CCE) let programmer to control the parallelism explicitly.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2051.png)

> **[图片提取文字 (image.png)]:**
> ## Solution with Multi-thread?
> 
> How about support hardware multi-thread feature?
> 
> - The code in each thread is sequential
> - CUBE is a share resource between threads
> - It has hardware cost
> 
> ![](_page_0_Figure_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2052.png)

软件编译生成静态多线程序列

> **[图片提取文字 (image.png)]:**
> ## How does it work - TIK
> 
> - Typical sequential Davinci code is a combination of nested FOR loops
> - Software multi-thread can be added to any FOR loop body (iterator kernel).
> 
> ```
> 1 #preload
> 2 mov out to ub(deq scale)
> 3 mov out to ub(l0c offset)
>  4 duplicate(loc offset)
> 5 load2d(weight_matrix)
>  6 #burst leavel
>  7 for(burst level)
>        brc()
>        #pipe level
>        for(pipe_level)
>     mov_out_to_ll(ll_fmi, out_fmi)
>             load3d(l0a u8, l1 fmi, weight matrix)
> 12
>            mmad(l0c s32,
> 13
>        mov_l0c32_to_ub(ub_fp16, l0c_s32)
>        vconv(ub_u8, ub_fp16)
> 15
>        mov ub to out(out fmo, ub u8)
> 16
> 17
> ```
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ## Programmer view of multi-thread
> 
> Kernel2 – Two threads, original M iteration is divided by 2
> 
> ![](_page_0_Figure_2.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2054.png)

task scheduler（μ-controller）将指令序列派发给AI Core。

> **[图片提取文字 (image.png)]:**
> ## **Advanced Compiler Techniques**
> 
> - Architecture independent DSL→ C → Binary lowering process
> - Traversal order determines data reuse factor
> - Millions of legitimate mappings
> - Find optimal mapping to
>   - bridge the 2,000x memory bandwidth gap
> 
> ![](_page_0_Picture_6.jpeg)
> 
> ![](_page_0_Figure_7.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2055.png)

> **[图片提取文字 (image.png)]:**
> ## **Putting All This Together**
> 
> ![](_page_0_Figure_1.jpeg)
> 
> - User program AI model using familiar frameworks
> - Extends operator library when necessary
> - The tasks are executed in a single node, or over a network cluster
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2056.png)

### SoC设计和3D IC

> **[图片提取文字 (image.png)]:**
> ## Mobile AP SoC
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2057.png)

> **[图片提取文字 (image.png)]:**
> ## **Automotive SoC**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2058.png)

> **[图片提取文字 (image.png)]:**
> ## Al Inference SoC
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Chip
> 
> Chip
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2059.png)

> **[图片提取文字 (image.png)]:**
> ## **AI Training SoC**
> 
> ![](_page_0_Figure_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2060.png)

> **[图片提取文字 (image.png)]:**
> ## Al Training SoC: Logic + 3DSRAM + 12 HBM
> 
> ![](_page_0_Figure_1.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> - Customized HBM2E with two Stacks to increase HBM bandwidth
> - Large 3D-SRAM as AI core cache
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2061.png)

> **[图片提取文字 (image.png)]:**
> ## Mobile AP: LoL + MoL
> 
> ![](_page_0_Picture_1.jpeg)
> 
> ## Step 1
> 
> - One logic die + 3D DRAM
> - 3DM+POP LPDDR
> 
> ![](_page_0_Figure_5.jpeg)
> 
> ## Step 3:
> 
> - Multi-layer 3D DRAM (remove POP LPDDR)
> - Multi-layer Logic die
> 
> ## Step 2
> 
> - Two logic die + 3D DRAM
> - POP LPDDR
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2062.png)

> **[图片提取文字 (image.png)]:**
> ## **Ascend910 Die Shot**
> 
> - Total 8 Dies integrated
>   - Two dummy dies are added to ensure mechanical uniformity
> - Total size:
>   456+168+96x4+110x2=1228mm<sup>2</sup>
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2063.png)

> **[图片提取文字 (image.png)]:**
> ## **Ascend310 Die Shot**
> 
> ![](_page_0_Picture_1.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2064.png)

## Groq LPU

[https://medium.com/%40cognidownunder/groqs-lpu-the-ai-accelerator-that-s-leaving-gpus-in-the-dust-bb6fff67a877](https://medium.com/%40cognidownunder/groqs-lpu-the-ai-accelerator-that-s-leaving-gpus-in-the-dust-bb6fff67a877)

[https://groq.com/blog/the-groq-lpu-explained?utm_source=chatgpt.com](https://groq.com/blog/the-groq-lpu-explained?utm_source=chatgpt.com)

编程性：程序员优化性能、线性代数完成功能

DataFlow架构，将MU分散在FU附近，指令控制FU取数据和存数据，控制空间Flow，硬件没有时间上的同步开销。

> **[图片提取文字 (image.png)]:**
> ## LPU Design Principle 1
> 
> ## Software First
> 
> The Groq LPU architecture started with the principle of software-first. The objective was to make the software developer's job of maximizing hardware utilization easier and put as much control as possible in the developer's hands.
> 
> GPUs are versatile and powerful; they can handle many different compute tasks. But they are also complex, putting extra burden on the software. It must account for variability in how a workload executes, within and across multiple chips, making scheduling runtime execution and maximizing hardware utilization much more challenging. To maximize hardware utilization on GPUs, every new AI model requires coding of model-specific kernels. This is where our software-first principle is so important – with GPUs, the software is always secondary to the hardware.
> 
> The Groq LPU was designed from the outset for linear algebra calculations – the primary requirement for AI inference. By limiting the focus to linear algebra compute and simplifying the multi-chip computation paradigm, Groq took a different approach to AI inference and chip design. The LPU employs a programmable assembly line architecture, which enables the AI inference technology to use a generic, model-independent compiler and stay true to its software-first principle. The software is always primary, in complete control of every step of inference.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2065.png)

> **[图片提取文字 (image.png)]:**
> easier and put as much control as possible in the developer's hands.
> 
> The objective was to make the software developer's job of maximizing hardware utilization
> 
> Software-first isn't just a design principle though – it is actually how Groq built its first generation GrogChip™ processor. We didn't touch chip design until the compiler's architecture was designed. The compiler accepts workloads from several different frameworks, running those workloads through multiple stages. As the compiler maps and schedules a program to run across one or multiple LPUs, it optimizes performance and utilization. The result is a program encompassing all data movement information throughout execution.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2066.png)

> **[图片提取文字 (image.png)]:**
> ## LPU Design Principle 2
> 
> ## Programmable Assembly Line Architecture
> 
> The primary defining characteristic of the Groq LPU is its programmable assembly line architecture.
> 
> The LPU features data "conveyor belts" which move instructions and data between the chip's SIMD (single instruction/multiple data) function units. At each step of the assembly process, the function unit receives instructions via the conveyor belt. The instructions inform the function unit where it should go to get the input data (which conveyor belt), which function it should perform with that data, and where it should place the output data. This process is all software-controlled; no synchronization is required within the hardware.
> 
> The LPU programmable streaming architecture supports an assembly line process within a chip as well as between chips. There is ample chip-to-chip bandwidth, which enables the data conveyor belts to flow between chips as easily as within a chip. There is no need for routers or controllers for inter-chip connectivity, even at maximum capacity.
> 
> The assembly line process within and across chips eliminates bottlenecks. There is no waiting for compute or memory resources to complete a task. There is no need for additional controllers on the chip given there are no bottlenecks to manage. The assembly line moves smoothly and efficiently, perfectly in sync.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2067.png)

> **[图片提取文字 (image.png)]:**
> This is a big improvement compared to how GPUs work. GPUs operate in a multi-core "hub and spoke" model, where an inefficient data paging approach requires significant overhead to shuttle data back and forth between the compute and memory units within and across chips. GPUs also utilize multiple hierarchies of external switches and networking chips, both within and across racks, to communicate among themselves, further exacerbating the software's scheduling complexity. The result is a hard-to-program, multi-core approach.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2068.png)

LPU：Line process，req内pipeline并行

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2069.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> What's a sentence that uses all 26 letters of the alphabet?
> 
> ![](_page_0_Picture_2.jpeg)
> 
> The quick brown fox j
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
> 
> Output **Tokens**
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2070.png)

GPU：hub and spoke，集中计算，集中存储

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> What's a sentence that uses all 26 letters of the alphabet?
> 
> ![](_page_0_Picture_2.jpeg)
> 
> The qu
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2071.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Input
> 
> What's a sentence that uses all 26 letters of the alphabet?
> 
> ![](_page_0_Picture_2.jpeg)
> 
> The quic
> 
> ![](_page_0_Figure_4.jpeg)
> 
> ![](_page_0_Picture_5.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2072.png)

LPU内和LPU之间的计算、存取的延迟（CC）是确定的，可以预测的。消除对FU和MU的竞争。

计算、存储分布式设置在片上，pipeline的吞吐量更大。

> **[图片提取文字 (image.png)]:**
> ## LPU Design Principle 3
> 
> ## Deterministic Compute & Networking
> 
> For an assembly line to operate efficiently, there needs to be a high degree of certainty about exactly how long each step will take. If there is excessive variability in how long a particular task takes to execute, that variability manifests across the entire assembly line. An efficient assembly line requires highly precise determinism.
> 
> The LPU architecture is deterministic, meaning every execution step is completely predictable to the smallest execution period (also known as clock cycle). The software-controlled hardware knows with a high degree of precision exactly when and where an operation will occur and how long it will take.
> 
> The Groq LPU achieves its high degree of determinism by eliminating contention for critical resources, namely data bandwidth and compute. There is ample capacity for routing data around the chip (the conveyor belts) and plenty of compute in the chip's functional units. There is no issue with different tasks using the same resource, so there are no execution delays due to resource bottlenecks.
> 
> The same is true for routing data between chips. The LPU data conveyor belts also operate between chips, so connecting chips results in a larger programmable assembly line. Data flow is statically scheduled by the software during compilation, and executes the same way every time the program runs.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2073.png)

> **[图片提取文字 (image.png)]:**
> LPU Design Principle 4
> 
> On-chip Memory
> 
> LPUs include both memory and compute on-chip, vastly improving the speed of storing and retrieving data while eliminating timing variation. While determinism ensures the assembly line runs efficiently and eliminates the variability of each compute stage, on-chip memory enables it to run much faster.
> 
> GPUs utilize separate high-bandwidth memory chips, introducing complexity – multiple layers of memory cache, switches, and routers to move the data back and forth – while also consuming significant energy. Having the memory on the same chip improves the efficiency and speed of each I/O action and removes complexity and uncertainty.
> 
> Groq on-chip SRAM has memory bandwidth upwards of 80 terabytes/second, while GPU off-chip HBM clocks in at about eight terabytes/second. That difference alone gives LPUs up to a 10X speed advantage, on top of the boost LPUs get from not having to go back and forth to a separate memory chip to retrieve data.
> 
> The assembly line process within and across chips eliminates bottlenecks. There is no waiting for compute or memory resources to complete a task.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2074.png)

## Sambanova RDU

[https://sambanova.ai/blog/9-predictions-for-ai-in-2025?utm_source=chatgpt.com](https://sambanova.ai/blog/9-predictions-for-ai-in-2025?utm_source=chatgpt.com)

[https://sambanova.ai/blog/sn40l-chip-best-inference-solution](https://sambanova.ai/blog/sn40l-chip-best-inference-solution)

[https://sambanova.ai/blog/open-source-deep-research-agents](https://sambanova.ai/blog/open-source-deep-research-agents)

[https://sambanova.ai/blog/from-insight-to-action-with-sambanova-agents?utm_source=chatgpt.com](https://sambanova.ai/blog/from-insight-to-action-with-sambanova-agents?utm_source=chatgpt.com)

SambaNova SN40L Reconfigurable Dataflow Unit

PMU和PCU类似积木？

> **[图片提取文字 (image.png)]:**
> ## **Dataflow Architecture**
> 
> Making it ideal for inference workloads, the SN40L is built with a dataflow architecture. GPUs utilize an architecture that was not designed for Al inference and requires multiple kernel calls to run Al models. Essentially what this means is that when running Al models, GPUs have to make multiple, redundant calls back to memory. This needless overhead adds latency and slows the process down. In comparison, the RDU utilizes a dataflow architecture that combines multiple operations into a single kernel call which can handle all of the compute operations. In effect, data flows from one step of the process to the next, without calling back to memory. The result is the elimination of the additional overhead incurred from launching multiple kernels, significantly accelerating Al model processing.
> 
> Further, GPUs have low data locality, meaning that data is not always stored near where it is processed. This leads to inefficiencies as the system spends time moving data around. RDUs, with a three tier memory design, have high data locality which reduces the time spent moving data and increases efficiency.
> 
> ![](_page_0_Picture_3.jpeg)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2075.png)

DRAM > HBM > SRAM

> **[图片提取文字 (image.png)]:**
> ## **Three-Tier Memory**
> 
> The SN40L has a three-tier memory design that includes very fast memory (SRAM), high bandwidth memory (HBM), and very large memory (DRAM). The way Al models are processed is that when a user enters a prompt, the entire model is loaded onto active memory, then every possible outcome is calculated. The most likely correct response, based on its training data, is then served up to the user. The model is then removed from active memory and upon the next user prompt the process starts over again. With other systems, the model is stored in off-chip memory and then loaded into active memory, incurring needless latency. The SN40L has DRAM to hold large numbers of different models. The HBM enables the movement of a given model from DRAM to SRAM very quickly, so the time to load and run a model is drastically reduced. Since models are held in memory, switching between them can be done almost instantly.
> 
> ![](_page_0_Picture_2.jpeg)
> 
> The inherent efficiency of the SN40L architecture means that less power is required to perform inference than GPUs. The SambaRack system, which is the base system of all SambaNova solutions, consumes an average of only 10kW of power and is air cooled. This is in contrast to the latest GPU-based systems which require as much as 140kW and expensive liquid cooling infrastructure.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2076.png)

## 其他idea

transformer中的layernorm能被**tanh**替代：Stronger Normalization-Free Transformer。

也能被误差函数erf替代，但erf是个高斯函数的积分。

qwen-image-layered 速度很慢，图像分图层 

turbo-diffusion生成视频

Agent调用多个AI model？perplexity？多任务场景。

mHC: Manifold-Constrained Hyper-Connections：替代残差块，有约束随机矩阵训练。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (b) Hyper-Connections (HC)
> 
> (c) Manifold-Constrained HC (mHC)
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2077.png)

> **[图片提取文字 (image.png)]:**
> input-dependent one and the global one, referred to as dynamic mappings and static mappings, respectively. Formally, HC computes the coefficients as follows:
> 
> In the HC formulation, learnable mappings are composed of two parts of coefficients: the
> 
> $$\begin{cases} \tilde{\mathbf{x}}_{l} = \text{RMSNorm}(\mathbf{x}_{l}) \\ \mathcal{H}_{l}^{\text{pre}} = \alpha_{l}^{\text{pre}} \cdot \tanh(\theta_{l}^{\text{pre}} \tilde{\mathbf{x}}_{l}^{\top}) + \mathbf{b}_{l}^{\text{pre}} \\ \mathcal{H}_{l}^{\text{post}} = \alpha_{l}^{\text{post}} \cdot \tanh(\theta_{l}^{\text{post}} \tilde{\mathbf{x}}_{l}^{\top}) + \mathbf{b}_{l}^{\text{post}} \\ \mathcal{H}_{l}^{\text{res}} = \alpha_{l}^{\text{res}} \cdot \tanh(\theta_{l}^{\text{res}} \tilde{\mathbf{x}}_{l}^{\top}) + \mathbf{b}_{l}^{\text{res}}, \end{cases}$$
> 
> $$(5)$$
> 
>  $\left(\mathcal{H}_{l}^{\mathrm{res}} = \alpha_{l}^{\mathrm{res}} \cdot \tanh(\theta_{l}^{\mathrm{res}} \tilde{\mathbf{x}}_{l}^{\mathsf{T}}) + \mathbf{b}_{l}^{\mathrm{res}},$  where RMSNorm(·) (Zhang and Sennrich, 2019) is applied to the last dimension, and the scalars  $\alpha_{l}^{\mathrm{pre}}$ ,  $\alpha_{l}^{\mathrm{post}}$  and  $\alpha_{l}^{\mathrm{res}} \in \mathbb{R}$  are learnable gating factors initialized to small values. The dynamic
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2078.png)

> **[图片提取文字 (image.png)]:**
> In this section, we detail the calculation process of  $\mathcal{H}_l^{\text{pre}}$ ,  $\mathcal{H}_l^{\text{post}}$ , and  $\mathcal{H}_l^{\text{res}}$  in mHC. Given the input hidden matrix  $\mathbf{x}_l \in \mathbb{R}^{n \times C}$  at the l-th layer, we first flatten it into a vector  $\vec{\mathbf{x}}_l = \text{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}$  to preserve full context information. Then, we follow the original HC formulation to get the dynamic mappings and the static mappings as follows:
> 
> $$\begin{cases} \vec{\mathbf{x}}_{l}' = \text{RMSNorm}(\vec{\mathbf{x}}_{l}) \\ \tilde{\mathcal{H}}_{l}^{\text{pre}} = \alpha_{l}^{\text{pre}} \cdot (\vec{\mathbf{x}}_{l}' \varphi_{l}^{\text{pre}}) + \mathbf{b}_{l}^{\text{pre}} \\ \tilde{\mathcal{H}}_{l}^{\text{post}} = \alpha_{l}^{\text{post}} \cdot (\vec{\mathbf{x}}_{l}' \varphi_{l}^{\text{post}}) + \mathbf{b}_{l}^{\text{post}} \\ \tilde{\mathcal{H}}_{l}^{\text{res}} = \alpha_{l}^{\text{res}} \cdot \text{mat}(\vec{\mathbf{x}}_{l}' \varphi_{l}^{\text{res}}) + \mathbf{b}_{l}^{\text{res}}, \end{cases}$$
> 
> $$(7)$$
> 
> where  $\varphi_l^{\text{pre}}$ ,  $\varphi_l^{\text{post}} \in \mathbb{R}^{nC \times n}$  and  $\varphi_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$  are linear projections for dynamic mappings and mat(·) is a reshape function from  $\mathbb{R}^{1 \times n^2}$  to  $\mathbb{R}^{n \times n}$ .
> 
> Then, the final constrained mappings are obtained via:
> 
> $$\begin{cases} \mathcal{H}_{l}^{\text{pre}} = \sigma(\tilde{\mathcal{H}}_{l}^{\text{pre}}) \\ \mathcal{H}_{l}^{\text{post}} = 2\sigma(\tilde{\mathcal{H}}_{l}^{\text{post}}) \\ \mathcal{H}_{l}^{\text{res}} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_{l}^{\text{res}}), \end{cases}$$
> (8)
> 
> where  $\sigma(\cdot)$  denotes the Sigmoid function. The Sinkhorn-Knopp(·) operator firstly makes all elements to be positive via an exponent operator and then conducts iterative normalization process that alternately rescales rows and columns to sum to 1. Specifically, given a positive matrix  $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_i^{\text{res}})$  as the start point, the normalization iteration proceeds as:
> 
> $$\mathbf{M}^{(t)} = \mathcal{T}_r \left( \mathcal{T}_c(\mathbf{M}^{(t-1)}) \right), \tag{9}$$
> 
> where  $\mathcal{T}_r$  and  $\mathcal{T}_c$  denote row and column normalization, respectively. This process converges to a doubly stochastic matrix  $\mathcal{H}_l^{\text{res}} = \mathbf{M}^{(t_{\text{max}})}$  as  $t_{\text{max}} \to \infty$ . We choose  $t_{\text{max}} = 20$  as a practical value in our experiments.
![image.png](meeting-26-01-09%EF%BC%88%E5%AE%9E%E9%AA%8C%E5%B9%B3%E5%8F%B0%E3%80%81length-adaptive%20TF%E3%80%81%E5%A4%9A%E4%BB%BB%E5%8A%A1%E3%80%81idea%E3%80%81/image%2079.png)

## 项目

架构：RV-CPU、AI芯片作推理

目前：FPGA AI core实现pipeline。

需要：提供模型输入输出、架构、CFG，给AI core。

AI core输入指令集定义的指令序列。

性能、网络大小

提供：**目标检测tiny**（256MB）、语义识别、超分辨率（FSR）、

xiaomi（3层ResBlk）