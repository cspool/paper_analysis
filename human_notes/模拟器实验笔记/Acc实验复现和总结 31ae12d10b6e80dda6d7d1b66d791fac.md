# Acc实验复现和总结

我需要的实验环境：

       架构创新的性能仿真，RTL实现。

       架构对系统的影响仿真，系统性能仿真。

我能否利用GPU（profile？）快速验证下我的想法是否正确？类似任意精度TC的GPU。

总结

仿真的意义是确定性能提升足够，有价值流片来实现生产。

rtl实现评估的意义是部分评估达成性能的成本。

复现实验。

下载paper的概览？？？ 

## 1、PISA Acc（基于2，cursor复现）

ref：PISA: Supporting Dynamic Block Sparse Attention in Long-Context LLMs via Programmable and Decoupled PIM System

没有给出实验代码，参考其他PIM-SIm的实现。

为什么不使用其他PIM-SIM？因为其他基于Ram1.0的实验只支持HBM1，太老了，需要HBM3.

复现的意义是验证ramulator2的使用和PIM仿真的环境搭建。

难以复现和验证，建议移后，等待论文开放仓库。或等待idea和baseline进行复现。

学习下实验做法。

## 2、SpAttn Acc（调研，找好用的）

ref：SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning

结合tile Acc或Acc的sim。

这些实验可能更容易复现。

host和Acc+dram的仿真。

找一个更多引用，更接近的论文复现。

chipyard

[https://github.com/ucb-bar/chipyard](https://github.com/ucb-bar/chipyard)

2.2k*；3m；

**Ramulator2**

[https://github.com/CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2)

502*；2m；

**Ramulator**

[https://github.com/CMU-SAFARI/ramulator](https://github.com/CMU-SAFARI/ramulator)

681*；4y；

**ramulator-pim**

[https://github.com/CMU-SAFARI/ramulator-pim](https://github.com/CMU-SAFARI/ramulator-pim)

182*；6y；

**PIMSimulator**

[https://github.com/SAITPublic/PIMSimulator](https://github.com/SAITPublic/PIMSimulator)

223*；2y；

**PIMsimulation**

[https://github.com/RohSiHyun/PiMsimulation](https://github.com/RohSiHyun/PiMsimulation)

9*；3y；

**UniNDP**

[https://github.com/UniNDP-hpca25-ae/UniNDP](https://github.com/UniNDP-hpca25-ae/UniNDP)

53*；7m；

**SpAtten**

[https://github.com/mit-han-lab/spatten/tree/main](https://github.com/mit-han-lab/spatten/tree/main)

125*；3y；

**Butterfly-Acc**

[https://github.com/SamsungLabs/Butterfly_Acc/tree/main](https://github.com/SamsungLabs/Butterfly_Acc/tree/main)

15*；4y；

**vAttention**

[https://github.com/microsoft/vattention/tree/main](https://github.com/microsoft/vattention/tree/main)

465*；2y；

**PowerInfer**

[https://github.com/Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer)

8.8k*；1y；

### PytorchSim

[https://github.com/PSAL-POSTECH/PyTorchSim](https://github.com/PSAL-POSTECH/PyTorchSim)

96*；3m；

**DataFlow**

[https://github.com/OpenDCAI/DataFlow](https://github.com/OpenDCAI/DataFlow)

2.9k*；3d；

## 3、ButterFly Acc（RTL实现上板，host+Acc仿真）

ref：Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design

### 实验环境（用什么做实验？是否细化环境？）

Benchmark：**任务**负载（模型、模块）。

软件实现：任务如何**映射**到架构，Pytorch编译、手写汇编、AI编译器、加速库。

硬件实现：架构/模块的**RTL/Scala实现**后，（Vivado）编译实现。

> **[图片提取文字 (image.png)]:**
> **Benchmarks.** To evaluate the algorithmic and hardware performance of our approach on workloads with long sequences, we choose five tasks from Long-Range-Arena [39], including hierarchical data classification (ListOPs), bytelevel text classification (*Text*), byte-level document retrieval (Retrieval), image classification for sequences of pixels (*Image*), classification of long-range spatial dependency (Pathfinder). The input sequences of these datasets range from 1024 to 4096.
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image.png)

> **[图片提取文字 (image.png)]:**
> Hardware Implementation. We implement our hardware accelerators using Verilog. To evaluate performance in different scenarios, two Xilinx FPGA boards are used in our experiments: VCU128 for cloud/server scenarios and Zynq 7045 for edge/mobile settings. Xilinx Vivado 2019.1 is used for synthesis and implementation. While the maximum clock
> 
> frequencies of our designs depend on the particular FPGA
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%201.png)

> **[图片提取文字 (image.png)]:**
> **Software Implementation.** We implement the vanilla *Trans*former [3], FNet [34] and our FABNet models using PyTorch (v1.10) [40]. The pretrained models are obtained from Huggingface 4.16 [41]. The batch size is 256 for both Image and Pathfinder tasks, and 32 for the rest of datasets during training. The learning rate is set to 0.0001, except for the *Image* and *Pathfinder* tasks where we use 0.01 and 0.0005 respectively. Multiple Nvidia A100 and V100 GPUs are used for training. To use FFT cores on Nvidia GPUs, the PyTorch API "rfft2" is used to implement the FFT operation required in both *FNet* and *FABNet*. The highperformance CUDA implementation [32] of butterfly linear transformation is adopted to accelerate both GPU training
> 
> and inference. We define two models with different default settings: FABNet-Base ( $D_{\text{hid}} = 768$ ,  $R_{\text{ffn}} = 4$ ,  $N_{\text{total}} = 12$ ,  $N_{ABfly} = 0$ ) and FABNet-Large ( $D_{hid} = 1024$ ,  $R_{ffn} = 4$ ,  $N_{\text{total}} = 24, N_{\text{ABfly}} = 0$ .
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%202.png)

实验所需设计参数的搜索，细化任务、硬件参数。

**新任务（模型）模块设计**的评估（baseline是老设计/老模块）。

Co-Design：算法模型的硬件HP的协同搜索，即**特定算法模块参数和特定硬件参数的组合性能最优**。

> **[图片提取文字 (image.png)]:**
> ## B. Algorithmic Performance
> 
> The FBfly introduced in Section III-B is an efficient alternative to the vanilla attention block. To evaluate its algorithmic impact on end-to-end models, we take a six-layer Transformer as an example and compress it with different numbers of FBfly blocks, starting from the last block to the first block. Figure 16 shows the accuracy results on LRA-Text and LRA-Image. Although the accuracy fluctuates with different numbers of compressed layers, FBfly shows higher accuracy than the non-compressed Transformer with 4 and 1 compressed layers on LRA-Text and LRA-Image, respectively, demonstrating the improved algorithmic performance of our approach on end-to-end models.
> 
> To obtain the best possible algorithmic performance of each model, we use the optimized configuration specified in [42] for both vanilla *Transformer* and *FNet*. We perform a simple grid search to optimize the hyperparameters of our *FABNet*. Table [III] presents the optimized accuracy of different models. *FABNet* achieves higher accuracy than both *Transformer* and *FNet* on three out of five tasks, including *ListOPs*, *Retrieval* and *Image*. On average, *FABNet* achieves the same accuracy as *Transformer*. To investigate the efficiency of *FABNet*, Figure [17] shows the compression
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%203.png)

> **[图片提取文字 (image.png)]:**
> ## C. Effectiveness of Co-design
> 
> We evaluate the effectiveness of our co-design approach in finding the optimal algorithm and hardware designs. For demonstration, we use LRA-Text as the target dataset and VCU128 FPGA as the target device. We select  $D_{\rm hid}$ ,  $R_{\rm ffn}$ ,  $N_{\text{ABfly}}$  and  $N_{\text{total}}$  from {64, 128, 256, 512, 1024}, {1, 2,  $\{4\}, \{0, 1\}$  and  $\{1, 2\}$  respectively. Parameters for hardware parallelism ( $P_{be}$ ,  $P_{bu}$ ,  $P_{qk}$  and  $P_{sv}$ ) are chosen from  $\{0, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,$ 16, 32, 64, 128}. Figure 18 shows the points in the accuracylatency design space. The orange line represents the accuracy loss, which is constrained to be less than 1% compared with the vanilla *Transformer*. The Pareto front is indicated by the brown line and the other blue points represent designs with less optimized software-related hyperparameters (Figure 16) or hardware design parameters. Among the design points that satisfy the accuracy constraint, we choose the point with the lowest latency in the Pareto front as our point of comparison. Within our design space, the selected point is up to 10% more accurate than the points in the same latency range and up to  $130 \times$  faster than points in the same accuracy range, underlining the advantages of our co-design
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%204.png)

### 和baseline对比（baseline？比什么？怎么采集数据/measure？）

和baseline的多指标（**speedup**）对比：没有流片，所以使用**相同的实现环境和测试环境**（比如RTL实现、上板测试）。

和GPU、CPU的功耗效率（**OPS/W**）对比：实机运行 vs. 模拟器。

和SOTA的**端到端性能（latency）**、**功耗（W）**对比：**相同任务**（适当改造SOTA来支持），通过**scaling**运算单元数量，**拉平计算容量/峰值算力（freq * #ALU）后对比**。

> **[图片提取文字 (image.png)]:**
> ## D. Comparison with Baseline Design To evaluate the speedup brought by our algorithm (FAB-
> 
> line design for comparison [3]. The baseline hardware is designed with multiple multiply-accumulate (MAC) units to accelerate the linear transform and the matrix multiplications between query, key and value vectors. Each MAC is composed of a multiplier array followed by an adder tree. The fine-grained intra- and inter-layer pipeline techniques [43], 44 are used to optimize the hardware performance. We allocate the parallelism of each MAC unit according to its workload in order to achieve load-balanced execution between different pipeline stages. For a fair comparison,
> 
> *Net*) and hardware (butterfly accelerator), we use a base-
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%205.png)

> **[图片提取文字 (image.png)]:**
> we implement both baseline and butterfly accelerators on a VCU128 FPGA using 2048 multipliers. The high bandwidth memory (HBM) is used as the external memory. Both designs are clocked at 200 MHz. We evaluate both base (12 layers) and large (24 layers) versions of each model using four different input sequences (128, 256, 512 and 1024).
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%206.png)

> **[图片提取文字 (image.png)]:**
> ## E. Comparison with GPU and CPU
> 
> We compare our butterfly accelerator against GPU and CPU in both edge and server scenarios. In the edge scenario, our butterfly accelerator is implemented on a Xilinx Zynq 7045 FPGA. DDR4 is used as external memory and 512 multipliers are used for computation. Nvidia Jetson Nano GPU and Raspberry Pi4 are used as the GPU and CPU platforms, respectively. In the server scenario, the butterfly accelerator is implemented on a Xilinx VCU128 FPGA. HBM is used as external memory and the design consumes 1920 multipliers. We use Nvidia V100 and TITAN Xp GPUs for comparison, with highly-optimized CUDA implementations [32]. FPGA designs are clocked at 200 MHz.
> 
> We evaluate both *FABNet-Base* and *FABNet-Large* using 128, 256, 512 and 1024 input sequences. Figure 20 shows the results in term of speedup and energy efficiency. We represent energy efficiency using Giga operations per second
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%207.png)

> **[图片提取文字 (image.png)]:**
> ## F. Comparison with SOTA Accelerators
> 
> Table V compares our butterfly accelerator with existing state-of-the-art (SOTA) accelerators in terms of speed and power consumption. Instead of comparing the effective throughput [6], [15], we use the end-to-end latency to represent the actual execution speed of the hardware. The energy efficiency is represented by the number of predictions per Joule (Pred./J). Following the experimental setting of [18], we compare all other SOTA accelerators on LRA-Image dataset with one-layer vanilla Transformer. Among these accelerators, only SpAtten [6] and DOTA [18] report the end-to-end performance. For the rest of the accelerators that
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%208.png)

> **[图片提取文字 (image.png)]:**
> only support attention, we estimate their performance by reusing their available multipliers to accelerate FFN. Furthermore, in both [6] and [15], the authors compare different ASIC and FPGA designs based on the assumption that all the ASIC designs are clocked at 1 GHz with 128 multipliers. For a fair comparison, we follow the same assumption in our experiments. For designs with more than 128 multipliers, we follow the scaling approach of [6], [15] to linearly scale down its throughput to get their end-to-end performance. For instance, DOTA [18] achieves 11.4× speedup over Nvidia V100 using 12,000 multipliers with 12 TOPS throughput. We scale down its throughput by 12,000/128 = 93.75, which leads to  $0.123 \times$  speedup over V100. To obtain the power consumption, we use the same linear scaling approach. For instance, Sanger [15] reports the power consumption of a design with 1024 multipliers. We divide the power consumption of their systolic array (2243 mW) by 1024/128 = 8, which leads to 280.375 mW. Together with the power of other modules such as pre-processing and memory, their total power consumption is 0.801 W. To match the computational capacity of ASIC designs, we use 640 DSPs in the VCU128 FPGA. As our FPGA-based design is clocked at 200 MHz, this ensures that we have the same  $640 \times 200M = 128$  GOPS theoretical peak performance as ASIC designs  $(128 \times 1G = 128 \text{ GOPS})$ . While this is a simple approximation, it allows us to compare different hardware architectures regardless of their underlying target platforms.
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%209.png)

### 设计实现的评估

**片外带宽BW**对架构的端到端延迟的影响。

架构的**功耗**和**资源**评估：RTL实现后通过Vivado XPE评估。

> **[图片提取文字 (image.png)]:**
> different designs with 16, 32, 64 and 128 BEs executing FABNet-Large with 24 layers. To understand the bandwidth requirements under both short and long input lengths, we evaluate each design using three input sequences (128, 1024
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%2010.png)

> **[图片提取文字 (image.png)]:**
> and 4096). The results are shown in Figure 21. For a smallscale design of 16 BEs, a bandwidth of 50 GB/s is enough for the design to reach its peak performance under different input sequences. For the largest design of 128 BEs, the achieved performance saturates once the bandwidth reaches 100 GB/s.
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## H. Power and Resource Analysis
> 
> Table VI shows the power consumption breakdown based on the report generated from the Vivado XPE tool. We implement two designs with 120 BEs (BE-120) and 40 BEs (BE-40) on a VCU128 FPGA, which have been used in Section VI-E and Section VI-F, respectively. In both designs, the dynamic power accounts for more than 70% of the total power consumption. The memory resources, including both BRAM and HBM, consume more than 25% of the dynamic power. Furthermore, when the number of BEs scales from 40 to 120, the power of clocking, logic & signal and DSPs increases from 2.688 W, 2.381 W and 0.338 W to 6.882 W, 7.732 W and 1.437 W, respectively.
> 
> Table VIII presents the resource consumption of both *BE-40* and *BE-120* designs on the same VCU120 FPGA. Due to the use of FFT and butterfly matrices, our *FABNet* becomes less memory-intensive than the vanilla attention-based NNs. Since the theoretical memory bandwidth of a single HBM (450 GB/s) can already satisfy the requirement of our accelerator (Section VI-G), we use one HBM in both designs to reduce the resource and power consumption. When the number of *BE*s decreases from 120 to 40, the BRAM usage is reduced from 978 to 338. This reduction can also be observed on the LUT and register resources.
![image.png](Acc%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E5%92%8C%E6%80%BB%E7%BB%93/image%2012.png)