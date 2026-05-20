## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出两个专用硬件dataflow：(1) Unified Attention Core的Row+OP可重构attention dataflow：Big Attention使用Row dataflow（PE Cluster按attention head分配，标准Row-stationary attention dataflow），Small Attention使用OP dataflow（PE Cell/Node改为output-stationary dataflow），通过Snooper配置不同PE Cell接收的packet ID，再由Fat Tree分发到各lane实现dataflow动态切换。Row+OP MAC通过Divide-and-Conquer和Fluid Zone Detection技术减少FP累加功耗。(2) Radix Sort Core的大K TopK dataflow：TP（Parallel-to-Sequential Converter）将并行confidence vector转串行→CountBin按radix digit分桶计数→PrefixSum计算前缀和定位candidate bin→SelectBin定位含第K大元素的bin→Filter从candidate bin筛选TopK元素并输出。加入Locality-aware Scheduling：根据mask map产生history table标记高置信空间区域，PE分组并行在不同区域各自执行Radix Select，利用confidence map空间偏斜优先处理靠近已解码token的区域。Radix Sort Core将大K TopK从通用排序问题（传统Bitonic Sort+Merge Sort在大K上需反复读写重排，N=4096时K=1936时TopK仅占3.5%操作数却占20.9%延迟）变为固定4阶段pipeline消除全局排序开销。实验比较：在VAR-Turbo accelerator上评估Radix Sort Core的延迟贡献降低效果；Attain Core Row vs OP dataflow的硬件效率对比（避免为两类attention分别放独立core造成的低利用率）。

- 后端平台是什么，配置是什么。
  VAR-Turbo accelerator (TSMC 28nm+HPC, 1P8M CMOS, TT 25C, 7.09 mm², 1.98 W, 目标频率论文未明确说明)。片外DRAM：2×64bit HBM2 channel @2GHz, 32GB/s。通用对比baseline：NVIDIA V100 GPU (14 TFLOPS FP32, 16 HBM2 channels 512GB/s)。

- 评估性能的软件/脚本是什么。修改了什么。
  自研cycle-accurate simulator + RTL仿真（Synopsys VCS + Design Compiler + PrimeTime PX）。RTL仿真通过测试用例校验Attention Core Row/OP dataflow切换功能正确性和Radix Sort Core四阶段pipeline功能正确性。模拟器与RTL延迟匹配率0.90。ViTCoD/AdapTiV baseline在相同工艺和simulator框架下重新评估。

