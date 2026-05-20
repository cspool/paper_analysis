## V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

- 属于硬件架构的实现是什么？实验比较什么？
  提出V-Rex，首个面向streaming video LLM的软件-硬件协同加速器，核心硬件为DRE（Dynamic KV Cache Retrieval Engine）。V-Rex accelerator由LXE（LLM Execution Engine）和DRE组成。LXE基于LPU架构[23]，含DPE（MAC trees, NDPE_h×NDPE_w=64×64）和VPE（vector units, NVPE_h×NVPE_w=1×64），BF16精度。DRE含KVPU和KVMU：(1) HCU（Hash-bit Cluster Unit）：含current hash-bit memory、key cache hash-bit memory、NHCU_h=1个parallel XOR accumulator（NHCU_w=16 inputs），做bit-level Hamming distance clustering；(2) WTU（WiCSum Threshold Unit）：NWTU_h=1个WTU core（NWTU_w=16），含score memory/token count memory/upper-lower bucket sorters/multipliers/adder tree/bucket range updater，做early-exit sorting+thresholding；(3) KVMU（KV Cache Management Unit）：管理hierarchical memory system（V-Rex memory→CPU memory→storage三级）+ cluster-wise memory mapping（同cluster token连续存储最大化PCIe BW）。V-Rex单核RTL实现，Synopsys Design Compiler 14nm工艺综合，0.8V 800MHz。单核面积1.89mm²，功耗2.61W，其中DRE仅占2.0%面积（0.04mm²）和2.4%功耗（57mW）。V-Rex8面积15.12mm²（vs AGX Orin 200mm²），V-Rex48面积90.57mm²（vs A100 826mm²）。片上memory：LXE 384KB + DRE 20.125KB（HC table等metadata）。实验比较：(1) V-Rex8/V-Rex48的latency/FPS/energy efficiency vs AGX Orin/A100上FlexGen/InfiniGen/InfiniGenP/ReKV；(2) 端到端延迟breakdown (AGX Orin vs V-Rex8)；(3) 消融实验（AGX+ReSV→+KVPU→+KVMU）；(4) FPS对比vs SOTA accelerator Oaken；(5) Bandwidth overlap分析；(6) Roofline model throughput analysis；(7) Power/area breakdown。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  Custom cycle-level simulator（自研，未开源）+ DRAMSim3（开源 https://github.com/umd-memsys/DRAMSim3）+ MQSim（开源 https://github.com/CMU-SAFARI/MQSim）。硬件RTL实现：Synopsys Design Compiler (14nm) + PrimeTime PX power analysis。DRAM behavior：HBM2e/DDR4建模用DRAMSim3，LPDDR5能耗参考vendor reports [11][15]。PCIe power 3W/lane，SSD power基于Kioxia BG6 [1]。GPU power通过NVIDIA-SMI [25]和tegrastats [26]实测。LPU baseline architecture参考[23]。

- 模拟器模拟什么的性能，修改了什么。
  Custom simulator模拟streaming video LLM end-to-end推理：(1) LXE计算延迟和能耗（DPE矩阵乘法+VPE向量操作，BF16）；(2) DRE的HCU bit-level clustering和WTU early-exit thresholding延迟/能耗；(3) KVMU的分层memory管理（recent/offloaded/retrieved KV movement）和cluster-wise memory mapping下的PCIe bandwidth utilization；(4) DRAM/SSD access通过DRAMSim3和MQSim建模；(5) 系统级power包括V-Rex cores+DRAM+PCIe+SSD。论文修改：在custom simulator上集成DRAMSim3和MQSim，增加V-Rex-specific模块（HCU XOR accumulator/ WTU bucket sorter+early-exit pipeline/ KVMU memory hierarchy+cluster mapping），建模与GPU baseline的公平对比。

- 开源情况。模拟器如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：V-Rex custom simulator和RTL未开源（HPCA 2026）。DRAMSim3开源，MQSim开源。硬件架构使用流程：
  1. 系统配置：edge配置V-Rex8（8 cores, DPE 64×64, VPE 1×64, HCU 1×16, WTU 1×16, LPDDR5 204.8 GB/s, PCIe 3.0 x4 4 GB/s, M.2 NVMe SSD, 32 GB capacity）；server配置V-Rex48（48 cores, HBM2e 1935 GB/s, PCIe 4.0 x16 32 GB/s, DDR4 CPU memory 80 GB）
  2. 编译/RTL综合：V-Rex单核Verilog RTL→Synopsys Design Compiler 14nm→0.8V 800MHz→面积1.89mm²/功耗2.61W→多核scale-up到8/48 cores
  3. 模拟执行：custom simulator加载Llama-3 8B model spec + COIN benchmark workload（26 frames, 25 question tokens, 39 answer tokens average case）→遍历KV cache sizes (1K-40K) + batch sizes (1/4/8)→输出per-frame latency/FPS/TPOT/energy/bandwidth utilization/roofline throughput
  4. 面积功耗：Synopsys综合报告→Table III breakdown (DPE 72.79% area/88.58% power, DRE total 2.0% area/2.4% power)
  5. 系统级功耗：V-Rex cores + DRAM (DRAMSim3/LPDDR5 vendor) + PCIe (3W/lane) + SSD (Kioxia BG6 spec)→V-Rex8总功耗35W (vs AGX Orin 40W)，V-Rex48总功耗203.68W (vs A100 300W)
