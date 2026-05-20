## V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出DRE（Dynamic KV Cache Retrieval Engine），一个硬件加速单元，负责streaming video LLM中KV cache retrieval的runtime计算和调度。DRE包含两大模块：(1) KVPU（KV Cache Prediction Unit）：集成HCU（Hash-bit Cluster Unit）和WTU（WiCSum Threshold Unit），加速ReSV中bit-level聚类和early-exit thresholding两类不规则计算——这些计算在GPU上因条件分支和数据依赖导致严重underutilization和延迟。(2) KVMU（KV Cache Management Unit）：管理分层KV cache memory（recent KV→V-Rex memory, old KV→CPU/storage offload, retrieved KV→prefetch回V-Rex memory），实现cluster-wise memory mapping使同cluster的token连续存储以最大化PCIe带宽利用。runtime pipeline：LXE生成hash-bit→HCU做Hamming distance clustering更新HC table→LXE计算Q×KeyCluster^T→WTU做early-exit WiCSum thresholding输出selected token indices→KVMU预取selected KV entries→attention与KV prediction/concurrent fetch重叠。实验比较：edge (V-Rex8)和server (V-Rex48)的latency/FPS/energy efficiency/bandwidth overlap across KV cache sizes (1K-40K)，对比AGX Orin和A100上的FlexGen/InfiniGen/InfiniGenP/ReKV。消融实验：AGX+ReSV (2.8× speedup但KV prediction仍占48% latency) → +KVPU (6.0× speedup, 9.2× energy reduction) → +KVMU (8.1× speedup, 10.2× energy saving)。Bandwidth分析显示KV prediction短时达600 GB/s但可hidden in attention，KV retrieval仅占~1% DRAM bandwidth可与attention/FFN并发。Roofline分析：V-Rex8达理论峰值71.5% throughput。

- 后端平台是什么，配置是什么。
  Edge: V-Rex8 (8 cores, 53.3 TFLOPS BF16, LPDDR5 204.8 GB/s 256-bit, PCIe 3.0 x4 4 GB/s, M.2 NVMe SSD)。Server: V-Rex48 (48 cores, 319.5 TFLOPS BF16, HBM2e 1935 GB/s 5120-bit, PCIe 4.0 x16 32 GB/s, DDR4 CPU memory)。Baseline GPU: NVIDIA Jetson AGX Orin (FP16 54 TFLOPS, 32 GB LPDDR5, ~40W)、NVIDIA A100 (FP16 312 TFLOPS, 80 GB HBM2e, ~300W)。V-Rex单核RTL 14nm Synopsys Design Compiler综合，0.8V 800MHz。

- 评估性能的软件/脚本是什么。修改了什么。
  自研custom cycle-level simulator + DRAMSim3 (DRAM) + MQSim (SSD)。模拟器建模GPU/CPU data movement bandwidth（A100和AGX Orin实测参数），集成DRAMSim3和MQSim进行系统级评估。V-Rex单核：Verilog RTL→Synopsys Design Compiler 14nm综合→PrimeTime PX功耗分析。GPU power通过NVIDIA-SMI和tegrastats实测。PCIe power按3W/lane，SSD power基于Kioxia BG6 spec。所有参数集成到custom simulator中。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：论文未提供V-Rex simulator/RTL开源（HPCA 2026）。DRAMSim3开源 (https://github.com/umd-memsys/DRAMSim3)，MQSim开源 (https://github.com/CMU-SAFARI/MQSim)。runtime pipeline使用流程：
  1. Prefill开始：LXE的VPE做hash-bit generation（Key×Hyperplane→binarize），输出当前frame的hash-bit vector
  2. HCU clustering：HCU接收hash-bit→从key cache hash-bit memory读取已有KeyCluster hash-bit→NHCU_h个并行XOR accumulator计算Hamming distance→与Thhd=7比较→更新HC table (cluster id/token idx/KeyCluster/token count)，HC table存于8KB on-chip memory
  3. ScoreCluster计算：LXE的DPE做Query×KeyCluster^T矩阵乘法（BF16）→得ScoreCluster→送入WTU
  4. WTU thresholding：WTU cores并行处理→preprocess step预计算weighted sum/min/max/Th_wics→token selection step从高分bucket开始bucket sort→cumulative sum→与Th_wics比较→early-exit→输出selected cluster bitmask→汇总所有row→通过HC table映射为token indices
  5. KVMU prefetch：根据selected token indices通过PCIe从storage/CPU memory预取KV entries→cluster-wise memory mapping使同cluster token连续存放→batch fetch提高PCIe有效带宽
  6. Light Attention执行：仅对selected K/V token做attention→同时KV prediction for next layer与current attention/FFN重叠（bandwidth analysis证实prediction spike 600 GB/s可hidden, retrieval仅占1% DRAM BW可concurrent execution）

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出ZipGEMM，一个fused decompression-GEMM kernel，将TCA-TBE解压与Tensor Core矩阵乘法融合为单一CUDA kernel。关键设计：(1) 基于split-K tiling，每个thread block分4阶段：Tile Loading（LDGSTS.128异步加载压缩权重+激活到shared memory）→ Warp-Level Decoding（每个warp独立解压权重到register，利用bitwise OR/POPC/shfl指令做spatial indicator判断和dynamic addressing）→ Activation Register Transfer（LDSM.M88将激活从shared memory搬入register）→ Tensor Core Computation（mma.m16n8k16执行BF16 GEMM）；(2) 两级software pipeline：coarse-level tile double buffering重叠global→shared memory传输与计算；fine-level slice-wise interleaving重叠shared→register解压与Tensor Core计算；(3) TCA-TBE的三层tiling (FragTile 8×8, TensorCoreTile 16×16, BlockTile 64×64) 直接对齐Tensor Core mma operand layout，消除runtime坐标变换。Decompressor设计三阶段：spatial bitmap indicator（bitwise OR三bitmap得64-bit indicator mask，每bit标识元素压缩/fallback状态）→ dynamic addressing（POPC并行prefix sum计算每个线程的buffer offset）→ fast exponent reassembly（base_exp + codeword算术恢复exponent，无shared memory table lookup）。实验比较kernel-level speedup：对比cuBLAS_TC、DietGPU、nvCOMP、DFloat11，在RTX4090上平均1.31× (peak 1.71×)，L40S上平均1.36× (peak 2.21×)。还对比了standalone Decompression kernel性能：ZipServ-Decomp平均2.14×/1.83×/1.10× over DietGPU/nvCOMP/DFloat11。跨代GPU forward compatibility在RTX5090上验证（1.34×–1.87× over cuBLAS）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX4090 (Ada Lovelace, 24GB, CC 8.9, SM频率2520 MHz)、NVIDIA L40S (Ada Lovelace, 48GB)、NVIDIA RTX5090 (Blackwell, 32GB, CC 12.0)。也对比了A100 (1410 MHz)和H800 (datacenter GPUs)。编译：NVCC 12.4 (RTX5090用NVCC 12.8)。

- 评估性能的软件/脚本是什么。修改了什么。
  Nsight Compute (NCU) profiler用于micro-level分析：测量DRAM reads (↓29.3%)、ALU utilization (66.0% LOP3/IADD/POPC指令)、Tensor Core utilization (保持cuBLAS的71.6%)、shared memory bank conflicts (仅~4.7K，vs DietGPU百万级)。benchmark脚本执行100 warm-up + 1000 timed iterations。ZipGEMM kernel通过PTX内联汇编实现mma.m16n8k16调用，使用LDGSTS.128 bypass L1 cache直接写入shared memory，cp.async.wait_group<0>() + __syncthreads()实现hierarchical barrier同步，__popc()和__shfl_sync()实现warp-level prefix sum。split-K配置和tile size针对不同matrix shape调优。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/HPMLL/ZipServ_ASPLOS26.git。ZipGEMM kernel编译为独立.so库（nvcc编译）。使用流程：
  1. 编译：`mkdir build && cd build && cmake .. && make` 生成libzipgemm.so
  2. kernel benchmark：通过C++ API调用，传入TCA-TBE格式的压缩权重buffer、激活tensor和matrix dimensions
  3. 性能分析：Nsight Compute `ncu --set full -o profile ./benchmark` 采集micro-architecture counters
  4. 例如LLaMA3.1-8B GateUp_proj layer (M=14336, K=4096, N=32)在RTX4090上，ZipGEMM 0.194ms vs cuBLAS_TC 0.275ms (1.42×)。DRAM读从~3.2GB降至~2.3GB (−29.3%)，ALU指令增加但被两级pipeline隐藏，Tensor Core利用率保持71.6%。shared memory bank conflict仅~4.7K（对DietGPU的百万级），因为TCA-TBE的triple bitmap layout确保coalesced access。

