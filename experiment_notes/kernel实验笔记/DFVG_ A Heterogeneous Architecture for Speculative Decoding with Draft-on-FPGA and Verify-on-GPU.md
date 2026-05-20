## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出TreeSort-Verify机制，将tree-based speculative decoding的irregular causal attention masks转换为高效block-diagonal lower triangular矩阵，实现GPU上block-parallel的tree verification。核心设计：(1) Path-Packing Reordering：定义重排序函数π将token tree节点按ancestor关系排序（parent先于child），使重排后的causal mask矩阵M_reordered[i,j] = 1 iff π(t_j) ≤ π(t_i)且t_j是t_i的ancestor，形成block-diagonal lower triangular结构；(2) Block Decomposition：将重排序列分区为K个连续block，每个block内独立使用标准lower triangular mask，tree attention分解为Att_tree = ⊕_{k=1}^{K} Att_block(Q_Bk, K_Bk, V_Bk, M_Bk)，每个block直接调用高度优化的cuBLAS GEMM kernel；(3) Memory-Friendly Verification：连续block布局improves GPU memory locality，KV-cache compact存储，block-diagonal结构支持GPU SM间的pipelined parallel执行。此外FPGA侧：(4) Multi-Branch Mapping to Block Events：利用shared prefix，draft model将多分支映射到block event，在Linear阶段增加weight reuse（多分支共享前缀权重），在Q×K^T阶段复用shared prefix并在pipeline末尾仅改变loading address以产生额外token，在S×V阶段通过最后round accumulation将额外token归并回原sequence length；(5) Ping-Pong PE调度：KERload = PE_Num × Data_width / Bandwidth，IFMload = KERload + CAS_Latency，实现computation与data loading重叠。实验比较：(1) TreeSort-Verify ablation贡献（2.46× vs HW-Branch-only 2.21×）；(2) FPGA operator execution efficiency（matrix multiplication loading和computation均达86.2%-97.5%）；(3) 不同tree verification方式的mask结构对比（sequence-based vs tree-based vs TreeSort-Verify）。

- 后端平台是什么，配置是什么。
  GPU侧：NVIDIA RTX 4090 (512 Tensor Cores, 2230MHz, 24GB DRAM, 1008 GB/s BW) 和 NVIDIA A100 (432 Tensor Cores, 1410MHz, 80GB DRAM, 1935 GB/s BW)，CUDA 12.1，cuBLAS。FPGA侧：AMD V80 (300MHz, 10848 DSPs, HBM) 和 AMD U200 (300MHz, 6480 DSPs, DDR)，Xilinx Vivado 2024.1。CPU：Intel Xeon 4310。

- 评估性能的软件/脚本是什么。修改了什么。
  TreeSort-Verify在GPU上实现，通过path-packing对token tree节点排序后划分block，每个block调用cuBLAS GEMM。FPGA kernel使用Verilog HDL定制PE微架构，含branch concatenation（多weight buffer + 选择连线）和DSP packing（单DSP双BF16×BF16乘法）。性能测量：Xilinx xbutil (FPGA power)、nvidia-smi (GPU power)。TreeSort-Verify修改了tree attention的causal mask计算方式，从irregular per-path mask转为block-diagonal形式。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/ShaoqiangLu/DFVG。TreeSort-Verify使用流程：
  1. GPU kernel编译：`cd gpu/ && make all`
  2. TreeSort-Verify在verify阶段自动触发：FPGA生成的token tree通过PCIe传输到GPU→TreeSort-Verify对树节点做path-packing重排序→按block划分→每个block独立调用cuBLAS GEMM→结果按原始index顺序recombine
  3. 例如FPGA draft model生成一棵含D1-D6的token tree（深度γ，分支数k_max），TreeSort-Verify按ancestor关系重排序后划分为若干连续block，每个block内部使用标准lower triangular mask做attention，避免了传统tree-based方法中irregular sparse mask导致的GPU memory divergence和vectorized computing underutilization。

