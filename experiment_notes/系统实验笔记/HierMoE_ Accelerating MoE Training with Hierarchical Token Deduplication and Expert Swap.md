## HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

- 属于Serving调度的实现是什么？实验比较什么？
  - HierMoE 修改 Megatron-LM 训练框架的 AlltoAll 通信调度，实现拓扑感知的分层 token 去重和 expert 交换：
    1. **HierD-AlltoAll（分层去重 AlltoAll）**：替代 Megatron-LM 原生的标准 AlltoAll 和 Tutel-2DH 的二维分层 AlltoAll。根据 GPU 集群的实际分层拓扑（Node/QPI/NVLink/Intra-GPU 四层），自动选择最优层级维度 d* 执行分层 AlltoAll。在每层执行 token 去重（按 bitwise OR 合并同一 expert group 内的 token 选择），消除因多个 expert 位于同一 GPU 导致的 token 重复传输。高层（Inter-Node, 低带宽 IB）通信量大幅减少，更多通信转移到高带宽 Intra-node 链路（NVLink）。
    2. **HierD-ES（分层 Expert Swap）**：在 HierD-AlltoAll 基础上，通过交换 expert 在 GPU 间的位置来平衡各 hierarchical group 的通信负载。每 iteration 或每 N iterations 计算交换任意两 expert 后的通信时间估计矩阵，选择使总通信时间最小化的 expert pair 进行交换。
  - 实验比较：
    - Baselines：Megatron-LM（标准 AlltoAll）、Tutel-2DH（二维分层 AlltoAll）、SmartMoE（expert placement）
    - 消融对比：HD2-MoE（仅 2D 去重）、HD2-MoE-Smart（2D 去重+SmartMoE swap）、HD-MoE（HierD-AlltoAll 无 swap）、HierMoE（完整方案）
    - AlltoAll 通信时间加速比：1.55×-3.32× over baselines
    - 端到端训练加速比：1.18×-1.27× over Megatron-LM

- 硬件平台是什么，配置是什么。
  - **32-GPU 集群**：4 nodes × 8 NVIDIA RTX A6000-48G GPUs
  - 每节点：Dual Intel Xeon Platinum 8358 @ 2.60GHz，512GB DDR4
  - GPU 互联：NVLink 112.5GB/s (4× link per GPU)，PCIe 4.0 x16
  - 跨节点互联：Mellanox MT28908 InfiniBand @ 200Gb/s (ConnectX-6)
  - 四层拓扑结构：Inter-Node (IB) → Inter-QPI → Inter-NVLink → Intra-NVLink
  - 软件：Ubuntu 20.04，CUDA 12.1，PyTorch 2.1.2，NCCL 2.18.5

- 开源Serving框架是什么。修改了什么。
  - 框架：**Megatron-LM** (https://github.com/NVIDIA/Megatron-LM/)，NVIDIA 的大规模 LLM 分布式训练框架，原生支持 MoE 模型的 Expert Parallelism（EP）训练。
  - HierMoE 本身未公开独立开源仓库，在 Megatron-LM 之上以原型系统实现。
  - 核心修改：
    1. **AlltoAll 通信原语替换**：将 Megatron-LM MoE layer 中用于 token dispatch 和 combine 的标准 AlltoAll（NCCL AlltoAll）替换为分层去重 AlltoAll（HierD-AlltoAll）。新增逻辑在每层 AlltoAll 前计算去重 token 分布，根据当前 iteration 的路由结果 I_route 在各层级维度上执行 token deduplication。
    2. **性能模型参数采集**：利用 NCCL collective primitives + nccl-tests (https://github.com/NVIDIA/nccl-tests) 在集群启动时测量 7 种 AlltoAll 通信变体的 α（启动延迟）和 β（每字节传输时间）参数，通过最小二乘法拟合线性模型 t = α + n · β。r² 值均在 0.997 以上。参数采集在训练前一次性完成（<300s 测量 + <10ms 拟合）。
    3. **Expert Swap Manager**：新增 HierD-ES 模块，在每 iteration 根据 token routing 结果计算交换每对 expert 的估计通信时间矩阵 Q_d*（增量更新方式，O(D·T·K·E)），选择最优 expert pair 交换。Expert 交换执行时间仅占端到端时间的 ~1%。
    4. **最优维度选择器**：在每 MoE layer 的 AlltoAll dispatch 前，计算 d=1 到 d=D（集群拓扑层数）各维度的估计通信时间 td，选择 td 最小的维度 d* 作为当前 iteration 的 HierD-AlltoAll 维度。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - HierMoE 原型未公开独立仓库，基于 Megatron-LM 实现。
  - HierMoE MoE 训练全流程（单 MoE layer，32 GPUs EP=32）：

  ① **输入**：batch tokens → embedding → hidden states x ∈ R^{T×M}（T tokens, M embedding dim）。所有 non-expert 权重（Attention, LayerNorm）按 DP 分布在各 GPU 上。Experts 按 EP 分布：每 GPU 持有 E/32 个 expert 的完整参数。

  ② **Router/Gating**（GPU 本地计算）：对每个 token 计算 softmax gating + Top-K 选择 → routing mask I_route ∈ R^{T×E}（boolean，标记 token i 是否选择 expert j）。Gating 计算量小，在 GPU 上本地完成。

  ③ **HierD-AlltoAll 维度选择**（CPU 侧控制逻辑）：
     - 从 I_route 计算各层级的去重 token 分布：对 d=1..D，将 E 个 expert 聚合到 U[d] 个 group，去重（OR）后统计每组 token 数
     - 代入性能模型公式 (3)-(5) 计算各维度的估计通信时间 td
     - 选择 d* = argmin td
     - 复杂度 O(D·T·K)，T=1024·batch，K=8，D=4 → 微秒级

  ④ **HierD-AlltoAll Dispatch**（GPU 间 NCCL 通信）：
     - 按选定的 d* 维度执行分层 AlltoAll
     - 例 d*=3: Inter-Node AlltoAll (IB, 200Gb/s, U[1]=4 groups) → 去重后跨节点传输
       → Inter-QPI AlltoAll (QPI, U[2]=8 groups) → 去重后跨 QPI domain 传输
       → Intra-QPI (NVLink) AlltoAll (112.5GB/s, Intra-level-2, U[3]=G=32) → GPU 级分发
     - 每层传输前执行 token 去重：同一 group 内多 expert 被选中 → 只传一份 token 副本

  ⑤ **Expert FFN 计算**（GPU 本地计算）：
     - 每个 GPU 对收到的 tokens 执行本地 experts 的 FFN 前向计算
     - W_gate GEMM → activation → W_up GEMM → element-wise → W_down GEMM
     - cuBLAS GEMM kernel 在 GPU SM 上执行

  ⑥ **HierD-AlltoAll Combine**（GPU 间 NCCL 通信）：
     - 逆向分层 AlltoAll，将 expert 输出按原 token 位置合并回各 GPU
     - 同样使用 HierD-AlltoAll 选定的 d* 维度

  ⑦ **HierD-ES Expert Swap**（每 iteration 可选，~1% 时间开销）：
     - CPU 侧：从 I_route 增量更新 Z 矩阵 (E×E×G)，计算 Q_d*[r,c] 估计时间矩阵
     - 用 smooth-max (γ=10) 平滑后选择 (r*,c*) = argmin Q_d*[r,c]
     - GPU 侧：通过 NCCL P2P Send/Recv 交换 expert r* 和 c* 的参数+优化器状态

  ⑧ **Backward**：expert FFN backward → HierD-AlltoAll combine backward (等同于 dispatch) → HierD-AlltoAll dispatch backward (等同于 combine) → Attention backward

  ⑨ **输出**：完成一层 MoE layer 的前向+反向，token hidden states 传递至下一层

- 属于Serving调度的实现是什么？实验比较什么？
  - HOBBIT 在 Llama.cpp 之上构建了一个混合精度 Expert Offloading 推理系统，核心 Serving 调度实现包括三个层次：
    1. **Token-level Dynamic Expert Loading (Section 3.2)**：根据 gating output ||G(x)|| 动态评估 expert 重要性，计算 unimportance degree score s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||，用双阈值 T1=0.6/T2=0.9 将 cache-miss expert 分为三组 — 高精度 (FP16/INT8)、低精度 (INT4/INT2)、跳过。低精度 expert 加载量减少最高 4×。
    2. **Layer-level Adaptive Expert Prefetching (Section 3.3)**：利用 MoE 层间 gating input 的高余弦相似度（相邻层 top-1 expert 预测准确率平均 96%），通过 Stacking Computer 一次性批量计算所有后续层的 gating output，实现自适应预取。结合混合精度预取来降低错误预测的惩罚。
    3. **Sequence-level Multidimensional Expert Caching (Section 3.4)**：提出 LHU (Least High Precision Frequently Used) 策略，结合 LRU + LFU + LHU + FLD 四种策略的加权线性组合管理混合精度 expert cache，最小化 cache miss penalty（高精度 miss 代价为 C，低精度 miss 代价为 (B_l/B_h)·C）。
  - 实现量：在 Llama.cpp 基础上增加 8,000 行 C++/C 代码。
  - 实验比较：Transformers (TF)、DeepSpeed-Inference (DS)、Llama.cpp (LL)、MoE-Offloading (MO)、MoE-Infinity (MI)、Fiddler (FD) 共 6 个 baseline。三组硬件对比配置（见表 2）。指标：prefill latency (s)、decoding speed (tokens/s)。四种 I/O 长度组合：[16,32], [16,128], [128,32], [128,128]，batch_size=1。

- 硬件平台是什么，配置是什么。
  - **RTX 4090**（edge server）：24GB GPU memory，256GB CPU memory，64 CPU cores，PCIe 4.0 (32GB/s theoretical)，Samsung NVMe SSD 980 PRO (7,000 MB/s theoretical, ~3,000 MB/s practical)。
  - **Jetson AGX Orin**（end device）：32GB unified memory（CPU/GPU 共享），12 CPU cores。SSD 同上。
  - 三组测试配置：
    - Jetson AGX Orin + INT8 模型：HB vs LL, MI
    - RTX 4090 + FP16 模型：HB vs TF, DS, MO, MI
    - RTX 4090 + CPU + FP16 模型：HB vs LL, FD

- 开源Serving框架是什么。修改了什么。
  - 开源框架：**Llama.cpp** (https://github.com/ggerganov/llama.cpp)。
  - 核心修改：
    1. **权重分布修改**：将所有 non-expert 权重 + 部分多精度 expert 置于 GPU memory，全部 expert 权重存储于 CPU memory。这与 Llama.cpp 原生的 "足够多完整层放 GPU + 剩余层放 CPU" 模式不同，针对 MoE 稀疏激活特性优化。
    2. **两种计算模式**：
       - GPU-centric：主线程在 GPU 上计算，scheduler 线程从 CPU/SSD 加载所需精度 expert 到 GPU。
       - CPU-GPU cooperative：expert cache miss 时，主线程发送 expert 输入到 CPU helper 线程计算并返回结果。
    3. **Dynamic Expert Loader**：Expert Scorer 基于 ||G(x)|| 动态计算重要度分数，生成不同精度加载任务入 Task Queue，Expert Scheduler 通过 read() 系统调用异步加载。
    4. **Adaptive Expert Predictor**：Stacking Computer（stack + matmul + top-k）一次性计算所有后续层 gating，自适应选择需预取的 expert。
    5. **Multidimensional Cache Manager**：Policy Performer 维护 LRU/LFU/LHU 优先级记录，按加权和公式 evict 最低优先级 expert。高/低精度 cache 分离管理。新 sequence 开始时重置记录。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供 HOBBIT 完整开源仓库链接（"we implement HOBBIT on top of Llama.cpp with 8,000 additional lines of C++/C code"）。
  - HOBBIT MoE 推理全流程（GPU-centric 模式，单 token 解码为例）：

  ① **输入**：token → embedding → hidden state x，所有 non-expert 权重（Attention + LayerNorm 等）常驻 GPU VRAM。
  ② **第 i 层 MoE 计算**：
     - Router/Gating 在 GPU 上计算，得到 top-K expert IDs 和 gate weights ||G(x)||。
     - Adaptive Expert Predictor 的 Stacking Computer 利用当前 gating input 一次性计算后续层的预测 expert IDs。
  ③ **Cache 检查**：Multidimensional Cache Manager 检查 on-demand experts 和 prediction experts 是否在 expert cache (GPU VRAM) 中。
  ④ **Expert 加载（cache miss）**：
     - Expert Scorer 按公式 s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}|| 计算专家不重要度分数。
     - s_{e_i} ≤ 0.6 → 加载高精度 expert (FP16/INT8) 从 CPU memory via PCIe read()
     - 0.6 < s_{e_i} ≤ 0.9 → 加载低精度 expert (INT4/INT2)，传输量减少 4×
     - s_{e_i} > 0.9 → 跳过该 expert
     - Expert Scheduler 异步执行加载任务。
  ⑤ **Expert 替换**：Cache Manager 按加权优先级公式 p_t = w_lru·p_t^lru + w_lfu·p_t^lfu + w_lhu·p_t^lhu + w_fld·p_t^fld 选择最低优先级 expert evict，写入新 expert。高/低精度 cache 分别管理。
  ⑥ **计算**：GPU cores 对所有 on-demand experts 执行 FFN 计算：y = Σ G(x)_{e_i} · E_{e_i}(x)。
  ⑦ **推进**：hidden state 传递至第 i+1 层，prediction experts 已被预取到 cache 中。
  ⑧ **输出**：最后一层 logits → softmax → 采样 → next token。
