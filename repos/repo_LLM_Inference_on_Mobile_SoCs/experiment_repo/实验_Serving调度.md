## Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：HeteroInfer，一个面向移动端异构 SoC 的 LLM 推理引擎，能够同时利用 GPU 和 NPU 进行计算。核心实现包括：(1) layer-level GPU-NPU 执行——根据算子的计算亲和性将 Matmul 分配给 NPU，将 RMSNorm、SwiGLU 等分配给 GPU；(2) tensor-level GPU-NPU 并行——通过 weight-centric、activation-centric 和 hybrid 三种 tensor 分区策略实现 GPU 和 NPU 的并发计算；(3) fast synchronization——利用 unified memory architecture (UMA) 和可预测的 GPU kernel 等待时间实现微秒级同步；(4) offline profiler + solver——离线测量 GPU/NPU 性能矩阵，在线根据序列长度动态选择最优分区策略。
  - 实验比较：(1) 端到端延迟对比 HeteroInfer vs. llama.cpp (CPU)、MLC (GPU)、MNN (GPU)、llm.npu (NPU)、PowerInfer-2 (NPU) 在 multi-turn dialogue (BELLE)、simple QA (GSM8K)、long-text processing (LongBench-TriviaQA) 三个 benchmark 上的表现；(2) prefill 阶段固定序列长度（64/256/1024）下各框架的 prefill speed 对比；(3) 动态序列长度下 Hetero-tensor vs. Online-prepare、Padding、NPU-pipe 的 prefill latency 对比；(4) decoding 阶段各框架 decoding rate 对比（prompt seq_len=256）；(5) fast synchronization 消融实验（prefill + decoding）；(6) GPU performance interference——与手游 (League of Legends: Wild Rift) 并发运行时的 FPS 和推理速度影响；(7) 能耗对比（GPU-only / Hetero-layer / Hetero-tensor）。

- 硬件平台是什么，配置是什么。
  - 主测试平台：Qualcomm Snapdragon 8 Gen 3 SoC（Arm CPU + Adreno 750 GPU ~1 TFLOPS FP16 实际/2.8 TFLOPS 理论峰值 + Hexagon NPU ~10 TFLOPS FP16 实际/17 TFLOPS 理论峰值），统一内存架构，理论内存带宽 68 GB/s
  - 辅助平台：Qualcomm Snapdragon 8 Elite（用于验证跨代 SoC 可移植性，内存带宽与 8 Gen 3 相同故 decoding 性能相似，prefill 性能提升约 10.5%）

- 开源Serving框架是什么。修改了什么。
  - HeteroInfer 是自建的工业级 LLM 推理引擎，并非基于现有开源 serving 框架修改。其 GPU kernel 使用 OpenCL 开发，NPU 算子通过 Qualcomm QNN (Qualcomm AI Engine Direct SDK) 集成。
  - 对比的 baseline 框架包括：llama.cpp [11]（CPU/GPU）、MLC-LLM [34]（GPU）、MNN-LLM [55]（GPU）、llm.npu [56]（NPU）、PowerInfer-2 [60]（NPU）、Qualcomm-AI [42]（NPU）、Onnxruntime [33]（CPU/NPU）。这些框架均仅使用单一加速器（GPU-only 或 NPU-only）。
  - HeteroInfer 的核心修改：(1) 新增 GPU-NPU 异构并行调度层——在 layer-level 和 tensor-level 两个粒度协调 GPU 和 NPU 的计算任务分配；(2) 新增 fast synchronization 机制——替代传统的 clFinish（~400μs）同步方式，利用 UMA 共享内存 + 可预测等待 + CPU 轮询 flag bit 实现微秒级同步；(3) 新增 tensor partitioning solver——基于 offline profiling 结果，在线为每个算子选择最优分区策略（weight-centric / activation-centric / hybrid / no-partition）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明 HeteroInfer 自身代码是否已开源。通过 web search 未找到公开 GitHub 仓库。论文 arXiv 版本: https://arxiv.org/html/2501.14794。
  
  - **HeteroInfer Serving 框架输入到硬件执行全过程（以 Llama-8B 的 prefill 阶段为例）**：
    
    **阶段 1：离线 Profiling & Solver**
    ```
    输入: 目标 SoC (Snapdragon 8 Gen 3), 目标模型 (Llama-8B)
    
    1. Performance Profiler:
       对 LLM 中所有 Matmul 算子的权重张量形状（如 [4096,4096], [28672,4096], [4096,14336]）
       在 GPU (OpenCL) 和 NPU (QNN) 上分别测量：
         - 各标准激活形状（seq_len=64/128/256/512/1024）下的执行延迟
         - 内存带宽
         - 同步开销
       约束条件：
         - 仅考虑 LLM 权重张量形状（非全空间搜索）
         - NPU stage performance 限制子张量最小尺寸（≥32，对应 32×32 systolic array）
         - 激活张量仅使用预定义标准序列长度
       耗时：< 20 分钟
    
    2. Tensor Partitioning Solver:
       对每个 Matmul 算子枚举所有可行并行策略:
         min( max(T_GPU_partition1, T_NPU_partition2) + T_sync + T_copy,
              T_GPU_all, T_NPU_all + T_sync + T_copy )
         s.t. Partition1 + Partition2 = All
       
       输出: 每种 (weight_shape, activation_shape) 组合下的最优 partition strategy 和 partition ratio
       示例输出 (Table 3):
         - [4096,4096] × [4096,1]   → Weight-centric, 1:1 (GPU:NPU)
         - [28672,4096] × [4096,1]  → Weight-centric, 3:1
         - [4096,14336] × [14336,1] → No partition, GPU-only
         - [4096,4096] × [4096,128] → No partition, NPU-only
         - [4096,4096] × [4096,256] → No partition, NPU-only
         - [4096,4096] × [4096,257-272] → Activation-centric, Dynamic:256
         - [4096,14336] × [14336,257-384] → Hybrid, 2:3 (Weight)
    
    3. Graph Pre-generation:
       对标准形状预生成 NPU 计算图（QNN），避免运行时 graph compilation 开销。
    ```
    
    **阶段 2：在线推理 — Prefill 阶段（NPU-dominant）**
    ```
    输入: 用户 prompt tokens（例如 seq_len=320）, 模型权重 (W4A16)
    
    1. Control Plane Decider 根据 solver 输出和当前 seq_len 选择执行策略:
       - 逐层遍历 decoder block 中的算子
       - 对 Attention Projection (Q/K/V/O): 根据 affinity 分配
       - 对 FFN Up/Gate/Down: 根据 (weight_shape, activation_shape) 查表选策略
    
    2. 以 FFN-down layer [4096,14336] × [14336,320] 为例（hybrid partition）:
       
       Step A — Activation-centric 分解 (处理动态 seq_len=320):
         activation [14336, 320] → 拆分为 [14336, 256] + [14336, 64]
         - [14336, 256]: 标准形状 → NPU 预生成图直接执行
         - [14336, 64]: 动态形状 → 分配给 GPU
    
       Step B — Weight-centric 分解 (处理 shape-sensitive NPU):
         weight [4096, 14336] → 对 NPU 子任务按行拆分
         因 NPU-3 shape-sensitive: 14336 列过大，NPU 效率仅 ~0.5-1.5× GPU
         partition ratio = 2:3 (40% GPU, 60% NPU)
         - GPU 子任务: [4096*0.4, 14336] 权重 × [14336, 64] 激活 → OpenCL kernel
         - NPU 子任务: [4096*0.6, 14336] 权重 × [14336, 256] 激活 → QNN graph
    
       Step C — Fast Synchronization (NPU-dominant prefill):
         NPU 提交任务 → GPU 提交任务 → CPU sync thread sleep(predicted_wait)
         → CPU 轮询 GPU output tensor 的 flag bit（数微秒）
         → flag bit 置位 → 立即通知 NPU 继续下一层
    
       Step D — Result Merge:
         GPU output [seq_len, 4096*0.4] + NPU output [seq_len, 4096*0.6]
         → 拼接为 [seq_len, 4096] → 传给下一层
    
    3. 逐层重复直至最后一个 decoder block → 输出 first token
    
    注: 对于 Matmul 算子，利用计算不变量交换张量顺序:
      [M,N] × [N,K] → [[K,N] × [N,M]]^T
    使较小的 K 维作为 weight 维度，利用 NPU weight-stall 范式。
    ```
    
    **阶段 3：在线推理 — Decoding 阶段（GPU-dominant）**
    ```
    输入: 已生成的 token（seq_len=1）, KV cache
    
    1. Decoding 阶段每个 Matmul 变为 matrix-vector 乘法 [M,K] × [K,1]
       - 由于 NPU-1 stage performance（32×32 systolic array 对小 K 效率低）
       - GPU 在矩阵-向量操作上性能更优且带宽更稳定
       → GPU 成为主要计算单元
    
    2. Weight-centric partition（关注内存带宽利用）:
       - GPU 承担主要计算（~75% partition）
       - NPU 承担辅助计算（~25% partition）
       - 并行执行使内存带宽从 ~43.3 GB/s (GPU-only) 提升至 ~59.5 GB/s
    
    3. Fast Synchronization (GPU-dominant decoding):
       NPU 提交任务 → GPU 提交任务（GPUkernel_1）
       → NPU 完成后立即 enqueue GPUkernel_2（GPU queue ordering 保证正确同步）
       → GPU 连续执行无需额外同步开销
       → NPU 执行时间被 GPU 执行时间完全覆盖
    
    4. 逐层重复 → 输出 next token → 直至 EOS
    ```
    
    **端到端流程示例（GSM8K simple QA，prefill=296 tokens，decode=340 tokens）**：
    ```
    User Prompt (296 tokens)
    ↓
    [Control Plane: 查询 solver output → 确定每层 partition strategy]
    ↓
    [Prefill Phase (NPU-dominant)]:
      Layer 0-31 逐层执行:
        Attention Q/K/V/O: NPU 执行 Matmul, GPU 执行 RMSNorm
        FFN Up/Gate:     NPU 执行 Matmul (大 M×K, 适合 NPU)
        FFN Down:        Hybrid partition (GPU-NPU 并行)
        Synchronization: Fast sync (NPU 执行覆盖 GPU + sync 延迟)
      → TTFT (Time to First Token)
    ↓
    [Decoding Phase × 340 iterations (GPU-dominant)]:
      每 iteration:
        Attention Q/K/V: GPU 执行 (matrix-vector, 高效)
        FFN Up/Gate:     Weight-centric partition (GPU:NPU = 3:1)
        FFN Down:        GPU-only (因 [4096,14336] 不适合 NPU 小激活)
        KV Cache 更新:   UMA 共享内存, 无需数据拷贝
      → 每个 output token
    ↓
    端到端延迟: HeteroInfer 2.62× 平均加速 vs. baselines
    ```

## Fast On-device LLM Inference with NPUs

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：llm.npu 在 MLLM 推理框架上实现两种 Serving 级调度技术：(1) **Chunk-sharing graph execution（块共享图执行）**——将变长 prompt 分割为固定长度 chunk（默认 256 tokens），利用 decoder-only LLM 的因果依赖特性，预构建固定大小的 NPU 计算图。进一步将 LLM 算子分为静态算子（Linear、LayerNorm，仅依赖 chunk 长度，可跨 chunk 共享）和动态算子（Attention，依赖 chunk 序列位置，不可共享），将共享子图构建一次、非共享子图按 chunk 独立构建。120/144 子图可共享，减少 up to 75% (7.2GB) 内存开销。(2) **Out-of-order subgraph execution（乱序子图执行，OOE）**——利用 chunk 和子图两级划分后子图间的依赖关系（跨 chunk 依赖和 chunk 内依赖），打破严格 chunk 顺序调度，采用在线启发式贪心算法：以 NPU 为关键路径，每次选择对减少 NPU 停顿贡献最大的待执行子图。贡献值 C = 若子图在 CPU/GPU 执行则为释放出的 NPU 子图总时间；若在 NPU 执行则为负的释放的 NPU 子图总时间。调度开销为微秒级。
  - 实验比较：(1) llm.npu prefill speed vs. llama.cpp-CPU、MNN-CPU、MLC-GPU、TFLite-GPU、PowerInfer-V2-NPU，在 prompt length=64/256/1024 下评估，在 Redmi K70 Pro 和 Redmi K60 Pro 两台设备上测量；(2) 端到端延迟对比（prefill+decode），涵盖 LongBench (2wikimqa, TriviaQA)、DroidTask (clock/setting)、Persona-Chat 三个真实应用场景；(3) 消融实验：naive NPU offload → +chunk-sharing graph → +shadow outlier execution → +OOE 的逐级 prefill speed 对比；(4) GPU-NPU vs CPU-NPU 协同的 prefill speed 和端到端延迟对比（GPU-NPU 端到端减少 80–90ms）；(5) 能耗对比（llm.npu vs llama.cpp-CPU、MLC-GPU、TFLite-GPU，Redmi K60 Pro）。

- 硬件平台是什么，配置是什么。
  - Redmi K70 Pro：Qualcomm Snapdragon 8 Gen 3 SoC，Hexagon NPU，24GB 内存，Adreno 750 GPU，Android 13
  - Redmi K60 Pro：Qualcomm Snapdragon 8 Gen 2 SoC，Hexagon NPU，16GB 内存，Android 13
  - NPU：1024-bit INT8 SIMD，73 TOPS (INT8)，500–750 MHz，统一内存架构
  - GPU 对比：TFLite-GPU 在 Redmi K70 Pro 上评估

- 开源Serving框架是什么。修改了什么。
  - 基础框架：MLLM（https://github.com/UbiquitousLearning/mllm）——移动端 LLM 推理引擎；QNN（Qualcomm Neural Processing SDK）——Qualcomm NPU 推理框架
  - llm.npu 修改：(1) 新增 chunk-sharing graph 构建模块——在 preparation stage 将 LLM 模型拆分为共享/非共享子图，预生成固定大小 NPU 计算图，处理变长 prompt 的图编译开销（原 QNN 对 Gemma-2B 需 11.54s 图优化，chunk 方法消除每次重建开销）；(2) 新增 out-of-order scheduler——在 execution stage 实现微秒级在线调度算法，管理 CPU/GPU 和 NPU 间的子图执行顺序，减少执行气泡；(3) 新增 shadow outlier execution 的 CPU-NPU 同步机制；(4) 实现 KVCache、SiLU、RMSNorm、ROPE 等 LLM 特定算子（QNN 原生不支持）；(5) shared buffer 同步——利用统一内存架构减少 CPU/GPU/NPU 间上下文切换开销。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：代码开源在 https://github.com/UbiquitousLearning/mllm（MIT 许可证），约 10K 行 C/C++ + 汇编。Artifact DOI: https://doi.org/10.5281/zenodo.14392760。
  
  - **llm.npu Serving 框架输入到硬件执行全过程（以 Qwen1.5-1.8B, prompt_len=1024 为例）**：

    ```
    # ========== 阶段 0：准备阶段 (Preparation Stage, 离线, ~分钟级) ==========
    # 仅在模型首次加载时执行一次

    1. 模型量化:
       FP16 模型 → 增强 per-tensor W8A8 量化 (max-min symmetric)
       异常值重要性分析 → 标记 85% 层为剪枝层
       热通道识别 → CPU 内存保留热通道权重副本

    2. Chunk-sharing graph 构建 (chunk_len=256):
       # 算子分类
       静态算子: Linear (Q/K/V/O/Gate/Up/Down), LayerNorm
                → 仅依赖 chunk_len, 可跨 chunk 共享
       动态算子: Attention (Q·K^T, Softmax, attn·V)
                → 依赖 chunk 序列位置, 每个 chunk 独立构建

       # 子图划分 (以 Qwen1.5-1.8B 为例, 共 144 子图)
       Graph_shared_1:  Q/K/V Linear → 共享
       Graph_dynamic_1: RoPE + Attention Score + Softmax
                        → chunk 0: [32,hds]×[32,hds]
                           chunk 1: [32,hds]×[64,hds]
                           ...
                           chunk N: [32,hds]×[256+32*(N-1),hds] (动态维度)
       Graph_shared_2:  O Linear + Residual + LayerNorm → 共享
       Graph_shared_3:  FFN Gate/Up Linear + SiLU → 共享
       Graph_shared_4:  FFN Down Linear + Residual + LayerNorm → 共享
       # 120/144 子图可共享, 节省 75% 内存 (7.2GB, prompt_len=1024)

    3. Profile 所有子图在 NPU/CPU 上的执行时间 (offline):
       对每种可能形状的子图, 测量:
         T_npu(graph_shape)  # NPU 执行时间
         T_cpu(graph_shape)  # CPU 执行时间
       构建子图间依赖关系 DAG:
         跨 chunk 依赖: G[i,j] ← G[0,j-1], G[1,j-1], ..., G[i,j-1]  (Attention)
         块内依赖:     G[i,j] ← G[i,j-1]                           (其余算子)

    4. 形状优化: profile 张量等价形状, 选择 NPU 最高效形状
       e.g. [1024,1,2048] → reshape → [32,32,2048] (1.62× 加速)

    # ========== 阶段 1：推理执行阶段 (Execution Stage, 在线) ==========

    输入: user_prompt = "Forward the unread emails to Alice about the Q3 budget..."
          经过 tokenizer → [t1, t2, ..., t1024], 共 1024 tokens
          chunk_len = 256
          num_chunks = ceil(1024/256) = 4

    Step 1 — Prompt 分块:
      chunk_0: tokens[0:256]     + padding (如有)
      chunk_1: tokens[256:512]
      chunk_2: tokens[512:768]
      chunk_3: tokens[768:1024]
      # 最后一个 chunk 不足 256 的用零填充

    Step 2 — 初始化调度器:
      pending_subgraphs = [
        (chunk=0, graph_idx=1),   # 每个 chunk 的第一个子图初始化就绪
        (chunk=1, graph_idx=1),
        (chunk=2, graph_idx=1),
        (chunk=3, graph_idx=1),
      ]
      ready_queue = NPU: [G[0,1], G[1,1], G[2,1], G[3,1]]
                    CPU: []
      completed = {}

    Step 3 — Out-of-Order 调度循环 (微秒级每步):
      while pending_subgraphs 非空:
        # 对每个就绪子图计算贡献值 C
        for each ready subgraph g:
          S = {g' | g' 在 g 完成后变为就绪}
          if g 分配到 CPU/GPU:
            C(g) = sum(T_npu(g') for g' in S if g' 分配到 NPU)
            # 越大越好: 释放更多 NPU 工作量
          else:  # g 分配到 NPU
            C(g) = -sum(T_npu(g') for g' in S if g' 分配到 CPU/GPU)
            # 越大越好: NPU 执行完此子图后, 释放的 CPU 工作越少越好

        选择 C 最大的子图 g* 提交到对应处理器执行
        更新 ready_queue 和 pending_subgraphs

      # 实际执行时间线示例 (Qwen1.5-1.8B, prompt=1024, NPU:CPU ≈ 2:1):
      # NPU:  |===G[0,1]===|===G[1,1]===|===G[0,3]===|===G[2,1]===| ...
      # CPU:     |G[0,2]|               |G[1,2]|  |G[0,4]|           ...
      # OOE 将 bubble rate 从 naive 37% 降至最低

    Step 4 — 子图在硬件上的执行:
      对分配到 NPU 的子图 (shared/dynamic):
        QNN 加载预构建计算图 → 从 unified memory 读取输入激活
        → Hexagon NPU 1024-bit SIMD 执行 INT8 MatMul
        → 写回 unified memory (shared buffer)
      对分配到 CPU 的子图:
        MLLM CPU backend 执行 FP16 算子 (LayerNorm, Attention, RoPE, SiLU)
        + shadow outlier execution (如果该层未被剪枝)
        → 结果写回 unified memory
      同步: 通过 unified memory flag bit 传递完成信号

    Step 5 — 结果收集:
      所有 chunk 的所有子图完成后:
        KV Cache: 已按 chunk 顺序累积在 unified memory 中
        Last hidden state: chunk_3 的最后一层输出
        → 传递给 decoding 阶段 (MLLM CPU backend, 逐 token 生成)

    # ========== Chunk Length 与性能权衡 ==========
    # 在 Xiaomi 14 (Snapdragon 8 Gen 3) 上的 profile:
    chunk_len=32:   NPU 利用率低, per-token latency 高 (小 MatMul 不能充分用 NPU)
    chunk_len=64:   QKV Linear per-token latency ~0.020ms, FFN ~0.030ms
    chunk_len=128:  QKV ~0.012ms, FFN ~0.018ms
    chunk_len=256:  QKV ~0.008ms, FFN ~0.012ms  ← llm.npu 选择 (最优)
    chunk_len=512:  QKV ~0.008ms, FFN ~0.012ms  (与 256 持平, 但 padding 增加)
    chunk_len=1024: 大量 padding 浪费 (短 prompt 时严重)
    ```

  - **端到端真实应用性能示例（Redmi K70 Pro, Qwen1.5-1.8B）**：
    - LongBench 2wiki-Multi-doc QA (prompt ~1600 tokens, output ~3 tokens):
      - 端到端: llm.npu 1.7s vs llama.cpp-CPU 45.6s (26.8×) vs MLC-GPU 78.4s
      - Prefill: llm.npu 1.49s vs llama.cpp 45.43s
    - DroidTask clock (prompt ~700 tokens, output ~3 tokens):
      - 端到端: llm.npu 1.4s vs llama.cpp-CPU 21s (15×) vs MLC-GPU 46.6s
    - Persona-Chat (prompt ~530 tokens, output ~46 tokens):
      - 端到端: llm.npu 6.72s vs MLC-GPU 18.74s (2.8×)
      - Speedup 降低原因: 解码 token 多, CPU decoding 成为瓶颈

## LLM as a System Service on Mobile Devices

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：LLMS，一个面向移动设备的 LLM-as-a-System-Service (LLMaaS) 原型系统，核心理念是将 LLM 作为系统级服务暴露给所有 app 共享使用（类似 Android 的 Location Service），而非每个 app 各自加载 LLM。LLMS 的核心是实现高效的 LLM context 内存管理，解耦 app 内存与 LLM context 内存的管理，通过细粒度、chunk-wise、全局优化的 KV cache 压缩与交换来最小化 context switching 延迟。LLMS 包含三个关键技术：(1) Tolerance-Aware Compression（容忍度感知压缩）——利用 attention scores 计算每个 chunk 的信息密度来决定压缩率，在保证全局平均压缩比的前提下最大化整体信息强度；(2) Swapping-Recompute Pipeline（交换-重计算流水线）——将部分 chunk 的重计算与其余 chunk 的磁盘 I/O 以流水线方式重叠执行，修改 LLM 的 position encoding 和 causal mask 以支持不连续 chunk 的重计算；(3) Chunk Lifecycle Management（chunk 生命周期管理）——使用 LCTRU (Least Compression-Tolerable and Recently-Used) 队列决定 eviction 优先级，采用 ahead-of-time (AoT) swapping-out 在 callLLM() 返回阶段提前换出已修改 chunk 以隐藏回收延迟。
  - 实验比较：(1) 端到端 context switching latency 对比 LLMS vs. LMK（Android low-memory killer）、Swapping（整上下文交换）、VLLM-S（chunk-wise KV cache 管理无压缩）、VLLM-SQ（chunk-wise + 统一 INT8 量化），在 72 小时合成 trace 上评估 2/4/6/8/12/16 个 active contexts 的切换延迟；(2) 不同 memory budget（1GB/2GB/3GB）下 max number of active contexts 对比；(3) 不同 maximal context length（256-4096 tokens）下 active contexts 数量对比；(4) 三种 context switching pattern（Random/Markov/Gaussian）下的性能一致性；(5) 压缩效率对比：LLMS tolerance-aware compression vs. 静态均匀量化（4-bit/2-bit）的 accuracy-compression ratio trade-off；(6) 消融实验：逐步移除 tolerance-aware compression / swapping-recompute pipeline / chunk lifecycle management 的影响；(7) chunk size 选择实验（不同 token/chunk 数下 switching latency 变化）；(8) LLM 推理性能稳定性分析（LLMS 对正常推理的影响，within 5%）；(9) Service calling frequency 敏感性分析。

- 硬件平台是什么，配置是什么。
  - Jetson Orin NX：8 GB RAM，NVMe SSD，1024-core Ampere™ GPU
  - Jetson TX2：8 GB RAM，SATA HDD（磁盘带宽较低），256-core Pascal™ GPU
  - MI14 Smartphone：8 GB RAM，UFS 4.0 存储，Hexagon™ 8Gen3 NPU（Qualcomm Snapdragon 8 Gen 3）
  - 注：所有设备均为 8 GB RAM，LLMS 在 TX2 上因 SATA HDD 低带宽导致整体 switching latency 更长，但仍显著优于 baseline

- 开源Serving框架是什么。修改了什么。
  - LLMS 不是基于现有 serving 框架修改，而是自建的 LLM Service 原型（3.5k LoC Python/C++），构建在 Huggingface Transformers 和 mllm 之上（分别用于 Jetson 设备和 MI14 智能手机）。LLMS 作为独立进程运行，通过 socket IPC 接收来自客户端进程的推理请求。
  - LLMS 修改/新增的核心模块：
    (1) Context Memory Management Module：嵌入 LLM Service 内部，实现 claim/reclaim/load/fault 四个内存操作原语——Claim 直接分配空闲内存给 chunk；Reclaim 在内存压力下将 chunk 换出到磁盘；Load 在调用 callLLM() 前将缺失 chunk 从磁盘加载到内存；Fault 在每次 LLM 推理迭代时按需加载缺失 chunk（保留用于异常处理如系统崩溃）。
    (2) Tolerance-Aware Compression：在 LLM 推理框架现有 KV cache 量化（如 LMDeploy 的 INT8）之上，对低信息密度 chunk 进一步执行 channel-wise 线性量化（4-bit/2-bit），使用并行 bit-shift 操作将 sub-byte 数据打包为 INT8 格式。
    (3) Swapping-Recompute Pipeline：多线程实现——独立 I/O 线程从磁盘加载 chunk 到内存，计算线程在当前层 I/O 完成后才进入下一层（流水线同步），修改 position encoding（全局编码不连续 token）和 causal mask 以支持不连续 chunk 的正确重计算。
    (4) Chunk Lifecycle Management：LCTRU 队列由多个按压缩率分组的 LRU 子队列串联组成，AoT swapping-out 在 callLLM() 返回阶段将已修改 chunk 换出。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确说明 LLMS 自身代码是否已开源。论文声明将公开实验使用的 context switching traces。LLMS 依赖的开源组件：Huggingface Transformers、mllm、GPTQ、Pickle (Python/C++)、Llama2-7B、OPT-6.7B。

  - **LLMS Serving 框架输入到硬件执行全过程（以 MI14 上 Llama2-7B 的 context switching 为例）**：

    **阶段 0：LLM Service 初始化与 LLMS API**
    ```
    # LLMS 定义四类 API（兼容 Android Service 模式）

    Class LLMService:              # 系统级 LLM 服务类，类似 Android Service
    Class LLMCtx:                  # LLM 上下文类，封装状态（主要含 KV cache）
    Method bindLLMService(app):    # App 绑定 LLM Service
    Method newLLMCtx(sysPrompt):   # 创建新 LLM 上下文（返回 LLMCtxStub）
    Method callLLM(ctx, prompt):   # 调用 LLM（输入上下文 + 新 prompt，返回输出 + 更新上下文）
    Method delLLMCtx(ctx):         # 删除上下文

    # 配置参数（OS 可配置）：
    # - K: 每个 app 最大 active context 数
    # - max_context_length: 每个 context 最大 token 数（Llama2=4K, OPT=2K）
    # - ratio_global: 全局平均 KV cache 压缩比（默认 50%）
    # - chunk_size: 每个 chunk 包含的 token 数（默认 16）
    # - {ratio_w}: 可用压缩级别 {8/8, 4/8, 2/8}（即 INT8/INT4/INT2）
    ```

    **阶段 1：App 调用 LLM Service（以 Chatbot 为例）**
    ```
    输入: App 发送 socket IPC 请求到 LLM Service 独立进程
          ctx = "历史对话 KV cache", newPrompt = "What's the weather today?"

    1. LLMS 接收 callLLM(CtxStub, newPrompt):
       - 解析 CtxStub → 定位磁盘和内存中的 chunk 状态
       - 触发 Load primitive: 将缺失 chunk 从磁盘加载到内存

    2. 内存布局（LLMS Memory Model）:
       Context 分为两部分:
         Memory-resident fragment: prompt/output text（不可换出，始终在内存）
         Swappable fragment: KV cache chunks（可压缩、可换出到磁盘）
       KV cache layout: token 维度增长，每 chunk = 16 tokens × 所有层 × 所有 head
       e.g. Llama2-7B, 4K tokens: 256 chunks per context
            每个 chunk 原始大小 ≈ 2GB / 256 ≈ 8 MB
            （INT8 后 ~8 MB, INT4 后 ~4 MB, INT2 后 ~2 MB）
    ```

    **阶段 2：Context Switching — Load 阶段（三种技术协同工作）**

    ```
    # === 子阶段 2A: Tolerance-Aware Compression 决定 chunk 压缩率 ===
    # 压缩决策在上一次 token generation 完成时已执行，此处说明原理

    对 context 中每个 chunk_i（16 tokens）:
      1. 计算 attention score matrix A [R×C]:
         A = softmax(mask(Q·K^T / sqrt(d_k)))
         对每列 col, 对每个 head h 和 layer l:
           计算 token-level 信息密度 = avg(attention scores paid to this token)
       
      2. 计算 chunk-level 信息密度 D_i (Equation 1):
         D_i = 1/(q-p) · Σ_col Σ_layer Σ_head avg(A[*, col] in head h, layer l)
         含义: 被更多其他 token "关注" 的 chunk 信息量更大，压缩容忍度更低
       
      3. 排序: Rank_i = D_i 在所有 chunk 中的百分位排名
       
      4. 确定压缩阈值 {σ_ratio} (Equation 3):
         在全局平均压缩比 ratio_global=50% 约束下
         maximize ctxInfo = Σ_w (1/ratio_w) · Σ_{chunk in [σ_{w+1}, σ_w]} D_i
         → σ_8/8 (top 排名 chunk: 保持 INT8, 不额外压缩)
         → σ_4/8 (中间排名: 从 INT8 再压到 INT4)
         → σ_2/8 (低排名: 从 INT8 再压到 INT2)

    结果示例: 256 chunks 中 top ~30% 保持 INT8, middle ~40% 压到 INT4, bottom ~30% 压到 INT2
              总内存: 256×8MB×(0.3×1 + 0.4×0.5 + 0.3×0.25) ≈ 1178 MB
              vs 不压缩 2048 MB → 节省 ~43% 内存，无明显精度损失
    ```

    ```
    # === 子阶段 2B: Swapping-Recompute Pipeline 加载缺失 chunk ===
    # 假设 callLLM() 触发时，目标 context 有 60% chunk 已在内存，40% 在磁盘

    1. Profiling (安装时 offline 一次性):
       对当前设备测量:
         T_re(x, f, e): 重计算 x 个 chunk 的延迟（取决于 chunk 数、CPU/GPU 频率、能耗模式）
         T_IO(m): 从磁盘加载 m MB 的延迟
       实践中用线性函数近似，通过离散测试点拟合

    2. Planning (Equation 4, 线性规划求解):
       minimize pipelineDelay = max(T_re(Σx_re), T_IO(m_onload - Σratio_w · x_re))
       s.t. x_re^{ratio_w} < x^{ratio_w}
       
       给定: m_onload ≈ 471 MB（需加载的总内存大小）
             {x^{8/8}=20, x^{4/8}=50, x^{2/8}=30}（各压缩率下缺失的 chunk 数）
       
       求解: 决定重计算哪些 chunk 来最大化 I/O 与 recompute 的重叠
             优先重计算压缩率低的 chunk（无压缩 → 重计算不涉及 I/O，直接利用 CPU/GPU）

    3. Pipeline 执行 (多线程):
       I/O Thread:       |==Load L0 K/V==|==Load L1 K/V==|==Load L2 K/V==| ...
       Compute Thread:  |==Recompute L0==|==Recompute L1==| ...
       同步条件: Compute thread 在当前层 I/O 完成后才进入下一层

    4. Chunk Recompute Procedure (处理不连续 chunk):
       例如 context 文本 "a b c d e f" 中 c 和 e 的 KV cache 被换出:
         → Embed "c" 和 "e" 为 token → 全局 position encoding (pos_c=3, pos_e=5)
         → 重计算 Q, K, V → 插入到已有 K/V 的对应位置
         → 应用 causal mask: c 只能看到 a b c, e 只能看到 a b c d e
         → 每层重计算后进入下一层 pipeline
    ```

    ```
    # === 子阶段 2C: Chunk Lifecycle Management 决定 eviction ===
    # 在 callLLM() 返回阶段和后续内存压力时触发

    1. AoT (Ahead-of-Time) Swapping-out:
       callLLM() returning stage:
         识别所有在本次推理中被修改的 chunk → 立即写入磁盘
         即使当前无内存压力也执行（writeback 而非 eviction）
         延迟对 caller 不可感知（发生在 token 生成完成后到返回前）

    2. LCTRU Queue Eviction (当 Reclaim primitive 被触发):
       LCTRU queue 结构:
         Q_{LCTRU} = [Q_{8/8}] → [Q_{4/8}] → [Q_{2/8}]
         每个子队列按最近访问时间排序（LRU）
         头部的子队列（低压缩率 → heavy chunk）优先被 evict
       
       Eviction 决策（需要释放 M bytes 时）:
         从 Q_{8/8} 头部开始 pop → 换出到磁盘
         若 Q_{8/8} 空 → pop Q_{4/8}
         若 Q_{4/8} 空 → pop Q_{2/8}
       
       设计原理:
         - Heavy chunk first: 重计算 pipeline 中，更少的 chunk 数 → 更低的重计算延迟
         - LRU within same ratio: 利用 context 访问的时间局部性
       
    3. Working Set Lock:
       callLLM() 执行期间，LLMS 锁定当前 context 的所有 chunk
       → 禁止 Reclaim 回收自己的 chunk（避免 thrashing）
       → Fault primitive 保留用于异常处理（如系统崩溃后恢复）
    ```

    **阶段 3：LLM 推理执行与 Token Generation**
    ```
    所有 chunk 就绪后，LLM 正常执行自回归推理:
      Prefill phase: 新 prompt tokens → 生成新 KV cache
      Decode phase:  逐 token 生成
      LLMS 不干预 LLM 推理过程本身，仅负责推理前的 context 准备
    
    LLMS 使用的推理配置:
      - 权重量化: GPTQ W4A16 (4-bit INT)
      - KV cache 默认: INT8 (SmoothQuant 类方法)
      - LLMS 额外压缩: chunk-wise 4-bit/2-bit (基于 tolerance-aware)
      - 滑动窗口 attention: streaming LLM
      - 框架: Jetson → HuggingFace Transformers + PyTorch; MI14 → mllm
    ```

    **端到端 Context Switching 示例（MI14, Llama2-7B, 8 active contexts, Markov pattern）**：
    ```
    App 发送 callLLM(ctx_id=5, "Summarize the previous emails about Q3 budget")
    ↓
    [LLMS Load Phase]:
      - 识别 ctx_5 当前状态: 256 chunks 中 150 在内存, 106 在磁盘
      - LCTRU queue: 决策是否需先 evict 其他 context 的 chunk（若内存不足）
      - AoT: 前一次 callLLM() 返回时 ctx_5 的修改 chunk 已在磁盘（无需额外 I/O）
    ↓
    [LLMS Swapping-Recompute Pipeline]:
      - Planning: T_re vs T_IO 分析 → 决定重计算 40 个 INT4/INT2 chunk + I/O 加载 66 个
      - Pipeline 执行: I/O 和 recompute 重叠 → ~0.27s (vs. Swapping baseline ~27s)
    ↓
    [LLM Inference]:
      - Prefill: new prompt tokens → 生成新 KV cache chunk
      - Decode: 逐 token 生成 "The Q3 budget emails discuss..."
      - 新生成的 KV cache chunk 按 tolerance-aware compression 压缩
    ↓
    [LLMS AoT Writeback]:
      - callLLM() 返回前: ctx_5 新修改 chunk → 写入磁盘
      - LCTRU queue 更新: ctx_5 的 chunk 移到各自子队列末尾
    ↓
    返回: 生成的 tokens + 更新后的 LLMCtxStub 给 App
    总 context switching 延迟: ~0.27s（LLMS）vs. LMK recompute ~22.92s（85× reduction）
    ```

  - **关键性能数据**（72h trace, 平均 switching latency）：
    - LLMS vs LMK: up to 2 orders of magnitude reduction
    - LLMS vs Swapping: 1-2 orders of magnitude reduction
    - LLMS vs VLLM-SQ (chunk + INT8): up to 20×, average 9.7× reduction
    - 消融: 全部技术 → 0.27s; 去掉 lifecycle mgmt → 0.62s; 去掉 tolerance compression → 0.42s; 去掉 recompute pipeline → 1.62s
    - 10ms latency constraint, 3GB budget: LLMS 支持 16.32 contexts vs baseline 5.73 (2.85×)

## Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：FUSE（Fully Unified Scheduling for Energy），一个面向移动端 LLM 推理的统一能耗感知 DVFS governor，作为 llama.cpp 的扩展实现（~2K 行 Python）。核心实现包括：(1) 离线 profiling-based 频率搜索——给定 LLM 模型，对 prefill（5 个代表性长度范围）和 decode（1 个采样长度）分别搜索最优 CPU/GPU/内存频率组合，搜索过程分两步（先搜 GPU 频率抑制 antagonistic effect，再搜 CPU 频率微调），将搜索次数从全组合 2808 降至平均 14.5 + 30.8 次推理；(2) 运行时频率固定——在线推理时，根据 inference framework 的阶段通知（prefill 开始/结束、decode 开始/结束）查表并 pin 住 CPU/GPU 到最优频率组合；(3) 支持两种目标：G1（给定能耗预算最小化延迟）和 G2（给定延迟目标最小化能耗）。
  - 实验比较：(1) FUSE vs. 默认 Android DVFS governors (Gov: sched-pixel EAS + Quickstep GPU + Interactive Memory) 在 6 个模型上的 TTFT、TPOT、能耗对比（G1: 同能耗比延迟; G2: 同延迟比能耗）；(2) FUSE 在 ShareGPT 真实 trace（200 请求，prefill ≤512 tokens, decode ≤256 tokens）上的端到端性能（TTFT/TPOT/E2E/energy）对比 Gov；(3) FUSE 频率搜索效率——搜索步数和总搜索时间（分钟级，vs. 全组合搜索需 374× 更多推理）；(4) 各 governor 隔离实验（GPU governor only / EAS only / Memory governor only）的延迟-能耗分析；(5) antagonistic effect 实时 trace 捕获与根因分析实验（CPU 和 GPU 频率的交叉影响 CDF、双向级联下降实时记录）。

- 硬件平台是什么，配置是什么。
  - Google Pixel 7：Google Tensor G2 SoC（ARM Cortex-X1 性能核心 + ARM Cortex-A55 LITTLE 核心）、ARM Mali-G710 MP7 GPU、8GB DRAM、Android 13（stock, rooted, 电池 bypass）
  - Google Pixel 7 Pro：Google Tensor G2 SoC、ARM Mali-G710 MP7 GPU、12GB DRAM、Android 13（用于 Llama-2 7B 等大模型实验）
  - 电源测量：Monsoon power monitor，每 0.2ms 报告细粒度功耗
  - CPU 可用频率：18 档（500–2850 MHz）；GPU：12 档（151–848 MHz）；Memory：13 档（421–3172 MHz）
  - 实验中屏幕关闭以避免屏幕功耗干扰

- 开源Serving框架是什么。修改了什么。
  - 基础框架：llama.cpp（tag b2202），C++ 跨平台 LLM 推理库，GPU 推理通过 OpenCL + CLBlast 库支持
  - FUSE 修改/新增：（1）新增离线 profiling 模块——自动执行频率搜索（枚举候选 GPU/CPU 频率并测量延迟和功耗），通过写 scaling_min_freq/max_freq 或 min_freq/max_freq 来 pin 住各组件频率；（2）新增运行时 governor 模块——在 llama.cpp 推理流程中插入 prefill/decode 阶段通知 hook，hook 触发时查表并设置对应频率组合；（3）新增 profiling daemon——在手机上自动执行 benchmark 脚本（因 adb 在实验中不可用——电池 bypass 状态下 USB 被 Monsoon 占用）
  - FUSE 不修改 llama.cpp 的模型推理逻辑，而是在推理框架外部（governor 层）通过 sysfs 接口控制 DVFS 频率

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文声明 FUSE 作为 llama.cpp 扩展已发布（"We have released FUSE as an extension to the llama.cpp framework"），但文中未给出具体 GitHub 仓库 URL。通过 web search 未能确认独立开源仓库。llama.cpp 自身开源在 https://github.com/ggerganov/llama.cpp（MIT 许可证）。
  
  - **FUSE Serving 框架输入到硬件执行全过程（以 TinyLlama-1.1B 在 Pixel 7 上的 G1 目标为例）**：

    **阶段 0：离线 Profiling & 频率搜索（模型安装/首次加载时执行一次）**
    ```
    输入: 目标模型 (TinyLlama-1.1B, 4-bit 量化), 目标设备 (Pixel 7)
          能耗预算 = 默认 governors 下测量的 energy-per-token
          （prefill: energy_budget_prefill, decode: energy_budget_decode）
          6 个 setting: decode × 32 tokens, prefill × {32, 64, 128, 256, 512} tokens
    
    Step 1 — GPU 频率搜索 (G1 目标):
      从最高 GPU 频率 (848 MHz) 开始递减:
        对每个 GPU 频率 f_gpu:
          固定 CPU 频率 = 默认 EAS 下 effective CPU frequency
          固定 Memory 频率 = 默认 governor
          执行一次推理 → 测量 energy_per_token(f_gpu)
          若 energy_per_token(f_gpu) ≤ energy_budget:
            停止 → 选中 f_gpu 为 candidate_1
            同时保留 f_gpu_prev (energy > budget 的前一个频率) 为 candidate_2
            （因为 candidate_2 配合更高 CPU 频率可能仍满足 budget）
    
      示例结果 (TinyLlama decode 32 tokens):
        848 MHz: 396 mJ > budget(392 mJ) → 继续
        762 MHz: 330 mJ ≤ budget → candidate_1 = 762, candidate_2 = 848
        平均 Step 1 需 2.4 次推理/setting
    
    Step 2 — CPU 频率搜索 (对每个 GPU candidate):
      对每个 candidate GPU 频率 f_gpu ∈ {762, 848}:
        固定 GPU = f_gpu, Memory = default governor
        从最高 CPU 频率 (2850 MHz) 开始递减:
          执行一次推理 → 测量 energy_per_token(f_cpu | f_gpu)
          若 energy_per_token ≤ energy_budget:
            停止 → 记录 (f_cpu, f_gpu, latency)
    
      选择 latency 最低的组合:
        若 candidate_2 (848 MHz) 配合某 CPU 频率满足 budget 且延迟更低:
          选 candidate_2 组合
        否则选 candidate_1 组合
    
      平均 Step 2 需 5.1 次推理/setting
    
    总计: 6 settings × (2.4 + 5.1) ≈ 45 次推理
    vs. 穷举 2808 组合 → 减少 62× (paper 报告 374×，因 Step 2 也需搜索)
    
    搜索耗时（per model）:
      TinyLlama-1.1B:   17.7 分钟
      StableLM-Zephyr-3B: 43.1 分钟
      Llama-2-7B:         78.5 分钟
    ```

    **阶段 1：运行时推理 — Prefill 阶段**
    ```
    输入: 用户 prompt tokens, prefill length 经分类落入对应 range
    
    1. llama.cpp 推理开始 → FUSE 接收 prefill_start 通知
    2. FUSE 查表: prefill_len=232 → 落入 range 256 → 查得 (f_cpu, f_gpu)
    3. FUSE 写 sysfs:
       echo f_cpu > /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
       echo f_cpu > /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq
       echo f_gpu > /sys/class/kgsl/kgsl-3d0/max_gpuclk
       echo f_gpu > /sys/class/kgsl/kgsl-3d0/min_gpuclk
       （Memory 保持默认 governor — §5.2 发现 memory governor 已近最优）
    4. llama.cpp 正常执行 prefill:
       CPU (ARM Cortex-X1): OpenCL runtime — 管理 command queue,
          将 GPU kernel 逐个提交到 Mali-G710 GPU（最多 2 个 outstanding entries）
       GPU (Mali-G710 MP7): 执行 OpenCL kernel — MatMul (CLBlast), Attention, FFN
       Memory (LPDDR5): 权重读取 + KV cache 读写
       CPU 与 GPU 间通过 OpenCL event/callback 同步
    5. prefill 完成 → first token 生成 → FUSE 接收 prefill_end 通知
    ```

    **阶段 2：运行时推理 — Decode 阶段**
    ```
    6. FUSE 接收 decode_start 通知 → 查表得 decode 最优频率组合
       （decode 的 CPU/GPU 频率通常高于 prefill —— decode 阶段 GPU 利用率低，
        需更高频率补偿以避免 governor 将其过度降低）
    7. FUSE 重新写 sysfs 切换到 decode 频率组合
    8. llama.cpp 逐 token decode（自回归循环）:
       每 iteration:
         CPU: 提交 GPU kernel → 管理 queue
         GPU: MatMul (batch_size=1 → matrix-vector, memory-bound)
         Memory: 读取权重 + 读写 KV cache
       decode 持续直至 EOS
    9. EOS → FUSE 接收 decode_end 通知 → 恢复默认 governors
    ```

    **对比: Default Governors (Gov) 在同一次推理中的行为**
    ```
    同一 TinyLlama-1.1B decode 32 tokens:
    
    默认 GPU governor (Quickstep):
      - 检测 GPU utilization ~50-70% (decode 阶段低利用率)
      - 查 dvfs_table → 利用率低于 target range → 降频
      - 有效 GPU 频率: 424.4 MHz (vs. FUSE pin 的 762-848 MHz)
      - 结果: TPOT 215.1 ms, 402.7 mJ/token
    
    默认 CPU governor (EAS):
      - 检测 CPU utilization ~17-25% (LLM 推理中 CPU 主要在等待 GPU)
      - 任务 load 随时间衰减 → 选择更低频率
      - 有效 CPU 频率: 1130.8 MHz (vs. FUSE pin 的 2252 MHz)
    
    Antagonistic Effect (两个 governor 同时运行):
      CPU 低频 → OpenCL runtime 慢 → GPU task 提交慢 → GPU utilization ↓
      → GPU governor 降 GPU 频率 → GPU kernel 执行慢 → CPU 等待 GPU 时间 ↑
      → CPU utilization ↓ → EAS 降 CPU 频率 → ...循环 (downward spiral)
      最终: GPU 151 MHz (最低), CPU 500 MHz (最低)
      TPOT 与 FUSE 差距可达 41%
    
    FUSE 的关键创新: 通过 offline profiling 预先知道最优频率组合,
    运行时直接 pin 住频率, 完全避开 governor 间的 antagonistic effect。
    ```

    **端到端示例（ShareGPT trace, TinyLlama-1.1B, G1 同能耗）**:
    ```
    200 个请求 (avg prefill=232.4 tokens, avg decode=70.0 tokens)
    
    Gov (默认):
      TTFT avg: 10.56s, TPOT avg: 210.7ms, E2E avg: 25.2s
      总能耗: 738.1 mAh
    
    FUSE (G1, 同能耗 737.8 mAh):
      TTFT avg: 9.04s  (-14.4%)
      TPOT avg: 157.2ms (-25.4%)
      E2E avg:  19.6s  (-22.1%)
    
    更大模型收益更显著:
      DeepSeek-R1-Distill-Qwen 1.5B: TPOT -36.8%, E2E -28.0%
      StableLM-Zephyr 2.7B:        TPOT -35.2%, E2E -24.7%
    ```

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：论文在 llama.cpp 上新增了 Hexagon NPU backend，将 NPU 作为独立计算后端集成到现有推理框架中，支持 test-time scaling（Best-of-N、Beam Search）workload。核心调度实现包括：
    **(1) CPU-NPU 协同推理调度**：使用 rpcmem (kernel dmabuf wrapper) 管理 CPU-NPU 共享物理内存，消除跨处理器数据拷贝。通过 FastRPC 启动远程 NPU session，NPU 端线程持续轮询共享内存区域的 command queue 接收 CPU 提交的计算请求。手动管理 L2 cache coherence（Snapdragon SoC 仅支持单向 CPU→NPU coherence）。算子未在 NPU 实现时自动 fallback 到 CPU（如 lm_head/logits 因 NPU 32-bit 地址空间限制保留在 CPU）。
    **(2) Test-Time Scaling Batch Execution**：利用 NPU 矩阵单元在 decoding 阶段利用率低的特性（GEMM→GEMV，31/32 行 activation tile 浪费），通过增加 batch size（多路径并行采样）高效利用 NPU 闲置算力。Batch size 从 1 增至 16 时 decoding throughput 显著提升（HMX tile 利用率从 1/32 增至 16/32），理论解码开销几乎不增。
    **(3) 无 QNN 依赖的独立推理栈**：完全绕过 Qualcomm QNN 闭源 SDK（避免固定的静态计算图和 per-tensor 量化限制），直接使用 Hexagon SDK LLVM toolchain + reverse-engineered FP16 HMX 指令，支持动态 batch size 和灵活的量化方案。
    **(4) 内存与地址空间管理**：NPU 仅有 32-bit 虚拟地址空间（最大 4 GiB，实际可用更少），lm_head（vocab_size × batch × FP16）和对应权重因尺寸过大无法放入 NPU → 保留在 CPU 执行。使用 pmap 测量 NPU dmabuf 使用量（1.5B 模型 ~1056 MiB, 3B 模型 ~2090 MiB, context=4096 tokens）。CPU memory 由 rpcmem shared buffer 统一管理。
  - 实验比较：(1) End-to-end decoding throughput 在不同 batch size (1/4/8/16) 下的扩展特性，三台设备对比；(2) NPU-based system vs. GPU-based (llama.cpp OpenCL backend) 的 decoding 和 prefill throughput 对比；(3) 与 QNN FP16 参考数据的 prefill 性能对比；(4) Test-time scaling（Best-of-N/Beam Search）accuracy-latency Pareto frontier 分析；(5) 能耗测量：通过 sysfs 接口测量不同 batch size 下 decoding 功耗和归一化能耗，与 3B 单 batch baseline 对比；(6) CPU utilization 和 memory consumption（resident + dmabuf）随 batch size 变化；(7) Prompt length（512-4096 tokens）对 decoding throughput 的影响。

- 硬件平台是什么，配置是什么。
  - OnePlus Ace3：Snapdragon 8 Gen 2（Hexagon NPU V73, NPU 32-bit VA 限制 2 GiB，≥3B 模型无法运行）
  - OnePlus 12：Snapdragon 8 Gen 3（Hexagon NPU V75），performance mode 启用
  - OnePlus Ace5 Pro：Snapdragon 8 Elite（Hexagon NPU V79）
  - CPU-NPU 通信：rpcmem shared memory + FastRPC；仅单向 cache coherence（CPU→NPU）
  - NPU：TCM 8 MiB + L2 1 MiB，DMA ≥60 GB/s DDR read

- 开源Serving框架是什么。修改了什么。
  - 基础框架：llama.cpp（https://github.com/ggml-org/llama.cpp），使用 rpcmem shared memory 作为底层 buffer 类型
  - 对比 baseline：llama.cpp OpenCL backend（commit 1caae7f，Adreno GPU，含 Q4_0 优化 kernel）。其他 NPU baseline 未作为主要对比：llm.npu 在 decoding 阶段不使用 NPU，QNN-based 系统精度过低（PowerServe per-channel W4A16 在 MATH500 仅 2.1%），PowerInfer-2 和 HeteroLLM 未开源
  - 论文修改/新增：
    (1) **新增 Hexagon NPU backend**（约 7K 行 C/C++ + inline assembly）——集成到 llama.cpp 的 backend 框架中，与现有 CPU/GPU backend 并列
    (2) **新增 NPU operator library**（编译为独立 Hexagon DSP shared object）——实现 GEMM、FlashAttention、dequantization、Softmax 等算子；包含 power management、hardware resource management、computation thread pool
    (3) **新增 shared memory 通信层**——rpcmem 分配 shared buffer；FastRPC 启动/管理 remote NPU session；NPU 端 polling-based command dispatch；manual cache maintenance（L2 invalidate/writeback）
    (4) **新增算子级 CPU fallback**——对未在 NPU 实现的算子（lm_head, logits）自动调度到 CPU，通过 shared memory 传递中间结果
    (5) **支持动态 batch size**（QNN 因其固定静态计算图无法支持）——运行时可变 batch，适配 test-time scaling 的 generation budget 调整

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：主仓库 https://github.com/haozixu/llama.cpp-npu（MIT 许可证）；算子库 https://github.com/haozixu/htp-ops-lib。无 QNN 依赖，使用 Hexagon SDK 6.0.0.2 LLVM toolchain。

  - **llama.cpp NPU Backend 输入到硬件执行全过程（以 Qwen2.5-1.5B Best-of-N, N=8, OnePlus 12 为例）**：

    ```
    # ========== 阶段 0：系统初始化（首次加载模型）==========
    输入: Qwen2.5-1.5B Instruct (Q4_0 量化, 约 1.0 GiB), 目标设备 OnePlus 12

    1. CPU 侧 llama.cpp 启动:
       - 加载模型权重文件 (GGUF 格式) → CPU memory
       - 初始化 Hexagon NPU backend:
         a. FastRPC 启动远程 NPU session (加载 DSP shared object)
         b. 分配 rpcmem shared buffers:
            - 权重 buffer: ~1 GiB (Q4_0 量化, 包含 offline permuted tile layout)
            - KV cache buffer: ~context_len × n_layers × 2 × n_kv_heads × head_dim × FP16
            - 激活/中间 buffer: 多块, 按需分配
            - 通信 buffer: command descriptors + flags
         c. NPU 端进入 polling loop
       - 预计算 LUT: 64 KiB exp LUT → TCM (Softmax 加速)
       - 设置 context budget = 4096 tokens

    2. NPU 端初始化:
       - 配置 power management (performance mode)
       - 初始化 hardware resource manager (HVX 线程池, HMX 访问仲裁)
       - 进入 command polling loop:
         while true:
           cache_invalidate(communication_buffer)
           if command_flag == READY:
             parse command descriptor
             execute kernel (GEMM / FlashAttention / ...)
             cache_writeback(output_buffers)
             completion_flag = DONE

    # ========== 阶段 1：Prefill（处理 prompt）==========
    输入: user_prompt = "Solve: If x^2 + y^2 = 25 and xy = 12, find x+y."
          经过 tokenizer → [t1, t2, ..., t_N] (N tokens)
          test-time scaling: Best-of-N, N=8, budget=8

    # Prefill 在 NPU 上以 batch_size=1 执行（单 prompt, 多 token）
    对每个 token 或 chunk，CPU 通过 shared memory 发送 GEMM/Attention 请求:
      CPU: 填充 activation tensor → cache writeback → 写 command descriptor → 设 READY
      NPU: 检测 READY → cache invalidate → 执行 HMX GEMM / FlashAttention
           → 结果写回 shared buffer → cache writeback → 设 DONE
      CPU: 检测 DONE → cache invalidate → 继续下一层

    # 特殊处理:
    # - Attention: FP16 FlashAttention kernel（HMX MatMul + HVX LUT exp）
    # - lm_head (prefill 最后): 回退到 CPU 执行
    #   原因: [1, vocab_size] 虽可放入 NPU，但权重在 CPU memory 中
    #   为保持架构简洁，lm_head 统一在 CPU 执行

    Prefill 完成后 → first token 生成 → 进入 decoding 阶段

    # ========== 阶段 2：Decoding（Test-Time Scaling, Best-of-N, N=8）==========

    # Best-of-N: 生成 N=8 条独立路径，每条自回归至 EOS/max_len
    # 每条路径共享同一 prompt KV cache（prefill 时已计算）

    # Decoding loop（每条路径每 token 一次 forward）:
    for path_id in range(8):  # 可并行（batch），此处以 batch=8 说明
        for step in range(max_new_tokens):

            # === Step A: CPU 侧准备 batch activation ===
            # 从 8 条路径收集当前 token → [8, hidden_dim] activation
            batch_act = collect_current_tokens(paths[0:8])

            # === Step B: CPU → NPU 提交 GEMM 请求 ===
            # 对每层 decoder layer 的每个 MatMul:
            for layer in decoder_layers:
                # Q/K/V projection
                for proj in [Q, K, V]:
                    cpu_fill_activation(batch_act)         # [8, hidden_dim]
                    cpu_cache_writeback(act_buffer)
                    cpu_set_command(GETMM_Q4_0, act_addr, weight_addr, out_addr)
                    cpu_set_flag(READY)
                    npu_poll_and_execute()                  # HVX dequant + HMX MatMul
                    cpu_wait_flag(DONE)
                    cpu_cache_invalidate(out_buffer)

                # Attention (FlashAttention on NPU):
                #   Q [8, n_heads, head_dim], K/V [8+ctx, n_kv_heads, head_dim]
                #   HMX MatMul(QK^T) → HVX LUT exp(Softmax) → HMX MatMul(PV)
                npu_execute_flash_attention(Q, K, V, O)

                # O projection
                npu_execute_gemm(O, W_O)

                # FFN Gate/Up/Down
                npu_execute_gemm(x, W_gate) → SiLU (HVX 近似或 CPU)
                npu_execute_gemm(x, W_up)
                npu_execute_gemm(gate * up, W_down)

            # === Step C: lm_head 回退到 CPU ===
            # hidden_state [8, hidden_dim] → CPU MatMul → logits [8, vocab_size]
            # vocab_size ~152K for Qwen2.5 → logits 过大无法放 NPU
            cpu_execute_lm_head(hidden_state, W_lm_head) → logits

            # === Step D: Token selection (CPU) ===
            for path_id in range(8):
                next_token = sample(logits[path_id])
                paths[path_id].append(next_token)
                # KV cache 更新（写入 shared buffer, NPU 后续可见）

        # === Step E: Best-of-N selection (CPU) ===
        # 每条路径完成 → 使用 Skywork-1.5B-PRM 评分
        scores = [PRM.score(path) for path in paths]
        best_path = paths[argmax(scores)]
        output = best_path.generated_text

    # ========== 性能特征 ==========
    # batch=1 (conventional decoding):
    #   HMX tile 利用率: activation [1→32 padded, 32] × weight [32, 32]
    #   → 1/32 有效行（~3% utilization）
    #   → NPU 算力大量浪费
    #
    # batch=8 (Best-of-N, N=8):
    #   HMX tile 利用率: activation [8→32 padded, 32] × weight [32, 32]
    #   → 8/32 有效行（25% utilization）
    #   → NPU 算力更充分利用
    #   → decoding throughput 显著提升（Fig 11）
    #   → 理论: NPU GEMM 延迟与 batch=1 相同（tile padding 后）
    #     实际: 非 NPU 部分（lm_head CPU, memory）随 batch 增长
    #     batch=16 时 CPU logits 占比 ≥50%

    # ========== 与 GPU baseline 对比 ==========
    # GPU (llama.cpp OpenCL):
    #   batch=1: GPU 更快（decoding 低延迟, GEMV 优化好）
    #   batch≥4: NPU 更快且扩展性更好（HMX tile 利用率提升）
    #   → NPU 适合 test-time scaling（大 batch decoding）
    #   → GPU 适合传统单路径 decoding

    # ========== 与 QNN 对比 ==========
    # QNN FP16:
    #   - 仅支持固定静态计算图 → batch size 不可变
    #   - 仅支持 per-tensor/per-channel 量化 → 精度严重损失
    #     (MATH500: QNN W4A16 2.1% vs. AutoAWQ 15.9%)
    #   - Prefill 性能可比较（某些 workload 下论文方法达到 QNN 水平）
    ```

  - **关键性能数据**：
    - Decoding throughput (Qwen2.5-1.5B, OnePlus 12): batch=1 → ~5 tok/s, batch=8 → ~22 tok/s, batch=16 → ~28 tok/s
    - Prefill throughput (Qwen2.5-1.5B, OnePlus 12, seq_len=256): 论文方法 ~130 tok/s, GPU OpenCL ~90 tok/s, QNN FP16 ~150 tok/s
    - 功耗 (OnePlus 12, 1.5B): batch 增大时功耗有所增加但总体 <5W；3B model 稳定在 ~4.3W
    - 能耗 (归一化): 1.5B batch=8 解码能耗 < 3B batch=1 解码能耗，同时 test-time scaling 精度可比
    - CPU memory: 1.5B ~1.3 GiB total (resident + dmabuf), 3B ~2.4 GiB total (context=4096)
    - CPU utilization: batch=16 时 4 核利用率最高（因 lm_head CPU 计算增加）

