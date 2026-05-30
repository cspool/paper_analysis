## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 D2MoE 的在线执行引擎，包含三个核心 Serving 调度组件：(1) **Bit-Width-Aware I/O-Compute Pipeline**：将传统"按 expert ID 顺序执行 I/O+Compute"改为按 fine-grained bit-width 级重新排序（Figure 9d），利用 MWQ 嵌套结构使低 bit-width 权重可被多个请求复用，减少 I/O 等待时间。(2) **Hottest-Expert-Bit-First (HEBF) 调度算法**：为每个 expert 构建按 bit-width 升序排列的队列 Q_i，每轮从各队列头部弹出元素并选择激活频率最高的（hottest）优先进入 I/O 队列；加载完立即开始计算，使高激活频率 expert 的长计算时间与后续 expert 加载重叠，最小化 I/O-compute bubble。满足三项约束：(6a) 计算在加载完成后开始；(6b) 同 expert 按 bit-width 升序加载以最大化低 bit-width 复用；(6c) 等待时间 = 当前计算完成时间 - 上一计算完成时间 - 当前计算量。(3) **Memory Budget Scheduler (Algorithm 2)**：引入可配置参数 M（GPU 分配给 expert 的内存上限），每层检查当前所需内存是否超出预算 — 若超出则逐层释放高 bit-width 权重（lines 4-6）；若仍不足则释放低 bit-width 权重（lines 7-8）；满足预算后执行 bit-width-aware pipeline 并更新预算。

  实验比较：(a) D2MoE vs EdgeMoE (基于重要性的固定 bit-width + 预加载预测)、Hold-in-Memory-AWQ (INT4 全量 GPU)、MoQE-DynaIO (统一 bit-width + on-demand 加载) 在不同 memory budget M (200MB~2500MB) 下的吞吐量 (tokens/s)，覆盖 LLaMA-MoE-3.5B 和 Mixtral 8×7B 在 Environment 1 和 2 上的结果；(b) D2MoE 扩展到 dense LLM (LLaMA2-13B) 对比固定 INT4 loading 方法的吞吐和峰值内存；(c) 消融实验（Figure 14）：+Router → +MWQ → +HEBF → +Budget 四步累积的吞吐提升分解；(d) 系统开销分析：bit-width router 开销（<0.5% 计算/内存，~1.5% 延迟）、dequantization 开销（4 requests 时 ~20.77% / 32 requests 时 ~5.3%）、parallelism planning 开销。

- 硬件平台是什么，配置是什么。
  Environment 1: NVIDIA RTX 3060 (6GB GPU) + Intel Core i7-11800H (32GB CPU) + Samsung 970 EVO (1TB NVMe, 3.5 GB/s read)。Environment 2: NVIDIA Jetson AGX Orin 64GB (SoC, shared memory) + ARM Cortex-A78AE + Samsung 970 EVO (1TB)。离线预处理: GPU server with 2× NVIDIA RTX A6000。单卡推理，非分布式，无 NVLink/InfiniBand。

- 开源Serving框架是什么。修改了什么。
  D2MoE 是**自研的端侧 MoE Serving 引擎**（~2,500 LOC Python + CUDA），并非基于现有开源 Serving 框架修改。基于 PyTorch，使用 Triton 进行 I/O-compute 并行编程，CUDA kernel 基于 NVIDIA Ampere/Ada Lovelace 架构。论文提到其设计可适配 TensorRT 和 vLLM。

  D2MoE 的 Serving 架构分为两阶段：
  
  **Offline 预处理阶段**（部署前执行一次）：
  - Token-adaptive bit-width router 微调：使用 C4 通用数据集微调每层每个 expert 的 bit-width router
  - MWQ 量化：使用 calibration 数据集（128 random 2048-token segments）对所有 expert 权重执行 MWQ 量化
  - Offline Profiling：测量各 bit-width 的 I/O 延迟 T_io(b_k) 和计算延迟 T_comp(b_k)（data-independent，一次测量可复用）

  **Online 执行阶段**修改：
  - **Bit-Width-Aware I/O-Compute Pipeline**：将传统的"按 expert ID 顺序 I/O → 按 expert ID 顺序 Compute"替换为 fine-grained bit-width 级重排序 + HEBF 调度
  - **Memory Budget Scheduler**：替换无内存限制的专家加载策略，增加动态内存预算检查与释放逻辑
  - **Dequantization Kernel 集成**：在每个 expert 计算前插入 MWQ 反量化步骤

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供公开开源代码仓库。基于论文描述的 D2MoE 引擎架构。

  **D2MoE Serving 框架输入到硬件执行全过程（以 LLaMA-MoE-3.5B, Environment 1, M=1600MB, 32 requests 为例）**：

  ```
  === 系统初始化 ===
  1. 离线预处理完成: bit-width routers 已微调，expert 权重已 MWQ 量化 (b_1=2, b_K=4)
  2. 加载非 expert 参数到 GPU（Attention, LayerNorm, Embedding, Router 等常驻）
  3. Offline profiling 完成: 记录 T_io(INT2), T_io(INT3), T_io(INT4), T_comp(INT2/3/4)
  4. Memory budget M=1600MB 配置完成

  === 逐 Token 推理流程（per MoE layer） ===

  Step 1: Attention (常驻 GPU，无加载开销)
    h = FlashAttention(x)

  Step 2: Dual Routing (GPU compute)
    # Router 1: Original MoE expert routing
    gate_logits = W_gate @ h                    // [E=8] per token
    selected_experts = TopK(Softmax(gate_logits), K=2)  // 选 top-2 experts

    # Router 2: Bit-width routing (per selected expert)
    For each selected expert e_i:
      bw_logits = W_bw_router[e_i] @ h         // [K=3] bit-width logits (INT2/3/4)
      b_k = argmax(Softmax(bw_logits))          // 选择最优 bit-width
    # 输出: [(expert_id, bit_width), ...] for all tokens in batch

  Step 3: Memory Budget Check (CPU scheduler)
    For current layer j:
      total_expert_mem = Σ memory(expert_id, bit_width)  // 本层所有被选 experts 总内存
      If total_expert_mem > M:
        # 释放高 bit-width 权重 (Algorithm 2, lines 4-6)
        For k = K-1 down to 0:
          Free(layer[j-1][k])                  // 释放上一层的高 bit-width experts
          Update M
        # 若仍不足，释放低 bit-width 权重 (lines 7-8)
        If layers[j] > M:
          Free(layer[j-1][1])                  // 释放上层的低 bit-width experts
      # 加载本层所需 experts
      Load and Store(layer[j])

  Step 4: HEBF Scheduling (CPU scheduler)
    # 为每个 expert 构建按 bit-width 升序的队列
    For each expert e:
      Q_e = sorted([(bit_width, count) for activations of expert e], key=bit_width asc)

    # HEBF 算法主循环
    While any Q_e is non-empty:
      candidates = []
      For each expert e with non-empty Q_e:
        (bw, count) = Q_e.front()               // 当前最低 bit-width
        candidates.append((e, bw, count))
      
      # 选激活频率最高的 (hottest expert-bit)
      (e*, bw*, count*) = argmax(count) in candidates
      Enqueue (e*, bw*) to I/O Queue             // 优先 I/O 加载
      Pop from Q_{e*}
      
      # I/O 完成后立即提交到 Compute Queue
      # 高频率 expert 的计算时间长 → 可与后续 expert 的 I/O 重叠

  Step 5: Bit-Width-Aware I/O-Compute Pipeline (GPU + I/O)
    磁盘 → GPU (I/O Stream):     | Load E2(INT2) | Load E5(INT2) | Load E2(INT3) | ...
                                  |<- 重叠 ——>|
    GPU Compute Stream:           | idle | Dequant+FFN(E2,INT2) | Dequant+FFN(E5,INT2) | ...
                                            |<- E5 I/O 与 E2 Compute 重叠 ->|
    
    # INT2 权重被所有选 INT2/3/4 的请求共用（MWQ 嵌套）
    # INT3 权重仅被选 INT3/4 的请求额外加载
    # INT4 权重仅被选 INT4 的请求额外加载
    # → 低 bit-width 权重加载次数远多于高 bit-width → HEBF 优先加载

  Step 6: Expert FFN Computation (GPU)
    For each completed I/O load:
      W_fp16 = MWQ_dequant(Q_W_b1, Q_W_b2, ..., s_b1, s_b2, ..., z_b1)
      # Dequantization: CUDA Core 执行 binary shift 操作
      output = W_fp16 @ h                       // Tensor Core GEMM
      output = output * gate_score              // Gating 权重加权

  Step 7: Combine & Continue
    final_output = Σ routed_expert_outputs
    h = h + final_output                        // Residual
    Continue to next layer

  === 关键约束满足 ===
  - (6a) Compute 仅在 I/O 完成后启动: HEBF 保证 L(s+1,j,k) ≤ C(s,j,k)
  - (6b) 同 expert 按 bit-width 升序加载: Q_e 始终升序排列
  - (6c) 等待时间公式: T_wait = C(s,j,k) - C(s-1,j,k) - B_{j,k}·T_comp(k)
    → HEBF 最小化 T_wait，使高频率 expert 计算时间覆盖后续 I/O
  ```

  **关键性能特征**：
  - Mixtral 8×7B 在 Environment 1 (6GB GPU) 上：EdgeMoE 和 Hold-in-Memory-AWQ 失败（显存不足），D2MoE 达到 38.07 tokens/s
  - LLaMA-MoE-3.5B 在 Environment 1: D2MoE 吞吐比 EdgeMoE 高 1.06×-1.16×，内存减少 33%-53%
  - 随 memory budget 增大，D2MoE 吞吐接近 Hold-in-Memory-AWQ（无 I/O 加载开销的理想上界）
  - 32 requests, M=200MB → 66.45 tokens/s; M=1600MB → 83.14 tokens/s (near-linear scaling)
  - Parallelism planning 开销：虽然随 request 数增长，但占总推理时间比例递减
