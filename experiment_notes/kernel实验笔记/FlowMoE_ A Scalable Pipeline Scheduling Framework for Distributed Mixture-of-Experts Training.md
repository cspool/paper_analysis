## FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FlowMoE 包含三个 kernel 调度/运行时计算层面的实现：
    1. **Unified Pipeline Scheduling（统一流水线调度）**：将 Transformer block 内的全部任务——MHA 计算、gating 路由、expert 计算、all-to-all (A2A) 通信——统一纳入流水线调度。将每个 Transformer block 的输入 tensor 按 **R 个等分** 切分，除 all-reduce 外的所有计算和通信任务均拆分为 R 个独立子任务，通过统一的 feed-forward 和 backward 顺序编排，使 MHA、gating、expert 和 A2A 在同一条流水线上交错执行。
    2. **Tensor Chunk-Based Priority Scheduling（张量分块优先级调度）**：在反向传播期间，将每层 all-reduce 的梯度张量切成大小为 **S_p** 的 chunk，放入通信任务池。A2A 任务具有最高优先级，all-reduce chunk 仅在没有 A2A 任务 pending 时立即执行，填充通信间隙，最大化计算-通信重叠。
    3. **Bayesian Optimization Auto-Tuning（贝叶斯优化自动调参）**：轻量级 BO 自动搜索最优 all-reduce chunk 大小 S_p。仅需约 8 次采样即可收敛到近优值（如 BERT-Large-MoE 上 ~2.5MB），开销 < 1% 迭代时间。硬件环境变化时重新执行。
  - 实验比较：FlowMoE vs vanillaEP、FasterMoE、Tutel、FSMoE、ScheMoE 在 4 个真实 MoE 模型上的 per-iteration training time、energy consumption、memory usage，以及 675 个自定义 MoE 层配置上的加速比。消融实验对比 Pipe-MoE (仅 MoE 层流水线)、Pipe-AT (加入 MHA+gating)、Pipe-AR (加入 all-reduce, w/ w/o BO) 的逐模块贡献。

- 后端平台是什么，配置是什么。
  - Cluster 1: 2 节点 × 8 × NVIDIA RTX 3090 (24GB)，共 16 GPU，100Gb/s 跨节点网络，Intel Xeon Gold 6248R CPU。
  - Cluster 2: 4 节点 × 2 × NVIDIA RTX 2080Ti (11-12GB)，共 8 GPU，10Gb/s 跨节点网络，Intel Xeon Gold 5118 CPU。
  - 所有参数和梯度使用 32-bit 单精度浮点。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch** 构建，利用 **Tutel** 优化通信（Tutel 是集成到 PyTorch 中的 MoE 加速库，支持通信和计算任务的异步执行，被 DeepSpeed 作为默认 MoE 训练模块）。
  - 修改内容：
    - **新增三个队列**：DataQueue（任务间数据传递）、A2AQueue（all-to-all 通信任务）、ARQueue（all-reduce 通信任务）。
    - **新增通信池管理器**：在后台线程运行，优先级逻辑确保 A2A 任务优先执行，all-reduce chunk 填充间隙。
    - **Tensor 分区**：将 all-reduce 张量按 S_p 切块。
    - **贝叶斯优化集成**：自动调优 S_p。
  - 性能指标测量：per-iteration training time（平均 1000 次迭代）、energy consumption（NVIDIA SMI 每 5ms 采样，时域积分）、memory usage（NVIDIA SMI 每 1s 监控）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：代码开源在 https://github.com/ZJU-CNLAB/FlowMoE
  - **FlowMoE 统一流水线调度执行原理全过程**：
    ```
    ┌── Kernel/Scheduling Input ──────────────────────────────────────┐
    │ 每个 Transformer block 的输入 tensor: x^(l) [B, N, H]           │
    │ R: 流水线度（pipelining degree），通常 R=2                        │
    │ S_p: all-reduce 切块大小（BO 自动调优）                          │
    │                                                                  │
    │ 前向任务集（每层 l）:                                            │
    │   AT_r^(l): MHA + gating 计算子任务                              │
    │   D_r^(l):  Dispatch A2A 子任务                                  │
    │   E_r^(l):  Expert 计算子任务                                    │
    │   C_r^(l):  Combine A2A 子任务                                   │
    │                                                                  │
    │ 反向任务集（每层 l）:                                            │
    │   AT_r'^(l): MHA + gating 反向子任务                             │
    │   D_r'^(l):  Dispatch A2A 反向子任务                             │
    │   E_r'^(l):  Expert 反向子任务                                   │
    │   C_r'^(l):  Combine A2A 反向子任务                              │
    │   AR^(l):    All-reduce 梯度子任务（切成 chunk）                  │
    └──────────────────────────────────────────────────────────────────┘

    ┌── 前向调度 ───────────────────────────────────────────────────┐
    │ 顺序: AT_1→AT_2→...→AT_R→E_1→E_2→...→E_R→AT_1^(l+1)→...       │
    │       (计算任务按层流动)                                         │
    │                                                                    │
    │        D_1→...→D_R→C_1→...→C_R→D_1^(l+1)→...                     │
    │       (A2A 通信任务按层流动，与计算交错)                          │
    └──────────────────────────────────────────────────────────────────┘

    ┌── 反向调度（核心创新）─────────────────────────────────────────┐
    │ 计算任务:                                                         │
    │   E_R^(l+1)→...→AT_1^(l+1)→E_R^(l)→...→E_1^(l)→AT_R^(l)→...→AT_1^(l)
    │                                                                    │
    │ A2A 任务:                                                         │
    │   C_R^(l+1)→...→D_1^(l+1)→C_R^(l)→...→C_1^(l)→D_R^(l)→...→D_1^(l)
    │                                                                    │
    │ All-Reduce Chunk 插入（Theorem 1）:                              │
    │   在 A2A 任务的间隙中插入 all-reduce chunk:                       │
    │   C_j^(l) → [AR_chunk if A2A idle] → D_j^(l) → ...               │
    │   优先级: A2A tasks > all-reduce chunks                           │
    │   更小的 S_p → 更细粒度的 gap filling → 更优（Theorem 2）        │
    └──────────────────────────────────────────────────────────────────┘

    ┌── Baseline vs FlowMoE 调度对比 ───────────────────────────────┐
    │                                                                   │
    │ Tutel/ScheMoE (仅 MoE 层流水线):                                 │
    │ Time →                                                            │
    │ [MHA][Gate][==== A2A Dispatch + Expert + A2A Combine =====]       │
    │                                         [All-Reduce]              │
    │  // MHA 和 All-Reduce 串行，占 30-40% 迭代时间                   │
    │                                                                   │
    │ FlowMoE (全 block 流水线):                                        │
    │ Time →                                                            │
    │ [AT_1][AT_2][E_1][E_2][AT_1^(l+1)][E_1^(l+1)]...          (前向) │
    │ [D_1][D_2][C_1][C_2][D_1^(l+1)][C_1^(l+1)]...             (A2A)  │
    │         [AR_chunk_1]      [AR_chunk_2]    [AR_chunk_3]    (AR)   │
    │  // MHA、A2A、AR 全重叠，消除 30-40% 串行开销                    │
    └──────────────────────────────────────────────────────────────────┘
    ```

  - **Bayesian Optimization 调参原理**：
    1. **目标函数**：f(S_p) = per-iteration training time（平均 10 次迭代）
    2. **采样**：随机初始化若干 (S_p, time) 对
    3. **GP 模型**：高斯过程拟合 f(S_p) 的 posterior distribution
    4. **采集函数**：Expected Improvement (EI) 选择下一个采样点
    5. **终止**：约 8 次采样后收敛，BO 开销 < 1% 迭代时间
    6. **输出**：最优 S_p（如 BERT-Large-MoE 上 ~2.5MB）

  - **评估数据集与模型**：
    - GPT2-Tiny-MoE (M=256, H=512, L=12, E=P, k=2, OpenWebText)
    - BERT-Large-MoE (M=512, H=1024, L=24, E/P=2, k=1, wikitext-103)
    - LLaMA2-MoE (M=1024, H=4096, L=32, E=P, k=1, wikitext-103)
    - LLaMA2-MoE-L (M=1024, H=4096, L=64, E=P, k=1, wikitext-103)
    - DeepSeek-V2-S (M=5120, H=1536, L=4, E/P=2, k=8, OpenWebText)
    - DeepSeek-V2-M (M=5120, H=1536, L=7, E/P=2, k=1, OpenWebText)
    - 675 个自定义 MoE 层配置：B∈{2,4,8}, f∈{1.0,1.1,1.2}, N∈{512,1024,2048}, M∈{512..8192}, H∈{512..8192}

  - **关键性能数据**（Cluster 1, 16 × RTX 3090）：
    | Model | vanillaEP | FasterMoE | Tutel | FSMoE | ScheMoE | FlowMoE |
    |-------|-----------|-----------|-------|-------|--------|---------|
    | GPT2-Tiny-MoE | 169.5ms | 135.3ms | 129.3ms | 114.8ms | 116.4ms | **95.6ms** |
    | BERT-Large-MoE | 537.8ms | 490.8ms | 501.1ms | 421.9ms | 405.6ms | **351.9ms** |
    | LLaMA2-MoE | 1987.7ms | 1759.1ms | 1534.1ms | 1292.6ms | 1374.3ms | **1124.0ms** |
    | DeepSeek-V2-S | 5843.3ms | 4562.5ms | 4481.4ms | 3895.6ms | 4093.7ms | **3205.3ms** |
    
  - **消融实验**（M=8192, H=8192, 16 GPU）：
    | Configuration | Pipe-MoE | Pipe-AT | Pipe-AR | Time | Speedup vs vanillaEP |
    |--------------|----------|---------|---------|------|---------------------|
    | vanillaEP | ✗ | ✗ | ✗ | 1630.8ms | 1.00× |
    | Tutel | ✓ | ✗ | ✗ | 1115.2ms | 1.46× |
    | FlowMoE-AT | ✓ | ✓ | ✗ | 1012.6ms | 1.61× |
    | FlowMoE-AR | ✓ | ✗ | ✓ (w/o BO) | 971.5ms | 1.68× |
    | FlowMoE-AR(BO) | ✓ | ✗ | ✓ (w/ BO) | 895.3ms | 1.82× |
    | FlowMoE | ✓ | ✓ | ✓ | **796.1ms** | **2.05×** |
