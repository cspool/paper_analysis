## SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：SwapMoE 在 Serving 调度层面实现了完整的内存受限 MoE 推理框架，核心调度组件包括：(1) **运行时 Virtual Experts 管理**——维护动态更新的 Virtual Experts 子集，运行时根据 importance score 选择最重要的 experts 驻留在主存，不重要的 experts 按需从外部存储（CPU memory 或 SSD）加载/卸载。(2) **Amortized Expert Loading**——不在每个 sample 后同步更新所有 experts，而是跨多个样本摊销 expert 加载开销，异步进行 expert 置换，避免 IO 阻塞计算。(3) **Profiling-guided Memory Planning**——离线阶段对每个 expert 进行细粒度 profiling（memory footprint、inference latency、loading time、IO bandwidth），建立 config→performance 映射模型（E_accuracy, E_memory, E_latency），使用 Genetic Algorithm 搜索最优配置（包括 expert update frequency 和每层 Virtual Experts 数量 #experts_l）。(4) **Expert I/O Frequency 优化**——从低频率向高频率递增测试，找到 expert 更新不影响推理延迟的拐点频率。(5) **Layer Space Allocation**——利用遗传算法将有限 memory budget 分配到不同 MoE 层，更重要的层（如中间层）获得更多 Virtual Experts。
  - 实验比较：(a) **Overall Runtime Performance**：SwapMoE vs Pruning vs On-demand loading vs Original MoE，在不同 memory budget 下的 end-to-end latency 和 accuracy（ROUGE-2 / Perplexity）；(b) **Offline Planning Performance**：遗传算法找到的配置 vs 实际运行时 memory/latency/accuracy vs 给定 constraints；(c) **Robustness Analysis**：不同 expert 数量（16/32/64）的 Switch Transformer 的资源-性能 tradeoff；(d) **Ablation Study**：Simple scheduling (token counting) vs Simple planning (均匀分配) vs Full SwapMoE；(e) **Overhead Analysis**：峰值/平均 IO overhead、external memory consumption。

- 硬件平台是什么，配置是什么。
  - 设备：Jetson Nano（最大 GPU 内存 4GB）和 Jetson AGX ORIN
  - batch size = 1，模拟边缘设备连续 serving 场景（如个人 assistant 逐 token 生成）
  - external memory hierarchy：GPU main memory → CPU memory → SSD（PCIe 和 CPU-SSD 两种 IO 路径）
  - IO bandwidth reference：GPU-CPU PCIe 10-30 GiB/s，CPU-SSD 300-600 MiB/s

- 开源Serving框架是什么。修改了什么。
  - 框架：HuggingFace Transformers (Wolf et al., 2019)
  - 修改内容：(1) 在 MoE layer 的 forward 中插入 Masked Gating 逻辑——原始 router 输出后乘以 Virtual Expert mask，renormalize，将推理重定向到 VE subset；(2) 加入 Runtime Scheduler——管理 expert importance score 计算、Virtual Expert 更新（amortized + async loading/unloading）、IO 与计算协调；(3) 加入 Offline Memory Planner——基于 genetic algorithm 搜索最优 layer-wise expert 分配方案；(4) 加入 Fine-grained Profiler——profile 每个 expert 的 memory/latency/loading time，训练 E_accuracy 小 DNN 模型。
  - 关键修改点：SwapMoE 的核心调度逻辑可概括为——在 HuggingFace MoE layer 的 router 和 expert FFN 之间插入 Virtual Expert selection 和 update 逻辑，将原本的 full expert set 替换为动态维护的 subset，通过 coordinated I/O 和 computation scheduling 最小化 memory 和 latency。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供开源链接，实现基于 HuggingFace Transformers
  - 框架输入→硬件执行全过程（边缘设备连续 serving，batch_size=1）：
    ```
    输入：token sequence X = [x_1, ..., x_T]，memory budget LIMIT_memory
    
    离线阶段（一次性）：
    1. 对目标设备进行 Fine-grained Profiling：
       - 对每个 MoE layer 的一个 expert 进行 inference profiling
         (memory footprint, computation latency, parameter loading time via I/O)
       - profile I/O bandwidth (GPU↔CPU PCIe, CPU↔SSD)
    2. 收集 profiling dataset（少量 labeled samples from deployment scenario）
    3. 训练 E_accuracy(config) = small_DNN(config)，minimize Σ(actual - predicted)²
    4. 运行 Genetic Algorithm 搜索最优 config：
       - Random init population of configurations
       - For each iteration:
           mutate: randomly change one parameter in each config
           crossover: exchange or average existing configs
           evaluate: E_accuracy, E_memory, E_latency
           selection: remove configs violating constraints or suboptimal
       - Output: config* = {frequency*, #experts_1*, ..., #experts_L*}
    
    在线阶段（每个 sample）：
    1. Token 输入：X 进入 Transformer decoder layer
    2. Self-Attention：Q/K/V projection → FlashAttention → output（正常执行）
    3. MoE Layer（SwapMoE 调度路径）：
       a) Router：gating_scores = softmax(router(X))
       b) Masked Gating：
          mask = [1 if i∈VE else 0 for i in 1..num_experts]
          masked_scores = gating_scores * mask
          masked_scores = masked_scores / sum(masked_scores)
       c) Expert 计算：
          仅对 i ∈ VE 执行 E_i(X)，跳过非 VE experts
          y = Σ_{i∈VE} masked_scores[i] * E_i(X)
       d) Importance Score 收集：
          对每个 expert 计算 importance(E_i, X) = Σ_{x∈X_i} ||x|| * |G(x)_i| * ||E_i||
       e) Virtual Expert Update（每 frequency 个 samples 触发）：
          排序 experts by importance → 选择 top-k 为 VE_new
          VE_to_load = VE_new - VE_old → async I/O load from external memory
          VE_to_evict = VE_old - VE_new → release from main memory
    4. LM Head：hidden state → token logits → sample/argmax
    5. 输出 token y_t，作为下一轮 autoregressive 输入
    
    硬件执行路径（Jetson AGX ORIN）：
    - 当前 VE 的 expert 参数已在 GPU main memory → 直接参与 GEMM 计算
    - 需要加载的 expert 参数通过 async copy engine 从 CPU memory/SSD → GPU memory
    - 不重要的 expert 参数从 GPU memory 释放
    - IO overhead: peak ~40 MiB/s, mean ~20 MiB/s（远低于 PCIe/SSD bandwidth）
    ```
