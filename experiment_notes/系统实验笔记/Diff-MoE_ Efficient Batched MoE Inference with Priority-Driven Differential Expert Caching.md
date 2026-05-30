## Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Diff-MoE 是一个面向 MoE 稀疏 LLM 的高吞吐 batched inference 框架，基于 NVIDIA FasterTransformer 库构建。核心实现包括三部分：(1) 优先级管理器（Priority Manager），在离线微调阶段识别 globally hot experts（top-N per layer，分配 MaxP 优先级并永久固定），在线推理阶段动态调整 non-global experts 的优先级分数（激活 +Δinc，不活跃缓存 -Δdec_in，不活跃未缓存 -Δdec_out）；(2) 差分缓存层级（Differential Cache Hierarchy），将 GPU 显存组织为 per-layer high-priority cache (HPCi, 永久存储 globally hot experts)、per-layer medium-priority cache (MPCi, 动态管理 locally hot experts，容量为 HPCi 的 2 倍)、以及跨层共享的 low-priority cache (LPC, 临时存储 cold experts 和预取 experts，用后即清)；(3) 轻量级 GRU 预测器（6 层 GRU，在微调阶段收集的 expert 激活序列上训练），预测下一 MoE 层可能激活的 experts 并预取到 LPC 中。
  - 实验比较：Diff-MoE 与 3 种 SOTA offloading 方案对比：(a) DeepSpeed-Offload — 按需加载激活 experts，计算后立即驱逐；(b) Pre-gated MoE — 预取类方法，修改 gating 机制提前预取下一层所有 activated experts；(c) MoE-Infinity — 缓存类方法，全局共享缓存 + 基于估计重用概率的驱逐策略。比较指标：Cache Hit Rate、End-to-End Throughput (tokens/s)、Peak GPU Memory (GB)、Memory Efficiency (tokens/(GB·s))。Diff-MoE 平均吞吐提升 2.74× (vs DeepSpeed)、2.22× (vs Pre-gated MoE)、1.55× (vs MoE-Infinity)。

- 硬件平台是什么，配置是什么。
  - GPU：单卡 NVIDIA H200（141 GB HBM）。
  - CPU：2 × Intel Xeon Gold 6430。
  - Host DRAM：1 TB。
  - 互联：PCIe 5.0，双向带宽 128 GB/s。
  - 操作系统：Ubuntu 22.04。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：NVIDIA FasterTransformer v5.2（https://github.com/NVIDIA/FasterTransformer）。
  - 论文开源代码：https://github.com/ceciliawinter/Diff-MoE.git（DOI: 10.5281/zenodo.15879848）。
  - FasterTransformer 修改内容：
    1. **Expert 参数粒度拆分**：将原始 HuggingFace 格式模型的 bin 文件拆分为细粒度 expert 参数文件，使每个 expert 可独立从 host 加载到 GPU。
    2. **差分缓存层级注入**：在 FasterTransformer 的 MoE 层执行路径中注入 HPC/MPC/LPC 三级缓存管理逻辑。HPC 在推理启动前预先加载 globally hot experts，推理过程中不驱逐；MPC 按优先级驱动替换策略动态管理；LPC 作为临时缓冲，当前层计算完成后清空。
    3. **Gating 后拦截**：在 gating network 输出 top-K experts 后，拦截 expert 加载流程——先检查 HPCi ∪ MPCi ∪ LPC 是否已缓存目标 expert，仅对缺失的 expert 触发 host→GPU 数据传输。
    4. **优先级更新钩子**：在每层 MoE 计算结束后，按公式 (1) 更新所有 non-global experts 的优先级分数，触发 locality-preserving cache replacement。
    5. **预取流水线**：在当前层 expert 加载完成后，利用 GRU predictor 预测下一层所需 experts，异步预取未缓存的 top-1/2 experts 到 LPC，与当前层计算并行。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：代码已开源在 https://github.com/ceciliawinter/Diff-MoE.git，同时提供完整的 README 复现指南。
  - 框架输入到硬件执行全过程（以 Switch-Base, XSum, batch_size=64 为例）：

    **阶段 0 — 离线准备**：
    1. 下载 HuggingFace 格式的 pretrained Switch-Base 模型（7B, 12 layers, 6 MoE layers × 128 experts）。
    2. 使用提供的脚本将模型 bin 文件拆分为 per-expert 细粒度参数文件（T1）。
    3. 在各下游任务数据集（XSum/SQuAD/CoQA）上微调模型，同时记录每个 MoE layer 的 expert 激活频率，按激活频率降序选出 top-2 globally hot experts per layer（默认 α=5% 缓存比，HPC=2, MPC=4）（T2）。
    4. 用微调阶段收集的 expert 激活序列训练 6 层 GRU predictor（训练/验证 8:2 划分）（T3）。

    **阶段 1 — 推理初始化**：
    5. 将非 MoE 参数（attention weights, embeddings, layernorm 等）常驻 GPU memory。
    6. 将每个 MoE layer 的 globally hot experts（HPCi，各层 2 个，共 6×2=12 experts）从 host 加载到 GPU HPC，永久锁定。
    7. 将所有其他 expert 参数保留在 host DRAM 中，初始化优先级分数为 0。

    **阶段 2 — 在线推理循环（以第 i 个 MoE layer 为例）**：
    8. **Token Embedding 输入**：batch_size=64 的 token embeddings X ∈ ℝ^(64×S×H) 经过前置 attention layer 后进入 MoE layer i。
    9. **Gating**：FasterTransformer 执行 `Softmax(LinearGate(X))` → 得到 gating 权重 G，`TopK(G, k=1)` → 每 token 选出 1 个 activated expert，集合为 A。batch=64 时大约 30-34 个不同 experts 被激活。
    10. **Expert 查找**：遍历 A 中每个 expert E_k^i，检查是否在 HPCi ∪ MPCi ∪ LPC 中。若缺失（cache miss），触发 cudaMemcpy 从 host DRAM 经 PCIe 5.0 加载 expert 参数到 GPU LPC 的激活缓冲区。
    11. **并行计算**：Token-to-Expert Dispatch——将 batch 中每个 token 分配到其 gating 选择的 expert。各 expert 在 GPU 上并行执行 FFN 计算。结果按 token 聚合为输出 Y。
    12. **优先级更新**：按公式 (1) 更新所有 non-global experts 的优先级：p_k^i = clip(p_k^i + 1) for E_k^i ∈ A；p_k^i = clip(p_k^i - 0.4) for inactive cached；p_k^i = clip(p_k^i - 0.2) for inactive uncached。阈值 threshold_hot = 1。
    13. **LPC → MPC 晋升**：locality-preserving replacement —— 若当前 activated 但未在 MPCi 的 expert 优先级 ≥ threshold_hot（即刚激活一次），候选按优先级降序排列；MPCi 中优先级 < threshold_hot 的 resident 按优先级升序排列；用最高优先级候选替换最低优先级 resident，直到无符合条件者。所有冷 experts 从 LPC 驱逐。
    14. **下一层预测与预取**：GRU predictor 接收当前层 A 中的 expert IDs，通过隐藏状态建模历史激活模式，输出下一层 i' 的各 expert 概率分布。batch_size>4 时聚合各样本分布取 top-2 未缓存 expert，异步 cudaMemcpy 预取到 LPC 预取缓冲区。

    **阶段 3 — 测量**：
    15. 所有性能指标（throughput, memory, cache hit rate）从 FasterTransformer 内嵌的日志钩子输出。每个配置重复 3 次取平均。

    Diff-MoE 的核心作用：通过分级缓存 + 优先级策略减少 expert migration 次数（提升 cache hit rate），通过 GRU predictor 将 host→GPU 传输与 GPU 计算重叠（隐藏通信延迟），在 batched MoE 推理中克服 PCIe 带宽瓶颈。
