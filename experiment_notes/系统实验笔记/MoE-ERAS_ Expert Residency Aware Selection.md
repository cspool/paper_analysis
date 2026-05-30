## MoE-ERAS: Expert Residency Aware Selection

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoE-ERAS 在 MoE 推理 serving 场景中，修改 gating network 的 expert 选择策略，使其根据 expert 当前是否驻留在 GPU HBM 中来调整选择决策。核心修改在 serving 框架的 expert dispatch 阶段：
    1. 维护每个 MoE layer 的 expert residency 状态表（记录 expert 在 HBM 还是 CPU DRAM）。
    2. 在每层 gating 输出后、Top-K 选择前，插入 thresholding 或 biasing 操作调整 logits/weights，使路由器倾向选择已驻留的 expert。
    3. 当路由器选择 on-chip expert 而非 off-chip expert 时，避免了一次 CPU→GPU 的 expert 参数传输（在 memory-bound 解码阶段节省显著延迟）。
  - 实验比较：(1) Baseline（Top-K routing + quantization + LRU caching from dvmazur/mixtral-offloading）vs Thresholding（α=0.05, 0.15, 0.25）vs Biasing（β=1）；(2) 不同 offload per layer 设置（offload 1-7 experts per layer）下的 relative speedup；(3) Sequential decoding 的 wall clock latency 和 throughput 对比（100 token sequences, 50 iterations）。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA H100（图 2 展示 H100 HBM vs CPU DRAM 的 expert read time 对比：CPU 读取时间比 GPU 高数个数量级）
  - 主机内存：CPU DRAM 用于 offload 未常驻的 expert 参数
  - 配置文件：baseline 框架可在 Tesla T4 16GB 上运行 Mixtral-8x7B（通过 quantization + LRU caching + expert offloading）

- 开源Serving框架是什么。修改了什么。
  - 基座框架：`dvmazur/mixtral-offloading`（https://github.com/dvmazur/mixtral-offloading），该框架提供 expert 量化、LRU caching 和 expert offloading 功能。
  - 修改内容：
    1. 在 gating network 输出与 Top-K selection 之间插入 residency-aware routing 逻辑（thresholding 或 biasing）。
    2. 维护 expert residency table：跟踪每个 layer 的 expert 当前在 HBM 还是 DRAM。
    3. Profiling 模块：在 inference 前收集 expert activation frequency（用于 biasing 的 freq 参数）。
  - 服务流程：请求进入 → self-attention 计算 → gating network 输出 logits → residency-aware routing (thresholding/biasing) 调整 logits/weights → Top-K 选择 expert → 若选中 off-chip expert 则触发 CPU→GPU 传输 → expert MLP 计算 → 输出 token → 更新 LRU cache 和 residency table。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：MoE-ERAS 自身代码未开源。Baseline `dvmazur/mixtral-offloading` 开源。
  - 框架执行全过程（以 thresholding α=0.15, offload=3 experts/layer 为例）：
    1. **输入**：用户 prompt token 序列，模型权重（expert 参数部分在 HBM，部分在 CPU DRAM）。
    2. **Prefill 阶段**：prompt tokens 经 embedding → attention layers → 到达 MoE layer i。Gating network 计算 Logits = H_i @ W_exp。Residency-aware routing：Weights = Softmax(Logits)，对 HBM 中的 5 个 expert 加 α=0.15。SelectTopK(Weights, k=2) → 若选择的 2 个 expert 均在 HBM 中，零传输开销；若选择 off-chip expert，CPU→GPU PCIe 传输 ~数百 MB 的 expert 权重（图 2：CPU 读取延迟 >> GPU 读取延迟）。Expert MLP 计算输出。LRU cache 更新 residency 状态。
    3. **Decode 阶段**：逐 token 生成，每步经过所有层。当 α=0.15 且 3 experts offloaded 时，约减少 10-13% 的解码延迟；offload 越多、α 越大，节省越显著（最大 21.2% reduction）。
    4. **输出**：生成的 token 序列 + 各层 expert 激活记录（用于 biasing 方法的 freq 更新）。
