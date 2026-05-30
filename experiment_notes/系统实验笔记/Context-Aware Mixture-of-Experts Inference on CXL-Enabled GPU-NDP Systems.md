## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Context-Aware Expert Placement Module**，在 GPU-NDP 异构系统上动态决定每层哪些 experts 驻留 GPU HBM（FP16 全精度），哪些 experts 驻留 CXL-NDP 设备（量化低精度）。核心机制：(1) 在 prefill 阶段收集每 expert 的激活频率 $P_{l,e}$ 和累计路由评分 $W_{l,e}$；(2) 计算归一化重要性分数 $S_{l,e} = \alpha\widetilde{P}_{l,e} + (1-\alpha)\widetilde{W}_{l,e}$；(3) 按 $S_{l,e}$ 降序选择每层 top-K experts 迁移至 GPU，其余保留在 NDP；(4) placement 仅执行一次（prefill 后），decoding 阶段不再迁移。将传统 GPU-NDP 系统以"Parameter Movement"为代价的 expert offloading 转化为以 "Activation Movement" 为核心的 NDP 近数据执行。

  实验比较：
  - vs **MoNDE** [18] (context-agnostic GPU-NDP expert offloading)：Ours-3bit 端到端 6.6-8.3× speedup，Ours-2bit 7.9-10.6×
  - vs **HOBBIT** [31] (GPU-only mixed-precision offloading)：Ours-2bit 达 18-19× speedup
  - Decoding throughput：Ours-3bit 8.7×, Ours-2bit 11.2× (Mixtral-8×7B)
  - NDP 侧 latency reduction：Ours-3bit ~5×, Ours-2bit ~8×

- 硬件平台是什么，配置是什么。
  系统：1× H100 GPU (80GB HBM3, 132 SM, 989.4 TFLOP/s) + 1× DDR-based CXL-NDP device (512 GB DDR, 512 GB/s bandwidth, 64×(4×4) systolic arrays @ 1 GHz)。PCIe Gen4 ×16 互联。

- 开源Serving框架是什么。修改了什么。
  论文未基于现有开源 Serving 框架（如 vLLM/SGLang）修改。系统评价使用 Ramulator [19] 模拟 NDP 设备，并自行实现 MoE 推理 pipeline。Baseline MoNDE 为 GPU-NDP MoE 系统，论文与其在同一模拟环境下对比。

  **相对于 MoNDE（context-agnostic）的核心修改**：
  1. **Prefill 统计注入**：在 MoE forward pass 的 prefill 阶段，每层 Gate/Router 计算后额外收集 $(P_{l,e}, W_{l,e})$，通过轻量级累加器实现（metadata 开销可忽略）。
  2. **单次 Expert Placement 调度**：MoNDE 使用 on-demand swapping 或 static placement——experts 在 GPU↔NDP 间动态迁移或固定分配，导致频繁迁移开销和带宽争用。本论文改为 prefill-guided once-per-sequence placement，消除 decoding 期间的 expert migration。
  3. **Hot/Cold 动态识别**：MoNDE 的 hot/cold 分类基于全局历史频率统计，忽略 context dependence。本论文使用 per-sequence prefill 统计做动态识别，捕捉不同输入序列的 expert 激活变化。
  4. **GPU-NDP 计算重叠**：GPU 执行其 hot experts 的 FFN 计算时，NDP 并行执行 cold experts 的量化计算，两者 overlap 最大化 pipeline 效率。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未公开独立代码仓库。NDP 模拟基于 Ramulator [19] (https://github.com/CMU-SAFARI/ramulator)。量化使用 GPTQ [9]。

  **Context-Aware GPU-NDP MoE Serving 全过程（以 Mixtral-8×7B, K=4 GPU/4 NDP experts/layer, Ours-3bit, 一个推理请求为例）**：

  1. **请求到达**：用户 prompt tokens 到达系统 → 非-expert 参数（Attention, Router, LayerNorm, shared params）常驻 GPU HBM。
  
  2. **Prefill 阶段（含统计收集）**：
     - 所有 prompt tokens 经 GPU 执行 Attention (FlashAttention on H100) → Router: logits = Softmax(W_g · x) → TopK(k=2) → 每层 8 experts 中选出 top-2
     - **统计收集**（论文创新点）：每层 l，维护 counter array [E=8]：
       - $P_{l,e}$ += 1（若 expert e 被任何 token 选中）
       - $W_{l,e}$ += routing_score（累计门控输出的 softmax 权重）
     - Expert FFN 计算：prefill 期间所有 experts 仍在 GPU 执行（因 prefill tokens 多，expert 激活均匀）
     - 每层输出传递至下一层，统计累加器一同传递

  3. **Expert Importance 计算**（prefill 结束后，解码前）：
     - 每层归一化：$\widetilde{P}_{l,e} = P_{l,e} / \sum_e P_{l,e}$, $\widetilde{W}_{l,e} = W_{l,e} / \sum_e W_{l,e}$
     - 重要性：$S_{l,e} = 0.5 \times \widetilde{P}_{l,e} + 0.5 \times \widetilde{W}_{l,e}$
     - 每层按 $S_{l,e}$ 降序 → top-4 experts → $\mathcal{H}_l$ (GPU, FP16)
     - 其余 4 experts → $\mathcal{C}_l$ (NDP, 由 Bitwidth Selector 分配 1-4 bit)

  4. **一次性 Expert 迁移**（仅在 prefill 后执行一次）：
     - GPU→NDP：$\mathcal{C}_l$ 中原本在 GPU 的 experts 的量化权重（pre-cached 1/2/3/4-bit GPTQ replicas）通过 PCIe → NDP memory。仅传输量化权重（如 3-bit：~45.1B × 3/8 ≈ 16.9 GB / 32 layers × 4 experts/layer ≈ 2.1 GB per layer），远小于全精度传输。
     - NDP→GPU：若 $\mathcal{H}_l$ 中有 expert 原在 NDP，其 FP16 权重 → GPU HBM。
     - 此后 decoding 阶段 zero migration。

  5. **Decoding 阶段（GPU-NDP 重叠执行）**：
     - 每个 decoding step，token x_t 经 GPU Router → top-2 experts 选择
     - **Case 1: 两个均在 GPU** → GPU FFN 直接计算 (FP16 GEMM on H100 tensor cores) → 输出
     - **Case 2: 一个 GPU + 一个 NDP** → GPU 计算 hot expert FFN 同时，activation x_t 通过 PCIe → NDP device → NDP 用指定的 b_{l,e} bitwidth 量化权重执行 FFN（64×(4×4) systolic arrays 并行）→ NDP 输出 activation 通过 PCIe → GPU → 加权求和
     - **Case 3: 两个均在 NDP** → x_t 经 PCIe → NDP → 两个 cold experts 依次执行 → 两个 output activations 经 PCIe → GPU → 求和
     - GPU 和 NDP 的计算在 per-layer 粒度实现 pipeline overlap：GPU 计算本层 hot experts 时，NDP 已开始下层 cold experts 的量化计算

  6. **关键优势——Activation Movement vs Parameter Movement**：
     - Baseline (MoNDE on-demand): 每次需要 cold expert → expert weight (FP16, ~45.1B/8/32×K MB per expert) 从 NDP → GPU → 大参数传输
     - Ours (prefill-guided): 仅传输 activation x_t (4096-dim FP16 = 8KB per expert per token) 从 GPU → NDP 或 NDP → GPU → 小激活传输
     - 单 decoding step 数据移动量对比：~数百 MB (parameter) vs ~数 KB (activation)，减少约 10^4-10^5×

  7. **输出返回**：autoregressive 生成完成的 tokens 流式返回客户端。
