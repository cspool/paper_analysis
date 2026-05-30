## Sparsity-Aware Expert Cache (稀疏感知专家缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparsity-Aware Expert Cache 是 MoE-Infinity 提出的面向个人机器 MoE 推理的专家缓存设计范式。核心理念是：在 batch_size=1 的个人部署场景下，MoE 模型 expert 激活呈现高度稀疏性和请求内偏斜重用模式，缓存策略应利用这些稀疏激活特征（而非忽略它们）来指导 expert 的预取（prefetching）和淘汰（eviction）。Sparsity-Aware 体现在三个层面：(1) 请求级稀疏追踪——EAMC 按请求粒度记录 expert 激活模式，捕获分组激活 (Grouped) 和稀疏性 (Sparsity)；(2) 激活感知预测——PredictEAM 通过余弦匹配历史模式预测未来 expert 使用概率，捕获重用 (Reuse)；(3) 概率感知淘汰——eviction priority 综合考虑 expert 的预测概率、历史频率和层位置。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Sparsity-Aware Expert Cache 的完整运行时流程（对比 LRU 和 Statistical Count 两种传统方法）：

```
场景: MoE 模型有 4 层，每层 2 个 expert。当前处理第 2 层，
      token 被路由到 E[2,1]。GPU cache 容量 = 2 experts。
      Layer 3 将激活 E[3,2]（未来信息，cache 策略不可见）。

传统 LRU Cache（vLLM, DeepSpeed, Llama.cpp）:
  Cache = [E[1,1], E[2,1]]
  进入 Layer 3: 预取 E[3,1] (依赖关系) → evict E[1,1]
  Router 选择 E[3,2] → Cache MISS → FetchOnDemand E[3,2]
  问题: 基于执行顺序预取，不感知哪些 expert 会实际被激活，
        evict 了未来可能需要的 expert (E[1,1] 或 E[2,2])

传统 Statistical Count（BrainStorm, DeepUM）:
  Cache = [E[1,1], E[2,1]]
  全局频率: 各 expert 使用接近均匀 → 无法区分优先级
  预取 E[3,1] 或 E[3,2]: 无法确定 → 可能预取错误的 expert
  问题: 跨请求的均匀分布冲淡了请求内的偏斜模式

Sparsity-Aware Expert Cache（MoE-Infinity）:
  iEAM = [[1,0], [0,1], [0,0], [0,0]]  (前 2 层已执行)
  PredictEAM(iEAM, EAMC):
    → 匹配到历史 rEAM = [[1,2], [0,3], [0,3], [0,3]]
    → 聚合归一化 → pEAM 显示 E[2,2] 在 layer 2 高概率
                  E[3,2] 在 layer 3 高概率
  Cache 决策:
    保留 E[2,2]（pEAM 显示将被重用）→ 不 evict
    预取 E[3,2]（pEAM 显示 layer 3 将激活）→ 提前加载
  进入 Layer 3: Router 选 E[3,2] → Cache HIT ✓
```

Eviction Priority 公式（Algorithm 1）:
```
p = n_token / ((pEAM[e.layer_idx] + ε) × (1 - e.layer_idx/L))
```
三个因子：(1) n_token = expert 在匹配的历史模式中的 token 数，代表重用频率；(2) pEAM[e.layer_idx] + ε = 该 expert 在当前预测中的概率，ε 防零；(3) (1 - e.layer_idx/L) = 层衰减——浅层 expert 获得更高缓存优先级（因为浅层预取预测不够准确）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：MoE-Infinity 基于 PyTorch 自建推理运行时，GPU 侧维护固定大小的 expert cache（由 GPU 显存减去 attention/KV-cache 后的剩余空间决定）。CPU 侧维护 EAMC 和执行 PredictEAM。GPU-CPU 数据传输使用 pinned memory + DMA（每个 GPU 一个独立 I/O 线程，PCIe 4.0 单线程可打满 32GB/s）。支持 PyTorch 和 HuggingFace checkpoint 格式，集成 FlashAttention。
- **实验效果**（A5000 24GB GPU, PCIe 4.0）：
  - Mixtral-8x7B: 836ms TPOT，比 DeepSpeed-Inference/Ollama/vLLM 提升 1.4×
  - DeepSeek-V2-Lite: 155ms TPOT，比 vLLM (485ms) 提升 3.1×，比 Ollama (2590ms) 提升 16.7×
  - Arctic-128x4B (900GB): 唯一可在单 GPU 上提供可竞争推理性能的系统
  - Long context (2^12→2^17 tokens): 在 2^16 token 前保持低延迟，之后因 KV-cache 挤占 cache 空间而降级为 on-demand fetching（增量延迟 137ms，仍低于 vLLM 和 Mixtral-Offloading）
- **局限性**：(1) 假设 batch_size=1 的个人机器场景，高 batch 下 expert 激活趋于均匀分布，Sparsity-Aware 的收益递减；(2) EAMC 需要 warm-up 阶段（约 50 个请求恢复最优延迟 after workload shift）；(3) 对 expert 数量少但单个 expert 大的模型（如 Mixtral-8x7B，仅 8 experts/layer），稀疏性有限，收益相对较小（1.4× vs 3.1–16.7×）。

涉及论文标题：
- MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving
