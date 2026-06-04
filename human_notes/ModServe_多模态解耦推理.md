# ModServe：模态感知的多模态模型解耦推理

- 导出时间：2026-06-04 10:36 CST
- 来源：当前 Claude Code session 可用对话上下文
- 保存路径：human_notes/ModServe_多模态解耦推理.md
- 范围：仅用户输入与 Claude 最终输出

## 对话记录

### 001 User
（选中 draft/review_draft.md 第 32 行：多模态调度 | ModServe Modality-Aware Disaggregation? | Image Instances (CPU preprocessing + GPU encoding) ↔ Text Instances (LLM prefill + decode)独立autoscaling；modality-aware routing | 模态级独立扩缩容，InternVL-26B上6.8× throughput vs vLLM monolith | Q2.2, vault: paper_secs/ModServe (2224.6)）
教我

### 002 Claude
好的，我来结合论文原文和你的综述表格，系统讲解 **ModServe** 这篇论文。

---

## ModServe：模态感知的多模态模型解耦推理

### 1. 背景：为什么要做 Modality-Aware Disaggregation？

现有的多模态大模型（LMM）serving 系统（如 vLLM）采用 **monolithic（单体）部署**：图像预处理、图像编码、LLM prefill/decode 全部耦合在同一个服务实例中，统一扩缩容。

这带来三个核心问题：

```
┌─────────────────────────────────────────────────────────────────┐
│  Monolithic 部署的问题                                            │
├─────────────────────────────────────────────────────────────────┤
│  1. 图像编码成为 TTFT 瓶颈                                        │
│     - CroAttn 模型中，图像编码占 TTFT 的 65%-79%                   │
│     - DecOnly 模型中，图像编码占 TTFT 的 25%-54%                   │
│                                                                 │
│  2. 各阶段计算特性不同，统一 batching/并行策略低效                   │
│     - 图像编码：compute-bound（类似 prefill）                      │
│     - LLM prefill：compute-bound                                 │
│     - LLM decode：memory-bound                                   │
│                                                                 │
│  3. 生产流量中图像请求有独立突发模式                                │
│     - 图像请求突发与文本请求突发时间不相关                           │
│     - 图像数/请求呈 heavy-tail 分布                               │
│     - 现有 LLM 流量预测误差率高达 79%                              │
└─────────────────────────────────────────────────────────────────┘
```

用论文图1的直观结果：当请求中图像数量增加时（视频/多图场景），monolithic vLLM 的 TTFT 急剧恶化，而 ModServe 通过解耦图像编码器到独立 GPU，TTFT 保持稳定。

### 2. 核心设计：Image Instances ↔ Text Instances 解耦

ModServe 的核心思想是把 LMM 推理管线拆成两个独立的资源池：

```
                      ┌──────────────────────┐
                      │   Text-only requests  │
                      │   (直接跳过图像阶段)    │
                      └──────────┬───────────┘
                                 │
                                 ▼
┌─────────────────────┐    ┌──────────────────────────────┐
│   Image Instances   │    │      Text Instances          │
│                     │    │                              │
│  CPU: 图像预处理     │    │  LLM Prefill (compute-bound) │
│  GPU: 图像编码(ViT)  │───▶│  LLM Decode  (memory-bound)  │
│                     │    │  + Connector (colocated)     │
│  Image Pool Manager │    │                              │
│       ↓             │    │  Text Pool Manager           │
│  独立 autoscaling    │    │       ↓                      │
│  独立 TP 配置        │    │  独立 autoscaling             │
│  独立 batch sizing   │    │  独立 TP 配置                 │
└─────────────────────┘    └──────────────────────────────┘
          │                           │
          │    RDMA pull-based        │
          │    图像 token 传输          │
          └───────────────────────────┘
```

**关键设计决策**：

| 设计点 | 做法 | 原因 |
|--------|------|------|
| **解耦维度** | 按模态分离（Image ↔ Text），而非传统的 prefill/decode 分离 | 多模态流量中图像/文本有独立突发模式 |
| **Token 传输** | Pull-based RDMA（非 push） | 等所有图像 token 就绪后再选最优 Text Instance，信息更充分 |
| **Connector 放置** | 与 Text Instance 共置 | Connector < 0.1% 参数量，< 0.4% TTFT，单独放浪费 GPU |
| **物理共置策略** | 同一服务器内可混合放置（如 1×TP-4 Text + 2×TP-2 Image） | 避免跨机 RDMA 开销，利用空闲 GPU |

### 3. 三大核心机制

#### 3.1 Token-Aware Pool Autoscaling（令牌感知的池化自动扩缩容）

不是按请求速率扩缩，而是按 **token 吞吐量**扩缩：

$$N_{\text{replicas}} = \lceil \frac{ML}{MC} \rceil$$

- $ML$ = 模态特定负载（Image: image tokens/sec, Text: prompt tokens/sec）
- $MC$ = 单实例在不违反 SLO 下的最大容量（来自离线 profiling）

这能同时应对 **请求速率变化** 和 **请求大小变化**（如突然多图像请求增多）。

#### 3.2 Modality-Aware Routing（模态感知路由）

```
Image-text request 到达
        │
        ▼
┌───────────────────────────────┐
│  路由到 image-token load 最少   │  ← 按图像负载均衡，非 round-robin
│  的 Image Instance            │
│                               │
│  大请求（多图）→ 拆分到多个       │  ← image-level parallelism
│  Image Instance 并行编码        │     (因图像间无 attention 依赖)
│                               │
│  编码完成后 → 路由到 pending     │
│  tokens 最少的 Text Instance   │  ← DecOnly: total tokens
│                               │     CroAttn: text tokens only
└───────────────────────────────┘
```

#### 3.3 SLO-Driven Scheduling（SLO 驱动的调度）

Text Instance 内部，不用 FIFO（会产生 HoL blocking），而是根据 SLO 紧迫度和请求大小做优先级调度，短请求（如纯文本、小图）优先。

### 4. 与你表格中其他方法的对比

你的综述表格中同一行还有其他三个相关方法，它们的关系如下：

| 方法 | 解耦方式 | 与 ModServe 的关系 |
|------|----------|-------------------|
| **EEVEE** | 不做解耦，通过 Modal Cache 缓存视觉特征 | 互补：EEVEE 减少重复编码，ModServe 让编码独立扩缩 |
| **vLLM-Omni** | Stage Disaggregation（Thinker→Talker→DiT Vocoder） | 按功能阶段拆，ModServe 按模态拆；两者可组合 |
| **EPD-Serve** | Encode→Prefill→Decode 三阶段解耦 | 按计算特性拆，但把 Encode 单列出来这一层与 ModServe 共享动机 |
| **ModServe** | Image Instances ↔ Text Instances 模态级解耦 | **按模态解耦 + 模态感知路由**是核心创新点 |

关键区别：ModServe 的 **模态感知路由** 是独有机制——不仅解耦了组件，还根据 **DecOnly vs CroAttn 架构差异** 采用不同的负载均衡策略（DecOnly 按 total tokens 路由，CroAttn 按 text tokens 路由），因为 CroAttn 中图像 token 不参与 self-attention，prefill 成本随图像比例升高反而降低。

### 5. 实验结果要点

| 指标 | 结果 |
|------|------|
| 静态分配下 | **3.3–5.5×** higher throughput vs vLLM monolith |
| 动态场景 | **25–41.3%** 成本节省（满足 P99 TTFT SLO） |
| InternVL-26B | **6.8×** throughput vs vLLM monolith（表格中的数据点） |
| 可组合性 | 与 PD disaggregation 兼容，可在 Text Instance 上叠加 |

### 6. 一句话总结

> **ModServe 的核心洞察是：多模态推理中，图像处理和文本处理的计算特性、流量模式、burst 行为都不同，把它们混在一起统一扩缩容必然低效。解决办法是按模态拆成两个独立资源池，各自用最适合的策略 autoscaling、batching、TP 配置，再通过模态感知路由把两者高效连接起来。**

---

如果你想继续深入某个方面（比如 ModServe 和 EPD-Serve 的详细对比、CroAttn vs DecOnly 架构差异的 serving 含义、或者想把这部分写入 Obsidian 笔记），告诉我即可。
