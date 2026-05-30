## Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：基于 **DeepSpeed inference engine** 构建的 expert-aware 推理系统。核心调度创新有三项：(1) **Expert-aware Batching（Algorithm 1）**：利用 pre-gating router 在推理前已知每个 token 的 expert 选择，将选择同一 expert 的 tokens 组 batch，最小化每 batch 所需激活的 unique expert 数量。伪代码：维护 ReqQueueByExpert（每个 expert 的请求队列），每次从请求最多的 expert 队列取 tokens 填充 batch（最多 MaxTokenLen），确保 batch 内 token 共享同一 expert。(2) **Fine-grained Prefetching**：利用 pre-gating 已知未来层所需 expert，在计算第 i 层时并行预取第 i+1 层的 expert 权重（compute stream 与 loading stream 流水线重叠），隐藏 expert 加载延迟。(3) **Belady-inspired Optimal Caching**：Belady 算法为理论最优离线缓存替换策略（替换未来最远访问的对象），传统上因无法预知未来访问而不可行。Read-ME 通过 pre-gating 预知所有 token 在所有层的 expert 需求，实现近似最优缓存替换：C(t-1)={e_1,...,e_k}，evict e* = argmax_{e∈C} F(e,t)（F(e,t) 为 expert e 下一次被请求的时间）。
  - 实验比较：(1) **Batching 策略**：Read-ME expert-aware batching vs Decoding-prioritized batching [38] vs Prefill-prioritized batching [39,47]，在 Chatbot Arena Dataset 重放负载下评估端到端延迟分布和 p95 延迟；(2) **Prefetching**：Read-ME Prefetching vs On-demand Loading [25]，在不同 expert cache capacity 下对比端到端延迟；(3) **Cache 策略**：Random vs LRU vs Belady 在不同 cache capacity (2/3/4/5 experts) 下对比 cache hit ratio；(4) **单请求延迟分解**：OpenMoE (layerwise router) vs Read-ME (pre-gating router) vs Dense Llama2，分解为 Router/Attention/Expert(MLP) 各部分延迟。

- 硬件平台是什么，配置是什么。
  - 推理系统评估：单卡 NVIDIA A100 80GB GPU。Host CPU memory 用于 expert offloading（论文未明确说明具体 CPU 配置）。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：DeepSpeed inference engine [38]（https://github.com/microsoft/DeepSpeed）。
  - 修改内容：
    1. **Expert-aware Batch Scheduler**：新增 ReqQueueByExpert 数据结构（每个 expert 一个 FIFO queue），Scheduler 收集 pre-gating router 输出的 expert assignment，将 tokens 按 expert 分入对应 queue。Algorithm 1 从 queue 中构建 batch：优先从请求最多的 expert 取 tokens，最大化 batch 内 expert 共享。
    2. **Prefetch Pipeline**：在 DeepSpeed 的 layer-wise 推理循环中插入异步 expert 加载流。Compute stream 执行第 i 层 expert FFN，同时 loading stream 从 host memory 向 GPU memory 传输第 i+1 层所需 expert 权重（cudaMemcpyAsync / PCIe transfer）。
    3. **Belady Cache Manager**：新增 expert cache 模块，维护 k 个 expert slots。Router 预计算所有 pending requests 在未来时间步所需的 expert 序列，构造 F(e,t) 映射（每个 expert 的下次访问时间）。Cache eviction 时选择 max F(e,t) 的 expert 驱逐。Cache 跨所有并发请求共享。
    4. **Pre-gating Router Integration**：将 Read-ME 的 pre-gating router 作为推理 pipeline 的第一步执行。Router 输出每个 token 在所有层的 expert assignment，传递给 Scheduler 进行 batch 规划和 cache 预热。
  - 开源：论文代码开源 https://github.com/VITA-Group/READ-ME。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/VITA-Group/READ-ME
  - 框架输入到硬件执行全过程（Read-ME 4.7B-17B, single A100 80GB, Chatbot Arena workload）：
    
    **阶段 0 — 系统初始化**：
    1. 加载 Read-ME model checkpoint（experts + pre-gating router + permanent expert）到 host memory。
    2. 初始化 DeepSpeed inference engine，替换原生 MoE inference pipeline 为 Read-ME pipeline。
    3. 初始化 Expert Cache：在 GPU memory 分配 k 个 expert slots（默认 k=5），初始为空。
    4. 初始化 ReqQueueByExpert[N]：N=8 个 expert queue（+ 1 permanent expert queue）。
    
    **阶段 1 — Pre-gating（推理前一次性路由）**：
    5. 多个 conversation requests 到达，Scheduler 收集所有 request tokens。
    6. Pre-gating Router G 执行：对每个 token sequence x_{≤t}，Router（1-layer Transformer, causal attention）计算 gating weights G(x_{≤t}) ∈ R^N，选 top-K=2 experts。因 router 与 layer index 无关，所有层 expert 选择一致。
    7. Router 输出：每个 token → {expert_i, expert_j}（跨所有 32 层相同）。
    
    **阶段 2 — Expert-aware Batching**：
    8. Scheduler 按 Algorithm 1 构建 batch：扫描 N 个 ReqQueueByExpert，从请求最多的 expert 开始，取 tokens 直到 batch size = MaxTokenLen。
    9. 结果：batch 内所有 tokens 共享同一 expert → 单 batch 仅需加载 1 个 expert（vs layerwise MoE 的 ~7.63/8 experts per batch）。
    10. Scheduler 将 batch 提交给 Inference Engine。
    
    **阶段 3 — 流水线推理执行**：
    11. Layer 1 开始：检查 Expert Cache——若 required expert e_1 在 cache → 直接使用；否则触发 cache miss，从 host memory 加载 e_1 到 GPU（PCIe 4.0, ~25 GB/s）。
    12. 同时启动 Prefetch Stream：异步加载 Layer 2 所需 expert e_2。
    13. Compute Stream：执行 Layer 1 attention（causal self-attention）→ Expert e_1 FFN（SwiGLU, d=5504）→ permanent expert FFN → 输出 hidden states。
    14. Cache Manager：Router 已预知 e_1 的下次使用时间 F(e_1, t)，若 cache 满，evict max F(e,t) 的 expert（Belady 策略）。
    15. 重复 Layer 2...32：每层计算与下层 expert 加载流水线重叠。
    
    **阶段 4 — Token Generation 循环**：
    16. 新生成的 token 通过 Router G（仅需对新 token 执行，causal attention 利用 KV cache）。
    17. 新 token 的 expert assignment 加入对应 ReqQueueByExpert。
    18. Scheduler 动态重组 batch，重复阶段 2-3。
    
    **关键结果**：
    - Expert-aware batching 将平均 unique experts/batch 从 5.08（decode-prioritized）/ 5.21（prefill-prioritized）降至 3.51。
    - 端到端平均延迟降低 5.0-6.1%，p95 延迟降低 9.5-10.0%。
    - Prefetching vs On-demand Loading：最高 30% 延迟改善。
    - Belady caching 在 cache capacity=4 时 hit ratio 77.21% vs LRU 66.95%（+10.26%）。
