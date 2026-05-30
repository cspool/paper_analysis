## EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend

- baseline方法是什么？
  多模态推理的 **monolithic 架构**（以 vLLM v0.11.0 为代表）：将 Encode（视觉编码器处理图像/视频）、Prefill（LLM 首次前向生成首 token 及 KVCache）、Decode（自回归逐 token 生成）三个阶段串行绑定在同一硬件资源上执行，不存在阶段级别的逻辑隔离或物理资源划分。

  全栈执行例子（openPangu-7B-VL on Ascend Atlas 800I A2，处理一条含图像的多模态请求）：
  - **模型推理算法层**：ViT (0.7B) 编码图像 → 特征 token 序列 V_m；文本 prompt token 化与 V_m 拼接 → 输入 LLM (7B) 执行 Prefill（生成首 token O_1 + 构建 KVCache KV1）→ 自回归 Decode（基于 O_i 和 KV_i 生成 O_{i+1} 直至 max_length 或 <eos>）。
  - **系统框架层**：vLLM v0.11.0 monolithic——E/P/D 三阶段在同一 NPU 上串行执行。PagedAttention 管理 KV cache。请求间通过 continuous batching 共享 Decode 阶段。没有按模态或阶段分拆的调度路径（多模态和纯文本请求混在同一队列）。
  - **编译框架层**：论文未明确说明。使用 Ascend CANN（Compute Architecture for Neural Networks）编译框架，标准 PyTorch 前端。
  - **kernel调度层**：Ascend NPU 上 MatMul、Attention 等算子由 CANN 运行时调度到 AI Core、AI Vector 等计算单元，无阶段级算子系统调度优化。论文未明确说明 kernel 级细节。
  - **硬件架构层**：华为 Ascend Atlas 800I A2，单 NPU 64 GB HBM。AI Core（矩阵/向量密集计算）+ AI Vector（AllReduce 等通信算子）。E/P/D 全部串行占据同一 NPU 资源，无跨 NPU 通信优化。

  Baseline 缺陷：
  - (a) **阶段耦合导致执行干扰**：视觉编码（compute-heavy Encode）和文本 Prefill 共享同一 NPU，无隔离机制。多模态请求可能阻塞纯文本请求，推高 TTFT，并打乱 Decode 调度节奏，恶化 TPOT 和整体吞吐。
  - (b) **统一并行策略不匹配异构阶段需求**：Encode 偏好数据并行或序列并行，Decode 偏好张量并行以降低延迟。Monolithic 的统一并行策略无法为每个阶段单独优化，限制扩展性。
  - (c) **串行执行阻止资源复用**：E/P/D 严格串行，尽管三阶段在计算-访存特征上互补（Encode compute-heavy + Decode memory-heavy），大量 NPU 计算资源在阶段切换时闲置。
  - (d) **跨阶段 tensor 传输无优化**：Monolithic 无跨 NPU 的 E-P 特征传输和 P-D KV 传输需求（同一 NPU 内存共享），但扩展到解耦场景时缺少传输优化机制。
  - (e) **无模态感知路由**：多模态请求和纯文本请求无差别的混合调度，高负载多模态请求会抢占资源影响纯文本请求的延迟。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **EPD-Serve：将多模态推理 pipeline 按 E/P/D 三阶段解耦为独立可调度实例，配合异步跨阶段 tensor 传输优化和灵活物理共置策略**。核心设计：(i) E-P-D 阶段级解耦——将 Encode/Prefill/Decode 拆分为独立实例进程，通过 Proxy 统一路由；(ii) E-P 异步特征预取——基于 Mooncake Store 构建 MM Store（hash→feature 缓存池），事件驱动只传 hash 不传全量特征，异步预取隐藏通信延迟；(iii) P-D 分层分组 KV 传输——按 Transformer 层分组打包 KV cache，延迟调度对齐通信与 Prefill 计算；(iv) 模态感知多路径调度——按请求模态分派到 E-P-D 完整管道或 P-D 纯文本管道；(v) 灵活物理共置——逻辑隔离 + 硬件空间复用（AI Core/AI Vector 算子级并行）。

  全栈执行例子（openPangu-7B-VL on Ascend Atlas 800I A2, (E-P)-D 部署, 2 NPU, ShareGPT-4o, 10 req/s）：
  - **模型推理算法层**：同一 ViT + LLM 推理 pipeline。Encode 生成 V_m ∈ R^{n×3584}，Prefill 计算 KV cache，Decode 自回归生成。不修改模型结构与计算逻辑。
  - **系统框架层**：EPD-Serve 将 vLLM monolithic 替换为三实例架构——(E-P) 实例处理 Encode+Prefill，D 实例独立处理 Decode。Proxy 接收请求 → 模态感知路由（多模态→E-P-D 管道，纯文本→P-D 管道）→ 实例级最少负载优先调度。MM Store 缓存已编码的多模态特征（key=hash(input)，value=feature vector）。E-P 异步预取：Encode 完成后仅发 hash 事件；Prefill listener 收到后从 MM Store 检索并加载特征。P-D 分层传输：Prefill 计算 L+1 层时异步传输 L 层 KV cache 至 Decode 实例。分组打包减少握手频率，延迟调度避开通信峰值。
  - **编译框架层**：论文未明确说明。使用 Ascend CANN 编译框架，未修改底层编译路径。
  - **kernel调度层**：NPU 上的算子调度由 Ascend CANN 运行时管理。EPD-Serve 在算子层面的关键优化是**物理共置空间复用**：(E-P) 共置 NPU 上，当 Prefill 的 MatMul（AI Core）完成而该阶段等待 P-D 传输时，Encode 的 AllReduce（AI Vector）可利用空闲 AI Core 执行；反之亦然。通过算子级硬件资源互补（operator complementarity 详见 Figure 6），减少 NPU 的 idle 周期。论文未明确说明自定义 kernel。
  - **硬件架构层**：Ascend Atlas 800I A2，每 NPU 64 GB。利用 AI Core（矩阵乘）和 AI Vector（规约/通信）的异构计算单元实现算子级并行复用。(E-P)-D 中 Encode+Prefill 共置 1 NPU，Decode 独占 1 NPU。跨 NPU 通信通过 Mooncake Store 的异步传输接口实现。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 阶段耦合导致执行干扰 (a)** → 方案：E/P/D 拆分为独立实例进程，通过 Proxy 统一路由和负载均衡。多模态请求和纯文本请求按模态分派到不同管道，隔离跨模态的调度干扰。
  - **defect: 统一并行策略不匹配异构阶段需求 (b)** → 方案：每个阶段实例独立配置并行策略和资源分配，可按需弹性伸缩（如 Encode 实例多副本、Decode 实例张量并行）。
  - **defect: 串行执行阻止资源复用 (c)** → 方案：物理共置 + 空间复用——逻辑层独立调度，物理层共享 NPU。通过 operator-level co-location（Figure 6），将硬件资源需求差异大的算子（MatMul vs AllReduce）在时间线上交错复用，提升 NPU 利用率。
  - **defect: 跨阶段 tensor 传输无优化 (d)** → 方案：(i) E-P 异步特征预取：MM Store 缓存 + 仅传 hash → Prefill 异步检索本地缓存，transmission overlap ratio 接近 100%（主流分辨率下）；(ii) P-D 分层分组 KV 传输：overlap ratio 从 15.27%→98.78%（seq_len=1024），bandwidth utilization 提升 58%。
  - **defect: 无模态感知路由 (e)** → 方案：多路径调度——按请求模态分派到 E-P-D（多模态）或 P-D（纯文本）管道，配合实例级最少负载调度，实现异构流量分离和动态负载均衡。
  - **额外设计：灵活部署拓扑**：支持 E-P-D / EP-D / ED-P / E-PD / (E-P)-D / (E-D)-P / (E-PD) 等拓扑按 SLO 需求切换：(E-P)-D 兼顾低 TTFT+低 TPOT（balancing）；(E-D)-P 优化 TTFT（first-token priority）；(E-PD) 最大化吞吐（throughput priority under relaxed SLO）。
