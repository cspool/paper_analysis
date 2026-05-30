## EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：EPD-Serve 是一个支持 Encode-Prefill-Decode 三阶段解耦的多模态推理 Serving 系统，部署于华为 Ascend NPU。核心实现包括：(i) E-P 异步特征预取机制——基于 Mooncake Store 构建共享多模态缓存池（MM Store），通过事件驱动的 hash 通知实现跨节点特征传输与 Encode 计算重叠；(ii) P-D 分层分组 KV cache 传输机制——按 Transformer 层将 KV cache 分组打包，延迟调度以对齐通信与 Prefill 计算，最大化通信-计算重叠（overlap ratio 从 baseline 15.27% 提升至 98.78% at seq_len=1024）；(iii) 模态感知多路径调度——根据请求模态（纯文本 vs 多模态）路由到不同管道（P-D vs E-P-D），配合实例级最少负载优先的全局调度策略；(iv) 灵活阶段解耦与物理共置——支持 E-P-D、EP-D、ED-P、E-PD、(E-P)-D、(E-D)-P、(E-PD) 等部署拓扑，逻辑隔离 + NPU 空间复用实现算子级并行（MatMul 用 AI Core、AllReduce 用 AI Vector 互补执行）。
  - 实验比较：(i) EPD-Serve 多种部署拓扑（E-PD、(E-PD)、(E-P)-D、(E-D)-P、EP-D、E-P-D、TP1、TP2）在吞吐、TTFT、TPOT、SLO 达成率上的全面对比；(ii) 传输优化消融实验——对比 Baseline(E-P-D) vs +E-P 异步预取 vs +P-D 分层分组 vs 两者全开；(iii) Encode 解耦效益分析——(E-PD) vs TP1/TP2/E-PD；(iv) Decode 解耦效益分析——EP-D/(E-P)-D/(E-D)-P vs TP1/TP2；(v) 全解耦效益分析——E-P-D vs (E-P)-D vs (E-D)-P vs EP-D at 10 req/s 高负载。

- 硬件平台是什么，配置是什么。
  - 华为 Ascend Atlas 800I A2 服务器，每 NPU 64 GB 片上内存。单机多 NPU 环境。所有对比实验在相同硬件配置下进行以保证公平性。

- 开源Serving框架是什么。修改了什么。
  - Baseline 框架：vLLM v0.11.0（默认 monolithic 架构——Encode/Prefill/Decode 串行在同一计算资源上执行）。
  - EPD-Serve 修改：(i) 将 E/P/D 拆分为独立实例进程，支持独立调度和弹性伸缩；(ii) 引入统一 Proxy 组件执行跨实例请求路由和负载均衡；(iii) 基于 Mooncake Store（参考 Mooncake [12]）实现 MM Store 多模态缓存池；(iv) 实现 E-P 异步特征预取和 P-D 分层分组 KV 传输模块；(v) 增加模态感知路由逻辑；(vi) 支持多种 flexible deployment topology 的物理共置。底层 PyTorch/Ascend 计算框架未修改。
  - 论文未声明 EPD-Serve 开源，未提供代码仓库 URL。关联工作 Mooncake Store 开源在 https://github.com/kvcache-ai/Mooncake。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未声明开源（2026年5月检索未找到公开代码仓库）。底层传输组件 Mooncake Store 关联开源。
  - 使用例子（openPangu-7B-VL on ShareGPT-4o, (E-P)-D 部署，2 NPU，10 req/s）：
    1. **输入**：API Server 接收请求，判断输入是否含图像/音频/视频模态。多模态请求路由到 E-P-D 完整管道，纯文本请求路由到 P-D 管道。AISBench 控制请求注入速率 10 req/s。
    2. **Encode + Prefill 阶段**（共置 NPU 1 (E-P)）：Vision Encoder (ViT 0.7B) 编码图像 I_m → 特征 V_m ∈ R^{n×3584}。完成后事件驱动异步发送特征 hash（非完整 tensor）到 Prefill 实例的 listener。Prefill listener 从 MM Store（hash→feature vector）检索并写入本地缓存 → 若 miss 则本地重算（fault-tolerant）。文本提示 I_t 编码为 V_t → 拼接 V_m+V_t 输入 LLM (7B) → 逐层 Prefill 计算 KVCache。
    3. **P-D 传输**：当 Prefill 开始计算 L+1 层时，L 层的 KVCache 异步传输至 Decode 实例。多层 KV 分组打包减少握手次数，延迟调度避免通信峰值。通过分层分组，KV 传输 overlap ratio 从 baseline 15.27% 提升至 98.78%（seq_len=1024）。
    4. **Decode 阶段**（独立 NPU 2 (D)）：接收分层到达的 KVCache，按自回归逐 token 生成 O_i+1。独立 Decode NPU 不受 Encode/Prefill 资源竞争影响，稳定低 TPOT。输出固定 64 tokens 或至 <eos>。
    5. **物理共置空间复用**：NPU 1 上 Encode 与 Prefill 共享 AI Core/AI Vector——当一个阶段等待通信时另一阶段利用空闲计算单元。MatMul（AI Core compute-heavy）与 AllReduce（AI Vector communication-heavy）交替执行实现算子级并行。
    6. **输出**：Proxy 收集各实例结果返回客户端。SLO 约束：TTFT ≤ 2000ms, TPOT ≤ 50ms（Decode-disaggregated 时）。Per-NPU effective throughput = 77.36 tokens/s（(E-P)-D at 10 req/s），SLO attainment rate = 26.17%。
  - **作用**：在 Ascend NPU 上实现多模态推理三阶段灵活解耦与物理共置优化。(E-P)-D 在 12 req/s 高并发下比 PD-disaggregated EP-D 提升吞吐 57.37-69.48%；全解耦 E-P-D (3 NPU) 在 10 req/s 下 SLO 达成率 94.34%；单 NPU 共置 (E-PD) 比 monolithic TP1 提升吞吐 12.87-14.88%、降低 TTFT 2.7-3.25%、降低 TPOT 69.58-70.39%。
