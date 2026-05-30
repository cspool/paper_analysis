## MoEsaic: Shared Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoEsaic 在 vLLM 上实现多租户 MoE 模型的专家共享。核心实现包括四部分：(1) Expert Deduplication——在模型加载时计算每个 expert 张量的 128-bit hash digest，通过 in-memory dictionary 检测并去重跨模型实例的相同专家，使多个 client 共享同一份 GPU 显存；(2) Lazy Memory Allocation——使用 tiny pseudo experts 初始化模型，加载时才扩容并填充参数，避免预分配导致的内存不足；(3) Independent Expert Representation——将 vLLM 中 per-layer 的 co-located expert tensor 拆分为单个 expert 独立表示，支持张量级别的单独共享；(4) Fused Gate——将多个 model instance 的 gate 合并为单一 fused gate，批量处理路由请求，避免逐模型串行调用 CUDA kernel；(5) Merged Expert Representation——将去重后的相同专家合并为单一 nn.Parameter，使来自不同 client 的请求在专家计算时自动批处理。
  - 实验比较：MoEsaic 与 dedicated MoE instances (baseline) 对比。比较指标：(a) GPU Memory (GB)——MoEsaic 减少内存占用；(b) Inter-token Latency——fused gate vs separate gate 的路由延迟差异；(c) Throughput (tokens/s)——token 生成速率；(d) GPU Utilization——NVIDIA Nsight 测量的 SM 占用率；(e) Model Loading Time (s)——初始化耗时。MoEsaic 可服务 7× 更多 Mixtral-8x7B 变体且对推理性能影响不大。

- 硬件平台是什么，配置是什么。
  - GPU：8 × NVIDIA A100（40GB）。
  - CPU：64 × AMD EPYC 7742。
  - 推理流量：自定义 chat message 数据集，生成 512 token 序列。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：vLLM（https://github.com/vllm-project/vllm）。
  - 论文未提供独立开源仓库链接（SoCC '24 论文，发表于 2024 年 11 月，IBM Research），代码开源情况：论文未明确说明独立代码仓库链接。
  - vLLM 修改内容：
    1. **Lazy Allocation of Memory**：vLLM 原在模型初始化时预分配所有 expert GPU 显存。MoEsaic 改为用 tiny pseudo experts 初始化，加载时再扩容并填充参数。去重后最大仅占用去重后专家的显存量，避免"全部模型加载完才能去重"导致的内存峰值。
    2. **Independent Representation of Experts**：vLLM 中每层所有 expert 以单个 tensor co-located。MoEsaic 将每个 expert 独立表示为单独的 nn.Parameter 对象，使其内存可独立管理——相同专家共享底层 tensor 但保持独立参数对象。
    3. **Expert Population Tracking**：vLLM 中 in-memory 表示与 in-file 表示不同（多个 in-file tensor 对应一个 in-memory tensor）。MoEsaic 跟踪每个 expert 的张量分配状态，expert 完全填充后标记为"可去重候选"。
    4. **Tensor-Parallel Expert Loading**：vLLM 原生支持 TP 加载但不支持向已部署模型添加新 expert 的 TP。MoEsaic 新增 Ray workers——每个 worker 负责加载指定 GPU 上的 expert shard，新 expert 继承初始模型的 sharding 方式（如 4-way TP），去重在 shard 级别进行。
    5. **Fused Gate**：在每层 MoE layer 中合并多个 model instance 的 gating network 为单一 fused gate，一次性批量完成路由，对比 separate gate 减少了逐模型调用 CUDA kernel 的延迟开销。
    6. **Merged Expert Representation**：模型初始化后，将去重后共享相同底层 tensor 的 expert 合并为单一 nn.Parameter。每个 MoE 的 gate 将 expert ID 映射到合并后的表示，使 Triton kernel 中来自不同 client 的请求在专家计算时被自动批处理。
    7. **Non-disruptive Add/Remove**：支持在运行中动态添加/移除 model instance，无需系统重启（但不可在活跃推理期间执行）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文基于 vLLM（开源），但 MoEsaic 自身的修改代码未提供独立仓库链接。论文未明确说明是否开源。
  - 框架输入到硬件执行全过程（以 Mixtral-4x7B, 4 model instances, 2 shared experts 为例）：

    **阶段 0 — 模型加载与初始化**：
    1. 第一个 client 提交 Mixtral-4x7B 模型。vLLM 启动时，MoEsaic 用 tiny pseudo experts 初始化所有 expert 结构（不占用实际参数显存）。
    2. 从模型文件逐 tensor 加载参数。每加载完一个 expert（即所有 in-file tensor segment 聚合完整），计算该 expert 的 128-bit hash digest，存入 in-memory dictionary。
    3. 后续模型加载时，对每个新 expert 计算 hash，查 dictionary：若命中→新 expert 引用已有 tensor（共享显存）；若未命中→分配新 GPU 显存。
    4. 若 TP=4，生成 4 个 Ray workers，每个 worker 负责加载对应 GPU 上的 expert shard。新 expert 继承初始模型的 4-way sharding 策略。
    5. 所有 expert 加载去重后，将共享底层 tensor 的相同 expert 合并为单一 nn.Parameter 表示，供后续批处理使用。

    **阶段 1 — Gating 与路由**：
    6. 推理请求到达时，每个请求携带其所属 client 的 model_id。
    7. 在每层 MoE layer，MoEsaic 的 fused gate 接收所有请求的 hidden states X ∈ ℝ^(B×H) 和对应的 model_id 列表。
    8. Fused gate 在单次 CUDA kernel 调用中完成所有 model instance 的 gating 计算：对每个 model instance i，执行 Softmax(W_gate^i · X[model_i]) → TopK 选择专家。gate mapping 表将每个 model instance 的原始 expert ID 映射到合并后的 merged expert ID。
    9. 路由结果：每个 token 被分配到一个 merged expert ID，即使来自不同 client，只要路由到相同专家即进入同一个计算批次。

    **阶段 2 — Expert 计算**：
    10. Token-to-Expert Dispatch：按 merged expert ID 将 batch 中所有 token 分配至对应 expert。
    11. Triton kernel 执行各 expert 的 FFN 计算。由于共享专家使用单一 nn.Parameter，来自多个 client 的请求被自然批处理——例如 client 1 的 16 个 token + client 2 的 12 个 token 共 28 个 token 在同一 batch 中由 expert A 处理。
    12. 计算结果按 token 聚合为输出 Y，返回给各 client。

    **阶段 3 — 测量**：
    13. Inter-token latency：从发送请求到生成 token 的平均时间（包含 gating + expert 计算 + attention + KV cache 等全部开销）。
    14. Throughput：tokens/second。
    15. GPU Utilization：NVIDIA Nsight 测量的 SM 平均占用率。

    MoEsaic 的核心作用：在 multi-tenant MoE serving 场景下，通过专家去重减少显存占用（约 1.6× 到 7× 更多模型实例），并通过合并专家表示将多 client 请求自动批处理以提升 GPU 计算效率。
