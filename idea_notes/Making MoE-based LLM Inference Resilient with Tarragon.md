## Making MoE-based LLM Inference Resilient with Tarragon

- baseline方法是什么？
  Baseline 为 **MegaScale-Infer**（解耦 attention-expert 部署）和 **vLLM**（单体部署）的粗粒度故障恢复机制。以 MegaScale-Infer 为例说明全栈执行路径：
  - **算法层**：MoE 推理执行 layer-wise synchronized 的前向传播。Decoding 阶段，每个 AW 对 layer ℓ 执行 attention 计算（更新 KV cache），gating network 选 top-k expert，通过 NCCL all-to-all 或 M2N 通信将 token embeddings 发送到对应 EW。EW 聚合同 layer 同 expert 的 tokens 成 batch 执行 FFN，返回结果给 AW。AW 等待所有 expert 返回后才进入 layer ℓ+1。此过程每层重复，形成严格的同步屏障。
  - **系统框架层**：MegaScale-Infer 使用 vLLM 作为 compute engine，AW 与 EW 之间通过 M2N 或 NCCL all-to-all 通信。Expert placement 是**静态**的——每个 logical expert 被永久绑定到一个物理 EW/GPU，路由固化在 datapath 中。当任一 worker（AW 或 EW）故障时：CCL/NCCL 将 worker set 视为静态通信组，任一 worker 不可用即 abort 整个 communicator，触发所有 worker 被 kill 并 restart。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：EW 使用标准 PyTorch CUDA kernel 执行 expert FFN（libtorch），按 layer-wise batch 聚合 tokens。AW 侧 attention kernel 为 vLLM 内置。无故障感知的 kernel 调度。
  - **硬件架构层**：GCP A3 Ultra 节点，8x H200 GPU (141 GB)，8x 400 Gbps ConnectX-7 RDMA，NVLink 3.6 Tbps。故障后整个 pipeline 停滞 64 秒（包含 worker restart ~18.5s + 重放 prefill + 重放 decoding）。
  - Baseline 核心缺陷：
    1. **粗粒度故障域**：单个 worker 故障导致整个推理任务被 tear down 并 restart，所有 in-flight 状态（KV cache、partial outputs）被丢弃。在 40 节点（320 GPU）部署中，任意时刻至少 1 个节点故障的概率约为 18.1%。
    2. **用户可见 stall**：故障恢复期间 pipeline 完全停止产生新 token（T_stall ~64s for MegaScale-Infer），对交互式 LLM 服务造成严重用户体验退化。
    3. **浪费的计算**：KV cache 和 expert outputs 被丢弃并从零重算。已 decoding 的 token 越多，GPU 计算浪费越大（与已 decoding token 数 i 呈线性增长）。
    4. **静态 expert placement**：expert 与物理 GPU 绑定，该 GPU 故障直接导致对应 expert 不可用，必须等待 replacement worker 启动。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TARRAGON 通过**可重构数据通路 + 双向自愈 + 异步 KV cache checkpointing**三层次设计解决上述缺陷。全栈执行路径（以 decoding 阶段 AW 故障为例，Mixtral-8×7B, 8 AWs + 8 EWs）：
  - **算法层 — 异步增量 KV Cache Checkpointing + Per-Request Restoration**：
    1. 每个 AW 在每层 attention 完成后，利用 AW-EW 通信间隙（link idle 时段，如图 8 的 bursty 流量模式所示），通过 one-sided RDMA write 将新产生的 KV cache segment（每 token 每层一个小 segment，大小为 C = 2×H_kv×(N_hidden_size/H_attn)×S_elem，对 Mixtral-8×7B 约为 expert traffic V 的 12.5%）异步写入 checkpoint store。使用 "async log + commit record" 设计——每个 RDMA write 携带单调递增的 work request ID 作为 sequence number 保证顺序。
    2. AW 故障时，Orchestrator 识别该 AW 上所有活跃请求及其最后 committed token。对于每个被重分配的请求，checkpoint store 通过 GPUDirect one-sided RDMA write 将完整 KV cache segments 直接注入替代 AW 的 GPU 显存。替代 AW 从 committed token+1 开始继续 decoding，无需重放任何 prefill/decoding。
  - **系统框架层 — 可重构数据通路 (REFE + ERT)**：
    1. REFE（Reconfigurable Forwarding Engine）是 AW 侧 C++ 扩展，通过双 QP 设计（control-plane QP for liveness probe + data-plane QP for token embeddings via GPUDirect RDMA）替代 NCCL/M2N 的静态通信模式。对外暴露 `expert_io(expert_id, layer_id, token_embeddings)` API。
    2. ERT（Expert Routing Table）将 logical expert ID 与 physical EW/GPU **解耦**，每个 AW 独立维护一份 ERT，由 Orchestrator 动态更新。故障时只需更新 ERT 条目即可实现流量重路由，无需全局重启。
    3. **AW 侧自愈**：REFE 对 EW 响应设超时→探测 liveness→立即重路由到健康 EW 或 shadow expert→重播 token embeddings。因 expert 计算 deterministic（纯函数），重播产生相同结果。
    4. **EW 侧自愈**：EW 不再等待所有 AW 输入。当收到"足够"健康 AW 的 tokens（或 batch 达最小区间）即开始 expert FFN，将未响应 AW 的 slots 从 batch 中省略。消除全局同步屏障。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层 — Shadow Experts**：
    1. 在每个 EW GPU 显存中预加载 inactive expert 副本（shadow expert），包含与 primary 相同的权重和计算 kernel。Primary 故障时立即激活，避免从存储重新加载权重（数百毫秒到秒级延迟）。
    2. Shadow expert 在无故障时不消耗任何 compute 资源（仅占 GPU 显存，约 2.5 GB per expert for DeepSeek-R1），不引入 kernel-level interference（单 expert 执行延迟与"shadow loaded + primary active"时完全一致，如图 14）。
  - **硬件架构层**：GCP A3 Ultra，3 节点（AWs + EWs + Checkpoint store 各 1 节点），8x H200 (141 GB) per node，ConnectX-7 RDMA with GPUDirect。故障后 stall 从 ~64s 降至 0.3s (EW) / 0.4s (AW)，稳态吞吐与 MegaScale-Infer 匹配（偏差 <2.8%）。
  - 对比 baseline 的改进映射：
    - **粗粒度故障域 → Worker 级故障域（REFE + ERT）**：将 expert identity 与物理 location 解耦，故障时仅更新 ERT 条目而非重启所有 worker。任何 AW 可向任何 EW 发送请求，数据通路完全可重构。AW 和 EW 形成独立故障域，互不强制全局重启。
    - **用户可见 stall → 双向自愈消除等待**：AW 侧自愈（EW 故障）：AW 本地超时重路由，不等待 orchestrator，其他 AW 继续前进。EW 侧自愈（AW 故障）：EW 不等待所有 AW，用部分输入开始 expert 计算。结果：stall 从 64s 降至 0.3-0.4s（160-213× improvement）。
    - **浪费的计算（KV cache 重建）→ 异步增量 checkpointing + Per-request 恢复**：AW 故障后无需重放 prefill 和全部 decoding。替代 AW 直接从 checkpoint store 注入 KV cache，从 committed token 继续。恢复延迟比 Sequential replay 低 1800×，传输流量低 8×，GPU 重计算消除。
    - **静态 expert placement → Shadow Expert 快速激活**：EW 故障后 shadow expert 立即接管（已在 GPU 显存中），避免 T_w（18.5s worker 初始化）的等待。后台 provisioning 并行进行，恢复容量时不影响在线推理。
    - **Spurious 无故障开销 → <3%**：异步 checkpointing 利用 link idle 间隙，不干扰正常 AW-EW 流量。Failure detection probing (10ms) 和 ERT remapping 的总 Steady-state overhead <3%（Ablation study, Fig. 15）。
