## Accelerating Distributed MoE Training and Inference with Lina

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 Lina 系统，基于 DeepSpeed MoE (Microsoft) 构建，**在 Training 端**通过 tensor partitioning + micro-op priority scheduling 优先 all-to-all 避免与 allreduce 争抢网络带宽，并引入 expert packing 提升流水线效率；**在 Inference 端**通过 token-level expert selection pattern 估算 expert popularity，动态调度 expert-device 映射（两阶段调度：phase 1 基于估算预分配、phase 2 偏差过大时微调），以均衡设备负载和 all-to-all 带宽。
  
  实验比较：
  - **Training**: Lina vs DeepSpeed Baseline vs Tutel，比较 training step time、MoE layer time（前向/反向）、all-to-all time、GPU utilization、pipelining efficiency
  - **Inference**: Lina（完整版） vs Lina w/o estimation vs Lina w/o fine-tuning vs Baseline vs Ideal（perfectly balanced gating），比较 median/95%ile inference time、MoE layer time、tail all-to-all time
  
  Benchmark: Transformer-XL (24L, text generation on Enwik8), BERT-Large (12L, translation on WMT En-De), 以及 IMDB Reviews/Twitter sentiment analysis, WMT French/Russian 等泛化任务。

- 硬件平台是什么，配置是什么。
  4 个 worker 节点，每节点 4 块 NVIDIA Ampere A100 GPU（40GB HBM），节点间 100Gbps InfiniBand 互联。Training 使用与 expert 数量相等的 GPU（最多 16 GPUs，即 4 节点×4 GPU），Inference 同理。

- 开源Serving框架是什么。修改了什么。
  **开源框架**: DeepSpeed MoE (https://github.com/microsoft/DeepSpeed)，PyTorch 1.10 + CUDA 11 + NCCL 2.10。
  **论文代码开源情况**: 论文未明确提供独立开源仓库，约 7500 LoC 修改（C++/Python）基于 DeepSpeed MoE 和 PyTorch。
  
  **修改内容**:
  1. **Training 端——Communication Scheduler**：在 backward pass 中将 all-to-all 和 allreduce 分解为 micro-ops（tensor partitioning 到固定大小 chunk 如 30MB），使用 priority queue 调度，保证 all-to-all 优先获满带宽，allreduce micro-ops 仅在无 all-to-all 待处理时发射。修改 PyTorch DistributedDataParallel 的 bucketing 机制（不再 fuse gradients，而是分区每个梯度）。
  2. **Training 端——Expert Packing Coordinator**：动态调整每设备 expert 数量（powers of two，从 1→2→4），当 FFN micro-op 时间短于 all-to-all micro-op 时触发 packing；不足时使用 DRAM-offloading swap expert param。
  3. **Inference 端——Resource Scheduler**：在 device 0 上运行独立线程，管理 expert-device 映射。两阶段调度：Phase 1 基于 profiling 阶段采集的 token-level expert selection path patterns 估算 expert popularity（样本路径长度 l=3），按 `n_e = N × Σ P(e) / N_t` 分配设备，使用 first-fit-decreasing heuristic 打包；Phase 2 比较估计与实际 routing 结果，top-2k 不一致时重新计算分配。
  4. **Inference 端——All-to-All Coordination**：使用 unequal split all-to-all 避免多 process group 开销；无 token 导向某 device 时传 placeholder pointer。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **Lina Training 端执行全过程（以 16-expert Transformer-XL 一个 training step 为例）**:
  1. **输入**: batch tokens 分布在 16 GPU 上（data parallelism），每个 GPU 持 1 个 expert（expert parallelism degree=1，或 packing 后每 GPU 2-4 experts）
  2. **Forward Pass**: 非 MoE 层（Attention/LayerNorm）本地计算 → Gate 路由选择 top-2 experts → all-to-all dispatch tokens → Expert FFN 计算（如 packing=2，两 expert 串行执行）→ all-to-all combine → 输出 combine
  3. **Backward Pass**: 反向传播开始
     - **Commission Scheduler** 将 gradient tensor 按 30MB chunk 分为 micro-ops
     - **Priority Queue**: 当 all-to-all micro-op 到达时置顶；allreduce micro-op 仅在队列无 all-to-all 时发射
     - 避免了 all-to-all 与 allreduce 同时使用网络带宽（baseline 中两 CUDA stream 各自发射导致带宽均分）
  4. **Expert Packing**: 每 4 steps 检查 FFN vs all-to-all micro-op 时间比 → 若 FFN < all-to-all → 增倍 packing → 通过一次 synchronous all-to-all 交换 expert 参数 → 下次 iteration 生效
  5. **优化器步**: 所有 gradient allreduce 完成后 optimizer.step()

  **Lina Inference 端执行全过程（以 16-expert BERT-Large 推理一个 batch 为例）**:
  1. **输入**: 推理请求 batch tokens 分布在 16 GPU 上
  2. **Layer 1-3（warm-up）**: 标准 MoE 推理（Gate → All-to-All → Expert → All-to-All），不作调度（累积 expert selection path）
  3. **Layer 4+ —— Phase 1（预调度）**:
     - Device 0 Scheduler: 根据 profiled expert selection patterns `{Ψ}` 和当前 token sample paths（长度 l=3），估算下一层各 expert 的 popularity `n_e`
     - 估算信息 piggyback 在第一个 all-to-all 中发送到 device 0
     - Scheduler 计算新 expert-device 映射（first-fit-decreasing），popular expert 复制到多 device → unpopular 打包到少 device
     - 映射结果通过第二个 all-to-all 下发 → 各 device 从 host DRAM swap in 对应 expert 权重
  4. **Phase 2（微调）**:
     - Gate 执行后各 device 对比实际 routing vs 估算 → 通过 NCCL send 报告
     - 若 top-2k expert 一致（~77% cases）→ scheduler 广播 resume 信号 → 模型继续
     - 若不一致 → scheduler 重算 expert-device mapping → 广播新映射 → 模型 blocked 直到收到命令
  5. **Expert Computation**: 多 expert 的 device 串行执行各 expert FFN（每次 load 一个 expert 权重）
  6. **Unequal Split All-to-All**: combine 阶段按 device 实际 token 量发送，非均匀拆分
  7. **重复 3-6** 直至所有 MoE layer 完成

  **Baseline (DeepSpeed MoE) 对比**:
  - Training: backward pass 中 all-to-all (stream b) 与 allreduce (stream c) 独立发射无协调，网络带宽公平共享 → all-to-all 被延长 1.83x~4.14x
  - Inference: 所有 device 均匀持有 1 个 expert → popular expert 处理 token 量远多于 unpopular（最大 5.56x）→ 延迟尾部拖长
