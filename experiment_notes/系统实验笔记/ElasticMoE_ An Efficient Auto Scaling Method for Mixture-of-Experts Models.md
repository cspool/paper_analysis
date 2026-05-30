## ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **ElasticMoE 弹性自动伸缩框架**，在 vLLM 之上新增三个模块实现 MoE 模型的细粒度、低延迟、零停机垂直缩放：(1) **Coordinator**：用户请求入口，维护活跃请求队列，监控 SLO 达标率，触发 scale-up/scale-down 指令，并在新旧配置间无缝切换流量；(2) **HBM Management Module (HMM)**：全局控制面（Python/Ray）+ 每设备 worker（C++/CANN API），管理 HBM 上的模型权重和 KV cache，与推理执行解耦。权重只加载一次，跨推理实例通过 zero-copy IPC 共享，缩放时计算最小代价的权重再分配计划（最大化 zero-copy 复用、最小化 P2P 传输）；(3) **Inference Management Module (IMM)**：管理多个推理实例（基于 vLLM），同一时刻仅一个活跃。通过 LRU 缓存 pre-initialized standby 实例，缩放时从 HMM 获取 zero-copy 引用句柄附加权重和 KV cache，快速激活新配置。核心 primitives：`zero-copy`（Ascend IPC 跨进程共享张量）、`p2p-copy`（HCCL P2P 传输绕过 host memory）、`vpage-remap`（虚拟内存管理 expert 权重，避免大缓冲区重分配）、`disk-copy`（选择性磁盘加载避免重复读取）。缩放策略：固定 TP 度，仅调整 DP 和 EP 度。实验比较：(a) Scaling Latency：Horizontal (Replica)、Vertical (Cold Restart)、Vertical (Extravagant)、Vertical (Colocated) vs ElasticMoE，在 DeepSeekV2 Lite、Qwen3-30B-A3B、DeepSeek V3 三个模型上，ElasticMoE 缩放延迟仅为最佳 baseline 的 ≈0.11×（提升约 80.9%）；(b) SLO Recovery：scale-up 4→6 NPU 下 SLO 恢复速度和 scale-down 6→4 NPU 下 SLO/NPU 成本效率；(c) SLO Compliance：RPS 递增负载下各方法维持 SLO≥90% 的能力；(d) Throughput During Scaling：缩放窗口前后和期间吞吐量对比；(e) Ablation：逐步禁用 IPCAlloc、HCCL P2P、PreInit、ZeroCopy 各组件的缩放延迟和 downtime。

- 硬件平台是什么，配置是什么。
  **Huawei CloudMatrix384 supernode**：集成 384×Ascend 910C NPU（每颗 64 GB HBM），192×Kunpeng 920 CPU，分布在 24 个节点。每节点 16×Ascend 910C + 4×Kunpeng 920（1.5 TB 系统 RAM）。所有 CPU 和 NPU 通过 Unified Bus (UB) 互联，提供 non-blocking all-to-all 连接，近均匀的节点内/节点间通信延迟。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：**vLLM**（https://github.com/vllm-project/vllm）。基于 ascend-vLLM（华为 Ascend NPU 适配版）实现。修改/新增：(a) **HMM 模块**：新增 HBM 管理守护进程，负责模型权重持久化加载、KV cache 管理、缩放计划计算和权重再分配。控制面用 Python/Ray，数据面用 C++/CANN API + PyBind11；(b) **IMM 模块**：新增 `ZeroCopyLoader` 替代传统 `DiskLoader`，新增 `Instance Manager` 管理推理实例生命周期、LRU cache 和流量切换；(c) **Coordinator 模块**：新增 `SLO-aware Load Estimator` 监控 SLO 并触发缩放，新增流量无缝切换逻辑；(d) **低层 primitives**：`IpcSafeAllocator` 覆盖 PyTorch 默认内存分配器（torch.ones/empty/full），`p2p-copy`（HCCL isend/irecv/broadcast），`zero-copy`（rtIpcSetMemoryName/rtIpcOpenMemory），`vpage-remap`（aclrtMallocPhysical/aclrtReserveMemAddress/aclrtMapMem），`disk-copy`（按名称/partition/layer 选择性加载），`add-nodes`（运行时动态扩展 HMM 管理的节点和 NPU）；(e) **整体架构**：Coordinator→HMM→IMM 三级模块通过 ZMQ/UNIX domain socket IPC 通信，Coordinator 对外暴露 TCP API（OpenAI-style inference API）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供独立开源代码仓库，但基于开源 vLLM + 华为 ascend-vLLM 和 CANN API 实现，HMM 支持 vLLM model loader backend。ElasticMoE 的弹性 Serving 全流程如下：

  ```
  === 初始化阶段 ===
  Step 1: HMM 从磁盘加载初始配置（如 DP2-TP2-EP4 on NPU 0-3）的模型权重和 KV cache
  Step 2: HMM 通过 IpcSafeAllocator 分配 IPC 兼容内存，export_handle 导出张量引用
  Step 3: IMM 创建活跃推理实例（vLLM），通过 ZeroCopyLoader.open_tensor 获取 HMM 引用句柄附加权重
  Step 4: IMM 可选 pre-initialize standby 实例（仅 CPU 内存），存入 LRU cache
  Step 5: Coordinator 开始路由请求到活跃实例

  === Scale-Up 操作 (NPU 0-3 → NPU 0-5, DP2→DP3, EP4→EP6) ===
  Step 1: Coordinator 的 SLO-aware Load Estimator 检测到 SLO 持续低于 90%，触发 scale-up 命令
  Step 2: HMM 分析旧配置 DP2-TP2-EP4 和新配置 DP3-TP2-EP6，生成最小代价再分配计划
  Step 3: Attention 权重：NPU 0-3 上保持不变（TP 固定），通过 zero-copy 复用；NPU 4-5 通过 p2p-copy (HCCL) 从 NPU 0-1 异步传输
  Step 4: KV Cache：NPU 0-3 已存在的 KV cache 通过 zero-copy 直接复用（无重复分配），旧实例继续使用同一份 cache 服务 in-flight 请求；NPU 4-5 初始化新 KV cache
  Step 5: Expert 权重：全局 remap 专家→NPU 映射以平衡负载，通过 p2p-copy 迁移专家到新 NPU，通过 vpage-remap 更新虚拟→物理映射（旧映射保持活跃直到新实例接管）
  Step 6: IMM 从 LRU cache 取/创建 6-NPU 推理实例，通过 zero-copy 附加权重和 KV cache，标记为 ready
  Step 7: Coordinator 停止向旧实例路由新请求，等待 in-flight 请求完成，旧实例标记 inactive，流量切到新实例
  Step 8: 旧实例终止，释放不再使用的物理内存页

  === 推理运行 ===
  用户请求 → Coordinator TCP API → OpenAI-style forward → 活跃 IMM 实例 (vLLM on Ascend) → HCCL 集合通信 (all-to-all/TP) → Ascend 910C NPU 执行 attention/expert GEMM → 返回 token
  ```

  关键性能收益：(a) Scale-up latency 约 0.11× 最佳 baseline（≈9× 改善），scale-up 2.43s (DP3→DP4, DeepSeek V2 Lite)；(b) Peak memory 仅比 Cold Restart 高 2-3%，比 Extravagant 低 35-40%；(c) 缩放到 4→6 NPU 后几乎立即恢复 SLO 合规（≥90%），Cold Restart 需要额外数十秒恢复；(d) 缩放期间 throughput 达 Cold Restart 的 ≈2×；(e) 零 downtime。
