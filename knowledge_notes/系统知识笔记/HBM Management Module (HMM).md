## HBM Management Module (HMM)

术语是什么？

HBM Management Module (HMM) 是 ElasticMoE 的核心组件，负责将 NPU HBM 上的模型权重和 KV cache 管理与推理执行解耦。HMM 以持久守护进程运行，加载权重一次后跨多个推理实例复用，推理实例不直接执行磁盘 I/O 而是通过 zero-copy IPC 获取 HMM 管理的内存引用。HMM 分两层：(1) **全局控制面**（Python + Ray 分布式 runtime）：维护集群范围状态，track 资源使用，分发 zero-copy 引用给活跃推理实例，协调缩放时机和工作流；(2) **每设备 worker**（C++，绑定具体 NPU）：执行数据面操作——管理 Ascend IPC 跨进程内存句柄、通过 HCCL 执行 P2P 权重传输、在 EP 重配置时通过 vpage-remap 重映射 expert 权重、通过 disk-copy 选择性从磁盘加载张量。

从系统架构角度拆解术语：

HMM 在 scale-up 中的操作：
```
HMM 控制面:
1. 接收 scale-up trigger → 分析 current_config vs target_config
2. 生成 scaling_plan = {
     zero_copy_keep: [NPU 0-3 attention_weights, KV_cache]  // TP 固定不变
     p2p_transfer: [(NPU0→NPU4, attn_shard), (NPU1→NPU5, attn_shard)]
     expert_migration: [(expert_E1: NPU0→NPU4), (expert_E2: NPU2→NPU5), ...]
     expert_remap: [vpage 更新映射, 无 buffer 重分配]
   }
3. 协调 per-device workers 执行计划

HMM per-device worker:
4. export_handle(tensor) → IPC 句柄 → 发送给 IMM via UNIX socket
5. p2p_copy(src_NPU, dst_NPU, tensor_slice) → HCCL async transfer
6. vpage_remap(expert, new_physical_pages) → 更新虚拟映射
7. 信号 ready → Coordinator → 流量切换
```

术语一般如何实现？如何使用？

控制面 Python + Ray Core (Actor 模型管理 worker 生命周期，分布式 object store 传递引用)，数据面 C++ + CANN API，PyBind11 桥接。通信层：HMM ↔ IMM 通过 ZMQ UNIX domain socket IPC。支持 vLLM model loader backend，可集成任意 vLLM 兼容 MoE 模型。CUDA 扩展方案中控制面逻辑不变，仅替换底层 API（CAN→CUDA IPC, HCCL→NCCL, CANN vpage→CUDA virtual memory）。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models
