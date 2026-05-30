## Two-Level Checkpointing Management (Snapshot-PEC + Persist-PEC / 两级检查点管理)

术语是什么？
Two-Level Checkpointing Management 是 MoC-System 提出的分层 checkpoint 管理策略，将传统两阶段流程（GPU→CPU→Storage）升级为 PEC 感知的两级系统。(1) Snapshot-PEC：GPU→CPU 时保存 K_snapshot 个 expert（利用 PCIe 高带宽，保留更多 expert 降低 PLT）；(2) Persist-PEC：CPU→Storage 时仅持久化 K_persist 个 expert（K_persist ≤ K_snapshot，利用持久存储可靠性但避免大 I/O）。恢复时正常节点可直接从 in-memory snapshot 恢复 K_snapshot 个 expert（比存储中的更近期），降低 Persist-PEC 导致的 PLT。同时将 K_pec 拆分引入 K_snapshot 和 K_persist 两个独立超参数，实现效率与精度的精细权衡。

从系统架构角度拆解术语：
两级的保存与恢复流程：

```
# === Two-Level Saving ===
# Snapshot-PEC (每个 rank 的异步线程)
for each MoE layer l:
    non_expert_states → GPU-to-CPU copy          # 完整非 expert
    experts[selected[0:K_snapshot]] → GPU-to-CPU  # K_snapshot 个 expert
    # snapshot buffer ready → persist trigger

# Persist-PEC (异步线程)
wait(snapshot_complete)
for each MoE layer l:
    non_expert_states → serialize + write to storage
    experts[selected[0:K_persist]] → serialize + write  # ≤K_snapshot
    # done → buffer becomes "recovery buffer"

# === Two-Level Recovery ===
# Faulty nodes (data lost):
load_all_from_storage(K_persist subset + non_expert)

# Healthy nodes (snapshot intact):
load_experts_from_cpu_snapshot(K_snapshot subset)  # more recent!
load_non_expert_from_storage(sharded parts)
```

术语一般如何实现？如何使用？
- 基于 Triple Buffering 实现：snapshot/persist/recovery 三个 buffer 状态机轮转。代理线程在每个节点上管理 buffer 生命周期。
- 配置策略：若 PCIe snapshot 可被 F&B 完全覆盖，K_snapshot 设较大值（如 4→降低 PLT）；Persist duration 决定 I_ckpt 下界，K_persist 设较小值（如 1→减小 persist 时间，缩小 I_ckpt 最小可行值）。
- Adaptive Configuration：根据硬件（PCIe 带宽、存储 I/O、GPU 算力）和训练配置（模型大小、并行度）自动推导最优 (K_snapshot, K_persist, I_ckpt)。

涉及论文标题：
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
