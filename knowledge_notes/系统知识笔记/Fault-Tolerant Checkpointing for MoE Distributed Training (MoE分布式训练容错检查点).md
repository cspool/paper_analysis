## Fault-Tolerant Checkpointing for MoE Distributed Training (MoE分布式训练容错检查点)

术语是什么？
Fault-Tolerant Checkpointing 是分布式训练系统中通过周期性保存和恢复模型状态来保证故障后训练可继续的机制。在 MoE 训练中面临独特挑战：(1) MoE 模型因大量 expert FFN 导致 checkpoint 体积远超等计算量的 dense 模型——expert 部分占 checkpoint 总量的约 86%（以 GPT-350M-16E 为例：expert optimizer states 74%、expert weights 12%）；(2) MoE 的 forward-backward (F&B) 时间不随 expert 数量成比例增加，但 checkpoint 数据量随 expert 数线性增长——GPU-to-CPU snapshot 无法被 F&B 完全重叠，导致 checkpoint stall；(3) 现有框架 sharding 效率低——Megatron-DeepSpeed baseline 仅用 EP-Group-0 保存，未利用全部 ranks 带宽。总 checkpoint 开销模型：O_ckpt ≈ O_save · I_total / I_ckpt + Σ (O_restart + I_ckpt/2)，优化方向为减小 O_save 和 I_ckpt。

从系统架构角度拆解术语：
标准两阶段异步 checkpoint 流程：

```
# Phase 1: GPU-to-CPU Snapshot (PCIe, 必须与 F&B 重叠)
for each rank in parallel:
    snapshot_thread.start()
    for tensor in model_states:
        cudaMemcpyAsync(cpu_buffer, gpu_tensor)    # 异步复制
    # snapshot 须在 weight update 前完成，否则 stall

# Phase 2: CPU-to-Storage Persist (网络 I/O, 可完全重叠)
for each rank in parallel:
    serialize(cpu_buffer)                            # 序列化 tensor
    write_to_distributed_fs(serialized_data)         # 写入分布式文件系统

# Recovery after fault:
restart_all_nodes()
load_checkpoint_from_storage(latest_completed_ckpt)
resume_training(from_iteration)
```

MoE 场景中 PEC 的优化：snapshot duration ⊥ checkpoint stall origin ∝ (P_ne + K_pec/N · P_e) · (B_w + B_o) / PCIe_BW，减少 K_pec 有效缩短 snapshot 时间使其可被 F&B 完全覆盖。

术语一般如何实现？如何使用？
- 主流框架：Megatron-DeepSpeed、PyTorch FSDP、ByteCheckpoint、DataStates-LLM、CheckFreq、Gemini 等。
- MoE 特化方案：MoC-System (PEC + Fully Sharded + Two-Level)、MoEtion (sparse incremental checkpoint + lightweight recovery log)。
- 使用场景：千卡以上 GPU 集群训练百 B 级 MoE 模型（如 DeepSeek-V2、Mixtral 等），MTBF 可能低至数小时。I_ckpt 典型设置 50-2000 iterations，需平衡 I/O 频率与故障丢失量。

涉及论文标题：
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
