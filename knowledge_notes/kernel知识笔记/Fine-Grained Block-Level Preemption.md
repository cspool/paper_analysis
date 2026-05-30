## Fine-Grained Block-Level Preemption

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fine-Grained Block-Level Preemption 是本文提出的概念性 GPU 调度机制，定义为：thread block scheduler 在任意时刻中断**任意一个或一组 thread block** 的执行，保存其 context（register file、shared memory、warp program counter/state），并在之后恢复执行的能力。与 NVIDIA 现有机制的关键区别：(i) Priority Streams/Leftover Policy 完全不支持抢占（只能等 block 自然完成）；(ii) Time-Slicing 支持 coarse-grained 抢占但必须清空整个 GPU；Fine-Grained 可以在 sub-GPU 粒度（单个或多个 SM 的 blocks）上部分抢占，保留其余 blocks 继续执行。(iii) 抢占后的空间可被更高优先级 kernel 使用，实现真正的 GPU spatial-temporal multiplexing。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fine-Grained Preemption 的调度流程（与 Priority Streams 对比）：

```
// 当前状态：training kernel 占据所有 SM
// 事件：inference kernel 到达（高优先级）

// Priority Streams (现有): 等待
while SM has running training_blocks:
    wait()                          // compounded delay!
schedule inference_blocks when space frees

// Fine-Grained Preemption (提出的): 立即抢占
target_blocks = select_blocks_to_preempt(
    num_sms_needed = ceil(inference_kernel.grid_size / blocks_per_sm),
    policy = "least-recently-scheduled"  // 或其他策略
)
for each block in target_blocks:
    save_context(block)             // registers + shared memory → global memory
    mark_sm_slot_available(block.sm)
schedule inference_blocks immediately
// 恢复:
for each preempted block:
    restore_context(block)
    resume_execution(block)
```

抢占成本估算（基于 NVIDIA RTX 3090 参数）：
- Per-SM context: 128KB L1/shared memory + 256KB register file + 64KB constant memory = 448KB
- Per-SM bandwidth: 936 GB/s / 82 SMs ≈ 11.4 GB/s
- Per-SM save time: 448KB / 11.4 GB/s ≈ 37μs
- 基于 time-slicing 实测：145μs 总切换 / 2 ≈ 73μs per save（论文实际测量）

抢占隐藏策略（论文 O8-O9）：
(a) 利用 H2D memory transfer latency 并行执行抢占；
(b) 在小 kernel 执行期间预抢占训练 blocks 为即将到达的大 kernel 腾空间（利用 DL kernel 序列的可预测性）；
(c) 小 kernel 完成后不立即填充训练 blocks，保留空间给后继 kernel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

当前 NVIDIA GPU 硬件不支持 fine-grained block-level preemption，论文建议：(i) 复用现有 time-slicing 的 context-switching 硬件（如 Falcon 微控制器管理的 context save/restore）；(ii) 结合 contention-aware block placement policy 提高 predictability；(iii) 可与 MPS 的 thread limit 机制结合实现 "minimum resource guarantee + priority over-allocation"。论文建议使用 GPU 模拟器 Accel-Sim 进行先期验证（因实际硬件需 NVIDIA 合作修改闭源组件）。相关工作 (Tanasic et al., Park et al.) 曾在模拟器上探索过 context-switching、SM-draining、SM-flushing 等预抢占策略的 trade-off。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
