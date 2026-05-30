## Adaptive Offloading (自适应卸载 / 多级 Expert Offloading)

术语是什么？
Adaptive Offloading 是 ES-MoE 提出的动态 offloading 策略选择机制，根据当前训练配置（expert 数量、GPU 内存容量、CPU 内存容量）自动在三种 offloading 模式间切换，以在任何设置下获得最优训练吞吐。三种模式：(1) **GPU-only**：当所有 expert 参数 + optimizer states 能放入 GPU 聚合显存时，不卸载任何参数，但仍享受顺序 expert 计算 + zero-padding 消除的收益；(2) **CPU Offload + Expert Pinning**：expert 参数超出 GPU 显存但能放入 CPU 内存时，卸载到 CPU 并固定 top 25% 最热门的 expert 在 GPU 上以减少 I/O；(3) **CPU+SSD Offload**：当 CPU 内存也不足时，扩展到 SSD，使用 LRU cache + DMA-able pinned memory prefetching。

从系统架构角度拆解：
三种模式的选择逻辑：

```
Algorithm: Adaptive Offloading Decision
Input: model_config (num_experts, expert_size), 
       gpu_mem_capacity, cpu_mem_capacity
Output: offload_mode

total_expert_params = num_experts * expert_size * bytes_per_param(fp16=2)
total_opt_states = num_experts * expert_size * bytes_per_opt_state(fp32=12)
total_model_memory = total_expert_params + total_opt_states + non_expert_memory

if total_model_memory <= aggregate_gpu_memory:
    mode = GPU_ONLY
    # 所有 experts 常驻 GPU，按需顺序计算
    # 获益：更大的 microbatch（无需 offload 通信）
elif total_model_memory <= cpu_memory:
    mode = CPU_OFFLOAD_WITH_PINNING
    # Experts 卸载到 CPU，GPU 固定 top 25% experts
    # Expert pinning: 根据上一 iteration 的 token load 选择热门 expert
    pin_top_n_experts_on_each_gpu(n_pin = 0.25 * experts_per_gpu)
else:
    mode = CPU_SSD_OFFLOAD
    # CPU 内存作为 SSD 的 cache
    # LRU eviction policy + application-level prefetching
    # 利用 training 的确定性访问模式：forward → backward
    # Prefetch 到 DMA-able pinned memory，避免 naive VM 的 page fault stall
```

三种模式的转换点（基于 ES-MoE 论文 MoE-M 实验）：
- ≤ 32 experts (6.6B params): GPU-only mode，吞吐高于 Tutel（更大 microbatch + 零 padding 消除）
- 33-104 experts: CPU offload with pinning mode，expert pinning 贡献 22.8% 吞吐提升
- > 104 experts: CPU+SSD offload mode，SSD offload 使 ES-MoE 可扩展 67× 更多 expert（vs baselines）

术语一般如何实现？如何使用？
- **Expert Pinning**: 保留上一 iteration 中 token 数最多的 top 25% experts 在 GPU，这些 expert 不用重新加载。Token 分布在不同 iteration 间变化缓慢，使 pinning 有效
- **SSD Offloading**: 需要 DMA-able (pinned/non-pageable) memory 作为 CPU↔SSD 传输的中转站；使用 LRU cache 策略管理有限的 CPU 内存窗口
- **Adaptive Switching**: 模式切换在训练配置改变时（如 experiment scaling）自动触发，无需手动干预
- 局限性：GPU-only 模式下无法使用 dynamic expert placement（experts 常驻 GPU 且静态分配）

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
