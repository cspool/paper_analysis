## Expert Packing (专家打包)

术语解释
Expert Packing 是 Lina 提出的 Training 端优化：当单个 expert 的 FFN 计算时间远小于 All-to-All micro-op 通信时间时，将多个 experts 打包到同一 GPU device 上运行。通过增加每 device 的计算量使 FFN time 对齐 All-to-All time，消除 pipeline bubble，最大化通信-计算重叠效率。

术语是什么？
在 Tensor Partitioning + Pipelining 场景下，FFN micro-op 处理 token subset 的时间往往远小于单次 All-to-All micro-op 的时间（例如 FFN: 0.5ms vs All-to-All: 2.0ms），导致计算流在 All-to-All 期间大量空闲。Expert Packing 通过 2^n 递增每 device expert 数（1→2→4→...），使 FFN total time 接近 All-to-All micro-op time。若 GPU memory 不足则使用 DRAM-offloading 暂存非活跃 expert。

从系统架构角度拆解术语。
```
# Expert Packing 决策流程
def packing_controller(ffn_time, alltoall_time, current_experts_per_device):
    """每10步检查一次"""
    if ffn_time < alltoall_time:
        # FFN too short → increase experts per device
        new_n = min(current_experts_per_device * 2, max_experts_per_device)
        # Init new process groups
        new_pg = create_process_groups(new_experts_per_device=new_n)
        # One-time sync all-to-all to swap expert parameters
        sync_alltoall_exchange_expert_params(new_pg)
        # Multi-stream parallel execution for forward+backward
        enable_multi_stream(new_n)
        return new_n
    return current_experts_per_device

# Pipelining efficiency 对比
# Without packing:  FFN=0.5ms << A2A=2.0ms → efficiency 33%
# With packing (4 experts/dev): FFN total=2.0ms ≈ A2A=2.0ms → efficiency 86%
```

配置规则：
- 起始: 1 expert/device
- 递增: 2^n (2, 4, 8, ...)
- 停止: FFN micro-op time >= All-to-All micro-op time 或 GPU memory 满
- 16-expert models: 通常 2 experts/device 最优（8 devices total），Transformer-XL 使用 4 experts/device
- 调整频率: 每 10 training steps → 稳定后每 4 steps

术语一般如何实现？如何使用？
- Expert Packing Coordinator: MoE 模型中嵌入单线程 controller
- LibTorch 多 stream 并行执行（多 expert forward/backward 在不同 stream）
- DRAM-offloading: ZeRO-Offload 风格，将非活跃 expert param 移至 CPU host memory
- One-time synchronous all-to-all 交换 expert 参数（插入下一次 iteration）
- 效果: Pipeline efficiency 从 33%→86%（Transformer-XL 16-expert）, GPU utilization +17.6%

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---
