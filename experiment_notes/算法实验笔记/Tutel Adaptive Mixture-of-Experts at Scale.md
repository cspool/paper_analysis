## Tutel Adaptive Mixture-of-Experts at Scale

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：(1) **自适应并行切换**——通过统一的张量分布布局（ZeRO-DP Stage-3 风格的分片），支持 DP（r=0）和 EP+DP+MP（r=1 到 r=⌈W/E⌉）之间的零成本运行时切换（无参数迁移、无张量重整开销）。通过通信复杂度分析（Table 4）将 7 种并行策略缩减为两种等价覆盖策略。(2) **自适应流水线**——根据 capacity factor f 动态选择最优的流水线度（d∈{1,2,4,8}）和 All-to-All 算法（Linear 或 2DH）。(3) **字典式最优策略查找**——预构建 hash map ⌊c/R⌋ → {r*, d*, a*}，运行时通过 O(1) 查表选择最优并行度和流水线策略。
  - 实验比较：(a) adaptive:r（DP ↔ EP+DP ↔ EP+DP+MP）在不同 capacity factor f（1.0~8.0）下的吞吐量对比（Figure 12），Base (H=2K) 和 Large (H=32K) 两种配置；(b) adaptive pipelining 在 243 种 MoE 模型配置（E_g∈{0.5,1,2}, D∈{1024,2048,4096}, H∈{1024,2048,4096}, tokens/step∈{4096,16384,65536}）下 16~256 GPU 的平均提升（Table 6a/6b）；(c) 单 MoE 层 scaling（16→2048 GPU）的优化逐项叠加分解（Figure 14），从 Fairseq baseline → +Kernel → +2DH A2A → +Flexible A2A → +Adaptive Pipelining → 最终 4.96×/5.75× speedup；(d) SwinV2-MoE 端到端训练/推理速度对比（Table 7, 8~128 GPUs），训练 1.14×~1.55× 加速，推理 1.95×~2.11× 加速；(e) SwinV2-MoE vs 稠密 SwinV2 准确率对比（Table 8）。

- 硬件平台是什么，配置是什么。
  - Azure Standard_ND96amsr_A100_v4 VMs：每 VM 配备 8× NVIDIA A100 SXM 80GB GPU，8× 200 Gbps HDR InfiniBand，96× 2nd-gen AMD Epyc CPU cores，1.9 TiB 内存。节点内 GPU 通过 3rd-gen NVLink + NVSwitch 互联，节点间通过 1,600 Gbps InfiniBand non-blocking 网络（adaptive routing）。实验规模最大 2,048 A100 GPUs (256 VMs)。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) SwinV2-MoE（Swin Transformer V2 的 MoE 版本，每两个 FFN 层替换为 MoE 层，前两个 stage 除外），SwinV2-S（~65.8M active params）和 SwinV2-B（~109.3M active params）两种 size，E=8~128 experts，top-k=1/2，capacity factor f=1.0/1.25；(2) 合成 MoE 配置用于 micro-benchmark：fflayer hidden size H∈{1K,2K,4K,16K,32K}，fflayer channel size D∈{1K,2K,4K}，E_g∈{0.5,1,2} local experts per GPU，tokens/step∈{4096,16384,65536}。
  - 数据集：ImageNet-22K（14.2M images, 22K classes）预训练；ImageNet-1K 微调/5-shot 线性评估；COCO object detection。
  - Benchmarks：ImageNet-22K acc@1、ImageNet-1K acc@1 (ft)、ImageNet-1K 5-shot acc@1、COCO box/mask AP。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/microsoft/tutel，已集成到 Fairseq 和 DeepSpeed。
  - 算法 Pipeline 核心——自适应并行切换的伪代码：

```
# === 自适应并行选择（运行时，O(1) 查表） ===
c = current_capacity_factor * k * T / E     # 当前 expert capacity
key = floor(c / R)                           # R=128 窗口大小
r_opt, d_opt, a_opt = dictionary[key]        # 查表得最优参数

# === Switchable DP (r=0): 等效 ZeRO-DP Stage-3 ===
# forward: all-gather weights across W GPUs
W_sliced = all_gather(W_local, group=range(W))
output = expert_ffn(W_sliced, local_tokens)
# backward: reduce-scatter gradients
grad_W = reduce_scatter(grad_W_local, group=range(W))

# === Switchable EP+DP+MP (r in [1, ceil(W/E)]): ===
# 将 W GPUs 分为 W/(ceil(W/E)/r) 组，组内 DP，组间 MP
group_size = ceil(W/E) / r
# Step 1: LOCAL_REPEAT — 生成 r 份 gating 结果副本
gating_replicated = repeat(gating_result, r)  # shape: (T*r, ...)
# Step 2: All-to-All dispatch (基于 replicated gating)
dispatched = all_to_all(dispatch_input)
# Step 3: Expert FFN 计算
expert_out = expert_ffn(dispatched)
# Step 4: All-to-All combine
combined = all_to_all(expert_out)
# Step 5: LOCAL_SUM — 对 r 份输出求和
output = reduce_sum(combined.reshape(r, T, ...), dim=0)
# Step 6: DP All-Gather（仅在 group_size > 1 时）
if group_size > 1:
    W_sliced = all_gather(W_local, group=groups)
```

  - 自适应流水线核心——Token 分区多流重叠（以 degree=2 为例）：

```
# 输入: (E, C_g, D) 沿 C 维度拆分为 C_0 和 C_1
C_0, C_1 = split(input, dim=C, partitions=2)
# Stream 0: C_0 → A2A_dispatch → Expert_FFN → A2A_combine
# Stream 1: C_1 → A2A_dispatch → Expert_FFN → A2A_combine
# 两流异步执行，A2A (通信流) 与 Expert FFN (计算流) 互相重叠
stream0: A2A_dispatch(C_0) → Expert_FFN(...) → A2A_combine(...)
stream1: A2A_dispatch(C_1) → Expert_FFN(...) → A2A_combine(...)
barrier()  # 等待所有流完成
output = concat([C_0_out, C_1_out], dim=C)
```

  - 字典构建（预搜索，执行一次）：

```
# 对每个 key = floor(c/R):
for r in TernarySearch([1, ceil(W/E)-1]) + [0, ceil(W/E)]:
    for d in {1, 2, 4, 8}:
        for a in {Linear, 2DH}:
            measure throughput(r, d, a, key)
dictionary[key] = argmax_r,d,a(throughput)
```
