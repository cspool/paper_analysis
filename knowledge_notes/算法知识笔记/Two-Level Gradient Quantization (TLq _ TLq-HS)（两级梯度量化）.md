## Two-Level Gradient Quantization (TLq / TLq-HS)（两级梯度量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Level Gradient Quantization (TLq) 是 SDP4Bit 提出的梯度通信压缩策略，针对 ShardedDP 中梯度同步的两阶段通信模式设计。将梯度通信分为两级：(1) **Intra-node（8-bit）**：节点内 all-to-all 通信走 NVLink/NVSwitch 高带宽链路，使用 INT8 量化以保持高精度；(2) **Inter-node（4-bit）**：跨节点 all-to-all 通信走 InfiniBand/Slingshot 相对低带宽链路，使用 INT4 量化大幅压缩通信量。两级间的衔接：intra-node all-to-all 后的数据先做 local reduce（将来自同节点内其他 GPU 的数据归约），再量化到 INT4 进行 inter-node all-to-all。其增强版 TLq-HS 额外在量化前施加 Hadamard Transform（32×32 Walsh-Hadamard 矩阵），将梯度中的 outlier 信息分散到邻近元素，产生更平滑的分布，从而显著降低量化误差。相比于 ZeRO++ 的 Uniform Level quantization (ULq) — 两级均用 4-bit — TLq-HS 用较小的通信开销增加（intra-node 8-bit vs 4-bit）换取了大幅的精度提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 中 TLq-HS 的完整流程（Algorithm 3）：
```
# 输入: grad (FP32, shape [N])
# 输出: g_final (FP32, 归约后的梯度)

# Step 1: Hadamard Transform 平滑
g_hat = H_32 @ grad @ H_32.T     # H_32 ∈ {+1,-1}^{32×32}

# Step 2: INT8 量化 + Intra-node AlltoAll
qg8 = round(clip(g_hat, -s8, s8) / s8 * 127)  # INT8, group=512
list_qg8 = IntraAlltoAll(qg8)    # NVLink/NVSwitch

# Step 3: 反量化 + Local Reduce（省略 Hadamard 逆向 ∵ H·H=I）
g_local = sum([dequantize(x) * s8/127 for x in list_qg8])

# Step 4: Hadamard + INT4 量化 + Inter-node AlltoAll
g_hat2 = H_32 @ g_local @ H_32.T
qg4 = round(clip(g_hat2, -s4, s4) / s4 * 7)  # INT4, group=128
list_qg4 = InterAlltoAll(qg4)    # InfiniBand/Slingshot

# Step 5: 反量化 + Final Reduce + Inverse Hadamard
g_reduced = sum([dequantize(x) * s4/7 for x in list_qg4])
g_final = H_32 @ g_reduced @ H_32.T   # 最终逆变换
```

优化技巧（Section 3.3）：
1. 利用 $H \cdot H = I$ 省略 Step 3 中的 Hadamard 逆向
2. 利用 $\sum_i H g_i = H \sum_i g_i$ 将 Step 5 的 Hadamard 从 dequant 之后移到最终 reduction 之后，使每轮 transform 次数从 6 降至 2

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TLq-HS 的实现基于 Megatron-LM + NCCL 的 all-to-all 集体通信原语：(1) Intra-node all-to-all 通过 `ncclGroupStart/End` 组织 per-rank P2P send/recv；(2) Inter-node all-to-all 同理跨节点 P2P。Hadamard transform 被融合到量化/反量化 CUDA kernel 中（Fused Hadamard Kernel），要求 group_size 能被 H 矩阵大小整除（SDP4Bit 设 H=32×32, group_size=512 intra / 128 inter）。启用参数：`--quantized-gradients --gradient-quantization-bits-intra 8 --gq-group-size-intra 512 --gradient-quantization-bits-inter 4 --gq-group-size-inter 128 --hadamard-transform --gradient-alltoall-pipeline 4`。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- ZeRO++: Extremely Efficient Collective Communication for Giant Model Training

---
