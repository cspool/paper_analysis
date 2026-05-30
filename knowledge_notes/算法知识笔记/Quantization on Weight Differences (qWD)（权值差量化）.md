## Quantization on Weight Differences (qWD)（权值差量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization on Weight Differences (qWD) 是 SDP4Bit 提出的权值通信压缩策略。核心思想：不直接量化当前权值 $w_t$，而是量化两次迭代间的权值差值 $\delta w_t = w_t - w_{t-1}$（即 optimizer update 导致的变化量）。在 ShardedDP 的每轮迭代中，optimizer 更新 main weights 后计算 $\delta w = w_{main} - w_{model}$，对该差值做 INT4 group-wise 对称量化后通过 AllGather 分发，接收方反量化后加到本地 model weights $w_{model}$ 上。其有效性基于：(1) 经验上，差值分布比权值本身更均匀且数值范围更小（Fig. 4 直方图），INT4 量化误差更小；(2) 理论上，差值量化的相对误差 $\frac{\|q(\delta w_t) - \delta w_t\|}{\|w_t\|}$ 小于直接量化权值的误差 $\frac{\|q(w_t) - w_t\|}{\|w_t\|}$；(3) 收敛分析上，差值量化兼容 biased compressor（如 top-k sparsifier）而直接权值量化与 biased compressor 组合会收敛失败（Counterexample 4.1 证明 ternary quantizer 直接量化权重使 SGD 卡在初始值）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 中 qWD 的完整计算流程（Algorithm 2, 结合 Megatron-LM）：
```
# 每轮 iteration:
# 前置状态: w_main[p] (FP32 sharded), w_model (BF16 full, Megatron-LM 维护)

# Step 1: Optimizer 更新 main weights
w_main[p] = AdamW(g_main[p], w_main[p])

# Step 2: 计算权值差值
d[p] = w_main[p] - w_model[p]     # BF16 精度

# Step 3: INT4 group-wise 对称量化差值
# group_size = 2048
for g in range(0, len(d[p]), 2048):
    group = d[p][g:g+2048]
    s_g = max(abs(group))
    d_q[g:g+2048] = round(clip(group, -s_g, s_g) / s_g * 7)

# Step 4: AllGather 量化差值（带宽仅为 BF16 的 1/4）
d_q_global = AllGather({d_q, scales})

# Step 5: 反量化并更新 model weights
for each received shard:
    d_deq[p'] = dequantize(d_q[p'], scales[p'])
    w_model[p'_offset : p'_offset + size] += d_deq[p']

# Step 6: Forward pass 使用更新后的 w_model
output = ForwardPass(w_model, input)
```
与直接权值量化 qW 对比：qW 中 Step 2-5 被替换为直接量化 $w_{main}$ → AllGather → 反量化 → 赋值 $w_{model}$ = deq($w_{main}$)。qWD 的关键区别在于利用 $w_{model}$ 的历史值作为"锚点"，仅传输变化部分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
qWD 的实现依赖训练框架维持完整 model weights 副本——Megatron-LM 的 Distributed Optimizer 模式下天然支持（因不释放 weights），而 ZeRO-3/FSDP 等释放 weights 的框架需要额外适配（需在 backward 前重新 all-gather weights，对 qWD 的实现更复杂）。实现要点：(a) 差值计算需在 optimizer step 后、all-gather 前完成；(b) BF16 精度下的差值计算需注意数值稳定性——如果 $w_{main}$ 是 FP32 需先 cast 到 BF16；(c) group_size=2048 是 SDP4Bit 的默认配置，经 ablation 验证可达全精度训练准确率。SDP4Bit 开源代码在 Megatron-LM 中通过 `--quantized-weights --weight-quantization-bits 4 --wq-group-size 2048` 启用。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

---
