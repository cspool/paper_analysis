## Test-Time Temporal Sampling (T3S / 测试时时间采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Test-Time Temporal Sampling (T3S) 是一种训练无关、即插即用的 MLLM 视频推理包装器。核心思想：不处理单条长视频 token 序列，而是生成 m 个短且多样化的视频子序列，将它们打包到单次前向传播中并行处理，最后通过 logit 聚合输出最终预测。整个流程分为三个阶段：(1) Multi-Trial Frame Sampling——对视频进行 m 次独立随机帧采样，每次抽取 N 帧；(2) Token Subsampling——每试次保留 αᵢ 比例的 visual token（默认使用均匀随机 patch 级采样）；(3) Multi-Subsequence Inference & Logit Aggregation——将 m 个子序列打包，使用块对角线 attention mask 确保子序列间不互相关注，推理后对各试次 logit 做均值、置信度加权或双试次交叉验证聚合。关键数学性质：self-attention 复杂度从 baseline 的 O(L²) 降为 O(∑αᵢ²L²)，当 ∑αᵢ² < 1 时获得理论加速。m=2、α₁=0.5、α₂=0.3 时 ∑αᵢ²=0.34，理论节省 66%，Qwen2.5-VL-7B 上实测加速 2.04× 同时准确率提升 3.1%（LongVideoBench）。与 FastV、AdaReTake 等 training-free 方法的核心区别：T3S 通过多试次随机采样的统计覆盖性补偿信息损失，而非依赖 attention score 的重要性排序。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === T3S Inference Pipeline ===
# 输入: 视频 V (F帧), 文本 tokens t
# 参数: N=256 (帧数/试次), m=2 (试次数), 
#        α₁=0.5, α₂=0.3 (token保留率), k=2 (top-k)

# Stage 1: Multi-Trial Frame Sampling
for i = 1 to m:
    P_i = RandomSample({1,...,F}, N)   # 随机选N帧索引
    V̂_i = V[P_i]                        # 提取子序列

# Stage 2: Vision Encoding + Token Subsampling
for i = 1 to m:
    v^(i) = VisionEncoder(V̂_i)          # |v^(i)| = L = N×M (每帧M个patch)
    idx = RandomSample({1,...,L}, ⌊αᵢL⌋)  # 均匀随机选token索引
    v̂^(i) = v^(i)[idx]                 # |v̂^(i)| = ⌊αᵢL⌋

# Stage 3: Multi-Subsequence Inference (单次前向传播)
# Pack: concat所有子序列 + block-diagonal attention mask
input_seq = concat(v̂^(1), t, v̂^(2), t)   # 或两个独立序列
{o₁, o₂} = MLLM.forward(input_seq, 
              attn_mask=BlockDiagonal(mask_size=[|v̂^(1)|+|t|, |v̂^(2)|+|t|]))
# 各子序列仅与自身tokens计算attention, 不跨子序列交互

# Stage 4: Logit Aggregation (Two-Trial Cross-Refinement, m=2)
K = TopK(o₁, k=2)                    # 试次1提出top-k候选
t* = argmax_{token∈K} o₂[token]      # 试次2在候选中重新排序

# 若 m>2, 使用均值聚合:
# o_avg = (1/m) Σ oᵢ; t* = argmax o_avg
```

时间复杂度分析：
- Baseline: O(L²) —— 单序列 self-attention
- T3S (m=2, packed): O((α₁²+α₂²)·L²) = O(0.34·L²)
- 实际打包后总序列长度 = (α₁+α₂)L，短于原始 L，进一步降低实际延迟

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/kaibinwang3/T3S。评估使用 VLMEvalKit 工具包，支持 Qwen2.5-VL-7B、LLaVA-Video-7B、Oryx-1.5-7B 等开源 MLLM。使用方式：(1) 加载预训练 MLLM；(2) 用 T3S wrapper 包裹模型推理接口，设置 m、N、αᵢ、k 参数；(3) 对每个视频采样 m 个子序列，调用一次包装后的 forward 获得聚合预测。超参数推荐：m=2（性价比最优，m>2 收益递减），α₁=0.5、α₂=0.3（平衡速度与准确率），k=2（对 k 值不敏感，2-100 范围内波动 <1%）。局限性：(1) 单 GPU 上各 chunk 计算已饱和，无法实现真正的序列级并行；(2) 每步生成 m 个不同 next-token 候选，随生成进行显存占用逐渐增加。代码已开源。

涉及论文标题：
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding
