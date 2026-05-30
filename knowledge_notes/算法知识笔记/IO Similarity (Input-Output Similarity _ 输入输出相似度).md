## IO Similarity (Input-Output Similarity / 输入输出相似度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

IO Similarity 是衡量 Transformer 模型中某个模块（layer、attention sublayer 或 FFN sublayer）对前向传播贡献程度重要性的度量。其直观含义：如果某个 Transformer 模块的输入向量经过该模块处理后，输出向量与输入向量高度相似（余弦相似度接近 1），说明该模块对信息几乎没有变换——输入"穿过"该模块后变化极小，因此该模块在当前上下文中的重要性较低，可以被跳过（skip）以节省计算。

数学定义：给定两个 n 维向量 $\vec{a}$（输入）和 $\vec{b}$（输出），余弦相似度为：

$$Similarity(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

AdaSkip 论文通过实验验证了 IO Similarity 与模块重要性之间的强相关性：采用 LeastSkip 策略（跳过 IO Similarity 最低的层，即最重要的层）在仅跳 1 层时 GPT score 即降至 1.0 以下；而 MostSkip 策略（跳过 IO Similarity 最高的层，即最不重要的层）在跳 1/3/5 层时 GPT score 分别为 8.9/6.1/4.2，明显更优。

在长上下文推理中，不同模型（LLaMA3.1-8B-128k、InternLM-7B-8k、Vicuna-v1.5-7B-16k）的 IO Similarity 分布差异极大：InternLM 的高 Similarity 层集中在中部（如 layers 12-14），而 LLaMA3.1 的高 Similarity 层集中在尾部（layers 25-29），且曲线近似单调递增。这种差异性正是 AdaSkip 需要 per-model 自适应学习 IO Similarity 分布的动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

在 AdaSkip 的算法 pipeline 中，IO Similarity 是核心决策依据，贯穿 Prefilling 和 Decoding 两个阶段：

**阶段 1：Offline Importance Learning（Prefilling）**
```
// 在 N 个历史样本上累积各 sublayer 的平均 IO Similarity
for each sample i with |T_i| tokens:
    for each sublayer j in attention_sublayers ∪ ffn_sublayers:  // 共 2M 个
        for each token t in Prefilling phase:
            Simi_j += cosine_sim(a_it^j, b_it^j)  // 输入向量 vs 输出向量

Simi_j = Simi_j / sum(|T_i|)  // 归一化为平均 IO Similarity
```

**阶段 2：Sublayer 排序与选择**
```
sorted = sort_by(Simi_j, descending=True)  // Similarity 越高 → 越不重要 → 越应跳过
m = M - M/α  // 根据加速比 α 确定跳过的 layer 数
skipped = sorted[0:2m]  // 跳过 Similarity 最高的前 2m 个 sublayer
```

**阶段 3：Online Importance Learning（Decoding）**
```
// 用前 P 个 decoded token 计算当前上下文的 IO Similarity
for each FFN sublayer j not in skipped:
    for token t in 1..P:  // online learning window
        Simi_j^P += cosine_sim(a_t^j, b_t^j)
    Simi_j^P /= P

// 用阈值 β 筛选额外可跳过的 FFN
β = min(Simi_j for j in skipped)  // skipped set 中的最低 Similarity
for each FFN j not in skipped:
    if Simi_j^P > β:
        skipped^P += j  // 当前上下文也高 Similarity → 也可跳过
```

**IO Similarity 跨任务泛化性**（关键发现）：
AdaSkip 发现 offline 学习的 IO Similarity 在不同数据集间具有高 hit rate（如 TriviaQA → MFieldQA: ATTN top-10 hit 9.31/10, FFN top-10 hit 9.56/10），说明 IO Similarity 分布是模型内在特性而非任务特定特征。

**IO Similarity 的 Phase 特性**（Observation 3）：
Attention sublayer 和 FFN sublayer 在 prefill 和 decoding 阶段有相似的趋势但波动程度不同。特别是 FFN sublayer 在 decoding 阶段的 IO Similarity 高于 prefill 阶段——这意味着更多 FFN 可在 decoding 阶段额外跳过。

术语一般如何实现？如何使用？

IO Similarity 的计算可通过对推理框架的前向传播进行 hook（拦截）来实现。具体做法：
1. 在 HuggingFace Transformers 的每个 attention 和 FFN 模块前后注册 forward hook，捕获输入和输出 hidden states
2. 推理过程中实时计算 `cosine_sim(a, b) = (a·b)/(|a||b|)`，利用 PyTorch `torch.nn.functional.cosine_similarity` 高效实现
3. 对多 token 的 Similarity 取平均，将 per-sublayer 统计量存储为元数据
4. 后续推理的每个 sublayer 入口处检查是否 ∈ skipped set，如是则执行 identity shortcut
5. 跳过的 sublayer 输出用 Scale_j * a 近似补偿

AdaSkip 开源：https://github.com/ASISys/AdaSkip

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
