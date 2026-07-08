## Test-Time Scaling / Parallel Test-Time Compute（测试时计算扩展）

术语是什么？
Test-Time Scaling（亦称 Test-Time Compute、Inference-Time Compute）是一种在推理阶段通过增加计算量来提升 LLM 生成质量的新范式，无需修改模型参数。核心理念：将额外的推理算力（增大 batch size、多次采样、搜索多条生成路径）转化为模型能力的提升。两类典型方法：(1) **Best-of-N (BoN)**：并行生成 N 条完整候选回答，通过 Outcome Reward Model (ORM) 或外部 verifier 评分，选择最高分者；(2) **Beam Search (Step-level)**：逐 step 生成多个候选 token/path，通过 Process Reward Model (PRM) 对中间步骤评分，动态剪枝低质量路径，仅保留 top-k beams 继续探索。所有并行 test-time scaling 方法的共同特征是解码阶段 batch size > 1（多路径同时推进），这使得原本在 batch=1 下浪费的硬件算力得以利用。

从算法 pipeline 角度拆解术语：
Best-of-N 和 Beam Search 的算法流程对比：

```
=== Best-of-N (以 MATH500 数学推理为例) ===
输入: prompt, generation budget N, ORM scorer
输出: 最优回答

1. Parallel Generation:
   for i in range(N):                         # N 条路径并行
     path_i = autoregressive_generate(prompt)  # temperature > 0, 独立采样
     # 每条路径独立 decode 至 EOS 或 max_tokens
     # GPU/NPU 以 batch=N 执行 GEMM + Attention

2. Scoring:
   for i in range(N):
     score_i = ORM(path_i, prompt)  # Outcome Reward Model 评分
     # 如 Skywork-1.5B-PRM: 输入完整回答 + prompt → 标量 reward

3. Selection:
   best_idx = argmax(scores)
   return path_best_idx

=== Step-level Beam Search (以 MATH500 数学推理为例) ===
输入: prompt, beam_width K, PRM scorer
输出: 最优回答

1. Initialization:
   beams = [(prompt, score=0.0)]              # K 个活跃 beam

2. For step t = 1..max_steps:
   # 2a. Expansion
   candidates = []
   for (prefix, score) in beams:
     for k in range(samples_per_beam):        # 每 beam 采样多个 next step
       next_step = generate_one_step(prefix)  # 生成一个推理步骤
       candidates.append((prefix + next_step, score))

   # 2b. Scoring
   for (prefix, _) in candidates:
     step_score = PRM(prefix)                 # Process Reward Model 评分
     candidates[i].score += step_score

   # 2c. Pruning (保留 top-K)
   candidates.sort(key=score, descending=True)
   beams = candidates[:K]

   # 2d. Termination check
   if all(is_complete(beam.prefix) for beam in beams):
     break

3. Selection:
   return beams[0].prefix  # 最高分 beam
```

关键差异：
| 维度 | Best-of-N | Beam Search |
|------|-----------|-------------|
| 并行度 | N 条完整路径并行 | 每步生成后需同步评分 |
| Verifier | ORM (仅最终输出) | PRM (每步评分) |
| 计算特征 | Batch decode (GEMM batch=N) | 逐步扩展+剪枝 |
| 适合场景 | Easy/Medium 问题 | Hard 问题 |
| NPU 适用性 | 高 (大 batch 填充 HMX tile) | 中 (batch 随剪枝减小) |

术语一般如何实现？如何使用？
- 框架：HuggingFace TGI、vLLM 的 `best_of` 参数、llama.cpp 的 `--batch-size` 支持多路径并行采样。论文在 llama.cpp NPU backend 中支持 Best-of-N 和 Beam Search 两种模式。
- Verifier 模型：Skywork-1.5B-PRM（论文使用，340M 参数）、Math-Shepherd（PRM）、RLHF reward models。
- 核心洞察（论文）：Mobile NPU 的 HMX 矩阵单元在常规 decode (batch=1) 下利用率极低（activation tile [1,32]→1/32 有效行）。Test-time scaling 通过增大 batch size 将闲置 HMX 算力转化为生成质量提升，实现"零额外延迟成本"的精度增益。实验表明 Qwen2.5-1.5B + Best-of-N (N=8) 精度超 3B baseline；2.5-3B + Best-of-N 精度超 7B baseline。
- 局限：(1) 需要 verifier 模型（额外内存和计算开销）；(2) BoN 对 easy/medium 问题效果显著，hard 问题收益递减；(3) Beam Search 的 step 同步降低并行度。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

---

## Fine-Grained Group Quantization / W4A16（细粒度分组量化）

术语是什么？
Fine-Grained Group Quantization（细粒度分组量化）是一种 LLM 权重量化方案，将权重矩阵沿 accumulation axis（列维度）划分为大小为 G（通常 G=32 或 128）的连续组，每组内独立计算 scale（和可选的 zero-point），实现 W4A16（权重 4-bit 整数 / 激活 16-bit 浮点）或更低比特的混合精度推理。与 coarse-grained quantization（per-tensor 或 per-channel，一组 scale 覆盖整个 tensor/channel）相比，fine-grained grouping 将量化误差限制在更小的局部区域，显著降低低比特量化的精度损失，是现代 LLM 部署的关键技术。代表方案包括 GPTQ、AWQ、GGUF 的 Q4_K_M 等。

从算法 pipeline 角度拆解术语：
W4A16 Fine-Grained Group Quantization 的推理数据流：

```
模型存储格式 (Q4_0 symmetric, group_size=32):
  权重矩阵 W ∈ R^{K × N}
  对于每 32 个沿列方向的元素: 16 bytes INT4 values + 2 bytes FP16 scale
  Total BPW ≈ 4 + 16/(32×4) ≈ 4.5 bits

推理时反量化 GEMM (Y = X @ W_deq):
  输入: X ∈ FP16 [M, K] (activations)
        W_q ∈ INT4 (packed, 每 32 个一组)
        W_s ∈ FP16 (每组的 scale)

  对 W 的每个 group g:
    w_deq[g*32:(g+1)*32] = W_q[g] * W_s[g]  // INT4→FP16
    　　　　　　　　　　　　　　　　　　　　　 // 沿 K 维度，逐 group 反量化
  对每个 M 行:
    Y[i, :] = Σ_k X[i, k] * w_deq[k, :]      // FP16 GEMM

通用反量化公式 (Q4_0 对称量化):
  w_deq = q_int4 × scale_fp16
  其中 q_int4 ∈ {-8, -7, ..., 6, 7}
  scale_fp16 = max(|W_group|) / 7.0

非对称量化 (IQ4_NL, 论文使用):
  w_deq = LUT[q_int4]  // 非均匀量化表, 16-entry
  # LUT 预学习 (importance-weighted calibration)
  # 每 group 有独立的 16-entry LUT + FP16 scale
```

与 QNN Coarse-Grained Quantization 的精度对比（Table 1）：
| 量化方案 | MATH500 (†) | GSM8K (↑) | Wiki PPL (↓) |
|---------|------------|----------|-------------|
| AutoAWQ (per-group W4A16) | 15.9 | 32.6 | 19.42 |
| QNN (per-channel W4A16) | 2.1 | 3.4 | 28.99 |

Per-channel（QNN）在推理任务上精度崩溃 → test-time scaling 不可行。Per-group 保持可用精度。

术语一般如何实现？如何使用？
- 主流工具：AutoAWQ、AutoGPTQ、llama.cpp 的 `llama-quantize`（GGUF 格式）、ExLlamaV2。
- 论文使用 IQ4_NL+Q8_0 hybrid scheme：attention proj + FFN gate/up → IQ4_NL (4.5 BPW)；FFN down → Q8_0 (8.5 BPW，保留关键层精度)。量化在离线阶段完成，推理时无额外精度开销。
- 关键挑战：(1) 反量化引入 runtime 开销——需仔细设计 kernel 将反量化与 GEMM 融合；(2) Group size 越小精度越高但反量化开销越大——G=32 是常见权衡点；(3) NPU 原生不支持 per-group 量化——论文通过 tile-group quantization 在 NPU 上实现高效 per-group 推理。
- Hybrid 量化策略：混合使用不同比特宽度（4-bit + 8-bit），在精度敏感层（FFN down）使用高精度。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

术语是什么？
Mixture-of-Experts (MoE) 是一种神经网络架构范式，通过将模型中的标准前馈网络（FFN）层替换为多个并行的"专家"子网络（experts），并引入一个可学习的门控网络（gating network / router）来决定每个输入 token 应由哪些专家处理。每个 token 仅激活少数专家（top-k，通常 k=1 或 2），而非所有专家，从而实现条件计算（conditional computation）——模型总参数量极大但每个 token 的实际计算量可控。

核心计算流程：
1. 输入 token 表示 $x \in \mathbb{R}^{d_{model}}$ 传入第 $l$ 层 MoE 层
2. Gating network 计算分数向量 $s = W_g \cdot x$，其中 $W_g \in \mathbb{R}^{N \times d_{model}}$，$N$ 为专家数
3. Top-K 选择：保留分数最高的 $k$ 个专家索引，其余置为 $-\infty$
4. Softmax 归一化：$g = \text{softmax}(\text{TopK}(s))$
5. 专家计算：对每个被选中的专家 $e$，计算 $y_e = \text{FFN}_e(x)$（通常为两层 MLP：$W_{e,2} \cdot \sigma(W_{e,1} \cdot x)$）
6. 加权合并输出：$y = \sum_{e \in \text{TopK}} g_e \cdot y_e$

从算法pipeline角度拆解：
MoE 以 token-to-expert routing 方式工作。以 LLaMA-MoE（32 层，每层 16 专家，top-2 gating）为例：
- 每个 token 进入 MoE 层 → gate 输出 16 维 logits → top-2 选出两个专家 → softmax 得到两专家权重 → 两专家的 FFN 分别计算 → 加权求和得到该层输出
- token 在 32 层中的 forward path：t → layer1(gate→expert_a, expert_b) → layer2(gate→expert_c, expert_d) → ... → layer32(gate→expert_x, expert_y) → output
- 专家专业化：不同专家在不同数据子集上激活，形成自然专业化。如 FLUX 论文 Figure 2 所示，某些专家（如 layer1 的 expert-8）激活频率高（>30%），而其他专家（如 expert-3）激活频率 <5%
- 专家激活模式跨层变化：浅层（如 layer 1）往往高度偏斜（少数专家高频激活），深层（如 layer 31）趋向均匀分布

术语一般如何实现？如何使用？
- 实现框架：HuggingFace Transformers 中的 `MoEModel` 类、Megatron-LM、DeepSpeed-MoE
- 典型 MoE LLM：Mixtral 8x7B（64 层，8 专家/layer，top-2）、DeepSeek-MoE（28 层，64 专家/layer，包含 shared experts + routed experts）、LLaMA-MoE（从 LLaMA 通过 continual pre-training 构建，32 层，16 专家/layer）
- 训练挑战：负载均衡（需 auxiliary load-balancing loss 防止 router collapse）、专家并行（不同专家分布在不同 GPU 上）、通信开销（all-to-all dispatch/combine）
- 推理挑战：所有专家参数需加载到显存（即使只激活少数），是 FLUX 论文要解决的核心问题

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Sparse Activation

术语是什么？
Sparse Activation（稀疏激活）是 MoE 架构的核心特性：对于每个输入 token，仅激活所有可用专家中的一小部分（通常 top-1 或 top-2），而非激活全部专家。这使得模型总参数量可以极大（如 Mixtral 8x7B 的 46.7B 参数），但每个 token 的实际 FLOPs 仅相当于激活的专家子集的计算量（相当于 ~12B 密集模型的计算量）。

从算法pipeline角度拆解：
稀疏激活的具体流程（以 top-2 gating 为例）：
```
输入: token embedding x, 所有 expert 参数 {W_e1, W_e2}_{e=1..N}
1. gate_logits = Linear_gate(x)           # shape: (N,)
2. topk_values, topk_indices = topk(gate_logits, k=2)
3. gate_weights = softmax(topk_values)     # shape: (2,)
4. output = zeros_like(x)
5. for i in [0, 1]:
      e = topk_indices[i]
      h = activation(Linear_e1(x))        # expert e 的 FFN layer1
      y_e = Linear_e2(h)                  # expert e 的 FFN layer2
      output += gate_weights[i] * y_e
6. return output
```
稀疏激活意味着第 5 步的 for 循环只执行 k=2 次（而非 N=16 或 64 次），大幅节省计算。

术语一般如何实现？如何使用？
- 在 MoE 训练中，稀疏激活通过 `torch.gather` 或自定义 CUDA kernel 实现 token-to-expert dispatch
- 关键优化：expert capacity（每个 expert 最多处理的 token 数上限，超出的 token 被 dropped）、load-balancing loss（鼓励 router 均匀分配 token 到各 expert）
- 在 FLUX 论文中，稀疏激活导致的核心问题是 expert activation frequency 高度不均（部分 expert 激活频率 <5%），影响 fine-tuning 效率——FLUX 通过 profiling 识别激活模式并据此决定 expert merging 和 role assignment

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Expert Merging

术语是什么？
Expert Merging（专家合并）是将 MoE 模型中多个功能相似的 expert 参数合并为更少的合并专家（merged expert）的技术。目的是在保留模型关键行为的同时减少专家数量，降低显存占用和计算开销。

FLUX 的合并策略包含三个层次：
1. **Adaptive Layer Size（§5.1）**：不同层的合并预算 $B_i^{non}(l)$ 不同。公式：$B_i^{non}(l) = \lfloor \frac{b_i^l}{\sum_k b_i^l} \times B_i^{non} \rfloor$，其中 $b_i^l = \frac{L-l+1}{v_i^l}$。$v_i^l$ 是层 l 的 expert 激活频率方差，$L-l+1$ 使浅层获得更大预算（因误差在浅层传播累积更大）。
2. **Similarity-based Clustering（§5.2）**：PCA 降维 expert 参数 → 跨层融合 K-Means 聚类（40× 加速 vs 逐层独立聚类） → 同 cluster 内的 expert 合并。
3. **Importance-based Weighted Merging（§5.3）**：$W_{merged} = \sum_{e \in E_c} \frac{\alpha_e}{\sum_{e' \in E_c} \alpha_{e'}} W_e$，其中 $\alpha_e = f_e \cdot \bar{a}_e$（激活频率 $f_e$ × 平均 attention score $\bar{a}_e$）。

从算法pipeline角度拆解：
Expert merging 在每个 training round 的 forward pass 前执行：
```
1. 量化模型估计 f_e, ā_e
2. 对 non-tuning experts:
   a. 每层计算 B_i^{non}(l) （公式1，考虑 variance 和层深度）
   b. PCA(W_e) → low-dim features
   c. 跨层 K-Means → clusters（所有层 experts 一起聚类但标注层标签）
   d. 每 cluster 内加权合并（公式2，α_e = f_e × ā_e）
   e. Gate re-routing: 更新 gating network expert index mapping
3. 合并后的 merged experts 冻结（不参与 backprop），仅参与 forward pass
```

术语一般如何实现？如何使用？
- 通用 expert merging 工具：mergekit（用于合并 fine-tuned LLM）、MC-MoE（Merge then Compress）
- FLUX 实现 API：`Flux.moe.customized_moe(model, exps_config)` 构建每层不同专家数的 MoE
- 关键细节：(1) 合并后需做 gate re-routing；(2) attention score 作为重要性权重比纯频率更准确（Figure 17，额外减少 19.2% output error）；(3) 跨层融合聚类消除重复 centroid 初始化，40× 加速
- 使用场景：资源受限的 MoE 推理/训练、模型压缩、边缘部署

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Expert Activation Profiling

术语是什么？
Expert Activation Profiling（专家激活分析）是在 MoE 模型训练或推理前，通过运行模型 forward pass 来测量每个 expert 的激活统计信息的过程。核心统计量包括：激活频率 $f_e$（流经 expert e 的 token 数 / 总 token 数）、每个 expert 处理的 token 子集 $D_i^e$、以及流经 expert e 的 token 的平均 attention score $\bar{a}_e$。

FLUX 提出 **Quantization-based Profiling**：使用 INT4 量化的 MoE 模型替代全精度模型做 profiling——量化模型的激活模式与全精度模型高度近似（4-bit 估计误差约 11.01%），但计算开销大幅降低。此外，**Stale Profiling** 机制使 profiling 与 parameter aggregation 并行执行，使用上一轮的 stale profile 做 merging，同时并发量化+profiling 最新模型。

从算法pipeline角度拆解：
```
Round r（并发执行）:
  Thread A: Parameter Server Aggregation
  Thread B: Quantize(w^{r-1}) → Forward on local data → Profile(f_e, D_i^e, ā_e)

Round r+1:
  使用 Round r 的 profile（stale by 1 round）做 Expert Merging
  误差增长 <2%，round time 减少 28.2%
```

术语一般如何实现？如何使用？
- 通用实现：hook gating network 的 top-k indices，统计每个 expert 被选中的次数
- FLUX：支持 INT4/INT8 等不同量化级别，participant 可根据算力灵活选择
- Profiling 结果用途：(1) 决定 tuning expert 的训练数据；(2) 计算 merging 权重；(3) 初始化 expert utility（第一轮）
- 关键考量：activation pattern 随时间变化，profiling 不能是一次性的

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Expert Role Assignment

术语是什么？
Expert Role Assignment（专家角色分配）是在资源受限的 MoE fine-tuning 中，决定每个 expert 在每轮训练中是"调优专家（tuning expert，完整 FP32 更新）"还是"非调优专家（non-tuning expert，合并或冻结）"的优化问题。

FLUX 的角色分配包含三个关键设计：
1. **Expert Utility 定义**：$u_i^e = |D_i^e| \sqrt{\frac{1}{|D_i^e|} \sum_{k \in D_i^e} \nabla g_k}$。结合 data utilization（$|D_i^e|$）和 gradient magnitude。
2. **Exploration-Exploitation**：ε 比例选最高 utility expert（exploitation），(1-ε) 比例随机选（exploration），ε 动态增长。
3. **全局优化**：$\max \sum_i \sum_e x_i^e u_i^e$，s.t. $\sum_e x_i^e \leq B_i^{tune}, \forall i$。Server 求解后下发。

从算法pipeline角度拆解：
```
Server-side (每轮):
1. 收集所有 participant 上报的 {u_i^e}
2. 求解优化问题 → 候选集 E_i
3. E_i^{exp} = top ε|E_i| experts by utility（exploitation, backprop 训练）
4. E_i^{exl} = random (1-ε)|E_i| experts（exploration, forward-only 梯度估计）
5. 下发 E_i^{exp}, E_i^{exl} 到各 participant
6. ε 动态调整：ε_t = min(ε_max, ε_0 + Δε × t)
```

术语一般如何实现？如何使用？
- FLUX 实现：角色分配逻辑在 parameter server 端运行
- 第一轮用 Norm(activation frequency) 初始化 utility
- Exploration experts 使用 forward-only gradient estimation 省 backprop
- 使用场景：联邦 MoE fine-tuning、异构 participant 资源下的自适应训练

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Gating Network (Router)

术语是什么？
Gating Network（门控网络，亦称 Router）是 MoE 层中的核心组件，负责为每个输入 token 选择应激活哪些专家。典型实现为一个线性层 $W_g \in \mathbb{R}^{N \times d_{model}}$，将 token 表示映射为 N 维 logits，经 top-k 选择 + softmax 得到各专家的组合权重。

从算法pipeline角度拆解：
```
输入: x ∈ R^{d_model}
1. logits = W_g @ x              # (N,) — 每个专家的原始分数
2. topk_vals, topk_idx = topk(logits, k)  # 选 k 个最高分
3. weights = softmax(topk_vals)   # (k,) — 归一化权重
4. for each selected expert e:
      y_e = Expert_e(x)           # FFN forward
5. output = Σ weights[i] * y_{topk_idx[i]}
```

术语一般如何实现？如何使用？
- PyTorch 实现：`nn.Linear(d_model, num_experts)` + `torch.topk` + `F.softmax`
- 在 FLUX 中，gating network 在 expert merging 后需做 **gate re-routing**——因某些 expert 被合并，需更新 gating 输出到合并后 expert 的映射
- Gating 网络参数量很小（d_model × N，通常 < 0.1% 总参数），但设计直接影响模型质量和训练稳定性
- 关键设计：Top-K（k=1/2）、Load Balancing（auxiliary loss 或 dynamic bias）、Capacity Factor

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Forward-only Gradient Estimation

术语是什么？
Forward-only Gradient Estimation（仅前向梯度估计，亦称 Zeroth-Order Optimization）是一种不通过反向传播计算梯度的技术。核心思想：对参数施加微小随机扰动 $\xi$，通过两次前向传播的 loss 差商来近似梯度：$\hat{\nabla} \approx \frac{L(W + \xi) - L(W)}{\xi}$。

FLUX 将此技术用于 exploration experts——仅需评估 utility 而非实际更新参数，无需精确 backprop 梯度。

从算法pipeline角度拆解：
```
输入: expert e 的参数 W_e, batch data, 扰动尺度 σ
1. ξ ~ N(0, σ²)
2. loss_plus = forward(W_e + ξ, batch)
3. loss_orig = forward(W_e, batch)
4. ∇̂_e = (loss_plus - loss_orig) / ξ
5. u_i^e = |D_i^e| * sqrt(avg(∇̂_e))  （代入 utility 公式）
```

术语一般如何实现？如何使用？
- 其他名称：Zeroth-Order Optimization (ZOO)、Evolutionary Strategies (ES)
- 相关工作：BAFFLE (Backpropagation-Free Federated Learning)
- 优势：省 backprop 显存和计算（forward 比 forward+backward 快约 2×）
- 局限：梯度估计有噪声，收敛慢于精确一阶方法——FLUX 仅在 exploration experts 上使用
- FLUX 实验：估计梯度与 ground truth 的平均 cosine distance 约 0.29

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

---

## Exploration-Exploitation Strategy

术语是什么？
Exploration-Exploitation Strategy（探索-利用策略）是强化学习和在线优化中的经典决策框架。Exploitation = 利用当前已知最优选择最大化即时收益；Exploration = 尝试未充分了解的选项以获取更多信息。在 FLUX 中用于 expert role assignment：exploitation 选高 utility experts 做 tuning，exploration 随机选 experts 更新其 utility 估计。

从算法pipeline角度拆解：
```
每轮:
1. 候选集 E_i（解优化问题 公式4 得到）
2. Exploitation: E_i^{exp} = top ε|E_i| experts by utility
3. Exploration: E_i^{exl} = random (1-ε)|E_i| experts
4. 动态 ε: ε_t = min(ε_max, ε_0 + Δε × t)
   - 早期 ε 小（多探索，utility 估计不可靠）
   - 后期 ε 大（多利用，utility 估计趋于准确）
```

术语一般如何实现？如何使用？
- 经典实现：ε-Greedy、UCB、Thompson Sampling
- FLUX：ε-Greedy 变体，ε 随时间动态递减探索比例
- 联邦学习中的应用：Oort 在 participant selection 中使用类似策略
- FLUX 创新：在 expert 粒度而非 participant 粒度应用，与 MoE 稀疏激活特性结合
- 实验效果（Figure 19）：动态 ε 比固定 ε=0.3（不稳定）或 ε=0.7（收敛慢）更快

涉及论文标题：
- Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices

## Cross-Encoder Reranker

术语是什么？
Cross-Encoder Reranker（交叉编码器重排序器）是一种用于语义排序任务的 Transformer 模型架构。与 Bi-Encoder 将 query 和候选文档分别独立编码为固定向量不同，Cross-Encoder 将 query 和每个候选文档拼接为一个联合输入序列 `[CLS] query [SEP] candidate [SEP]`，通过多层 Transformer 的 self-attention 机制实现 query 与候选文档之间的 token 级深度交互，最后通过一个轻量 classifier head（作用于最后一层 hidden state 的最后一个 token 或 [CLS] token）输出一个标量 relevance score。

核心架构分为两类主流变体：
1. **Encoder-only**（如 BERT-style, Bge-Reranker-v2-M3）：双向 self-attention，所有 token 互相可见，适合理解 query-document 对的充分语义关系。
2. **Decoder-only**（如 GPT-style, Qwen3-Reranker 系列）：因果 self-attention（causal mask），仅左侧 token 对右侧 token 可见，适配 instruction-following 和 reasoning-intensive 任务。

从算法pipeline角度拆解：
Cross-Encoder Reranker 的推理 pipeline（以 Qwen3-Reranker-0.6B decoder-only 为例，单个 query-candidate pair）：

```
输入: query q (tokens), candidate d (tokens)
1. 拼接: input = [BOS] q [SEP] d [EOS]
   # tokenized: input_ids ∈ Z^{L}, L ≤ 512 (典型 reranker 的 max_length)

2. Embedding: h_0 = Embedding(input_ids)  # [L, D]
   # D = hidden_dim（如 0.6B 模型 D≈1024）

3. Transformer Layers (0..27):
   for layer_i in range(num_layers):
       # 3a. Self-Attention (causal for decoder-only)
       Q, K, V = h @ W_Q, h @ W_K, h @ W_V    # [L, D]
       attn_scores = Q @ K^T / sqrt(d_head)     # [L, L], causal masked
       attn_weights = softmax(attn_scores)       # [L, L]
       h_attn = attn_weights @ V                # [L, L] × [L, D] = [L, D]

       # 3b. FFN (SwiGLU or standard MLP)
       h_ffn = Activation(h_attn @ W_up) * (h_attn @ W_gate)
       h = h + h_ffn @ W_down                   # [L, D], residual

       # 3c. RMSNorm / LayerNorm (omitted for brevity)

4. Score Head:
   score = classifier_head(h[-1, :])  # 取 last-token hidden state
   # 或对于 encoder-only: score = classifier_head(h[0, :])（[CLS] token）
   # score ∈ R — 单个标量 relevance score

5. 对 N 个候选文档重复步骤 1-4（传统方式），或 monolithic forwarding 一次处理所有（PRISM 方式）
```

关键复杂度：
- Self-Attention: $O(L^2 \cdot D)$（causal 减半为 $O(L^2 \cdot D/2)$）
- FFN: $O(L \cdot D^2)$
- 总延迟 ∝ N（候选数）× L（层数），传统方式下每候选需独立完整前向

术语一般如何实现？如何使用？
- 主流框架：HuggingFace Transformers 中的 `AutoModelForSequenceClassification`，加载如 `BAAI/bge-reranker-v2-m3`、`Qwen/Qwen3-Reranker-0.6B` 等模型
- 使用方式：`model(input_ids, attention_mask).logits` → scalar score per pair
- 典型场景：RAG pipeline 的重排序阶段（retrieval → reranking → generation）、Agent Memory 的轨迹选择、长上下文相关性选择
- 精度优势：vs Bi-Encoder 提升 15-25% retrieval precision（BEIR benchmark），因为 token 级交互捕获了 Bi-Encoder 独立编码所丢失的细粒度语义
- 代价：latency ∝ N，每候选需完整前向（0.6B 模型处理 20 候选→top-5 需 5,754 ms on Mac Mini）

涉及论文标题：
- On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

---

## Bi-Encoder

术语是什么？
Bi-Encoder（双编码器）是一种将 query 和候选文档分别通过独立编码器（通常共享权重或镜像架构）映射为固定维度 embedding 向量的检索架构。query 编码为向量 $e_q \in \mathbb{R}^d$，每个候选文档独立编码为向量 $e_d \in \mathbb{R}^d$，最终通过余弦相似度或点积计算相关性分数：$score(q, d) = \cos(e_q, e_d)$ 或 $e_q^T e_d$。代表性模型包括 DPR (Dense Passage Retrieval)、SBERT、E5、BGE-Embedding 等。

从算法pipeline角度拆解：
Bi-Encoder 的检索 pipeline：
```
离线阶段（Indexing）:
  for each document d in corpus:
      e_d = Encoder(d)  # [d] 维向量，仅编码一次
  存储所有 e_d 到向量数据库（如 FAISS、Milvus）

在线阶段（Retrieval）:
  e_q = Encoder(q)                             # [d] 维，单次编码
  scores = {cos(e_q, e_d) for all d in corpus}  # 近似最近邻搜索（ANN）
  top_k = argsort(scores)[:k]                   # 选 top-k
```

关键特性：
- query 和候选文档独立编码，无 token 级交互 → 速度快但精度受限
- 候选文档 embedding 可预计算并索引 → 适合大规模语料库（百万级+）
- 复杂度：$O(1)$ encoder forward（query 一次）+ $O(\log N)$ ANN 搜索 → 远低于 Cross-Encoder 的 $O(N)$ forward passes

术语一般如何实现？如何使用？
- 主流实现：Sentence-Transformers 库（`SentenceTransformer`）、HuggingFace `AutoModel` + mean pooling
- 向量数据库：FAISS（Meta，IVF/HNSW 索引）、Milvus（DiskANN-based）、Chroma
- 典型使用：RAG 的 first-stage retrieval（粗筛），从海量文档中快速召回 top-N（N=20-100）候选，再送 Cross-Encoder 做 fine-grained reranking → top-K（K=1-10）
- 精度劣势：vs Cross-Encoder 低 15-25%（BEIR benchmark），因为无法捕获 query-document 的 token 级细微语义交互
- PRISM 论文中的角色：作为 Cross-Encoder 的对比 baseline——Bi-Encoder 速度快但精度不足，Cross-Encoder 精度高但延迟大，PRISM 解决 Cross-Encoder 的延迟问题

涉及论文标题：
- On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

---

## Top-K Semantic Selection

术语是什么？
Top-K Semantic Selection（语义 Top-K 选择）是从 N 个候选项目中识别语义上最相关的前 K 个项目的任务。它是现代 on-device AI 服务的核心组件，广泛出现在 RAG（检索增强生成）、AI Agent Memory（智能体记忆）、个性化推荐等场景中。Top-K 选择的关键特性是：**只需相对排名（relative ranking），而非精确绝对分数（absolute scores）**——只要能正确区分 top-K 内和 top-K 外的候选即可。

从算法pipeline角度拆解：
典型的 on-device semantic selection pipeline（以 RAG 为例）：
```
Stage 1: Coarse Retrieval（粗筛）
  - Sparse Retrieval: 关键词搜索（如 BM25）→ top-m_sparse 候选
  - Dense Retrieval: 向量搜索（Bi-Encoder + ANN）→ top-m_dense 候选
  - 合并去重: N = m_sparse + m_dense 候选（通常 N=10-30）

Stage 2: Fine-grained Reranking（精排）← Top-K Selection 核心
  - Cross-Encoder Reranker 对 N 个候选打分
  - 选出 top-K 最相关候选（K = 1-10）
  - 本阶段贡献端到端延迟的 >95%

Stage 3: Downstream Task（下游任务）
  - Top-K 候选 + query → 拼接为 prompt → LLM 生成回答
  - 或直接返回 top-K 结果给用户/UI agent
```

关键观察（PRISM 论文）：top-K 选择只需相对排名，不需要为每个候选计算精确的绝对分数。这一 insight 催生了 PRISM 的 progressive cluster pruning 策略——只要能在中间层区分哪些候选属于 top-K、哪些不属于，就可以提前剪枝，无需完成全部层的计算。

术语一般如何实现？如何使用？
- 实现 pipeline：Hybrid Search (sparse + dense) → Cross-Encoder Reranker → LLM
- 评估指标：Precision@K = |top-K 中相关项| / min(K, |ground truth|)；NDCG@K；MRR
- 关键权衡：K 越小对 reranker 精度要求越高（选错代价大），K 越大 latency 压力越大（更多候选需精确排序）
- PRISM 的贡献：识别到 top-K 选择的"相对排名"特性，用 sequence-level sparsity + progressive cluster pruning 将 latency 降低 up to 89.2%

涉及论文标题：
- On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

---

## Sequence-Level Sparsity

术语是什么？
Sequence-Level Sparsity（序列级稀疏性）是 PRISM 论文发现的一种 cross-encoder reranker 特有的推理特性：当所有候选文档作为一个整体（monolithic batch）通过 Transformer 各层时，候选文档的 relevance scores 在中间层即逐步分化为统计显著的聚类（clusters），**聚类间的相对排名（inter-cluster rankings）在中间层就趋于稳定并与最终排名一致**，而聚类内的排名（intra-cluster rankings）仍在持续变化。这种序列级别的稀疏性意味着：无需完成全部层的计算即可确定哪些候选文档已稳居 top-K 或已确定出局。

验证方法：
- **Goodman and Kruskal's Gamma ($\gamma$)**：衡量中间层排名与最终排名的一致性。$\gamma = \frac{N_c - N_d}{N_c + N_d}$，其中 $N_c$ 为中间层与最终层排名一致的候选对数量，$N_d$ 为排名反转的对数。
- **Cluster Gamma**：仅计算**不同聚类间**候选对的排名一致性。论文在 18 个数据集上验证，cluster $\gamma$ 在所有层始终接近 1.0，证明聚类间排名早在中间层即稳定。

从算法pipeline角度拆解：
Sequence-level sparsity 的观察和利用流程：
```
1. 将所有 N 个候选的 scores 在每个 Transformer layer 后记录下来
2. 观察 scores 跨层的演化：
   - 早期层（layer 0-5）：所有候选 scores 聚集在一起，无明显分化
   - 中间层（layer 6-15）：scores 开始分化为 3-5 个统计显著的聚类
     - 聚类间相对排名已稳定（与最终层一致）
     - 聚类内排名仍在波动
   - 后期层（layer 16-27）：聚类进一步细化为更小的子聚类

3. 利用：在中间层识别聚类 → 剪枝已确定进/出 top-K 的聚类 → 仅对边界聚类继续计算

归因：Transformer 的 coarse-to-fine understanding 特性
  - 浅层：捕获广泛语义特征，区分明显相关的候选
  - 深层：细化细微语义差异，区分高度相似的候选
```

术语一般如何实现？如何使用？
- PRISM 利用此特性实现 Progressive Cluster Pruning：每层用 CV（变异系数）衡量 score dispersion → 超过 threshold 时触发 K-Means 聚类 → 识别 boundary cluster → 三路路由（selected/dropped/deferred）
- 通用性：论文在 18 个数据集和所有主流模型架构（encoder-only 和 decoder-only）上验证了此特性
- 区别于 token-level sparsity：token-level sparsity（如 LazyLLM 的 token pruning）利用长序列中大量 token 的信息冗余，对 reranker 的短而信息密集的输入效果不佳；sequence-level sparsity 利用的是候选间的相对排名关系，与序列长度无关

涉及论文标题：
- On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

---

## Progressive Cluster Pruning

术语是什么？
Progressive Cluster Pruning（渐进式聚类剪枝）是 PRISM 系统中利用 sequence-level sparsity 实现的训练无关（training-free）候选剪枝算法。它在每一层 Transformer 执行后，用 model 的原始 classifier head 计算所有候选的当前 relevance scores，通过变异系数（Coefficient of Variation, CV）判断排名是否已稳定分化，若稳定则触发 K-Means 聚类，将候选文档三路路由：selected（已稳居 top-K，停止计算）、dropped（已出局，停止计算）、deferred（处于边界，继续下一层）。该算法将传统的"N 个候选 × L 层的全量计算"转变为"逐层递减候选数的自适应计算"。

从算法pipeline角度拆解：

```
输入: N 个候选文档, target K, dispersion_threshold θ
输出: top-K 候选文档

final_selected = []               # 已确认的 top-K
remaining = all N candidates      # 待继续处理的候选

for layer_i in range(num_layers):
    # 1. 前向传播（monolithic batch，仅 remaining candidates）
    hidden_states = Transformer_Layer_i(remaining)  # chunked execution 内

    # 2. 计算当前 scores
    scores = classifier_head(hidden_states[:, last_token, :])  # [num_remaining]
    
    # 3. 计算变异系数
    cv = abs(std(scores) / mean(scores))
    
    # 4. 判断是否触发剪枝
    if cv > θ:  # 排名已稳定分化
        # 4a. K-Means 聚类（CPU, ~1ms）
        n_clusters = auto_detect(scores)  # 或固定 k=3-5
        clusters = KMeans(scores.reshape(-1, 1), n_clusters=n_clusters)
        
        # 4b. 按 cluster mean score 降序排列
        sorted_clusters = sort_by_mean_score(clusters)
        
        # 4c. 找 boundary cluster（包含第 K-th ranked candidate）
        cum_count = 0
        for idx, cluster in enumerate(sorted_clusters):
            cum_count += len(cluster)
            if cum_count >= K:
                boundary_idx = idx
                break
        
        # 4d. 三路路由
        selected = flatten(sorted_clusters[:boundary_idx])
        deferred = sorted_clusters[boundary_idx]
        dropped = flatten(sorted_clusters[boundary_idx+1:])
        
        final_selected.extend(selected)
        remaining = deferred  # 仅边界簇继续
        
        # 4e. 提前终止条件
        if len(final_selected) + len(deferred) <= K:
            final_selected.extend(deferred)
            return final_selected[:K]  # 提前终止！
    else:
        # CV ≤ θ，排名未分化，继续下一层
        continue

# 最终层：余下候选的排名自然确定
return final_selected + sort_by_score(remaining)[:K - len(final_selected)]
```

Dispersion Threshold ($\theta$) 的自动校准：
```
每 N 个请求采样一次：
  sampled_result = PRISM(sampled_request, current_θ)
  在设备空闲时重跑完整推理（无剪枝）→ ground_truth
  precision = |sampled_result ∩ ground_truth| / K
  if precision < target_precision:
      θ = θ + Δ  # 提高 threshold → 更保守 → 更高精度
  else:
      θ = θ - Δ  # 降低 threshold → 更激进 → 更低延迟
```

术语一般如何实现？如何使用？
- 实现：PRISM 在 HuggingFace Transformers 的每层 forward 后插入 pruning check，K-Means 使用 `sklearn.cluster.KMeans` 在 CPU 上运行（~1ms）
- 超参数：dispersion threshold $\theta$ 是唯一的 tuning knob。论文提供两种模式：手动指定或自动校准（指定 target precision）
- 关键设计选择：
  - 聚类在 CPU 而非 GPU 上执行——GPU 忙于下一层的 chunked execution
  - classifier head 复用原始模型权重——无需额外训练或校准
  - 剪枝在 cluster 粒度而非 candidate 粒度——因为单个 candidate 的排名不稳定，而 cluster 间排名已稳定
- 效果：单独贡献 49.0% latency reduction（ablation study），综合系统达 up to 89.2%

涉及论文标题：
- On-device Semantic Selection Made Low Latency and Memory Efficient with Monolithic Forwarding

---

## IndexSoftmax

术语是什么？

IndexSoftmax 是一种训练无关（training-free）的全整数 softmax 替代算子，专为 ARM CPU 上 INT8 量化的 Transformer 注意力设计。它完全消除浮点 softmax 路径中的 dequantize → softmax → requantize 瓶颈，通过三个纯整数组件实现：稀疏感知裁剪（Sparsity-aware Clipping）、基于查找表的指数函数（LUT Exponential）和整数归一化（Integer Normalization）。整个数据流为 S32 → U8（int32 logits 输入，uint8 概率输出），无需任何浮点类型转换。

IndexSoftmax 的核心设计动机来自一个观察：在 INT8 GEMM 加速 QK^T 和 PV 矩阵乘法后，中间 softmax 路径（dequantize → FP32 exp/sum/div → requantize）在注意力延迟中的占比可高达 65%（RK3588S2 ARM CPU 实测）。因为 INT8 矩阵乘的速度提升了 2× 以上，但 softmax 仍停滞在浮点域，成为新的瓶颈。

三个组件的协同设计：
1. **Clipping**：softmax 由 logits 中接近最大值的元素主导，远离最大值的 logits 贡献可忽略。对 int32 logits 做 max-subtraction 后裁剪至 [0, c=6.6] 范围，跳过大量无效指数运算。裁剪同时使指数输入域有界，这是 LUT 替代指数函数的前提。
2. **LUT Exponential**：利用裁剪后的有界指数域 [0, c=6.6]，使用 32-entry UINT8 查表（每 entry 1 字节，总 32 bytes）近似 exp 函数。LUT[i] = round(exp(-i/2^b) × 255)，b=5。32 bytes 的 LUT 可完全驻留在 NEON 寄存器中，查表通过 TBL 指令实现 16 路并行。
3. **Integer Normalization**：将 P 矩阵（softmax 输出概率）表示为 UINT8 [0, 255] 而非 INT8 [-128, 127]。UINT8 在相同 32-byte 表预算下利用全部 256 个值（vs INT8 仅用一半非负值），精度 4× 更高。归一化使用定点乘除 `(prob * 255 + sum/2) / sum`，无浮点除法。

从算法pipeline角度拆解术语：

在标准 Transformer 自注意力的算法 pipeline 中（S = QK^T/√d → P = softmax(S) → O = PV），IndexSoftmax 替换了 softmax 步骤。整个注意力层的算法流程变为：

```
输入：INT8 Q, K, V（S8 格式），量化 scale

=== QK^T 累积（标准 INT8 GEMM） ===
S_int32 = INT8_GEMM(Q_int8, K_int8)  // S8×S8→S32

=== IndexSoftmax 替换 FP32 Softmax ===
对 S_int32 的每一行 s：

  步骤 1 - Clipping:
    s_max = max(s)                          // 行内最大值
    c_int = round(6.6 / scale)              // c 转为整数阈值
    mask = (s - s_max > -c_int)             // 仅保留有效 logits

  步骤 2 - LUT Exponential:
    LUT[0..31] = {round(exp(-i/32) × 255)} // 32-entry UINT8
    idx = clamp(round(-(s - s_max) × 32 / c_int), 0, 31)
    prob = LUT[idx]                         // UINT8, [0, 255]

  步骤 3 - Integer Normalization:
    sum_prob = sum(prob)
    P_row = (prob * 255 + sum_prob/2) / sum_prob  // 定点归一化
    // 裁剪元素: P_row[j] = 0

输出：P_uint8 ∈ [0, 255]^{L×L}

=== PV 混合 ===
O_int32 = INT8_GEMM(P_uint8, V_int8)  // U8×S8→S32
```

与 FP32 softmax 的关键差异：
| 维度 | FP32 Softmax | IndexSoftmax |
|------|-------------|-------------|
| 输入 | FP32 | INT32（S32 GEMM 累加器） |
| 指数 | expf()（标量，12-20 cycles/elem） | LUT TBL 查表（<1 cycle/elem, 16 路并行） |
| 中间格式 | FP32 | UINT8 |
| 归一化 | FP32 除法 | 定点乘除 |
| 输出 | FP32（需 requantize 回 INT8） | UINT8（直接供下游 INT8 GEMM） |
| 类型转换 | 2 次（dequant + requant） | 0 次 |
| 缓存往返 | 3 次 | 1 次（融合 kernel） |

术语一般如何实现？如何使用？

IntAttention 论文在 Arm Compute Library (ACL) 中实现了 IndexSoftmax：
1. **ACL kernel 实现**（`add_impl_for_ACL.patch`）：在 ACL 中新增 `NEIndexSoftmax` 函数，Clipping 使用 NEArithmeticOps + NEComparison，LUT 通过 NEON `TBL` 指令并行查表，Normalization 使用 NEON `umull` + `ushr` 定点乘除。融合为单个 kernel。
2. **PyTorch 模拟**（`pysimulation/`）：使用 FP32 算术模拟整数行为，用于 GPU 精度验证。
3. **使用方式**：作为 INT8 注意力 pipeline 的 drop-in 替换。c=6.6, b=5 为全局默认超参数，无需模型特定校准。适用条件：Q/K/V 已 INT8 量化、QK^T GEMM 输出 S32、目标平台 ARM CPU。

涉及论文标题：
- IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

## Fully Integer Attention Pipeline（全整数注意力流水线）

术语是什么？

全整数注意力流水线是指 Transformer 注意力层从 QK^T 矩阵乘法到 PV 矩阵乘法的整个概率计算路径均在整数域中完成，消除所有浮点类型转换的推理方案。其核心数据流为 `S8×S8 → S32 → U8 → S8×S8 → S32`：Q/K 以 INT8 格式做 GEMM 产生 INT32 logits；logits 经过全整数 softmax 替代算子输出 UINT8 概率；概率与 INT8 V 做 GEMM 产生 INT32 输出。全程无 S32↔FP32 或 FP32↔S8 转换。

在传统"Quant-Only"方案中，仅矩阵乘法做了 INT8 加速，中间 softmax 仍为浮点（dequantize → softmax → requantize）。随着 INT8 GEMM 将矩阵乘加速 2×+，softmax 路径成为占比 ≤65% 的新瓶颈。全整数流水线通过消除此瓶颈，在 RK3588S2 上实现 vs Quant-Only 2.0× 加速和 37% 节能。

从算法pipeline角度拆解术语：

```
Step 1: S32 Logits = Q_s8 ⊗ K_s8^T  // ACL NEGEMMLowp, S8×S8→S32
Step 2: P_u8 = IntegerSoftmax(S32_logits)  // 全整数替换 FP32 softmax
Step 3: O_s32 = P_u8 ⊗ V_s8  // ACL NEGEMMLowp, U8×S8→S32
Step 4 (可选): O_s8 = Requantize(O_s32)  // 供下游层
```

整数域中 scale 的处理：QK^T GEMM 输出的 S32 值实际表示 `round(real_logit / (scale_Q × scale_K))`。IndexSoftmax 中裁剪阈值 c 需换算为整数阈值：`c_int = round(6.6 / (scale_Q × scale_K / sqrt(d)))`。LUT 的索引映射同样需 scale 校准。这些 scale 校准可离线预计算，在线无开销。

术语一般如何实现？如何使用？

- ACL + `add_impl_for_ACL.patch`：替换 NEGEMMLowpMatrixMultiplyCore 的中间路径
- 自定义 C++ NEON intrinsics：手写 S32→U8 kernel（`bench_speed.cpp` 的方式）
- PyTorch 模拟：`acc_llm.py --method int_attention`，替换 HF 模型的 attention forward
- 适用条件：Q/K/V 预量化为 INT8、GEMM 支持 INT8（ACL/cuBLAS/MKL-DNN）、training-free

涉及论文标题：
- IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

## Training-free Attention Quantization（训练无关注意力量化）

术语是什么？

训练无关注意力量化指在无需对预训练模型进行任何重新训练、微调或量化感知训练（QAT）的前提下，直接将注意力层的计算从浮点精度替换为整数量化精度的技术。这是一种"即插即用"（drop-in replacement）的后训练优化策略。

与 QAT（需要训练数据、GPU 算力和数小时训练）不同，训练无关方法仅需：访问模型权重（前向传播）、确定量化超参数（scale、clipping range、LUT 精度）、替换 attention 计算图节点。IntAttention 的 IndexSoftmax 通过一次性的超参数 sweep（c=6.6, b=5 在 WikiText 上确定为稳定区域）而非训练学习来确定参数，对同架构模型族可通用。

从算法pipeline角度拆解术语：

```
阶段 1（离线，一次性）：超参数选择
  for c in [4.0..8.0], b in [3..8]:
    替换所有 attention 层的 softmax 为 IndexSoftmax(c, b)
    在 WikiText-2/ImageNet 上评估 PPL/Accuracy
  选 PPL 最低且稳定的 (c, b) → c=6.6, b=5

阶段 2（在线，推理时）：直接替换
  加载预训练模型 → 对每层 attention 替换 softmax 为 IndexSoftmax
  Q/K/V 在线量化 → 推理，无训练
```

术语一般如何实现？如何使用？

- **算子替换**：`torch.fx` 图变换 / `register_forward_hook` / ONNX 节点替换 / ACL C++ 层替换
- **与 QAT 对比**：Training-free 耗时分钟级（sweep）vs QAT 数小时，无需 GPU vs 需 GPU，精度损失 <0.5% vs 几乎无损
- **IntAttention 使用**：`acc_llm.py --method int_attention` 自动下载模型 → 替换 softmax → eval；`bench_speed --pipe 3` 跑 ARM CPU 延迟

涉及论文标题：
- IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

## LLM-guided spec search（LLM引导的Spec搜索）

术语是什么？
LLM-guided spec search 是 OPENJARVIS 提出的一种本地-云端协作优化算法，用于联合优化个人AI系统的四个可编辑原语（Intelligence/Engine/Agents/Tools & Memory）。云端前沿模型在搜索时诊断失败痕迹并提出跨原语协调编辑，由 held-out gate 仅接受非退化编辑，推理时优化后的 spec 完全在本地执行。该算法属于 greedy gated search 类别——单步可跨多个原语字段的贪心搜索，每步编辑由独立门控验证。

从算法pipeline角度拆解术语：
LLM-guided spec search 的核心算法流程（伪代码）：

```
Algorithm: LLM-guided spec search
Require: Spec S_0, teacher T, gate G, tolerance ε, budget B
1: S ← S_0
2: while not converged and cost < B do
3:   C ← T.diagnose(traces(S))
       // 教师LM读取trace corpus → 按失败模式聚类 → 每cluster标注
       // student vs teacher 成功率 + 技能差距的自然语言描述
4:   e ← T.propose(S, C)
       // 教师提出跨Intelligence/Engine/Agents/Tools & Memory的协调编辑
       // 单次proposal可同时: 改写tool description + 调整prompt 
       // + 切换Engine backend + 更改量化格式 + 触发LoRA/GRPO训练
5:   S' ← apply(S, e)
6:   if G_c(S') > G_c(S) AND ∀c'≠c: G_c'(S') ≥ G_c'(S) - ε then
7:     S ← S'              // greedy accept
8:   end if
9: end while
10: return S
```

各阶段详解：
- **Diagnose（诊断）**：教师 LM 摄取 eligible trace corpus（benchmark traces/合成 traces/用户批准的脱敏 traces）→ 按失败模式分组为 failure clusters → 每个 cluster 标注 student vs teacher 成功率 + 技能差距的自然语言描述（如"student 在需要日历查找的多跳问题上失败，因为不调用 calendar tool"）。
- **Propose（提议）**：教师提出四类编辑——Intelligence edits（模型选择、量化格式变更、LoRA/GRPO 训练触发）、Engine edits（backend 切换、batch size、KV-cache）、Agent edits（prompt 重写、few-shot exemplars、agent type 切换、turn limits）、Tools & Memory edits（添加/移除工具、修改 tool description、切换 memory backend）。所有编辑被记录，教师可检查自己的干预历史。
- **Gate（门控验证）**：Gate 由三类信号组成——合成标注或用户批准的脱敏 traces、大规模 agentic datasets（GeneralThoughtArchive 431K traces, ToolScale）、标准 benchmark splits（MMLU-Pro, GAIA, τ-bench）。编辑仅当目标 failure cluster 改进且非目标 cluster 退化 ≤ ε（默认 1%）时接受。
- **Repeat（迭代）**：接受编辑 → 更新 spec → 重复循环。停止条件：gate score 在 k=5 个 session 内停滞或预算耗尽。

复合奖励（仅用于触发训练的 Intelligence edits）：
$$R(q,y) = \alpha R_{\text{acc}}(q,y) - \beta \hat{E}(q,y) - \gamma \hat{L}(q,y) - \delta \hat{C}(q,y)$$
默认权重 (α, β, γ, δ) = (0.5, 0.1, 0.1, 0.3)。Ê, L̂, Ĉ 为 z-score 归一化值。

术语一般如何实现？如何使用？
实现方式：基于 OPENJARVIS 参考实现（Apache 2.0 开源，PyPI: `pip install OpenJarvis`）。Learning 原语中的 `spec_distillation` policy 实现 LLM-guided spec search。使用方式：`jarvis learn --benchmark PinchBench --teacher claude-opus-4-6` → 教师分析 traces → 提出编辑 → gate 验证 → 更新 spec。关键设计要点：（1）**Failure cluster 驱动的编辑分配**——教师将 failure mode 映射到对应原语（retrieval failure→Tool edit 65%, reasoning failure→Intelligence edit 52%, control-flow failure→Agent edit 51%, efficiency-bounded→Engine edit 58%）；（2）**跨原语协调编辑**——单次 proposal 可同时修改多个原语，实现单原语优化器无法表达的联合优化；（3）**非退化门控**——GateOK 防止过拟合；（4）**搜索-推理分离**——云端教师仅在搜索时使用（\$15.6/benchmark），推理时零云端调用。实验效果：比最佳单原语 baseline 高 5.5–16.5 pp，比 prompt-only baseline 高 5.0–18.8 pp，优化成本低 7.1–10.9×。

涉及论文标题：
- OpenJarvis

## Failure Clusters（失败聚类诊断）

术语是什么？
Failure Clusters 是 LLM-guided spec search 中 Diagnose 阶段的输出——教师 LM 将 student spec 的失败痕迹按共同失败模式聚类，每个 cluster 标注 student vs teacher 成功率和技能差距的自然语言描述。这是将非结构化 traces 转化为可操作编辑信号的关键机制。每个 cluster 对应一种可被特定原语编辑解决的技能差距，使教师能够有针对性地提出编辑（而非盲目搜索）。

从算法pipeline角度拆解术语：
Failure clustering 在 LLM-guided spec search 中的角色——**将问题空间结构化以指导搜索方向**：

```
Diagnose(traces) → Failure Clusters:
  Cluster c1: "student fails on multi-hop questions requiring 
              calendar lookups → does not invoke calendar tool"
              student_success: 12%, teacher_success: 94%
  Cluster c2: "student generates syntactically incorrect code 
              for Python data analysis tasks"
              student_success: 35%, teacher_success: 89%
  Cluster c3: "student responses too slow for interactive use
              (>5s per turn)"
              avg_latency_student: 8.2s, avg_latency_teacher: 1.1s
```

聚类过程：
1. 教师 LM 读取 trace corpus（JSON 格式的 benchmark traces/合成 traces/用户脱敏 traces）
2. 教师在 held-out tasks 上比较自己与 student 的输出，识别 student 表现不佳的查询类别
3. 将相似失败模式的 traces 分组——例如"所有需要多步工具调用的查询"或"所有 Python 代码生成任务"
4. 每个 cluster 生成自然语言描述（"student fails on X because Y"）和量化指标（student vs teacher 成功率）

编辑分配逻辑（基于 OPENJARVIS 实验数据，Qwen3.5-9B + Claude Opus 4.6）：
- Retrieval failures → Tool edits (65% of accepted edits for this cluster type)
- Reasoning failures → Intelligence edits (52%)
- Control-flow failures → Agent edits (51%)
- Efficiency-bounded failures → Engine edits (58%)
- Format/output failures → Agent edits (53%)

术语一般如何实现？如何使用？
实现方式：作为 LLM-guided spec search 的 Diagnose 阶段实现。教师 LM（如 Claude Opus 4.6/GPT 5.4/Gemini 3.1 Pro）通过 API 调用接收 trace corpus → 使用 grep/脚本分析 JSON traces 中的字段 → 通过 LLM 推理识别重复失败模式 → 生成 failure clusters。使用方式：完全自动化——`jarvis learn` 触发后自动执行 diagnose→propose→gate 循环，用户无需手动标注失败模式。每个 cluster 的 skill gap 描述被记录到搜索日志中，教师可检查自己的干预历史跨 sessions 识别有效/无效编辑。

涉及论文标题：
- OpenJarvis

## Gate Function / Gated Acceptance（门控接受函数）

术语是什么？
Gate Function（门控函数）是 LLM-guided spec search 中的 held-out 验证机制——用于判断候选 spec 编辑是否应被接受。其核心逻辑：编辑仅当目标 failure cluster 的 gate score 改进且所有非目标 cluster 的 gate score 退化不超过容忍阈值 ε（默认 1%）时接受。这是保证搜索不退化（non-regressing）的关键机制。

从算法pipeline角度拆解术语：
Gate Function 的数学定义：

令 G(S) 为 spec S 的 held-out gate score，G_c(S) 为限制在 failure cluster c 上的 gate score。对于目标 cluster c 的编辑 e，S' = apply(S, e)，接受条件：

$$G_c(S') > G_c(S) \quad \text{且} \quad G_{c'}(S') \ge G_{c'}(S) - \epsilon \quad \forall c' \ne c$$

Gate score 的计算基于三类信号源：
1. **合成标注/用户批准的脱敏 traces**：教师标注合成 traces 或用户批准的脱敏 traces 为 ground-truth labels
2. **大规模 agentic datasets**：GeneralThoughtArchive（431K reasoning traces with verifier scores）、ToolScale（diverse verifiable queries）
3. **标准 benchmark splits**：MMLU-Pro/GAIA/τ-bench 的 train/test splits

Gate 的作用：
- **防止过拟合**：编辑只在 held-out 数据上评估，不接受仅对训练数据有效的编辑
- **防止退化**：即使 target cluster 改进，若任何非 target cluster 显著退化（>ε），编辑仍被拒绝
- **统一度量**：同一 GateOK 规则适用于 Intelligence weights/Agent prompts/Tool descriptions/Memory config/Engine settings 的所有编辑类型
- **成本效率**：拒绝的编辑不触发完整的训练运行（如 LoRA/GRPO），只浪费 proposal API 调用——解释了 LLM-guided spec search 比单原语 baseline 低 7.1–10.9× 的优化成本

术语一般如何实现？如何使用？
实现方式：在 OPENJARVIS 中作为 `GateOK(S', S, C, ε)` 函数实现。Gate 评估在每次候选编辑的 Execute 阶段执行。接受编辑 → 更新 spec → 继续搜索；拒绝编辑 → 记录 gate 结果到日志 → 教师可在下一轮利用被拒编辑的信息提出改进 proposal。默认 ε=1%，可根据部署场景调整——敏感应用可使用更严格的 ε（如 0.5%），探索性应用可使用更宽松的 ε（如 2%）。使用方式：完全自动化，作为 LLM-guided spec search 的门控步骤运行。

涉及论文标题：
- OpenJarvis

## Composite Reward for Intelligence Edits（Intelligence编辑的复合奖励函数）

术语是什么？
Composite Reward 是 LLM-guided spec search 中当 Intelligence 编辑触发模型训练（LoRA/GRPO）时使用的多目标奖励函数。它在精度、能耗、延迟和成本四个维度上平衡候选响应的质量，使训练出的模型不仅精度高，同时在效率维度上也表现良好。

从算法pipeline角度拆解术语：
复合奖励的数学定义：

$$R(q,y) = \alpha R_{\text{acc}}(q,y) - \beta \hat{E}(q,y) - \gamma \hat{L}(q,y) - \delta \hat{C}(q,y)$$

其中：
- R_acc(q,y)：query q 对候选响应 y 的精度奖励（如 exact match 或 LLM-judge score）
- Ê(q,y)：z-score 归一化的能耗偏差，Ê = (E − μ_E) / σ_E（在 benchmark 内归一化）
- L̂(q,y)：z-score 归一化的延迟偏差
- Ĉ(q,y)：z-score 归一化的美元成本偏差
- 默认权重：(α, β, γ, δ) = (0.5, 0.1, 0.1, 0.3)

归一化处理：每个效率量 X ∈ {E, L, C} 在 benchmark 内计算 z-score 归一化（X̂ = (X−μ_X)/σ_X），使奖励函数交易的是无量纲偏差而非原始焦耳/秒/美元。

奖励使用流程：
1. Intelligence edit 触发 GRPO 训练 → 对每个候选响应 y 计算 R(q,y)
2. GRPO 使用 composite reward 进行策略优化
3. 训练完成后，gate 评估完整 spec（包含更新后的 weights + 其余原语配置）的端到端精度
4. Gate 接受/拒绝基于精度而非复合奖励——复合奖励仅影响训练方向

鲁棒性：reward weight 扰动实验（LiveCodeBench, Qwen3.5-9B + Claude Opus 4.6）显示，所有合理权重设置下最终精度在 4.6 pp 带内（80.7%–85.4%），默认权重 (0.5, 0.1, 0.1, 0.3) 比最佳变体（accuracy-only α=1 达 85.4%）低 2.4 pp——这个精度差异是主动用 2.4 pp 换取在能耗/延迟/成本上的 Pareto 优。部署可根据优先级调整权重。

术语一般如何实现？如何使用？
实现方式：在 OPENJARVIS 的 Learning 原语中，当 Intelligence edit 指定 GRPO training 时启用。奖励函数在 GRPO 训练循环内部计算——每步对候选响应 scoring。归一化参数 μ_X 和 σ_X 在每个 benchmark 上预计算。使用方式：默认权重适用于通用部署；accuracy-first 部署可设置 α=1, β=γ=δ=0（约 +2.4 pp 精度 gain）；energy-first 部署可提高 β（约 −1 pp 精度 cost）；cost-sensitive 部署可提高 δ。奖励函数仅在 Intelligence edit 的训练阶段使用；推理时不需要奖励计算。

涉及论文标题：
- OpenJarvis

## Prefill & Decode Stages (LLM Inference Pipeline)

术语是什么？

Prefill 和 Decode 是自回归 LLM 推理的两个截然不同的计算阶段，由 Transformer decoder 的因果注意力（causal attention）机制和自回归生成范式自然划分。

**Prefill 阶段**（亦称 prompt processing / initiation phase）：一次性并行处理所有输入 prompt tokens（$N_{prefill}$ 个），执行完整的多层 Transformer forward pass，产生第一个输出 token 的 logits 并为所有层生成完整的 KV Cache。计算特性：
- **Compute-bound**：Self-attention 计算量为 $O(N_{prefill}^2 \cdot D)$（QK^T 的完整 $N_{prefill} \times N_{prefill}$ 矩阵乘），FFN 计算量为 $O(N_{prefill} \cdot D^2)$
- **高并行度**：所有 prompt tokens 同时处理，矩阵维度大 → GPU 高利用率（>80%）
- **高内存需求**：KV Cache 全部从零构建，大小为 $N_{prefill} \times L \times D \times 2$（K 和 V 各一份，FP16）
- **一次执行**：每个 request 仅执行一次 prefill

**Decode 阶段**（亦称 token generation / autoregressive phase）：逐 token 迭代生成，每次仅输入 1 个新 token（上一 step 采样的 token），利用已缓存的 KV Cache（前序 tokens 的 K/V 已存储，无需重算），通过单层 forward pass 产生下一个 token 的 logits，然后采样（argmax/top-p/top-k）输出。重复此过程直到达到 max_tokens 或 EOS。计算特性：
- **Memory-bound**：Self-attention 计算量为 $O(N_{kv} \cdot D)$（新 token 的 Q 与所有历史 K 做 attention），远小于 prefill 的 $O(N^2 \cdot D)$
- **低并行度**：每次仅 1 token → 矩阵乘的 batch 维度极小 → GPU 利用率低（<20% 典型）
- **KV Cache 持续增长**：每步追加 1 token 的 K/V → $N_{kv}$ 线性增长 → attention 计算量也线性增长
- **重复执行**：每个 request 执行 $N_{decode}$ 次（$N_{decode}$ = 输出 token 数）

从算法pipeline角度拆解术语：

以 TinyLlama 1.1B（22 layers, d=2048, 32 heads, d_head=64）为例的一次完整推理流程：

```
输入: prompt = "Explain DVFS", N_prefill = 3 tokens
      max_decode = 4 tokens

=== Phase 1: Prefill (一次性, 3 tokens 并行) ===

1. Tokenization + Embedding:
   input_ids = [15496, 261, 1234]  # "Explain", "DV", "FS"
   h = Embedding(input_ids)  # [3, 2048]

2. Layer 0 Prefill (重复 layer 1..21):
   a. Self-Attention:
      Q, K, V = h @ W_Q, h @ W_K, h @ W_V  # [3, 2048] each
      
      # QK^T 计算 — O(N^2 × D) compute cost
      scores = Q @ K^T / sqrt(64)  # [3, 3] causal masked
      attn_weights = softmax(scores)  # [3, 3]
      
      # K/V 存储为 KV Cache: [3, 2048] each (FP16)
      KV_cache[layer_0] = (K, V)
      
      # PV 计算
      attn_out = attn_weights @ V  # [3, 2048]
   
   b. FFN (SwiGLU):
      gate = SiLU(h @ W_gate)  # [3, 5504] (FFN intermediate)
      up = h @ W_up            # [3, 5504]
      ffn_out = (gate * up) @ W_down  # [3, 2048]
   
   c. Residual + RMSNorm:
      h = RMSNorm(attn_out + ffn_out + h)  # [3, 2048]

3. LM Head:
   logits = h[-1, :] @ W_lm_head  # [vocab_size] = [32000]
   # 取最后一个 token 的 hidden state

=== Phase 2: Decode (逐 token, 4 steps) ===

Step 1 (token "DVFS"):
   1. Embedding: h = Embedding[sampled_token]  # [1, 2048]
   
   2. Layer 0 Decode:
      Q = h @ W_Q  # [1, 2048]
      K_new = h @ W_K  # [1, 2048]
      V_new = h @ W_V  # [1, 2048]
      
      # 追加到 KV Cache
      K = concat(KV_cache[layer_0].K, K_new)  # [4, 2048]
      V = concat(KV_cache[layer_0].V, V_new)  # [4, 2048]
      KV_cache[layer_0] = (K, V)
      
      # QK^T — O(N_kv × D) memory-bound
      scores = Q @ K^T / sqrt(64)  # [1, 4]
      attn_weights = softmax(scores)  # [1, 4]
      attn_out = attn_weights @ V  # [1, 2048]
      
      # FFN 同 prefill, 但 batch=1
      ...
   
   3. Sampling: next_token = argmax(logits) or top_p/top_k

Step 2-4: 重复 Step 1，KV Cache 增长至 [7, 2048]

=== 计算特性对比 ===
| 维度          | Prefill              | Decode                |
|--------------|----------------------|----------------------|
| 输入 shape    | [N_prefill, D]       | [1, D]               |
| Attention 复杂度 | O(N²×D) compute-bound | O(N×D) memory-bound  |
| GPU 利用率    | >80% (大矩阵乘)      | <20% (1 token/batch) |
| KV Cache      | 首次构建, N_prefill  | 每步追加 1 token     |
| 延迟特征      | ~100ms (232 tokens)  | ~10-50 ms/token      |
| 每 request 次数 | 1 次                 | N_decode 次          |
| 功耗特征      | 高功耗 (~5W GPU)     | 中低功耗 (~2W GPU)   |

关键观察（CORE 论文的核心利用）：
- Prefill 是 compute-bound → 高频率直接加速 → 用高 GPU/CPU 频率
- Decode 是 memory-bound → 高频率不加速（带宽瓶颈） → 用中频率省电
- 两者最优频率不同 → 需阶段感知的 DVFS 调度
```

术语一般如何实现？如何使用？

1. **llama.cpp 中的实现**：
   ```cpp
   // llama_decode 内部
   if (n_tokens > 1) {
       // Prefill: 并行处理所有 tokens
       ggml_cgraph *gf = llm_build_graph(ctx, batch, true);  // prefill=true
   } else {
       // Decode: 单 token
       ggml_cgraph *gf = llm_build_graph(ctx, batch, false); // prefill=false
   }
   ```

2. **CORE 的阶段感知**：
   - Python daemon 检测 prefill 开始（首次 batch>1）和 decode 开始（batch=1）
   - `llama.cpp` 在 `llama_decode` 前后写入 `/tmp/llama_phase`（"prefill"/"decode"）
   - CORE daemon 轮询该文件 → 查表 → 切换频率

3. **关键指标**：
   - **TTFT (Time-To-First-Token)**：Prefill 延迟 = 从 prompt 到第一个 token 的时间。CORE 改善 8.5-17.7%
   - **TPOT (Time-Per-Output-Token)**：Decode 每步延迟 = 输出 N 个 token 的平均每 token 时间。CORE 改善 27.8-39.6%
   - **E2E Latency** = TTFT + N_decode × TPOT
   - 为什么 TPOT 改善远超 TTFT？Decode 阶段 governor 的"向下螺旋"效应更严重（利用率更低 → 降频更多 → TPOT 恶化），CORE 纠正效果更大。

涉及论文标题：
- Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE
