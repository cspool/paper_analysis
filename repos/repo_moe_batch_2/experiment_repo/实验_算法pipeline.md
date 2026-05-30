## LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - LExI 提出一种 data-free 的 post-training 优化技术，通过两阶段 pipeline 为预训练 MoE 模型的每一层静态分配最优的 active expert 数量（top-k），无需任何 calibration 数据集或微调：
    1. **Stage 1 — 逐层 Top-K 扰动敏感性分析（Monte Carlo Profiling）**：对每个 MoE layer，从标准正态分布 N(0,1) 采样随机输入张量 X ∈ R^{B×L×H}。先用 baseline top-k 计算输出 Y_base，再用候选 top-k ∈ {1, 2, ..., top-k_baseline} 计算扰动输出 Y_perturbed。用 Frobenius 范数 ||Y_perturbed - Y_base||_F 度量输出偏差，重复 N_iter 次取平均得到该层的 top-k 敏感性 profile D_j(k)。整个过程仅使用模型权重，无需真实数据。
    2. **Stage 2 — 进化搜索（Evolutionary Search with Proxy）**：将 Stage 1 的敏感性值作为 proxy，用进化算法搜索全局最优的逐层 top-k 分配 k* = (k_1, ..., k_L)。目标：最小化总敏感损失 Σ_j D_j(k_j)，约束：Σ_j k_j = B（总 active expert budget），k_min ≤ k_j ≤ k_max。使用 tournament selection + uniform crossover + mutation（每层 ±1 同时保证总和不变），迭代 G_max 代后返回最优分配。
  - 实验比较：
    - LExI vs Baseline（pretrained fixed top-k）vs Inter-Expert Pruning (NAEE, 12.5%/25%/50%) vs Intra-Expert Pruning (MoE-I², 12.5%/25%/50%)
    - 指标：Throughput (tokens/s) vs Accuracy/F1/Perplexity 的 Pareto trade-off
    - LM-Eval：9 个语言理解任务 (ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande) 平均准确率
    - LongBench (Qasper)：F1 score vs throughput
    - Passkey Retrieval：准确率 vs throughput（100 iterations，varying depths）
    - Perplexity：C4, PTB, WikiText-103 上的 PPL vs throughput
    - VLMEvalKit：MME, MMMU, ScienceQA（仅 DeepSeekVL2-Tiny）
  - 结果：OLMoE-1B-7B 上 LExI (B=100) 可达与 50% intra-pruning 相同的 throughput 同时准确率高 10%；Qwen1.5-MoE 上 LExI 获得比 inter/intra pruning 高 5.1% 的吞吐量同时准确率高 0.5%；Mixtral-8x7B 上 LExI 在相同吞吐量下比 inter-pruning 准确率高 10%。

- 硬件平台是什么，配置是什么。
  - **NVIDIA H100 80GB GPUs**（支持 Tensor Cores）
  - 大多数 LLM 模型使用 4 GPUs；DeepSeek-V2-Lite-Chat 和 DeepSeekVL2-Tiny 使用 2 GPUs
  - 多 GPU 间使用 Tensor Parallelism
  - 批量推理 batch size = 16，input/output 序列长度因模型而异（遵循各模型最大 context length 约束）
  - 论文未明确说明 CPU、内存、互联类型（NVLink/NVSwitch）等详细配置

- 模型是什么。数据集和bench分别是什么。
  - **LLM 模型**：
    - Mixtral-8x7B-Instruct-v0.1 (46.7B, 32 layers, 8 experts, top-k=2)
    - Qwen1.5-MoE-A2.7B-Chat (14.3B, 24 layers, 60 experts, top-k=4)
    - OLMoE-1B-7B-0924-Instruct (6.92B, 16 layers, 64 experts, top-k=8)
    - MiniCPM-MoE-8x2B (17B, 40 layers, 8 experts, top-k=2)
    - DeepSeek-V2-Lite-Chat (15.7B, 27 layers, 64 experts, top-k=6)
  - **VLM 模型**：DeepSeekVL2-Tiny (3B, 12 layers, 64 experts, top-k=6)
  - **Benchmarks**：
    - lm-eval-harness：ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande（报告平均准确率）
    - LongBench：Qasper（F1 score）
    - Passkey Retrieval（准确率，100 iterations with varying depths）
    - Perplexity：C4, Penn Treebank (PTB), WikiText-103
    - VLMEvalKit：MME, MMMU, ScienceQA（VLM 评估）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **未开源**。论文未提供代码仓库链接或代码可用性声明。
  - LExI 算法 pipeline 伪代码（基于论文 Algorithm 1 + Algorithm 2）：
    ```
    # Stage 1: Per-Layer Sensitivity Profiling
    D = {}  # key: top-k value, value: list of Frobenius norms
    for i in range(N_iter):
        X = randn(B, L, H)  # random Gaussian input
        set_topk(model, k_base)
        Y_base = moe_forward(X)
        for k in T:  # T = [1, 2, ..., k_base]
            set_topk(model, k)
            Y_k = moe_forward(X)
            D[k].append(||Y_k - Y_base||_F)
    for k in T:
        D[k] = mean(D[k])  # average perturbation loss

    # Stage 2: Evolutionary Search
    population = random_feasible_allocations(N_pop, L, B, k_min, k_max)
    for g in range(G_max):
        p1, p2 = tournament_select(population)  # min φ(k) = Σ D_j(k_j)
        offspring = uniform_crossover(p1, p2)    # each layer randomly from parent
        offspring = mutate(offspring, η_mut)      # ±1 per layer, ΣΔ = 0
        offspring = project_to_feasible(offspring, B, k_min, k_max)
        population.append(offspring)
    k_star = argmin_{k in population} Σ D_j(k_j)
    return k_star  # (k_1, ..., k_L) per-layer top-k
    ```
  - 核心张量计算：MoE 层输出 y = Σ_{i=1}^{top-k} G(x)_i · E_i(x)，其中 G(x) = Softmax(TopK[x·W_g])。LExI 通过改变每层的 top-k 参数（set_topk 操作）控制激活的 expert 数量。Frobenius 范数 ||Y_k - Y_base||_F = sqrt(Σ_{i=1}^{B×L×H} (Y_k[i] - Y_base[i])²) 衡量改变 top-k 对输出的扰动程度。


## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- 属于算法pipeline的实现是什么？实验比较什么？
  - Lancet 提出两种算法 pipeline 优化：
    1. **Weight Gradient Computation Scheduling（反向传播）**：将 MoE 模型训练反向传播中的 weight gradient computation (dW) 算子调度到与 all-to-all 通信重叠执行。分析 IR 依赖图，用 BFS/DFS 识别与每个 all-to-all 无依赖路径的 dW 指令集合。然后采用 best-fit greedy 算法：顺序遍历 all-to-all 指令，对每个 all-to-all 从可用 dW 池中贪心选取总执行时间最接近 current unoverlapped time 的 dW 算子集，使 all-to-all 被 dW 计算最大程度覆盖。
    2. **Operator Partition with Dynamic Programming（前向传播）**：将前向传播中的 non-MoE 计算（如 self-attention、前一个/后一个 Transformer layer 的 FFN）分区并与 all-to-all + expert 计算组成 computation-communication pipeline。使用 DP 公式 `T(n) = min_{1<i<n-1} {T(i) + min_{1<k<K} P(i,n,k)}` 搜索最优 partition range（包含哪些 non-MoE 算子和多少个 partitions）。其中 `P(i,n,k)` 为指令 i 到 n 被分为 k 个 partition 并经 pipeline scheduling 后的端到端时间。Partition axis 通过约束满足问题（CSP）求解，使用 OR-Tools。Pipeline scheduler 按 stage 组织 partitioned 算子并模拟时间线得到 P(i,n,k)。
  - 实验比较：
    - Lancet vs DeepSpeed 0.5.8、Tutel 0.3、RAF（无 Lancet 优化的 baseline）
    - 训练吞吐量（tokens/s 或 iteration time）、通信重叠度（non-overlapped communication time 减少量）
    - 两种 gating 方法：Switch gate（允许 pre-MoE 和 post-MoE 分区）和 Batch Prioritized gate（只允许 post-MoE 分区）
    - Ablation study：仅 scheduling vs 仅 pipelining vs 两者组合的加速比
  - 结果：non-overlapped communication 减少最多 77%（vs Tutel on V100），端到端加速最高 1.3x

- 硬件平台是什么，配置是什么。
  - **A100 Cluster**: Amazon EC2 p4de.24xlarge × 8 nodes，每 node 8× NVIDIA A100 80GB GPU，4×100 Gbps NIC
  - **V100 Cluster**: Amazon EC2 p3dn.24xlarge × 8 nodes，每 node 8× NVIDIA V100 32GB GPU，1×100 Gbps NIC
  - 软件环境：Ubuntu 20.06, CUDA 11.3, NCCL 2.12.12 (PXN enabled), Docker
  - Weak scaling 评测，从 1 node (8 GPUs) 扩展到 8 nodes (64 GPUs)

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - **GPT2-S-MoE**：12 layers, hidden dim 768, 每 GPU 2 experts
    - **GPT2-L-MoE**：24 layers, hidden dim 1024, 每 GPU 2 experts
    - 基于 Huggingface Transformers v4.18.0 的 GPT-2，每隔一个 Transformer block 的 FFN 替换为 MoE layer
    - Expert 数量随 GPU 数量线性扩展（per-GPU 保持 2 experts）
  - **数据集**：WikiText (Merity et al., 2016)
  - **Benchmark Metrics**：training iteration time, throughput, non-overlapped communication time
  - Input sequence length 固定 512；A100 上 GPT2-S-MoE batch size 24/GPU, GPT2-L-MoE 48/GPU；V100 上 16 和 8

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：GitHub https://github.com/hikettei/Lancet (Apache-2.0)，AWS Labs 镜像 https://github.com/awslabs/Lancet-Accelerating-MoE-Training-via-Whole-Graph-Computation-Communication-Overlapping
  - **Weight Gradient Computation Scheduling 伪代码**：
```
Input: instruction sequence I (model IR)
  G = CreateDependencyGraph(I)
  Ia = [all-to-all instructions in I]
  For each all-to-all Ij^a:
    Wj = {Ik in I | no directed path between Ij^a and Ik}
  t^a, t^W = profile execution time of all instructions
  W_used = {}; Asg = {}
  For each all-to-all Ii^a in Ia:
    tu = ti^a  // unoverlapped time
    While tu > 0 AND W_i \ W_used != empty:
      jmin = argmin_j{|tu - tj^W| | Ij^W in W_i, Ij^W not in W_used}
      tu = tu - t^W_{jmin}
      W_used.add(I^W_{jmin}); Asg[I^W_{jmin}] = Ii^a
  I' = ReorderInstructions(Asg)
    // place dW instructions right after their assigned all-to-all
```
  - **DP Partition Range Selection**：
```
T(n) = min_{1<i<n-1} { T(i) + min_{1<k<K} P(i,n,k) }
// P(i,n,k): pipeline time of instructions i..n partitioned k ways
// For each P(i,n,k):
//   1. PartitionAxisInferencer (CSP) -> axes for all tensors
//   2. PipelineScheduler -> simulate timeline -> end-to-end time
// K is max partitions (default 8), limited by batch dim size
```
  - **Partition Axis CSP** (以矩阵乘 Y=XW 为例)：
    - `(ax1=0 ∧ ax2=-1 ∧ ay1=0) ∨ (ax1=-1 ∧ ax2=1 ∧ ay1=1)`
    - ax1=0 表示沿 X 的 row 维度分区（W 不变，Y 沿 row 分区）
    - ax2=1 表示沿 W 的 column 维度分区（X 不变，Y 沿 column 分区）
    - -1 表示不分该维度
  - **Pipeline Scheduling**：将 partitioned 指令按 stage 组织（所有 computation 连续执行为一个 stage，所有 communication 为一个 stage），各 partition 按 partition index 顺序调度，每个指令的 start time = max(依赖指令 end time, 同类型前一个 partition 指令的 end time)

## IFMoE: An Inference Framework Design for Fine-grained MoE

- 属于算法pipeline的实现是什么？实验比较什么？
  - IFMoE 提出的算法 pipeline 核心是 **Self-Draft Speculative Decoding with KV-cache Revision**：
    1. **Self-Draft 机制**：不引入额外的小型 draft model，而是利用 fine-grained MoE 模型自身在激活更少 expert 时仍能保持较好性能的特性。Draft 阶段仅激活 decode_topk Dk=2 个 expert（全量为 Ek=6），每个 decode step 的计算量大幅降低（GroupedGEMM 的 expert 数从 6 降到 2），实现快速 draft。
    2. **KV-cache Revision**：每 α=10 步 draft 后，用全量 Ek=6 experts 对 buffer 内所有 token 做一次 encode forward，将 KV-cache 中对应的 key/value 更新为"全量 expert 应产生的值"。这是 IFMoE 区别于传统 speculative decoding 的关键——传统方法用 draft logits 与 target logits 比较做 accept/reject，IFMoE 接受全部 draft token 但修正 KV-cache。
    3. **Accept All 策略**：IFMoE 接受 draft model 生成的全部 token（不做逐 token 的 accept/reject），仅通过 KV-cache revision 来补偿用更少 expert draft 造成的信息损失。
  - 实验比较：
    - Full model（标准全量 expert decode）vs IFMoE（Dk=2 draft + Ek=6 revision）
    - 下游任务：XSum（ROUGE）、GSM8K（准确率）、TruthfulQA-Gen、IFEval
    - 超参：α=10, encode_topk Ek=6, decode_topk Dk=2
    - 结果：下游性能与 full model 接近（如 Qwen2: GSM8K 75.4→71.1, XSum 13.7→13.5），benchmark 延迟和吞吐均提升 >30%

- 硬件平台是什么，配置是什么。
  - Qwen2-57B-A14B-Instruct：4× NVIDIA A6000 GPUs
  - Deepseek-Lite-Chat：2× NVIDIA A6000 GPUs
  - 论文未明确说明 CPU、内存、互联等配置细节

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - **Qwen2-57B-A14B-Instruct**：fine-grained MoE，64 experts/layer，14B 激活参数
    - **Deepseek-Lite-Chat**：fine-grained MoE，64 experts/layer（附录 Table 2 含 Deepseek-v2，160 experts/layer，用于内存分析但未用于 benchmark）
  - **数据集/Benchmark**：
    - **XSum**：摘要生成，评估 ROUGE
    - **GSM8K**：数学推理，评估准确率
    - **TruthfulQA-Gen**：真实性评估
    - **IFEval**：指令遵循评估
  - Benchmark 实验：最大 batch size 256（Qwen2）/ 200（Deepseek-Lite），测量 decoding 阶段的 latency 和 throughput

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **未开源**。论文 Checklist 标注 "IFMoE is still under develop with future features"。
  - **伪代码**（直接引用论文 Algorithm 1）：
    ```
    Input: α, encode_topk Ek, decode_topk Dk,
           fine-grained MoE model M
    Initialize: terminate = False, buffer = []
    while not terminate do
      for each step in α do
        buffer.append(M.decode(topk = Dk))   # Draft: 仅激活 Dk=2 experts
      end for
      # Revise KV Cache
      M.encode(buffer, topk = Ek)             # Verification: 全量 Ek=6 experts 重算 KV
      terminate = detect_terminate()
      buffer = []
    end while
    ```
  - **张量计算流程**：
    1. **Draft Decode Step**（per token）：
       - Attention: Q,K,V = W_Q·x, W_K·x, W_V·x → softmax(QK^T/√d)·V
       - Router: g_i = softmax(W_r·x), select top-Dk=2 experts
       - GroupedGEMM (Cutlass): 对选中的 2 个 expert 并行做 y_i = W_i^up·x → σ → W_i^down·σ(W_i^gate·x)
       - Combine: y = Σ g_i · y_i（仅 2 个 expert 的结果加权和）
    2. **KV-cache Revision**（per α steps）：
       - 对 buffer 中 α 个 token，重新过 Router: select top-Ek=6 experts
       - GroupedGEMM (Cutlass): 对选中的 6 个 expert 并行计算
       - 用新计算的 key/value 覆盖 KV-cache 中对应位置
       - Attention 层在下一次 decode 时自动读取修正后的 KV-cache

## I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - I²MoE 提出一种端到端的 MoE 框架，用于显式建模多模态交互（modality interaction），基于 Partial Information Decomposition (PID) 将模态交互分解为四种类型：唯一性(uniqueness for modality 1)、唯一性(uniqueness for modality 2)、协同(synergy)、冗余(redundancy)。核心设计：
    1. **Interaction Experts**：使用 n+2 个交互专家（n 个唯一性专家 + 1 个协同专家 + 1 个冗余专家），每个专家是一个完整的融合模型（fusion model + prediction head），通过弱监督的交互损失（interaction loss）学习专精于特定交互类型。
    2. **Perturbation-based Weak Supervision**：训练时使用随机向量替换某个模态的嵌入来模拟单模态场景（masked modality），对不同专家施加不同的交互损失：唯一性专家使用 Triplet Margin Loss 使完整模态输出接近未遮蔽目标模态、远离遮蔽模态；协同专家使用 Cosine Similarity 使完整模态输出与所有遮蔽输出最大化差异；冗余专家使用 Cosine Similarity 使完整模态输出与所有遮蔽输出最大化相似。
    3. **Re-weighting Model**：一个 MLP 根据所有模态嵌入输出 soft weights [w_uni1, w_uni2, ..., w_syn, w_red]，对每个交互专家的预测加权求和得到最终预测 ŷ = Σ w_i · ŷ_i。
    4. **Dual-objective Loss**：L_total = L_task + λ_int · L_int，其中 L_task 用带权融合输出计算，L_int 鼓励专家分化。
  - 实验比较：
    - Baselines：Early Fusion (EF)、Late Fusion (LF)、Low-Rank Multimodal Fusion (LRMF)、Multimodal Transformer (MulT)、InterpretCC、SwitchGate (Switch Transformer)、MoE++
    - 消融实验：(1) No-Interaction（去掉交互损失）、(2) Latent-Contrastive（在隐空间嵌入而非输出上施加交互损失）、(3) Simple-Weight（用全局可学习权重替代MLP重加权）、(4) Less-Forward（仅遮蔽2个模态而非全部）、(5) Synergy-Redundancy（仅保留协同和冗余专家）
    - 通用性实验：I²MoE 与 SwitchGate、InterpretCC、MoE++ 结合
    - 评估指标：Accuracy、AUROC、Micro F1、Macro F1
    - 可解释性评估：样本级局部解释（logits + weights 分解）、数据集级全局解释（权重分布统计）、人类评估（15人，300次评价）

- 硬件平台是什么，配置是什么。
  - **单卡 NVIDIA A100 GPU**（论文 Table 6 记录所有实验在单 A100 上运行）
  - 训练/推理时间及参数量详见表6

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - Modality-Specific Encoders：ADNI用3D-CNN(图像)+MLP(基因组/临床/生物样本)；MIMIC用LSTM(所有模态)；MOSI用GRU(视觉/音频/文本)；ENRICO用VGG11(截图+线框图)；IMDB用VGG16(图像)+Google Word2vec(语言)
    - Prediction Head：所有实验使用线性分类头
    - 参数规模：最小 673,935 (MOSI)，最大 6,696,728 (ADNI)
  - **数据集**：
    - **ADNI**：2,380样本，阿尔茨海默病三分类(CN/MCI/AD)，四模态(Image/Genetic/Clinical/Biospecimen)
    - **MIMIC-IV**：9,003患者记录，一年死亡率二分类，三模态(Lab/Notes/Codes)
    - **IMDB**：25,959电影，23类多标签分类，双模态(Image/Language)
    - **MOSI**：2,199 YouTube片段，情感分析回归[-3,3]→二分类，三模态(Vision/Audio/Text)
    - **ENRICO**：1,460 Android截图，20类UI设计分类，双模态(Screenshot/Wireframe)
  - **训练配置**：70%/15%/15% train/val/test split，batch_size=32，Adam优化器，lr=0.0001，训练30-50 epochs，3次随机种子取平均

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/Raina-Xin/I2MoE
  - 算法伪代码（I²MoE 训练流程，以两模态为例）：
    ```
    输入：modalities X_1, X_2, 标签 T
    编码器：E1, E2
    交互专家：F_uni1, F_uni2, F_syn, F_red (各含融合模型+预测头H_i)
    重加权模型：W (MLP)

    训练阶段：
    1. 编码模态: e1 = E1(X_1), e2 = E2(X_2), r ~ random vector
    2. 对每个专家 F_i, i ∈ {uni1,uni2,syn,red}:
       - 完整模态前向: ŷ_i^(12) = H_i(F_i(e1, e2))
       - 遮蔽模态2前向: ŷ_i^(-2) = H_i(F_i(e1, r))
       - 遮蔽模态1前向: ŷ_i^(-1) = H_i(F_i(r, e2))
       - 交互损失:
         * L_uni1 = TripletLoss(ŷ_uni1^(12), ŷ_uni1^(-2), ŷ_uni1^(-1))  # anchor, positive, negative
         * L_uni2 = TripletLoss(ŷ_uni2^(12), ŷ_uni2^(-1), ŷ_uni2^(-2))
         * L_syn = CosSim(norm(ŷ_syn^(12)), norm(ŷ_syn^(-1))) + CosSim(norm(ŷ_syn^(12)), norm(ŷ_syn^(-2)))  # 最小化相似度
         * L_red = [1 - CosSim(norm(ŷ_red^(12)), norm(ŷ_red^(-1)))] + [1 - CosSim(norm(ŷ_red^(12)), norm(ŷ_red^(-2)))]
    3. 计算权重: [w_uni1, w_uni2, w_syn, w_red] = softmax(W(e1, e2) / temperature)
    4. 融合预测: ŷ = Σ w_i · ŷ_i^(12)
    5. 任务损失: L_task = CrossEntropy(ŷ, T)
    6. 总损失: L_total = L_task + λ_int · (mean of all L_i)
    7. 反向传播更新所有参数

    推理阶段：
    1. e1 = E1(X_1), e2 = E2(X_2)
    2. 每个专家 F_i 仅计算完整模态前向: ŷ_i^(12) = H_i(F_i(e1, e2))
    3. 权重: [w_i] = softmax(W(e1, e2) / temperature)
    4. 输出: ŷ = Σ w_i · ŷ_i^(12)
    5. 可选：返回各专家预测 ŷ_i 和权重 w_i 用于可解释性分析
    ```
  - 扩展至 n 模态：交互专家数 = n + 2（n个唯一性专家 + 1个协同 + 1个冗余）。每个专家要进行 1+n 次前向（1次完整输入 + n次遮蔽单个模态）。唯一性专家 i 以完整输出为 anchor，遮蔽模态 i 为 negative，其余遮蔽为 positive。协同专家所有遮蔽为 negative。冗余专家所有遮蔽为 positive。

## HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

- 属于算法pipeline的实现是什么？实验比较什么？
  - HierMoE 提出两种拓扑感知的算法优化来加速 MoE 训练中的 AlltoAll 通信：
    1. **Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll)**：利用 GPU 集群的分层拓扑结构（如 4 层：Inter-Node/InfiniBand → Inter-QPI → Inter-NVLink → Intra-NVLink），在不同层次维度上对 token 进行去重（deduplication），消除因多个 expert 位于同一 GPU/group 而导致的 token 重复传输。通过性能模型公式选择最优维度 d*，在高层（如 Inter-Node）减少低带宽链路上的通信量，将更多通信转移到高带宽的 Intra-node 链路。
    2. **Hierarchical Expert Swap (HierD-ES)**：在 HierD-AlltoAll 基础上，通过交换 expert 在 GPU 间的位置来平衡各 hierarchical group 的通信负载。计算交换任意两个 expert 后的估计通信时间矩阵 Q_d*，选择使通信时间最小化的 expert pair 进行交换。使用 smooth-max 函数平滑 Q_d 的梯度，提升优化稳定性。
  - 实验比较：
    - Baselines：Megatron-LM（标准 AlltoAll）、Tutel-2DH（二维分层 AlltoAll）、SmartMoE（expert placement 优化）
    - 消融：HD2-MoE（仅 2D 去重）、HD2-MoE-Smart（2D 去重+SmartMoE swap）、HD-MoE（HierD-AlltoAll 无 HierD-ES）、HierMoE（完整方案）
    - 评估指标：端到端训练加速比、AlltoAll 通信加速比
    - Ablation：不同 K（top-K experts）、E（expert 数）、G（GPU 数）下的加速比；不同层级维度的效果；不同 max 函数类型；不同 expert swap 更新频率

- 硬件平台是什么，配置是什么。
  - **32-GPU 集群**：4 节点 × 8 NVIDIA RTX A6000-48G GPU
  - 每节点配置：Dual Intel Xeon Platinum 8358 @ 2.60GHz，512GB DDR4，8× A6000-48G @ 1.46GHz
  - 互联：NVLink 112.5GB/s (4× link)，PCIe 4.0 (x16)，Mellanox MT28908 @ 200Gb/s InfiniBand
  - 软件环境：Ubuntu 20.04，CUDA 12.1，PyTorch 2.1.2，NCCL 2.18.5

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - DeepSeek-V3（hidden/model dim 减半至原始的 1/2，6 layers，EP degree=32）
    - Qwen3-30B-A3B（32 layers，EP degree=32）
  - **训练配置**：micro batch size=1，sequence length=1024
  - **数据集**：论文未明确说明具体训练数据集名称
  - **指标**：AlltoAll 通信时间加速比、端到端训练迭代时间加速比

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - HierMoE 本身未公开独立开源仓库，基于 Megatron-LM (https://github.com/NVIDIA/Megatron-LM/) 实现。NCCL 通信性能参数通过 nccl-tests (https://github.com/NVIDIA/nccl-tests) 采集拟合。
  - HierD-AlltoAll 算法 pipeline 伪代码（单 MoE layer 的 AlltoAll Dispatch 流程）：

```
=== 初始化阶段（集群启动时执行一次） ===
Input: 集群拓扑 D（4 层: Node/QPI/NVLink/Intra-GPU），GPU 数 G
Output: U[0..D-1]（各层 expert group 数），α/β 参数（AlltoAll 性能模型）

1. 通过 nccl-tests 测量 7 种 AlltoAll 通信的 α, β:
   - 标准 AlltoAll (HD1), Inter-Node (HD2-Inter1), Intra-Node (HD2-Intra1)
   - Inter-QPI (HD3-Inter2), Intra-QPI (HD3-Intra2)
   - Inter-NVLink (HD4-Inter3), Intra-NVLink (HD4-Intra3)
2. 使用最小二乘法拟合线性模型: t = α + n · β
3. U ← [1, 4, 8, 16, 32]  // 对应每层的 expert group 数
   例: Inter-Node AlltoAll 按 4 个 node 分成 4 组 (U[1]=4)
       Inter-QPI 每个 node 内再分 2 组 (U[2]=8)
       Inter-NVLink 再分 2 组 (U[3]=16)
       Intra-NVLink 最终 32 GPU (U[4]=G=32)

=== 每 iteration 的 HierD-AlltoAll Dispatch 流程 ===
Input: 路由结果 mask I_route ∈ R^{T×E} (T 为 token 数, E 为 expert 数)
       M (embedding 维度), G, E, D, U[], α/β 参数
Output: 最优维度 d*, 完成 token dispatch

Step 1: 计算 HD1-AlltoAll 的通信时间 t1
  m ← E/G                                    // 每 GPU 的 expert 数
  I_route^(1,G)[i,j] ← OR over j1 in [(j-1)m+1, j·m] of I_route[i,j1]
  p[j] ← sum_i I(I_route^(1,G)[i,j])          // 每 expert group 的去重 token 数
  n_a2a ← G · max(p) · M · v                  // 通信量 (v=字节/维度, FP16 下 v=2)
  t1 ← α_a2a + n_a2a · β_a2a

Step 2: 对 d = 2..D 计算 HDd-AlltoAll 通信时间 td
  for k = 1 to D-1:                            // 遍历各层级的 Inter-level
    m ← E/U[k]
    // 将 routing mask 聚合到 U[k] 个 expert group (去重)
    I_route^(k,U[k])[i,j] ← OR over j1 in [(j-1)m+1, j·m] of I_route^(k,E)[i,j1]
    p_a2a^(k,U[k])[j] ← sum_i I(I_route^(k,U[k])[i,j])
    n_a2a^Inter(k) ← (U[k]/U[k-1]) · max(p_a2a^(k,U[k])) · M · v
    // 更新 routing mask 以反映 Inter-level-k 通信后的 expert 分布
    I_route^(k+1,E) ← process(I_route^(k,E))
    p_a2a^(k+1,G)[j] ← sum_i I(I_route^(k+1,E)[i,j])
  // Intra-level-(d-1) 通信量
  n_a2a^Intra(d-1) ← (G/U[d-1]) · max(p_a2a^(d,G)) · M · v
  // 总时间 = Σ(Inter-level 各层) + Intra-level
  td ← Σ_{i=1}^{d-1} (n_a2a^Inter(i) · β_a2a^Inter(i) + α_a2a^Inter(i))
       + n_a2a^Intra(d-1) · β_a2a^Intra(d-1) + α_a2a^Intra(d-1)

Step 3: 选择最优维度 d*
  d* ← argmin_{1≤d≤D} td
  复杂度: O(D·T·K)

=== HierD-ES Expert Swap (每 iteration 可选执行) ===
Input: d*, routing mask, 当前 expert-to-GPU placement
Output: 交换的 expert pair (r*, c*)

1. 初始化 Z ∈ R^{E×E×U[d*]} 和 Z_intra ∈ R^{E×E×G}
   // Z[r,c,k]: 交换 expert r 和 c 后，第 k 个 expert group 的去重 token 数
2. for each token t (选中的 K 个 experts):
     for each expert pair (A, B) where A 被选中, B 未被选中:
       - Case 1/2: B 所在 group 无其他选中 expert → 该 group 计数+1
       - Case 2/4: A 是 group 内唯一选中 expert → A 原 group 计数-1
       - Case 1/3: A 所在 group 有 ≥2 选中 expert → A 原 group 不变
       - Case 3/4: B 所在 group 有选中 expert → B 所在 group 不变
   // 通过增量更新降低复杂度从 O(D·T·K·E²) 到 O(D·T·K·E)
3. 基于公式计算交换每对 expert 后的通信时间 Q_d*[r,c]
4. 使用 smooth-max (γ=10) 平滑: smooth_max(x,γ)=max(x)·(Σ_i (x[i]/max(x))^γ)^(1/γ)
5. (r*, c*) ← argmin Q_d*[r,c]
6. 交换 expert r* 和 c* 在 GPU 间的位置 (约 1% end-to-end 时间开销)
7. 将 Z 和 Z_intra 重置为无交换状态用于下一轮
```

- 去重效果量化：当 R=4（expert 组数）, K=8 时，重复率达 55%（表 II），HierD-AlltoAll 可消除这些重复。高 K（如 DeepSeek-V3 的 K=8）+ 低组数 → 高重复率 → 去重收益更大。

- 属于算法pipeline的实现是什么？实验比较什么？
  - Hecate 提出 **Fully Sharded Sparse Data Parallelism (FSSDP)**，一种全新的 MoE 训练范式，核心算法组件：
    1. **FSSDP Sharding Phase**：将每个 MoE layer 的 parameters 和 optimizer states 划分为 |𝒟| 个不相交的 MoE shards，每个 shard 包含一组 expert 的完整参数+优化器状态，全局只保留一份 optimizer states 副本，实现最小且均衡的内存占用。
    2. **SparseAllGather (spAG)**：用于每 iteration 从 MoE shards 中稀疏物化 (sparsely materialize) expert placement。形式化定义为 spAG(𝒫₀, 𝒫₁)，其中 𝒫₀ 为 surjective 的前置条件（每个 chunk 唯一归属于某 device），𝒫₁ 为 𝒫₀ 的超集。通信量上界 O(λS)，λ = |Ĉ|/|C| 为稀疏度，λ << 1 时远小于 FSDP 中 AllGather 的 O(S)。
    3. **SparseReduceScatter (spRS)**：用于将物化专家的梯度 reduce 回对应 MoE shard 所在 device。定义为 spRS(𝒫₀, 𝒫₁)，𝒫₁ ⊆ 𝒫₀ 且 surjective。每个 spAG(𝒫, 𝒫') 与对称的 spRS(𝒫', 𝒫) 配对。
    4. **Heterogeneous Sharding (Algorithm 2)**：跨所有 MoE layer 统一调度 sharding，允许每个 MoE shard 包含任意数量 expert（0 到 |ℰ|），同时保证跨 device 内存均衡。先放置 underloaded expert（负载变化慢），再填充 overloaded expert。re-sharding 低频触发（每 100 iterations）。
    5. **Sparse Materialization (Algorithm 1)**：拓扑感知的启发式搜索算法，在 overlap degree t 和 memory capacity m 两个约束下，搜索近似最优的 expert placement。t ≤ m 时，将 top-t overloaded expert 物化到所有 device；否则按负载比例分配 replica slots，优先 intra-node 通信。
    6. **Re-materialization**：物化的 expert 参数在用后立即释放，后续 backward 需要时再次通过 spAG 物化，将参数内存额外开销降低 90.2%。
  - 实验比较：baseline 包括 EP (原生 Expert Parallelism)、FasterMoE、SmartMoE、FlexMoE。评估 GPT-MoE-S (1.84B)、GPT-MoE-L (7.36B)、BERT-MoE (3.27B)、BERT-MoE-Deep (6.54B) 四种 MoE 模型在不同 GPU 规模（16/32 GPU）下的端到端训练加速比。消融实验分别验证 heterogeneous sharding 和 sparse materialization 各组件的贡献（图 15），以及 re-materialization 的 memory/performance trade-off（图 14）。layer-wise speedup 分析（图 11）和 critical path 时间分解（图 12）。

- 硬件平台是什么，配置是什么。
  - **Cluster A**：4× AWS p3dn.24xlarge nodes，每 node 8× NVIDIA V100-32G GPU（NVLink 300 GB/s），node 间 100 Gbps 网络。
  - **Cluster B**：4× AWS p4d.24xlarge nodes，每 node 8× NVIDIA A100-40G GPU（NVSwitch 600 GB/s），node 间 400 Gbps 网络。
  - 总规模：每个 cluster 32 GPUs（4 nodes × 8 GPUs），16 GPU 实验使用 2 nodes。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：GPT-3 和 BERT 的 MoE 变体。FFN 替换为 MoE layer（expert 仍为 FFN，d_ffn = 2 × d_model），GShard Top-2 gating。
    - GPT-MoE-S：d_model=768, SeqLen=2048, 12 layers, 64 experts, 1.84B params
    - GPT-MoE-L：d_model=1536, SeqLen=2048, 12 layers, 64 experts, 7.36B params
    - BERT-MoE：d_model=1024, SeqLen=512, 12 layers, 64 experts, 3.27B params
    - BERT-MoE-Deep：d_model=1024, SeqLen=512, 24 layers, 64 experts, 6.54B params
  - **数据集**：论文未明确说明具体数据集；训练框架使用 Megatron-LM，采用 weak scaling 方式（16 GPU 用 32 experts，32 GPU 用 64 experts）。
  - **指标**：端到端训练加速比（vs EP）、layer-wise 加速比、peak memory usage（optimizer states/gradients/parameters 分项）、critical path 时间分解。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未公开 Hecate 完整代码。基于 PyTorch + NCCL 实现，使用 Megatron-LM 作为训练框架。
  - FSSDP 算法 pipeline 伪代码（单 MoE layer l 在一个 iteration 中的执行流程）：

```
=== SHARDING PHASE (iterator 间低频执行) ===
Input: F^g (all MoE layers 的 expert load 分布), t (overlap degree)
Output: P^g = {P_0, P_1, ..., P_L} (各层 sharding plan)

1. J ← 各层 top-t overloaded experts
2. J' ← E^g - J  // underloaded experts
3. slots_per_device ← |E^g| / |D|  // 每 device 可用 slots
4. for each layer l in sortByMaxLoadDescending(L):
5.     for each expert e in J'_l (sorted by load descending):
6.         n ← least-loaded node (优先剩余 slots 少的)
7.         d ← least-loaded device on node n
8.         P_l ← P_l ∪ {(d, e)};  S_d ← S_d - 1
9. 将 J 中剩余 experts 任意分配到剩余 slots

=== MATERIALIZATION PHASE (每 iteration 执行) ===
Input: P (sharded placement), F (estimated expert loads),
       t (overlap degree), m (memory capacity)
Output: P' (materialization plan)

1. t ← min(t, |E|), m ← min(m, t)
2. P' ← P
3. if t ≤ m:
4.     E^topT ← Top t experts by load F
5.     P' ← P' ∪ (D × E^topT)  // 物化到所有 device
6. else:
7.     totSlots ← |D| · m
8.     for each e in sortByLoadDescending(E^topT):
9.         n ← assignSlotsByLoad(e, totSlots, F)
10.        P^e ← 在 nodes/devices 间分配 n 个 replica
               (优先有空闲 slots 的 node)
11.        P' ← P' ∪ P^e

=== FORWARD PASS of MoE layer l ===
1. // 通信与前一 Attention 计算重叠
2. P_l' ← Scheduler(P_l, F_l, t, m)  // Algorithm 1
3. spAG(P_l, P_l')  // SparseAllGather: 物化 expert 参数
4. // 可选 Calibration: 用 MoE gate 实际输出重新运行 Algorithm 1
5. Token dispatching (topology-aware All-to-All):
   - 优先 intra-node 通信
   - 同 expert 多 replica: 均匀分配 tokens
6. Expert FFN computation on materialized parameters
7. Release materialized parameters (若启用 re-materialization)

=== BACKWARD PASS of MoE layer l ===
1. spAG(P_l, P_l')  // re-materialize expert 参数
2. Expert backward computation
3. spRS(P_l', P_l)  // SparseReduceScatter: reduce gradients to source
4. Release materialized parameters

=== OPTIMIZER STEP ===
1. 各 device 在其 MoE shards 上用同步后的 gradients 更新
   optimizer states 和 model parameters
```

  - 关键张量计算：设 expert e_i 的参数为 W_i ∈ R^{d_model × d_ffn}（FFN 三层），SparseAllGather 从持有 e_i 的 source device d_src 以 Broadcast 方式将 W_i 发送到需要 e_i 的 target devices，通信量为 |W_i| × |target_devices|。spRS 逆过程，将各 device 上 e_i 的梯度 reduce（求和）回 d_src。整体通信量上界 O(2λS)，与同 placement 下 rearrangement 系统 AllReduce 通信量等价。

## HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - HOBBIT 提出 **Mixed Precision Expert Offloading** 算法：利用 MoE 中不同 expert 的重要性差异，对不重要的 cache-miss expert 动态替换为低精度（量化）版本以加速加载，同时保持模型精度。核心算法组件：
    1. **Expert 重要性动态估计**：使用 ||G(x)_{e_i}||（gating output 的 magnitude）作为 expert 重要性的计算高效代理，实验验证与 ||G(x)_{e_i}E_{e_i}(x)||（expert output magnitude）的 Pearson 相关系数为 0.99。
    2. **Unimportance Degree Score**：s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||（已归一化）。对所有 top-K experts 按 ||G(x)|| 降序排列后计算累积分数。
    3. **双阈值精度决策**：T1=0.6（高精度阈值），T2=0.9（跳过阈值）。s ≤ 0.6 → 高精度 (FP16/INT8)；0.6 < s ≤ 0.9 → 低精度 (INT4/INT2)，加载量减少 4×；s > 0.9 → 跳过该 expert。分布比例约 67%/30%/3%（以 Mixtral-8x7B 为例）。
    4. **混合精度预取**：即使预测精度低时，低精度 expert 预取加载的惩罚远低于高精度（图 9），使预取在任何精度下都有正向收益。
    5. **多维混合精度缓存淘汰**：LHU (Least High Precision Frequently Used) 策略 + LRU + LFU + FLD 加权组合，H_t 记录高精度使用频次，最小化混合精度 miss penalty（高精度 miss 代价为 C，低精度为 B_l/B_h · C）。
  - 实验比较：6 个 baseline（Transformers, DeepSpeed-Inference, Llama.cpp, MoE-Offloading, MoE-Infinity, Fiddler）。精度验证使用 GSM8K（数学推理 accuracy）和 TruthfulQA（truth/info 分数）。消融实验分别验证动态加载（1.19-1.57× speedup）、自适应预取（~5% decoding speedup）、多维缓存（4.69%-8.68% miss penalty reduction vs LRU）。

- 硬件平台是什么，配置是什么。
  - **RTX 4090**（24GB GPU, 256GB CPU, 64 cores, PCIe 4.0 32GB/s）作为 edge server。
  - **Jetson AGX Orin**（32GB unified memory, 12 CPU cores）作为 end device。
  - 存储：Samsung NVMe SSD 980 PRO (7,000 MB/s read, ~3,000 MB/s 实测)。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - Mixtral-8x7B：45B 总参数，32 layers，8 experts/layer，top-2 routing，14B 激活参数/token，权重 87GB（expert 占 96%）。
    - Phi-MoE：42B 总参数，32 layers，16 experts/layer，top-2 routing，6.6B 激活参数/token，权重 78GB（expert 占 96%）。
  - **精度配置**：
    - FP16 高精度 + INT4 低精度（RTX 4090 实验）
    - INT8 高精度 + INT2 低精度（Jetson Orin 实验）
  - **数据集与 benchmark**：
    - 速度测试：Alpaca 数据集 60 高质量样本（一半 input length=16，一半 input length=128），四种 I/O 组合 [16,32]/[16,128]/[128,32]/[128,128]。
    - 精度测试：GSM8K（数学推理，top-1 accuracy）、TruthfulQA（truth 和 info 分数）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未公开 HOBBIT 完整代码；基于 Llama.cpp (https://github.com/ggerganov/llama.cpp) 修改。
  - 混合精度 Expert Offloading 算法伪代码：

```python
# === HOBBIT Dynamic Mixed Precision Expert Loading ===
# 输入: hidden state x, gating function G (Linear + Softmax + TopK)
# 输出: MoE layer output y

# 超参数
T1 = 0.6  # 高/低精度分界阈值
T2 = 0.9  # 低精度/跳过分界阈值

def moe_layer_forward(x, expert_cache, expert_storage):
    # Step 1: Gating（GPU 计算）
    gate_logits = G.linear(x)           # [1, num_experts]
    gate_probs = softmax(gate_logits)   # [1, num_experts]
    topk_vals, topk_ids = topk(gate_probs, K=K)
    
    # Step 2: 归一化 gating weights 并按降序排序
    gate_norm = topk_vals / topk_vals.sum()  # 归一化
    sorted_idx = argsort(gate_norm, descending=True)
    
    # Step 3: 计算 unimportance degree score
    scores = zeros(K)
    cumulative = 0.0
    for i in range(K):
        e_i = sorted_idx[i]
        scores[e_i] = cumulative
        cumulative += gate_norm[e_i]
    
    # Step 4: 动态决定每个 expert 的精度
    load_tasks = []
    for i, e_i in enumerate(topk_ids):
        s = scores[i]
        if s <= T1:
            precision = "high"   # FP16 或 INT8
            skip = False
        elif s <= T2:
            precision = "low"    # INT4 或 INT2
            skip = False
        else:
            skip = True
        
        if not skip and e_i not in expert_cache:
            load_tasks.append((e_i, precision))
    
    # Step 5: 异步加载缺失的 experts
    for e_i, precision in load_tasks:
        weight = expert_storage.read(e_i, precision)  # PCIe/SSD read
        expert_cache.insert(e_i, weight, precision)   # 由 Cache Manager 管理
    
    # Step 6: Expert FFN 计算（GPU）
    y = zeros_like(x)
    for i, e_i in enumerate(topk_ids):
        if not skip_for(e_i):
            weight = expert_cache.get(e_i)
            # FFN: y_e = W_o · (SiLU(W_g · x) ⊙ (W_p · x))
            gate_out = silu(matmul(x, weight.W_g))  # [1, d_ffn]
            up_out = matmul(x, weight.W_p)           # [1, d_ffn]
            expert_out = matmul(gate_out * up_out, weight.W_o)  # [1, d_model]
            y += gate_norm[i] * expert_out
    
    return y

# === Adaptive Expert Prefetching ===
def predict_experts(x, stacking_computer, num_layers_ahead=3):
    # Stacking Computer: 一次性矩阵乘计算后续层 gating
    # gate_weights_stacked: [num_layers_ahead, d_model, num_experts]
    gate_logits_all = matmul(x, gate_weights_stacked)  # [num_layers_ahead, num_experts]
    topk_ids_all = topk(gate_logits_all, K=K, dim=-1)   # [num_layers_ahead, K]
    return topk_ids_all  # 后续层预取 expert IDs

# === Multidimensional Cache Priority ===
def compute_cache_priority(expert_t, current_layer, current_token, records):
    T = current_token
    l_n = total_layers
    l_i = current_layer
    l_t = expert_t.layer_id
    
    p_lru = records.R[t] / T      # 最近使用时间
    p_lfu = records.F[t] / T      # 使用频率
    p_lhu = records.H[t] / T      # 高精度使用频率
    p_fld = 1 - ((l_t - l_i + l_n) % l_n) / l_n  # 层距离
    
    p = w_lru*p_lru + w_lfu*p_lfu + w_lhu*p_lhu + w_fld*p_fld
    return p  # 越高优先级越高，evict 最低优先级的 expert
```

  关键设计原理：
  - Pearson 相关系数 0.99 验证了 ||G(x)|| 可作为 expert 贡献度的近似，避免实际计算 E(x)。
  - 双阈值设计基于 gating 分布的统计分析：top-1 expert 始终得 0 分（保持高精度），约 67% 选高精度、30% 选低精度、3% 跳过。
  - 总 expert cache miss penalty 相比 LFU 降低 2.13%-4.19%，相比 LRU 降低 4.69%-8.68%。

## GLaM: Efficient Scaling of Language Models with Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - GLaM 提出基于稀疏激活 Mixture-of-Experts (MoE) 的 decoder-only 语言模型系列。核心实现：(1) 每隔一个 Transformer 层将标准 FFN 替换为 MoE 层（64 个 expert FFN），每 token 通过 top-2 softmax gating 仅激活 2 个 expert，加权组合输出；(2) 非 MoE 层使用 Gated Linear Unit (GLU) + GeGLU 替代标准 ReLU+Linear；(3) 使用 per-layer relative positional bias 替代绝对位置编码；(4) 通过 GSPMD 2D sharding 将 expert 权重 [E, M, H] 沿 E 和 H 维度划分、输入激活 [B, S, M] 沿 B 和 M 维度划分，实现无冗余并行；(5) 使用 GShard 辅助负载均衡损失（系数 0.01）鼓励 expert 均匀使用。最大 GLaM (64B/64E) 拥有 1.2T 总参数，每 token 仅激活 96.6B（8%），推理 FLOPs/Token 约 GPT-3 的 51.4%，训练能耗约 GPT-3 的 1/3。
  - 实验比较：(1) MoE vs Dense 对比：GLaM MoE 系列 vs 同等 nact-params 的 GLaM Dense 系列（0.1B→137B dense），在 29 个 NLP benchmark 上 zero/one/few-shot 性能；(2) vs GPT-3 (175B Dense)：最大 GLaM (64B/64E) vs GPT-3 在相同 benchmark 上的 zero/one/few-shot, FLOPs/token 和训练能耗对比；(3) Scaling 实验：expert 数量 scaling（1→256 experts）、data quality 消融（filtered vs unfiltered web data）；(4) Data Efficiency：不同 training token 量（up to 630B）下 MoE vs Dense 的 learning curve；(5) 参考对比：Gopher (280B), Megatron-NLG (530B), Switch-C (1.5T MoE)。

- 硬件平台是什么，配置是什么。
  - 训练硬件：1,024 块 Cloud TPU-V4 芯片（最大 GLaM 64B/64E 模型）。TPU-v4 单芯片实测系统功耗 326W。数据中心 PUE=1.11（训练期间）。
  - 训练框架：使用 GSPMD (Xu et al. 2021) 进行 2D sharding 模型并行，基于 Lingvo 框架 (Shen et al. 2019) 实现。
  - 网络拓扑：TPU 集群的 2D device mesh 拓扑，expert 沿 device 维度分布，同一 index 的 expert 跨不同 MoE 层放置于同一 device 上。

- 模型是什么。数据集和bench分别是什么。
  - 模型系列：GLaM (0.1B/64E)、GLaM (1.7B/32E, 64E, 128E, 256E)、GLaM (8B/64E)、GLaM (64B/64E) 以及对应 Dense 基线 GLaM (0.1B)、GLaM (1.7B)、GLaM (8B)、GLaM (137B)。关键架构参数：最大模型 L=64 layers, M=8192, H=32768 (MoE expert hidden), nheads=128, dhead=128。
  - 训练数据集：1.6T tokens，混合权重：Filtered Webpages 42% (143B tokens, Pareto 采样)、Books 20% (390B)、Conversations 28% (174B)、Forums 2% (247B)、Wikipedia 6% (3B)、News 2% (650B)。质量过滤使用 feature hash linear classifier 对 curated text vs webpages 分类后 Pareto 采样。
  - 评估 benchmark（29 个）：
    - NLG (8): TriviaQA, Natural Questions (NQS), Web Questions (WebQS), SQuADv2, LAMBADA, DROP, QuAC, CoQA。指标：Exact Match, F1。
    - NLU (21): HellaSwag, StoryCloze, Winograd, WinoGrande, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, BoolQ, COPA, RTE, WiC, MultiRC, WSC, ReCoRD, CB, ANLI R1/R2/R3, RACE-m, RACE-h。指标：Accuracy（除 MultiRC 用 F1a）。
  - 评估协议：zero-shot（直接评估）、one-shot（随机取 1 训练例作为 demonstration）、few-shot（每个 task 用 up to GPT-3 使用的 shot 数）。NLG 任务使用 beam search width=4。NLU 任务基于条件对数似然 log P(option|context) 长度归一化。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未公开模型权重或训练代码。GSPMD 2D sharding 算法论文开源（Xu et al. 2021, arXiv:2105.04663）。训练基于 Google 内部 Lingvo 框架（开源: https://github.com/tensorflow/lingvo）。GShard MoE 架构参考 Lepikhin et al. 2021（开源: https://github.com/google-research/google-research/tree/master/gshard）。
  - GLaM MoE 算法 pipeline 伪代码：

```python
# === GLaM Decoder-only MoE Transformer 前向传播 ===

# 输入 token_ids: [B, S], 经过 embedding + relative positional bias
x = embedding(token_ids)  # [B, S, M]

for layer in range(L):
    # --- Attention (非 MoE 层，标准 multi-head self-attention) ---
    attn_out = multi_head_self_attention(
        x, relative_positional_bias=per_layer_rel_bias[layer]
    )  # [B, S, M]
    x = x + attn_out  # residual

    if layer % 2 == 1:  # 每隔一层: MoE FFN 层
        # --- MoE Gating ---
        # gating_logits: [B, S, E], softmax over experts
        gating_logits = softmax(x @ W_gate)  # W_gate: [M, E]
        # top-2 gating: 选前 2 大 gate 值的 expert index
        gate_vals, expert_indices = top_k(gating_logits, k=2)  # [B, S, 2]
        # 归一化 top-k gate 值
        gate_vals = gate_vals / sum(gate_vals, dim=-1, keepdim=True)

        # --- Sparse Expert Computation ---
        moe_out = zeros_like(x)  # [B, S, M]
        for e in range(E):  # E=64 experts
            # 找到分配给 expert e 的所有 (batch, seq) 位置
            mask_e = (expert_indices == e).any(dim=-1)  # [B, S]
            if mask_e.sum() == 0: continue
            x_e = x[mask_e]  # [n_e, M]
            # Expert FFN: 两层线性 + GeGLU 激活
            # W_e1: [M, H], W_e2: [H, M]
            h = GeGLU(x_e @ W_e1)  # [n_e, H]
            out_e = h @ W_e2  # [n_e, M]
            # 按 gate 值加权（gate_vals 对应 expert e 的那一列）
            gate_e = gate_vals[mask_e][expert_indices[mask_e] == e]  # [n_e]
            moe_out[mask_e] += gate_e.unsqueeze(-1) * out_e

        x = x + moe_out  # MoE 输出 residual
    else:  # 非 MoE 层: 标准 GLU/GeGLU FFN
        # GLU: component-wise product of two linear transforms
        # gate = GeGLU(x @ W_g), value = x @ W_v
        ff_gate = GeGLU(x @ W_g)  # W_g: [M, H]
        ff_value = x @ W_v          # W_v: [M, H]
        ff_out = (ff_gate * ff_value) @ W_o  # W_o: [H, M]
        x = x + ff_out

# --- 负载均衡辅助损失 (训练时) ---
# 对每个 MoE 层计算 GShard auxiliary loss:
# L_aux = 0.01 * Σ_E (f_e * p_e), 
#   其中 f_e = fraction of tokens dispatched to expert e
#   p_e = average gating probability for expert e
L_total = L_cross_entropy + L_aux
```

```python
# === GSPMD 2D Sharding 张量划分 ===
# Expert 权重: 形状 [E, M, H] → 沿 E 和 H 维度 partition
#   device[{E_partition}, {H_partition}]
# 输入激活: 形状 [B, S, M] → 沿 B 和 M 维度 partition
#   device[{B_partition}, {M_partition}]
# 
# 同一 index expert 跨层放在同一 device:
#   device_i 持有 Layer_0/expert_i, Layer_2/expert_i, ...
# 
# while_loop 包装重复 MoE 层计算图以降低编译时间
```

- GLaM pipeline 关键张量流（inference 单 token）：
  input token → Embedding [1, M=8192] → GSPMD shard 到 device mesh → 64 层 Transformer，其中 32 层为 MoE 层（64 experts 分布在 N/E devices，top-2 softmax gating 选 2 experts，加权 sum）→ 32 层为 GLU/GeGLU FFN 层 → final LM head [M, V=256K] → output logits。

- 训练配置：Optimizer = Adafactor (β1=0, β2=0.99, update clipping=1.0, factored second-moment)。初始 LR=0.01 保持 10K steps 后 inverse sqrt 衰减。Max seq length=1024，每 batch 1M tokens。Dropout=0。float32 weights, bfloat16 activations。NaN/Inf 梯度检测 + 跳过更新 + checkpoint 回退恢复机制。SentencePiece tokenizer (vocab 256K)。

## GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 GraphMETRO：一种基于 Mixture-of-Experts (MoE) 架构的 GNN 泛化方法，将未知分布偏移（distribution shift）分解为多个 shift components，通过 gating model + multiple expert models 生成 referentially invariant representations。核心实现：(1) Gating model ϕ（GNN encoder）输入图数据输出权重向量 w ∈ R^{K+1}，预测各 shift component 的贡献；(2) K+1 个 Expert models {ξ_i}，每个对应一个 shift component，其中 ξ_0 为 reference model，其他 expert 对其分配的 shift component 产生 referentially invariant representations（定义：ξ_0(G) ≈ ξ_i(τ_i(G))，∀G ∈ supp(D_s)）；(3) 对 expert 输出进行 softmax 加权聚合 h(G) = Softmax(w) · [z_0, ..., z_K]^T；(4) 训练目标 L = L1 + L2：L1 为 BCE loss 优化 gating 预测混合成分，L2 为 CE + λ·Frobenius distance loss 优化 expert 分类和与 reference model 的对齐（L2 不反向传播到 gating model）。
  - 实验比较：(1) 真实数据集（GOOD benchmark）：WebKB、Twitch、Twitter、GraphSST2 上 vs ERM、DANN、IRM、VREx、GroupDRO、Deep Coral、SRGNN、EERM、OODGAT、DIR、G-Mixup、GSAT、CIGA 的分类准确率/ROC-AUC；(2) 合成数据集（DBLP、CiteSeer、IMDB-MULTI、REDDIT-BINARY）上 vs ERM 和 ERM-Aug 在不同单/多 shift component 组合（14种环境）下的准确率；(3) Ablation：移除 L1 loss、shared encoder vs independent encoder、移除 alignment term (λ=0)、不同数量和类型的 transform functions（2-7个）；(4) Invariance matrix 可视化（验证每个 expert 对其对应 shift component 的不变性）；(5) Distribution shift discovery（gating model 输出的 mixture 揭示目标分布的 shift 类型，如 WebKB 以 "add_edge" 为主导，Twitch 以 "noisy_node_feat" 和 "drop_node" 为主）。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA GPU（论文 checklist 声明提供了 GPU 信息，但正文未列出具体 GPU 型号）。论文未明确说明具体 GPU 型号或数量。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - 真实实验 backbone：Graph Convolutional Network (GCN) [25]（node-level：WebKB, Twitch），Graph Isomorphism Network (GIN) with Virtual Node [70, 18]（graph-level：Twitter, SST2）
    - 合成实验 backbone：Graph Attention Networks (GAT) [63]
    - GraphMETRO 架构：gating model ϕ（一个 GNN encoder），K+1 个 expert models ξ_i（每个为独立 GNN encoder，或共享 GNN + 独立 MLP），classifier μ（MLP + softmax）
    - 激活函数：ReLU（真实）/ PReLU（合成）
  - 数据集：
    - 真实数据集（来自 GOOD benchmark [20]）：
      - WebKB：5-class 节点分类，按大学域名划分 train/test split（natural covariate shift）
      - Twitch：二分类节点分类（预测是否 streaming mature content），按用户语言域划分 split，metric=ROC-AUC
      - Twitter：grammar tree graph 分类（不同 domain 的句子长度和语言风格不同）
      - GraphSST2：sentiment tree graph 分类，metric=accuracy
    - 合成数据集：
      - 节点分类：DBLP [16]、CiteSeer [73]
      - 图分类：IMDB-MULTI、REDDIT-BINARY [46]
  - Benchmark：GOOD (Graph Out-of-Distribution) benchmark [20]

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/Wuyxin/GraphMETRO
  - 基于 PyG (PyTorch Geometric)：https://github.com/pyg-team/pytorch_geometric
  - GOOD benchmark：https://github.com/divelab/GOOD/tree/GOODv1
  - 算法 pipeline 伪代码：

```
# 输入：图 G，源分布 Ds，K 个 stochastic transform functions {τ_i}_{i=1}^K
# 模块定义：
#   ϕ: Gating GNN，输出 w ∈ R^{K+1}（各 shift component 的权重）
#   {ξ_i}_{i=0}^K: Expert GNNs (独立 GNN encoder)
#   μ: Classifier (MLP + softmax)

# ===== 训练阶段（每次梯度步） =====
for (G, y) in Ds:
    # 1. 采样 joint stochastic transform τ^{(k)}（k 个 transform 的组合）
    τ^{(k)} = sample_k_transforms({τ_1, ..., τ_K})
    G_transformed = τ^{(k)}(G)
    
    # 2. Gating Loss L1（预测 mixture）
    w = ϕ(G_transformed)  # w ∈ R^{K+1}
    Y_gt[i] = 1 if τ_i in τ^{(k)} else 0  # ground truth binary vector
    L1 = BCE(w, Y_gt)
    
    # 3. Expert Loss L2（分类 + 对齐，不反向传播到 ϕ）
    z_i = ξ_i(G_transformed) for i = 0..K  # 每个 expert 生成表示 [z_i ∈ R^v]
    z_0_ref = ξ_0(G)  # reference model 在原图上的表示
    
    # Softmax 加权聚合
    w_norm = Softmax(w)
    h = Σ_{i=0}^{K} w_norm[i] · z_i  # h ∈ R^v
    
    # 分类 + Frobenius 距离对齐
    y_pred = μ(h)  # classifier 输出
    d = (1/n) · ||h - z_0_ref||_F  # Frobenius norm, λ=1
    
    L2 = CE(y_pred, y) + λ · d  # d 为 referential alignment 项
    
    # 总 loss
    L_total = L1 + L2
    # 梯度更新：L1 → ϕ; L2 → {ξ_i}, μ (不更新 ϕ)

# ===== 推理阶段 =====
for G_test in D_test:
    w = ϕ(G_test)  # gating 预测 shift mixture
    z_i = ξ_i(G_test) for i = 0..K
    h = Softmax(w) · [z_0, ..., z_K]^T  # weighted sum aggregation
    y_pred = μ(h)  # 最终预测
```

  - **Stochastic Transform Functions**（基于 PyG 构建，共 11 种，论文实验中用了 5 种）：
    ```
    {mask_edge_feat, noisy_edge_feat, edge_feat_shift,
     mask_node_feat, noisy_node_feat, node_feat_shift,
     add_edge, drop_edge, drop_node, drop_path, random_subgraph}
    ```
    每个函数允许一或多个超参数控制变换程度（如 Bernoulli drop probability 在 [0.3, 0.5]），保留随机性确保多样性。
  - **关键超参数**：Adam optimizer, weight_decay=0, hidden_dim=64(CiteSeer:32)/128(graph)/300(Twitter/SST2), num_layers=2(graph)/3(node), dropout=0.0(合成)/0.5(真实), learning_rate=1e-3 或 1e-2（GraphMETRO 使用更高的 lr 做 grid search）, epoch=100-200, batch_size=32(graph)/NA(node)。
  - **训练计算复杂度**：前向 O(K) encoder passes，训练 O(K²|Ds|)（因 extrapolation 将数据集扩大 K+1 倍）。

## LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - LatentMoE 提出一种新的 MoE 架构，通过将 token 投影到低维 latent space 来解耦 expert routing 和计算，并利用节省的 FLOPs/内存/通信来扩展 expert 数量和 top-k。核心实现包含两种变体：
    1. **ℓ-MoE_eff（效率导向变体）**：将 routed expert 的输入维度从 d 压缩到 ℓ（压缩比 α = d/ℓ），同时将总 expert 数 N 扩展 α 倍，保持 top-K 不变。communication cost 和 memory bandwidth cost 分别降低 α 倍，同时保持模型精度不变。定义为 ℓ-MoE_eff(x) = W_↑ · (Σ_{i∈T_{K,N'}} p'_i · E_i(W_↓·x; ℓ)) + Σ_{j=N'+1}^{N'+S} E_j(x; d)，其中 N' = α·N。
    2. **ℓ-MoE_acc（精度导向变体，推荐）**：在 ℓ-MoE_eff 的基础上，同时将 top-K 也扩展 α 倍（K' = α·K）。communication cost 和 memory bandwidth cost 与 standard MoE 相同，但通过指数级扩大 expert combination 空间（C(αN, αK) ≥ C(N, K)^α）提升了模型精度。
    3. **五项 Design Principles**：(I) 低延迟场景 memory bandwidth 是瓶颈，最大化 accuracy per parameter；(II) 高吞吐场景 all-to-all 通信是瓶颈，减少 routed hidden dimension d；(III) 保持 U_eff = K·m 非线性预算以维持质量；(IV) 存在任务特定特征秩 r_eff 作为 d 压缩下界；(V) 同时增加 N 和 K 指数级扩大专家组合空间。
  - 实验比较：
    - **Ablations (16BT-2BA)**：压缩比 α sweep (α=1,2,4,8)，验证 α≤4 时质量保留；expert scaling 消融（压缩 4× 有/无 expert 扩展），验证 expert scaling 必要性；ℓ-MoE_eff vs ℓ-MoE_acc vs baseline validation loss 对比。
    - **Scaling (95BT-8BA Transformer + Hybrid-73BT-8BA Mamba-Attention MoE)**：ℓ-MoE_eff/ℓ-MoE_acc vs baseline 在 MMLU Pro, MMLU, Code, Math, Commonsense 上的 accuracy。
    - **Inference Performance**：Hybrid-73BT-8BA 在 2× H100 GPU 上 vLLM FP8 的 tokens/s/GPU（concurrency=1,4,16,64,128），LatentMoE vs Standard MoE 最多差 6% throughput。
    - **Projected Trillion-Parameter**：Kimi-K2-1T vs Kimi-K2-1T-LatentMoE（EPM≈1.35×）的 throughput-latency Pareto frontier，1.24×-3.46× slowdown for iso-accuracy standard MoE。

- 硬件平台是什么，配置是什么。
  - **系统分析基准**：NVIDIA GB200 GPUs, NVLink ~1800 GB/s bidirectional, 900 GB/s per direction。EP=64 GPUs, FP4=10 PFLOPs, HBM BW=8 TB/s。
  - **Inference 实测**：2× NVIDIA H100 GPUs, vLLM, FP8 per-tensor quantization。
  - **Trillion-parameter 投影**：proprietary performance simulator, 200K+ operating points。
  - **训练硬件**：论文未明确说明具体 GPU 型号和数量，基于 DeepSeek-v2-lite 架构和超参训练。

- 模型是什么。数据集和bench分别是什么。
  - **模型配置**（Table 2）：
    - **16BT-2BA**：L=27, d=2048, N=64, K=6, S=2, m=1408, SwiGLU, 16 heads。基于 DeepSeek-v2-lite。
    - **95BT-8BA**：L=32, d=4096, N=128, K=6, S=2, m=2688, Squared-ReLU, 32 heads, cosine LR(max=1.2e-3), seqlen=8192, batch=768(~6M tokens)。
    - **Hybrid-73BT-8BA**：L=52(24 Mamba/MoE+4 Attn), d=4096, N=128, K=6, S=2, m=2688, Squared-ReLU, WSD LR schedule(max=8e-4)。
  - **LatentMoE 配置**：压缩比 α=4 (ℓ=512 for 16B, ℓ=1024 for 95B/Hybrid), N'=αN, K'=αK=24 (ℓ-MoE_acc)。
  - **Benchmarks**：MMLU Pro, MMLU, Code(HumanEval/+/MBPP/+ avg), Math(GSM8K CoT+MATH-500 avg), Commonsense(RACE, ARC-Challenge, HellaSwag, Winogrande avg)。
  - **训练数据**：论文未明确说明具体数据集，16B ablation 和 95B/Hybrid 分别训练至 300B-1T tokens。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未给出独立 LatentMoE 代码仓库。LatentMoE 已集成到 NVIDIA Nemotron-3 Super/Ultra（arXiv:2512.20856）。TensorRT-LLM v1.2.0+ 支持 LatentMoE。Nemotron 配方开源：https://github.com/NVIDIA-NeMo/Nemotron。
  - **LatentMoE ℓ-MoE_acc 算法伪代码**：

```python
# === LatentMoE MoE Layer Forward (ℓ-MoE_acc) ===
# 输入: x ∈ R^{B×S×d}, latent dim ℓ, 压缩比 α = d/ℓ
# 共享: W_↓ ∈ R^{ℓ×d}, W_↑ ∈ R^{d×ℓ}, Router W_r' ∈ R^{N'×d}
# Routed Experts E_i(·;ℓ): W_FC1^{(i)}∈R^{m×ℓ}, W_gate^{(i)}∈R^{m×ℓ}, W_FC2^{(i)}∈R^{ℓ×m}
# Shared Experts E_j(·;d): 在原始空间 d 操作

def latente_moe_acc_forward(x):
    # 1. Router（原始空间 d）
    gate_logits = x @ W_r'.T                  # [B,S,N'] where N'=αN
    topk_vals, topk_ids = topk(softmax(gate_logits), K')  # K'=αK
    topk_vals /= topk_vals.sum(dim=-1, keepdim=True)
    
    # 2. Shared Down-Projection
    z = x.reshape(-1, d) @ W_↓.T              # [B×S, ℓ]
    
    # 3. Routed Expert 计算（在 latent space ℓ）
    moe_out = zeros(B*S, ℓ)
    for e in selected_experts:
        mask_e = token_assigned_to_expert(e)
        z_e = z[mask_e]                        # [n_e, ℓ]
        # SwiGLU/Squared-ReLU FFN in latent space:
        h_g = activation(z_e @ W_gate^{(e)}.T) # [n_e, m]
        h_u = z_e @ W_FC1^{(e)}.T              # [n_e, m]
        h   = h_g * h_u                        # [n_e, m]
        e_out = h @ W_FC2^{(e)}.T              # [n_e, ℓ]  FC2 down
        moe_out[mask_e] += gate_e * e_out
    
    # 4. Shared Up-Projection
    routed_y = moe_out @ W_↑.T                 # [B×S, d]
    
    # 5. Shared Experts（原始空间）
    shared_y = sum(shared_expert_ffn(x, E_j) for j in 1..S)
    return routed_y + shared_y
```

  - **关键张量流（ℓ-MoE_acc，单 token，d=4096, ℓ=1024, α=4, K'=24）**：
    1. Router: x[1,4096] @ W_r'[512,4096] → probs[1,512] → top-24 → 24 gate_weights
    2. Down-proj: W_↓[1024,4096] @ x → z[1,1024]（共享，所有 experts 复用）
    3. All-to-All dispatch（latent space）: z[1024] → 24 experts。通信量 ∝ 24×1024 = 24576（vs standard MoE 6×4096 = 24576，相同）
    4. Expert FFN（latent space）: 每个 expert 权重 m×ℓ+ℓ×m（减少 4× vs d×m），memory BW per expert ↓4×
    5. All-to-All combine + Up-proj: W_↑[4096,1024] @ z_combined → routed_out[1,4096]
    6. Shared Experts（d=4096）: 2 shared experts 在原始空间计算
    7. Expert 组合空间: C(512,24) vs C(128,6)，指数级增长

- 属于算法pipeline的实现是什么？实验比较什么？
  - FlyLoRA 提出一种受果蝇嗅觉回路启发的隐式 MoE-based LoRA 变体，核心设计：(1) 将 LoRA 的下投影矩阵 A 替换为**冻结的稀疏随机投影矩阵**（每行仅 p < n 个非零元素，采样自 N(0, 1/r²)），(2) 在 B 矩阵中执行 **rank-wise top-k 专家激活**——对 Ax 投影结果的 r 维分量中取 top-k 幅值，仅激活对应的 B 列。A 同时承担下投影和隐式 router 的双重角色，**无需显式 router 参数**。通过稀疏随机投影的近似距离保持性（Theorem 3.1）实现隐式路由，并通过 top-k 稀疏性实现 rank 间梯度去相关（Theorem 3.3: 协方差降低因子约 k²/r²），同时不同 task 的独立随机 A_i/A_j 天然近似正交（Theorem 3.4），实现多任务模型合并时的 inter-task 去耦合。
  - 实验比较：(1) 单任务 SFT：vanilla LoRA(r=8)、LoRA(r=32)、Split-LoRA(4×8)；更强 baseline AdaLoRA、SoRA、HydraLoRA。(2) 多任务模型合并（weight averaging、TIES-MERGING、DARE）：与上述 LoRA 变体比较合并前后性能下降幅度。额外与 KnOTS、L-LoRA 等高级合并方法比较。(3) 消融：负载均衡策略（loss-free vs loss-controlled）、A 冻结/可训练、sparsity ratio、activated rank k、total rank r、k-selection 策略、A 初始化方案等。

- 硬件平台是什么，配置是什么。
  - 主要实验：Linux server, Ubuntu 20.04.4 LTS, Intel Xeon Platinum 8358P CPU @2.60GHz, 8× NVIDIA GeForce RTX 3090 (24GB), CUDA 11.7。
  - Qwen-2.5-14B 实验：8× NVIDIA A100 GPUs。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1-8B, Qwen-2.5-7B, Qwen-2.5-14B（扩展实验）。
  - 训练数据集：MMLU（99,842条多选题，涵盖57学科）、ScienceQA（12,726条，仅用文本部分）、GSM8K（7,473条数学应用题）、CodeAlpaca-20k（20,022条代码指令对）。
  - 评估 benchmark：MMLU（14,042条测试）、ScienceQA（4,241条测试）、GSM8K（1,319条测试）、HumanEval（164条 Python 编程题，Pass@1/5/10）。
  - 评估方式：zero-shot，HumanEval 用 pass@k，其余用 accuracy。所有结果报告 3 个随机种子误差条。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源链接：https://github.com/gfyddha/FlyLoRA
  - FlyLoRA 算法 pipeline（基于 Section 3.1, Eq. 7-11）：
    ```
    Input: 输入 token x ∈ R^n, 预训练权重 W0 ∈ R^{m×n}, 
           冻结稀疏随机投影 A ∈ R^{r×n}（每行 p 个非零 ~N(0,1/r²)）,
           可训练 B ∈ R^{m×r}, 负载均衡偏置 d ∈ R^r,
           激活 rank k, 总 rank r, scaling factor α=2r
    Forward:
      y = A @ x                                  # sparse projection: O(r·p)
      y_biased = y + d                           # load balancing bias
      I_topk = argtopk(y_biased, k)              # 选择 top-k 激活维度
      mask = zeros(r); mask[I_topk] = 1          # k 个 1, r-k 个 0
      delta = (α/r) * (B * mask) @ y             # 仅激活 k 个 B 列, O(m·k)
      output = W0 @ x + delta
    Load Balancing Update (每步, loss-free):
      for i in 1..r:
        d_i += u * sign(expected_count_i - actual_count_i)
    
    Backward: 仅 B 被更新，A 保持冻结
      grad_B_masked = grad_loss @ y^T ⊙ mask   # 只有 k 列有非零梯度
      B = B - lr * grad_B_masked
    ```
  - FlyLoRA 与 baseline 的关键张量计算对比：
    - LoRA(r=8):  W0·x + B_{m×8}·(A_{8×n}·x)，激活参数 2·d·8
    - LoRA(r=32): W0·x + B_{m×32}·(A_{32×n}·x)，激活参数 2·d·32
    - Split-LoRA(4×8): W0·x + Σ_i G(x)_i·B_i·(A_i·x)，4个8-rank expert + router W_g∈R^{4×n}，激活参数 2·d·8 + d·4
    - FlyLoRA(k=8): W0·x + Σ_{i∈I_topk} b_i·(a_i·x)，仅激活 8 个 rank-1 expert 在 B，激活参数 d·8（无 router 开销）
  - 训练配置（Appendix C）：Total rank r=32, activated k=8, scaling factor α=64, target modules {q,k,v,o,gate,down,up}_proj, optimizer AdamW, warmup ratio 0.01, gradient accumulated batch 128, dropout 0.0。数据集特定配置见 Table 19（epochs 1~20, max seq len 128~512, micro batch size 8, learning rate 3e-4~6e-4）。混合精度训练（16-bit）。

## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  - FloE 提出 **Hybrid Compression（混合压缩）** 方案，对 MoE expert 内部的三组投影矩阵采用不同压缩策略：
    1. **Contextual Sparsification（上下文化稀疏化，Section 3.2.1）**：对 gate projection W_gate 和 down projection W_down 执行基于激活幅值的剪枝。具体为对 up projection 的输出激活 a_up = x·W_up 按阈值 t 做稀疏化 S_t(a_up)，保留 |a_up|≥t 的通道，将对应 gate projection 列和 down projection 行一同剪枝。阈值 t 由目标稀疏率 k 从样本数据集的激活绝对值经验CDF反向确定（t = min{t': F(t') ≥ k}）。
    2. **Ultra-Low-Bit Quantization（超低位量化，Section 3.2.2）**：仅对 up projection W_up 施加 INT2 HQQ 量化，因为 up projection 对量化最不敏感（在 INT2 下 perplexity 仅为 gate 量化的 46% 和 down 量化的 27%）。
    3. **Dual Sparsity Predictors（双稀疏预测器，Section 3.3）**：
       - Inter-expert 学习型预测器：利用当前层 hidden state 和历史 expert 选择轨迹，通过单层/双层 MLP（32K~2M 参数）预测下一层激活 expert，平均 precision 0.88。
       - Intra-expert 复用型预测器：用当前层 hidden state 与下一层 W_up（复用）直接做矩阵乘法，近似估计 up projection 输出激活，预计算上下文稀疏分布，平均 recall 0.95，零额外内存开销。
  - 实验比较：在 7 个下游任务（ARC-Easy/Challenge, BoolQ, SciQ, OpenBookQA, Winogrande, MMLU@5）上对比 CATS（激活稀疏化）、CHESS（通道级阈值稀疏化）、HQQ 量化；在 WikiText-2 上评估 perplexity；在 ShareGPT 上评估端到端生成速度。

- 硬件平台是什么，配置是什么。
  - 端到端延迟测试：GeForce RTX 3090（24GB VRAM），64核 CPU @2.3GHz，256GB DRAM，PCIe 4.0。
  - 单 expert 延迟测试：H100, A100, A6000, GeForce RTX 3090。

- 模型是什么。数据集和bench分别是什么。
  - 主要模型：Mixtral-8×7B（FP16，每层 8 expert，每 token 激活 2 expert）。
  - 验证模型：Phi-3.5-MoE-instruct, DeepSeek-MoE-16B-Base, Qwen1.5-MoE-A2.7B, DeepSeek-V2。
  - 数据集：C4（稀疏性分析和单 expert 延迟），WikiText-2（perplexity），ShareGPT（端到端生成效率），EleutherAI LM Harness（下游任务：ARC-Easy/Challenge, BoolQ, SciQ, OpenBookQA, Winogrande, MMLU@5）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文发表于 ICML 2025。论文未明确给出代码开源链接。可参考的相关开源项目：HQQ（https://github.com/mobiusml/hqq）、CATS（https://openreview.net/forum?id=v3w2a7EInO）。
  - 算法 pipeline 伪代码（基于 Algorithm 1）：
    ```
    Input: hidden state x, sparse threshold t_ij, expert weights {W_gate, W_down^T, W_up}
    1. v = x @ W_up                          # 全精度 up projection
    2. mask = (|v| > t_ij)                   # 按阈值生成稀疏掩码
    3. x' = SiLU(x @ W_gate[mask]) ⊙ v[mask] # 仅加载被掩码选中的 gate 列
    4. y = (W_down^T[mask] @ x')^T           # 仅加载被掩码选中的 down 列（列主序转置存储）
    Return: y
    ```
    核心张量操作链：hidden state x (1×4096) → up projection 全精度计算 → 幅值阈值化 → 掩码选择 gate 列和 down 行 → 稀疏 gate GEMV + SiLU → Hadamard 积 → 稀疏 down GEMV → 输出。结合混合压缩，实际传输量从 ~300MB/expert（FP16）降至约 ~32MB/expert（9.3×压缩）。

## HMoE: Heterogeneous Mixture of Experts for Language Modeling

- 属于算法pipeline的实现是什么？实验比较什么？
  - HMoE 提出异构 Mixture of Experts 预训练语言模型，核心实现：(1) 为 MoE 层中的不同 expert 分配不同的 FFN hidden dimension（即不同大小的 expert），以引入专家异质性。每个 expert FFN 沿用 LLaMA 的 SiLU-gated 设计：e_i(x) = W_{o,i} · (SiLU(W_{g,i} · x) ⊙ (W_{p,i} · x))，其中 W_{g,i} ∈ R^{h_input × h_ffn,i}, W_{p,i} ∈ R^{h_input × h_ffn,i}, W_{o,i} ∈ R^{h_ffn,i × h_input}，通过改变 h_ffn,i 控制各 expert 大小；(2) 提出 P-Penalty loss (Parameter Penalty) L_P-Penalty = N · Σ M_i · P̂_i，其中 M_i = (1/T) Σ 1{e_i ∈ E^t} × h_ffn,i，将 expert 大小纳入损失，鼓励激活更小的 expert，防止 router 过度偏好大 expert；(3) 配合 Top-P routing 时额外使用 router entropy loss L_entropy = N · Σ P_i · log(P_i) 抑制激活 expert 数量增长；(4) 探索三种 expert 大小分布策略：Geometric（几何级数如 {1,2,4,8,16,32,64,128}）、Arithmetic（等差级数如 {9,11,13,15,17,19,21,23}）、Hybrid（混合如 {1,1,1,1,2,2,4,4}）。
  - 实验比较：(1) HMoE vs Homogeneous MoE vs Dense：0.4B 和 3B 总参数量级，在等 FLOPs 预算下对比 Top-K (k=2) 和 Top-P (p=0.6) 路由；(2) isoFLOP 分析：不同训练 FLOPs 下 HMoE vs Homogeneous MoE 的最优激活参数量和 loss 曲线；(3) Ablation：P-Penalty loss vs load balancing loss、三种 expert 分布策略 (geometric/arithmetic/hybrid)、不同 expert 大小方差（最大/最小 expert size ratio）的影响；(4) Expert 分析：expert 间相似度 (Wasserstein distance)、协同度 (KL divergence)、不同难度 token 的 expert 激活模式、层间激活参数分布。

- 硬件平台是什么，配置是什么。
  - 训练硬件：NVIDIA A800 (80GB 显存) 或 H800 (80GB 显存) GPU。
  - 训练加速：使用 DeepSpeed Zero2 策略进行分布式训练，配合 gradient checkpointing 节省 GPU 显存。
  - 高效训练支持：Megablocks (Gale et al. 2022) 实现 block-sparse 矩阵乘法 kernel 处理不规则形状 expert 的批量计算；ES-MoE (Kim et al. 2024) 引入 expert-wise offloading 和动态 expert 放置策略（论文在 Related Work/Efficient Training 中引述）。

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：基于 LLaMA Transformer decoder-only 架构。Dense-0.4B: 12 layers, FFN hidden=12288, 12 heads×64 dim。Dense-1B: 12 layers, FFN hidden=32768, 16 heads×64 dim。MoE/HMoE-0.4B: 12 layers, 8 experts/layer, 总 expert hidden=12288。MoE/HMoE-3B: 12 layers, 8 experts/layer, 总 expert hidden=32768。Attention 层规格与对应 Dense 一致。LLaMA2 tokenizer, vocab=32000。
  - HMoE expert 大小分布（主实验）：arithmetic 策略，相对大小 {9, 11, 13, 15, 17, 19, 21, 23}（归一化后 expert 实际 hidden dim 分别为 2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888 for 3B model）。
  - 训练数据集：RedPajama（开源），包含 Common Crawl, C4, GitHub, Wikipedia, Books (the Pile), arXiv, StackExchange。
  - 评估 benchmark（6 个）：PIQA（物理常识）、hellaswag（句子补全常识推理）、BoolQ（是非问答）、ARC-Easy（科学推理）、winogrande（代词消歧）、SIQA（社交常识推理）。评估协议：基于相同训练 FLOPs 预算（7×10^19 和 2.6×10^20）而非训练 token 数，因为不同方法激活参数量不同。使用 lm-evaluation-harness (Gao et al. 2021) 框架。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声明 "Codes will be released upon acceptance"，当前未公开代码。
  - HMoE 算法 pipeline 伪代码（基于论文公式和 LLaMA FFN 设计）：

```python
# === HMoE Decoder Layer 前向传播 ===

# 输入: x [B, S, h_input], 当前层 hidden states
# N: 专家数量
# h_ffn_list: 各专家 hidden dim 列表，如 [2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888]

# 1. Router (gating)
P = softmax(x @ W_r)  # W_r: [h_input, N], P: [B, S, N]

# 2. Top-K 或 Top-P 路由选择
if routing == "topk":
    top_k_vals, top_k_idx = topk(P, k=2, dim=-1)  # [B, S, 2]
    top_k_vals = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)  # 归一化
    # 被选中专家的 gate 值 = 归一化后的 top_k_vals，未选中 = 0

elif routing == "topp":
    P_sorted, sort_idx = sort(P, descending=True, dim=-1)
    # 如果 P_sorted[0] > p: 选 1 个
    # 否则累加直到 cumsum >= p
    t = argmin_k(cumsum(P_sorted) >= p)  # t 个专家
    selected_idx = sort_idx[:, :, :t]
    gate_vals = P.gather(dim=-1, index=selected_idx)
    gate_vals = gate_vals / gate_vals.sum(dim=-1, keepdim=True)

# 3. Expert Computation (异构)
output = zeros([B, S, h_input])
for i in range(N):
    mask_i = (expert i is selected for this token)  # [B, S]
    if mask_i.sum() == 0: continue
    x_i = x[mask_i]  # [n_i, h_input]
    gate_i = gate_vals[mask_i][对应于 expert i 的 gate 值]  # [n_i]

    # LLaMA-style FFN with expert-specific hidden dim h_ffn,i
    # W_g,i: [h_input, h_ffn,i], W_p,i: [h_input, h_ffn,i], W_o,i: [h_ffn,i, h_input]
    h_i = SiLU(x_i @ W_g,i) * (x_i @ W_p,i)  # [n_i, h_ffn,i]
    expert_out_i = h_i @ W_o,i  # [n_i, h_input]
    output[mask_i] += gate_i.unsqueeze(-1) * expert_out_i

# 4. Auxiliary Losses
# P-Penalty Loss (替代传统 load balancing loss):
L_pp = N * sum_i(M_i * P_hat_i)
# 其中 M_i = (1/T) * sum_t(1{e_i activated for token t} * h_ffn,i)
# P_hat_i = (1/T) * sum_t(P_i,t)
# h_ffn,i 是 expert i 的 hidden dim，大 expert 贡献更大 penalty

# Router Entropy Loss (仅 Top-P):
L_entropy = N * sum_i(P_i * log(P_i))  # 防止激活过多 expert

# 最终训练 loss:
L_total = L_lm + α * L_pp + β * L_entropy  # α=0.1, β=3e-2 (Top-P only)
```

张量计算示意（expert i, h_ffn,i=2304 vs expert j, h_ffn,j=5888）:
- Small expert i: input [n_i, 4096] → W_g [4096, 2304] → gate [n_i, 2304]; W_p [4096, 2304] → up [n_i, 2304]; SiLU(gate) ⊙ up → hidden [n_i, 2304]; W_o [2304, 4096] → output [n_i, 4096]。参数量 ≈ 3 × 4096 × 2304
- Large expert j: input [n_j, 4096] → W_g [4096, 5888] → gate [n_j, 5888]; 参数量 ≈ 3 × 4096 × 5888

异构 expert 使用 Megablocks block-sparse kernel 进行批量计算，避免不同形状 expert 的 padding 开销。

## HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

- 属于算法pipeline的实现是什么？实验比较什么？
  - HEXA-MoE 提出三个算法层面的 MoE 训练优化：(1) **Expert-Specific Operators**：用三个专用算子——ESMM (Expert-Specific Matrix Multiplication)、ESS (Expert-Specific Summation)、ESTMM (Expert-Specific Transposed Matrix Multiplication)——替代传统 GeMM 或 grouped GeMM 接口，实现 in-place 计算，消除 token padding/discarding 带来的冗余 FLOPs；(2) **Data-Centric 与 Model-Centric 双模式并行**：对大规模 workload 使用 data-centric 配置（tensor parallelism 切分 FFN intermediate size，各设备 all gather 完整 MoE 参数后本地计算），对小规模 workload 使用 model-centric 配置（all gather 同步本地数据批次，各设备计算本地参数 chunk）。引入 pipeline-shared cache 解决 data-centric 模式下 backward pass 需保存全部 MoE 参数导致内存膨胀的问题；(3) **Heterogeneous-Aware Expert Allocation**：基于各设备计算能力（通过 benchmark 测量平均延迟）按反比分配 workload——data-centric 下调整各设备 local batch size，model-centric 下调整各设备 FFN intermediate sub-dimension。
  - 实验比较：(1) Memory Analysis: HEXA-MoE vs Tutel vs MegaBlocks 的 GPU 内存占用，Swin-MoE Small/Base, 8 global experts, top-1~top-8 routing；(2) Latency Analysis: 平均每步训练延迟对比，4 homogeneous GPUs, 4 experts, 不同 batch size；(3) Data-Centric vs Model-Centric: 不同 batch size 下的延迟对比；(4) Heterogeneous Experiments: 异构设备（TITAN RTX + RTX 2080 Ti）上 data-centric 和 model-centric 配置下不同 workload 分配比例的延迟对比；(5) Ablation: expert-specific operators、pipeline-shared cache、fused kernel、data-/model-centric、memory optimization 各组件的 memory footprint 和 latency 贡献分解。

- 硬件平台是什么，配置是什么。
  - 同构机器 M_homo：CPU 2× Intel Xeon Platinum 8352V 2.10GHz, 1008 GB RAM；GPU 4× NVIDIA GeForce RTX 4090 (24 GB)。同构实验均在 M_homo 上进行。
  - 异构机器 M_hete：CPU 2× Intel Xeon Gold 6130 2.10GHz, 62.5 GB RAM；GPU D0: 1× NVIDIA TITAN RTX (24 GB), D1: 1× NVIDIA GeForce RTX 2080 Ti (11 GB)。
  - 软件栈：PyTorch + NCCL 通信后端，automatic mixed precision 训练。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Swin-Transformer-MoE（Small 和 Base 两种规模），遵循 Tutel (Hwang et al., 2023) 的配置。全局 experts 数=4 或 8，routing 从 top-1 到 top-8。默认使用 top-k routing + atomicAdd 聚合各 expert 输出。
  - Benchmark：Swin-MoE 训练过程作为 benchmark，评估指标为平均 GPU 内存占用 (GB) 和每训练步平均延迟 (s)。Latency 实验记录 2k steps 的平均值，Memory 实验记录各设备平均 GPU 内存占用。
  - 计算能力 benchmark（异构实验）：Algorithm 5 中的 proxy task——1024 次循环大矩阵乘法（size=2048），测量完成时间作为计算能力指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/UNITES-Lab/HEXA-MoE（另有作者个人仓库 https://github.com/luoshuqing2001/hexa_moe）
  - 算法 pipeline 伪代码（以 top-1 routing 为例）：

```python
# === HEXA-MoE Forward & Backward with Expert-Specific Operators ===
# 输入: x [N, D_i], 路由选择 R(x) [N], 权重 W1 [E, D_i, D_mid], W2 [E, D_mid, D_o]
# Forward:
y1 = ESMM(x, W1, b1, R(x))       # [N, D_mid], 每个 token 用其路由 expert 的 W1 计算
y2 = F(y1)                         # 激活函数 (如 GELU)
y  = ESMM(y2, W2, b2, R(x))       # [N, D_o]

# Backward (auto-diff 提供 ∂ℓ/∂y):
∂ℓ/∂b2 = ESS(∂ℓ/∂y, R(x))         # [E, D_o], 按 expert 累加
∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x))   # [E, D_mid, D_o]
∂ℓ/∂y2 = ESMM(∂ℓ/∂y, W2^T, null, R(x))  # [N, D_mid]
∂ℓ/∂y1 = ∂ℓ/∂y2 ⊙ F'(y1)          # element-wise
∂ℓ/∂b1 = ESS(∂ℓ/∂y1, R(x))        # [E, D_mid]
∂ℓ/∂W1 = ESTMM(x, ∂ℓ/∂y1, R(x))   # [E, D_i, D_mid]
∂ℓ/∂x  = ESMM(∂ℓ/∂y1, W1^T, null, R(x))  # [N, D_i]
```

  - Expert-Specific Operators 定义：
    - ESMM(x, W, b, R): y_i = x_i @ W_{R(x_i)} + b_{R(x_i)}，每个 token 仅与其路由 expert 的权重做矩阵乘法
    - ESS(x, R): y[e] = Σ_{i: R(x_i)=e} x_i，按 expert 分组累加
    - ESTMM(x1, x2, R): y[e, i, j] = Σ_{m: R(x_m)=e} x1[m,i] · x2[m,j]，expert-wise 外积累加

  - Top-k routing 扩展（k>1）：对 k 个路由选择分别执行 ESMM，最终输出为 k 个 ESMM 结果的累加（使用 atomicAdd）。中间结果 tensor 的内存分配扩展为 k 倍。

  - Data-Centric 配置（大规模 workload）：
    ```
    # 各设备沿 FFN intermediate size 切分 expert 权重
    # 每个 MoE layer: all gather 完整参数 → ESMM 本地计算 → 下一层
    # pipeline-shared cache: 每设备分配额外 HBM 区域动态缓存 gathered shards
    # all gather 与 attention/router 计算 overlap
    ```

  - Model-Centric 配置（小规模 workload）：
    ```
    # 各设备沿 FFN intermediate size 切分 expert 权重
    # 每个 MoE layer: all gather 数据批次 → ESMM 用本地参数 chunk 计算 → all reduce sum 聚合
    ```

  - Heterogeneous Allocation:
    ```
    # 先测量各设备计算能力 t_i (proxy task 延迟)
    # Data-centric: B_i = (1/t_i) / Σ(1/t_j) · B_global
    # Model-centric: h_i = (1/t_i) / Σ(1/t_j) · H
    ```

  - 对比 baseline (Tutel) 的 pipeline：Tutel 使用 GeMM + token padding/discarding → dispatch/combine + all-to-all 通信；HEXA-MoE 使用 ESMM in-place 计算 + tensor parallelism 替代 expert parallelism → 无 token padding、无冗余 FLOPs、无 all-to-all 通信。

## Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

> ⚠️ 注意：此论文目录名为 "KTransformers Unleashing the Full Potential of CPUGPU Hybrid Inference for MoE Models"，但 PDF 实际内容为轮式装载机自主导航的机器人控制论文（ICRA 2025，arXiv:2409.15717）。按"提出新的算法模型"归类为算法pipeline 层次，为弱匹配。

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出 **Goal-Conditioned Actor-Critic MPC**：将 Lyapunov-based Actor-Critic RL（ALAC 算法）训练的 critic 网络作为非线性 MPC 的 terminal cost 和 stage cost，替代传统的高层轨迹规划器。核心算法组件：
    1. **Lyapunov-based RL 训练（ALAC）**：训练 critic L_ψ 满足 sampling-based Lyapunov 稳定性条件（Theorem 3.1）。引入 gradient penalty（Eq. 16）鼓励 1-Lipschitz，为下游 MPC 提供平滑优化景观。
    2. **Critic 作为 Terminal Cost**：l_f(x_N,g) = L_ψ(x_N, 0, g)，动作替换为零向量。
    3. **Critic 二阶 Taylor 近似作为 Stage Cost**：l(x_n,u_n,g) = Δt · L̃(x_n,u_n,g)，L̃ 为 critic 在上一 MPC 解处的二阶 Taylor 展开（Eq. 25），缓解仅用 terminal cost 时的犹豫行为。
    4. **输入延迟补偿**：将 MPC 初始状态向前传播 200ms 匹配执行器延迟。
  - 实验比较：
    - **Baseline**：基于 CasADi + IPOPT 的非线性轨迹优化（Eq. 27-28），T=25s horizon，direct collocation 离散化（采样 200ms），求解 >5s（AMD Ryzen 3900x）。
    - **场景 (a)/(b)**：真机实验（Avant 635），短装载循环和 180° 紧凑转弯，收敛时间与 baseline 相当或更优。
    - **场景 (c)**：多障碍物导航仅仿真（N=20 使 MPC 求解 200-300ms > 实时 100ms 要求）。
    - **128 场景仿真**：Actor-Critic MPC 平均收敛 10.92s vs baseline 14.33s（快 23.80%），全部成功。
    - **指标**：收敛时间（||x-g|| < 0.1）、速度跟踪误差。

- 硬件平台是什么，配置是什么。
  - **真机**：NVIDIA Jetson AGX Orin（32GB unified memory，12 cores），搭载于 Avant 635 小型轮式装载机。
  - **Baseline**：桌面 AMD Ryzen 3900x CPU（离线求解，>5s）。
  - **执行器**：液压转向 + 柴油发动机，输入延迟约 200ms。

- 模型是什么。数据集和bench分别是什么。
  - **Actor/Critic 网络**：前馈 NN，层结构 (48,96,144,96,48)，SoftPlus 激活。Critic 构造 L=Q·Q^T（确保正输出）。编码器将绝对位姿转为相对位姿，heading 用 sin/cos 编码。Actor 为 Gaussian policy（tanh + 缩放至 a_max）。
  - **运动学模型**：6D 状态 [x_f, y_f, θ_f, β, β̇, v_f]，2D 控制 [β̈, a_f]。4 阶 Runge-Kutta 离散化（Δt=0.2s）。
  - **场景**：无标准 benchmark。真机：(a) 短装载循环，(b) 180° 转弯。仿真：(c) 多障碍物场景，128 随机目标位姿。
  - **RL 训练**：PyTorch + stable-baselines3。MPC：CasADi + Acados + L4CasADi（NN 集成），SQP-RTI + HPIPM QP solver，N=10（N=20 for obstacles）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **未找到公开代码仓库**（搜索 arXiv:2409.15717 + "Aleksi Mäki-Penttilä" + GitHub 无匹配）。
  - **算法 pipeline 伪代码**：

```
=== 阶段一：RL 离线训练（仿真环境） ===
Input: 运动学模型 f(x,u), 目标 goal g
Output: 训练好的 critic L_ψ (满足 Lyapunov 条件)

1. 初始化: Actor π_φ, Critic L_ψ = Q_ψ·Q_ψ^T
2. for each episode:
3.     s0 ← 重置; g ← 采样目标
4.     for each step t:
5.         a_t ~ π_φ(s_t, g)              # Gaussian policy + tanh 限幅
6.         s_{t+1} = f(s_t, a_t)           # 4阶 Runge-Kutta, Δt=0.2s
7.         存储 (s_t,a_t,c_t,s_{t+1},g) → replay buffer D
8.         从 D 采样 mini-batch
9.         # Critic 更新 (Eq. 12-16):
10.        L_target = c(s,a,g) + γ·L̄_ψ̄(s', a', g)
11.        Ĵ_c = E[(L_ψ - L_target)²] + ρ·E[(1-||∇L_ψ||₂)²]
12.        # Actor 更新 (Eq. 17-18):
13.        J_φ = λ_e·(log π_φ + H) + λ_l·ΔL_ψ
14.        # Lagrange multipliers 自适应 (Eq. 19):
15.        当 λ_l < 1 → L_ψ 为有效 Lyapunov 函数
16. 训练至 λ_l 收敛 → 0.8 (实验设定)

=== 阶段二：MPC 在线求解（真机 Jetson，每步 <100ms） ===
Input: 当前状态 x_init, 目标 g, 训练好的 critic L_ψ

1. 输入延迟补偿: x_init ← propagate(x_init, 200ms) via Eq. 5
2. 构建 NLP (Eq. 20):
   min_{x,u} Σ_{n=0}^{N-1} [Δt · L̃(x_n,u_n,g)] + L_ψ(x_N,0,g)
   s.t. x_{i+1}=f(x_i,u_i), 状态约束 (Eq. 6), 控制约束 (Eq. 21)
        可选: 障碍物约束 (Eq. 22)
3. Stage cost L̃: critic 在上一解 (x*,u*) 处的二阶 Taylor 近似 (Eq. 25)
   设 z_n=[x_n;u_n], z*_{n+1}=[x*_{n+1};u*_{n+1}]:
   L̃(x_n,u_n,g) = ∂L/∂z_n · (z*_{n+1}-z_n) + 0.5 ∂²L/∂z_n² · (z*_{n+1}-z_n)²
4. Warm-start SQP-RTI solver (HPIPM QP solver)
5. solver → 最优轨迹 [x₀..x_N], 取 x₁ 的 (β̇,v_f)
6. 发送 (β̇_cmd, v_f_cmd) → 低层反馈控制器 → 液压/发动机
```

  - **张量计算流**（单 MPC 迭代，N=10）：
    1. 编码：绝对位姿 → 相对位姿 (dx, dy, sin(θ-θ_g), cos(θ-θ_g))
    2. Terminal: x_N → NN [48→96→144→96→48] SoftPlus → Q → L=Q·Q^T → scalar
    3. Stage: 在 (x*,u*) 处计算 ∂L/∂z 和 ∂²L/∂z²（各 N 步），构造二次型（比直接 NN forward 更轻量）
    4. SQP-RTI: 序列二次规划 → 输出 x₁ 的 β̇, v_f

> ⚠️ **近似层次匹配说明**：此论文属于机器人控制/AI 决策领域，非典型 AI 系统/LLM 推理领域。因"提出新的算法模型"（Goal-Conditioned Actor-Critic MPC）归类为算法pipeline，为弱匹配。其他层次（Serving调度/编译框架/kernel调度/硬件架构/芯片设计）均不适用。论文实验基于 kinematics simulation + 真机测试（Avant 635 装载机），硬件为 NVIDIA Jetson AGX Orin，无 GPU 集群或 AI 加速器。

## FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出三种算法层面的训练优化，用于加速大规模 MoE 模型的分布式训练：
    1. **Dynamic Shadowing（动态影子策略，Section 4.1）**：在运行时按迭代动态选择热门 expert，将其模型参数广播复制到所有 worker，替代原本的大量 token 输入传输。通过性能模型比较 `Lat_imbl`（不均衡下的延迟）和 `Lat_shadow`（影子化后的延迟），当满足条件 `B_max > rαH` 或 `(3(B_max - B'_max)αH) / (rαH - B_max) > P/W_net` 时启用。伪代码见 Algorithm 1（SelectShadowExperts）。
    2. **Fine-grained Smart Scheduling（细粒度智能调度，Section 4.2）**：将 all-to-all 通信拆分为分组 pairwise exchange 操作，计算也相应拆分。创建独立的 computation stream 和 communication stream，将 n 个 group 的 S（send）、C（compute）、R（receive）操作按依赖关系重新排列，使通信与计算异步并行执行。将最快的操作 S_{i,0} 和 R_{i,n-1} 放在首尾以最小化开销。
    3. **Topology-aware Gate（拓扑感知门控，Section 4.3）**：修改 expert 选择策略，限制跨节点 tokens 数量上限 L = (W_net / (M·W_local))·B，超过 L 的 tokens 在本地节点内重新选择 expert，减少上层网络链路拥塞。
  - 实验比较：
    - 整体加速比 vs ZeRO stage 1/2/3（数学等价，不修改 expert 选择）和 FastMoE（expert parallelism baseline）
    - 动态影子策略单独效果（迭代延迟 vs 影子化 expert 数量）
    - 智能调度单独效果（每层实际加速比 vs 理论上界 `(Lat_comm + Lat_comp) / max{Lat_comm, Lat_comp}`）
    - 拓扑感知门控 vs GShard、BASE Layer 的收敛速度（training loss vs time/iterations）
    - 性能模型预测准确度（computation/communication 单独 + end-to-end，R²=0.987/0.967）

- 硬件平台是什么，配置是什么。
  - **johnny 集群**：16× NVIDIA Tesla V100-PCIE 32GB GPU，2 节点（每节点 8 GPU），GPU 通过 PCIe switch 连接 2 个 CPU socket。Infiniband EDR 但因缺少 ×16 PCIe 插槽实际带宽降级至 50Gb/s。
  - **trevor 集群**（天河二号超算分区）：64× NVIDIA Tesla V100-SXM2 32GB GPU，16 节点（每节点 4 GPU），节点内 NVLink 互连（异构环，半数边双链路双带宽）。Infiniband EDR 100Gb/s 节点间通信。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    | 模型 | 参数量 | 层数 | Experts | H（hidden） | α（MLP中间维度比） | 集群 |
    |------|--------|------|---------|-------------|-------------------|------|
    | MoE-GPT-S | 0.86B | 12 | 16 | 1024 | 2 | johnny |
    | MoE-GPT | 3.42B | 12 | 16 | 2048 | 2 | johnny |
    | MoE-GPT-L | 13.7B | 12 | 16 | 4096 | 2 | johnny |
    | MoE-BERT-Deep | 1.71B | 24 | 16 | 1024 | 2 | johnny |
    | MoE-BERT-Deep-L | 27.4B | 24 | 16 | 4096 | 2 | johnny |
    | MoE-BERT-Wide | 3.27B | 12 | 64 | 1024 | 2 | trevor |
    | MoE-BERT-Wide-L | 13.1B | 12 | 64 | 2048 | 2 | trevor |
  - **数据集**：
    - 性能评测使用 expert selection dataset（从真实训练过程中记录的 token-to-expert 分配），可在 https://pacman.cs.tsinghua.edu.cn/laekov/fastermoe-data/dumps.tgz 下载，为 16 experts 生成。
    - 训练实验使用预处理后的 wikidataset，可在 https://pacman.cs.tsinghua.edu.cn/laekov/fastermoe-data/wikidataset.tgz 下载。
  - **Bench**：无标准 benchmark 数据集。性能延迟测量基于重放 expert selection dataset，收敛实验基于 wikidataset 训练 loss 曲线。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：源代码开源于 https://github.com/thu-pacman/FasterMoE，artifact evaluation 脚本在 https://github.com/laekov/fastermoe-ae。
  - **系统依赖**：基于 FastMoE [7] 实现，依赖 CUDA、NCCL（≥2.9.9）、PyTorch（v1.10.0）。Baseline 系统包括 Megatron-LM（修改 MLP 模块用于 MoE）、DeepSpeed v0.4.4（ZeRO Optimizer）、FairSeq（BASE Layer）。
  - **算法 pipeline 解释（以动态影子策略为例，伪代码对应 Algorithm 1）**：

  ```
  # Dynamic Shadowing 核心流程（Algorithm 1: SelectShadowExperts）
  # 输入: B[N], 每个worker的batch size（token数量）
  # 输出: E_s, 需要影子化的expert集合
  # 每iteration在所有worker上执行

  def SelectShadowExperts(B):  # B[w] = sum_i T[i][w]
      B_max = max(B)
      c_min = Lat_imbl(B_max)    # 当前不均衡配置的延迟
      E_s = []                    # 影子化expert集合

      # 按local batch大小降序遍历
      for i, B_i in sorted(enumerate(B), key=lambda x: -x[1]):
          B_i = T[i][i]           # 保留本地tokens
          for j != i:             # 其他worker的tokens: 在本地计算
              B_i += T[i][i]      # 影子化后在本地执行

          B_max_prime = max(B)    # 影子化后的最大batch
          c = Lat_shadow(len(E_s)+1, B_max_prime)

          if c < c_min:           # 影子化降低延迟则采纳
              c_min = c
              E_s.append(i)
          else:
              return E_s          # 一旦不改善即停止
      return E_s

  # 影子化延迟模型 (Eq. 8):
  # Lat_shadow(r, B') = max_w{3 * 4*B'_w*α*H²/P} + 2r * 2αH²/W_net
  #   - 第一项: 影子化后均衡的computation
  #   - 第二项: 广播r个expert参数的通信开销（forward 1次 + backward reduce 1次）

  # 影子化启用条件 (Eq. 9/10):
  # 条件1: B_max > rαH  (token传输开销 > 模型传输开销)
  # 条件2: 3(B_max - B'_max)αH / (rαH - B_max) > P/W_net  (减少的computation > 增加的communication)
  ```

  **拓扑感知门控算法（Section 4.3）**：

  ```
  # 拓扑感知门控：限制跨节点tokens
  # L = (W_net / (M * W_local)) * B
  #   W_net: 跨节点带宽, W_local: 节点内带宽
  #   M: 每节点worker数, B: batch size

  def TopologyAwareGate(tokens, scores, L):
      for each token x with top-k expert scores:
          if expert_is_on_remote_node(x.best_expert):
              remote_candidates.append((x, x.score))
  
      # 仅允许分数最高的L个跨节点
      remote_candidates.sort(key=lambda t: -t[1])
      allowed = remote_candidates[:L]
  
      # 其余token在本地节点内重新选择expert
      for token in remote_candidates[L:]:
          token.reselect_expert(local_node_only=True)
  ```

  **智能调度张量计算流（Section 4.2）**：
  ```
  # n个group, 每个worker在step j执行:
  # S_{i,j}: 发送tokens到group (i+j) mod n，接收来自group (i-j) mod n
  # C_{i,j}: 对来自group (i-j) mod n的tokens用本地expert计算
  # R_{i,j}: 接收本地tokens输出从group (i+j) mod n，发送输出到group (i-j) mod n
  #
  # comm stream: S_{i,0}, S_{i,1}, ..., S_{i,n-1}, R_{i,0}, ..., R_{i,n-1}
  # comp stream: C_{i,0}, C_{i,1}, ..., C_{i,n-1}
  # 两stream并行执行，依赖关系由数据依赖保证
  ```

## Faster MoE LLM Inference for Extremely Large Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文对 fine-grained MoE（DeepSeek-V2-Lite / DeepSeek-V3）提出两种推理阶段的算法优化策略：
    1. **Inference Time Expert Skipping（Section 5）**：在推理时按层级别动态减少每 token 的激活 expert 数 na。通过定义一个四元组 (b, h, e, p) 来描述跨层的 expert 分配——首层选 b 个、第 p 层选 h 个、末层选 e 个、其余层线性插值——实现 ascending/descending/peak/valley 等多种分配模式。探索了 softmax 激活（V2）和 sigmoid 激活（V3）两种路由机制下的不同行为。
    2. **Pre-Inference Expert Pruning（Section 6）**：在推理开始前减少总 expert 数 ne，通过多种选择策略（Random、Structured/Odd-Even-FirstHalf-LastHalf、Activate Count、Soft Count）从 ne 个 expert 中选择 ne' 个保留，其余丢弃不加载。
    3. **Roofline-based 效率分析（Section 4）**：从 Roofline 模型出发，推导 MoE 层的 I/O、FLOPS 和算术强度公式，分析 MoE 相比 FFN 的批次效应弱化原因——因 token 间很少复用同一 expert，增加 token 数反而增加额外 expert 参数加载开销。
  - 实验比较：
    - Expert skipping 效率：不同 na（2-8）在不同并发度（2-768）下的 throughput 和 speedup ratio
    - Expert skipping 性能：不同 (b,h,e,p) 四元组策略在 ARC-C, ARC-E, BoolQ, OBQA, RTE, WinoGrande 上的 Avg 得分
    - Expert pruning 效率：不同 ne（8-64）在不同并发度（2-784）下的 throughput 和 speedup
    - Expert pruning 性能：不同选择策略（Random/Structured/Activate Count/Soft Count）在不同 ne' (16/32/48) 下的 benchmark 得分

- 硬件平台是什么，配置是什么。
  - **DeepSeek-V2-Lite**：2× NVIDIA Tesla A800 80G PCI-e, Intel Xeon Silver 4314 CPU @ 2.40GHz
  - **DeepSeek-V3**：8× NVIDIA Tesla H200 141G SXM5, Intel Xeon Platinum 8558 CPU @ 2.10GHz
  - 效率测试固定 1024 input + 1024 output tokens，使用 sglang v0.4.4 post 1 + sglang.bench

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - DeepSeek-V2-Lite: 16B params, ne=64, na=6, d=2048, de=1408, ds=10944, da=8448, softmax routing
    - DeepSeek-V3: 671B params, ne=256, na=8, d=7168, de=2048, ds=18432, da=16384, sigmoid routing
  - **Benchmark 数据集**：ARC-Challenge, ARC-Easy, BoolQ, OpenBookQA (OBQA), RTE, WinoGrande（Avg 为 6 个 benchmark 的平均，baseline=36 为纯猜测基线）
  - 效率评测无特定 benchmark 数据集，使用随机生成 token 序列

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供独立开源代码仓库。基于开源 sglang 和 PyTorch 实现。
  - **算法 pipeline 解释**：

  **Expert Skipping 算法（Section 5.2）**：

  给定模型 L 层，原始每层激活 na 个 expert。定义四元组 (b, h, e, p) 控制跨层分配：

  ```
  # 输入: b (首层 expert 数), h (第 p 层 expert 数), e (末层 expert 数), p (峰值/谷值位置)
  # 输出: 每层的 na(l) for l in 0..L-1

  na[0] = b                      # 首层
  na[p] = h                      # 第 p 层
  na[L-1] = e                    # 末层

  # 线性插值填充中间层
  for l in 1..p-1:
      na[l] = b + (h - b) * l / p

  for l in p+1..L-2:
      na[l] = h + (e - h) * (l - p) / (L - 1 - p)
  ```

  **四种典型分配模式（Figure 6）**：
  - Ascending: b < h < e（expert 逐层递增，V3 最优）
  - Descending: b > h > e（expert 逐层递减，V2-Lite 最优）
  - Peak: h > b 且 h > e（中间层最多 expert）
  - Valley: h < b 且 h < e（中间层最少 expert）

  **MoE 推理效率模型（Section 3.1, Eq. 3）**：
  ```
  # FFN/MoE 的 Memory I/O, FLOPS, Arithmetic Intensity
  I/O(d, di, L) = 3 * di * d + 2 * L * (d + di)
  FLOPS(d, di, L) = 6 * L * (di * d)
  AI(d, di, L) = (6 * L * di * d) / (3 * di * d + 2 * L * (di + d))

  # 对于 MoE, di = da = de × na (激活 expert 的总中间维度)
  # 但实际 I/O 还包括所有被选中 expert 的参数加载
  ```

  **Expert Skipping 前向计算**（以 na'=2, 原始 na=6 为例）：
  ```
  # 输入: hidden_states h ∈ R^(d), batch_size B
  # 原始: na=6, router 选 top-6 expert
  # Skipping: na'=2, router 选 top-2 expert

  # Step 1: Router gate
  r' = W_r @ h                    # R^ne, router logits

  # DeepSeek-V2-Lite: softmax
  r = softmax(F_r(r'))            # 可选 load balancing modifier
  topk_indices = topk(r, k=2)     # 原 k=6, 改为 k=2
  topk_weights = softmax(r[topk_indices])  # renormalize

  # DeepSeek-V3: sigmoid
  r = sigmoid(F_r(r'))            # 无 softmax normalization
  topk_indices = topk(r, k=2)     # 原 k=8, 改为 k=2
  topk_weights = r[topk_indices]  # 直接用 sigmoid 值, 不 renormalize

  # Step 2: Expert FFN (仅 top-k)
  out = 0
  for idx, w in zip(topk_indices, topk_weights):
      # Expert FFN = GLU
      gate = W_g[idx] @ h         # R^(de) gate projection
      up = W_u[idx] @ h           # R^(de) up projection
      act = SiLU(gate) * up
      out += w * (W_d[idx] @ act) # down projection, R^(d)

  # Step 3: Shared Expert (always activated)
  gate_s = W_g_shared @ h
  up_s = W_u_shared @ h
  out += W_d_shared @ (SiLU(gate_s) * up_s)

  return out
  ```

  **Expert Pruning 算法（Section 6.2）**：

  在推理前从 ne 个 expert 中选择 ne' 个保留：

  ```
  # Soft Count 方法 (最佳方法):
  # 1. 在 calibration 数据上运行 forward pass
  expert_activation_count = zeros(ne, L)  # 记录每层每个 expert 被激活次数

  for batch in calibration_data:
      for layer in 0..L-1:
          gate_logits = router[layer](hidden_states)
          topk_idx = topk(gate_logits, k=na)
          expert_activation_count[layer][topk_idx] += 1

  # 2. 按激活次数排序, 选 top-ne' experts per layer
  for layer in 0..L-1:
      sorted_experts = argsort(expert_activation_count[layer], descending=True)
      selected[layer] = sorted_experts[:ne']  # 仅保留 ne' 个最活跃 expert

  # 3. 推理时仅加载 selected experts, 其余无视
  ```

  **V2 vs V3 行为差异（Section 5.2）**：
  - **V2 (softmax)**：低排名 expert 权重显著小于 top-1 expert → expert skipping 性能退化更平滑（descending 策略最优）
  - **V3 (sigmoid)**：expert 权重极化（趋于 0 或 1）→ 跳过权重接近 1 的 expert 会导致显著性能下降（ascending 策略最优）
  - 结论：不存在 universal skipping strategy，策略与模型强相关

  **关键性能数据**：
  - Expert skipping: na 从 6→2, V2-Lite 性能下降仅 7.5%（best 6%）；na 平均 3.3 时下降 <1%
  - V3: best method 可提高 throughput ≥10% 且零性能退化
  - Expert pruning (ne 64→48): best method (soft count) Avg 64.2 vs baseline 66.0（−2.7%）
  - Expert pruning (ne 64→32): best method Avg 57.8（−12.4%）
  - Expert pruning (ne 64→16): best method Avg 47.8（−27.6%），随机选择几乎丧失语言能力

## FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

- 属于算法pipeline的实现是什么？实验比较什么？
  - FOLDMOE 提出将 token-level 的 all-to-all（A2A）通信与计算的 overlapping 从 MoE 层扩展到整个 Transformer block，通过 attention-MoE pipelining 实现。核心实现包括三部分：
    1. **1A1M 调度（1-Attention-1-MoE schedule）**：将 Transformer block 划分为四个流水线阶段（attention computation → A2A dispatch → expert computation → A2A combine），通过交错执行 attention 和 expert computation，减少 aAaM 调度中因阶段不平衡导致的流水线气泡。
    2. **Token Buffer 与时间均匀微批次（Time-Uniform Micro-Batching）**：在 attention 层和 MoE 层之间引入 token buffer 解耦二者的微批次划分，使 attention 层可按时间均匀（非 token 数量均匀）切片，MoE 层仍保持 token 数量均匀的微批次。使用基于 FLOPs 建模的启发式算法（Algorithm 1: Quick-start time-uniform attention slicing）确定切片方案。
    3. **与 FlashAttention、TP、SP 的兼容**：FOLDMOE 在不改变 attention 因果掩码的前提下与 FlashAttention 兼容；与 TP 正交（TP 切分算子，FOLDMOE 沿序列维度切分数据）；与 SP 兼容（SP 仅作用于 layernorm、dropout 等非 attention/非 MoE 区域）。
  - 实验比较 FOLDMOE 与 Megatron-MoE（无 overlapping baseline）和 Tutel（SOTA token-level MoE-only overlapping baseline）在 GPT-MoE 模型训练上的每迭代延迟（per-iteration latency）加速比。

- 硬件平台是什么，配置是什么。
  - 2 个 AWS g5.48xlarge 节点，每节点 8 张 NVIDIA A10G-24G GPU，共 16 GPU。
  - 节点间通过 100 Gbps 网络互联。
  - 训练配置：2-way cross-node DP + 8-way intra-node TP+SP（attention 层）+ 16-way EP（MoE 层）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：GPT-MoE 系列，基于 GPT-2 的 MoE 变体：
    - GPT-MoE-S: n_layer=6, d_model=512, n_heads=8, expert_hidden_size=1024
    - GPT-MoE-M: n_layer=6, d_model=768, n_heads=8, expert_hidden_size=1536
    - GPT-MoE-L: n_layer=12, d_model=1024, n_heads=8, expert_hidden_size=2048
    - 每隔一个 Transformer block 将 FFN 替换为 MoE 层，使用 top-1 GShard gate。
  - 数据集：Wikipedia dataset。
  - 序列长度：4K 到 32K（均为 2 的幂）。
  - 指标：per-iteration training latency（平均 per-block 延迟），training throughput。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供开源代码链接，在 ACL Anthology 和 web search 中均未找到公开的代码仓库。
  - **算法 pipeline 解释**：

  FOLDMOE 的核心是将每个 Transformer block 内的 attention 层和 MoE 层组成四级流水线：

  ```
  阶段1: Attn(X_{i:j}; K_{1:i-1}, V_{1:i-1}) → Z_{i:j}
  阶段2: A2A dispatch(Z_{i:j})          → Z'_{i:j}  (发送到对应 expert 所在 GPU)
  阶段3: Expert(E_i, Z'_{i:j})          → Y'_{i:j}  (各 expert 独立计算)
  阶段4: A2A combine(Y'_{i:j})          → Y_{i:j}   (收集回原 GPU)
  ```

  **1A1M 调度伪代码**（单 Transformer block 前向）：
  ```
  # 输入: sequence X[0..L-1]，切片方案 S={l1, l2, ..., ld}
  # K/V cache 初始为空
  K_prev, V_prev = [], []
  start = 0

  for mb_idx = 0 to d-1:
      l = S[mb_idx]  # 当前 micro-batch 的 token 数
      X_mb = X[start : start+l]

      # 阶段1: Attention (可与上一 micro-batch 的 A2A combine 重叠)
      K_mb, V_mb = compute_kv(X_mb)
      K_attn = concat(K_prev, K_mb)
      V_attn = concat(V_prev, V_mb)
      Z_mb = flash_attn(X_mb, K_attn, V_attn, causal=True)

      # 存入 token buffer
      buffer.append(Z_mb)

      # 从 buffer 中取 token-uniform 微批次 (MoE 侧固定大小 m = ceil(L/d))
      while len(buffer) >= m:
          Z_moe = buffer.pop(m)  # FIFO 取出 m 个 token
          # 阶段2: A2A dispatch (与下一 attention 微批次重叠)
          Z_disp = all_to_all_dispatch(Z_moe)
          # 阶段3: Expert computation
          Y_expert = moe_experts(Z_disp, gate)
          # 阶段4: A2A combine
          Y_moe = all_to_all_combine(Y_expert)
          Y.concat(Y_moe)

      K_prev = concat(K_prev, K_mb)
      V_prev = concat(V_prev, V_mb)
      start += l

  # drain buffer
  while buffer not empty:
      Z_moe = buffer.pop(min(m, len(buffer)))
      Y_moe = a2a_dispatch → expert → a2a_combine(Z_moe)
      Y.concat(Y_moe)

  return Y
  ```

  **时间均匀切片算法（Algorithm 1）**：
  - 输入：序列总长 L，overlap degree d，理想切片时间 t̂
  - 首先分配 quick-start slice（大小为 ceil(L/d)），最小化启动 A2A 的延迟
  - 然后基于 attention FLOPs 建模 `FLOPs(l, c) = (4H + 3h)lc + 8H²l` 迭代确定后续切片边界，使每个 attention 微批次的计算时间接近 t̂
  - 时间复杂度 O(L)

  **反向传播**：流水线调度按相反顺序执行（A2A combine → expert grad → A2A dispatch → attention grad），保持与正向相同的重叠模式。

## Fast Inference of Mixture-of-Experts Language Models with Offloading

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出三种针对 MoE 模型的推理加速策略：
    1. **Expert LRU Cache**：利用相邻 token 间 expert 复用的局部性（locality），在 GPU 显存中为每个 MoE 层缓存 k 个最近使用的 expert（LRU 策略）。当后续 token 复用同一 expert 时可即时获取。Mixtral-8x7B 用 k=2（12GB GPU）或 k=4（16GB GPU）。
    2. **Speculative Expert Loading（投机 expert 预加载）**：利用 Transformer 残差连接带来的归纳偏置——当前层的 hidden states 可作为下一层 hidden states 的近似估计。将下一层 MoE gate 函数应用于当前层 hidden states，预测下一层最可能被激活的 1-2 个 expert，在当前层计算期间在后台预取这些 expert 权重到 GPU。
    3. **混合量化（Mixed MoE Quantization）**：使用 HQQ（Half Quadratic Quantization）对 attention 层保持 4-bit，expert 层量化到 2-3 bit，获得最优尺寸-质量权衡。
  - 实验比较：
    - 4.1 节：LRU cache 命中率 vs cache 大小 k，speculative loading recall vs 预取 expert 数量（OpenAssistant 数据集，Mixtral-8x7B-Instruct）
    - 4.2 节：不同量化方案下 Mixtral-8x7B 的 WikiText2/C4 perplexity 和 MMLU 准确率
    - 4.3 节：完整系统在 T4/RTX 3060/RTX 3080 Mobile/A100 上的 tokens/sec，消融 LRU cache 和 pre-loading 的效果

- 硬件平台是什么，配置是什么。
  - T4 (free-tier Google Colab): 16GB VRAM, PCIe Gen.3
  - RTX 3080 Mobile (gaming laptop): 16GB, PCIe Gen.4
  - RTX 3060 (midrange desktop): 12GB, PCIe Gen.3
  - A100-80GB-SXM (data-center server, 用于参考对比)
  - 目标场景：足够系统内存容纳模型参数（量化后），GPU 显存仅能容纳 non-expert 层 + k 个缓存 expert

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B（46.7B 总参数，expert 占 45.1B/96.6%），Mixtral-8x7B-Instruct
  - 数据集/benchmark：
    - OpenAssistant（对话生成，测量 tokens/sec）
    - WikiText2 perplexity（语言建模）
    - C4 perplexity（语言建模）
    - MMLU 5-shot accuracy（多任务语言理解）
    - 推理评测使用 batch size 1，按预测概率采样（无 temperature/nucleus sampling）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：代码开源在 https://github.com/dvmazur/mixtral-offloading
  - **算法 pipeline 解释**：

  **LRU Expert Cache 策略（Section 3.1）**：
  对于 Mixtral-8x7B 的每个 MoE 层（共 32 层），每层有 8 个 expert。每个 token 由 gate 选择 top-2 expert 参与计算。LRU cache 维护每层 k 个最近使用过的 expert 在 GPU 显存中：

  ```
  # 对于每个 MoE 层 l, 维护 LRU cache C_l (max size = k)
  # 初始化：C_l = 空（或随机 k 个 expert）
  
  for each token t:
      for each MoE layer l:
          gate_scores = W_gate[l] @ hidden_states[l]  # (num_experts,)
          top2_indices = topk(gate_scores, k=2)        # 选 top-2 experts
          
          for expert_id in top2_indices:
              if expert_id in C_l:
                  # cache hit: expert 已在 GPU，直接使用
                  expert_weights = GPU_cache[expert_id]
                  # 将该 expert 标记为 most recently used
              else:
                  # cache miss: 从 host RAM 加载 expert 到 GPU
                  expert_weights = load_from_host(l, expert_id)
                  if len(C_l) >= k:
                      evicted = C_l.evict_lru()  # 淘汰最久未用的 expert
                  C_l.add(expert_id)
                  GPU_cache[expert_id] = expert_weights
          
          # Expert computation (仅 top-2)
          hidden_states[l+1] = hidden_states[l]
          for expert_id in top2_indices:
              weight = gate_scores[expert_id] / sum(gate_scores[top2_indices])
              hidden_states[l+1] += weight * expert_ffn(expert_id, hidden_states[l])
  ```

  **Speculative Expert Loading（Section 3.2）**：
  利用残差连接的归纳偏置，用当前层的 hidden states 预测下一层的 gate 选择：

  ```
  # 在处理 MoE 层 l 时，同时预测并预取层 l+1 的 expert
  # 当前层 hidden states: h_l (pre-MoE gate input)
  
  # Step 1: 当前层 gate（正常执行）
  current_gate_scores = W_gate[l] @ h_l
  current_top2 = topk(current_gate_scores, k=2)
  
  # Step 2: 投机预测下一层 gate
  # 利用 h_l 近似 h_{l+1}（残差连接的归纳偏置）
  predicted_gate_scores = W_gate[l+1] @ h_l  # 用当前激活值运行下一层的 gate
  predicted_top2 = topk(predicted_gate_scores, k=2)
  
  # Step 3: 在 CUDA stream 上异步预取预测的 expert
  async_load(l+1, predicted_top2[0])  # 后台加载
  async_load(l+1, predicted_top2[1])  # 后台加载
  
  # Step 4: 继续当前层 expert 计算
  hidden = expert_compute(current_top2, h_l)
  
  # 当进入下一层时，投机加载的 expert 可能已就绪
  # 如果预测正确 → 即时可用；如果错误 → 重新加载正确 expert
  ```

  **系统内存管理（Section 3.3）**：
  - Expert 参数在连续内存 buffer 中分配，单次 host-to-device copy 完成传输
  - Host 侧使用 pinned memory（tensor.pin_memory()）加速传输
  - 分配 b=4 个临时 device buffer 用于异步拷贝/预取，所有 MoE 层共享
  - 总内存 = num_layers × num_experts 个 expert buffer（split 在 host/device 间）+ b=4 临时 buffer
  - 混合量化方案：attention 层 4-bit HQQ（group size 64, scale group size 256），expert 层 2-bit（group size 16, scale group size 128）或 3-bit（group size 64, scale group size 128）

## Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - Fair-MoE 提出面向医疗 VLM 公平性的 MoE 框架，包含两个核心组件：
    1. **FO-MoE（Fairness-Oriented Mixture of Experts）**：在 CLIP 的图像和文本 encoder 中引入两类 MoE 层——**patch embedding-based MoE**（替换最后一个 attention block 的 MLP 层）和 **feature-based MoE**（放置在 encoder 之后）。Embedding-based MoE 通过专家容量（capacity C）和 top-c 筛选过滤偏置 patch embedding；Feature-based MoE 进一步消除偏置特征，提取公平的任务相关特征。两类 MoE 均采用 sparse gating：`W = softmax(G(I))`，`Ŵ = Top_c(Top_r(W, k), α)`，其中 α = C(N+1)k/M。
    2. **FOL（Fairness-Oriented Loss）**：由五部分组成——F_EI（图像 embedding-based MoE 方差优化）、F_ET（文本 embedding-based MoE 方差优化）、F_FI（图像 feature-based MoE 方差优化）、F_FT（文本 feature-based MoE 方差优化）和 L_distance（Sinkhorn distance loss）。FOL 的核心创新是将 MoE load balance 中使用的 variance 度量同时用于 fairness：`F_EI = Σ_{p∈P} Σ_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2`，同时优化不同属性组分布的距离（Sinkhorn）和离散度（variance difference）。
  - 实验比较 Fair-MoE 与 CLIP（Vanilla，b16/l14）和 FairCLIP（SOTA 公平性 VLM，b16/l14）在青光眼诊断任务上的公平性和准确性。消融研究包括：FO-MoE 组件有效性、FOL 各子损失有效性、embedding-based vs feature-based MoE、Text vs Image MoE 模块。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA GeForce RTX 3090 GPU（24GB 显存）。
  - 训练协议与 FairCLIP 保持一致。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 CLIP 架构（ViT-B/16 ~200M 参数，ViT-L/14 ~500M 参数）。对比模型包括 CLIP/b16, CLIP/l14, FairCLIP/b16, FairCLIP/l14, FairMoE/b16, FairMoE/l14。
  - 数据集：Harvard-FairVLMed（青光眼多模态数据集），7000 训练 / 1000 验证 / 2000 测试样本，每样本包含 SLO 眼底图像 + 临床笔记 + 标签，4 个受保护属性：Race（种族）、Gender（性别/GEN）、Ethnicity（民族/ETH）、Language（语言/LAN）。
  - Benchmark 指标：
    - AUC（Area Under the Curve）：整体性能
    - DPD（Demographic Parity Difference）：公平性，衡量不同组获得正向结果的概率差
    - EOD（Equal Opportunity Difference）：公平性，同时考虑 TPR 和 FPR
    - ES-AUC（Equity-Scaled AUC）：性能与公平性的权衡，`ES-AUC_s = AUC_s / (1 + Σ_a |AUC_s - AUC_{s,a}|)`

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：代码已开源在 https://github.com/LinjieT/Fair-MoE-Medical-Fairness-Oriented-Mixture-of-Experts-in-Vision-Language-Models。论文发表于 MICCAI 2025，arXiv: 2502.06094。
  - **算法 pipeline 解释**：

  Fair-MoE 基于 CLIP 的对比学习框架，在图像和文本 encoder 中替换/增加 MoE 层：

  **前向流程（单张图像 + 文本对）**：
  ```
  # 图像侧
  I_image = ViT_patch_embed(image)              # (N+1)×D patch embeddings
  I_enc = attention_blocks[0..K-2](I_image)     # 前 K-1 个 attention block
  I^1 = I_enc                                     # 输入 embedding-based MoE
  W^1 = softmax(G^1(I^1))                         # Gate: R^{(N+1)×D} → R^{(N+1)×M^1}
  Ŵ^1 = Top_c(Top_r(W^1, k^1), α)                 # Sparse + capacity filtering
  I^2_a = Σ_{b=0}^{M^1-1} Ŵ^1_{a,b} · E^1_b(I^1_a)  # Expert 加权聚合
  I_feat = I^2_0                                  # 取 [CLS] token
  W^2 = Top_r(softmax(G^2(I_feat)), k^2)          # Feature-based MoE gate
  I^3 = Σ_{b=0}^{M^2-1} Ŵ^2_b · E^2_b(I_feat)     # Fair image feature

  # 文本侧（对称结构）
  T_text = tokenize + embed(report)              # L×D text embeddings
  # ... 同样经过 embedding-based MoE 和 feature-based MoE ...
  T^3 = fair_text_feature                         # Fair text feature

  # 对比学习 + FOL
  similarity = cosine(I^3, T^3)
  L_CLIP = contrastive_loss(similarity, labels)
  L_FOL = F_EI + F_ET + F_FI + F_FT + L_distance
  L_total = L_CLIP + λ · L_FOL
  ```

  **FOL 方差优化核心逻辑（以图像 embedding-based MoE 为例）**：
  ```
  # 从整个数据集和特定属性组分别采样 N 个 batch
  # 收集 gate weights 矩阵 O_N, O_{N|p}
  for p in ProtectedAttributes:  # race, gender, ethnicity, language
      for j in range(M^1):       # 每个 expert
          loss += (Var(O_N[:, j]) - Var(O_{N|p}[:, j]))^2
  ```

  **Expert 结构**：
  ```
  E^1_b(x) = T̃^1_b · σ(W̃^1_b · x)   # 两层 MLP + 激活函数
  ```
  其中 σ 为激活函数，W̃^1_b 和 T̃^1_b 为可学习参数。

  **对比学习范式**：论文遵循 CLIP 的对比学习训练方式，将匹配的图像-文本对作为正样本，不匹配的作为负样本，通过 InfoNCE loss 进行优化，并在其上叠加 FOL。

## FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - FarSkip-Collective 提出一种修改 MoE 模型架构连接性的方法，通过使计算能够在通信进行期间使用"过时"（outdated）或"部分"（partial）激活值来消除阻塞通信模式。核心实现包括两部分：
    1. **FarSkip-Collective 架构修改**：修改 Transformer 层的残差连接，使得下一子块的计算输入 $o_k^*$ 不再等待当前子块的完整输出 $o_k$。提出两种变体：
       - (8a) "Outdated"：$o_k^* = o_{k-1}$，直接使用上一层的完整输出
       - (8b) "Partial"：$o_k^* = o_{k-1} + f_k^*(o_{k-1}^*)$，使用当前子块中不依赖通信的部分计算结果
       对于 Attention 子块输入，使用 partial activation：$\text{attn-in}_k = o_{k-2} + \text{attn-out}_{k-1} + \text{shared-exp-out}_{k-1}$（缺失 routed-exp-out_{k-1}，使得 Combine 通信可被重叠）。对于 MLP 子块输入，使用 outdated activation：$\text{mlp-in}_k = o_{k-1}$（使 Dispatch 通信可被重叠）。
    2. **FCSD（FarSkip-Collective Self-Distill）**：通过 KL 散度知识蒸馏将原始模型转化为 FarSkip-Collective 模型。以原始模型为 teacher，FarSkip 修改后的模型为 student，使用 KL 散度 loss $\mathcal{L}_{KD}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} [\sum_t KL(q(\cdot \mid x, y_{<t}) \parallel p_{\theta}(\cdot \mid x, y_{<t}))]$ 训练。训练配方：AdamW + cosine-annealing LR scheduler + 1000-step warmup，batch-size 从 $\{2^{16}, 2^{17}, 2^{18}\}$ 中 sweep 选择，learning rate 从 {2e-5, 4e-5, 8e-5} 中 sweep 选择，最多训练 10B tokens，使用 MBPP+ 作为 early stopping 验证集（patience=20 evals, delta=2%）。
  - 实验比较：FCSD 蒸馏的 FarSkip-Collective 模型 vs 原始模型 vs SFT baseline，在 11 个下游评测上对比准确性（Tab. 1）。蒸馏方法消融：KL vs KL+Inter.L2 vs SFT vs KL+Embed Freeze vs 不同 batch-size（Tab. 2）。层数消融：不同比例（50%/75%/90%/100%）和不同位置（从首层/从末层）的 FarSkip 层替换（Fig. 3）。Pretraining from scratch：从头预训练 FarSkip 架构 vs 常规架构的 loss curve 对比（Fig. 8, Tab. 4）。

- 硬件平台是什么，配置是什么。
  - 训练：1× AMD MI325X 8GPU 机器（单节点）；多节点扩展：4 节点 × 4×MI325X（每节点 8GPU），节点间 400Gbps 互联。
  - 推理：1× AMD MI300X 8GPU 机器；多节点：2 节点系统，8×400Gbs NIC 互联。
  - 软件环境：PyTorch（torch.dist async_op + CUDA Stream），Megatron-LM（训练），vLLM & SGLang（推理），HIP/CUDA-graphs。

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-V2-Lite (16B-A3B, 64 experts), Qwen-3-30B MoE (30B-A3B), Llama-4-Scout (109B-A17B), DeepSeek-V2 (235B), DeepSeek-V3 (671B, 用于训练的缩短版 L=6 约 71B)。
  - 训练数据：GenQA [43] 和 Infinity Instruct [22] 的 SFT 数据，最多 10B tokens。
  - 下游评测 Benchmark（11 个）：PIQA, ARC-Easy, ARC-Challenge, HellaSwag, CommonsenseQA, WinoGrande, HumanEval+, MMLU, OpenBookQA, GSM-8K, MBPP+。
  - Pretraining 评测：ARC-C, ARC-E, BoolQ, HellaSwag, MMLU, OpenBookQA, PIQA, SCIQ, WinoGrande。
  - 性能指标：通信重叠率（overlap %），端到端加速比（speed-up），Time-To-First-Token (TTFT)，Time-Between-Tokens (TBT)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文声明 "We plan to open-source our implementation and modified model checkpoints"，截至分析时未在 web search 中发现公开代码仓库。
  - **算法 pipeline 解释**：

  **FarSkip-Collective 架构修改**（以单个 MoE Transformer 层为例）：

  原始 MoE 层前向（常规连接性）：
  ```
  # 输入: o_{k-1} (上一层输出)
  # Step 1: Attention sub-block
  attn_out = Attention(LayerNorm(o_{k-1}))
  o_k_attn = o_{k-1} + attn_out         # 残差连接

  # Step 2: MoE gating (阻塞等待 o_k_attn)
  gate_scores = Router(LayerNorm(o_k_attn))

  # Step 3: Dispatch all-to-all (阻塞通信)
  tokens = AllToAllDispatch(o_k_attn, gate_scores)  # 暴露通信气泡

  # Step 4: Routed experts + Shared experts
  routed_out = RoutedExperts(tokens)
  shared_out = SharedExperts(o_k_attn)

  # Step 5: Combine all-to-all (阻塞通信)
  combined = AllToAllCombine(routed_out)  # 暴露通信气泡

  # Step 6: 最终输出
  o_k = o_k_attn + shared_out + combined
  ```

  FarSkip-Collective 修改后的前向（Section 4.1 训练执行顺序）：
  ```
  # 输入:
  #   o_{k-2}: 上上层输出
  #   attn_out_{k-1}: 上一层 attention 输出
  #   shared_out_{k-1}: 上一层 shared expert 输出
  #   routed_out_{k-1}: 上一层 routed expert 输出 (待 Combine)

  # Step 1: Attention part (a) — q, k, v 准备
  #   attn-in_k = o_{k-2} + attn_out_{k-1} + shared_out_{k-1}  (partial)
  q, k, v = MLA_prepare(attn-in_k)     # 不依赖 routed_out_{k-1}

  # Step 2: 同步上一层的 Combine (如果上一层是 FarSkip MoE 层)
  WaitCombine(routed_out_{k-1})         # 此时 Combine 已被重叠

  # Step 3: MoE gating
  gate_scores = Router(LayerNorm(o_{k-1}))

  # Step 4: 异步启动 Dispatch
  DispatchAsync(tokens, gate_scores)    # async_op=True, 立即返回

  # Step 5: Attention part (b) — core attention + output projection
  #   Dispatch 在后台运行，与 attention 计算重叠
  attn_out_k = MLA_core_attn(q, k, v)

  # Step 6: 同步 Dispatch，执行 routed experts
  WaitDispatch()
  routed_out_k = RoutedExperts(dispatched_tokens)

  # Step 7: 异步启动 Combine
  CombineAsync(routed_out_k)            # 后台运行

  # Step 8: Shared experts (与 Combine 重叠)
  shared_out_k = SharedExperts(o_{k-1})

  # 最终输出在下一层同步 Combine 时获取
  ```

  **重叠窗口条件**（Eq. 9）：
  $$T_{\text{Dispatch}} + T_{\text{Combine}} \le T_{\text{overlappable}} = T_{\text{layer}} - (T_{\text{Routed Experts}} + T_{\text{Gate}})$$

  **FCSD 训练伪代码**：
  ```python
  # 加载原始模型作为 teacher（冻结）
  teacher = load_checkpoint("original_moe")
  teacher.eval()
  for p in teacher.parameters():
      p.requires_grad = False

  # 初始化 FarSkip-Collective student（与 teacher 参数形状相同，仅连接性不同）
  student = convert_to_farskip(teacher)  # 修改 skip connections
  student.train()

  # Sweep: batch_size ∈ {2^16, 2^17, 2^18}, lr ∈ {2e-5, 4e-5, 8e-5}
  optimizer = AdamW(student.parameters(), lr=best_lr)
  scheduler = CosineAnnealingLR(optimizer)

  for step in range(max_steps):
      x = next_batch()  # SFT data (GenQA + Infinity Instruct)
      with torch.no_grad():
          teacher_logits = teacher(x)
      student_logits = student(x)
      loss = KL_divergence(teacher_logits, student_logits)
      loss.backward()
      optimizer.step()
      scheduler.step()

      # Early stopping: 每 1000 steps 评估 MBPP+
      if step % 1000 == 0:
          mbpp_score = evaluate_mbpp_plus(student)
          if is_instability(mbpp_score, patience=20, delta=0.02):
              break
  ```

  **训练中的 Sequence Number Hijacking**（反向传播）：
  在反向传播中，FarSkip 使用 PyTorch autograd 的 Sequence Number 机制重新排序计算优先级。默认 autograd 按节点创建顺序（与正向相同顺序）处理就绪节点。FarSkip 重新分配 Sequence Number，将子块反向计算节点优先级提高，将通向通信输入的节点优先级降低，使得在通信等待期间先执行子块计算，最大化重叠窗口。

  **Pretraining from scratch 结果**（Fig. 8, Tab. 4）：
  从头预训练 DeepSeek-V2-Lite 架构（16B, 64 experts）50B tokens，FarSkip vs Regular：
  - 最终 training loss: 2.205 vs 2.187（最后 50 步平均）
  - 下游评测平均分: 54.7 vs 54.4

## FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - FedMoE 提出一个基于 MoE 架构的个性化联邦学习框架，包含两阶段训练流程：
    1. **Stage One: Coarse-grained Submodel Initialization**：首先通过 PEFT（LoRA）在客户端进行少量轮次（约5轮）内存高效微调，收集各 expert 的激活概率（$p_{i,j} = n_{i,j}/N$）。对于内存不足的客户端，云端基于同任务其他客户端的数据量加权平均估算其激活概率。然后云端对每个客户端执行启发式子模型搜索——建模为优化问题：在内存约束下最大化每层保留 expert 的激活概率阈值 θ。使用二分搜索在 [0,1] 范围内寻找最优 θ，对每个 θ 构建满足阈值的最小 expert 子集，验证是否超出内存限制 $\alpha \cdot M$，动态调整上下界。
    2. **Stage Two: Federated Training and Fine-grained Submodel Adjustment**：
       - **Modular Aggregation**：dense 层使用 FedAvg 聚合；sparse 层按模块粒度——未被任何客户端激活的 expert 保持不变，仅被单个客户端使用的 expert 直接更新，被多个客户端共享的 expert 使用 FedAvg 聚合。Router 对应的维度按相同模式更新。
       - **Expert Recommendation**：若客户端多轮性能无提升（达到瓶颈），云端基于各客户端 expert 激活概率的 cosine similarity（Eq. 4）找到 top-K 最相似客户端作为参考。若参考客户端平均 expert 数多于当前客户端，则推荐增加 expert（按估算激活概率 $\hat{p}_{expert}$ 排序）；否则推荐裁剪低效 expert。调整具有探索性，若性能未改善则回退并固定结构。
  - 实验比较：
    - End-to-end 性能：FedMoE vs randomMoE（随机选 expert 子集）、FedProx（正则化联邦优化）、SCAFFOLD（控制变量）在 4 种 FL 设置下对比 task performance + communication volume + memory usage
    - 收敛速度：FedMoE vs 三 baseline 的 99%/90% 相对目标性能加速比
    - 鲁棒性：各方法在不同设置下的 Coefficient of Variation (CV) 和 Composite Variation Index (CVI)
    - 消融实验：FedMoE vs w/o stage1 vs w/o stage2 对比任务性能和 expert 数量演变

- 硬件平台是什么，配置是什么。
  - 模拟 FL 环境：30 个客户端，每轮随机选择 5 个（Standard 设置）或强制选择 3 个不同任务类型的客户端（Enforced 设置）。
  - 客户端内存容量：18GB–24GB，典型高性能智能手机和边缘计算平台。
  - 云端服务器：维护全局 MoE 模型，执行子模型搜索、聚合和 expert 推荐。
  - 训练使用 Hugging Face Transformers 框架，模型权重从 Hugging Face 直接下载预训练 Switch Transformers。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Switch Transformers 架构。FedMoE 和 randomMoE 的全局模型配置为每层 32 experts；FedProx 和 SCAFFOLD 由于边缘设备内存限制配置为每层 8 experts。预训练权重从 Hugging Face 下载。PEFT 阶段使用 LoRA 进行内存高效微调。
  - **数据集**：
    - AG News（文本分类 task-TC，评价指标 accuracy）
    - SQuAD（阅读理解 task-RC，评价指标 F1 score）
    - XSum（文本摘要 task-TS，评价指标 Rouge-2）
  - **FL 设置**（4 种模拟真实场景）：
    1. Standard-Hetero-T：30 客户端，异构任务，每轮随机选 5 个
    2. Standard-Hetero-TD：在 Hetero-T 基础上引入 label-skewed non-IID 数据分布（不均匀标签分配）
    3. Enforced-Hetero-T：强制每轮选 3 个不同任务类型客户端，制造更强冲突
    4. Enforced-Hetero-TD：Enforced 客户端选择 + label-skewed non-IID 数据
  - **Benchmark 指标**：task accuracy/F1/Rouge-2，communication volume (GB)，peak memory usage (GB)，convergence speedup，CV/CVI 鲁棒性指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未明确说明开源代码仓库链接。
  - **算法 pipeline 解释（Stage One 启发式子模型搜索）**：

  ```
  # 输入: 每层所有 expert 的激活概率 p[i][j], 内存约束 M, 预留比例 α
  # 输出: 每层保留 expert 的二值指示 x[i][j]

  def HeuristicSubmodelSearch(p, M, α, w_d, w_expert):
      lo, hi = 0.0, 1.0
      while hi - lo > epsilon:
          θ = (lo + hi) / 2
          feasible = True
          for each layer i:
              sorted_experts = sort_by_p_desc(i)  # 按激活概率降序
              cum_prob = 0
              for expert j in sorted_experts:
                  x[i][j] = 1
                  cum_prob += p[i][j]
                  if cum_prob >= θ:
                      break
          # 计算子模型总内存
          mem = w_d + sum(x[i][j] * w_expert[i][j])
          if mem <= α * M:
              lo = θ  # 可行，尝试更大 θ
          else:
              hi = θ  # 不可行，减小 θ
      return x  # 最优 expert 映射
  ```

  **Stage Two Federated Training（Algorithm 1 核心流程）**：

  ```
  for each round r = 1..R:
      S ← sample subset of clients
      for client u_k in S (parallel):
          w_k = subsample(global_model, client_expert_map[u_k])
          w_k* = TRAIN(w_k, D_k_train)       # 本地微调
          p_all, acc = VALIDATE(w_k*, D_k_val)  # 收集激活概率和验证分数
          send (w_k*, p_all, acc) to server

      # Modular Aggregation
      for param in dense_layers:
          w_global[param] = FedAvg(w_k*[param] for k in S)
      for expert e in sparse_layers:
          clients_with_e = {k: e ∈ w_k for k in S}
          if len(clients_with_e) == 0: continue   # 未激活不变
          elif len(clients_with_e) == 1:           # 单客户端直接更新
              w_global[e] = w_k*[e]
          else:                                    # 多客户端 FedAvg
              w_global[e] = FedAvg(w_k*[e] for k in clients_with_e)

      # Expert Recommendation
      for client u_k in S:
          if acc not improved:
              # 计算与其他客户端的 cosine similarity
              sim(u_k, u_a) = cosine_sim(p_u_k, p_u_a)  # Eq. 4
              S' ← top K similar clients
              n = AVG(n_expert(S')) - n_expert(u_k)
              if n > 0:  # 增加 expert
                  for expert outside w_k:
                      p_hat = weighted_avg_p(sim, p_from_S')  # Eq. 6
                  E ← top n experts by p_hat
                  add E to w_k
              else:      # 裁剪 expert
                  for expert inside w_k:
                      p_hat = weighted_avg_p(sim, p_from_S')
                  E ← top |n| experts by lowest p_hat
                  remove E from w_k
              if adjusted model not improved:
                  revert and fix structure
  ```

  **Modular Aggregation 张量计算**（以单个 MoE 层为例）：
  ```
  # 全局模型: W_global = {W_dense, W_router, W_expert[0..E-1]}
  # 客户端 k 的子模型: W_k = {W_dense, W_router[kept], W_expert[kept]}

  # Dense 层聚合 (FedAvg)
  W_dense_new = (1/|S|) * Σ_k W_k_dense

  # Sparse 层聚合 (Modular)
  for expert j in 0..E-1:
      S_j = {k: expert j ∈ W_k}
      if len(S_j) == 0:
          W_expert_new[j] = W_expert_old[j]  # 不变
      elif len(S_j) == 1:
          W_expert_new[j] = W_k_expert[j]    # 直接更新
      else:
          # FedAvg 加权聚合
          n_total = Σ_{k∈S_j} |D_k|
          W_expert_new[j] = Σ_{k∈S_j} (|D_k|/n_total) * W_k_expert[j]

      # Router 对应维度同步
      W_router_new[j] = same_pattern_as_expert(S_j)
  ```

  **Cosine Similarity for Expert Recommendation (Eq. 4)**：
  ```
  # u_k, u_a: 两个客户端在全部 expert 上的激活概率向量
  # p_{i,j} 为第 i 层第 j 个 expert 的激活概率
  sim(u_k, u_a) = Σ_i Σ_j p_{i,j}(u_k) · p_{i,j}(u_a)
                  / (||p(u_k)|| · ||p(u_a)||)
  ```

  **关键性能数据**：
  - Standard-Hetero-T: FedMoE 94.76/86.64/16.92 (TC/RC/TS) vs FedProx 92.92/87.99/11.94（FedProx 在 RC 上略优但 TS 显著落后）
  - Communication volume: FedMoE 1.76GB vs FedProx 2.30GB（−23.5%）
  - Memory usage: FedMoE 13.44GB vs FedProx 24.71GB（−45.6%）
  - 收敛加速（Enforced-Hetero-T）：1.35×–2.92× vs baselines（90% target）
  - Ablation: w/o stage1 性能显著下降（TS: 14.50 vs 16.92），expert 数不减反增（96→104）；w/o stage2 expert 数保持 78 不变

## FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - FSMoE 提出一个灵活的 MoE 分布式训练系统，通过三项核心技术优化任务调度：
    1. **MoE 模块化与统一抽象**：将 MoE 层分解为 6 个子模块（Gate、Order、I-Order、Dispatch、Combine、Expert），预实现 4 种路由函数（GShard、Sigmoid、X-MoE、Expert Choice），并通过在线 profiling 为不同 MoE 实现提供任务调度。
    2. **节点内/节点间通信与计算的协同调度**：在 MP 和 ESP group 对齐节点内 GPU 数量的常见场景下，将节点内通信（ESP-AllGather、ESP-ReduceScatter，走 NVLink/Shared Memory）与节点间通信（AlltoAll Dispatch/Combine，走 InfiniBand）以及专家计算进行流水线化（pipeline），通过 4 种 case 分类和 SLSQP 求解器确定最优流水线度（pipeline degree r）。
    3. **自适应梯度分区（Adaptive Gradient Partitioning）**：在反向传播中，将 Gradient-AllReduce 的梯度按 overlappable parts 分配到各 MoE 层，通过两阶段算法（Step 1: 贪心分配，Step 2: 差分进化优化剩余梯度分配）最大化隐藏梯度同步通信开销。
  - 实验比较 FSMoE 与 Tutel（w/ PipeMoE）、DeepSpeed-MoE 在配置 MoE 层和真实 MoE 模型（GPT-2 MoE、Mixtral-7B、Mixtral-22B）上的每迭代训练时间加速比。

- 硬件平台是什么，配置是什么。
  - **Testbed-A**：48 GPU 集群（6 节点），每节点 8×NVIDIA RTX A6000 @1.46GHz, 48GB，NVLink 112.5GB/s (4x)，Mellanox MT28908 @200Gb/s InfiniBand，PCIe 4.0 x16，CPU Dual Intel Xeon Platinum 8358 @2.60GHz，512GB DDR4。
  - **Testbed-B**：32 GPU 集群（8 节点），每节点 4×NVIDIA RTX 2080Ti @1.35GHz, 11GB，无 NVLink，Mellanox MT27800 @100Gb/s InfiniBand，PCIe 3.0 x16，CPU Dual Intel Xeon Gold 6230 @2.10GHz，512GB DDR4。
  - 软件环境：Ubuntu 20.04, CUDA 11.3, PyTorch 1.12, NCCL 2.12。
  - 并行配置：Testbed-A 上 N_MP=N_ESP=8；Testbed-B 上 N_MP=N_ESP=4。N_EP 等于节点数（6 或 8）。

- 模型是什么。数据集和bench分别是什么。
  - 真实模型：GPT-2 XL MoE、Mixtral-7B (7 layers on Testbed-B due to memory limit)、Mixtral-22B (33 layers on Testbed-A)。
  - 配置层实验：1458 个不同 MoE 配置组合，参数空间为 B∈{1,2,4}, N_heads∈{8,16,32}, L∈{512,1024,2048}/{256,512,1024}, M∈{1024,2048,4096}, N_hscale=H/M∈{2,3,4}, f∈{1.2,2.4,*}, ffn-type∈{simple,Mixtral}。
  - Benchmark：per-iteration training latency（ms），speedup vs baseline。
  - 数据集：论文使用语言模型训练的标准流程，具体数据集名称论文未明确说明（以 causal language modeling 和 masked language modeling 为训练目标）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：代码开源在 https://github.com/xpan413/FSMoE。
  - **系统架构解释**：

  FSMoE 基于 PyTorch + C/C++/CUDA 扩展实现，将 MoE 层模块化为 6 个子模块并支持自定义：

  ```python
  # 使用 FSMoE 构建 MoE 层
  from FSMoE import LinearGate, SimpleOrder, MOELayer
  gate_impl = LinearGate()
  order_impl = SimpleOrder()
  moe_module = MOELayer(gate_impl, order_impl, **kwargs)
  # moe_module 可作为普通 nn.Module 使用

  # 自定义 Expert 和 Hook
  from FSMoE import ExpertBase, CallbackBase
  class CustomizedExpert(ExpertBase):
      def do_experts(self, args): pass
  class CustomizedCallBack(CallbackBase):
      def before_moe_start_hook(self, args): pass
  ```

  **算法pipeline 执行流程**（DP+MP+EP+ESP 混合并行下的反向传播，pipeline degree r=4）：

  输入 tensor X 被切分为 r 个 chunk，在流水线中依次处理：

  ```
  # 对于每个 chunk i (0 ≤ i < r):
  for i in range(r):
      # 阶段1: 节点内 AllGather (ESP-AllGather)
      X_i_ag = AllGather(X_i)        # intra-node, NVLink

      # 阶段2: 节点间 AlltoAll Dispatch
      X_i_disp = AlltoAllDispatch(X_i_ag)  # inter-node, InfiniBand

      # 阶段3: 节点内 ReduceScatter (ESP-ReduceScatter)
      X_i_rs = ReduceScatter(X_i_disp)     # intra-node

      # 阶段4: Expert Computation
      Y_i = ExpertCompute(X_i_rs)          # GEMM on GPU

      # 阶段5: 节点内 AllGather (ESP-AllGather)
      Y_i_ag = AllGather(Y_i)              # intra-node

      # 阶段6: 节点间 AlltoAll Combine
      Y_i_comb = AlltoAllCombine(Y_i_ag)   # inter-node

      # 阶段7: 节点内 ReduceScatter (ESP-ReduceScatter)
      Y_i_rs = ReduceScatter(Y_i_comb)     # intra-node

  # Gradient-AllReduce 与最后一个 chunk 的 ESP-AllGather/ReduceScatter 及 expert 计算重叠
  GradientAllReduce(grads)
  ```

  **性能模型**（线性建模）：
  ```
  t_{a2a,r} = α_{a2a} + n_{a2a}/r · β_{a2a}
  t_{ag,r}  = α_{ag}  + n_{ag}/r  · β_{ag}
  t_{rs,r}  = α_{rs}  + n_{rs}/r  · β_{rs}
  t_{exp,r} = α_{exp} + n_{exp}/r · β_{exp}
  ```
  其中 α 为启动时间，β 为每字节/每单位计算量的传输时间，n 为通信量或计算量。

  **最优流水线度求解**（Algorithm 1: FindOptimalPipelineDegree）：
  ```
  输入: α_{a2a}, β_{a2a}, n_{a2a}, α_{ag}, β_{ag}, n_{ag},
        α_{rs}, β_{rs}, n_{rs}, α_{exp}, β_{exp}, n_{exp}, t_{gar}
  输出: r, t^{moe}

  1. r1, t1 = solve(f_1)  // Case1: 节点间通信主导，SLSQP求解
  2. r2, t2 = solve(f_2)  // Case2: 专家计算主导
  3. r3, t3 = solve(f_3)  // Case3: AlltoAll通信主导
  4. r4, t4 = solve(f_4)  // Case4: 节点内通信主导
  5. r = candidates[argmin(t1,t2,t3,t4)]
  6. return r, min(t1,t2,t3,t4)
  ```

  **自适应梯度分区两阶段算法**：
  - Step 1：以 t_{gar}=0 优化各 MoE 层流水线度，计算 overlappable parts 时间 t_{olp}，贪心分配梯度：n_first^i = g_grad^{-1}(min(t_grad(n_grad^{i-1}), t_{olp}^i))
  - Step 2：对剩余梯度，差分进化算法求解 min Σ f_moe^i(t_grad(x_g^i))，将剩余梯度最优分配到各层。

  **前向/反向分别调度**：前向 r_fwd 和反向 r_bwd 独立优化（反向计算量约为前向 2 倍，且含 Gradient-AllReduce），912/1458 配置下前反向最优度不同。

## Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Flex-MoE 框架，核心是两个算法组件：(1) **Missing Modality Bank Completion**：对缺失模态基于已观测模态组合从 learnable embedding bank 中查找补充，避免 zero-padding/imputation 破坏编码器训练；(2) **Expert Generalization & Specialization**：先用全模态样本通过 G-Router 训练通用 expert（含 load/importance balancing loss），再用 S-Router 结合交叉熵损失将 top-1 gate 绑定到目标 modality combination expert，剩余 top-(k-1) expert 仍做 load/importance balancing。实验比较 Flex-MoE 与单模态 baseline（3D CNN、VGG、ResNet-18/34）、多模态 baseline（TF、MulT、MAG、LIMoE、ShaSpec、mmFormer、FuseMoE）在 11 种 missing modality combination 场景下的 ACC、Macro-F1、AUC，以及消融实验（去除 expert specialization/generalization、去除 embedding bank、改变排序策略）、敏感度分析（expert 数量、SMoE 层数、top-k）和计算复杂度对比（mean time/iteration、GFLOPs、参数量）。
- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU。
- 模型是什么。数据集和bench分别是什么。
  - 模型：modality-specific encoders（3D-CNN 处理 MRI 图像、ResNet-34 处理 Genetic SNP 数据、MLP encoder 处理 Clinical/Biospecimen tabular 数据），输出 concat 后经 Transformer（FFN 替换为 SMoE layer，ADNI 用 16 experts, top-4 gating；MIMIC-IV 用 32 experts, top-3 gating），最后 1-layer MLP prediction head 做 AD 三分类或 MIMIC 二分类。Hidden dim=128, attention heads=4, batch size=8, learning rate=1e-4, 50 epochs（含 5 warm-up epochs），load/importance balancing loss coefficient=0.01。
  - 数据集：ADNI（Image/Genetic/Clinical/Biospecimen 四个模态，AD 阶段三分类：Dementia/CN/MCI，70/15/15 train/val/test split，test 和 val 取全模态交集以保证公平），MIMIC-IV（Lab&Vital/Clinical Notes/ICD-9 Codes 三个模态，一年死亡率二分类，每患者取最后一次访问）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/UNITES-Lab/flex-moe
  - 算法流程伪代码：
    ```
    # 输入: samples 按可用模态数降序排列
    # M = {I, C, B, G} = 4个模态
    # Missing modality bank: B ∈ R^(2^|M|-1 × |M| × d), d=128
    # MC_index: modality combination 索引, 如 "IGCB"=0, "IGC"=1, ..., "B"=14

    for each sample i in batch:
        for each modality m in {I, C, B, G}:
            if m is observed in sample i:
                e_i^m = Encoder_m(sample_i)    # 仅用有模态的样本训练encoder
            else:
                mc_idx = MC_index(observed_combinations(i))
                e_i^m = B[mc_idx][m]           # 从bank按观测组合查找缺失embedding
        h_i = concat([e_i^I, e_i^C, e_i^B, e_i^G])  # h_i ∈ R^(4×128)

    # === Phase 1: Expert Generalization (warm-up epochs, 仅全模态样本) ===
    if sample_i has ALL modalities:
        gate_logits = g(h_i)                   # g is 1-2 layer MLP
        gate_vals = TopK(softmax(gate_logits), k)
        y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)
        # G-Router uses standard load + importance balancing loss:
        #   L_balance = CV^2(importance) + CV^2(load)

    # === Phase 2: Expert Specialization (remaining epochs, all samples) ===
    gate_logits = g(h_i)
    top1_pred = argmax(gate_logits)
    target_expert = MC_index(observed_modalities_of(i))
    # Cross-entropy loss to bind top-1 gate to target expert:
    #   L_ce = - Σ_j one_hot(MC(x_j)) · log(softmax(gate_logits))
    # Load/importance balancing only on remaining top-(k-1) experts:
    #   L_balance = CV^2(Σ_i importance_{e≠etop1}) + CV^2(Σ_i load_{e≠etop1})
    gate_vals = TopK(softmax(gate_logits), k)
    y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)

    # Inference: 任意模态组合 → S-Router 激活对应expert + 其他top-(k-1)
    pred = MLP_head(y_i)
    ```
  - 张量计算要点：missing modality bank `B` 的索引由观测模态组合的位掩码确定，共 2^4-1=15 种非全组合。bank embedding 维度 d=128。Router 为 1-2 层 MLP，输出经 softmax 后 Top-K 选择（ADNI 用 k=4, MIMIC-IV 用 k=3）。GFLOPs 约 59.06-59.07（极低，因 SMoE 稀疏激活），参数量约 36.5M-36.9M（远低于 FuseMoE 的 264.7M-340.9M）。mean time/iteration 约 12.73-16.00s，优于 FuseMoE（18.68-20.71s）。expert 总数 16（ADNI）或 32（MIMIC-IV），expert indices 对应所有可能的 modality combinations 加 buffer experts。Encoder 训练只使用该 modality 被 observed 的样本，不做 zero-imputation。

## GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - GRACE-MoE 提出一套 lossless 协同优化框架，通过三个算法层面的设计联合优化 SMoE 推理中的通信开销和计算负载不均衡：
    1. **Non-Uniform Hierarchical Expert Grouping（Section 4.1）**：基于 spectral clustering 对 expert affinity matrix（共激活频率）进行分层分组——跨节点层面使用 fully non-uniform grouping 最大化 intra-node affinity 以最小化跨节点通信；节点内 GPU 间使用 controlled non-uniform grouping（由 non-uniformity ratio r 控制 size deviation δ=E·r）。通过绘制 (S(r), U(r)) 曲线选择 knee point 作为最优 r。Algorithm 2 给出完整 controlled non-uniform grouping 流程（光谱聚类→trim oversized groups→reassign overflow experts→balance undersized groups）。
    2. **Dynamic Expert Replication based on Load Skew（Section 4.2）**：定义 computational load skew factor ρ = Wmax/W，由 n_replica = min(max(1, floor(ρ)), n_gpu-1) 动态确定副本数。仅复制 heaviest group 中最热的 expert（cumulative load > Wmax·n_replica/(1+n_replica)），副本放置到 n_replica 个最空闲 GPU 作为 secondary copies。
    3. **Topology-Aware Routing with Locality Preference（Section 4.3）**：三级 locality-first 策略——(i) 优先同 GPU 副本；(ii) 其次同节点内其他 GPU 副本；(iii) 最后跨节点副本。每级内使用 Weighted Round-Robin with Load Prediction（基于 pre-replication load stats 预测 post-replication GPU 负载，weights ∝ 1/load）。
  - 实验比较：(1) 端到端性能：GRACE-MoE vs Tutel, Megablocks, vLLM, C2R, Occult（No-Prune）在 OLMoE/DeepSeek-v2-lite-chat/Qwen3-30B-A3B 三个 MoE 模型上，2 nodes×2 GPUs 和 2 nodes×4 GPUs 两种集群，两种 workload；(2) Component analysis：六种配置下（Occult→+HSC→+HG→+FR+WRR→+DR+WRR→+DR+TAR）的通信/负载指标增量影响；(3) Generalizability：跨 dataset transfer（WikiText-2→MATH→GitHub）的 end-to-end latency。

- 硬件平台是什么，配置是什么。
  - 2 节点，每节点 4× NVIDIA A100-SXM4 GPU (80GB)。节点内 NVLink（每 GPU 12 links，50 GB/s per direction）。节点间 25 Gbps Ethernet。软件：Megablocks + PyTorch 2.5 + Triton 3.1，BFloat16 精度。

- 模型是什么。数据集和bench分别是什么。
  - OLMoE（6.92B, 64 experts, top-8, 16 MoE layers）、DeepSeek-v2-lite-chat（15.7B, 64 experts, top-6, 26 MoE layers）、Qwen3-30B-A3B（30.5B, 128 experts, top-8, 48 MoE layers）。
  - 数据集：WikiText-2-v1, MATH, The Pile (GitHub subset)。指标：All-to-All time, cross-node/intra-node traffic, GPU idle time, per-layer GPU load std, MoE layer time, end-to-end latency。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声明 "code will be released upon acceptance"，基于 Megablocks 实现。
  - GRACE-MoE 算法 pipeline（offline + online 两阶段）：

**Offline Phase: Profiling → Hierarchical Grouping → Dynamic Replication**

```
# Step 1: Profiling - 构建 expert affinity matrix per layer
for each MoE layer l:
    for each token in calibration_data:
        topk = router[l](h_t)
        for i, j in topk: A[l][i][j] += 1  # co-activation count

# Step 2: Hierarchical Grouping
# Cross-node: fully non-uniform spectral clustering (N groups → N nodes)
C_nodes = SpectralClustering(A, D=N)
# Intra-node: controlled non-uniform with ratio r
#   E = floor(n_experts / D_gpu), delta = max(1, round(E * r))
#   num_min = max(1, E - delta), num_max = E + delta
#   Select r at knee point of (S(r), U(r)) curve:
#     U(r) = sum_{group C} sum_{i,j in C} A[i,j] / sum_{i<j} A[i,j]
#     S(r) = sqrt(1/D * sum (|C_d| - E)^2)
# Algorithm 2: SpectralClustering(A, D) → trim oversized → reassign overflow → balance undersized
C_gpus = ControlledNonUniformGrouping(A, D_gpu, r_opt)

# Step 3: Dynamic Replication (per layer)
W_max = max(sum(token_count for expert in group))
W_mean = mean(group_loads)
rho = W_max / W_mean
n_replica = min(max(1, floor(rho)), n_gpu - 1)
# In heaviest group: rank experts by load, select those with
#   cumulative_load > W_max * n_replica / (1 + n_replica)
# Place replicas on n_replica least-loaded GPUs
```

**Online Phase: HSC + Topology-Aware Routing**

```
# Hierarchical Sparse Communication (HSC, Section 5):
# Stage 1: Cross-node — physically global group, logically sparse
#   GPU aggregates tokens to same dest node → single cross-node send (zero-padded)
# Stage 2: Intra-node — tokens redistributed to expert GPUs via high-BW links
# Cross-node comm overlapped w/ intra-node routing decision computation

# Topology-Aware Routing (Algorithm 4):
for each token routed to expert e (with replicas on replica_gpus):
    if token_gpu_id in replica_gpus:
        selected = token_gpu_id              # local GPU first
    elif any(g in replica_gpus with Node(g) == token_node):
        candidates = [g in replica_gpus | Node(g) == token_node]
        selected = WRR(candidates, predicted_loads)  # intra-node WRR
    else:
        selected = WRR(replica_gpus, predicted_loads) # cross-node fallback

# WRR Load Prediction (Eq. 4):
# W_p = W_max / (n_replica + 1)  # per-instance load after replication
# W'_max = W_max - W_r + W_p
# W'_i = W_i + W_p  (for target replica-hosting GPU i)
# polling_weights ∝ 1 / W'  (inverse proportional)
```

  - 关键数据：最大 speedup 4.66×（OLMoE）、3.73×（DeepSeek）、4.47×（Qwen3）。MoE layer time 降低 up to 80.11%。HSC: All-to-All time −35.19%；HG: 额外 −18.56-24.69%；DR+WRR: GPU idle −19.71%；TAR: All-to-All −9.47% vs WRR, GPU idle 仅 +2.58%。Cross-dataset transfer 最差 +4.52% latency。

## GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - GatePro 提出一种**无参数的 MoE gating 优化方法**，通过局部竞争机制直接提升 expert 选择的多样性（diversity），而非仅关注负载均衡。核心实现：(1) **Gate Similarity Computation**：计算 gating weight 矩阵各行向量之间的 cosine similarity 矩阵 S∈R^{N×N}，识别功能冗余的 expert 对；(2) **Localized Competition Mechanism**：对每个 expert i 找到最相似的 expert j*(i)=argmax_{j≠i} S_{ij}，按 token 级 logit 比较决定 winner/loser，对 loser 施加固定惩罚 λ=10^{-4} 抑制其被选中，防止功能相似的 expert 被同时激活。该方法无额外可学习参数，可 hot-swappable（训练中途启用/禁用），计算开销极小（cosine similarity O(N²d)，竞争选择 O(N)/token）。
  - 实验比较：(1) GatePro vs baseline MoE（含 load balance loss）在 Seed-MoE-0.7B/7B 和 Seed-MoE-1.3B/13B 两种规模上从 100B→1.2T tokens 的 pretrain 全程性能跟踪；(2) Continuous Training (CT) 阶段的性能对比；(3) 在开源 OLMoE-1B/7B 架构上的泛化验证；(4) Expert utilization analysis：zero token count 随训练步数的下降曲线；(5) Expert gating similarity analysis：average cosine similarity、average angle、spectral entropy 三个 diversity 指标；(6) Hot-swappable analysis：不同 GatePro→MoE 切换时间点的性能影响；(7) 不同 expert pool size：128 vs 256 experts 的扩展实验。

- 硬件平台是什么，配置是什么。
  - 训练硬件：8 节点，共 64 GPUs（论文未明确说明 GPU 型号，但基于 ByteDance Seed 基础设施推断为 NVIDIA H800/A100 级别）。分布式训练使用 FSDP (Zhao et al. 2023) 和 Flash Attention (Dao et al. 2022)。
  - OLMoE 实验：论文未明确说明具体 GPU 配置，但遵循 OLMoE 原版开源配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型系列：
    - Seed-MoE-0.7B/7B：0.7B 激活参数 / 7B 总参数，top-k=6，默认 128 experts（扩展至 256）
    - Seed-MoE-1.3B/13B：1.3B 激活参数 / 13B 总参数，top-k=6，128/256 experts
    - OLMoE-1B/7B：开源 MoE 架构，遵循原版配置
  - 训练 tokens：Seed-MoE-0.7B/7B 最多 500B tokens；Seed-MoE-1.3B/13B 最多 1.2T tokens；OLMoE 训练 400B tokens。
  - 评估 benchmark：
    - 主要（Seed-MoE）：MMLU-Pro, MMLU, BBH, HellaSwag, GSM8K, MBPP
    - OLMoE 扩展：MMLU, HellaSwag, ARC-Challenge, PIQA, COPA
  - Hot-swappable 实验：GatePro-MoE 0.7B/14B, 256 experts, 500B tokens total。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确给出代码开源链接。论文发表于 2025 年 10 月，作者来自 ByteDance Seed + UC Berkeley。可参考 OLMoE 开源实现验证泛化性（https://github.com/allenai/OLMoE）。
  - GatePro 算法 pipeline 伪代码（基于 Algorithm 1）：

```python
# === GatePro MoE Forward Pass ===
# Input: token x ∈ R^d, gating weights W_g ∈ R^{N×d}, penalty λ=1e-4,
#        experts {E_1, ..., E_N} (each a FFN)

# Step 1: Original logits
logits = W_g @ x  # [N], router logits

# Step 2: Gate similarity matrix (pre-computed, updated periodically)
# S_{i,j} = <W_g[i], W_g[j]> / (|W_g[i]| * |W_g[j]|)  # cosine similarity
S = cosine_similarity(W_g)  # [N, N]
# 对角线除外: S[i, i] = -inf

# Step 3: For each expert i, find its most similar counterpart
j_star = argmax(S, dim=1)  # [N], j*(i) for each i

# Step 4: Localized competition
penalty_mask = zeros(N)
for i in range(N):
    if logits[i] < logits[j_star[i]]:
        penalty_mask[i] = -lambda  # loser gets penalized

# Step 5: Suppressed logits
logits_tilde = logits + penalty_mask  # [N]

# Step 6: Top-k expert selection on suppressed logits
topk_indices = topk(logits_tilde, k=6)  # k=6 in paper

# Step 7: Softmax over selected experts
alpha = softmax(logits_tilde[topk_indices])  # [k]

# Step 8: Sparse weighted combination
output = sum(alpha[j] * E[topk_indices[j]](x) for j in range(k))
return output
```

  - GatePro 关键张量计算流（inference 单 token，N=128 experts，k=6）：
    - input token x (1×d) → gating projection W_g·x (128) → GatePro penalty: 对 128 experts 逐一检查 loser (128 次比较) → suppressed logits (128) → top-6 selection → 6 expert FFN compute (forward: Linear[d→αd]→GeLU→Linear[αd→d]) → weighted sum → output (1×d)
    - 对比 baseline MoE：仅多了 cosine similarity 预计算 (offline/periodic) + per-expert penalty 比较 (online O(N))，无额外参数。
  - GatePro 的 hot-swappable 特性：`penalty_mask` 可在任意训练步通过标志位启用/禁用，模型权重完全不变。论文实验证明先用 GatePro 训练 400B tokens 再切换回标准 MoE 训练 100B tokens，性能几乎等同于全程 GatePro (500B) 训练，表明 GatePro 建立的 expert diversity 具有"训练遗产效应" (training legacy effect)。

## Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

- 属于算法pipeline的实现是什么？实验比较什么？
  - Hunyuan-Large 提出多项算法 pipeline 创新来训练大规模 MoE 模型（389B 总参数量, 52B 激活参数, 256K 上下文）：
    1. **MoE 混合专家路由策略（Shared + Specialized Experts + Recycle Routing）**：使用 1 个 Shared Expert（被所有 token 消费）和 16 个 Specialized Experts（每个 token 激活 top-1）。提出 Recycle Routing 策略：对传统 top-k 路由中因 capacity overflow 被丢弃的 token，随机重新分配到未超 capacity 的其他 specialized experts，避免关键信息丢失。
    2. **KV Cache 压缩（GQA + CLA）**：联合使用 Grouped-Query Attention（8 组 KV heads）和 Cross-Layer Attention（每 2 层共享 KV cache），将 KV cache 内存开销相比 MHA 减少约 95%（从 4nhdhl 降至 2ngdhl）。
    3. **Expert-Specific Learning Rate Scaling**：不同 expert（shared vs specialized）处理的 token 数不平衡（shared expert 处理所有 token, specialized expert 处理 1/16 的 token），因此 effective batch size 不同。为 shared expert 分配最优学习率 ε_opt(B)，为 specialized experts 按比例 ε_opt(B)/ε_opt(B/n) ≈ 0.31 缩小学习率。
    4. **MoE Scaling Laws**：训练 10M-1B 激活参数的 MoE 模型系列，拟合 N_opt = N_c * C_min^α 和 D_opt = D_c * C_min^β，确定最优激活参数数量（约 58.1B, 选 52B）和最优训练 token 数（约 5.6T, 选 7T）。
    5. **四步合成数据 pipeline**：Instruction Generation → Instruction Evolution → Response Generation → Response Filtering，生成约 1.5T tokens 高质量合成数据（包含数学、代码、低资源语言、高教育价值领域）。
  - 实验比较：
    - Pre-training baselines: LLama3.1-405B, LLama3.1-70B, Mixtral-8x22B, DeepSeek-V2
    - Post-training baselines: LLama3.1-405B-Instruct, LLama3.1-70B-Instruct, Mixtral-8x22B-Instruct, DeepSeek-V2.5-Chat
    - 评估指标：MMLU, MMLU-Pro, BBH, HellaSwag, CommonsenseQA, WinoGrande, PIQA, NaturalQuestions, DROP, ARC-C, TriviaQA, CMMLU, C-Eval, C3, GSM8K, MATH, CMATH, HumanEval, MBPP, AlignBench, MT-Bench, IFEval, Arena-Hard, AlpacaEval-2.0, RULER, LV-Eval, PenguinScrolls

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练硬件配置（未给出 GPU 型号、数量、节点数等具体信息）
  - 推理评估使用与 baseline 一致的配置，具体 GPU 和节点数论文未明确说明

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Hunyuan-Large, Transformer-based MoE。64 layers, 80 attention heads, 8 KV heads (GQA), 1 shared expert, 16 specialized experts (top-1 activated), hidden size 6400, SwiGLU activation, RoPE position embedding, vocabulary 128K tokens, total 389B params, 52B activated params
  - **数据集**：预训练数据共 7T tokens，其中 ~1.5T 为合成数据（数学、代码、低资源语言、高教育价值领域），其余为自然文本语料（中英文为主）；SFT 数据超 100 万条；长上下文预训练用 ~10B tokens × 2 stages（32K → 256K）；论文未公开具体数据集名称
  - **Benchmarks**：MMLU, MMLU-Pro, BBH, HellaSwag, CommonsenseQA, WinoGrande, PIQA, NaturalQuestions, DROP, ARC-C, TriviaQA, CMMLU, C-Eval, C3, GSM8K, MATH, CMATH, HumanEval, MBPP (pre-training)；AlignBench, MT-Bench, IFEval, Arena-Hard, AlpacaEval-2.0, GPQA_diamond, RULER, LV-Eval, PenguinScrolls (post-training + long-context)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：代码 https://github.com/Tencent/Tencent-Hunyuan-Large, 模型 https://huggingface.co/tencent/Tencent-Hunyuan-Large
  - **MoE 混合路由算法 pipeline（Recycle Routing 伪代码）**：
    ```
    # 输入: tokens x (B×L×d), router weights W_r (d×E), experts E[0..15], shared expert E_shared
    # 超参数: capacity_factor C, expert_capacity = (B×L / E) * C
    
    # Step 1: All tokens through shared expert
    shared_out = E_shared(x)  # (B×L, d)
    
    # Step 2: Router scores for specialized experts
    scores = softmax(x @ W_r, dim=-1)  # (B×L, 16)
    top1_vals, top1_indices = topk(scores, k=1, dim=-1)  # each token gets top-1 expert
    
    # Step 3: Recycle routing
    expert_counts = count(top1_indices)  # [16]
    expert_out = zeros(B×L, d)
    for token_i in range(B×L):
        expert_id = top1_indices[token_i]
        if expert_counts[expert_id] < expert_capacity[expert_id]:
            expert_counts[expert_id] += 1
            expert_out[token_i] = E[expert_id](x[token_i])
        else:
            # Recycle: randomly assign to any expert under capacity
            available = [e for e in range(16) if expert_counts[e] < expert_capacity[e]]
            if available:
                new_expert = random_choice(available)
                expert_counts[new_expert] += 1
                expert_out[token_i] = E[new_expert](x[token_i])
            # else: truly dropped (rare)
    
    # Step 4: Combine
    output = shared_out + expert_out
    return output
    ```
  - **KV Cache 压缩张量计算**：MHA 原始 KV cache = 4 × n_h × d_h × l bytes (bf16)；GQA 后 = 4 × n_g × d_h × l (8 groups, 80→8 heads 压缩)；CLA 后 = 2 × n_g × d_h × l (每 2 layers 共享, l→l/2)；最终 GQA+CLA = 2 × n_g × d_h × l，相比 MHA = (2 × n_g)/(4 × n_h) = n_g/(2×n_h) = 8/(2×80) = 5% 的 KV cache。
  - **Expert-Specific LR 张量计算**：给定 batch size B，噪声 batch size B_noise，最大学习率 ε_max。shared expert LR = ε_opt(B) = 2ε_max / (sqrt(B_noise/B) + sqrt(B/B_noise))。specialized expert LR = ε_opt(B/16)。ratio ≈ 0.31（代入 B 和 B_noise 计算）。
  - **MoE Scaling Law 计算预算**公式 C ≈ 9.59ND + 2.3×10^8 D（N=激活参数量, D=训练 tokens），结合临界 batch size B_crit(L) 得到最小计算预算 C_min = C / (1 + B/B_crit(L))，拟合 N_opt = 5.9×10^-3 × C_min^0.5305, D_opt = 3.2 × C_min^0.50。

## Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

- 属于算法pipeline的实现是什么？实验比较什么？
  - **MoE Experts Compression Suite (MC-Suite)**：从四个维度（权重、推理行为、激活、梯度）提出 16 种专家重要性评估准则，用于识别可安全丢弃的冗余专家：
    1. **Weight-Guided（4种）**：Expert Weight Similarity (EWS) — 计算专家权重的 pairwise cosine similarity；Router Weight Norm (RWN) — 路由矩阵中对每个专家的 l2-norm；Expert Weight Stable Rank (WSR) — 权重矩阵的 stable rank = Σσ_i²/σ_1²；Expert Weight Norm (EWN) — 专家权重的 l2-norm。
    2. **Inference-Guided（4种）**：Expert Usage Frequency (EUF) — 专家被多少 token 激活的比例；Expert-Expert Collaboration (ECC) — 两专家共同被路由到同一 token 的次数；Expert Vocabulary Coverage (EVTC) — 专家处理的唯一 token 占词表比例；Expert Input Token Similarity (ETS) — 跨专家输入 token 的重叠数。
    3. **Activation-Guided（4种）**：Expert Activation Similarity (EAS) — 专家激活的 pairwise cosine similarity；Expert Activation Entropy (EAE) — H(A_Ep) ∝ Σ_j log[σ(A_Ep^j)]，各隐藏维度标准差对数之和；Expert Activation Distribution Outliers (EAO) — μ ± 3σ 之外的激活异常值计数；Expert Activation Norm (EAN) — 累积激活的 l2-norm。
    4. **Gradient-Guided（4种）**：Expert Gradient Similarity (EGS) — 专家梯度的 pairwise cosine similarity；Expert Gradient Entropy (EGE) — H(W_Ep^g) ∝ Σ_i log[σ(W_Ep^{g^j})]；Expert Gradient Outliers (EGO) — 梯度异常值计数；Expert Gradient Norm (EGN) — 梯度的 l2-norm。
  - **MoE Lottery Subnetworks**：提出迭代 estimate-prune-finetune 三阶段流程，替代传统 one-shot pruning：
    1. 对每个 MoE layer，使用 MC-Suite 准则估算专家重要性
    2. 每轮丢弃 s/k% 的专家（k 轮总丢弃 s%），每轮丢弃后重新估算剩余专家重要性
    3. 丢弃后使用 task-agnostic budget finetuning（next-token prediction on C4）校正子网络的次优状态
    4. 仅需 ~1M training tokens 即可饱和 finetuning 收益
  - 实验比较：
    - **Baselines**：Random Dropping (one-shot, iterative, w/ MoE Lottery)、prior expert pruning methods (Lu et al., 2024; Muzio et al., 2024)
    - **LLM Weight Pruning 对比**：Random Pruning, Magnitude Pruning, Wanda (Sun et al., 2023) 在 2:4 structured sparsity 下
    - **消融实验**：One-shot vs Iterative vs MoE Lottery 三种剪枝策略对比
    - **Sparsity ratios**：12.5%, 25.0%, 37.5%, 50.0%, 62.5%, 75.0%
    - **评估指标**：C4/Wikitext Perplexity、MMLU accuracy、ARC-c accuracy、ARC-e accuracy、HellaSwag accuracy、WinoGrande accuracy、BoolQ accuracy、CommonsenseQA accuracy
    - **Instruction-following 恢复实验**：zero-shot → k-shot examples → SFT（supervised fine-tuning with instruction-tuning dataset）
    - **关键发现**：(1) Min-EAN (最小激活范数) 和 Min-EGE (最小梯度熵) 是最优准则，50% sparsity 下 perplexity 从 15.21 (Random One-shot) 降至 9.99 (Min-EAN MoE Lottery)；(2) MoE Lottery 在 ≥50% sparsity 下仍保持 robust 性能（≥1.27× speedup, ≤0.55× memory）；(3) Expert dropping 主要损害 instruction-following 能力，可通过 k-shot 或 SFT 恢复；(4) 迭代剪枝与 one-shot 剪枝选出的专家高度不一致（Figure 5a），而 MoE Lottery 与迭代剪枝选出专家高度一致（Figure 5b）；(5) Expert dropping 在 Base model 上执行优于在 Instruct model 上执行。

- 硬件平台是什么，配置是什么。
  - **8×NVIDIA A100 GPU**（论文 Appendix C 明确说明："With the availability of 8×A100, we use a batch size of 8"）
  - 使用 HuggingFace Transformers 加载 Mixtral-8×7B checkpoint

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Mixtral-8×7B（Base 和 Instruct 两个版本），32 层 MoE，每层 8 个 experts，top-2 routing，总参数 ~46.7B，激活参数 ~12.9B，float32 下 180GB 内存（激活 28GB/token）
  - **Calibration 数据集**：C4 validation set（256 samples, max_seq_len=2048），用于 MC-Suite 准则估计和 task-agnostic finetuning
  - **Evaluation Benchmarks**：MMLU（14042 test samples）、ARC-Challenge（ARC-c）、ARC-Easy（ARC-e）、HellaSwag、WinoGrande（1267 test samples）、BoolQ（3270 test samples）、CommonsenseQA（1221 test samples）
  - **Finetuning**：AdamW optimizer, cosine LR scheduler, max LR=1e-6, batch size=4~8，具体超参见 Table 6

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源状态**：截至 2025/05，论文未发布官方代码（Papers with Code 显示 "No code implementations yet"）。arXiv: 2504.05586。作者来自 UT Austin、Apple、UNC Chapel Hill。
  - **算法 Pipeline 伪代码**：

    ```
    # ========== MC-Suite: Expert Importance Estimation ==========
    # 给定 MoE 模型 M，层 l，n 个专家 E = {E_1,...,E_n}，router G with W_G^{d×n}
    # Calibration 数据集 X_calib

    # 示例准则 1：Minimum Activation Norm (Min-EAN)
    def estimate_activation_norm(M, l, X_calib):
        for each expert E_p in layer l:
            A_Ep = []  # accumulated activations
            for batch in X_calib:
                # register forward hook on E_p output
                a_Ep = forward_hook(M, layer=l, expert=p, batch)
                A_Ep.append(a_Ep)
            A_Ep = concat(A_Ep, dim=0)
            score[p] = sum(norm_l2(A_Ep, dim=0))
        drop_idx = argmin(score)  # 最小范数→最可丢弃

    # 示例准则 2：Minimum Gradient Entropy (Min-EGE)
    def estimate_gradient_entropy(M, l, X_calib):
        for batch in X_calib:
            loss = next_token_prediction(M(batch), batch_labels)
            loss.backward()  # 累积梯度
        for each expert E_p in layer l:
            W_grad_p = W_Ep.grad  # 形状与权重相同
            # H ∝ Σ_j log[σ(W_grad_p^j)]
            stds = [std(W_grad_p[j, :]) for j in range(d_hidden)]
            score[p] = sum(log(s) for s in stds if s > 0)
        drop_idx = argmin(score)  # 最小梯度熵→最可丢弃
    ```

    ```
    # ========== MoE Lottery Subnetworks ==========
    # 输入：full MoE model M, target sparsity s, k rounds, MC-Suite criterion c

    def moe_lottery_pruning(M, s, k, c, X_calib):
        drop_per_round = s / k  # e.g., 50% in 4 rounds → 12.5%/round
        for round in range(k):
            # Step 1: Estimate importance using criterion c
            for each MoE layer l in M:
                scores = estimate_criterion(c, M, l, X_calib)
                # 每个 layer 均匀丢弃 expert (per-layer uniform)
                n_drop = int(n_experts * drop_per_round)
                drop_experts[l] = argsort(scores)[:n_drop]

            # Step 2: Prune experts (delete from router + weights)
            for each MoE layer l in M:
                W_G_l^{d×n} → W_G_l^{d×(n-n_drop)}  # 从 router 删除对应列
                remove expert weights from memory

            # Step 3: Task-agnostic budget finetuning
            tokens = 0.2M * (2^round)  # progressive schedule
            for batch in X_calib:
                loss = next_token_prediction(M(batch), batch_labels)
                optimizer.step()
                if tokens_processed >= tokens: break
        return M  # MoE lottery subnetwork
    ```

  - **张量计算示例（Min-EAN 准则, Mixtral-8×7B, layer l）**：
    - 输入：X_calib 经 layer l-1 后的 hidden states H^{t×d}（t tokens, d=4096）
    - Router: G(H) = softmax(H @ W_G^{4096×8}) → top-2 routing → 每个 expert E_p 获得 tokens 子集 X_p^{t_p×4096}
    - Expert forward: A_p = SiLU(X_p @ W_{gate}^{4096×14336}) * (X_p @ W_{up}^{4096×14336}) @ W_{down}^{14336×4096}（标准 SwiGLU FFN）
    - Expert Activation Norm: score[p] = ||A_p||_2 = sqrt(Σ_{i=1}^{t_p} Σ_{j=1}^{4096} (A_p[i,j])²)
    - 选择 score 最小的 expert 丢弃 → 从 router 删除 W_G[:,p] → W_G^{4096×7}
    - 50% sparsity 时（4/8 experts 丢弃）: memory 180GB→99GB, speedup 1.27×

  - **关键结论张量化**：
    - Perplexity trend（w/ MoE Lottery）: Random 75% sparsity→33.05, Min-RWN→17.26, Min-ETS→16.03, Min-EGE→15.08, Min-EAN→14.02（Table 1, Mixtral Instruct）
    - Expert dropping 对 instruction-following 的影响：zero-shot MMLU@50% sparsity 18.91(one-shot)→40.79(MoE Lottery); 加 k-shot 后可接近 full-MoE baseline
    - 迭代 vs one-shot 专家选择差异：Figure 5a 显示 Dark pink（不一致）大面积存在，表明 one-shot 与迭代选出的子网络完全不同

## HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

- 属于算法pipeline的实现是什么？实验比较什么？
  - HybridEP 的 SR-Based Expert Compression（共享-残差专家压缩）是一种算法级参数压缩技术，用于减少跨 DC 的专家传输流量。核心设计：
    1. **Shared Expert（共享专家）**：将所有 expert 参数取平均得到共享专家，由所有 GPU 共享。共享专家学习各专家间的冗余/共性知识，通过 backward 阶段的 All-Reduce 同步梯度。
    2. **Residual Expert（残差专家）**：每个 expert 的残差 = expert - shared_expert。残差分布更集中、更稀疏（Figure 9a 中的 "res" 分布），因为不同 expert 的主要差异集中在少量参数上。
    3. **Top-k Sparse Compression**：对残差 expert 应用 Top-k 压缩（保留绝对值最大的 k 个参数），以稀疏 value-index 格式传输。压缩比 (CR) 最高 50× 时仍不损失模型精度（Figure 14）。
    4. **SREncode/SRDecode 流水线**：编码阶段计算残差 → Top-k → value-index 格式存储；解码阶段从稀疏格式恢复残差 → 与 shared expert 相加恢复完整 expert。SRDecode 中将恢复和加法操作 fused 以减少 overhead（Figure 15b，与 expert computation 融合可减少 45% overhead）。
  - 实验比较：
    - **HybridEP w/ S**（有共享专家）vs **HybridEP w/o S**（直接 Top-k 压缩，无共享专家）vs Tutel/FasterMoE/SmartMoE baseline
    - Loss 曲线分析（Figure 14）：HybridEP w/S 的 loss 与 baseline 几乎一致（50× CR），HybridEP w/o S 的 loss 显著偏高
    - 时间分解分析（Figure 15）：不同 expert 大小下 SREncode/SRDecode 的 overhead 及 fusion 效果（SREncode+optimizer step 融合减 30%，SRDecode+expert computation 融合减 45%）

- 硬件平台是什么，配置是什么。
  - 4 种模型配置在 GPU 集群上训练评估，压缩效果验证使用与 Serving 调度实验相同的集群（Cluster-S/M/L: NVIDIA A800 GPUs）
  - 具体模型与数据集见 Table II：Llama-Tiny (PennTreebank)、Mistral-Small (WikiText2)、GPT-Medium (OpenWebText-10k)、GPT-Large (WikiText103)
  - 所有模型 expert 数 E=32，激活 expert 数 K∈{1,2,4}

- 模型是什么。数据集和bench分别是什么。
  - **模型**（Table II）：
    - **Llama-Tiny**：E=32, H=512, P_E=2.1M, #Layers=12, Dataset=PennTreebank
    - **Mistral-Small**：E=32, H=768, P_E=4.7M, #Layers=12, Dataset=WikiText2
    - **GPT-Medium**：E=32, H=1024, P_E=8.4M, #Layers=12, Dataset=OpenWebText-10k
    - **GPT-Large**：E=32, H=1024, P_E=8.4M, #Layers=16, Dataset=WikiText103
  - 数据集均为语言建模标准 benchmark
  - 压缩效果评估使用 loss 曲线（训练 loss 对比），通过 loss 值判断压缩对精度的影响

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码未公开开源。论文未提供开源链接。SR 压缩算法描述详见论文 §IV-B。
  - SR-Based Expert Compression 伪代码：
    ```
    # === 初始化 ===
    # 所有 experts 参数: experts[0..E-1], 每个 expert 大小为 P_E
    # 共享专家: shared_expert = mean(experts[0..E-1])
    # CR: 压缩比 (e.g., 50x)

    # === SREncode (编码阶段, 与 optimizer step 融合) ===
    def SREncode(expert, shared_expert, CR):
        # 1. 计算残差
        residual = expert - shared_expert  # size: P_E
        
        # 2. Top-k 压缩: 保留绝对值最大的 k 个参数
        k = P_E // CR
        values, indices = topk(abs(residual), k)
        # values: 保留的 k 个残差值(含符号)
        # indices: 对应在 P_E 中的位置
        
        return (values, indices)  # 稀疏 value-index 格式
    
    # === SRDecode (解码阶段, 与 expert computation 融合) ===
    def SRDecode(values, indices, shared_expert):
        # 1. 从稀疏格式恢复残差
        residual_recovered = scatter(indices, values, size=P_E)
        # 未保留位置填 0
        
        # 2. 恢复完整 expert (fused with addition)
        expert_recovered = shared_expert + residual_recovered
        
        return expert_recovered
    
    # === 训练 iteration 中的使用 ===
    # -- 前一步: Initialization 阶段 (与 optimizer.step() 融合) --
    for each expert in local_experts:
        compressed = SREncode(expert, shared_expert, CR)
        send_queue.push(compressed)  # 存入异步发送队列
    
    # -- 当前步: Asyn-comm 阶段 --
    # GPU i 从 send_queue 取出压缩 expert 残差
    for compressed in send_queue:
        # NCCL All-Gather: 域内所有 GPU 收集彼此的压缩 expert
        all_compressed = all_gather(compressed, group=domain_group)
    
    for compressed in all_compressed:
        expert = SRDecode(compressed.values, compressed.indices, shared_expert)
        recv_queue.push(expert)  # 供 expert FFN 计算使用
    
    # -- Expert FFN 计算 --
    for expert, tokens in zip(recv_queue, token_batches):
        output = expert_ffn(expert, tokens)  # 标准 MoE expert 前向
    ```
  - **张量计算示例**（Mistral-Small, E=32, H=768, P_E=4.7M, CR=50, 单个 expert）：
    - 输入：expert 参数 W ∈ R^{4.7M}（gate/up/down 矩阵展平后的总参数量）
    - Shared expert：W_shared = mean(W_0, W_1, ..., W_31) ∈ R^{4.7M}
    - Residual：R = W - W_shared ∈ R^{4.7M}
    - Top-k (k = 4.7M/50 ≈ 94k)：保留 |R| 最大的 94k 个元素
    - 输出：values[94k] + indices[94k]，压缩后数据量 = 94k × (FP16 + INT32) ≈ 0.56 MB
    - 对比原始 P_E=4.7M × FP16 ≈ 9.4 MB，压缩比 ≈ 16.8×（与带宽相关）
    - 论文用 50× CR：P_E=0.094 MB per expert (Table IV, AG-only 配置)，与 P_E=4.7 MB 相比

  - **关键结论**：SR-Based Expert Compression 的核心洞察是 experts 间存在知识冗余（residual 分布比原始 weight 更集中，Figure 9a），通过 shared + residual 分解可以安全地以高压缩比（50×）压缩专家参数，在几乎不损失模型精度的情况下大幅减少跨 DC 传输流量。无 shared expert 的 naive Top-k 压缩会导致显著精度损失（Figure 14），证明 shared expert 对维护精度至关重要。

## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - JANUS 提出三个算法层面的 MoE 推理优化：
    1. **Activated-Expert-Balanced Scheduling (AEBS, Algorithm 1)**：将 MoE 层激活 expert 调度形式化为负载均衡问题——收集 batch 中所有 token 的 top-K routing 结果获得激活 expert 集合 → 单副本 expert 分配至唯一持有实例 → 多副本 expert 贪心分配至当前激活 expert 数最少的实例 → 重写每个 token 的路由结果为物理副本 ID。目标是 minimize a_max = max_i(activated experts on instance i)，因为 MoE 层延迟主要由 straggler（a_max 最大的实例）决定。每 MoE 实例独立运行相同 deterministic 算法，实现 synchronization-free。
    2. **Monte Carlo a_max Estimator + Theoretical Bound（Section 3.5, Appendix A）**：将 expert 激活建模为 balls-into-bins process，推导 closed-form upper bound: a_max ≤ min(C, ā_max + sqrt(2·ā_max·ln n_e)) + 1 (Eq. 5)。利用 recent activation trace 构建 Monte Carlo estimator â_max(n_e, B) 查找表——对每个候选 (n_e, B) 从 trace 采样 B tokens → 应用 AEBS 策略 → 记录结果 â_max。表格周期性重建以适应当前 workload。
    3. **Fine-Grained SLO-Aware Resource Scaling（Section 3.5, Algorithm 2 + Eq. 1-3）**：基于 Roofline + Little's Law 构建 TPOT 性能模型。Attention latency (Eq. 1b) 遵循 Roofline: memory-bound plateau c_a + computation/KV-cache access αb + c_kv·b·S_ctx。MoE latency (Eq. 1c): β·a_max + c_e。稳态 batch size B* 由 Little's Law: B* = λ·TPOT(B*) 求解 (bounded binary search)。枚举 (n_a, n_e) 搜索空间 → 求解 B* → 检查 SLO + memory feasibility → 选择 min(n_a+n_e)。Activation-Aware Replica Placement (Algorithm 3, Appendix B): min-max 优化 co-activation load I(g) = Σ a(e,e')，贪心放置 + bounded swap 解决。
  - 实验比较：
    - AEBS vs EPLB：a_max reduction (Fig. 13), MoE-layer latency (Fig. 14), scheduling overhead (Fig. 15)
    - Full JANUS vs ablations：2PC+EGate+AEBS vs 1PC+EGate vs 2PC+AGate (Fig. 12)
    - Scaling quality：搜索空间可视化 (Fig. 16)，验证 JANUS 选择的资源高效配置
    - Resource cost：24h production trace 下 GPU-hour 节省 39% vs SGLang, 16% vs MegaScale-Infer (Fig. 11)
    - Monte Carlo bound validation：Analytical bound vs â_max across n_e ∈ {6,8,12,16} (Fig. 17)

- 硬件平台是什么，配置是什么。
  - 4 节点 × 8× NVIDIA H100 80GB (共 32 GPU)
  - Intra-node: NVLink 900 GB/s, Inter-node: IB 400 Gbps
  - 模型：DeepSeek-V2, Qwen3-MoE, Scaled-DS variants (top-k=8, 160/200 experts)
  - 所有参数 KV 缓存 BF16 格式

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - DeepSeek-V2：MoE, top-k routing, 共享+路由 expert 架构
    - Qwen3-MoE (235B)：MoE, 含共享 expert
    - Scaled-DS-1：top-k=8, 160 experts/layer, expert intermediate=1024
    - Scaled-DS-2：top-k=8, 200 experts/layer, expert intermediate=1536
  - **数据集**：
    - ShareGPT：avg input 16 tokens + avg output 256 tokens，用于端到端 TPOT/TPG 测量
    - BurstGPT：合成动态到达 trace，模拟生产 LLM 服务负载
    - Production trace (24h)：真实 LLM 服务 trace，用于 scaling 行为评估
  - **Benchmark**：TPOT SLO 满足率，per-GPU throughput (TPG)，GPU-hour 消耗

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确给出 JANUS 公开开源仓库。基于 SGLang (https://github.com/sgl-project/sglang) 实现。
  - **AEBS 算法伪代码（Algorithm 1）**：

    ```
    Algorithm 1: Activated-Expert-Balanced Scheduling (AEBS)
    Input:
      T: number of tokens, n_e: number of MoE instances
      k: activated experts per token
      L(i,j): logical expert ID for token i, expert j
      R(e): number of replicas for expert e
      G(e): set of instances hosting replicas of expert e
      P(e,g): physical replica ID of expert e on instance g
    Output:
      O(i,j): physical replica ID for token i, expert j

    1. E ← ∪_{i=1}^{T} ∪_{j=1}^{k} {L(i,j)}  // 收集所有激活 experts
    2. Initialize actRep[e] ← -1 for all e ∈ E
    3. Initialize load[g] ← 0 for all g ∈ {1, ..., n_e}

    // Assign single-replica experts
    4. for all e ∈ E where R(e) == 1 do
    5.     g ← unique instance in G(e)
    6.     actRep[e] ← P(e,g)
    7.     load[g] ← load[g] + 1

    // Assign multi-replica experts via greedy load balancing
    8. for all e ∈ E where R(e) > 1 do
    9.     g* ← argmin_{g ∈ G(e)} load[g]  // 最少负载实例
    10.    actRep[e] ← P(e, g*)
    11.    load[g*] ← load[g*] + 1

    // Map token routing to physical replicas
    12. for i = 1 to T do
    13.     for j = 1 to k do
    14.         O(i,j) ← actRep[L(i,j)]

    // Synchronization-free: 每个 MoE instance 独立运行相同 AEBS
    // 产生相同 O(i,j)，通过确定性算法保证一致性
    ```

  - **SLO-Aware Scaling 算法伪代码（Algorithm 2）**：

    ```
    Algorithm 2: Fine-Grained, SLO-Aware Resource Scaling
    Input:
      n_max: upper bound of instance sizes
      n_e^min: lower bound of MoE instance sizes (= ⌈E/C⌉)
      B_max: upper bound of batch sizes (GPU memory budget)
    Output: (n_a*, n_e*, B*): optimal configuration

    1. opt ← ⊥; J* ← ∞
    2. for (n_a, n_e) ∈ {1,...,n_max} × {n_e^min,...,n_max} do
    3.     B* ← solve B = λ · TPOT(B, n_a, n_e, S_ctx) via binary search in [1, B_max]
    4.     if B* == ⊥ then continue  // 无可行解
    5.     T ← TPOT(B*, n_a, n_e, S_ctx)  // Eq. (1)
    6.     if T > SLO or not MemoryFeasible then continue
    7.     if n_a + n_e < J* then
    8.         opt ← (n_a, n_e, B*); J* ← n_a + n_e
    9. return opt
    ```

  - **TPOT 性能模型（Eq. 1）张量公式**：

    ```
    TPOT = Σ_{ℓ=1}^{L} [T_attn^(ℓ) + T_moe^(ℓ) + T_comm^(ℓ)]

    T_attn^(ℓ) = max(c_a^(ℓ), α^(ℓ)·b + c_kv^(ℓ)·b·S_ctx)
      // Roofline: memory-bound plateau vs computation+KV-cache
      // b = B/n_a (per-instance batch), S_ctx = avg context length

    T_moe^(ℓ) = β^(ℓ) · a_max^(ℓ)(n_e, B) + c_e^(ℓ)
      // Linear dependence on max activated expert count
      // a_max estimated via Monte Carlo from recent trace

    T_comm^(ℓ) = profiled cost of two-phase communication
    ```

  - **a_max Theoretical Bound（Appendix A, Eq. 5）**：

    ```
    Uniform activation: p_e = K/E
    E[a_g] ≤ C · [1 - (1 - K/E)^B]  // expected activated experts per instance
    ā_max = max_g E[a_g]              // bottleneck instance

    Tail bound (Bernstein + union):
    a_max ≤ min(C, ā_max + sqrt(2·ā_max·ln n_e)) + 1

    Two regimes:
    - Small B: ā_max << C, a_max grows with B → T_moe increases
    - Large B: ā_max → C, a_max plateaus → T_moe capped, T_attn dominates
    ```

  - **Monte Carlo â_max Estimator 使用原理**：
    1. 从最近的 activation trace 采样 B tokens（按 empirical distribution）
    2. 对每个 MoE layer ℓ，应用当前 AEBS 策略 + 候选配置 (n_e, B)
    3. 记录 â_max^(ℓ)(n_e, B) = 各 MoE instance 中 max distinct activated experts
    4. 构建 lookup table [n_e][B] → â_max^(ℓ)
    5. 周期性重建（如每 15 min）以跟踪 workload 变化
    6. 在 Algorithm 2 的 TPOT 评估中 constant-time 查表

  - **关键结果**：
    - AEBS vs EPLB: a_max 降低 2-5 experts (Fig. 13), MoE-layer latency 降低 up to 30% (Fig. 14)
    - AEBS overhead: batch=64 → <20μs, batch=4096 → <90μs (Fig. 15)
    - Full JANUS (2PC+EGate+AEBS): per-GPU throughput 4.7× SGLang, 2.2× MegaScale-Infer, 3.3× xDeepServe
    - Resource cost: 39% GPU-hour saving vs SGLang, 16% vs MegaScale-Infer (24h trace)

## Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出 **Joint MoE Scaling Laws**，将训练损失 L 表示为 active parameters N_act、训练 token 数 D 和 expert 数 E 的函数：
    L(N_act, D, Ê) = aÊ^δ · N_act^(α+γ·ln(Ê)) + bÊ^ω · D^(β+ζ·ln(Ê)) + c
    其中 Ê 是 E 的单调变换（见 Eq.4），c 为 dataset irreducible entropy。核心算法贡献：
    1. **联合形式推导**：从固定 E 的 Chinchilla 形式出发，引入 E 与 N_act、D 的交互项（通过 power-law exponent 中 ln(Ê) 项），统一描述 dense (E=1) 和 MoE (E≥2) 的 scaling behavior。
    2. **Compute Optimality 分析**：对固定 compute budget F=6·N_act·D，求解 argmin L(N_act,D,E)，得出 MoE 的 compute-optimal 配置：expert 越多 → 应减少 active parameters、增加 training tokens（Finding 1）。
    3. **Memory Optimality 分析**：引入 total parameter 约束 N_total ≤ M 和 KV-cache 约束，证明 MoE 可在 memory-constrained 场景下超越 dense 模型（Finding 2-3）。
    4. **Inference Optimality**：将 inference FLOPs (2·N_act·D_inf) 纳入 joint budget，给出训练+推理联合最优配置。
    5. **Learning Rate Scaling Law**：LR(N_act\e, E) = exp(8.39 - 0.81·ln(N_act\e) - 0.25·ln(E))，发现更多 expert 需要更低 LR（Finding 4）。
  - 实验比较：
    - 280+ 模型 runs，E ∈ {1,2,4,8,16,32}，N_act 最高 2.7B，N_total 最高 5B
    - 核心对比：同一 FLOPs budget 下不同 E 的 loss 曲线（IsoFLOP profiles, Fig.2）
    - Memory-matched 验证：1.1B 总参数 dense vs E={2,4} MoE，相同 FLOPs + memory budget 下 MoE 获得更低 loss（Fig.1b）
    - Scaling law fit quality：RMSE_v=0.0039 (validation), RMSE_t=0.0062 (training)；与独立 Chinchilla fit (RMSE_v=0.0041) 接近，验证联合公式的有效性
    - LR scaling law 验证：在 E={1,8} 上拟合，E=4 插值验证，E=32 外推验证（Fig.7）

- 硬件平台是什么，配置是什么。
  - 训练硬件：Polish HPC infrastructure PLGrid (ACK Cyfronet AGH)，以及 Writer.com 提供的计算资源
  - 论文未明确说明具体 GPU 型号和集群配置；在 memory constraint 分析中引用 H100 (80GB)、RTX 4090 (24GB)、8×H100 node (640GB) 作为典型 memory budget 场景
  - 论文未明确说明使用的 GPU 数量、节点互联、CPU 等具体配置

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Decoder-only Transformer，Switch MoE (Fedus et al. 2022)，每 token 激活 1 个 expert。SwiGLU activation，hidden size = 3×d_model。RoPE position embeddings。GPT-2 tokenizer (vocab=50,257)。配置规则：N_blocks = N_heads = d_model/64。Mixed precision 训练（attention、RoPE、router 保持高精度）。Router z-loss=0.001, load balancing loss=0.01。Weight initialization: truncated normal (scale=0.1)。详细模型配置见 Appendix E（N_total 从 79M 到 5.0B，d_model 从 512 到 2304，E 从 1 到 32）。
  - **数据集**：FineWeb-Edu (Penedo et al. 2024)，高过滤质量 web 数据。训练 token 数从 500M 到 80B（随模型配置不同）。
  - **Benchmark/指标**：最终训练 loss（cross-entropy），无下游 NLP benchmark 评估。评估协议：基于 loss 的 scaling law fit quality（RMSE, Huber loss δ=0.01）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源状态**：论文声明"计划开源模型 checkpoint 和代码"（Footnote 2），但截至论文发布未提供开源链接。拟合方法基于开源 LBFGS (PyTorch/SciPy)。
  - **Scaling Law 拟合 pseudocode**：
    ```
    # === Joint MoE Scaling Law Fitting ===
    # Input: 实验数据 {(N_act_i, D_i, E_i, L_i)} for i = 1..280+
    # Output: 拟合系数 {a, α, δ, γ, b, β, ω, ζ, E_start, E_max, c}
    
    1. 计算 Ê_i via Eq.4:  1/Ê = 1/(E-1+(1/E_start-1/E_max)^(-1)) + 1/E_max
    2. 对每对 (N_act_i, D_i)，计算：
         pred_i = a·Ê^δ · N_act^(α+γ·ln(Ê)) + b·Ê^ω · D^(β+ζ·ln(Ê)) + c
    3. 优化目标：Huber loss (δ=0.01) over log-space predictions
         L_huber = Σ_i Huber(log(L_i) - log(pred_i), δ=0.01)
    4. 优化器：LBFGS，lr=1e-4, weight_decay=1e-5
    5. 初始化网格搜索：
         α ∈ {0.05, 0.25, 0.5}, β ∈ {0.05, 0.25, 0.5}
         a,b ∈ {30, 100, 300}, c ∈ {0.5, 1, 2}
         δ,γ,ω,ζ ∈ {-0.5, 0, 0.5}
    6. 选择 training RMSE + validation RMSE 之和最小的系数
    ```
  - **Compute-optimal 配置求解（给定 E 和 budget F）**：
    ```
    1. 从 joint formula 退化为固定 E 的 Chinchilla 形式：
         m(E) = a·Ê^δ,           μ(E) = α + γ·ln(Ê)
         n(E) = b·Ê^ω,           ν(E) = β + ζ·ln(Ê)
       → L(N_act, D|E) = m·N_act^μ + n·D^ν + c
    
    2. 给定 FLOPs budget F = 6·N_act·D，求解 compute-optimal:
         G = (μ·m / (ν·n))^(1/(μ+ν))
         N_act_opt = G · (F/6)^(ν/(μ+ν))
         D_opt = G^(-1) · (F/6)^(μ/(μ+ν))
    
    3. Memory-optimal 配置（约束 N_total ≤ M）：
         在 {N_act, D, E} 空间搜索，满足 6·N_act·D=F 且 N_total(E) ≤ M
         使 L 最小化的配置
    ```
  - **张量计算上下文**：scaling law 本身不涉及张量计算，其推导基于 Switch MoE 的标准计算范式。每一 token 前向通过 Router 选择 top-1 expert → expert FFN 计算 (W_o · SwiGLU(W_g·x, W_p·x)) → 输出。FLOPs 计数遵循 F_train = 6·N_act·D, F_infer = 2·N_act·D_inf。

## Layerwise Recurrent Router for Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Layerwise Recurrent Router for Mixture-of-Experts (RMoE)，在 MoE 路由过程中引入轻量级 Gated Recurrent Unit (GRU)，逐层建立跨层路由决策的依赖关系。具体实现：在第 i 层，先用逐层独立的投影层（Proj_i）将 token hidden state x_i ∈ R^h 降维到 GRU 状态维度 x_i' ∈ R^p（p=128），然后与上一层 GRU 输出 h_{i-1} 拼接送入跨层共享的 GRU 单元，得到当前层 GRU 输出 h_i，最后将 h_i 输入该层 router (linear layer + softmax + top-k) 选择 expert 并执行标准 MoE 计算。
  - 实验比较：在 language modeling (Enwiki8 BPC, WikiText-103 PPL)、大规模 pre-training + SFT (0.91B, 15B 模型)、多 benchmark 评估 (ARC-Easy, Hellaswag, PIQA, SciQ, LAMBADA, MMLU, GSM8K, HumanEval) 下，与 SMoE (标准 linear router)、HyperMoE、SMoE-MLP、RandomMoE、CosineMoE、XMoE 等 baseline 比较。还与 XMoE 结合验证正交兼容性。消融实验拆解 layerwise recurrence、Recurrent Gradient、层投影器、GRU vs RNN vs LSTM 等组件贡献。

- 硬件平台是什么，配置是什么。
  - 8-layer 小模型 (hidden=352, 16 experts top-2)：1 张 NVIDIA A100 GPU，约 21 小时。
  - 0.91B 模型 (24-layer, hidden=1280, 16 experts top-4, fine-grained MoE)：8 张 NVIDIA A100 GPU，约 5 天 pre-training + 2 小时 SFT。
  - 15B 模型 (activate 2.7B)：使用 Megablocks 框架，论文未明确说明 GPU 数量，据训练规模推测为多卡 A100 集群。

- 模型是什么。数据集和bench分别是什么。
  - 模型：decoder-only transformer，小模型 8 层 hidden=352，中模型 24 层 hidden=1280 (Llama-style, RoPE + SwiGLU + RMSNorm)，大模型 15B total/2.7B activated (DeepSeek-MoE style, fine-grained + shared experts)。
  - 数据集：Enwiki8 (character-level LM, BPC)，WikiText-103 (word-level LM, PPL)，大规模 pre-training 使用多语言语料 (Wikipedia + 金融 + 法律文本, 40B/120B/400B tokens)，SFT 使用 Alpaca (52K instruction-response pairs)。
  - Benchmark：ARC-Easy (acc), Hellaswag (acc_norm), PIQA (acc_norm), SciQ (acc), LAMBADA (acc), MMLU (acc), GSM8K (acc), HumanEval (pass@k)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/qiuzh20/RMoE
  - 算法 pipeline 伪代码：

```
# 初始化：跨层共享的 GRU 单元 + 每层独立的 Proj_i 和 Router G_i
h_0 = zeros(p)  # 初始 GRU 状态，维度 p=128

for i in range(num_layers):
    # Step 1: 逐层投影降维
    x_i_prime = Proj_i(x_i)  # x_i ∈ R^h → x_i_prime ∈ R^p

    # Step 2: GRU 跨层循环（Eq.5）
    s_i = sigmoid(W_s @ x_i_prime + U_s @ h_{i-1})   # reset gate
    z_i = sigmoid(W_z @ x_i_prime + U_z @ h_{i-1})   # update gate
    h_tilde = tanh(W_h @ x_i_prime + s_i ⊙ (W_h @ h_{i-1}))
    h_i = (1 - z_i) ⊙ h_tilde + z_i ⊙ h_{i-1}

    # Step 3: 基于 GRU 输出的 MoE routing（Eq.6）
    score_i = softmax(h_i @ G_i)         # G_i ∈ R^(p, N), N 个 experts
    topk_idx, topk_val = topk(score_i, k)

    # Step 4: 稀疏 MoE 计算
    y_i = sum_{n in topk_idx} topk_val[n] * Expert_n(x_i)
```

  - 关键设计：(a) 每层使用独立 Proj_i 而非共享投影器，因为不同层的 hidden state norm/分布差异大；(b) 跨层共享 GRU 单元以引入跨层路由信息；(c) GRU 额外提供 Recurrent Gradient 路径，优化 router 训练；(d) 该设计正交于现有 MoE 方法（如 XMoE, DeepSeekMoE），可无缝组合。

## Llama 3 Meets MoE: Efficient Upcycling

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出从预训练 dense checkpoint 初始化 MoE 模型的 **Sparse Upcycling** 算法 pipeline：
    1. **Upcycling Technique**：将 dense checkpoint 中指定 FFN 层的权重复制 N 次，初始化 MoE layer 的 N 个 expert（每个 expert 是原始 FFN 的完整副本），同时添加随机初始化的 router。其余权重（embedding、attention 等）直接从 dense checkpoint 复制。
    2. **Online Upcycling in NeMo**：在分布式训练框架 NeMo 中实现在线 upcycling。根据并行训练配置将 dense checkpoint 按设备分片（shard），各设备独立完成权重 upcycling，无需跨设备权重复制，解决因总参数量激增导致的内存超限问题。
    3. **MoE Parallel Folding**：提出异构混合并行策略，解耦 Attention 和 MoE 组件的并行映射。Attention 层使用 TP×CP×DP×PP 四维并行；MoE 层使用 Expert-TP×EP×Expert-DP×PP 四维并行。将 Attention 和 MoE 层中通信密集的并行操作折叠到 NVLink 高带宽域内，减少跨节点通信开销。
    4. **5-D Hybrid Parallelism**：基于 Megatron-Core，同时使用 Tensor Parallelism (TP)、Expert Parallelism (EP)、Pipeline Parallelism (PP)、Context Parallelism (CP)、Data Parallelism (DP with ZeRO-1) 五种并行策略。
    5. **Router Algorithm 选择**：对比 Mixtral-type router（KeepTopK→Softmax，确保 upcycling 后初始前向输出与 dense 模型一致）和 ST-type router（Softmax→KeepTopK），选择收敛更快的 Mixtral-type。
  - 实验比较：
    - Llama 3-8B Base vs Llama 3-8B E8T2 (upcycled 8-Expert Top-2 MoE)：MMLU (0-shot/5-shot), TruthfulQA, PIQA, SciQ, LogiQA, BoolQ, OpenBookQA
    - Capacity Factor (CF) 消融：CF=1, 2, 4, Dropless (无限 CF) 下的 MFU 和 MMLU 准确率
    - Base Model CT (Continued Training) vs upcycled MoE 的 MFU 和 MMLU
    - 不同并行配置 (TP/CP/EP/PP) 下的 TFLOPS/GPU 和 MFU
    - Router 类型：Mixtral-type vs ST-type 训练 loss 曲线对比

- 硬件平台是什么，配置是什么。
  - 主训练：512× NVIDIA H100 GPUs，使用 bfloat16 精度
  - MFU 调优实验：128× NVIDIA H100 GPUs（不同 TP/CP/EP/PP 配置），含 NVLink 域内通信
  - 训练框架：NeMo (https://github.com/NVIDIA/NeMo) + Megatron-Core (https://github.com/NVIDIA/Megatron-LM)
  - 分布式并行：5-D Hybrid Parallelism (TP+EP+PP+CP+DP with ZeRO-1)

- 模型是什么。数据集和bench分别是什么。
  - **Base 模型**：Llama 3-8B（Meta 预训练 dense checkpoint）
  - **Upcycled 模型**：Llama 3-E8T2（34.4B 总参数，11.8B 激活参数，8 Experts Top-2 routing，FLOPs 约 dense 的 1.6×）
  - **训练数据集**：
    - RedPajama V2（经 CCNet pipeline 按 n-gram perplexity 分桶，取最低 perplexity 桶，约 0.89T tokens）
    - Academic data blend（多种开源学术 benchmark 数据集混合，约 2.7B tokens）
    - 两源混合比例 7:3
  - **训练量**：100B tokens（主实验），27B tokens（CF 消融实验）
  - **Benchmarks**（使用 lm-evaluation-harness 评估）：
    - MMLU (5-shot & 0-shot), TruthfulQA (0-shot), PIQA (0-shot), SciQ (0-shot), LogiQA (0-shot), BoolQ (0-shot), OpenBookQA (0-shot)
  - **训练超参**：初始 LR=3e-5，余弦退火至 3e-7，100 warmup steps，主实验 CF=4, EP=8, TP=2, PP=4, VPP=8

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：NeMo (https://github.com/NVIDIA/NeMo) 中已集成 online upcycling 功能；Megatron-Core (https://github.com/NVIDIA/Megatron-LM) 提供 5-D 并行训练支持
  - **Upcycling 算法伪代码**：
```
# === Sparse Upcycling: Dense → MoE 初始化 ===
# 输入: dense checkpoint Θ_dense, 目标 expert 数 N, Top-K
# 输出: MoE checkpoint Θ_moe

def upcycle_dense_to_moe(Θ_dense, N, K, moe_layer_indices):
    Θ_moe = copy(Θ_dense)  # 复制所有非 MoE 权重

    for layer_idx in moe_layer_indices:
        # 1. 复制 FFN 权重 N 次初始化 experts
        W_orig = Θ_dense[layer_idx].ffn  # 原始 FFN 权重
        for n in range(N):
            Θ_moe[layer_idx].expert[n] = copy(W_orig)

        # 2. 随机初始化 router
        Θ_moe[layer_idx].router.W_g = random_init()
        Θ_moe[layer_idx].router.W_noise = random_init()

    return Θ_moe
```

  - **MoE Layer 前向传播（Mixtral-type router）**：
```
# 输入: x [B, S, d_model]
# Router:
H(x) = x @ W_g + StandardNormal() * Softplus(x @ W_noise)  # [B, S, N]
G(x) = Softmax(KeepTopK(H(x), K=2))                        # TopK 后 Softmax

# Expert FFN (每个 expert E_i 为 SiLU-gated FFN):
for each token with selected experts (i1, i2):
    y = G(x)_i1 * E_i1(x) + G(x)_i2 * E_i2(x)

# Expert capacity 控制:
expert_capacity = (tokens_per_batch / N) * CF
# 超出容量的 token 被跳过，直接传递到下一层
```

  - **MoE Parallel Folding 配置示例**：
```
# Attention layer: TP=2, CP=2, DP×PP
# MoE layer: EP=8, TP=1
# 效果: Attention 的 TP×CP group (4 GPUs) 折叠到 MoE 的 EP group (8 GPUs)
#        Attention 的 2×2 TP×CP 在单节点 8 GPU 内通过 NVLink 完成
#        避免跨节点通信开销扩大
```

  - **关键训练调优实践**：
    1. TP 和 EP 保持在 NVLink 域内以最小化延迟；MoE 层 EP 通常优于 TP
    2. AllToAll-based token dispatcher 对 TopK=1-4 更高效（vs AllGather-based）
    3. CP 配合 GQA 可重叠通信与计算，减小 KV 通信量
    4. 跨节点扩展用 PP+DP，VPP 减少 pipeline bubble
    5. 早期训练阶段对 MoE 层启用 recomputation，缓解负载不均导致的 OOM

## Load Balancing Mixture of Experts with Similarity Preserving Routers

- 属于算法pipeline的实现是什么？实验比较什么？
  - SIMBAL（SIMilarity-preserving routers for MoE load BALancing）提出一种新的 MoE 负载均衡辅助损失 L_orth = ||R^T R - I_E||_1，通过鼓励 router 权重矩阵 R ∈ R^{D_M × E} 逼近正交矩阵来保持 token 间成对相似性。结合 Saxe et al. 2014 的正交初始化，SIMBAL 使得相似 token 获得相似的 expert 分布，减少 expert 间知识冗余。核心设计选择：(1) 使用 loss-based 方法替代显式正交参数化（QR分解），避免大模型训练中的计算开销和数值不稳定；(2) 损失函数数据集无关且计算便宜，对 batch size 不敏感；(3) 提出 Pairwise Expert Similarity (PES) 指标量化 expert 冗余度。
  - 实验比较：
    - SIMBAL vs LBL (Load Balancing Loss, Fedus et al. 2022) 为主比较
    - 两种模型规模：MoE-M (230M active/627M total) 和 MoE-L (761M active/3.14B total)
    - 无负载均衡 baseline (no loss) 验证 collapse 避免
    - Loss-Free (LF) balancing [Wang et al. 2024] 组合实验（附录A.1）
    - 评估指标：Validation Perplexity、收敛速度、PES、SEU、Router Entropy、Router Gram L2 距离
    - 下游 benchmark：ARC Challenge/Easy、HellaSwag、PIAQ、WinoGrande、GLUE
    - 推理时 expert pruning 协同实验 [Szatkowski et al. 2024]
  - 结果：SIMBAL 比 LBL 快 36% 收敛；MoE-M perplexity 13.685 vs 14.086，MoE-L 8.304 vs 8.517；MoE-L avg benchmark 45.19% vs 43.28%；PES 显著更低 (0.0044 vs 0.0255)；推理 pruning 下 7.4% speedup

- 硬件平台是什么，配置是什么。
  - MoE-M 训练：8× NVIDIA A100 40GB GPUs，Distributed Data Parallelism (DDP)
  - MoE-L 训练：8× AMD MI300X 192GB accelerators，DDP
  - 软件环境：PyTorch (bfloat16 训练精度)，基于 OLMo 开源代码库
  - 推理 pruning 实验：论文未明确说明具体 GPU 型号

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：Transformer backbone + RMSNorm + SwiGLU activations + RoPE + Z-loss (1e-5)
    - MoE-M：DM=768, Depth=8, Heads=8, DF(expert)=768, 32 experts, top-4, 230M active/627M total, RoPE θ=1e4, Peak LR=5e-4
    - MoE-L：DM=1536, Depth=12, Heads=12, DF(expert)=1536, 32 experts, top-4, 761M active/3.14B total, RoPE θ=1e5, Peak LR=3e-4
    - Dense baselines：Dense-M (230M) 和 Dense-L (761M)
    - 所有 FFN 层替换为 MoE 层
  - 训练配置：AdamW optimizer, weight decay=0.01, linear warmup (2000 steps) + cosine decay, bfloat16
  - 数据集：DCLM-pool-400m-1x [Li et al. 2025]，cl100k_base tokenizer (tiktoken)，77M tokens 验证集
  - 训练量：MoE-M 19.9B tokens, MoE-L 78.6B tokens
  - LBL baseline：loss coefficient 0.01

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 基于 OLMo 开源代码库 (https://github.com/allenai/OLMo)。LBL 参考 lucidrains/st-moe-pytorch。
  - SIMBAL loss 伪代码（来自论文 Appendix A.3 Figure 6）：
```python
def simbal_loss(router_linear, p=1):
    w = router_linear.weight           # [E, D_M]
    w_ortho = torch.matmul(w, w.T)     # Gram matrix R^T R
    eye = torch.eye(w.shape[0], device=w.device)
    loss = torch.norm(w_ortho - eye, p=p)  # ||R^T R - I||_1
    return loss
```
  - LBL baseline 伪代码：
```python
def balance_loss(gates):
    # gates: [batch_size, num_tokens, num_experts]
    expert_mask = gates > 0.0
    f_i = reduce(expert_mask.float(), "b t e -> b e", "mean")
    P_i = reduce(gates, "b t e -> b e", "mean")
    loss_per_batch = num_experts * torch.sum(f_i * P_i, dim=-1)
    return loss_per_batch.mean()
```
  - SIMBAL 单 token 张量流：x ∈ R^D_M → Router R (near-orthogonal) → scores ∈ R^E → softmax → top-4 experts → SwiGLU FFN per expert → weighted sum output
  - PES 指标：C_expert(x) = (2/(N(N-1))) Σ_i Σ_{j>i} cos(f_i(x), f_j(x))，PES 越低表示 expert 多样性越高、冗余越低

## LocMoE: A Low-overhead MoE for Large Language Model Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - LocMoE 提出三种算法 pipeline 优化用于 MoE 大语言模型训练：
    1. **GrAP（Grouped Average Pooling）正交门控权重层**：用分组平均池化替代传统 Dense 层计算门控值 G_{m,E_i}。权重矩阵 ω_i 为固定正交矩阵，元素为 0 或 1，按分组聚集：ω_{i,j} = 1{i·d/n ≤ j < (i+1)·d/n}。正交性使不相关 token 倾向被路由到不同领域的 expert，利于收敛和精度，同时避免 Dense 层的 FLOPs 开销。
    2. **局部性专家正则化（Locality Loss）**：在辅助负载均衡 loss L_aux = α·n·Σ f_i·P_i（α=0.01）基础上，增加 KL 散度正则化项 L_loc = μ·KL(D_c || D_l)，其中 D_c 为当前 expert 分配分布，D_l 为完全局部化的理想分布。该 loss 促使同一节点的 token 优先路由到本地 expert，将跨节点 All-to-All 通信转化为节点内高带宽通信。总 loss 为 L_task = L_aux + L_loc + L_cross。
    3. **专家容量下界理论（Expert Capacity Lower Bound）**：首次在 NLP 领域证明了 MoE expert capacity 存在临界值。基于高维球面均匀分布假设，推导出 ec_min ≥ 1/(n·erfc(√(δ²d/(2-δ²))))，其中 δ 为 token 与 gating weight 夹角余弦的最小阈值。实验测得 δ≈0.03，可据此下界安全降低 expert capacity 而不损失精度。
  - 系统层面：使用 MindSpore 内置的 Group-wise All-to-All 将通信拆分到 TP 域高速带宽 + EP 域，并实现 FFN 计算与 All-to-All 通信的切片重叠（slice-and-overlap），进一步隐藏通信延迟。
  - 实验比较：
    - LocMoE vs HashMoE（基于哈希函数的绝对均衡路由）vs SwitchMoE（Top-1 gating with auxiliary loss）
    - 每 epoch 训练时间减少：64N 下 12.68%~22.24%，128N 下也有显著加速
    - Expert 分配均衡性、收敛速度（valid perplexity）、多 NLP 任务推理精度
    - All-to-All 通信时间下降 5.13%（64N/128N）
    - Ablation：计算/通信/重叠/空闲时间占比分析，在不同规模（64N, 128N, 256N）下对比
  - 结果：64N 下相对 HashMoE 加速 1.15x，相对 SwitchMoE 加速 1.29x；LocMoE 在 256N 下不如 HashMoE（因部分节点无 expert，locality 失效）

- 硬件平台是什么，配置是什么。
  - **Ascend 910A NPU 集群**：3 组配置
    - 64N：8 节点 × 8 Ascend 910A（64 NPUs）
    - 128N：16 节点 × 8 Ascend 910A（128 NPUs）
    - 256N：32 节点 × 8 Ascend 910A（256 NPUs）
  - 每 Ascend 910A：32 AI Cores，最大内存 2TB，最大内存带宽 1.07TB/s，FP16 算力 320 TFLOPS，INT8 算力 640 TOPS
  - 服务器型号：Atlas 800 9000，每 8 个 NPU 通过 HCCS（Huawei Cache Coherence System）互联，节点间通过两级 Fat-tree 网络 + RoCE 互联
  - 软件栈：CANN 5.1.RC2.1 (toolkit 1.84, driver 23.0.rc2)，MindSpore 2.0.0
  - 通信库：HCCL（Huawei Collective Communication Library），支持 ring/mesh/HD/ring+HD/mesh+HD 算法

- 模型是什么。数据集和bench分别是什么。
  - **模型**：PanGu-Σ，1.085T 参数稀疏 MoE 模型（从密集模型 PanGu-α 扩展而来），包含 Dense + Sparse Transformer Encoder layers + Decoder layers + Query layer。稀疏层含 RRE（Random Routing with Expert selection）两级路由：第一级按领域分组，第二级（原为随机哈希路由 → 被 LocMoE 替换）路由到组内具体 expert。配置：16 experts，8 MoE layers，40 attention heads，batch size 32，expert parallel=16。
  - **数据集**：华为内部移动网络运营商服务文档语料
    - iCase（技术案例）：591,972 文档，387M tokens
    - Wiki（内部知识管理平台）：1,146,755 文档，1,162M tokens
    - 核心网/MML：223,898 文档，137M tokens
    - 配置翻译（Huawei/Cisco 产品文档）：1,460,680 文档，560M tokens
    - 特性文档（4G/5G FAQ, fault tree 等）：86,913 文档
    - 语料为中/英/双语，格式包括 Word/PDF/HDX/HTML
  - **Benchmark**：自定义 NLP 任务评估集（从 10 个业务角度提取），包括故障树节点识别（L1-L3 难度分级）、方案类、ICT 认证考试、标题改写等。每个任务 30~80 条 Q&A 作为评估集，人工评分。
  - 预训练目标：自回归语言模型（cross-entropy loss），验证指标为 valid perplexity

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供开源代码。LocMoE 基于 Huawei 内部 MindSpore 框架和 PanGu-Σ 模型实现，运行于 Ascend NPU 集群，均为商用/内部系统。MindSpore 开源（https://github.com/mindspore-ai/mindspore），但 LocMoE 特定修改未公开。
  - **算法 pipeline 伪代码**：
  ```python
  # LocMoE 前向传播核心流程
  # Input: token_embeddings x_m of shape [T, d]
  # Experts: {E_0, ..., E_{n-1}} on EP domain, n=16
  # Hyperparams: alpha=0.01, mu (locality weight)

  # Step 1: GrAP Gating (替代 Dense Gating)
  # 分组平均池化: 将 d 维 token 均分为 n 组，每组取均值作为门控值
  # x_m: [T, d] -> reshape [T, n, d//n] -> mean(dim=2) -> [T, n]
  # 等价于正交权重矩阵 omega 与 x_m 的内积
  # omega_i 满足: omega_{i,j} = 1 if i*d/n <= j < (i+1)*d/n else 0
  gate_logits = x_m.reshape(T, n, d // n).mean(dim=2)  # [T, n]

  # Step 2: Top-1 Routing with Softmax
  gate_probs = softmax(gate_logits, dim=1)  # [T, n]
  expert_idx = argmax(gate_probs, dim=1)    # [T]

  # Step 3: Locality-Aware Token Dispatch
  # Group-wise All-to-All: 将 All-to-All 按 TP 域拆分
  # 每个 device 在 EP 域内负责部分传输，All-Gather 在 TP 域同步
  tokens_dispatched = groupwise_all_to_all(x_m, expert_idx)

  # Step 4: Expert FFN 计算 (切片与 All-to-All 重叠)
  # 每个 expert 执行: W_out · GeLU(W_in · x)
  for expert_i in range(n):
      mask_i = (expert_idx == expert_i)
      tokens_i = tokens_dispatched[mask_i]
      if len(tokens_i) > 0:
          output_i = W_out[expert_i] @ GeLU(W_in[expert_i] @ tokens_i)

  # Step 5: Combined Loss
  # Auxiliary loss (Switch Transformer 风格负载均衡)
  f_i = sum(expert_idx == i) / T  # expert i 的 token 比例
  P_i = mean(gate_probs[:, i])    # router 选择 expert i 的平均概率
  L_aux = alpha * n * sum(f_i * P_i for i in range(n))

  # Locality loss (KL 散度促使局部路由)
  # D_c: 当前 batch 中 (节点, expert) 的 token 分配分布
  # D_l: 完全局部化的理想分布（token 只在本地 expert）
  L_loc = mu * KL_divergence(D_c, D_l)

  # Task loss
  L_task = L_aux + L_loc + L_cross
  ```
  - **Expert Capacity 下界推导**：
    1. GrAP 层的正交门控权重满足 Lemma 2（各 expert 被等概率选择：P{i_j = i'} = 1/n）
    2. 基于高维球面几何（Lemma 3），当 d 很大且 δ = Θ(1/√d) 时，token 应分配给某 expert 的概率 p_δ ≈ 0.3
    3. 当 δ 增大（即夹角变小，token 与 expert 更匹配），p_δ 快速衰减至 0（仅少量 token 为 class-discriminative）
    4. 由此得到 ec_min = 1/(n·[1 - I_{δ²}(1/2, (d-1)/2)])，在大 d 下退化为 ec_min ≥ 1/(n·erfc(√(δ²d/(2-δ²))))
    5. 论文实验测得 δ≈0.03，可据此计算安全的 expert capacity 下界，避免超量分配

## Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 **LTDR (Long-Tailed Distribution-aware Router)**，包含两个核心模块：
    1. **Modality-specific Distribution-aware Router (MsDaR)**：针对 LVLM 中 vision tokens 服从 long-tailed 分布、language tokens 服从 uniform 分布的模态差异，移除 vision TER 的 load balancing 约束（只保留 language TER 的 load balancing），让 vision tail tokens 能路由到专业化 experts 而非被负载均衡打散。
    2. **Vision-specific Dynamic Expert Activation (VsDEA)**：定义 vision tail token 为 RPV (Routing Probability Variance) 高于所有 vision tokens 平均 RPV 的 token（约 13%），对这些 tail tokens 激活更多 experts（Top-a, a > k），采用 renormalized softmax 权重，作为 data-augmentation 策略提升 tail tokens 的学习效果。
  - 实验比较：
    - **Vision-Language 任务**：vs MoE-LLaVA (4Top2, StableLM-1.6B / Phi2-2.7B)、Molmo (64Top8, OLMoE-1B-7B)
    - **Vision 任务**：vs GMoE (4Top2 / 6Top2, ViT-S/16)
    - **路由策略对比**：Task routing、Cluster routing、Instruct routing、Dynamic routing、STGC、Modality routing、Distribution routing、DeepSeekMoE
    - **Ablations**：MsDaR vs MsDaR+VsDEA 模块有效性、Vision Token Selection 策略（VHTs vs IATs vs 不同固定比例 VTTs）、策略互换（language+MsDaR vs vision+MsDaR）、跨 router 性能（Top-1/Top-2/Top-a/Dynamic）、load balancing 系数消减（0.01→0.001 vs 移除）
    - 结果：Vision-Language 平均提升 1.2%/2.1%，Vision 平均提升 1.6%，inference time 不显著增加

- 硬件平台是什么，配置是什么。
  - **V100-30G**：NVIDIA V100 30GB
  - **A800-80G**：NVIDIA A800 80GB
  - 训练配置：epoch=1, learning rate=2e-5, cosine schedule, weight decay=0.0, load balancing loss coefficient=0.01, text max length=2048, batch size per GPU=16, precision=FP16
  - MoE-LLaVA train steps=original（MoE-LLaVA 默认设定），Molmo train steps=2000
  - VsDEA 中 a=4（others, MoE-LLaVA 上）/ a=12（Molmo 上, others use Top8）

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - **MoE-LLaVA**：StableLM-1.6B / Phi2-2.7B 作为 LLM backbone，MoE 层替换 FFN，4 experts Top2 routing，CLIP visual encoder
    - **Molmo**：OLMoE-1B-7B backbone，64 experts Top8 routing
    - **GMoE**：ViT-S/16 backbone，4 或 6 experts Top2 routing
  - **Instruction-tuning 数据集**：
    - MoE-LLaVA 用：LLaVA (158K) + ShareGPT (40K) + VQAv2 (83K) + GQA (72K) + OKVQA (9K) + OCRVQA (80K) + A-OKVQA (66K) + TextCaps (22K) + RefCOCO (48K) + VG (86K)，合计 665K
    - Molmo 用：VQAv2 (440K) + TextVQA (35K) + OKVQA (9K) + ChartQA (28K) + DocVQA (39K) + InfographicVQA (24K) + AI2D (15K) + A-OKVQA (17K) + AndroidControl (300K) + ScienceQA (6K) + TabWMP (23K) + ST-VQA (25K) + TallyQA (250K)
    - 扩展数据集：Open-LLaVA-NeXT（额外 350K，总计 1021K）
  - **Benchmarks**：
    - Vision-Language：GQA, ScienceQA-IMG, TextVQA, POPE, MME, MMBench, MM-Vet（MoE-LLaVA 路径）；ChartQA, DocVQA, AI2D, VQAv2, AndroidControl, CountBenchQA（Molmo 路径）
    - Vision：PACS, VLCS, Office-Home（GMoE 路径）
    - High vision-token scenario：NLVR2（多图 VQA）
  - Training loss：Auto-regressive generative loss `L = -Σ log p(y_i | V, T, Y_{<i}, θ)`，仅计算生成文本部分

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确给出开源代码链接。基于 MoE-LLaVA（open source）、Molmo（open weights）、GMoE（open source）实现。
  - **LTDR 算法 Pipeline 伪代码**：
    ```
    # === 输入 ===
    # V ∈ R^{M×D}: vision tokens (M ≈ 576 from CLIP)
    # T ∈ R^{N×D}: language tokens
    # K = 4 (MoE-LLaVA) / 64 (Molmo): total experts
    # k = 2 (MoE-LLaVA) / 8 (Molmo): default activated experts
    # a = 4 (MoE-LLaVA) / 12 (Molmo): activated experts for tail tokens
    # W ∈ R^{D×K}: router weight matrix (trainable linear layer)

    # === Step 1: 标准 Router Forward ===
    f_v = V @ W           # [M, K] vision logits
    f_t = T @ W           # [N, K] language logits
    P_v = softmax(f_v)    # [M, K] routing probabilities
    P_t = softmax(f_t)    # [N, K] routing probabilities

    # === Step 2: MsDaR — Load Balancing（仅 language） ===
    for i in 1..K:
        F_i = count(T routed to expert i) / N    # language token fraction
        G_i = mean(P_t[:, i])                     # mean routing prob
    L_balancing = K * sum_{i=1..K} F_i * G_i
    # vision tokens 不参与 L_balancing 计算

    # === Step 3: VsDEA — RPV 计算与 Tail 分类 ===
    RPV_v = Variance(P_v, dim=1)              # [M], per-token variance
    threshold = Mean(RPV_v)                    # scalar, ~13th percentile
    is_tail = (RPV_v > threshold)              # [M] boolean mask

    # === Step 4: VsDEA — 差异化 Expert Activation ===
    # Vision tail tokens: activate a experts (a > k)
    for v_i in V where is_tail[i]:
        indices = TopK(f_v[i], a)              # select top-a experts
        w = softmax(f_v[i][indices])            # renormalized weights
        output = sum_{j=1..a} w_j * Expert_j(v_i)

    # All other tokens (vision head + all language): activate k experts
    for x in [V_head, T]:
        indices = TopK(f(x), k)
        w = softmax(f(x)[indices])
        output = sum_{j=1..k} w_j * Expert_j(x)
    ```
  - **关键数学关系**：
    - RPV: `Var(P(v)) = (1/K) Σ (P(v)_i - μ)²`，其中 μ = (1/K) Σ P(v)_i
    - Vision head/tail 阈值: `Mean(RPV(V)) = (1/M) Σ RPV(v_i)`，自适应动态阈值
    - 实验测得 vision tail tokens 约占 13%，head tokens 约占 87%
  - **Inference 性能**：all-to-all 通信速度由最慢 expert 决定，VsDEA 激活更多 experts 但不显著增加最慢 expert 负载，因此 inference time 几乎不变（V100 上 avg 1100s vs 1108s baseline，A800 上 846s vs 917s）

## LongCat-Flash Technical Report

- 属于算法pipeline的实现是什么？实验比较什么？
  - LongCat-Flash 提出四项算法 pipeline 创新：
    1. **Zero-Computation Experts（零计算专家）**：在 MoE 的 FFN expert pool 中引入 Z 个 zero-computation experts，其输出直接等于输入（identity function），不引入额外计算量。Router 从 N+Z 个 experts 中选 top-K，实际激活的 FFN experts 数量随 token 的上下文重要性动态变化（18.6B-31.3B 参数，平均 27B）。通过 expert bias + PID 控制器调节零计算专家选择比例，确保平均计算负载收敛到目标值。公式：`MoE(x_t) = Σ g_i · E_i(x_t)`，其中 `E_i(x_t) = FFN_i(x_t)` if `1 ≤ i ≤ N` else `E_i(x_t) = x_t`。
    2. **Shortcut-Connected MoE (ScMoE)**：引入跨层 shortcut 连接，从同一层第一个 MLA block 的输出直连到 MoE block，允许前一层的 Dense FFN 计算与当前层 MoE 的 dispatch/combine 通信并行执行。将 token 维度切分为两个 chunk，实现 chunk 间互相重叠以及与 dense FFN 的重叠。
    3. **Variance Alignment for MLA**：在 MLA 的低秩分解路径中引入 scale-correction 因子 α_q 和 α_kv。因 query 压缩维度 d_q 和 KV 压缩维度 d_kv 产生的 query 分量 q_t^C 和 key 分量 k_t^C 方差与 d_model 不同，通过 `α_q = √(d_model/d_q)` 和 `α_kv = √(d_model/d_kv)` 将低秩路径分量的方差对齐到 d_model 参考尺度，解决缩放过程中的注意力分数不稳定问题。
    4. **Variance Compensation for Experts Init**：fine-grained expert segmentation 将每个 expert 细分为 m 个小 expert 后，gating dilution 和 dimensional reduction 各使输出方差减少约 m 倍。通过聚合输出乘以缩放因子 `γ = m` 补偿方差，保持 MoE 层输出方差与分割前一致。
    5. **Multi-Token Prediction (MTP)**：单一 dense layer 作为 MTP head，在训练中期引入，接受率超 90%。
  - 实验比较：
    - with/without zero-computation experts 在匹配计算预算下的 validation loss（Figure 3a）：zero-expert 变体激活 4.2B-7.0B 参数但保持 8 FFN experts 期望，loss 持续低于固定 top-k=8 的 baseline。
    - with/without ScMoE 在四种模型配置（2.4B-16B MLA, 3B-20B MHA, 15B-193B GQA）下的 training loss 曲线（Figure 4）：loss 几乎完全相同，证明 ScMoE 是 quality-neutral。
    - with/without scale-correction MLA 在 1B activated MoE 上的 validation loss 收敛曲线（Figure 5a）：scale-correction 带来更低 loss。
    - Model growth vs random init 在 6B activated MoE 上的 validation loss（Figure 5b）：model growth 初期 loss 上升但最终收敛到更优值。
    - MTP head 结构对比（Table 5）：Dense layer (1.41% params, 92.1% accept rate) vs ScMoE layer (4.17% params, 92.9% accept rate)，Dense layer 以更少参数取得接近的接受率。
    - Base model vs DeepSeek-V3.1, Llama-4-Maverick, Kimi-K2 在 MMLU/MMLU-Pro/CEval 等全面 benchmark 上评估。
    - Chat model vs DeepSeek-V3.1, Qwen3-235B, Kimi-K2, GPT-4.1, Claude4-Sonnet, Gemini2.5-Flash 在 ArenaHard, IFEval, MATH500, AIME, SWE-Bench, τ²-Bench 等全面 benchmark 上评估。

- 硬件平台是什么，配置是什么。
  - **训练**：NVIDIA H800-80GB GPU × 数万张（tens of thousands），200Gb/s per accelerator RDMA 网络，NVLink intra-node 互联。以 Expert Parallelism Group (EP=32) 为基本单元，CP=8，V-ZB pipeline。训练持续 30 天，98.48% 可用率。
  - **推理**：NVIDIA H800-80GB GPU，128 GPUs 作为典型部署单元（2 nodes × 16 GPUs 为最小 PD-disaggregation 部署单元），NVLink intra-node + RDMA inter-node (GPUDirect RDMA)。EP 部署可根据需要伸缩到上千 GPUs。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LongCat-Flash，560B total params MoE，28 layers（不含 MTP layer），hidden dim 6144，64 attention heads per MLA，per-head dim 128。KV compression dim 512，query compression dim 1536。Dense FFN intermediate dim 12288，每个 FFN expert dim 2048。每层 512 FFN experts + 256 zero-computation experts，top-K=12。平均激活 ~27B params（18.6B-31.3B 范围）。Tokenizer: BPE，vocab size 131,072。
  - **Pre-training 数据**：~20T tokens（第一阶段 8k seqlen），多阶段包括通用预训练（two-stage data mixture）+ 推理代码增强（hundreds of billions of high-quality tokens）+ 长上下文扩展（80B → 32k + 20B → 128k tokens）。
  - **评估 Benchmark**：
    - Base model: MMLU, MMLU-Pro, CEval, CMMLU, GPQA, SuperGPQA, BBH, DROP, PIQA, WinoGrande, CLUEWSC, GSM8K, MATH, MBPP+, HumanEval+, MultiPL-E, CRUXEval
    - Chat model: MMLU, MMLU-Pro, ArenaHard-V2, CEval, CMMLU, IFEval, COLLIE, Meeseeks-zh, MATH500, AIME24, AIME25, BeyondAIME, GPQA-diamond, DROP, ZebraLogic, GraphWalks-128k, LiveCodeBench, Humaneval+, MBPP+, SWE-Bench-Verified, TerminalBench, τ²-Bench, AceBench, VitaBench
    - 内部 benchmarks: Meeseeks (multi-turn instruction-following), VitaBench (real-world agentic tasks from Meituan business scenarios)

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：模型权重开源在 Hugging Face (https://huggingface.co/meituan-longcat)，GitHub (https://github.com/meituan-longcat)。
  - **Zero-Computation Experts 例子（张量计算）**：
    假设 N=4 个 FFN experts，Z=2 个 zero-computation experts，K=3，K_e=2。
    ```
    # input x_t: [batch, seq, d_model]
    # router: softmax over N+Z=6 dimensions
    router_logits = router(x_t)  # [batch, seq, 6]
    router_probs = softmax(router_logits + expert_bias)  # expert_bias 由 PID controller 动态更新
    topk_indices = topk(router_probs, k=3)  # 从 6 个中选 3 个

    # Expert computation
    output = zeros_like(x_t)
    for idx in topk_indices:
        g_i = router_probs[idx]
        if idx < 4:  # FFN experts
            output += g_i * FFN[idx](x_t)
        else:        # Zero-computation experts
            output += g_i * x_t  # identity, no FLOPs

    # PID bias update (对 FFN experts only, zero-comp experts 不更新):
    # Δb_i = μ * (K_e/K * 1/N - T_i/(K*T_all))  for 1 <= i <= N
    ```
  - **ScMoE 例子（调度时序）**：
    Token 维度分为 chunk_a 和 chunk_b。单层执行顺序：
    ```
    Stage 1: MLA_0(x) → output → 分叉
    Stage 2 (并行):
      - Dense FFN(chunk_a_input) + Attn_0_QKV(chunk_a_input)
      - All-to-All Dispatch(chunk_b tokens)
    Stage 3: MoE GEMM(chunk_b) → All-to-All Combine(chunk_b)
    Stage 4 (并行):
      - All-to-All Combine(chunk_a) + Dense FFN(chunk_b_input)
      - Attn_1_Core(chunk_a): Core Attention + Output Projection
    ```
    通过 ScMoE shortcut，Dense FFN 计算可与 MoE 的 dispatch/combine 通信充分重叠，TPOT 理论值降低近 50%（vs DeepSeek-V3 的 TBO）。

## Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

- 属于算法pipeline的实现是什么？实验比较什么？
  - Lory 提出三种算法 pipeline 创新，实现完全可微的 MoE 自回归语言模型预训练：
    1. **Fully Differentiable Expert Merging**：在参数空间（而非激活空间）对专家进行软合并。给定路由权重 e_i，合并后的 FFN 参数为 θ̄ = Σ_i e_i · θ_i，然后用合并后的 FFN 处理输入：o_x = FFN(h_x; θ̄)。与稀疏 MoE 的 top-k 离散路由不同，整个过程完全可微，无需离散路由决策和辅助负载均衡损失。
    2. **Causal Segment Routing**：将输入序列分为 T=256 token 的固定长度段。对于第 k 段 S_k (k>1)，使用前一段 S_{k-1} 的隐藏表示平均值 h̄_{k-1} 计算路由权重并合并专家 FFN，然后用合并后的 FFN 处理当前段所有 token。对于第一段 S_1，使用自身段表示但施加 stop-gradient 防止信息泄露。推理时仅用 prompt 做一次路由决策，后续生成全程使用合并后的 FFN。
    3. **Similarity-based Data Batching**：使用 Contriever 计算文档相似度，通过贪心搜索算法将语义相似的文档拼接成训练实例，使相邻段来自相似领域，促进专家按领域/主题专业化。
  - 实验比较：
    - Lory MoE 模型 vs 参数匹配的 Dense 模型（0.3B/1.5B active params, 8/16/32 experts）
    - Lory vs Expert Choice (EC) MoE（段级路由和 token 级路由两种变体，capacity factor=1）
    - Ablation：causal segment routing vs prefix routing；similarity-based batching vs random batching
    - 扩展实验：7B/4E 模型（无 similarity batching）
    - 主要结果：0.3B/32E 在 Books 上 perplexity 改善 +13.9%，下游任务 averaged improvement +3.7% (commonsense), +3.3% (reading), +1.5% (QA), +11.1% (classification)

- 硬件平台是什么，配置是什么。
  - **训练**：最多 64 块 NVIDIA A100 GPU
  - **分布式训练**：数据并行 + ZeRO 优化（Rajbhandari et al., 2020）
  - 训练吞吐量（Table 3，A100）：0.3B dense 29,000 tokens/s/gpu；0.3B/8E 24,500；0.3B/16E 22,900；0.3B/32E 20,800
  - 软件环境：论文未明确说明具体 CUDA/PyTorch 版本

- 模型是什么。数据集和bench分别是什么。
  - **模型架构**（Table 4）：
    - 0.3B：24 layers, hidden dim 1024, 16 attention heads。MoE 变体 0.3B/8E (1.8B total), 0.3B/16E (3.5B), 0.3B/32E (6.8B)
    - 1.5B：48 layers, hidden dim 1536, 24 attention heads。MoE 变体 1.5B/8E (7.8B total), 1.5B/16E (15.0B), 1.5B/32E (29.5B)
    - 7B (extended)：32 layers, hidden dim 4096, 32 attention heads。7B/4E (19.7B total)
    - 解码器仅 Transformer，所有 FFN 层替换为 MoE 层，SwiGLU 激活，LLaMA tokenizer，context window 4096
  - **训练数据集**：CommonCrawl 的 150B token 随机子集（Wenzek et al., 2019），使用 similarity-based batching 构造训练实例。7B 实验使用 LLaMA2 的多语料混合的 200B token 子集，使用随机 batching。
  - **评估 Benchmark**：
    - **语言建模（Perplexity）**：arXiv, Books, Wikipedia, C4, Python code（各 1000 样本，4096 tokens/sample）
    - **Commonsense Reasoning**：BoolQ, PIQA, SIQA, HellaSwag, WinoGrande
    - **Reading Comprehension**：RACE-m, RACE-h, ARC-easy, ARC-challenge
    - **Closed-book QA**：Natural Questions, TriviaQA
    - **Text Classification**：AGNews, SST-2, Amazon, Yelp, FEVER, MRPC
  - 训练配置：AdamW (β1=0.9, β2=0.95), lr=2e-4, cosine schedule, batch size 1M tokens, 前 5% 训练步为 dense warmup 阶段（先训练参数匹配的 dense 模型，再复制 FFN 层初始化 MoE）

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **未开源**：论文未提供代码仓库链接，arXiv 页面和相关搜索均未找到开源实现。
  - **Causal Segment Routing 伪代码**（论文 Algorithm 1, Appendix A）：
```python
# B: batch size, L: sequence length, d: hidden dim
# E: number of experts, T: segment length (256)
# R: routing network (linear layer)
input: x  # (B, L, d)
N = L // T                     # number of segments per sample
seg_x = x.view(B*N, T, d)      # split into segments
repr = mean(seg_x, dim=1)      # segment representations (B*N, d)
e = softmax(R(repr), dim=-1)   # routing weights (B*N, E)
e_first = e.view(B, N, E)[:, 0] # first segment routing
e = roll(e, 1)                  # shift by 1 -> causal
e = e.view(B, N, E)
e[:, 0] = stop_grad(e_first)   # first segment uses own repr (no leakage)
e = e.view(B*N, E)
seg_y = moe_ffn(seg_x, e)      # merged FFN computed per segment
y = seg_y.view(B, L, d)        # back to instance view
```
  - **moe_ffn 函数实现（参数空间合并）**：
```python
def moe_ffn(seg_x, e):
    # seg_x: (B*N, T, d), e: (B*N, E)
    # experts: list of E expert FFNs, each = (W_gate, W_up, W_down)
    merged_W_gate = sum(e[:, i] * expert[i].W_gate for i in range(E))
    merged_W_up   = sum(e[:, i] * expert[i].W_up   for i in range(E))
    merged_W_down = sum(e[:, i] * expert[i].W_down for i in range(E))
    # SwiGLU FFN:
    gate = silu(matmul(seg_x, merged_W_gate))
    up   = matmul(seg_x, merged_W_up)
    out  = matmul(gate * up, merged_W_down)
    return out
```
  - 关键张量流（单 MoE 层，L=4096, T=256, E=32）：
    1. 输入 x (4096 tokens) → 分为 16 段 → 每段 256 tokens
    2. Segment 0 的 avg representation h̄_0 计算路由权重 e_0 (32-dim softmax) → stop_gradient → 合并 FFN_0 → 处理 Segment 0
    3. Segment 0 的 avg representation h̄_0 计算路由权重 e_1 → 合并 FFN_1 → 处理 Segment 1（causal shift）
    4. 依此类推，每个 segment 使用前一段的表示计算路由
    5. 合并操作 FLOPs 开销：E/T × (FFN FLOPs)，E=32, T=256 → ~12.5% 额外计算 vs Dense（MoE 层）。总模型开销更小（~15-28% 训练减速，Table 3）
  - 推理时：给定 prompt → 每层用 prompt 的平均隐藏表示计算一次路由权重 → 合并 FFN → 后续所有生成 token 使用该合并 FFN，与 Dense 模型推理效率相同
  - 合并操作仅每段执行一次（L/T 次），而非每 token 执行一次，这是 segment-level routing 的关键效率优势

## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- 属于算法pipeline的实现是什么？实验比较什么？
  - LYNX 提出了一种新的算法 pipeline —— **AffinityBinning（亲和力分箱）** 技术，用于在 batch 级别动态减少 MoE 推理中激活的专家数量。核心算法包括四个步骤：
    1. **Confidence Analysis（置信度分析）**：对每层的每个 token，计算其相对于 top-1 expert 的 log-ratio（即 router logits 之差，等价于 softmax 概率比的对数）。对于 sigmoid-based router（如 DeepSeek），使用 pre-sigmoid scores 的差值。将这些值按模型 sparsity ratio（k/N）决定的 bin width（由参数 α 控制）和 bin count（由参数 β 控制）进行离散化。bin=0 表示最高亲和力（与 top-1 expert 的 log-ratio 为 0），越负的 bin 表示亲和力差距越大。α 和 β 仅由模型架构的 sparsity ratio 决定，无需 task-specific tuning。
    2. **Adaptive Expert Scoring（自适应专家评分）**：对 batch 中所有 token 的 binned 偏好进行加权汇总。使用 batch_size 为底数的指数加权方案：score(e) = Σ_t (batch_size)^{bin(t,e)}，其中 bin(t,e) 为 token t 对 expert e 的 AffinityBinning 结果。高置信度 token 的偏好专家获得指数级更高权重，低置信度 token 的偏好被大幅降权。动态确定最终 active expert set 的大小。
    3. **Expert Remapping（专家重映射）**：将低置信度 token 的 expert assignment 重映射到 minimal critical expert set 内。高置信度 token 始终保留其 top-ranked expert。Preserve top-k 激活语义：每个 token 仍然激活 k 个 expert。
    4. **Phase-Aware Gating（相位感知门控）**：仅在 memory-bound decode iterations 中启用上述 pipeline，prefill 等 compute-bound 阶段直接绕过。
  - 实验比较：
    - Baseline：vLLM 默认推理（标准 top-k routing，无 expert reduction）
    - LYNX vs Baseline：TPOT、准确率、系统吞吐量、SLO-aware throughput
    - 对比范围：4 个模型家族（Qwen, Mixtral, DeepSeek, Llama），8 个 benchmark（GSM8K, HumanEval, MBPP, MATH, ChartQA, MMMU, AIME, GPQA）
    - 关键结果：median TPOT 降低 1.09-1.30x，准确率偏差 <1%，平均情况甚至提升准确率

- 硬件平台是什么，配置是什么。
  - NVIDIA H200 GPU (141 GB HBM)，SXM NVLink 互联
  - 2x AMD EPYC 9554 64-Core CPU，1.5 TB DRAM
  - Ubuntu 22.04.4 LTS，NVIDIA driver 560.35.05，CUDA 12.6
  - TP=2/4 配置，EP=2/4 实验使用 A100

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - Qwen2-57B-A14B-Instruct (k=8, N=64, sparsity ratio 0.125)
    - Qwen3-30B-A3B-Instruct
    - Qwen3-235B-A22B-Thinking-2507
    - Mixtral-8x7B-Instruct-v0.1 (k=2, N=8, sparsity ratio 0.25)
    - DeepSeek-V2-Coder (k=8, N=256, sparsity ratio 0.03)
    - Llama-4-Maverick-17B-128E-Instruct
    - Llama-4-Scout-17B-16E-Instruct
  - **数据集/Benchmark**：
    - 代码：HumanEval, MBPP
    - 数学：GSM8K, Minerva Math (Algebra)
    - 视觉推理：ChartQA, MMMU
    - 推理：AIME, GPQA
    - 真实 trace：ShareGPT, Mooncake
    - 准确率指标：Pass@1 (HE/MBPP), Exact Match (MATH/GSM8K)，遵循 EleutherAI LM evaluation harness 规范

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供开源代码链接。arxiv ID 2411.08982。
  - **算法 Pipeline 伪代码**：
    ```
    输入: batch tokens T = {t1, t2, ..., tB}, router logits L (B x N), sparsity ratio k/N
    参数: α (bin width factor), β (max bin count) - 由 k/N 决定

    # Step 1: Confidence Analysis (per-token, per-layer)
    for each token t in batch:
        top1_logit = max(L[t])
        for each expert e in top-k(t):
            log_ratio = L[t][e] - top1_logit  # difference of logits
            bin[t][e] = clamp(floor(log_ratio * α), -β, 0)  # discretize to [negative, 0]

    # Step 2: Adaptive Expert Scoring (batch-level)
    for each expert e in union of all top-k selections:
        score[e] = 0
        for each token t in batch:
            if e in top-k(t):
                score[e] += B^{bin[t][e]}  # exponential weighting by batch_size

    # Step 3: Determine Active Expert Set
    # Keep high-confidence tokens' top-1 experts unconditionally
    # Select additional experts based on score threshold (dynamic)
    active_experts = select_top_by_score(scores, threshold=determine_by_distribution(scores))

    # Step 4: Expert Remapping
    for each token t in batch:
        if confidence(t) < threshold:  # low-confidence token
            remap low-ranked experts to alternatives in active_experts
    ```
  - **关键张量计算流**（单 MoE layer, Qwen2-57B, batch B=16, N=64, k=8）：
    1. Router logits (B x 64) → softmax → top-8 per token
    2. Confidence Analyzer：对每个 (token, expert in top-8) 计算 log_ratio = logit[e] - logit[top1] → 离散化为 bin ∈ [-β, 0]（例如 β=5，则 6 个 bin）
    3. Adaptive Scorer：score[64] = Σ_t B^{bin[t][e]}，B=16 → 高置信度 token (bin=0) 贡献 16^0=1，低置信度 (bin=-5) 贡献 16^{-5}≈0.0001
    4. 动态阈值筛选 → active set 大小（例如从 25 个降至 15 个）
    5. Remapper：low-confidence tokens 的 lower-ranked experts 重映射到 active set 内 → 仍保持每个 token 8 个 expert
    6. 最终 dispatch：(B x 8) → (active_count x ...) 的 GEMM

## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- 属于算法pipeline的实现是什么？实验比较什么？
  - FineMoE 的算法 pipeline 创新包含：
    1. **Expert Map 数据结构**：记录每个 inference iteration 中每层 gate network 输出的全概率分布 P_l^{(i)} ∈ R^J（而非 coarse-grained 的 binary 激活或 hit count）。expert map 可退化恢复 coarse-grained 信息（对概率分布取 top-K + 聚合迭代）。直观上，expert map 不仅识别哪些 experts 被选择，还捕获 gate network 对所有 experts 的 confidence/preference 分布。
    2. **Semantic-based Expert Map Search**：利用 MoE 模型的 embedding layer 输出作为 semantic embedding，与 Expert Map Store 中历史 semantic embeddings 计算 pairwise cosine similarity，选择最相似的 historical iteration 的 expert map 指导 expert prefetching。该方法基于"语义相似的 prompts 具有相似的 expert 选择模式"的假设。
    3. **Trajectory-based Expert Map Search**：对第 l ∈ [d+1, L] 层，收集前 (l-d) 层已观察到的 expert probability trajectory（即前序层的 P_1,...,P_{l-d}），与 Expert Map Store 中历史 expert maps 对应的前 l-d 层计算 cosine similarity，选择最匹配的 expert map 的 P_l 指导当前层 prefetching。
    4. **Similarity-aware Dynamic Expert Selection**：对每层 l 根据 search confidence（cosine similarity score）动态计算 selection threshold δ_l = Clip(1-score, 0, 1)。高 similarity → 低 δ → 选择 fewer high-probability experts；低 similarity → 高 δ → 选择 more experts 防止 miss。
    5. **Expert Map Deduplication**：通过 unified redundancy score RDY = (d/L)*score^{sem} + ((L-d)/L)*score^{traj} 评估新 iteration 与 Expert Map Store 中旧 iterations 的冗余度，达到 capacity C 时剔除最相似（冗余）的旧 map 以维持多样性。理论分析表明保持 2LJ expert maps 可保证 ≥75% similarity lower bound，保持 (1/2)LJ*ln(LJ) maps 可保证 ≥98% similarity。
  - 实验比较（算法层面消融）：
    - Expert pattern tracking 对比：1) Speculate（speculative prediction），2) Hit count（request-level, MoE-Infinity 方式），3) Map(T)（仅 trajectory similarity），4) Map(T+S)（trajectory + semantic 但 static top-K selection），5) Map(T+S+δ)（全部 feature 启用）。结果：随 feature 递增恢复，expert hit rate 逐步提升
    - Pearson correlation 验证：semantic similarity 和 trajectory similarity 均与 expert hit rate 呈正相关（所有模型/数据集组合 Pearson coefficient > 0）

- 硬件平台是什么，配置是什么。
  - 主测试台：6× NVIDIA GeForce RTX 3090 24GB, NVLink, PCIe 4.0 32GB/s, AMD Ryzen Threadripper PRO 3955WX 32C, 480GB CPU RAM
  - 高配测试台：NVIDIA A100 80GB HBM2e, 2 TB/s 峰值带宽

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Mixtral-8×7B（12.9B active / 46.7B total, 2/8 experts/layer, 32 layers）、Qwen1.5-MoE（2.7B active / 14.3B total, 4/60 experts/layer, 24 layers）、Phi-3.5-MoE（6.6B active / 42B total, 2/16 experts/layer, 32 layers）
  - **数据集**：LMSYS-Chat-1M、ShareGPT；online 实验使用 Azure LLM 推理 traces 驱动请求
  - **Metrics**：TTFT（prefill）、TPOT（decode）、expert hit rate、CDF of end-to-end request latency

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：FineMoE 原型基于 MoE-Infinity 代码库（https://github.com/TorchMoE/MoE-Infinity），自身未发现独立开源仓库
  - **Expert Map Search 伪代码**：
  ```
  # Input: 新 prompt 的 semantic embedding sem_new ∈ R^{B×h}
  #        Expert Map Store: sem_old ∈ R^{C×h}, map_old ∈ R^{C×L×J}
  #        prefetch distance d, total layers L

  # 对每层 l ∈ [1, L]:
  for l in range(1, L+1):
      if l <= d:
          # Semantic-based search for initial layers
          score_sem = cosine_similarity(sem_new, sem_old)  # R^{B×C}
          best_iter = argmax(score_sem, dim=-1)  # 每 batch 选最相似 iteration
          # 使用 best_iter 的 expert map 中前 d 层指导 prefetch
          for target_l in range(1, d+1):
              P = map_old[best_iter, target_l, :]  # R^{B×J}
              prefetch_experts(P, scores=score_sem)
      else:
          # Trajectory-based search for later layers
          traj_new = concat([P_1, ..., P_{l-d}])  # R^{B×(l-d)J}
          traj_old = map_old[:, :(l-d), :].reshape(C, -1)  # R^{C×(l-d)J}
          score_traj = cosine_similarity(traj_new, traj_old)  # R^{B×C}
          best_iter = argmax(score_traj, dim=-1)
          P = map_old[best_iter, l, :]  # R^{B×J}
          prefetch_experts(P, scores=score_traj)
  ```
  - **Similarity-aware Expert Selection 伪代码**：
  ```
  # Input: searched P_l ∈ R^J (probability distribution), score ∈ [-1, 1]
  def select_experts_to_prefetch(P_l, score, K):
      δ = clip(1 - score, 0, 1)  # 相似度低时选更多 experts
      sorted_experts = argsort(P_l, descending=True)
      E_prefetch = []
      sum_p = 0.0
      for j in sorted_experts:
          E_prefetch.append(j)
          sum_p += P_l[j]
          if sum_p >= δ and len(E_prefetch) >= K:
              break
      return E_prefetch
  ```
  - **关键张量计算流**（Mixtral-8×7B, L=32, J=8, K=2, B=1）：
    1. Semantic embedding extraction: token_ids → embedding_layer → sem_new ∈ R^{1×4096}
    2. Semantic search: cos_sim(sem_new, sem_old[1..C]) → score_sem ∈ R^{1×C} → select best_iter
    3. Layer 1 (l ≤ d=3): P_1 = map_old[best_iter, 0, :] = {p_{1,1},...,p_{1,8}} → δ = clip(1-score_sem, 0, 1) → select experts until Σp ≥ δ and |E| ≥ 2 → prefetch E_prefetch
    4. Layer 4 (l > d): traj_new = concat(P_1, P_2, P_3) ∈ R^{1×24} → cos_sim(traj_new, traj_old[1..C]) → score_traj → best_iter → P_4 from map_old[best_iter] → select experts with δ = clip(1-score_traj, 0, 1)
    5. Repeat to Layer 32

## LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LSH-MoE框架，在MoE训练过程中使用Locality-Sensitive Hashing（LSH）对all-to-all通信前的token进行在线聚类压缩，仅传输聚类中心（centroid）替代完整token，从而减少通信量。核心包括两部分：(1) 基于cross-polytope hashing的高效LSH聚类算法，将token映射到桶中并计算聚类中心；(2) 基于残差的误差补偿方案（residual-based error compensation），记录每个token与其聚类中心的残差，在expert计算后将残差加回输出，弥补压缩带来的精度损失。
  - 实验比较：对比原始无压缩MoE训练与LSH-MoE（有/无error compensation）的收敛速度和下游任务精度。消融实验比较不同hash函数数量和类型（cross-polytope vs spherical-plane）对压缩率和模型质量的影响。

- 硬件平台是什么，配置是什么。
  - V100 Cluster: 2台服务器，每台8× NVIDIA V100 (32GB)，NVLink 2.0，跨机RDMA NIC 100 Gbps
  - A100 Cluster: 4台服务器，每台8× NVIDIA A100 (40GB)，NVLink 3.0，跨机双RDMA NIC 200 Gbps
  - 软件: Ubuntu 20.04, CUDA 11.3, cuDNN 8.2.0, NCCL 2.12.7, PyTorch 1.11

- 模型是什么。数据集和bench分别是什么。
  - 模型: RoBERTa-MoE (394M total, 16 experts), T5-MoE (~9.3B total, 16 experts), GPT-MoE 15B (16 experts, top-2 gating), GPT-MoE 52B (512 experts), Swin-MoE-L (946M, 32 experts)
  - 数据集: BooksCorpus (~800M words) + English Wikipedia (~2.5B words) for RoBERTa-MoE; 工业数据集 (~500M words) for T5-MoE span-masked LM pretraining; GLUE benchmark for GPT-MoE fine-tuning; ImageNet-1K for Swin-MoE fine-tuning

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文在补充材料中提交了代码，但T5-MoE因公司平台限制无法公开训练代码。论文明确指出方法是框架无关的，可应用于Hetu-MoE、DeepSpeed-MoE、Tutel等框架。
  - 算法pipeline执行流程（基于论文Algorithm 1）：
    1. Gate网络计算token到expert的映射：`ζ = G(X)`，将输入X分派到各expert的token集{X_i}
    2. 对每个expert i的token集X_i执行LSH聚类：
       - `IDX_i = LSH(X_i)` — 使用cross-polytope hashing将每个token映射到桶
       - `LSH(x) = argmax_{i∈{±1,...,±d}} |Rx|_i` — 随机旋转矩阵R将x映射到cross-polytope最近顶点
    3. 计算每个cluster j的聚类中心：`cluster_j = Mean(cluster_j)`
    4. 记录残差：`Δcluster_j = {x - cluster_j | x ∈ cluster_j}`
    5. 仅传输聚类中心C = {cluster_j}通过all-to-all通信（替代完整token）
    6. Expert对中心进行计算：`E(cluster_j)`
    7. 结果通过all-to-all传回
    8. 残差补偿还原输出：`Y_ij = {E(cluster_j) + ΔCluster_jk | k=1,...,N_j}`
  - 默认参数：6个hash函数，cross-polytope hashing，压缩率约20%时精度无损

