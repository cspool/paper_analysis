## PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文系统性地研究了构建高性能小语言模型（1B-1.5B参数级）的三个核心维度：神经架构设计、参数初始化策略和模型优化策略。
  - **神经架构——Compact Tokenizer**：发现从大模型（100k词表）直接继承tokenizer对小模型不友好，embedding+head层占38.19%参数。通过频率分析发现top-48k词汇覆盖97.86%训练语料，压缩至48k词表使embedding+head参数占比降至18.07%，性能最优。
  - **神经架构——Architecture Tweak**：在1B参数量约束下探索depth/width/FFN expansion rate的影响。Spearmanr系数显示depth与性能相关性最高（0.528），expansion rate无明显线性关系。推荐20层depth、expansion rate 2.77的配置，兼顾性能与推理速度（V100上29.49 tokens/s）。
  - **参数初始化——Parameter Inheritance**：从PanGu-π-7B通过learnable binary mask继承参数。关键发现：(1) Layer Selection：中间层冗余度高，首尾2-3层对性能至关重要；(2) Intra-layer Selection：数据驱动的learnable mask优于L1/L2/Taylor等启发式方法。
  - **模型优化——Multi-round Training**：小模型容量有限导致严重catastrophic forgetting。提出基于loss的概率采样策略（p_i = exp(l_i)/Σ_j exp(l_j)），50%采样率的第二轮训练即可获得主要收益。同时探索batch size与learning rate缩放关系（推荐r=0.5，batch size < 4M为安全范围）。
  - 实验比较：(1) 消融实验在50B tokens子集上验证各组件，用ARC-E/HellaSwag/C3评估；(2) PanGu-π-1B Pro vs 原版PanGu-π-1B（平均提升8.87）；(3) PanGu-π-1.5B Pro vs Qwen-1.8B/Phi2-2.7B/Open-LLaMA-3B等SOTA小模型（在C-Eval/CMMLU/MMLU/AGI-Eval/BoolQ/AX-b/PIQA/EPRSTMT/XSum/C3十个benchmark上全面对比）。

- 硬件平台是什么，配置是什么。
  - 训练：华为昇腾910 (Huawei Ascend 910) 集群
  - 推理速度测试：单卡NVIDIA V100 GPU，FP16精度，batch size 20，测试生成510个新token（前缀2 tokens）的端到端速度
  - 实现框架：PyTorch，基于LLaMA-like架构

- 模型是什么。数据集和bench分别是什么。
  - 模型：PanGu-π-1B Pro（depth=21/width=1792/vocab=48k/expansion=2.77，总~1B参数）和 PanGu-π-1.5B Pro（depth=22/width=2048/vocab=48k，总~1.5B参数）。架构基于LLaMA-like Transformer，从PanGu-π-7B通过learnable binary mask继承参数
  - 预训练数据：1.6T tokens，中英文~1:1比例，来源为互联网多元语料；扩展版本PanGu-π-1.5B Pro*使用6T tokens
  - Benchmarks：使用OpenCompass框架评估。Examination: C-Eval、CMMLU；Knowledge: MMLU；Reasoning: AGI-Eval、BoolQ、AX-b、PIQA；Understanding: EPRSTMT、XSum、C3。消融实验使用ARC-Easy、HellaSwag、C3

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/YuchuanTian/RethinkTinyLM
  - 算法Pipeline核心四步骤：

  **Step 1 — Compact Tokenizer构建**
  ```
  输入: 大词表 V_large (100883 vocab), 训练语料 D (1.6T tokens)
  1. 统计 D 中每个vocabulary的出现频率 freq[v] for v in V_large
  2. 按频率降序排序 vocabularies，计算累积覆盖率:
     cum_coverage(k) = Σ_{i=1}^{k} freq[v_i] / Σ_{i=1}^{N} freq[v_i]
  3. 选择最小 k 使得 cum_coverage(k) > 97% → k=48k
  4. 使用 SentencePiece BPE 在 D 上训练 48k tokenizer
  5. 小模型使用新tokenizer，embedding层: 48k × d_model, head层: d_model × 48k
  输出: 紧凑tokenizer，参数占比从 ~38% 降至 ~18%
  ```
  关键公式：PEHL = (2 × V × d_model) / total_params，推荐 PEHL < 20%

  **Step 2 — 架构配置搜索与选择**
  ```
  约束: total_params ≈ 1B, vocab_size = 48k 固定
  搜索空间: depth ∈ [9,40], width ∈ [1280,2560], expansion_rate ∈ [1.0,4.0]
  过程:
  for each (depth, width, expansion_rate) in sampled_configs(30):
      model = build_llama_like(depth, width, expansion_rate, vocab=48k)
      train(model, data=5B_tokens)
      metrics = evaluate(model, [ARC-E, HellaSwag, C3])
      record(depth, width, expansion_rate, metrics.avg)
  相关性分析:
  Spearmanr(depth, performance) = 0.528  # 强正相关
  Spearmanr(width, performance) = -0.528  # 强负相关（因为width与depth在固定参数下互斥）
  Spearmanr(expansion_rate, performance) ≈ 0  # 无明显线性关系
  选择: depth=21, width=1792, expansion_rate=2.77 (PanGu-π-1B Pro)
        depth=22, width=2048, expansion_rate=2.77 (PanGu-π-1.5B Pro)
  ```

  **Step 3 — 参数继承 (Learnable Mask Pruning)**
  ```
  输入: 大模型权重 W_large ∈ R^{d_large × ...}（PanGu-π-7B），目标架构 A_small
  1. Layer Selection:
     对 W_large 的每一层 i ∈ [1, L_large]:
         测量跳过该层后的性能下降 Δperf(i)
     发现: 前2-3层和最后几层Δperf大（关键层），中间层Δperf小（冗余）
     策略: 保留 L_small 层 = 前3层 + 后3层 + 中间均匀采样(L_small-6)层
  2. Intra-layer Selection (Learnable Mask):
     对每层参数 W ∈ R^{d_out × d_in}:
         初始化二值mask M ∈ {0,1}^{d_out × d_in}，通过Gumbel-Sigmoid可微近似
         mask训练: min L_task(f_M ⊙ W_large) + λ · ||M||_1
         f_M = σ((log(u) - log(1-u) + log(α)) / τ)，τ anneal至0
     提取: W_small = extract_submatrix(W_large, where M=1)
  3. 证明有效性（Table 5）:
     Learnable mask: Avg = 48.08（最优）> Taylor: 47.90 > L2: 47.00 > L1: 46.06 > Base(随机初始化): 42.06
  输出: 初始化的tiny model权重 W_small，具有大模型的表征能力
  ```

  **Step 4 — Multi-round Training**
  ```
  输入: 预训练数据 D, 模型参数 θ, 训练轮数 R=2, 采样率 r=0.5
  Round 1:
     将 D 随机均分为 K=8 个part: D = {P_1, P_2, ..., P_K}
     顺序训练: for i=1..K: θ ← SGD_step(θ, P_i)
     记录每个batch的loss: L_k = {l_1, l_2, ..., l_N_k} for each P_k
  Round 2:
     对每个 P_k:
         计算归一化采样概率: p_i = exp(l_i) / Σ_{j=1}^{N_k} exp(l_j) 
         采样 r × N_k 个batch（困难样本被采样概率更高）
     合并采样数据为 D' = {sampled_batches}
     继续训练: for batch in D': θ ← SGD_step(θ, batch)
  ```
  效果（Table 6-7）：Single round Avg=51.61 → Two round r=50% Avg=54.46 (+2.85) → Three round Avg=54.44（饱和）

  训练超参数：Optimizer AdamW (β1=0.9, β2=0.95)；LR Cosine decay, initial LR=2e-4；Batch size 2M tokens；Weight decay 0.1
