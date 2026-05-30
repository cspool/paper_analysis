## PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

- baseline方法是什么？
  Baseline为PanGu-π-1B原始模型（Wang et al., 2023），以及当时SOTA的开源小模型系列（TinyLLaMA-1.1B, MobileLLaMA-1.4B/2.7B, Sheared-LLaMA-1.3B, Phi-1.3B/Phi2-2.7B, Open-LLaMA-3B, Qwen-1.8B等）。Baseline模型在训练策略上直接沿用了大语言模型的开发方法：使用大模型的tokenizer（100k+词表）、LLaMA标准架构（12层/2048宽）、随机初始化或简单裁剪、单轮训练（所有数据仅使用一次）。

  全栈执行例子（以PanGu-π-1B baseline为例）：
  - 算法Pipeline：100k BPE tokenizer → 12层/2048宽/expansion_rate=2.77 LLaMA-like Transformer → 随机初始化参数N(0, σ²) → AdamW + Cosine LR训练1.6T tokens一轮 → 在OpenCompass十个benchmark上评估
  - 系统框架：PyTorch，Huawei Ascend 910集群训练
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明

  Baseline核心缺陷：
  1. **Tokenizer参数冗余**：大模型tokenizer（100k词汇）在小模型中embedding+head层占比高达38.19%，大量参数浪费在低频词汇的表征上（bottom 50k+ vocab仅覆盖<3%语料），挤压了模型本体的表达能力。
  2. **架构配置不匹配**：小模型直接沿用大模型的宽-浅架构（12层/2048宽），未针对1B参数的约束探索depth-width-expansion的最佳配比，导致性能欠优。
  3. **初始化信息缺失**：随机初始化使小模型从零开始学习表征，无法利用大模型已学到的强表征能力，收敛慢且最终性能受限。
  4. **严重灾难性遗忘**：小模型容量有限，在1.6T tokens的单轮训练中，后期数据会严重覆盖早期学到的知识，表现为"之前seen的数据loss大幅上升"的forgetting现象。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过四项核心设计系统性地解决了小模型训练的四个痛点：

  - **Idea 1（Compact Tokenizer）** 解决缺陷1：统计分析发现top-48k词汇覆盖97.86%训练语料（长尾效应），将词表从100k压缩至48k，使embedding+head层参数占比从38.19%降至18.07%，释放~20%参数给Transformer本体。本质是识别并移除对语义覆盖贡献极低的词汇冗余，在词表覆盖率与参数效率间找到最优平衡点。

  - **Idea 2（Architecture Tweak: Depth > Width）** 解决缺陷2：通过网格搜索和Spearman相关性分析发现depth是小模型性能的首要决定因素（Spearmanr=0.528），expansion rate几乎无影响。在1B参数量约束下，将架构从12层/2048宽改为21层/1792宽，性能从Avg=42.41提升至46.53（+4.12，含tokenizer改进）。深层窄架构以推理速度为代价换取更强的序列建模能力——符合Transformer深度增加有利于学习层次化特征的直觉。

  - **Idea 3（Parameter Inheritance with Learnable Masks）** 解决缺陷3：从大模型（PanGu-π-7B）通过数据驱动的learnable binary masks继承关键参数。Layer Selection阶段发现首尾层关键、中间层冗余的普适规律（在LLaMA2/InternLM/PanGu-π四个大模型上验证），据此移除中间冗余层。Intra-layer阶段用Gumbel-Sigmoid可微二值mask自动学习重要神经元，相比L1/L2/Taylor启发式标准显著提升（Learnable: 48.08 vs Taylor: 47.90 vs L2: 47.00）。核心思想是将大模型的表征能力"蒸馏"到小模型初始化中，使小模型从一开始就站在大模型的肩膀上。

  - **Idea 4（Multi-round Training with Loss-guided Data Sampling）** 解决缺陷4：发现小模型在单轮训练末尾，early data的loss已从训练时的低值大幅反弹（forgetting证据）。提出第二轮训练：按p_i ∝ exp(loss_i)的概率采样数据（困难样本高频出现），50%采样率可达到接近全量二轮训练的效果。两轮训练Avg从51.61提升至54.46（+2.85），三轮开始饱和，在性能与训练成本间取得平衡。

  全栈执行例子（PanGu-π-1B Pro）：
  - 算法Pipeline：
    Step 1: 频率分析 → BPE训练48k紧凑tokenizer
    Step 2: Depth=21/Width=1792/Expansion=2.77 架构
    Step 3: PanGu-π-7B → Layer Selection（移除中间冗余层）→ Learnable Mask训练（Gumbel-Sigmoid binarization）→ 提取子矩阵作为初始化
    Step 4: Round 1全量训练1.6T tokens（AdamW, LR=2e-4, bs=2M, cosine decay）→ 记录per-batch loss → Round 2按p_i=exp(l_i)/Σexp采样50%数据继续训练 → PanGu-π-1B Pro
    Step 5: OpenCompass十项benchmark评估 → Avg=51.28（vs baseline 42.41, +8.87）
  - 系统框架：PyTorch LLaMA-like架构，Huawei Ascend 910集群训练
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明

  各组件增量贡献（Figure 1，Avg性能提升）：
  PanGu-π-1B (42.41) → +Compact Tokenizer (44.11) → +Architecture Tweak (46.53) → +Parameter Inheritance (49.79) → +Multi-round Training (51.28)
  其中Parameter Inheritance贡献最大（+3.26），Multi-round Training次之（+1.49），两者合计贡献整个流程一半以上的提升。
