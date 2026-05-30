## Associative_Recurrent_Memory_Transformer

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是ARMT（Associative Recurrent Memory Transformer），在RMT（Recurrent Memory Transformer）的segment-level recurrence基础上，每层添加基于quasi-linear key-value memory（delta-rule, 源自Schlag et al. 2021）的层间关联记忆（associative memory）。每个segment的memory token通过线性映射生成key/value，经DPFP-3非线性变换后存入关联矩阵A_s^l，同时用γ-correction更新归一化向量z_s^l防止灾难性遗忘。实验对比：(1) BABILong benchmark上QA1-QA5任务（最高50M tokens），对比Mamba (130M)、RMT (137M)、RMT-R、GPT-4 (few-shot)、GPT-4+RAG；(2) Associative Retrieval Remember和Rewrite任务（最多200 key-value pairs），对比Mamba和RMT，评估记忆容量和动态改写能力；(3) Language Modeling (Wikitext-103)对比RMT；(4) 消融实验包括γ-correction有无和PRMT（Parallel Memory RMT，仅层间记忆无关联矩阵）对比。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体GPU型号和配置。致谢中提到"SberDevices for granting us access to additional computational resources"。基于训练设置（GPT-2 137M backbone、最大16k tokens训练）推测使用商用NVIDIA GPU集群（如A100或类似）。训练精度论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) BABILong实验：GPT-2 (137M) + ARMT/RMT扩展，segment size=512，总参数145M（ARMT）、137M（RMT）；对比Mamba-130M。训练最大长度16k tokens（32 segments × 512）。(2) Associative Retrieval实验：小模型约500k参数，4层，hidden dim=128，memory dim=32。训练最大200 pairs (Remember) / 50 pairs (Rewrite)。(3) LM实验：GPT-2 + ARMT/RMT，segment size=128，训练8 segments（1024 tokens）。
  数据集：(1) BABILong benchmark — 合成QA任务，从单事实(QA1)到三参数关系(QA5)，噪声句子+事实混合，支持生成1k-50M tokens序列；(2) Associative Retrieval — 自建数据集：Remember任务（唯一key-value对，评估记忆容量）、Rewrite任务（非唯一key，评估动态更新能力）；(3) Wikitext-103（语言建模评估）。(4) 训练数据使用curriculum learning：BABILong从2 segments递增至32 segments（16k tokens），Associative Retrieval从1对递增至200对。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/RodkinIvan/associative-recurrent-memory-transformer（Apache-2.0 license），基于PyTorch + Hugging Face Transformers + Accelerate，包含modeling_amt.py和modeling_rmt.py核心模块。

  ARMT算法pipeline核心——逐层前向传播（输入一个segment）：
  ```
  Input: X_s^l ∈ R^{seg_len×D} (segment s 在层l的hidden states)
         M_s^l ∈ R^{mem_tokens×D} (层l的memory tokens)
         A_{s-1}^l ∈ R^{D×D}, z_{s-1}^l ∈ R^D (上一层segment的关联记忆和归一化向量)
  
  For each layer l:
    // 1. 拼接输入和memory tokens
    input_concat = concat([X_s^l; M_s^l])  // ∈ R^{(seg_len+mem_tokens)×D}
  
    // 2. 从关联记忆中读取（类似线性注意力）
    for each token x_j in input_concat:
      q_j = W_Q x_j                    // ∈ R^D
      y_j = A_s^l φ(q_j) / (z_s^l)^T φ(q_j)  // 关联回忆, φ=DPFP-3非线性
    // y_j加到token表示中
  
    // 3. Transformer block处理（包含self-attention + FFN）
    [X_s^{l+1}; M_s^{l+1}] = TransformerBlock(input_concat + y)
    // self-attention仅在当前segment内（local context），不需要attend历史token
  
    // 4. 更新关联记忆（用新产生的memory tokens）
    for each memory token m_i ∈ M_s^{l+1}:
      k_i = W_K m_i                   // key投影
      v_i = W_V m_i                   // value投影
      β_i = σ(W_β m_i)                // importance scalar（sigmoid门控）
      
      // γ-correction: 从归一化向量中移除旧key的贡献
      v̄_i = A_{s-1}^l φ(k_i) / (z_{s-1}^l)^T φ(k_i)  // 回忆旧value
      γ_i = 1 - (z_{s-1}^l)^T φ(k_i) / ||φ(k_i)||²     // 归一化修正系数
      
      // Delta-rule更新关联矩阵
      A_s^l = A_{s-1}^l + Σ_i β_i (v_i - v̄_i) ⊗ φ(k_i)  // 外积更新, D×D
      z_s^l = z_{s-1}^l + Σ_i γ_i φ(k_i)                // 归一化向量更新
  
  Output: X_s^{l+1} (用于下一层或输出), M_s^{l+1} (传给下一segment同层), A_s^l, z_s^l (传给下一segment同层)
  ```

  复杂度分析（per segment, per layer）：
  - Local self-attention: O(seg_len² × D) — 仅segment内，与总序列长度无关
  - Associative memory read: O((seg_len+mem_tokens) × D²) — D×D矩阵-向量乘
  - Associative memory update: O(mem_tokens × D²) — 外积更新
  - 总空间: O(D²) per layer 存储固定大小A矩阵，与序列长度无关
  - 每segment处理时间和空间均为常量O(1)，与总序列长度无关

  训练配置：
  - Curriculum learning: 从短序列开始逐步增加segment数量
  - BABILong: 训练最大32 segments × 512 tokens = 16k tokens
  - Associative Retrieval: 训练最大200 pairs (Remember) / 50 pairs (Rewrite)
  - LM: 训练8 segments × 128 tokens = 1024 tokens
  - γ-correction的γ在训练时detach以改善收敛

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Artificial Hippocampus Networks (AHNs) —— 一种轻量级架构组件，在标准Transformer的每层attention旁添加RNN-like压缩模块。具体而言：模型保留滑动窗口attention（默认窗口W=32k）作为lossless short-term memory；当序列长度超过W时，AHN对离开窗口的KV pair (k_{t-W}, v_{t-W}) 进行recurrent压缩，更新固定大小的compressed long-term memory h_{t-W} = AHN((k_{t-W}, v_{t-W}), h_{t-W-1})。最终输出为attention输出与AHN压缩记忆输出的加和：y_t = y_AHN,t + Attention({(k_i,v_i)}_{i=t-W+1}^t, q_t)。AHN有三种实例化：AHN-Mamba2（基于Mamba2 SSM）、AHN-DN（基于DeltaNet的delta rule更新）、AHN-GDN（基于GatedDeltaNet的gated delta rule）。训练采用self-distillation：冻结base LLM全部参数，仅训练AHN参数（约0.4%参数量），以KL(p_teacher || p_student)为loss。实验在Qwen2.5-Instruct系列（3B/7B/14B）上对比：Full Attention、Sinks+SWA、Compressive Transformer (CT-Max/CT-Average pooling, 4x压缩率)、AHN-Mamba2/AHN-DN/AHN-GDN。评测包括LV-Eval (128k)、InfiniteBench (128k)、LongBench (6个>8k任务)、PG19 perplexity、RULER NIAH任务。

- 硬件平台是什么，配置是什么。
  训练：32块NVIDIA A100 GPU，训练7B模型约10小时。训练精度论文未明确说明（推测BF16/FP16混合精度）。推理评测：论文未明确说明推理硬件（推测同一A100集群）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-Instruct系列（3B、7B、14B），在其每层attention上添加AHN模块。AHN instantiation参数：每head的W_α∈R^{D×1}、W_β∈R^{D×1}、W_γ∈R^{D×1}（gating参数）和W_o∈R^{H×H}（per-head输出投影），总计约0.4% base model参数量。数据集：ChatQA2（1B tokens，开源长上下文任务集合），训练时最大序列长度24k，随机化滑动窗口大小（从[32,64,...,8192]中采样过滤后候选）。Benchmark：LV-Eval（128k context, 11 datasets）、InfiniteBench（128k context）、LongBench（6个平均长度>8k的任务：DuReader, HotpotQA, MuSiQue, NarrativeQA, QMSum, TriviaQA）、PG19（perplexity评测）、RULER（needle-in-a-haystack评测）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/ByteDance-Seed/AHN，模型：https://huggingface.co/ByteDance-Seed。实现基于PyTorch、LLaMA-Factory、Flash Linear Attention。

  AHN-GDN算法pipeline（per head, per layer）：
  ```
  Input: x_t ∈ R^D (当前token hidden state), QKV from attention projections
         KV cache window: {(k_i, v_i)}_{i=t-W+1}^t
         h_{t-W-1} ∈ R^{H×H} (上一步压缩记忆矩阵)

  // Step 1: 从attention的KV projection获取当前离开窗口的KV pair
  k_{t-W}, v_{t-W} ∈ R^H  // 第t-W位置的key和value, H为head_dim

  // Step 2: AHN-GDN 压缩记忆更新（Gated Delta Rule）
  α(x_{t-W}) = x_{t-W} W_α   // W_α ∈ R^{D×1}, α: scalar per head
  β(x_{t-W}) = x_{t-W} W_β   // W_β ∈ R^{D×1}, β: scalar per head

  h_{t-W} = α(x_{t-W}) * (I - β(x_{t-W}) * k_{t-W}^T k_{t-W}) * h_{t-W-1}
          + β(x_{t-W}) * k_{t-W}^T v_{t-W}
  // 即：记忆衰减 + 新信息写入，h ∈ R^{H×H}

  // Step 3: Query访问压缩记忆
  q_t = slice from attention's Q projection  // q_t ∈ R^H
  γ(x_t) = x_t W_γ   // W_γ ∈ R^{D×1}, gate scalar
  y_AHN,t = γ(x_t) * q_t * h_{t-W} * W_o   // W_o ∈ R^{H×H}, grouped by heads

  // Step 4: 与窗口attention输出求和
  y_t = y_AHN,t + Softmax(q_t {k_i}_{i=t-W+1}^T / √H) {v_i}_{i=t-W+1}

  复杂度（vs Full Attention）:
  - Memory cache: O(W) vs O(L), W=32k固定
  - FLOPs per token: O(W) vs O(L)
  - 当L ≤ W时，AHN不激活，模型等同于标准Transformer
  ```

  训练流程（Self-Distillation）：
  ```
  Teacher: 原始Qwen2.5-Instruct（Full Attention, 参数冻结）
  Student: Qwen2.5-Instruct + AHN（window attention + AHN, 仅AHN参数可训练）

  For each training step:
    x_teacher = x_student = input_sequence (max 24k tokens)
    p_teacher = Teacher(x_teacher)    // full attention forward
    p_student = Student(x_student)    // window attn + AHN forward
    loss = KL(p_teacher || p_student)
    loss.backward()  // 仅AHN参数有梯度

  超参数：
    - Optimizer: AdamW, LR=1e-4, linear warmup 10% steps + cosine decay
    - Batch size: 128 (global), 740 update steps (1 epoch over 1B tokens)
    - 训练窗口随机化：attention sink size ∈ [0,32,64,128,512,2048,4096]
      总lossless memory (sinks + window) ∈ [32,64,...,8192]
    - 推理默认：128 attention sinks + 32640 sliding window = 32768 lossless memory
  ```
