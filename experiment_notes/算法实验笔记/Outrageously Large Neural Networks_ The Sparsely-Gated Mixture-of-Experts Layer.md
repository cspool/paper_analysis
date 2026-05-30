## Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 Sparsely-Gated Mixture-of-Experts (MoE) Layer，核心组件包括：
    - **Noisy Top-K Gating**：对每个输入 x，计算 H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)，然后 KeepTopK 保留最大的 k 个值（其余设为 -∞），再经 Softmax 得到稀疏门控向量 G(x)。k 个专家中未被选中的专家无需计算 E_i(x)，实现条件计算。
    - **Hierarchical MoE**：两级层次结构。主门控网络 G_primary 选择稀疏加权组合的"次级专家"，每个次级专家本身包含一个次级 MoE（含次级门控网络）。输出 y_H = Σ_i Σ_j G_primary(x)_i · G_i(x)_j · E_{i,j}(x)。第一级分支因子通常等于 GPU 数量，第二级在每个设备内部选择。
    - **Load Balancing**：两个辅助损失——L_importance (CV of expert importance squared) 和 L_load (CV of smooth load estimator squared)，防止门控网络收敛到少数几个专家。
    - **混合数据并行与模型并行**：标准层和门控网络用数据并行，每个 expert 只保留一份共享副本（模型并行），同一组设备同时充当数据并行副本和模型并行分片。专家接收来自所有数据并行输入批次中相关样本的组合批次，batch size 放大 d 倍。
    - **卷积式应用 MoE**：在 stacked LSTM 之间插入 MoE 层，等待前一层对所有时间步完成后再将 MoE 卷积式应用于所有时间步，将时间步维度折叠进 batch 维度，进一步增大 expert 的 batch size。
  - 实验比较：
    - 1 Billion Word LM Benchmark（Chelba et al., 2013）：MoE-4/32/256/256-h/1024-h/4096-h（计算预算约 8M ops/timestep，k=4）vs 计算匹配的 baseline（LSTM-2048-512, 4xLSTM-512, MoE-1-Wide, MoE-1-Deep, MoE-4），以及高计算预算 MoE-34M/143M vs Jozefowicz et al. (2016) 的 best published result。
    - 100 Billion Word Google News Corpus：MoE-32/256-h/1024-h/4096-h/16384-h/65536-h/131072-h（最大 137B 参数）vs 4xLSTM-512 baseline 和 Kneser-Ney 5-gram。
    - WMT'14 En→Fr 和 En→De 机器翻译：MoE-2048（encoder 和 decoder 各一个 MoE 层）vs GNMT (Wu et al., 2016)、PBMT、DeepAtt、LSTM-6-layer。
    - Google Production En→Fr：MoE-2048 vs GNMT。
    - 多语言机器翻译（12 语言对）：MoE-Multi（8.7B 参数）vs GNMT-Mono（278M/model）和 GNMT-Multi（278M）。

- 硬件平台是什么，配置是什么。
  - NVIDIA Tesla K40 GPU 集群（单 GPU 理论峰值 4.29 TFLOPS）
  - 1 Billion Word 实验：16 K40 GPUs；高计算预算模型：32 K40 GPUs
  - 100 Billion Word 实验：32 K40 GPUs（最大两个模型用 64 和 128 K40 GPUs）
  - 机器翻译实验：最多 64 K40 GPUs

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：
    - 语言模型：Word Embedding (512) → LSTM (512) → MoE → LSTM (512) → Softmax。每层后加 dropout 和残差连接。MoE 层内每个 expert 为单隐藏层（1024 ReLU）全连接网络，输入/输出 512 维，约 1M 参数/expert。高计算预算模型用 1024/2048 维 embedding、8192 维 expert hidden layer。
    - 机器翻译模型：基于 GNMT (Wu et al., 2016) 修改版，encoder 3 层 LSTM + 1 层 MoE，decoder 2 层 LSTM + 1 层 MoE。LSTM hidden 2048、output projection 512。各 expert 单隐藏层 2048 ReLU，约 2M 参数/expert。使用 wordpieces（32K 词汇表）、beam search。
  - 数据集：
    - 1 Billion Word Language Modeling Benchmark (Chelba et al., 2013)：~829M words，词汇表 793,471 词
    - 100 Billion Word Google News Corpus：~100B words，来自 Google 内部新闻语料
    - WMT'14 En→Fr：36M 句对训练集，newstest2014 测试集
    - WMT'14 En→De：5M 句对训练集，newstest2014 测试集
    - Google Production En→Fr：Google 内部生产数据
    - 多语言翻译：12 语言对组合数据集（Johnson et al., 2016），约 3B 句对
  - Benchmark 指标：Perplexity（语言建模），BLEU score（机器翻译，multi-bleu.pl）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：原始 TensorFlow 实现是 Google Brain 内部代码，**未作为独立仓库开源**。论文发表于 ICLR 2017。后续 Google 将 MoE 概念集成到 tensor2tensor 库中。社区有多项 PyTorch 复现（如 davidmrau/mixture-of-experts，lucidrains/mixture-of-experts）。
  - 算法 Pipeline 伪代码：
    ```
    # 输入: x, shape [batch_size, seq_len, d_model]
    # 参数: W_g [d_model, n_experts], W_noise [d_model, n_experts]
    # 超参数: k (每个输入选择的专家数)
    
    def MoE_Layer(x):
        # 1. 计算门控信号 H(x)
        clean_logits = x @ W_g                    # [B*T, n]
        noise_std = Softplus(x @ W_noise)         # [B*T, n]
        noise = StandardNormal(clean_logits.shape) * noise_std
        H = clean_logits + noise                  # [B*T, n]
        
        # 2. Noisy Top-K 稀疏化
        top_k_vals, top_k_indices = KeepTopK(H, k)  # [B*T, k]
        # KeepTopK: 保留最大的 k 个值，其余设为 -inf
        G = Softmax(top_k_vals)                    # [B*T, k]，稀疏 gate 权重
        
        # 3. 条件执行 experts
        MoE_out = zeros([B*T, d_model])
        # 将输入按选中的 expert 分组（组 batch）
        for i in range(n_experts):
            mask_i = (top_k_indices contains i)    # 哪些输入选中了 expert i
            if mask_i.any():
                expert_inputs = x[mask_i]           # expert i 的子 batch
                expert_outputs = Expert_i(expert_inputs)  # W_in @ x -> ReLU -> W_out @ x
                gates_for_i = G[mask_i, i]          # 对应 gate 权重
                MoE_out[mask_i] += gates_for_i[:, None] * expert_outputs
        
        return MoE_out
    
    # Hierarchical MoE:
    # G_primary: [B*T] -> top-k1 选择 a 个次级 group
    # G_secondary_i: 仅在 G_primary(x)_i > 0 的样本上计算 -> top-k2
    # output = Σ_i Σ_j G_primary(x)_i * G_i(x)_j * E_{i,j}(x)
    
    # Load Balancing Losses:
    # Importance(X) = Σ_{x∈X} G(x)                    # [n], per-expert gate sum
    # L_importance = w * CV(Importance(X))^2           # CV = σ/μ
    # P(x,i) = Φ((clean_logits_i - kth_excluding(H,k,i)) / noise_std_i)
    # Load(X)_i = Σ_{x∈X} P(x,i)
    # L_load = w * CV(Load(X))^2
    # Total Loss = CrossEntropy + L_importance + L_load
    ```
