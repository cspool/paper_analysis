## Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Attention 将标准 softmax(QK^T)V 的 O(N²) 降至 O(N)（Katharopoulos et al. 2020; Schmidhuber 1992）。核心：用可分离特征映射 φ(Q)φ(K)^TV 替代 softmax，利用结合律先算 φ(K)^TV（固定大小），再与 φ(Q) 乘。等价于 RNN: s_t = s_{t-1} + φ(k_t)^T v_t, o_t = φ(q_t)s_t。每 token O(1) 且内存恒定。Naive linear attention 性能不及 softmax；RWKV/RetNet/GLA/Mamba 通过 decay/门控/数据依赖缩小差距。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RWKV WKV 是 linear attention 的直接改进——learnable per-channel exponential decay 替代等权求和：
```
s_t = diag(w)·s_{t-1} + k_t^T·v_t     # diag(w): channel-wise decay
wkv = diag(u)·k_t^T·v_t + s_{t-1}      # u: boost 当前 token
o_t = r_t @ wkv                         # r: receptance query
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
双模式：(1) RNN 模式（推理）：逐 token 更新 O(1)；(2) 并行模式（训练）：associative scan 沿时间维并行。RWKV 训练用 custom CUDA kernel SRAM-resident state 沿非时间维并行。FLA 库提供多种变体高效实现。

JRT论文使用Taylor 2阶近似特征图 φ(q)^Tφ(k)=1+q^Tk+(q^Tk)²/2（feature dim d̃=273 for d=16 base），在Based架构中结合gated convolution(kernel=3)+sliding window attention(window=128)+linear attention的hybrid layout。PLA将linear attention扩展为encoder-decoder：encoder区域用非因果sum预计算KV-state，decoder区域沿用causal cumsum。JRT-Prompt通过2×prefill重复context，利用linear attention的2N prefill仍快于attention的N prefill。

RWKV 原始论文（RWKV: Reinventing RNNs for the Transformer Era, EMNLP 2023）首次将 linear attention 扩展到 14B 参数规模，验证了线性注意力在大规模 LLM 中的可行性。其 WKV 算子使用通道级可学习指数衰减 w∈(R_{≥0})^d 替代等权特征图求和：`wkv_t = (Σ e^{-(t-1-i)w+k_i}⊙v_i + e^{u+k_t}⊙v_t)/(Σ e^{-(t-1-i)w+k_i} + e^{u+k_t})`，其中分母提供归一化（而非 feature map 的 Σ φ(k)），u 为当前 token bonus。数值稳定版本使用共享指数 p_t 技巧避免 exp 溢出：`q=max(p_{t-1}, u+k_t); wkv_t = (e^{p_{t-1}-q}⊙a'_{t-1}+e^{u+k_t-q}⊙v_t)/(e^{p_{t-1}-q}⊙b'_{t-1}+e^{u+k_t-q})`。RWKV 推理时将 WKV 递归化为 RNN：`a_t = e^{-w}⊙a_{t-1}+e^{k_t}⊙v_t; b_t = e^{-w}⊙b_{t-1}+e^{k_t}; wkv_t = a_t/b_t`，实现 O(d) 空间（仅需存储 a_t,b_t,p_t 三个 d 维向量）和 O(1) 时间 per token。论文证明 RWKV scaling law 与 Transformer 相同（r²=0.994），12 项 NLP benchmark 上 FLOP-matched 性能与 Pythia/OPT/BLOOM 相当。开源：https://github.com/BlinkDL/RWKV-LM，预训练模型：https://huggingface.co/RWKV。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models
- Linearizing_Large_Language_Models

SUPRA 的具体用法：SUPRA 用 MLP kernel φ(x)=ReLU(Wx+b) 作为可学习特征图，queries 和 keys 共享同一 MLP 权重。相似度函数变为 sim(q_i,k_j)=RoPE(φ(q_i))·RoPE(φ(k_j))，加入固定衰减向量 γ∈(0,1)^h。输出用 GroupNorm 而非分母除法归一化：v'_i=GroupNorm(Σ_{j=1}^{i} γ^{i-j}·sim(q_i,k_j)·v_j)。训练时用 Lightning Attention 2 的 Triton kernel 做序列并行，推理时切换为循环模式 O(1) per-token。关键区别：(1) MLP kernel 替代固定 ELU kernel；(2) GroupNorm 替代分母除法，解决大规模训练稳定性；(3) RoPE 提供相对位置编码；(4) 固定 decay 提供位置偏置。验证了线性注意力可通过 uptraining 从强预训练 Transformer 获得，不必从零训练。

---
