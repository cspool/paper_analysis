## Structural Symmetry between FFN and Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Structural Symmetry between FFN and Attention 是 Geva et al. (2020) 提出的 FFN 与单头注意力之间的结构等价性观察。核心等价推导：FFN(X) = φ(X·W_1^T)·W_2 可重新解释为 "X attends over W_1（作为 keys）to retrieve values from W_2（作为 values）"。具体来说：(1) 在 Attention(Q,K,V) = softmax(QK^T/√d_k)·V 中，将 Q 替换为 X，K 替换为 W_1，V 替换为 W_2^T；(2) 将 softmax 替换为 element-wise 非线性 φ(·)；(3) 两者形式上完全一致。因此 FFN 可被理解为 "对长度为 d_ff 的参数序列的注意力"——X query 通过 key W_1 访问 value W_2 中的存储知识（Geva et al. 的 "key-value memory" 解释）。对于 gated SwiGLU 变体，SwiGLU(X) = (SiLU(X·W_gate) ⊙ (X·W_up))·W_down，可定义 φ_s(Q,K) = SiLU(Q·K^{(g)T}) ⊙ Q·K^{(u)T}，同样重写为 φ_s(Q,K)·V，保持注意力类的结构。FlashMHF 论文基于此对称性提出 multi-head FFN 概念——正如 multi-head attention 从不同子空间并行学习，multi-head FFN 也应当从多个独立的参数子空间并行处理，增强表达力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# 单头Attention 与 FFN 的结构对称性证明（FlashMHF 论文公式1-4）：

# 1. 标准单头 Attention:
def single_head_attention(Q, K, V):
    # Q, K, V ∈ R^{L × d_k}
    scores = Q @ K.T / sqrt(d_k)     # [L×d_k] × [d_k×L] → [L×L]
    weights = softmax(scores, dim=-1)  # row-wise softmax: 每行之间归一化
    output = weights @ V              # [L×L] × [L×d_k] → [L×d_k]
    return output

# 2. 标准 FFN (vanilla):
def ffn(X, W1, W2):
    # X ∈ R^{L × d_model}, W1 ∈ R^{d_ff × d_model}, W2 ∈ R^{d_model × d_ff}
    hidden = activation(X @ W1.T)    # [L×d_model] × [d_model×d_ff] → [L×d_ff]
    output = hidden @ W2.T           # [L×d_ff] × [d_ff×d_model] → [L×d_model]
    return output

# 3. 对称性证明——用 Attention "模板" 表达 FFN:
#    令 Q = X, K = W1, V = W2^T, softmax → element-wise φ
#    → FFN = "X attends over W1 parameters to retrieve from W2 values"

# 4. Gated SwiGLU 的注意形式重写（FlashMHF 公式3-4）:
def swiglu_as_attention(X, K_gate, K_up, V):
    # K_gate, K_up ∈ R^{d_ff × d_model}, K = [K_gate, K_up]
    # V ∈ R^{d_ff × d_model} (即 W_down)
    Q = X                           # query = input
    gate = SiLU(Q @ K_gate.T)       # [L×d_model] × [d_model×d_ff] → [L×d_ff]
    up   = Q @ K_up.T               # [L×d_model] × [d_model×d_ff] → [L×d_ff]
    output = (gate ⊙ up) @ V.T      # element-wise gate × up, then project
    return output
    # 定义 φ_s(Q,K) = SiLU(Q·K^{(g)T}) ⊙ Q·K^{(u)T}
    # 则 output = φ_s(Q,K)·V，证明 SwiGLU 是广义注意力的一个实例

# 5. Multi-Head 推广（FlashMHF 核心思路）:
#    标准 attention 有 multi-head → FFN 也应有 multi-head
#    Q = split_H(X · W_in) ∈ R^{L×H×d_h}
#    每 head h 独立执行 FFÑ(Q[:,h,:]; K^h, U^h, V^h)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该对称性在 Transformer 研究中有多个系统的应用：(1) Geva et al. (2020) 利用此对称性分析 FFN 的知识存储能力——将 W_2 的每一行解释为一个 "learned pattern"，FFN 通过 softmax-like 选择检索最相关的 pattern；(2) Tokenformer (Wang et al., 2024) 将所有线性投影替换为 Token-Parameter Attention（PAttention），彻底 operationalize 此对称性；(3) FlashMHF 通过此对称性设计 multi-head FFN——正如 MHA 的 H 个头从 H 个不同子空间处理 Q 以丰富表示，MH-FFN 的 H 个头也从 H 个不同子空间处理 X 以增强表达力；(4) MLP-Mixer 和 DaViT 利用此对称性在 token-mixing（attention-like over tokens）和 feature-mixing（FFN-like over features/channels）之间建立对称操作。实际使用中，此对称性是 rethinking FFN architecture 从 "通用近似函数" 到 "结构化参数注意力" 的概念转变基石。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)
- Tokenformer (Wang et al., 2024)
