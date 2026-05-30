## SwiGLU FFN

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SwiGLU（Swish-Gated Linear Unit）是现代 LLM（LLaMA、LLaMA-2/3、Qwen、Mistral 等）中普遍采用的 gated FFN 变体。计算流程：SwiGLU(X) = (SiLU(X·W_gate^T) ⊙ (X·W_up^T)) · W_down。首先输入 X ∈ R^{L×d_model} 分别通过两个独立投影 W_gate（gate分支）和 W_up（up分支）映射到 d_ff 维（d_ff ≈ 8/3·d_model 是经验最优 ratio）；然后 gate 分支经 SiLU（Sigmoid Linear Unit = x·σ(x)）激活后与 up 分支做 element-wise 乘法 ⊙；最后经 W_down 投影回 d_model。SwiGLU 是 GLU family（Gated Linear Unit）的一种，其激活函数为 Swish/SiLU。相比原始 GELU-gated FFN、ReLU FFN 或标准 non-gated FFN，SwiGLU 在 scaling law 实验中和实际预训练中展现出更好的 perplexity 和收敛速度（Touvron et al., 2023; Shazeer, 2020）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SwiGLU FFN 完整 forward（PyTorch 风格，LLaMA-like）:
# 参数: W_gate, W_up ∈ R^{d_ff × d_model}, W_down ∈ R^{d_model × d_ff}
# 输入: X ∈ R^{L × d_model}

def swiglu_ffn(X, W_gate, W_up, W_down):
    # Step 1: Gate branch — 计算 gating signal
    gate_logits = X @ W_gate.T        # [L, d_model] × [d_model, d_ff] → [L, d_ff]
    gate = F.silu(gate_logits)        # SiLU(x) = x * sigmoid(x), element-wise
    
    # Step 2: Up branch — 计算 value stream
    up = X @ W_up.T                   # [L, d_model] × [d_model, d_ff] → [L, d_ff]
    
    # Step 3: Gating — element-wise multiply
    hidden = gate * up               # [L, d_ff] ⊙ [L, d_ff] → [L, d_ff]
    # 关键: gate ∈ (0, ~d_ff) 范围（SiLU 无上界），
    #       对 up 的每个 channel 做软性 important/unimportant 选择
    
    # Step 4: Output projection
    output = hidden @ W_down.T        # [L, d_ff] × [d_ff, d_model] → [L, d_model]
    return output

# 张量形状示例（LLaMA-7B: d_model=4096, d_ff=11008, L=2048）:
# gate_logits: [2048, 4096] × [4096, 11008] → [2048, 11008]  (约90M elements/bf16)
# hidden: [2048, 11008]  ≈ 45MB in bf16 → 必须写入HBM
# output: [2048, 11008] × [11008, 4096] → [2048, 4096]   (约17MB)
```

与标准 FFN 的关键区别：(1) gate 分支引入 channel-wise multiplicative gating——不同于简单 activation 的逐元素非线性，gating 机制允许 FFN 动态选择信息流中的哪些 channel 特征被保留/抑制；(2) 双投影设计（gate + up）使得参数量略多于标准 FFN（d_ff × d_model × 2 + d_ff × d_model = 3·d_ff·d_model vs 标准 FFN 的 2·d_ff·d_model），但这被证明是 parameter-efficient 的；(3) SiLU 的非单调性（在 x<0 时轻微负激活）相比 ReLU 的 hard-zero 提供更丰富的梯度流。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SwiGLU 在现代 GPU 上的实现主要由 cuBLAS GEMM 执行：三次矩阵乘法（gate/proj + up/proj + down/proj）通过高度优化的 tiled GEMM kernel 完成，分别由独立的 cublasGemmEx 调用。SiLU 激活和 element-wise multiply 在后续的 pointwise kernel/fused elementwise kernel 中完成。优化策略：(1) torch.compile 可将三次 GEMM + SiLU + multiply + down GEMM 融合为单个 fused kernel（部分场景），减少 kernel launch overhead；(2) 利用 NVIDIA cuBLASLt 的 fused epilogue 将 SiLU 激活与 gate GEMM 的 output write 融合；(3) FlashMHF 等工作正在探索更彻底的 I/O-aware fusion——通过 multi-head 设计分解大中间激活，再通过 SRAM-resident blockwise 计算消除 HBM round-trip。当 batch size 较小时（inference 典型 bs=1-8），中间激活 hidden ∈ R^{L×d_ff} 虽然远小于 attention 的 QK^T ∈ R^{L×L}，但仍占显著 HBM 带宽——LLaMA-7B 单层 SwiGLU 的 hidden tensor ≈ L·11008·2 bytes，L=4096 时约 90MB。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators (for GEMM-based Operator Chain)
