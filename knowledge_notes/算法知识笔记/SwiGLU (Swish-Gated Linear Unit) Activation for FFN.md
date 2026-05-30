## SwiGLU (Swish-Gated Linear Unit) Activation for FFN

术语解释
SwiGLU 是现代大语言模型（LLaMA、PaLM、Mistral、Qwen 等）中最广泛使用的 FFN 激活函数，结合了 Swish (SiLU) 激活和 Gated Linear Unit (GLU) 门控机制。它使用三个权重矩阵（gate_proj、up_proj、down_proj）替代传统 FFN 的两个矩阵，通过可学习的门控实现输入自适应的特征过滤。

术语是什么？
SwiGLU FFN 前向计算：
$$FFN_{SwiGLU}(x) = (SiLU(xW_{gate}) \odot (xW_{up})) W_{down}$$

其中 SiLU(x) = x · σ(x)（即 Swish 激活，σ 为 sigmoid），⊙ 为逐元素乘法（Hadamard product）。三步计算过程：
1. **Gate 投影**：h_gate = xW_gate，经 SiLU 激活 → [B, D_intermediate]
2. **Up 投影**：h_up = xW_up → [B, D_intermediate]
3. **门控融合 + 降维**：(SiLU(h_gate) ⊙ h_up)W_down → [B, d]

三个权重矩阵维度：W_gate ∈ R^{d×D}、W_up ∈ R^{d×D}、W_down ∈ R^{D×d}，其中 d 为 hidden dimension，D 为 intermediate/expansion dimension。

从算法pipeline角度拆解术语：
SwiGLU 在 Transformer FFN 层中的计算流程：

```
=== SwiGLU FFN Forward Pass ===

输入: x [B, d] — attention output hidden states

Step 1 — Gate Projection + SiLU Activation:
    gate = W_gate @ x         # [B, D]
    gate_act = gate * sigmoid(gate)  # SiLU activation, [B, D]
    # SiLU 性质: 光滑、非单调、负值区域有非零梯度

Step 2 — Up Projection (value path):
    up = W_up @ x             # [B, D]

Step 3 — Gated Feature Selection:
    hidden = gate_act * up    # element-wise, [B, D]
    # 关键: gate_act 的每个元素控制 up 对应元素的通过量
    # gate_act ≈ 0 → 该维度被抑制
    # gate_act ≈ up → 该维度保持
    # gate_act < 0 → 该维度被反转（SiLU 负值区域）

Step 4 — Down Projection:
    output = hidden @ W_down  # [B, d]

输出: output [B, d]
```

vs 传统 FFN（ReLU/GELU）：
- ReLU FFN：FFN(x) = ReLU(xW1 + b1)W2 + b2（2 个权重矩阵）
- SwiGLU FFN：3 个权重矩阵，但 D 通常缩放为原来的 2/3 以保持参数量可比
- 关键差异：SwiGLU 的"门控"是数据依赖的（data-dependent），不同输入产生不同的特征过滤模式

在 DSMoE 中的应用：DSMoE 将 SwiGLU FFN 的三个矩阵沿 intermediate dimension D 均等划分为 n 组，每组构成一个 expert，数学上保证了划分后所有 expert 输出之和等价于原始 FFN 输出。

术语一般如何实现？如何使用？
- **主流 LLM 框架**：HuggingFace Transformers 中的 LLaMA 系列默认使用 SwiGLU；PyTorch 中 `F.silu()` 为 SiLU 激活的原生实现
- **Fused Kernel 优化**：部分框架提供 fused SwiGLU kernel，将 gate projection + SiLU + element-wise multiply 融合为单次 kernel launch，减少 HBM 读写
- **GLU 变体对比**：SwiGLU（SiLU gate）、ReGLU（ReLU gate）、GEGLU（GELU gate）中 SwiGLU 在语言建模任务上经验表现最优（Shazeer, 2020, "GLU Variants Improve Transformer"）
- **维度选择惯例**：LLaMA-7B 使用 D=11008（≈ 8/3 × 4096 × 2/3 的取整）

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

---
