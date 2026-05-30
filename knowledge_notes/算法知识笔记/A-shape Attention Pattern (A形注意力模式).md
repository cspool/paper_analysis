## A-shape Attention Pattern (A形注意力模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

A-shape 注意力模式是 MInference 论文识别的三种长上下文注意力稀疏模式之一。其名称来源于注意力矩阵的视觉形状：attention weights 集中在**初始 token（attention sink / global tokens）**和**局部窗口（local window / recent tokens）**，在注意力矩阵热力图上形成类似字母 "A" 的形状——两侧有高注意力值（左侧=初始 token，右侧对角线=局部窗口），中间区域几乎为零。

A-shape 模式的特征：(1) **空间分布**：Static structured——无论输入内容如何变化，重要 token 的位置始终是初始若干 token + 末尾局部窗口；(2) **GPU 延迟**：Low——因为稀疏模式是结构化的、固定的，可以直接使用 FlashAttention 仅计算对应区域；(3) **索引构建时间**：Zero——完全静态，无需在线估计。

A-shape 模式的典型 attention head 负责处理局部语法结构或对初始 token（如 BOS token 或系统 prompt）的持续关注。StreamingLLM 论文（Xiao et al., 2024）首次系统性地识别了这一模式并用于 decoding 阶段的 KV cache 压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

A-shape 模式的计算极其简单——使用固定的因果掩码变体：

```
# A-shape 稀疏掩码定义
S = 131072  # 序列长度
M[i, j] = 1 if (j < GLOBAL) or (j >= i - LOCAL) else 0
# GLOBAL = 1024 (初始 global tokens)
# LOCAL = 4096 (局部 window tokens)

# 稀疏注意力计算
A = softmax(Q @ K^T / √d - c * (1 - M))  # c=1e5, 非M区域强制为0
y = A @ V
# 等效于在 FlashAttention 中仅遍历 global + local 区域
```

**具体执行**（LLaMA-3-8B, 128K context, A-shape head）：
- 仅计算: row 0-1023（global）的所有列 + row 1024-131071 的列 j∈[0,1024)∪[i-4096,i]
- FLOPs: ~1K × 128K + 127K × 5K ≈ $1.9 \times 10^9$（vs dense 的 $2.2 \times 10^{11}$）
- 稀疏率: ~99%

术语一般如何实现？如何使用？

实现方式：在 FlashAttention kernel 中，将 Q 的分块循环限制在 global token block 和 local window block 范围内，跳过中间所有 token blocks。由于模式完全固定，可以在 kernel 编译期就确定 loop range。

使用场景：A-shape 模式适合负责局部语法处理的 attention head，如相邻 token 的依存关系、局部上下文理解。不适合需要全局检索或多跳推理的 head（如 retrieval head）。MInference 论文的搜索结果显示，A-shape 模式主要在模型的中间层出现，占比较少（<<10%）。

主要局限：当关键信息位于 global window 和 local window 之间时（如中间位置的 passkey），A-shape head 完全无法捕获，导致 retrieval 类任务准确率崩溃。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
