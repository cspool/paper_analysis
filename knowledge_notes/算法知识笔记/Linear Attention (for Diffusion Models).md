## Linear Attention (for Diffusion Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Attention是通过解耦softmax将标准注意力从O(N²d)降至O(Nd²)的方法族。引入feature map φ(·)（激活函数），用φ(Q)φ(K)^T替代softmax(QK^T)，利用矩阵结合律重排：先算φ(K)^T V ∈ R^{d×d}，再乘φ(Q)得O ∈ R^{N×d}。关键技术公式：

$$H = \phi(K)^T V,\; Z = \operatorname{rowsum}(\phi(K)^T),\; O = \frac{\phi(Q)H}{\phi(Q)Z}$$

线性注意力的表达能力上限为rank d（映射维度），而full softmax注意力的stable rank可远大于d——这是线性注意力在许多场景失效的根本原因（Fan et al., 2025）。SLA实证：在Wan2.1视频生成中，Linear Only的VA=0.042（vs Full Attention 76.78），完全塌陷。但SLA的关键洞察是：去除top 8%大值后的注意力矩阵stable rank骤降至~20，远小于d，因此线性注意力可准确近似这92%的低秩部分。

从算法pipeline角度拆解术语：
```
Standard Attention:  S = QK^T/√d → P = softmax(S) → O = PV  // O(N²d)
Linear Attention:   H = φ(K)^T V | Z = rowsum(φ(K)^T)          // O(Nd²)
                    O = φ(Q)H / φ(Q)Z                           // no N×N matrices
```

SLA在marginal块（~85%）上使用线性注意力：预计算h_j = φ(K_j)^T V_j（d×d矩阵）后，每个marginal块仅需单次H_i += h_j加法，cost <0.5% full attention。φ函数的消融：softmax > elu+1 > hedgehog（表2）。

术语一般如何实现？如何使用？
主要φ选择：ELU(x)+1（Performer, Choromanski 2020）、ReLU(x)、softmax(x)（SLA推荐）。代表性线性注意力模型：Performer、Linear Transformer (Katharopoulos 2020)、cosFormer (Qin 2022)、Lightning Attention-2 (Qin 2024)、RetNet (Sun 2023)、Mamba2（SSM形式，数学等价线性注意力）。SLA中线性注意力不作为full attention的直接替代，而是通过Proj(O^l)投影和fine-tuning作为稀疏注意力的learnable compensation。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
