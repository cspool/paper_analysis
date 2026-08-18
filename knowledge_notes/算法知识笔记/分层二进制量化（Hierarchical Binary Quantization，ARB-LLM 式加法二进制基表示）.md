## 分层二进制量化（Hierarchical Binary Quantization，ARB-LLM 式加法二进制基表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 分层二进制量化（ARB-LLM [47]，Alternating Refined Binarization）把每个精度区域（bitwidth $b_k$）的权重张量表示为一组二进制基矩阵的加权和：$\mathbf{W}=\sum_{i=1}^{b_k}\alpha_{r,i}\,\alpha_{c,i}\,\mathbf{B}_i$，其中 $\mathbf{B}_i\in\{-1,+1\}^{och\times ich}$ 是第 i 个二进制基（逐元素 ±1），$\alpha_{r,i}$ 为行缩放因子、$\alpha_{c,i}$ 为列缩放因子。b_k 个二进制基逐层叠加逼近原权重（类似 OneBit [38] 的 1-bit 符号矩阵+轻量向量，但可扩展到多 bit 层级）。该表示的核心价值：二进制基的乘法只有符号翻转（±1×x = ±x），且与位串行硬件执行天然兼容——分配的 bitwidth 直接换算成硬件延迟与能耗。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（论文 Eq.6 与 Algorithm 1 的 QUANTIZE 步骤）：对每个量化块，按区域位宽 b_k 迭代：①交替精化求当前残差的符号基 $B_i=\mathrm{sign}(W-\sum_{j<i}\alpha_{r,j}\alpha_{c,j}B_j)$（ARB 的交替优化）；②按行/列求最优缩放因子 $\alpha_{r,i},\alpha_{c,i}$（最小二乘意义）；③累加进重建 $\hat{W}$。推理时张量计算例子（一个 4-bit 区域的线性层）：$Y=X\hat{W}=X\sum_{i=1}^4\alpha_{r,i}\alpha_{c,i}B_i=\sum_{i=1}^4\alpha_{r,i}((XB_i)\odot\alpha_{c,i})$——先做 4 次二进制乘加（$XB_i$ 只需取反/加），再按行/列缩放叠加；每减少 1 bit 就少一次基乘加，位宽与计算量线性挂钩。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：ARB-LLM 已有开源实现（https://github.com/ilacher/ARB-LLM，论文未引用具体链接）；训练后离线对每层逐区域精化（可并行）。在 SingularBit 中的使用：作为 rank 边界内所有区域（4/3/2/1-bit）的统一表示，输出 $\hat{U},\hat{V}^T$ 均以 $\sum_i\alpha_{r,i}\alpha_{c,i}B_i$ 存储；硬件侧 activation loader 只为 4 输入通道预计算 8 个 LUT 条目（0 通道编码符号、1–3 通道编码幅度组合），乘法被多路选择器+取反替代（见硬件架构层"LUT 位串行混合精度 Tensor Core"条目）。注意区分：本术语是"权重表示为多二进制基叠加"的表示法，与 LUT 查表量化（码本式非解析量化）不同，后者是聚类码本。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
