## SparseGPT 与 STR（LLM/CNN 权重剪枝方法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SparseGPT（Frantar & Alistarh, 2023）是面向生成式 LLM 的一次性（one-shot）权重剪枝方法：把大规模权重剪枝建模为逐层稀疏回归问题，用近似 Hessian 逆（基于层的输入激活二阶信息）在剪掉权重的同时最小化重建误差，无需重新训练即可把 100B+ 参数模型剪到 50% 稀疏度且保持困惑度；默认支持 2:4/4:8 等半结构化与任意非结构化稀疏。STR（Soft Threshold Reparameterization，Kusupati et al.）是 CNN 的结构化幅度剪枝方法：把阈值作为可学习参数，通过软阈值函数 S(x)=sign(x)·max(|x|−t,0) 对权重做可微重参数化，与网络联合端到端训练，训练后按阈值 t 得到真正的稀疏权重。Harmonia 用它们生成评估负载：对生成式 LLM（LLaMA-7B、OPT-1.3B，序列长 1024）应用 SparseGPT 得到整体密度 0.2/0.4/0.6 的权重；对视觉模型用 STR 把 ResNet-50 剪到平均权重密度 0.1/0.2、用幅度剪枝（magnitude-based pruning）把 VGG-16 剪到 0.1/0.32——这些剪枝权重正是 Harmonia 验证"端到端稀疏推理负载"的输入（attention/MLP 投影呈现严重的 token 级稀疏偏斜与动态变化）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SparseGPT 逐层剪枝（对每层）：采样该层输入激活 X，算海森矩阵 H=XᵀX，对每行权重做"稀疏回归"——用 Cholesky 分解近似 H 的逆，迭代：选当前最大幅值权重保留、其余剪掉，用 d = H⁻¹·(w−w_pruned) 调整剩余权重补偿误差（OBS 式更新）；伪代码骨架：
```
for layer in model.layers:
    H = X^T X + lambda I            # X 为该层校准激活
    L = cholesky(H)                 # 近似 H^-1
    for row in W_layer.rows:
        mask = keep_topk_by_magnitude(row, k)
        row[mask==0] = 0
        delta = solve(L, row)       # 用 H^-1 补偿剩余权重
        row -= mask * delta
```
STR 训练式剪枝（每轮）：权重 W 过软阈值 S(W)=sign(W)·max(|W|−t,0)（t 可学习、随训练更新），前向用阈值后的 W_hat=S(W)，反向经 STE 直通回传梯度给 W 与 t；训练结束按 t 生成最终稀疏权重。两者都输出"保持准确率的稀疏权重矩阵"，后续推理用稀疏 kernel（如 Harmonia 的 SpMSpM 数据流）执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：SparseGPT 官方实现开源（github.com/IST-DASLab/sparsegpt，PyTorch，支持 OPT/LLaMA 等，`python opt.py facebook/opt-1.3b c4 --sparsity 0.5` 一类命令行）；STR 有官方与社区实现（常用于 ResNet/ImageNet 稀疏训练）。Harmonia 中的用法：把剪枝后的权重矩阵作为 SpMSpM/稀疏 GEMM 的输入，与 SuiteSparse 矩阵一起构成 16 个评估 workload 的 DNN 子集，验证分层调度在真实剪枝网络（LLaMA-0.2/0.4/0.6、OPT-0.2/0.4/0.6、ResNet-0.1/0.2、VGG-0.1/0.32）上的端到端加速（平均 1.87×）与鲁棒性。注意：Harmonia 论文只用其生成稀疏权重，未修改剪枝算法本身。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

HiT 补充视角（ISCA'26，用幅值剪枝生成稀疏 LLM 评估负载）：HiT 对 Llama2-7B 的三个投影层（1024×11008、1024×4096、1024×11008，序列长 1024）的权重矩阵应用 magnitude-based pruning（幅值剪枝——按 |W| 从小到大置零到目标稀疏度），剪枝水平取 0.2/0.4/0.6，与近期 GPT 稀疏化研究（SparseGPT [49]、Wanda 类 [50]）一致；激活保持稠密（密度 1），故这些 workload 属于 MS×D（中稀疏权重 × 稠密激活）矩阵乘。它证明了幅值剪枝作为"生成可复现稀疏 DNN 评估负载"的简单手段的价值：无需重训练、无 Hessian/校准开销，直接以目标密度裁剪即可让硬件评估覆盖中稀疏度段——与 Harmonia 用 SparseGPT/STR 生成 LLM/CNN 稀疏权重是同一工作流，区别仅是剪枝方法更朴素、权重稀疏度更高（0.2-0.6 且投影层尺度 4096/11008）。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
