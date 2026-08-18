## LoRA（Low-Rank Adaptation，低秩适配子图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA（低秩适配）是参数高效微调方法：冻结原权重 W，训练低秩增量 ΔW = BA（B、A 为低秩分解矩阵），推理时输出 h = xW + xBA。QiMeng-Tensify（ISCA'26）把 LoRA 作为图级 benchmark 子图（Arch. 列标注 LLaMA3-lora）：LoRA 层含 3 个矩阵乘（xW、xB、A，其中 xB@A 沿 low-rank 维串行），计算密集 + 数据依赖链复杂，是"传统 autoscheduler 固定策略失效"的典型多算子图（论文 Background B 节明确举例"3 matrix multiplications in LoRA"）。端到端评估里 GPT-3-7B-LoRA 也是四个网络级 benchmark 之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LoRA 子图计算过程：
```
h = X @ W0              # 冻结权重投影 (B,S,H) @ (H,H)
h = h + (X @ A) @ B     # 低秩增量：A:(H,r) 下投影、B:(r,H) 上投影，r << H
# 或融合写法：h = X @ W0 + X @ A @ B，可整体看作 (B,S,H) 的线性映射
```
优化机会：把 X@A 与 (X@A)@B 的中间激活（B,S,r）融合/消除、与 X@W0 共享 X 的加载与并行 tile；QiMeng-Tensify 在该子图上编译时间 1.92h（A100，Table VIII）、FP16 子图平均加速比 PyTorch 达 6.49× 量级（含 LoRA 等不规则子图贡献，论文称 LoRA/GatedMLP 上最高 2.3× over PyTorch）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 benchmark 子图（Table VII），与 GQA/GatedMLP 等一起构成 9 子图集合；FP32/FP16 双精度对比 8 个 baseline。使用方式：代表"不规则数据流 + 多 GEMM 依赖链"（相对单算子），验证 QiMeng-Tensify 从算子级推广到子图级的一般化能力；GPT-3-7B-LoRA 用于端到端网络级评估（A100/H100、batch 1/8、seq 4096，vs PyTorch/TensorRT-LLM/Mirage）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
