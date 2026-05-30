## Offline Value Calibration (OVC) for Value Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

OVC 是 ReCalKV 针对 Value 投影矩阵低秩压缩的后校准策略。标准 SVD 低秩分解 W_v ≈ L_v·R_v 不保证最小化在激活分布 X 上的重建误差 E = ||L_v R_v X - W_v X||_F^2。OVC 通过闭式解分别校准 L_v 和 R_v 来直接最小化该误差。Fisher Information 分析显示 Value 投影的 Fisher 显著高于 Key 投影，校准对保持 Value 精度尤为重要。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 1. Initial SVD: W_v ≈ L_v·R_v
// 2. Calibrate L_v: dE/dL_v=0 →
L_v = W_v·X·X^T·R_v^T·(R_v·X·X^T·R_v^T)^{-1}
// 3. Calibrate R_v: dE/dR_v=0 →
R_v = (L_v^T·L_v)^{-1}·L_v^T·W_v
// 4. Matrix Fusion: W_o_fused = R_v·W_o
// 推理: output = Attention(Q, K, X@L_v) @ W_o_fused
```

LLaMA-2-7B, 80% 压缩率：OVC alone 将 WikiText2 PPL 从 9.34 降至 8.91，LongBench 从 9.01% 升至 13.09%。

术语一般如何实现？如何使用？

256 个 WikiText-2 样本做标定数据，PyTorch 矩阵运算 + `torch.linalg.inv()`，纯代数闭式解（无训练/无梯度）。R_v 通过 Matrix Fusion 与 W_o 合并，推理时零额外开销。r 较小（~64-256），每层校准仅需数秒。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

---
