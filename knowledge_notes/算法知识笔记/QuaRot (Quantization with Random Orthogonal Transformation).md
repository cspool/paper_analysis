## QuaRot (Quantization with Random Orthogonal Transformation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QuaRot 是基于 Hadamard 随机正交变换的 LLM PTQ 方法（Ashkboos et al., NeurIPS 2024）。核心思想：LLM 权重含 outliers，QuaRot 对权重施加随机 Hadamard 变换 W̃ = R₁WR₂^T 将 outliers 扩散到所有元素使分布平坦。由于 R 正交（RR^T=I），乘法不变（WX = R₁^T W̃R₂X），可无精度损失融合。实现 W4A4KV4 量化，是 OBR 的 backbone 旋转方法之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// QuaRot 旋转 + 在线融合
// 每个 Linear layer:
W_rot = R₁ × W × R₂^T
X_rot = R₂ × X                      // 在线 Hadamard 变换
Y_rot = W_rot × X_rot = R₁ × (WX)   // 等价性保证
// 下一层需吸收 R₁^T（或在线逆旋转）
// 量化：W_rot → INT4, X_rot → INT4

// Q/K/V 投影共享 R₂-QK, R₁-Q, R₁-K, R₁-V
// KV Cache 也量化到 INT4
```

FHT (Fast Hadamard Transform): O(d log d) 复杂度旋转。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/spcl/QuaRot (PyTorch + fast-hadamard-transform https://github.com/Dao-AILab/fast-hadamard-transform)。`--rotate` 标志启用。OBR 复用 QuaRot 旋转矩阵不重新训练，加入剪枝和误差补偿达成 W4A4KV4+50% sparsity。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
