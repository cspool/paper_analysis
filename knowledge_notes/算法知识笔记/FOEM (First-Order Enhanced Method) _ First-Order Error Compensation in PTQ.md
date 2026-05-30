## FOEM (First-Order Enhanced Method) / First-Order Error Compensation in PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FOEM 是 Zheng et al. (AAAI 2026) 提出的改进 GPTQ 的 PTQ 方法。核心创新：在权重量化的逐列误差补偿中**显式保留并近似一阶梯度项**。传统 GPTQ 沿袭 OBD/OBS 假设——全精度模型已收敛到局部最优，因此一阶梯度 g ≈ 0，仅用二阶 Hessian 项建模量化误差并补偿。FOEM 指出：逐列量化过程中，先量化列的补偿项持续更新后续 latent weights，导致 W 偏离原始 full-precision 权重 𝕎，产生不可忽略的一阶梯度。FOEM 通过 Taylor 展开近似 g(W) ≈ β(W − 𝕎)H（β=0.1 为稳定化因子），代入 Lagrangian 约束优化求解后，H 和 H^{−1} 在代数中自动消去，最终仅需 Cholesky 因子 T 和权重差分运算，无需显式 Hessian 计算或反向传播。开销极小（Llama3-8B 量化时间仅从 GPTQ 的 825.50s 增至 828.90s，+0.4%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FOEM 逐列量化流程（对比 GPTQ 仅二阶补偿）：

```
# === FOEM: 一阶增强的逐层量化 ===
# 输入: FP权重 W (m×n), 校准输入 X, block size B, β=0.1
H = X @ X.T                         # Hessian: 2XX^T（对称矩阵，n×n）
L = Inverse_Cholesky(H + λI)        # Cholesky 分解, H^{-1} = L L^T
T = L.T                              # 上三角矩阵 T = L^T
W_orig = W.clone()                   # 保存原始全精度权重 𝕎
Q = zeros(m, n)                      # 量化后权重
E = zeros(m, B)                      # block 误差矩阵

for i in range(0, n, B):            # 按 block 迭代
    for j in range(i, i+B):          # block 内逐列量化
        Q[:, j] = quant(W[:, j])     # RTN 量化当前列
        # === 一阶增强误差（GPTQ 仅 (w_q-ŵ_q)/T_{jj}）===
        E[:, j-i] = ((W[:, j] - Q[:, j])
                     - β * (W[:, j] - W_orig[:, j])) / T[j, j]
        # 补偿 block 内后续列（含一阶修正项）
        W[:, j:(i+B)] -= E[:, j-i].unsqueeze(1) * T[j, j:(i+B)]
        W[:, j:(i+B)] -= β * (W[:, j] - W_orig[:, j])
    # 补偿 block 外后续列 (lazy batch update)
    W[:, (i+B):] -= E @ T[i:(i+B), (i+B):]
```

**数学推导链**：
1. 保留一阶项：δE = g δw^T + ½ δw H δw^T
2. 带约束 Lagrangian：ℒ = g δw^T + ½ δw H δw^T + λ(e_q δw^T + w_q − ŵ_q)
3. 求导得最优：δw = −(w_q − ŵ_q − g H^{-1} e_q^T) / [H^{-1}]_{qq} · [H^{-1}]_{q,:} − g H^{-1}
4. 梯度近似：g ≈ β(W − 𝕎)H（g(𝕎)≈0 → Taylor 展开 g(W) ≈ g(𝕎) + (W−𝕎)H）
5. 代入消去 H/H^{-1}：δw = −((w_q − ŵ_q) − β(w_q − 𝕎 e_q^T)) / T_{qq} · T_{q,q:} − β(W − 𝕎)

与 GPTQ 的核心差异：分子中多减 β(w_q − 𝕎 e_q^T)，全局多减 β(W − 𝕎)。β=0 退化为 GPTQ。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/Xingyu-Zheng/FOEM，基于 gptqmodel 库集成。使用方式：
```python
from gptqmodel import GPTQModel, QuantizeConfig, FOEMConfig
foem_config = FOEMConfig(alpha=0, beta=0.1, device="cuda")
quant_config = QuantizeConfig(bits=4, group_size=128, foem_config=foem_config)
model = GPTQModel.load("meta-llama/Llama-3-8B", quant_config)
model.quantize(calibration_dataset)
```
β≤0.5 持续有效；β>0.5 因近似误差放大导致性能退化。FOEM 可与 SpinQuant 旋转矩阵无缝结合，在 W4A4KV4 下进一步缩小与 FP16 的差距。跨架构有效：在 SSM 模型 Mamba-1.4B 上 W3A16 PPL 从 GPTAQ 14.10 降至 FOEM 13.91。校准数据：C4 128 samples, seq_len=2048。评估：WikiText2/C4 PPL + 7 zero-shot benchmarks + 5-shot MMLU。

涉及论文标题：
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

---
