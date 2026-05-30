## Binary Residual Approximation（二进制残差逼近）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Binary Residual Approximation（二进制残差逼近）是 BiLLM（ICML 2024）提出的一种针对少数高显著度（salient）权重的低比特逼近方法。其核心思想是：不将 salient 权重保留为高精度（INT8/FP16）也不简单二值化，而是通过两阶段递归二值化来逼近原始权重——先用初始二值矩阵 B_o 逼近原始权重矩阵 W，再对残差 (W - α_o*·B_o*) 进行第二次二值化得到 B_r，最终用两个二值矩阵的加权和 α_o·B_o + α_r·B_r 表达原 salient 权重。数学上可证明 ε_rb = ||W - α_o·B_o - α_r·B_r||² ≤ ||W - α_o·B_o||² = ε_direct，即残差逼近的量化误差 ≤ 直接二值化的误差。这相当于用 2-bit 存储开销（两个二值矩阵 + 两个 scalar）达到了接近 8-bit 的 salient 权重保护效果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
BiLLM 中 Binary Residual Approximation 仅应用于 Hessian 选取的 salient 列，执行流程如下：
```
# Step 1: 初始二值化（针对 salient 列 W_sal ∈ R^{n×k}）
α_o = ||W_sal||_ℓ1 / (n × k)              # optimal scaling factor
B_o = α_o · sign(W_sal)                   # first binary matrix

# Step 2: 计算残差
R = W_sal - B_o                            # W_sal - α_o·B_o

# Step 3: 残差二值化
α_r = ||R||_ℓ1 / (n × k)                   # residual scaling factor
B_r = α_r · sign(R)                        # second binary matrix

# 最终逼近：Ŵ_sal = α_o·B_o + α_r·B_r （2-bit 有效位宽）
```
对比：PB-LLM 保留 salient 权重为 INT8（8-bit），BiLLM 用 2-bit（两个二值矩阵）达到更好效果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 BiLLM GitHub 仓库（github.com/Aaronhuang-778/BiLLM）中，binary.py 文件的 `res_approximation` 函数实现了该方法。适用于所有需要保护少数权重精度同时维持极低位宽的量化场景——理想的 salient 比例 r_salient=5-10%（BiLLM Table 1 显示 LLaMA-7B r_salient≈9%），额外存储仅 0.09 bit。该方法的关键洞察是：对 salient 权重，二次二值化逼近远比均匀量化高效，因为残差本身的数值范围远小于原始权重（残差分布更集中在 0 附近，利于二值化）。结合 OBC block-wise 补偿后，整体 binarization 过程约 0.5 小时/7B 模型（单 A100）。

涉及论文标题：
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---
