## Iterative Ternary Fitting (ITF, 迭代三值拟合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Iterative Ternary Fitting（ITF，迭代三值拟合）是 PT²-LLM 提出的三值参数无训练优化算法。ITF 将三值量化参数 (α, μ, T) 的优化建模为交替最小化问题：(1) 固定 T 时，通过最小化权重量化误差 E_w = ||W - (αT+μ)||²，对 α_i 和 μ_i 求偏导并置零，得到闭式解（Eq. 9）向量化并行逐行求解最优网格参数 (α*, μ*)；(2) 固定 (α*, μ*) 时，通过 Z_ij = (W_ij - μ_i) / α_i 将权重投影到归一化空间，弹性舍入到最近的三值：T*_ij = argmin_{t∈{−1,0,1}} |Z_ij - t|。两步骤交替进行，每一步贪心地减小 E_w，通常约 10 轮收敛（T 不再变化）。ITF 完全无训练、无梯度反传。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: W ∈ R^{n×m}
α, μ, T = Asymmetric_Ternary_Init(W)        # TWN风格初始化
T_prev = zeros_like(T)
while T ≠ T_prev:                           # 约 10 轮收敛
    T_prev = copy(T)
    # Step A: 闭式求解最优网格 (Eq. 9, 向量化逐行并行)
    α* = (m*(W∘T)1 - (T1)∘(W1)) / (m*(T∘T)1 - (T1)²)
    μ* = ((T∘T)1∘(W1) - (T1)∘((W∘T)1)) / (m*(T∘T)1 - (T1)²)
    # Step B: 弹性舍入更新 T
    Z_ij = (W_ij - μ*_i) / α*_i
    T*_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
```
ITF 将 LLaMA-2-7B 三值化的 WikiText2 PPL 从 22.88（初始化后）降至 15.47。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ITF 在 PyTorch 中通过向量化张量运算实现：(1) Build_Optimal_Grid 使用 Eq. 9 的逐元素和张量运算（无 Python 循环）；(2) Flexible_Round 通过 argmin 查找 Z 到 {−1,0,1} 的最近映射；(3) 收敛判断通过 (T != T_prev).any()。ITF 约 10 轮可收敛。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models
