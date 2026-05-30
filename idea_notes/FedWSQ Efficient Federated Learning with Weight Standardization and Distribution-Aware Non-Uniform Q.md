## FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

- baseline方法是什么？
  Baseline 是标准 **FedAvg** 及现有量化FL方法（FedPAQ、FedHQ+）。FedAvg的典型全栈执行例子（ResNet-18, 100 clients, CIFAR-100, RTX 4090）：

  - **算法pipeline**：Server每轮广播GMP W_g → 每个client用本地non-i.i.d.数据SGD训练K步得到LMP → client计算LMPU ΔW_i = W_i - W_g → client传输全精度（32-bit）ΔW_i至server → server加权聚合 Δ = Σ h_i ΔW_i → 更新GMP W_g ← W_g + Δ。若加量化（FedPAQ）：用absmax scaling将ΔW_i缩放到[-1,1] → uniform quantization到固定B-bit → 概率舍入避免QL集中 → 传输量化值+scale。FedHQ+在此基础上对每个client按量化误差加权。
  - **系统框架**：PyTorch，自实现FL simulator（100 clients × 5% participation），无Serving框架。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准PyTorch CUDA kernel（conv2d、BN/GN、linear），无自定义kernel。
  - **硬件架构**：NVIDIA RTX 4090 GPU，无自定义硬件。

  **Baseline 的核心缺陷：**
  1. **Client drift源于梯度偏差**：在non-i.i.d.数据下，local SGD梯度包含两个偏差分量：(a) 与当前LMP对齐的分量——local模型过拟合本地数据导致参数偏离GMP；(b) mini-batch梯度均值分量——biased toward local data distribution。这两个分量叠加导致各client的LMPU方向不一致，全局聚合后偏离最优参数。
  2. **absmax scaling对离群值敏感**：FedPAQ和FedHQ+使用absmax将张量缩放到[-1,1]，outlier会过度扩展动态范围，在低比特（1-bit/2-bit）下导致严重的underflow——大部分正常值被压缩到极窄区间内，量化后信息丢失严重。
  3. **Uniform quantization浪费容量**：LMPU实际近似正态分布（密集区域在均值附近），但UQ将范围均匀划分为2^B等间隔，在密集区域精度不足、稀疏区域容量浪费。
  4. **现有NUQ方法低比特乏力**：NF（NormalFloat）和FP（Floating Point）在1-bit/2-bit下性能急剧退化（如CIFAR-100 α=0.1时NF仅24.0%，FP仅7.0%），因为它们的QL设计未针对极端低比特优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FedWSQ = WS梯度过滤 + DANUQ分布感知量化**：

  **(1) WS梯度过滤解决client drift**
  Baseline local training的梯度直接沿raw gradient方向更新。FedWSQ在每层前向传播前对权重向量w_{n,m}应用WS：w̃_{n,m} = (ρ/σ(w_{n,m}))(I - P_1) w_{n,m}（先减均值去除DC分量，再除以标准差归一化）。反向传播时梯度经历双重投影（见Eq.6）：∂L/∂w_{n,m} = (ρ/σ)(I - P_1)(I - P_{w̃_{n,m}}) ∂L/∂w̃_{n,m}：
  - 第一重投影 (I - P_{w̃_{n,m}})：移除与WSP向量对齐的梯度分量（即local overfitting偏好的方向）
  - 第二重投影 (I - P_1)：移除mini-batch梯度均值分量
  结果：梯度被投影到 span{w̃_{n,m}, 1}^⊥，仅保留对global convergence有益的方向。这等价于一种隐式正则化，无需修改loss函数或优化器结构。

  FedWSQ传输PSP而非WSP（区别于FedWon），通过梯度过滤隐式纠正偏差而不强制client间参数统计一致，保留了本地适应性信息。

  **(2) DANUQ以标准差做scale + 正态分布最优QLs解决量化瓶颈**
  - scaling：不用absmax（对outlier敏感），改用LMPU的标准差σ作为scale factor。因为σ更稳健且与N(0,1)假设一致。Global EMA scale vector s_g = (1-β)s_g + β·mean(s_i) 在各client间共享，保证量化一致性。
  - QLs预计算：假设归一化后LMPU ∼ N(0,1)，求解 min_{q_1,...,q_R} E[(x-q)^2] 得到最优QLs。因closed-form不可得，用暴力搜索在合理范围内穷举，结果为：1-bit[-0.798,0.798] / 2-bit[-1.224,0,0.765,1.724] / 4-bit[16个非均匀间隔QLs]。这些QLs密集分布于高概率密度区域（均值附近），稀疏分布于尾部，比UQ同位数下信息损失小。
  - 无额外通信开销：QLs固定预计算，无需每轮传输量化参数或学习步长/零点。

  FedWSQ全栈执行例子（对比baseline）：
  - **算法pipeline**：Server广播 (W_g, s_g) → Client local training（WS前向+双投影梯度过滤，K步SGD） → Client DANUQ量化（ΔW_{i,l}/s_{g,l} → 查表映射到预计算QL → 得B-bit整数index） → Client上传 (ΔW̄_i, s_i) （量化值+1个scale/Layer的float） → Server dequantize（查表+乘scale还原全精度） → 聚合+EMA更新scale。
  - **系统框架**：PyTorch自实现FL simulator（同baseline），DANUQ为纯Python/CUDA查表操作，不依赖额外框架。
  - **编译框架/kernel调度/硬件架构**：论文未明确说明，与baseline相同的PyTorch CUDA kernel执行，无自定义kernel或硬件。

  **关键设计选择 vs Baseline缺陷对应**：
  - Baseline缺陷1（梯度偏差）→ WS投影过滤 (I-P₁)(I-P_{w̃}) 双重投影
  - Baseline缺陷2（absmax对outlier敏感）→ 标准差scaling + global EMA，对outlier更稳健
  - Baseline缺陷3（UQ容量浪费）→ DANUQ基于N(0,1) PDF设计非均匀QLs，密集区域细粒度、稀疏区域粗粒度
  - Baseline缺陷4（现有NUQ低比特弱）→ DANUQ直接为1/2/4-bit暴力搜索最优QLs，CIFAR-100 α=0.1 1-bit从NF 24.0%提升至84.8%
