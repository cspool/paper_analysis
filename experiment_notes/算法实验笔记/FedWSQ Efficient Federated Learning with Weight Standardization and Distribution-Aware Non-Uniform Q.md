## FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出FedWSQ，结合 Weight Standardization (WS) 梯度过滤和 Distribution-Aware Non-Uniform Quantization (DANUQ) 两个算法pipeline组件。WS在local training中通过连续投影（投影到 span{w˜_n,m, 1}^⊥）过滤掉与WSP向量对齐的分量和mini-batch梯度均值分量，从而缓解非i.i.d.数据导致的client drift。DANUQ基于标准正态分布先验，通过暴力搜索预先计算出最小化期望量化误差的最优量化级别（QLs），量化LMPU时使用共享global scaling vector（EMA更新），避免传输额外量化参数。论文对比实验包括：
  - 全精度FL方法：FedAvg、FedProx、FedAvgM、FedADAM、FedDyn、FedMLB、FedLC、FedNTD、FedSmoo、FedDecorr、FedWon、FedRCL、FedACG
  - 量化FL方法：FedPAQ（1-bit uniform quantization）、FedHQ+（4-bit/1-bit）
  - NUQ方法消融：NF（NormalFloat）、FP（Floating Point）vs DANUQ
  - 比特策略消融：FBA（固定比特分配）和DBA（动态比特分配，每轮随机1/2/4-bit，期望2.3bits）
  - 评估指标：CIFAR-10/100、Tiny-ImageNet测试集准确率（1000轮后）、收敛曲线、loss landscape Hessian top eigenvalue

- **硬件平台是什么，配置是什么。**
  NVIDIA RTX 4090 GPU。PyTorch框架实现。SGD优化器，初始学习率0.1，weight decay 0.001，指数衰减因子0.995。100个clients，5%参与率。每轮local training 5个epoch，batch size使每个local epoch含10次迭代。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：默认ResNet-18（将BN替换为GN），WS应用于每个GN层前。额外测试ShuffleNet、VGGNet-9、SqueezeNet、MobileViT作为backbone验证泛化性
  - 数据集：CIFAR-10（10类）、CIFAR-100（100类）、Tiny-ImageNet（200类）
  - 非i.i.d.设置：Dirichlet分布 α∈{0.1, 0.3, 0.6}，α越小数据异质性越高

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/Seongyeol-kim/FedWSQ

  **算法pipeline核心流程（参考Algorithm 1伪代码）**：

  1. Server端每轮采样clients S_t，广播GMP W_g^{t-1} 和 global scaling vector s_g^{t-1}
  2. Client端local training（K步迭代）：
     - Forward: 对每层权重向量w_{n,m}应用WS：w̃_{n,m} = (ρ/σ(w_{n,m})) · (I - P_1) w_{n,m}
       - (I - P_1): 减去均值（投影到span{1}^⊥），去除DC分量
       - 除以σ并缩放：标准化到零均值单位方差，乘以ρ控制scale
     - Backward: 梯度经双重投影过滤 ∂L/∂w_{n,m} = (ρ/σ) · (I - P_1)(I - P_{w̃_{n,m}}) ∂L/∂w̃_{n,m}
     - Optimizer step: W_i^k ← W_i^{k-1} - η∇f_i(W_i^{k-1})
  3. Client端量化LMPU ΔW_i = W_i^K - W_g：
     - 逐层归一化：ΔW_{i,l} / s_{g,l}（除以global scale，假设归一化后∼N(0,1)）
     - DANUQ量化：将归一化值映射到预计算的最优QLs
       - 1-bit QLs: [-0.798, 0.798]（省略q_0=0约束）
       - 2-bit QLs: [-1.224, 0, 0.765, 1.724]
       - 4-bit QLs: [-2.654, -1.974, -1.508, -1.149, -0.834, -0.544, -0.269, 0, 0.230, 0.465, 0.708, 0.966, 1.248, 1.568, 1.968, 2.649]
       - Quantization boundaries: u_r = (q_{r-1} + q_r)/2，将[0, +∞)分成R+1个区间
       - 量化规则：x ∈ [u_r, u_{r+1}) → q_r
     - 传输量化后的 ΔW̄_i 和 local scale vector s_i（未经量化）
  4. Server端dequantize并聚合：
     - Dequantize: Δ_i^t ← (ΔW̄_i^t, s_i^t)，还原为全精度
     - Aggregate: Δ^t ← Σ_{i∈S_t} h_i Δ_i^t
     - Update GMP: W_g^t ← W_g^{t-1} + Δ^t
     - Update global scale: s_g^t ← (1-β)s_g^{t-1} + β·(1/|S_t|)·Σ_{i∈S_t} s_i^t，β=0.1

  **DANUQ QLs预计算原理**：
  目标是最小化 E[(Δw - Δw̄)^2] = Σ_{r=0}^R ∫_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx，其中p(x)为N(0,1)的PDF。由于closed-form解难以获得（含高斯积分和误差函数），采用暴力搜索在合理范围内离散搜索最优QLs。搜索空间限制在经验范围内，使用并行处理加速。

  **FBA/DBA混合精度策略**：
  - FBA: 每个client固定比特宽度（从{1,2,4}中选择）
  - DBA: 每轮每个client随机分配比特宽度∼Uniform{1,2,4}，期望约2.3bits
