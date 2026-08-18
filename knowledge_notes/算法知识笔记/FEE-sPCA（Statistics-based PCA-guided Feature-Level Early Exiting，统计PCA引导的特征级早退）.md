## FEE-sPCA（Statistics-based PCA-guided Feature-Level Early Exiting，统计PCA引导的特征级早退）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FEE-sPCA 是 NASZIP 提出的算法级压缩技术，解决朴素 EE 部分距离收敛慢的问题：先对向量库做 PCA 变换使前 k 维集中最富信息的分量，再以"估计全维距离"d_est^k = α_k·d_part^k/β_k（α 放大、β 校正）与 threshold 比较，使 d_est^k≥d_part^k 从而更早触发早退；β 由统计方法（Chebyshev 不等式）保证估计不低估真实距离，避免误杀本应入队的候选、维持 recall。它把平均特征计算量削减约 50%（高维数据集更多：GIST 960 维中 80% 早退发生在第 193 维内）。Offline 阶段（一次）完成 PCA 与 α/β 计算，Online 搜索阶段仅查表缩放。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Offline 预处理（图 6 上半部分）：(1) 对数据库 P 做 PCA → 变换库 VD 与特征值 {λ_i}；(2) 由期望性质 E(‖v_1:d‖²/‖v‖²)=Σ_{i≤d}λ_i/Σ_{i≤D}λ_i 得 α_k = Σ_{i=1..D}λ_i / Σ_{i=1..k}λ_i；(3) 用 Chebyshev 不等式 P(|α_k·d_part^k/d_all − 1| ≤ ε_k) ≥ 1 − Var_k/ε_k²，取 1+ε_k=β_k 使 P(α_k·d_part^k/β_k < d_all) ≥ 1−Var_k/(2ε_k²)（论文取 ≥90%），Var_k 在索引构建时统计。Online 搜索（每候选向量，逐 burst 步进）：
```
d_part ← 0
for k = 1 to D（每 burst 读入 b 个特征）:
  d_part ← d_part + Σ_{i∈burst} (x_i − q_i)²
  d_est ← α_k · d_part / β_k                    # 估计全维距离
  if d_est ≥ threshold: return REJECT          # 更早触发早退
return ACCEPT
```
例子（论文 Fig.6b，SIFT 场景）：阈值 2.5，s2 在前 2 维算得 d_part²，d_est²=α₂·d_part²/β₂<threshold 继续；第 4 维后 d_est⁴≥threshold 即早退。对比朴素 EE（图 7）：朴素要算到第 109 维才触发，FEE-sPCA 第 4 维触发；对 d_all<threshold 的"应接受"向量，β 校正（黄色虚线）防止估计过高误触发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线 PCA 在 A100 GPU 上执行（BigANN 1B 约 430s、SIFT 1M 约 6.5s），α/β 表随索引存储；在线查询也需一次 PCA 变换（表 IV：0.1-0.8ms，占搜索延迟 0.1%-3.8%）。硬件上由 VPE 的 FEE 模块实现（按 burst 更新部分和并用 α/β 缩放比较，见硬件架构 VPE 条目）。开源（https://github.com/Intelligent-Computing-Research-Group/NasZip）中对应 preprocess_idx/ 的 PCA 预处理与 simulate/ 的搜索；离线过程仅在数据库更新达约 30% 时重跑。适用：高维向量检索（GIST/Wiki 收益最大），与 Dfloat 位级压缩正交组合。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
