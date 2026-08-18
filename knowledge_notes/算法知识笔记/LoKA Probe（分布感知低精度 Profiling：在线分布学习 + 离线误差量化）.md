## LoKA Probe（分布感知低精度 Profiling：在线分布学习 + 离线误差量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LoKA Probe 是 LoKA 三大组件之一，实现"分布感知 profiling"原则：在线学习每层激活与权重的统计分布（不存原始张量，避免存储爆炸与过拟合），再离线从学习分布采样合成输入/权重做统计显著的 MERE 与吞吐评估，定位"哪些层能安全又高效地用 FP8"。核心洞察：标准库用随机（正态）张量做基准会系统性低估真实量化误差——真实 LRM 激活重尾、相关、非平稳，随机基准漏检。
- 统计建模：激活按 batch 维独立，建模为多元高斯 T~G(μ,Σ)，把协方差存储从 O(M²N²) 降到 O(N²)（推荐模型避免跨 batch 算子如 BatchNorm 防信息泄漏，使 batch 独立成立）；用批量 Welford tracker 流式更新均值/散度（合并公式见下）。权重无维独立假设，建模为矩阵正态 W~MN(M,U,V)（vec(W)~N(vec(M),V⊗U)），Kronecker 积分解耦行列协方差，存储 O(M²+N²)，用 flip-flop 式 EMA + Cholesky 线性求解在线更新，trace 重归一化防尺度漂移。为控制开销：每 100 训练迭代激活统计、每 10000 迭代异步保存参数，总开销 ≤1%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在线合并（Welford，激活）：当前批 X∈R^{B×K}，批均值 μ_b、批散度 S_b=(X−1_Bμ_bᵀ)ᵀ(X−1_Bμ_bᵀ)；合并历史 (n_old,μ_old,Σ_old)：n_new=n_old+B、δ=μ_b−μ_old、μ_new=μ_old+(B/n_new)δ、Σ_new=Σ_old+S_b+(n_old·B/n_new)δδᵀ；样本协方差 Σ=Σ_new/(n_new−1)（FP32 累积）。
- 在线更新（权重，矩阵正态）：每 minibatch 解 L_VL_Vᵀ=V+εI 得 U'=(1/N)(W_c L_V⁻ᵀ)(W_c L_V⁻ᵀ)ᵀ；解 L_UL_Uᵀ=U+εI 得 V'=(1/M)(L_U⁻¹W_c)ᵀ(L_U⁻¹W_c)；EMA 平滑 U''=mU+(1−m)U' 后对称化+εI 正则；尺度重归一化 s=trace(U)/M、U←U/s、V←sV（m∈[0.9,0.99]，ε≈10⁻⁶×trace(U)/M）。
- 离线采样与评估：激活 T'=1_Bμᵀ+ZL_Σᵀ（Z~N(0,I_K)，L_Σ 为 Σ+εI 的 Cholesky）；权重 W'=M+L_UZL_Vᵀ。对每层采样 100 对，跑 FP8 vs TF32 算 MERE + 计时，MERE 高或加速比低的层标记为低精度不安全。
- Probe 关键发现（Wukong 分析）：bias 项发散（部分 bias 范数不收敛、≥0.1，级联导致越界/量化湮没小值）；归一化开销与 mean-cancellation 误差（LayerNorm 需高精度反量化/重量化往返）；sigmoid 型 Swish 指数运算放大离群值、量化损失剧增。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为训练钩子（hook）接入 PyTorch 线性层，流式维护每层统计量；离线 benchmark 模块采样合成张量驱动 TorchAO/DeepGEMM/FBGEMM kernel 评测。使用方式：训练早期启用收集分布 → 离线跑 MERE+吞吐矩阵 → 输出每层 (库, recipe) 的精度/性能表 → 供 LoKA Dispatch 过滤与选择。作用：把"哪里低精度安全"从拍脑袋变为统计可判，并发现标准基准漏检的误差（含 FBGEMM faulty test code，MERE 差 47×）。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
