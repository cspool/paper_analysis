## PPEP（Online Performance, Power, and Energy Prediction Framework）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PPEP（MICRO 2014，Bo Su 等，AMD Research）是在线 PPE 预测框架：用硬件性能事件实现每核 CPI 模型与功率模型，预测各 DVFS 状态下 CPU 的性能、功率与能耗，进而快速搜索 DVFS 空间（用于单步功率封顶、DVFS 空间探索）。功率模型：P_idle(V)=a₃V³+a₂V²+a₁V+a₀（idle 功率为 V 的三次多项式，系数离线用 idle 功率 trace 回归）+ P_active(V,E)=Σ_i(w_i E_i)·(V^γ+V)（active 功率为计数器加权×电压缩放项，w_i 在固定 V 下用 active 功率 trace 离线拟合，γ 对比不同电压下行为拟合）；性能模型为 CPI 关于频率的线性形式。平均功率模型误差约 4.6%（原始 PPEP 论文）。
- PowerGrad（ISCA'26）的建模方法学直接沿用 PPEP：Gradient Estimator 用同款六计数器（instruction-count、cycle-count、uops.executed、cache-misses、branch-misses、ldm_stalls_pending，PPEP 同款）按 PPEP 工作流拟合 a_i、w_i、γ，在线生成 P(V) 与 CPI(f) 模型。PowerGrad 的贡献在 PPEP 之上：把模型微分出性能梯度 ∂BIPS/∂P 并用于分层功率分配（PPEP 本身不做梯度、不跨节点分配）。功率模型实测 AAE：Legacy 4.1%（vs PPEP 原论文 4.6%）、Accelerated 2.5%（更多细粒度计数器如向量/AMX 指令数）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- PPEP 在 PowerGrad 中的角色 = 在线模型工厂：每 100ms 把"离线系数（架构固定）+ 在线计数器（负载相关）"组合成当前工作负载/阶段的功率-性能模型，供微分出梯度。运转流程：
```
离线（每架构一次）: PARSEC 3.0（Legacy）/ TorchBench（Accelerated）跑训练应用
  → 收集 idle/active 功率 trace + 计数器 → 回归 a_i（P_idle 三次多项式）、
    w_i（active 系数）、γ（电压缩放指数）
在线（每 100ms）: 读计数器 E^(t) 与频率 f^(t) → 套 (1)(2) 算 BCPS/CPI/MCPI/CCPI
  → 生成 P(V)、CPI(f) → 微分 ∂BIPS/∂P → 供控制器
```
- 例子：Legacy 上 Llama 请求在 prefill（计算密集）与 decode（内存密集）间切换时，同一套离线系数 + 变化的计数器/频率实时改变模型形状与梯度，控制器据此转移功率。可移植性依据：新架构只需重训离线系数（换训练应用集）即可，这正是 PowerGrad"易重定向到不同架构"（Haswell vs Emerald Rapids+AMX）的机制来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PPEP 原论文在 AMD CPU 上验证（约 4.6% 功率模型误差），未公开源码（联网搜索未发现公开仓库，IEEE Xplore 文档 7011408）。PowerGrad 按其方法学在 Intel Haswell/Emerald Rapids 上实现并扩展到梯度；训练集：Legacy 用 PARSEC 3.0（无 ML 加速的传统多线程基准）、Accelerated 用 TorchBench（PyTorch 基准套件，能触发 AMX 指令）。使用场景：在线 DVFS 空间探索、单步功率封顶（PPEP 原始用途）、梯度驱动分层功率管理（PowerGrad 用途）。局限：模型为线性/多项式近似，短 kernel 预测更差（PowerGrad 实测 R² 方差随 kernel 时长变化）；精度足够迭代优化收敛即可，不需完美预测。

涉及论文标题：
- PowerGrad: Hierarchical Power Management for Power-Limited ML Inference Clusters
