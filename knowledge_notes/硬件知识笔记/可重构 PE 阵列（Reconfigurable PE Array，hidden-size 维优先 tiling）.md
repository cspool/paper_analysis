## 可重构 PE 阵列（Reconfigurable PE Array，hidden-size 维优先 tiling）

术语解释
DiTPA 的矩阵计算阵列：12×6×2 可重构 PE 阵列，每 PE 每周期 64 MAC，通过 mode1/mode2 单周期状态机重构匹配不同算子的计算 tile；配合 hidden-size 维优先 tiling 与 ping-pong buffer 应对冗余消除后 token 长度动态变化（最高 10×）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
通用加速器的脉动阵列按固定 token 长度分 tile，而 DiTPA 的多级冗余消除使每步参与计算的 token 数动态变化（动作复用、去噪步跳过、模态跳过、列稀疏），利用率大幅下降（GPU <20%）。关键观察：无论 token 长度如何变，hidden-size 维不变。因此把矩阵运算沿 hidden-size 维分 tile，使每个 tile 长度恒等于 hidden-size、每 tile MAC 数恰好匹配 PE 阵列规模，token 长度变化不影响 tile 形状与利用率（实测 98.36%）。12×6×2 = 144 PE × 64 MAC/cycle。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
数据流按算子分为两 Case：Case 1（Q 投影等，激活宽 W 与权重高 H 都等于 hidden-size）：激活 tile 沿 PE 行方向广播、权重 tile 沿列方向广播并被所有 PE 行共享；激活静止、逐周期右移与相应权重列相乘，直到权重最后一列；完成一个激活 tile 后两者切到下一 tile 重复；ping-pong buffer 双缓冲重叠新激活 tile 的装载，隐藏 pipeline 切换。Case 2（FFN 第二个 Linear、S×V）：只有权重 W 对应 hidden-size，权重沿 W 分 tile；mode2 下激活与权重 tile 在 PE 间共享，每个激活元素 × 每个权重元素得中间结果存入寄存器，下一周期激活右移、权重下移并累加，直到权重最后一行；S×V 的多头输入全部头同时计算（并行最大化利用率）。重构：单周期状态机切换 mode1/mode2，与缓冲数据读取重叠、完全隐藏于流水线。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：28nm 工艺、500MHz、0.88V，Synopsys Design Compiler 综合估面积、PrimeTime PX 估功耗；PE 阵列占整芯片 59.63% 功耗、32.49% 面积（1.05W/4.37mm² 中的 623.80mW/1.42mm²）。使用：执行 Q/K/V 投影、attention S×V、FFN 三个 Linear 的 INT8 GEMM；与 SFU（LayerNorm/GELU）、多模态调度器协同完成每个去噪步；消融显示该设计使矩阵乘利用率 98.36%，是 GPU 上软件框架仅 2.3× 加速而硬件达 386.93× 的核心原因之一。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 加速器补充视角（ISCA'26，统一光栅化/MLP 可重构 PE 阵列）：本论文的 16×16=256 PE 阵列在运行时经 MUX 配置在同一数组上执行两种模式——rasterization 模式（每周期处理一个 Gaussian 对 16×16 tile 像素的 α 计算+α-blending）与 MLP 模式（每周期处理 256 个深度值、输出 256 个衰减因子 F(d)），由此消除专用排序引擎（排序 O(N log²N) 与光栅化 O(N) 复杂度异构、面积按 k·log²k 增长、tile 负载波动两个数量级导致 pipeline 失衡）；每 PE 6 MUL+6 ADD+1 EXP（FP16），MAC 分两组（M-{1~3}/A-{1~3} 独立、M-4-{1~3}/A-4-{1~3} 协作），广播寄存器 10 单元，Leaky ReLU 用符号检测+5-bit 整数加法器（FP16 指数减 3）实现。开销：相对同 MAC 数的专用光栅化阵列仅 +5% 面积/+6% 功耗/+2 cycle（模式配置+寄存器清空）；相对"MLP 与光栅化分用两个独立阵列"方案 1.91× 面积效率（吞吐/面积）、1.89× 能量效率（吞吐/功耗）；MLP 模式相对 32 并行 bitonic 排序网络 21.1~32.4× 加速。与 DiTPA 的可重构 PE 阵列（hidden-size tiling 的 mode1/mode2 GEMM 重构）同属"运行时重构提升利用率"思想，但本论文面向 3DGS 的光栅化-推理异构计算；TSMC 28nm、1 GHz、3.85mm²/1.64W。
