## 硬件噪声调度器（Noise Scheduler）与 SFU（Special Function Unit）

术语解释
DiTPA 片上两个计算辅助单元：noise scheduler 按去噪步索引驱动噪声逐步更新直至输出无噪动作（正常步走完整更新、特征复用步走低代价残差更新）；SFU 执行 LayerNorm、GELU 等非线性算子。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
反向扩散的每一步需要把 x_t 更新为 x_{t-1}（噪声调度递推），参数随步索引变化（如 √ᾱ 表、β_t 表）；DiTPA 将这一递推独立成片上 noise scheduler，与 PE 阵列的 attention/FFN 主计算分离。SFU 承接 GEMM 之后的非线性（LayerNorm 归一化、GELU 激活），避免非线性在 PE 阵列内打断矩阵数据流——这也是论文对 Ditto 类差分计算方案的批评点之一（差分计算对高精度非线性算子支持差、造成严重外存访问）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
一个去噪步的片内流转：PE 阵列完成 QKV/attention/FFN GEMM → SFU 做归一化/激活 →（特征复用步）跳过 GEMM 后仅 noise scheduler 用缓存特征做残差噪声更新 → 下一步或输出。交替去噪跳步时噪声更新仍是每步必做项，因此 noise scheduler 路径与 PE 路径按迭代索引表选择切换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：28nm，noise scheduler 41.54mW（3.97%）/0.06mm²（1.37%），SFU 104.58mW（10.00%）/0.34mm²（7.78%）。使用：与 multimodal scheduler 的迭代索引表联动实现"每 20 个跳过迭代重插一次完整去噪"；论文未给出 noise scheduler 内部精度格式（论文未明确说明定点/浮点实现细节）。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
