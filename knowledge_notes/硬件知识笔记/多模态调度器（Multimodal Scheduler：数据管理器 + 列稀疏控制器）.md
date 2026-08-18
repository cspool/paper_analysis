## 多模态调度器（Multimodal Scheduler：数据管理器 + 列稀疏控制器）

术语解释
DiTPA 中管理动态多模态数据与稀疏数据的片上调度单元：data manager 负责模态数据的缓冲/复用/更新与校准数据寻址，column sparsity controller 负责 action 模态注意力近零列的检测与跳过，使动态数据操作的时延可忽略（GPU 上该开销占 35.4% 总时延）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
三层冗余消除使输入 token 序列在去噪步间动态增删（跳过步、跳过模态、稀疏列），通用平台的 CPU 端数据搬移/索引管理成为瓶颈。多模态调度器把"什么 token 参与本步计算"解耦为可硬件解码的配置：两级索引表存离线生成的跳过配置（去噪步跳过模式、多模态 token 跳过模式），lifespan 序列生成器存运行时状态。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
data manager 流程：两级状态寄存器——action-level 寄存器存动作预测器的 skip 状态 + iteration 索引表的更新状态；iteration-level 寄存器存消除累积近似误差的计数器状态 → lifespan 序列生成器按状态解码本步 token 序列 → 与索引表默认序列逻辑组合 → 特征地址生成器 + 校准数据地址生成器更新输入 token 与校准数据（K/V 缓存）地址。column sparsity controller 流程：每个注意力头单元内含多组列比较器，判定 attention score 是否近零 → 结果存 bit 寄存器（1=近零）→ OR 归约判定整列稀疏 → 零列旁路、非零列索引入队 → 计算时按队列顺序出列，S×V 的稀疏列运算直接跳过 → dispatcher 把最长与最短头序列平均化，均衡各头计算时延。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：28nm 片上单元，功耗 29.54mW（2.82%）、面积 0.06mm²（1.37%）；配合 on-chip SRAM（2.44mm²/55.84% 面积、22.30% 功耗）缓存特征与校准数据。使用：与动作预测器（42.28% 动作跳过）协同后整体仅 2.87% 片上功耗却消除 42.28%+31.66% 计算；DRAM 侧权重重载降低 65.37%，是能效 2356.77×/8.71×/11.59× 的主要来源。外部 DRAM 为 LPDDR5，能耗按厂商规格 [24][29] 评估。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
