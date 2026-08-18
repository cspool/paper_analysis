## 全局缓冲（Global Buffer，GLB）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GLB 是加速器芯片上的共享主存储，位于 PE 阵列与 DRAM 之间，用多 bank 组织以支持高带宽并行访问。TensorPrism 的 GLB（2.5MB）承担四类数据：稀疏张量输入、输出（部分和+最终结果）、共现图元数据、稠密张量缓冲，按 1MB 稀疏元数据 + 1MB 稠密操作数 + 512KB 共现图元数据/中间缓冲分区。PE 阵列经 crossbar 与 GLB 交换数据（packet-switched 动态路由，区别于 TCP 电路交换固定拓扑），加速器经 DRAM 控制器接片外 HBM2（307.2GB/s）。GLB 占加速器总面积 50%（设计选择：大容量支撑复用）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GLB 容量 $M_{cap}$ 是式 5 DRAM 访问模型的硬约束：$(\prod_{V_{c_i}\in V_c}|V_{c_i}|+\prod_{V_{f_j}\in V_{f_1}}|V_{f_j}|)\times4\times M_t+4\times(2E_I^p+E_Y^p)\le M_{cap}$，决定 tiling 因子 $M_t$（$f_2$ 维稠密行驻留数）。数据路径：host → DRAM 控制器 → GLB（四类数据分区）→ CoG Scheduler 取元数据建图/划分 → PE 阵列经 crossbar 取稠密行/图条目 → 收缩引擎计算 → commit unit+MAG 写回 GLB → DRAM 写回 host。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：多 bank SRAM（CACTI 7.0 建模面积/能耗）+ crossbar 互连 + DRAM 控制器；与 PE 48KB 局部存储构成两级片上层次。作用：容量直接决定稠密行驻留与复用距离（amre 上 94% 访问来自片上缓冲）；DRAM 访问归一化 1/2.18/2.11/1.27/1.53（vs SPADE/HotTiles/GSpTC/TCP）；片外能耗 47.4%（vs TCP 79.1%）。使用场景：FROSTT 8 数据集 + LLaMA 注意力张量的端到端仿真（GLB 容量约束式 5 决定每数据集 tiling）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
