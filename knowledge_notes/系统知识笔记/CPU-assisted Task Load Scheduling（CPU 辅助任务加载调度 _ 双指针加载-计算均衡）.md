## CPU-assisted Task Load Scheduling（CPU 辅助任务加载调度 / 双指针加载-计算均衡）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU-assisted Task Load Scheduling 是 SMoE 的兜底调度策略，回答"既不能替换、也不能及时预取的未驻留专家，该加载到 GPU 还是留在 CPU 计算"：用 Algorithm 2 的双指针（two-pointer）贪心，按专家分数降序，把高分数专家优先分配给 PCIe 加载（load 侧）、低分数专家分配给 CPU 计算（CPU 侧），在累计成本上均衡 T_load = n_load × C_load 与 T_CPU = n_CPU × C_CPU，目标 min max(T_load, T_CPU)。C_load 为单个专家 PCIe 传输时间（主成本，GPU 计算时间可忽略故近似总时间），C_CPU 为单个专家 CPU 计算平均时间（低 batch 下视为常数，Fig.9 显示 CPU 时间对 batch 不敏感），用过去 p 次实测更新。它覆盖 prefetching 失败（高需求/低带宽）与预测错误的场景，以及 expert-cache router 找不到足够相近替换候选的 low-score 专家。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 2：S = 按 score 降序排列的未驻留专家 {uid_i}
T_CPU ← 0, T_load ← 0, n_load ← 0, n_CPU ← 0
l ← 0, r ← |S| − 1
while l ≤ r:
    if T_load ≤ T_CPU:                       # load 落后→ 给高分数专家加载
        T_load ← T_load + C_load; n_load += 1; l += 1
    else:                                    # CPU 落后→ 给低分数专家 CPU 计算
        T_CPU ← T_CPU + C_CPU; n_CPU += 1; r −= 1
```
运行流程（层 i 到 i+1 的 pipeline，Fig.10）：CPU 侧四类操作（expert-cache router 计算、CPU-assisted 调度、CPU 专家计算、cache eviction+protection shield）与 GPU 侧四类（common params 计算、直接专家计算、预取预测、新加载专家计算）及 PCIe 两类（预取层 i+1、加载层 i）并行流水线；数据依赖约束为 cache 更新在 gating/attention 后、GPU 专家计算在数据加载完成后、层间串行，bubble 被 PCIe 主导延迟掩盖。消融（+BA）相对 +Pre 再降 TPOT 34%（代价 cache ratio 降 3%——CPU 计算的专家不更新 GPU 缓存）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SMoE 运行时（https://github.com/goingshr/SMoE）中由 if_usecpu 字段开启 CPU 兜底；config 中 cpu_cores 配置（默认 4，n-1 用于计算、1 用于加载/后台 worker）。设计取舍：与 kTransformer（依赖 AMX/AVX-512 高端 CPU 指令）不同，SMoE 的 substitution-centric 设计把 CPU 计算量压到最低，兼容老旧/边缘 CPU（无专用指令），且消除对 batch 波动的敏感。局限（论文 Discussion）：S3 下 CPU 时间仅为 load 时间的 1/3，需大量专家才能平衡，且 MoE 每步激活专家数有限，难以始终完美均衡 PCIe 与 CPU 时间。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
