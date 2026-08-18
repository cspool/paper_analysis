## Lead Value 与 Straggler Wave（领先值与掉队波，kernel 起始时间差的检测指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Lead value 是 Lit Silicon 论文（ISCA'26）提出的检测指标：对每个 kernel k，比较其在所有 GPU 上的起始时间戳，leader GPU 比最晚（straggler）GPU 早开始的时间差即该 kernel 上该 GPU 的 lead value；把同一 GPU 上所有 kernel 的 lead value 聚合（sum=曲线下面积，默认；也可 max/last）得到每 GPU 的聚合 lead 值。straggler 的 lead≈0（总是最晚），leader 的 lead>0。Straggler wave 指 trace 中连接各 GPU 同一 kernel 起始时间形成的波前图：straggler 的波前最晚、leader 波前领先。
- 与 C3 的关联：straggler 通信起始晚 → leader 等待延长（C3 重叠变长、资源竞争）→ leader 变慢，lead 值在迭代内动态积累到 equilibrium 后重置，跨迭代重复。检测使用窗口平均（默认窗口 3 个采样）平滑。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 1: LEADVALUEDETECT
Input: Timestamp vector T[g,k] for g in GPUs, k in kernels
Output: Lead value vector L[g]
for each Kernel k:
    T_max <- max(T[all GPUs, k])
    for each GPU g:
        lead_value[g,k] <- T_max - T[g,k]
for each GPU g:
    L[g] <- sum_k lead_value[g,k]     # 或 max / last
return L
```
Annotations：T_max 是该 kernel 在节点内最晚的起始时间（straggler 的时间）。示例：GPU0 比 GPU1 早 10ms 开始某 kernel，则 GPU0 该 kernel lead=10ms；若某 GPU 的 lead 在 100 个 kernel 上从 0 线性涨到 10ms，其 sum 聚合 lead≈500ms。聚合方式选择：sum 在 equilibrium 期间也惩罚 GPU（利于在乘法性 C3 干扰下识别 leader），max/last 收敛更快但信息少。输出 L[g] 作为 Algorithm 2 功率上限增量计算的输入（norm_lead 归一化后乘 global 衰减与 max_inc 默认 15W）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时解析 PyTorch profile trace（Chopper 工具）获得每 GPU 每 kernel 起始时间戳，按上述算法计算聚合 lead 值；实际部署每 10 迭代采样一次（每样本 dump+处理约 4 秒），窗口平均后用于调整各 GPU 功率上限（amd-smi），收敛（约 20 样本/80 秒）后功率分布可停用或长周期复用。使用场景：检测多 GPU 节点内的 straggler/leader 归属（热致掉队的量化）、驱动节点级功率重分配（GPU-Red/GPU-Realloc/CPU-Slosh）、评估 C3 引起的性能波动；也用于 MoE 训练（all-to-all 不重叠时 lead 值小但有大 spike，仍可收敛）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
