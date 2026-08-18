## 热致掉队（Thermally Induced Straggling，热不平衡导致的 GPU 性能掉队）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 热致掉队描述因过热导致的性能下降：GPU 等设备在温度超过阈值后，设备级功率/频率管理（DVFS）自动降频（或降 IO 总线频率、启用高级 ECC）以保证可靠执行，使该设备比同节点其他设备更慢，成为"straggler（掉队者）"。文献报道过热可让 microbenchmark 性能降超 50%、macrobenchmark 降 3%-4%。Lit Silicon 论文（ISCA'26）把节点内更热更慢的 GPU 称为 straggler、更冷更快的称为 leader：实测 8×AMD MI300X 节点内最高温度与最低相差 1.155×、频率相差 1.062×，且 GPU 温度排名与频率排名（降序）几乎一致，强烈表明温度→频率→性能的因果链。
- 与工作负载无关：热致掉队是设备级 DVFS 的固有行为，即使所有 GPU 执行完全同构的负载（FSDP 训练）也会发生。论文还观察到 DVFS 管理"过度降频"现象：某个 GPU 不是最热却是第二热，频率却最低（推测温度超阈值后被过度压制）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（硬件内闭环）：GPU 计算产生功耗 → 片上温度传感器（thermal sensor）测到温度升高 → 超过 DVFS 阈值 → 时钟/电压管理单元（CMU/PMU）下调核心频率与电压 → SM 执行速率下降 → 该 GPU 的 kernel 运行时变长 → 在多 GPU 同步点（通信集合）拖慢整个节点。Lit Silicon 论文用 amd-smi 采样温度与频率验证（图 5）：三个训练迭代中，straggler 温度/频率保持在高/低两端，leader 反之。该硬件行为是节点级性能波动的物理根源，且与 C3（并发计算通信重叠）耦合形成负反馈循环（straggler 更慢→leader 等待→C3 重叠延长→leader 也变慢）。
- 影响量化：constant overlap kernel（0%/100% 重叠）上 straggler 比 leader 慢 5%-10%；varying overlap kernel 上反而 straggler 更快（1.5×），因为 leader 在等待中承担了更多 C3 资源竞争——这解释了为何"只优化通信/只修 straggler"不够，必须对齐频率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件侧 DVFS 与温度管理是芯片内置的（AMD/Intel/NVIDIA 各代 GPU 均有），软件可观测与干预：amd-smi（AMD，github.com/ROCm/rocm-systems）可读温度/频率/功率并设功率上限，nvidia-smi（NVIDIA）类似（nvml 查询温度、锁频）。使用方式：监控多 GPU 节点的温度/频率差异定位 straggler；Lit Silicon 论文用功率上限（power cap）而不是直接锁频来纠正掉队（功率上限比频率上限更可预测），把 leader 的功率让给 straggler 提升其频率，或直接降 leader 功率省电。部署前提是拥有管理员权限（多租户集群通常需要固件/离线方案）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads
