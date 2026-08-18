## TDP（Thermal Design Power，热设计功耗）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TDP 是芯片厂商给出的最大散热设计功耗上限：在该功耗下运行可保证可靠执行（不超温、不降频）。GPU 的 DVFS 在 TDP 内管理电压/频率以兼顾性能与功耗。Lit Silicon 论文（ISCA'26）把 TDP 作为功率上限的基准：baseline 是"所有 GPU 都在 TDP 运行"，功率管理算法（Algorithm 3 ADJPOWERNODE）调整各 GPU 功率上限时不得超 TDP；GPU-Red 用例把 leader 从 TDP 下调 15W 省电，GPU-Realloc 把 straggler 提到 TDP（且不超过）、GPU-Realloc/CPU-Slosh 评估了 700/650/600/550/500W 等不同初始功率上限（默认 700W）。
- 论文还引用规范：允许在毫秒级短暂超过 TDP（[66]），给 GPU-Realloc 更多空间；但长期稳定运行以 TDP 为界。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件作用：TDP 决定散热系统（散热器/风扇/液冷）设计容量、供电（VRM）规格与机架功率预算。运转流程：系统上电 → 固件读取各设备 TDP → 功率上限默认设为 TDP → DVFS 在 TDP 内调速 → 温度超限时降频（热致掉队）。节点级：8×MI300X 的节点功率 = Σ 各 GPU 功率（≤8×TDP）+ CPU 等；节点级功率上限（power cap）小于该和时，需要把功率在各 GPU 间分配——Lit Silicon 的 Algorithm 3 先按 Algorithm 2 增量提升 straggler，再把超出节点上限的部分均匀回退到所有 GPU（gpu_delta = ceil((node_power - P_n)/G)），并再按 TDP 调整（不超 TDP）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TDP 是出厂规格，硬件侧由固件/驱动默认执行（功率上限默认 = TDP）；软件侧 amd-smi / nvidia-smi 可查询与设置功率上限（≤TDP 或短暂超 TDP）。使用：数据中心按 TDP 做机架/PDU 功率预算；功率超订（oversubscription）场景下节点级功率上限 < ΣTDP，须用功率管理算法（如 Lit Silicon 的 GPU-Red/GPU-Realloc/CPU-Slosh）在设备间分配功率；评估时初始功率上限是影响收敛与收益的最主要旋钮（500W 时波动大、收益低，550W CPU-Slosh 时吞吐 +6%）。
- Power Sloshing 补充视角（ISCA'26 推理场景）：TDP 在本文指“设备软件可配置功率上限”而非厂商峰值额定（论文明确区分）。compound AI inference server 的典型静态分配为 8 GPU×1kW + CPU 300W；静态独立 TDP 的缺陷是：CPU 密集服务（预处理/后处理重）CPU 触顶节流而 GPU 闲置、GPU 密集服务（长上下文 LLM）反之——fleet 数据（Fig.3）60% 服务利用率不足其功率上限的 80%，同服务器 GPU 间功率高度不平衡（60% 服务器有 20-40% 的 GPU TDP 闲置可收割，Fig.6）。模块级功率上限 P_M（< Σ组件 TDP）正是“power sloshing”的运行约束：功率受限模式下保证 P_C + ΣP_Gi ≤ P_M，通过频率缩放把闲置 TDP 转给受限组件；评估对不同 P_M 做 sweep（Fig.12），紧功率封顶下 Performance/Watt 收益最大（最高 1.83×）。论文同时在 GPU 上用频率上限（53%-100% f_GM）而非功率上限做实验，以隔离 vendor 内置电源管理。
- RHODES 的 TDP 建模视角（ISCA'26，设计早期碳感知 DSE）：RHODES 按 HILP [55] 的方法从 TDP 估计功耗——每个候选 CPU 配置（核数）与 GPU 配置（SM 数×频率）对应 active 功耗向量 P_c/P_g 与 idle 功耗向量 P_c,idle/P_g,idle，作为功耗约束（Eq.7）的输入：CPU active + GPU idle 场景（P_c^T·c+P_g,idle^T·g≤P_max）与 GPU active + CPU idle 场景（P_c,idle^T·c+P_g^T·g≤P_max）分别约束。功耗还经运营碳 C_op=CI_use·t_operational·P 进入 tC 约束（Eq.10），因此 TDP 取值直接耦合鲁棒设计的碳最优性与约束满足（nominal 与 robust 配置差异主要来自功耗-时间-碳权衡）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
