## 节点级功率上限管理（Node-level Power Capping：GPU-Red / GPU-Realloc / CPU-Slosh）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 节点级功率上限管理是在节点内按 GPU/CPU 粒度分配功率预算的系统策略：以 GPU 功率上限（power cap，用 amd-smi 设置，比直接锁频更可预测）为杠杆，纠正热致掉队。Lit Silicon 论文（ISCA'26）定义三种用例：GPU-Red（无节点功率上限时，只降 leader 功率 → 节点/平均 GPU 功率降、吞吐不变）；GPU-Realloc（有节点上限时，把 leader 的功率重分配给 straggler → 节点功率不变、吞吐升）；CPU-Slosh（有节点上限时，把闲置 CPU 的功率预算转移给 GPU → 节点功率不变、平均 GPU 功率升、吞吐升）。训练时仅约 13.5% CPU 核心被利用，约 86.5% 核心功率（数百瓦）可转移。
- 算法栈：Algorithm 1 检测（lead value）→ Algorithm 2 计算每 GPU 功率上限增量（按相对 lead 与历史最大 lead 归一化，global scale 保证收敛后期调整幅度衰减）→ Algorithm 3 在节点上限 + TDP 双约束下均匀回退（先按 Algorithm 2 增量，超节点上限部分均匀分摊，再按 TDP 修正）。示例（单 straggler + 7 leader）：GPU-Red 下 leader 各 -15W、straggler 保持 TDP；GPU-Realloc 下节点上限比 TDP 总和低 120W（每 GPU 低 15W），straggler +15W 后全部 GPU 均匀 -15W/8；CPU-Slosh 下若每 GPU CPU 预算 ≥2W 则用 CPU 的 16W 覆盖增量、leader 不动。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（GPU-Realloc 例）：profiler 每 10 迭代采样一次 → Chopper 解析 kernel 时间戳 → Algorithm 1 算聚合 lead → Algorithm 2 给 straggler 功率上限 +15W → Algorithm 3 检查节点上限并均匀回退（全部 GPU -15W/8）→ amd-smi 逐 GPU 设置功率上限 → 下一采样周期观察 lead 是否下降 → 收敛后功率分布固定（可跨框架/模型/上限复用）。效果：straggler 频率提升、leader 频率微降，跨 GPU 频率拉齐后 lead 归零、C3 不再拖慢 leader；GPU-Realloc 吞吐 +3% 且节点功率不变，GPU-Red 平均功率 -4% 且吞吐不变，CPU-Slosh 吞吐 +4%（550W 上限 +6%）但 GPU 功率 +3%。敏感性：初始功率上限是最重要旋钮（500W 波动大），warm-up 不影响收敛，节点 0（更多 straggler）收益略低，收益随功率重分配递减（表 III 模型预测吞吐趋势一致、功率误差 ≤1%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时 Python/PyTorch 层（约 200 行）+ amd-smi 系统接口（需 root）；开源 https://github.com/UnaryLab/lit_silicon_tuning_amd（含 power_server.py/protobuf 功率服务、容器定义 primus.def、submit_all.sh 一键作业流水线）。使用方式：作为新节点级功率管理层部署在 GPU 级与集群级功率管理之间；收敛后功率分布长期稳定，可低频（周/月）重校准；对 MoE（DeepSeek V3 16B，Primus+torchtitan 8 路专家并行）同样有效（lead 小但有 spike，仍收敛）。多租户集群无管理员权限时改用固件在线调频（GPU 遥测同步）或离线校准钩子。论文测量：三个月调两次（3.5%/4% 节电），每样本约 4 秒、约 80 秒收敛。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
