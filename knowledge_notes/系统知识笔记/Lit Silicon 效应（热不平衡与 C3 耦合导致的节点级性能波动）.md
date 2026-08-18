## Lit Silicon 效应（热不平衡与 C3 耦合导致的节点级性能波动）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Lit Silicon 是 ISCA'26 论文提出的命名现象：在多 GPU 节点内，热不平衡导致的热致掉队（straggler 更热更慢）与 C3（并发计算与通信重叠）耦合，形成负反馈循环，放大同构工作负载（如 FSDP LLM 训练）下的节点级性能波动。机制三步：① 热不平衡产生 leader（冷快）与 straggler（热慢）；② leader 提前开始通信但必须等 straggler 完成集合，同时计算流在全部 GPU 上独立推进；③ 等待 straggler 使 leader 的通信被拉长、C3 重叠延长、资源竞争加剧，反而拖慢 leader。迭代内 lead 值动态积累到 equilibrium，迭代末 leader 空等 straggler，下一迭代重复——成为系统固有瓶颈。
- 命名来源：silicon（芯片）+ lit（热/发光）——芯片被"点亮/发热"的含义。论文首次识别该负反馈，并证明其与 workload 无关（dense 与 MoE 训练都发生），同样适用于 AI 推理（vLLM 等）。已开源：https://github.com/UnaryLab/lit_silicon_tuning_amd（约 200 行 PyTorch 解决方案，不改 GPU kernel）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一次训练迭代内的四阶段，图 6）：① 所有 GPU 一起开始迭代（性能波动不显著）；② 性能波动随层累积——constant overlap kernel（0%/100%）上 leader 更快、lead 值增长；③ straggler 通信起始晚，leader 必须等待（三个蓝色块同刻结束），varying overlap kernel 上 leader 因 C3 资源竞争反而更慢，抵消 constant overlap 的领先，达到 equilibrium（lead 值 b,c,d 相同）；④ 迭代末 leader 先完成所有 kernel、等待 straggler 收尾，虚线处下一迭代重启①-④。
- 节点级后果：节点吞吐受 straggler 约束（baseline 运行时 = t_max(constant overlap) + t_min(varying overlap)），leader 的空等与 C3 竞争造成 5%-10% 性能损失与 4% 可省功率。检测用 lead value（Algorithm 1），缓解用节点级功率重分配（Algorithm 2/3 + GPU-Red/GPU-Realloc/CPU-Slosh），与 GPU 级、集群级功率管理正交，构成新的节点级功率管理层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：约 200 行 PyTorch 运行时层——(1) 用 Chopper 解析 PyTorch trace 获取每 GPU kernel 起始时间；(2) Algorithm 1 算聚合 lead 值（sum 默认）；(3) Algorithm 2 按归一化 lead 生成各 GPU 功率上限增量（max_inc 默认 15W，global scale 衰减）；(4) Algorithm 3 在节点功率上限与 TDP 约束下均匀回退；(5) amd-smi 设置功率上限，采样-调整直到收敛（约 20 样本/80 秒），之后可停用或长周期采样（三个月调两次即可）。使用场景：数据中心节点级功率管理（省电/提吞吐/CPU 功率转移）；部署需管理员权限（多租户集群可用固件在线调频或离线校准钩子替代）。成果：GPU-Red 省电 4%、GPU-Realloc 吞吐 +3%、CPU-Slosh 吞吐 +4%（550W 上限达 +6%），单客户（6GW AMD 部署）年省电费估算约 $70M。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
