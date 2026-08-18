## 预取质量指标（Accuracy / Coverage / Timeliness）与预取激进性（Prefetch Aggressiveness）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 预取器质量用三个支柱指标衡量：Accuracy（准确率=有用预取数/总预取数）、Coverage（覆盖率=被预取消除的 miss/总 miss）、Timeliness（及时性=在 demand 访问前到达的预取比例）。激进性（aggressiveness）指预取的程度/数量（预取度 degree、距离 distance）。三指标有内在权衡：更高覆盖通常牺牲准确（更多无用预取→缓存污染/带宽浪费）；Moirai 的定位是"平衡画像"：覆盖率 18.18%（对紧凑模型可观）、准确率 43.63%（足够净正收益）、及时性 92.37%（最佳，因其低延迟架构把预测转成有用访存）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Moirai 的置信度三档激进性策略（Adaptive Control Unit，推理期按平均 loss L_avg 分级）：高置信（L_avg 低于低阈值 Linf）→ 只发主预测 delta 的 1 个预取（保准确）；中置信 → 发 5 个（D_pred±2）；低置信（L_avg 高但 <Ltrain）→ 发 9 个（D_pred±4，保覆盖）。该 intra-inference 分级与相位节流（training 阶段暂停全部 CaPNet 预取）互补：loss 越高预取越多直到越过 Ltrain 触发重训练。多核争用下 loss 上升自动降激进性→等效于 Pythia 的显式带宽感知 agent，但零额外成本。
- 相位控制（Figure 9 双指标策略）：训练→推理需 L_avg<Linf 且 counter>Ninf(=2048)，或训练超时；推理→训练需 L_avg>Ltrain(=0.5) 或 counter>Ntrain(=131072)。训练相位暂停预取防污染，辅助 stride 预取器兜底（消融显示辅助器贡献 <0.1%，但防冷启动/相位切换的性能悬崖）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件计数器/寄存器跟踪 L_avg、访问计数、阈值（Moirai Table III：N 32-bit、阈值 64×2、L_avg 32-bit 等，共约 0.27KB 核心状态）；预取度由组合逻辑按阈值分档输出。评估：ChampSim 统计每预取的 timely/early/late/eviction 分类、accuracy/coverage/timeliness 算术平均、L1-L2/L2-LLC/LLC-DRAM 三边界 traffic（demand/prefetch 分列）。使用场景：任何预取器评估与调参；Moirai 用它展示"不追求单项最优、以最低硬件成本给出净正性能"的设计点。

LIBRA 补充视角（ISCA'26，多 GPU 页面预取的 accuracy/coverage）：论文定义 prefetch accuracy=被预取页中被目的 GPU 实际访问的比例，prefetch coverage=访问页中已被预取且访问时位于本地内存的比例（页粒度，是 cache-line 粒度指标的粒度上移）。关键洞察：空间局部性预取器本质以 accuracy 换 coverage（TBNP 系 accuracy 33.8%–47.4%、coverage 42.2%–48.9%、预取页数 21333–28548），多 GPU 下独立 GPU 冗余预取同页进一步拉低两者；LIBRA 用 stride-based 动态深度预取达到 accuracy 81.8%、coverage 83.9%、平均预取页数 19967（baseline 页故障均值 24007）。FFT 等大 stride 负载 coverage 95%（TBNP 系仅约 12%）；irregular benchmark 子集 accuracy 79%、coverage 82%（TBNP 系 accuracy 40%）。预取阈值敏感性：每 +25% 概率多预取 1 页（25% 阈值平均最优）。

R-Max 补充视角（ISCA'26，Oracle 预取器质量画像）：R-Max 的指标画像——L2 上 prefetch coverage 93.2%–97.7%（compute fp 93.2%、compute int 95.6%、srv 97.7%、SPEC CPU2017 94.1%，对比 SPP 仅 13.7%–41.9%）、prefetch accuracy 95.75%–99.99%（各配置，因 oracle 几乎全对，少量 useless 来自访存重排）、timeliness 分布范围与真实预取器相近但平均更高（Fig.10，上界约 10⁸ cycles，预取过早反而有害）、DRAM utilization 相对无预取 LRU baseline 降 47.93%（对比 SPP 反而 +21.43%，因错误预取污染）。剩余 miss 归因：带宽/延迟/容量导致的 late prefetch（MSHR merge 或 prefetch 未跑在 demand 之前）与访存重排导致的 dropped prefetch。R-Max 用公式 accuracy=useful/(useful+useless) 计算（H.-Notes.md 明确给出）。
STEP 补充视角（ISCA'26，L2 空间足迹预取指标定义）：STEP 定义 Accuracy = N_useful/(N_useful+N_useless)（被 demand 消费的预取比例），Coverage = (N_miss^base − N_miss^pf)/N_miss^base（消除的 baseline demand miss 比例），Overprediction = 无用预取 + 重复预取（目标行已在缓存）相对 baseline demand miss 的百分比，Speedup = 预取/无预取 IPC 比。结果：STEP accuracy 74%（vBerti 89% 最高但 coverage 最低、eBingo 73%、Gaze 67%）、coverage 51%（最高，eBingo 50%），平衡 accuracy-coverage 画像对应最高 speedup。Prefetch Composition 分析（Fig.7）：STEP 把更大比例预取活动转为有用覆盖、控制浪费——CloudSuite 下发的预取更少但更有用/及时，SPEC06 更多来自 FOE 早触发（提 coverage/timeliness 但略升 overprediction）。多核/受限 way（1 way）/存储敏感性（32 KB 后收益递减）均保持领先。


TTP 补充视角（ISCA'26，RT 预取的 accuracy/coverage/efficiency）：TTP 定义 accuracy=被 demand load 实际访问的预取块比例（L1 98.92%、L2 89.81%），coverage=预取消除的 baseline miss 比例（L1 31.54%、L2 33.46%），prefetcher efficiency=miss 于 cache+MSHR 且有 MSHR 可用的预取比例（L1 58.56%、L2 64.85%）；speedup 与 coverage 强相关。MPKI 变化：DFS 下 L1/L2 RT read miss 降 28.28%/40.01%；BFS+N=4 下 L1/L2 降 44.10%/92.04%。limit study：perfect upward traversal 1.79x vs perfect downward 1.35x，与 pop streak 表征一致。
涉及论文标题：
- R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
- LIBRA: A High-Accuracy, Cost-Aware, and Coordinated Multi-GPU Page Prefetcher
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing

RoboCortex 视角（ISCA'26，RSU 引导的语义预取）：对指针链接的树/图等不规则数据结构，空间（stride）与时间（重复序列）预取器都失效——树节点无空间连续性、搜索控制流依赖数据语义。RoboCortex 让 RSU 在执行 DFS 时前瞻读取当前访问节点的全部子节点指针，在控制流决策定型前把潜在下一跳地址推入预取队列，再由预取器按 Algorithm 1 决定是否发起语义预取：对当前访问地址 MemReq.Addr，计算到左右子节点地址的偏移 LOffset/ROffset，若偏移绝对值 ≥ PageSize（跨页）则直接预取该子节点；两个子节点都不跨页则回退 stride 预取器。Octo-Tree 等多子节点结构取"上前左 + 下后右"两个代表子节点适配该二元判定。效果（Fig. 15）：预取后 L3 miss 占比显著下降、L1/L2 hit 上升（Kd-Tree 最明显）；Octo-Tree 收益有限因其树构建保证地址连续性、本身 L1/L2 hit 已最高。局限：属于"程序语义已知"的辅助预取，依赖 RSU 提供的下一跳信息，不能脱离加速器独立工作。
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
