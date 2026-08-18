## UVM 预取策略（Prefetching：TBP / FDP / RGP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPU UVM 中把"未来需要的页/区域"提前从 DRAM 迁到 HBM 以隐藏迁移延迟的运行时策略。默认 NVIDIA UVM 用 Tree-Based Prefetching（TBP）：每个 2MB 区域对应一棵完全二叉树（叶=64KB 页、根=2MB、内部节点=幂等大小子区域），每节点记录其子树内 HBM-resident 内存占比；页错误时若某祖先节点的驻留占比超过阈值（默认 51%）则预取该子树全部页。TBP 不自适应：固定阈值、无"预取是否有用"反馈，低局部性应用白耗 PCIe 带宽、高局部性应用错过整 2MB 预取。ObservUVM 用 observability 反馈设计两种策略：
  - FDP（Feedback-Driven Prefetching）：默认 TBP（阈值 51%），采样部分被预取区域做 observable，统计 access counter 通知（有用）与换出（无用）；>80% 被访问则把阈值降到 1（Aggressive Prefetching，整 2MB 一次迁入），否则维持保守 TBP（Algorithm 2）。
  - RGP（Region-Grain Prefetching）：类似 next-line 缓存预取器，检测分配区域内连续 fault 地址形成的流式访问模式；对已预取区域抽一个 64KB 触发页 T 做 observable，触发页被访问时预取下一个未预取的 2MB 区域（跨 2MB 边界），保证"不过早也不晚"的预取时机。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 ObservUVM 运行时中的流程（FDP，Algorithm 2）：PrefetchThreshold:=51；每收到反馈（AC=access counter 通知，EV=eviction）对应 CountAC/CountEV++；CountSum>T 时若 CountAC/CountSum>0.8 → PrefetchThreshold=1（切到 AP 整 2MB 预取），否则保持 51。RGP 流程：跟踪 fault 地址流 → 检测相邻页连续 fault → 判定流式访问 → 每个已预取区域设 trigger 采样页 → GPU 访问 trigger（access counter 通知）→ 驱动 setPrefetchRegion 预取下一 2MB 区域到 HBM。
- 效果：FDP 使除 GMM 外全部应用选择 AP（高空间局部性），TM+ 较 TM 页错误再降 36%（最高 84%）、时间降 8%（最高 15%）；RGP 再降 9% 页错误、+5% 提速；TM++ 较 UVM 页错误平均降约 78%。预取吞吐机会论据：论文测出 2MB 区域在换出时平均驻留占用率 >90%，说明"等 fault 流攒信心"的 TBP 过于保守。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 userspace 策略，intra-2MB 预取继承 ShallowPrefetch 基类、inter-2MB 预取继承 DeepPrefetch 基类；驱动提供 setPrefetchThreshold(int) 与 setPrefetchRegion(address) 下行接口执行预取。使用：在 ObservUVM 中注册 FDP/RGP（默认策略），运行 run_key.sh 复现 fig9/fig12/fig13。泛化：与 CPU 硬件预取器（next-line、stride）及 MoE 专家预取（knowledge_notes/系统知识笔记/Expert Prefetching（专家预取）.md，score 498）同属"预测未来访问 + 提前搬运隐藏延迟"范式，但 UVM 预取的对象是 2MB/64KB 页、信号来自 access counter 反馈而非网络/模型预测。

涉及论文标题：
- Observability-aided GPU Memory Oversubscription
