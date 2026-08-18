## 逐 Miss 根因分类（Real-Time Component-Level Diagnosis，实时组件级逐事件根因诊断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逐 Miss 根因分类是论文演示的 IPU 第四大能力（Real-Time Component-Level Diagnosis）：在事件发生的时刻，通过订阅组件内部决策点信号，把每个微架构失败事件（如 L1D demand miss）当场分类到根因类别，输出"每 miss、每 PC 的失败模式分布"。背景逻辑链：聚合计数器（PMU）只报告"组件失败了多少次"，无法解释"为什么失败"；采样工具（如 PEBS）缺乏预取器内部状态可见性；两者都区分不了"cold region（冷区，无预取覆盖）"、"no learned pattern（预取器没学到模式）"、"late prefetch（预取太晚）"、"prefetch failure（预取被逐出/失败）"这四类根因——而它们指向完全不同的微架构投资方向（PHT 容量 vs 训练触发 vs 访存调度/lookahead 距离 vs cache 容量/替换策略/预取节流）。IPU 通过"tapping 组件内部决策点信号"解决：不注入、不采样，直接看预取器决策路径上的信号（AT hit、PHT 的 missed_in_pt、MSHR in-flight 等），把诊断带到部署硅片上的真实负载，无需重新 tape-out，且同一二进制可部署到 fleet 上发现生产环境与设计期基准的根因分布差异。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
逐 miss 根因分类 introspection kernel 的计算过程（HIT=L1D cache+prefetcher，6 信号=132 bits，20-cycle 处理窗口）：
```
// 每 demand miss 到达（LSQ 经地址总线送 miss 虚拟地址与 PC）
classify_miss(pc, dem, at_hit, missed_in_pt, mshr_inflight):
    # dem: demand miss 指示; at_hit: Accumulation Table hit
    # missed_in_pt: PHT 在 region 毕业时是否含模式
    # mshr_inflight: 预取 in-flight 状态
    if not at_hit:                 # AT miss → 冷区，预取从未学到此区域
        category = COLD_REGION
    elif missed_in_pt:             # AT hit 但 PHT 无模式
        category = NO_LEARNED_PATTERN
    elif mshr_inflight:            # 预取已发出但在途（太晚/未覆盖本次 miss）
        category = LATE_PREFETCH
    else:
        category = PREFETCH_FAILURE  # 预取过但被逐出/失败
    bucket = hash(pc)              # 按 PC 累积
    hist[bucket][category] += 1    # 4 类失败模式计数
    every(20 cycles): fifo_send(hist)   # 周期性输出紧凑直方图
```
丢数据建模：20-cycle 处理窗口内到达的新 miss 被丢弃（L1D miss 事件相对稀少：Gaze 激活后残留 miss 率低，间隔 <20 cycles 的连续 miss 罕见），平均每 trace 类别比例误差 <3.5%、中位 2.46%；输出为每 PC 四类计数直方图（384B/报告周期，主机流量可忽略）。诊断含义：x264_s 的 PREFETCH_FAILURE 占比高 → cache 压力逐出预取数据 → 需改 cache 容量/替换/节流；NO_LEARNED_PATTERN 主导 → PHT 容量不足或训练触发保守；LATE_PREFETCH 主导 → 访存调度延迟或 lookahead 距离问题；COLD_REGION/NO_LEARNED_PATTERN 的 PC 是显式预取 hint 的候选。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：IPU_lite 上 50 行 introspection 代码（scratchpad 读 + RISC-V 指令，行为与 PICS 演示相似，面积 0.019 mm²、功耗 15mW）；模拟验证用 ChampSim + Gaze 作者实现（189 条 SPEC traces），离线分析脚本按 IPU 20-cycle 处理延迟建模丢数据并分类每个 demand miss（Gaze 是 HPCA 2025 空间预取器，SJTU，Zenodo 数据 artifact https://zenodo.org/records/14252372 提供 traces）。使用方式：硬件设计者据失败模式分布决定微架构投资方向（PHT/训练触发/访存调度/lookahead/cache 替换/节流）；软件开发者把标记的 PC 作为显式预取 hint 候选；fleet 部署同一二进制对比生产 vs 设计期根因分布。局限：需要预取器暴露内部决策点信号（论文证明 Gaze 的 AT/PHT/MSHR 信号在合理实现中现成可得），分类粒度受信号可见性与 20-cycle 处理窗口约束。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
