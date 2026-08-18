## Performance Signature Vector（PSV，性能签名向量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSV（Performance Signature Vector）是 TEA/IPU 的 PICS 生成中用到的核心数据结构：一个 bit-mask，表示"某个 PC 的某次动态指令实例上，发生了哪些性能事件"（如 DTLB miss、DCache miss、LLC miss、Drain-SQ full、Misspeculation 等事件的组合位图）。背景逻辑链：要生成逐指令周期栈（PICS），需要知道每个动态指令实例把时间花在哪些事件组合上；直接用 (PC, 事件组合) 的展开表存储爆炸，用 PSV 压缩为"每位一个事件"的紧凑位向量（TEA 用约 9 个事件、IPU 的 PICS 演示用 17 个信号覆盖事件+PC 控制），每个 PC 的动态实例维护一份 PSV。TEA 论文的设计值为每 400,000 cycles 采样/归并一次 PICS。论文的 IPU 版：IPU_lite 顺序核每 cycle 更新 PSV，事件发生时经 load-modify-store 序列把对应事件位置 '1'，flush 时把 PC 存进 IPU 内存供 commit 后引用；每 400,000 cycles 扫描活动 PSV 列表，把 (PC + 事件签名) 归并成 PICS 条目经 FIFO 发主机。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IPU 的 PICS introspection kernel 中 PSV 更新的计算过程（215 bits/cycle 输入，17 个信号）：
```
// 每 cycle（数据驱动：新 IORegs 数据到达即运行 _main）
loop:
  if itlb_miss(io_reg_x0):   # 事件检测：if-else-if 事件链
      psv = load(psv_table[pc])
      psv.bit0 = 1           # load-modify-store 置位
      store(psv_table[pc], psv)
  if icache_miss(io_reg_x1):
      psv = load(psv_table[pc])
      psv.bit1 = 1
      store(psv_table[pc], psv)
  ...                        # 其余事件（17 信号逐一检测）
  if flush_occurred(pc):     # flush 时保存 PC 供 commit 后引用
      store(flush_pc, pc)
  if cycle_count == 400000:  # TEA 设计值，定期归并
      for each active psv in psv_table:      # 扫描活动 PSV 列表
          fifo_send(PC + signature(psv))     # 归并成 PICS 条目发主机
      cycle_count = 0
```
事件输入到达率分析：215 bits 中 >75% 的 cycle 无事件（ROB 采样时未 stall/drain），~25% 出现单个长延迟事件，罕见 2-3 个事件；ROB 采样之外通常每 cycle 1-3 个事件触发 3-9 条指令执行。近似误差来源：事件在上一事件 PSV 生成窗口内到达会被丢弃（IORegs 保持旧数据丢弃新数据）——单 cycle 模拟 vs 每 cycle 模拟的平均相对误差 <3%（3 个应用 10-14%），PC 排序始终正确，丢 PC 覆盖 ≤0.37% cycles。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TEA 硬件版在 BOOM 核内为每个 in-flight 指令维护 PSV（约 9 事件、249B 存储、3.2mW、1.1% 性能开销、2.1% 平均误差）；IPU 软件版把 PSV 表放 32KB scratchpad，事件检测与置位用 RISC-V load-modify-store 序列（直方图/hash 指令作 intrinsics 优化）。使用方式：PSV 是"时间比例归因"的数据载体——主机后处理把每 PC 的 PSV 周期计数汇总成 PICS 栈（表 II 格式：PC 0x7912d0 DTLB miss+DCache Miss=50000000 cycles），开发者据此优化热点指令；IPU 版无需 BOOM RTL 改动、事件集合可软件扩展（对比 TEA 固定 9 事件）。验证：3 个 DARCHR microbenchmark（https://github.com/darchr/microbench）各只期望一个 PC 入栈，结果吻合。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
