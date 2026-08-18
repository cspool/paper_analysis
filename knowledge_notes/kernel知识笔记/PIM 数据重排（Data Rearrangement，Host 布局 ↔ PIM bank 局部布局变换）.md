## PIM 数据重排（Data Rearrangement，Host 布局 ↔ PIM bank 局部布局变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU-PIM 异构系统中，Host 与 PIM core 需要相反的数据布局，因此在 PIM kernel 执行前后必须做数据重排：Host 侧把连续元素跨 DRAM bank 分布（cache line 粒度访问、利用 bank/channel 级并行），PIM core 只能访问本地 bank、需要连续元素落在同一 bank 内以最大化本地带宽。由此 PIM kernel 执行固定为三步：①输入重排（把连续元素搬进各 PIM core 的本地 bank）→②PIM 计算→③输出重排（在 PIM core 间合并部分结果，或把连续元素重新跨 bank 分布以还给 Host 访问）。重排通常经 Host 内存总线完成（PIM 设备之外），开销巨大——DCC 动机实验中 TVM 式方案的数据重排占 kernel 端到端时间的 64.68%，是 PIM 编程与性能的最大痛点。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
核心矛盾（DCC §4.3 I）：(i) Host 侧顺序连续读利用 channel 并行，但写入时只能用一个 PIM channel → PIM 带宽利用不足；(ii) 多 PIM channel 并行写需要从 Host 内存同时读非连续地址 → Host channel 冲突、读性能下降。DCC 的解法（片上内存 staging）：
```
N = PIM 通道数; B = on_chip_memory_size / N   # 如 GPU shared memory 作 staging
for block in range(N):
    read(host_mem, block*B, on_chip[block])   # 串行 N 次 Host 读，读侧 channel 并行
for block in range(N):                         # 之后并行写
    parallel_write(PIM_channel[block], on_chip[block])  # N 路并行写 PIM
```
效果：数据搬移变成 N 次串行 Host 读 + N 路并行 PIM 写，两侧都拿到 channel 级并行。无可控片上内存的系统（如 CPU）退化为 PIM 后端默认 copy/DMA 接口；有对齐/交织等布局约束的后端再叠加额外 layout transformation pass。重排方向由 IR 的 `rearrange %t→%tt [to_PIM_core/to_host]` 显式编码，输入输出两条路径分别调度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器生成重排指令序列（LD/ST 或 DMA），与计算命令一起注入时序仿真器（DCC 用 Ramulator 2.0 的 LD/ST 命令仿真搬移成本）。使用：DCC 把重排成本纳入 draft 的端到端时间联合优化——同一 kernel 的不同数据分区对应不同重排成本，预测器按"重排+计算"总和选优；实测 DCC 同时优化计算（1.58×）与数据重排（1.65×），重排主导的 VA/RED/RELU 提速 1.67×、compute-heavy 的 GEMV/ATTN 仅 1.18×。对比：ATiM 式 compute-centric 流程先定计算模板再补重排，无法发现"稍差计算换便宜重排"的组合。

Taking Analytic Databases to the Bank 补充数据库视角（BLIMP OLAP relayout）：论文把该布局变换称为 relayout，并系统量化其代价与规避策略——(1) 代价：host 侧软件 relayout 平均吞吐 29GBps（vs 90GBps 峰值带宽）；把一个 64-bit 字写入特定 bank 需要 8 次内存写（读取 8 次读），因为标准 DDR 地址映射把字按字节 striping 到 8 个 chip；(2) 规避：提出 PIMDT（PIM Data Type）列式存储格式——把查询常用列以 PIM 友好布局常驻（整字单 bank），host 加载时无需查询时 relayout，只对少数"host 列"保留原布局；PIMDT 只支持定长类型（可变长字符串/blob 不兼容）且更新/插入需按字节重排；(3) 调度影响：查询规划必须最小化 relayout——end-to-end 评估显示隔离算子外推（每算子把输出 relayout 回 host）导致平均 22% 查询时间在 relayout、整体比 PIM-optimal 计划慢 3.2×；物化策略（Early/Hybrid/Late）决定何时发生 compute domain 转换与 relayout（bitvector 定长 vs value array 随选择性线性增长）；bushy join 树需多次重建哈希表（重复 relayout+build+broadcast），与最小化 relayout 目标相悖。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- Taking Analytic Databases to the Bank
