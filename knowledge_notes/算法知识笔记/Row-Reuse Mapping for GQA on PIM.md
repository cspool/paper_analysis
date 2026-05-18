## Row-Reuse Mapping for GQA on PIM

术语是什么？通过联网搜索让回答具体和精准。

Row-Reuse Mapping是PIM上执行GQA（Grouped Query Attention）时的优化策略：利用GQA中多个query heads共享同一组K/V cache的特性，优先在当前已打开的DRAM row上处理所有共享该KV row的query heads，减少DRAM row activation（ACT）和precharge（PRE）开销。在DRAM-based PIM中，读取不同row需要先precharge当前row再activate新row（ACT/PRE开销显著），row-reuse通过在同一open row上连续服务多个query head来摊薄此开销。但row-reuse引入了额外的WR-INP压力——需要为不同query head反复写入不同的GBuf entry，在静态PIM scheduling下这些input transfer stalls可能抵消row-reuse的收益。PIMphony的DCS通过dual-port GBuf/OBuf和entry-level dependency tracking在MAC消费当前GBuf entry时预取下一批query，将row-reuse的KV复用转化为真实吞吐收益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// GQA Row-Reuse Mapping on PIM（group_size=4, 4 query heads share 1 KV pair）:

// 无row-reuse（逐个head处理）:
for head h in 0..3:  // 4 query heads share same K/V
    ACT(row_K_h)     // 为每个head单独activate K row
    for t in 0..T-1:
        MAC(GBuf[q_h], K[t], OBuf[h][t])
    PRE(row_K_h)
// 总计: 4×ACT + 4×PRE overhead

// 有row-reuse（所有heads在同一open row上处理）:
ACT(row_K_shared)    // 一次activate, 4个head共享
WR_INP(GBuf[0], q_h0)  // 写入第一个head的query
for h in 0..3:
    if h > 0:
        WR_INP(GBuf[h], q_h)  // DCS可预取: GBuf[h]写入与GBuf[h-1]消费并行
    for t in 0..T-1:
        MAC(GBuf[h], K[t], OBuf[h][t])  // 所有h读同一open row
PRE(row_K_shared)
// 总计: 1×ACT + 1×PRE, 但4×WR-INP串行化（无DCS时）

// DCS优化: dual-port GBuf
//   port A: MAC读取GBuf[0] → 处理q_h0
//   port B: 同时WR-INP写入GBuf[1] → 准备q_h1
// → WR-INP latency被MAC重叠，消除input transfer stalls
```

PIMphony论文中，DCS配合row-reuse在GQA 128K模型上取得比non-GQA模型更大的收益（up to 11.3× speedup vs CENT），因为DCS将row-reuse减少的ACT/PRE overhead转化为真实吞吐而不会被WR-INP stalls抵消。对比ping-pong buffering baseline：ping-pong因需等两个region均idle才能切换（hand-off pipeline stalls），row-reuse的WR-INP压力导致更频繁的hand-off，DCS同buffer size下up to 1.4× higher compute-unit utilization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Row-reuse mapping由compiler或runtime的attention scheduler实现：识别GQA group structure→将共享同一KV pair的query heads聚合到同一PIM channel→调度MAC序列使同一DRAM row上的连续MAC操作最大化。PIMphony的MLIR compiler在pattern-matching阶段识别GQA config（group size g），在code generation阶段为共享KV的heads生成row-reuse optimized MAC序列并嵌入DCS dependency annotations使WR-INP与MAC overlap。GQA group size越大（如g=8），row-reuse收益越大，但WR-INP压力也越大——DCS的overlap能力在此trade-off中起关键作用。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

