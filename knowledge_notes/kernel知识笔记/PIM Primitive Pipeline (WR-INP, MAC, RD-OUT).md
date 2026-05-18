## PIM Primitive Pipeline (WR-INP, MAC, RD-OUT)

术语是什么？通过联网搜索让回答具体和精准。

PIM Primitive Pipeline（WR-INP / MAC / RD-OUT）是PIM系统中执行向量-矩阵运算的基本命令流水线。WR-INP（Write Input）：将32B input tile从host/PIM HUB写入PIM channel内的Global Buffer (GBuf)指定entry；MAC（Multiply-Accumulate）：从GBuf读取input tile和从DRAM bank读取weight/KV cache data，执行dot-product（GEMV）并将结果累加到Output Buffer (OBuf)对应entry；RD-OUT（Read Output）：从OBuf读出累加完成的结果返回PIM HUB/host。这一三阶段pipeline是PIM computation的基本构建块，一次完整的GEMV需要多轮WR-INP→MAC→RD-OUT序列（取决于维度大小和tile size）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// PIM primitive pipeline执行（32B tile粒度，FP16 GEMV: q[1×4096] × K[4096×T]）:
// 每个MAC处理一个32B tile = 16个FP16元素

for tile_idx in 0..(4096/16 - 1):   // 256个tile
    // Phase 1: WR-INP
    //   将q[tile_idx*16 : (tile_idx+1)*16]写入GBuf指定entry
    //   耗时: tWR-INP (DRAM write-to-GBuf latency)
    WR_INP(GBuf[entry_a], q[tile_idx*16 : (tile_idx+1)*16])

    // Phase 2: MAC
    //   读取GBuf[entry_a]的input tile
    //   读取DRAM bank中K[tile_idx*16 : (tile_idx+1)*16, t]的对应16个weight
    //   执行16个乘法+adder tree → 累加到OBuf[result_entry]
    //   耗时: tMAC (bank read + multiply + accumulate latency)
    MAC(GBuf[entry_a], K_addr[tile_idx*16 : (tile_idx+1)*16], OBuf[result_entry])

    // Phase 3: RD-OUT
    //   当所有tile累加完成，从OBuf读出最终结果
    //   耗时: tRD-OUT (OBuf read-to-HUB latency)
RD_OUT(OBuf[result_entry], output)

// 静态调度问题:
// 传统controller按: WR-INP0→MAC0→RD-OUT→WR-INP1→MAC1→... 串行
// 即使WR-INP1与MAC0无依赖也不能提前发射
// DCS改进: WR-INP0→(WR-INP1可以并行)→MAC0(等GBuf0就绪)→(WR-INP2并行)→MAC1→...
```

PIM primitive的关键时序特性：(1) WR-INP和RD-OUT是I/O操作，占用PIM内部总线但不占用MAC unit；(2) MAC是计算操作，占用MAC unit但总线可用于其他channel的I/O；(3) Attention的GEMV因dh小（128）、tile数少（8个tile），I/O占比高，静态调度下MAC大量idle（论文中Attention MAC utilization低至14.7%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PIM primitive以ISR（Instruction Set Register）指令编码，由compiler生成指令序列，PIM controller解码执行。在SK hynix AiM架构中，对应指令为MAC_ABK/MAC_SBK（All/Single-Bank MAC）、WR_ABK/WR_SBK和RD_ABK/RD_SBK。PIMphony通过MLIR compiler自动将attention subgraph映射为PIM primitive sequences，DCS controller在runtime做dependency-aware issue。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

