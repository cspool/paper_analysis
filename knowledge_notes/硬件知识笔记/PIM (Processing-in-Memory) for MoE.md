## PIM (Processing-in-Memory) for MoE

术语解释
存内计算（PIM）是将计算逻辑集成到内存芯片内部的架构方案，在MoE推理中使用PIM处理计算密度低但内存访问密集的expert操作，与GPU形成互补。

术语是什么？
Duplex是代表性的MoE PIM方案：
- 结合xPU（传统计算单元，如GPU）和Logic PIM（存内逻辑处理单元）
- 两种处理单元共享设备内存，通过高带宽TSV（Through-Silicon Via）实现DRAM die和Logic die间通信
- 关键洞察：不同操作有不同的Op/B（每字节操作数）特性：
  - Attention：高Op/B → 适合xPU（计算密集型）
  - Expert FFN：低Op/B → 适合PIM（内存密集型，需要高带宽低延迟访问expert权重）
- 为每层动态选择最优执行目的地（xPU vs Logic PIM）
- Expert和Attention阶段可并行执行以最大化吞吐

从硬件架构角度拆解术语。
Duplex的层执行目的地选择逻辑：
```
for each transformer layer l:
    # Op/B分析
    attn_op_per_byte = FLOPs(attention) / data_moved(attention)
    ffn_op_per_byte = FLOPs(expert_ffn) / data_moved(expert_ffn)
    
    # 目的地选择
    if attn_op_per_byte > threshold:
        attention.execute_on(xPU)     # 计算密集，GPU更高效
    else:
        attention.execute_on(PIM)     # 内存密集，PIM更高效
    
    if ffn_op_per_byte > threshold:
        expert_ffn.execute_on(xPU)
    else:
        expert_ffn.execute_on(PIM)    # Expert通常走PIM路径
    
    # 并行调度
    launch_async(attention, xPU)
    launch_async(expert_ffn, PIM)
    wait_all()
    # 合并结果
```

PIM微架构关键设计（Duplex）：
- Logic die上强大的处理单元 + DRAM die上的存储阵列
- 通过大量TSV实现高带宽die间通信
- 优化低Op/B操作（如expert FFN中的矩阵向量乘法）

术语一般如何实现？如何使用？
- PIM芯片：Samsung HBM-PIM、SK Hynix AiM（将计算单元放在HBM堆栈中）
- 互连：TSV提供比传统总线高得多的带宽
- 编程模型：通常需要offload API将特定操作标记为PIM可执行
- 编译支持：需要编译器自动识别低Op/B操作并调度到PIM
- 适用于：大规模MoE推理（expert分布存储，PIM可在内存侧高效计算）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models

---
