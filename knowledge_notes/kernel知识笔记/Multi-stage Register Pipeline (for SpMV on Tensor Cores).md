## Multi-stage Register Pipeline (for SpMV on Tensor Cores)

术语是什么？

Multi-stage Register Pipeline 是 Drawloom 为解决 SpMV 中 memory-compute 串行化导致的 warp stall 而提出的 GPU kernel 流水线设计。将 SpMV 的四大操作步骤——Fetch Sparse A (FS)、Fetch Column ID (FCid)、Load Vector X (LX)、TC Computation (Comp)——重构为五个流水线阶段：FillSMEM（async-copy 将 sparse A + CID 从 GMEM 异步拷贝到 SMEM）→ FillREG（从 SMEM 索引 prefetch vector X 到寄存器）→ Comp（TC MMA 计算）→ EmptySMEM（完成剩余 SMEM→REG 数据搬移）→ EmptyREG（完成剩余计算）。通过两个可调参数 delaySMEM 和 delayREG 控制流水线 overlap 深度：delaySMEM 控制 GMEM→SMEM 的 double buffering 阶段数，delayREG 控制 SMEM→REG + computation 的重叠阶段数。

从kernel调度角度拆解术语：

Multi-stage Register Pipeline 的伪代码执行流程：

```
// Pipeline参数
delaySMEM = 1  // double buffering: 2个SMEM buffer交替
delayREG = 2   // 3个REG set轮流使用

// 5阶段流水线
int stage = 0;

// 阶段1: FillSMEM — 填充SMEM buffer
while (stage < delaySMEM + 1):
    async_copy_g2s(FS[stage], &SMEM_A[stage % 2])  // sparse A
    async_copy_g2s(FCid[stage], &SMEM_CID[stage % 2])  // column IDs
    pipeline_commit()
    stage++

// 阶段2: FillREG — SMEM→REG + Load X
while (stage < delaySMEM + delayREG + 1):
    pipeline_wait_prior(stage - delaySMEM - 1)  // 等待SMEM就绪
    for tid in warp:
        cid = SMEM_CID[(stage-delaySMEM) % 2][tid]
        REG_X[stage % (delayREG+1)][tid] = LDG_instruction(X_base + cid)  // PTX LDG
    stage++

// 阶段3: Comp — TC Computation
while (stage < total_stages):
    pipeline_wait_prior(stage - delaySMEM - 1)
    for tid in warp:
        cid = SMEM_CID[(stage-delaySMEM) % 2][tid]
        REG_X[stage % (delayREG+1)][tid] = LDG_instruction(X_base + cid)
    
    // TC MMA: 使用前一阶段的REG数据
    TC_MMA(REG_A[stage-delaySMEM-delayREG], 
           REG_X[(stage-delayREG) % (delayREG+1)], 
           REG_C)
    stage++

// 阶段4-5: EmptySMEM & EmptyREG — 流水线排空
while pending_stages > 0:
    pipeline_wait_prior(...)
    TC_MMA(...)
    pending_stages--
```

2-stage pipeline（delaySMEM=1, delayREG=0）仅重叠GMEM→SMEM传输与后续操作；Multi-stage（delaySMEM+delayREG>1）进一步解耦SMEM→REG和计算，使多级寄存器集轮转消除 data dependency stall。

术语一般如何实现？如何使用？

实现依赖 A100+ 的 async-copy（`__pipeline_memcpy_async` + `__pipeline_wait_prior`）和 PTX LDG 指令（`ld.global.nc` 加载 X 向量绕过 L1 cache）。double buffering 通过两个 SMEM buffer 交替使用实现。REG prefetch 通过 delayREG 控制的多个寄存器集合轮转实现。在 Drawloom 的消融实验中，v4（+Multi-stage Pipeline）相比 v3 平均提速 1.46×（mip1 达 5.68×），warp stall 改善 3.02×-3.13×（在 majority representative matrices），memory throughput 提升 2.61×-2.75×。Pipeline 设计的关键限制是 SpMV 每次 memory access 只触发一个 TC 操作（与 SpMM 的大 tile 多 TC 操作不同），因此需要多级 REG pipeline 进一步重叠 short-latency TC computation 与 irregular X vector access。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

