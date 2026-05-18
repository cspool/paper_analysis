## Hazard-Aware Memory Orchestration

术语是什么？

Hazard-Aware Memory Orchestration是SLINFER中协调同一GPU node上多个model instance并发memory操作的机制：面对多个instance同时进行KV-cache scale-up/down和model loading/unloading的异步操作，通过optimistic budgeting（乐观更新可用memory budget）+ pessimistic scheduling（保守执行pending scale-up操作）的组合策略，既允许并行执行提高效率，又防止并发操作导致的OOM。uncoordinated scaling example：Ins-1 scale-up +20% + Ins-2 scale-up +10% → 总使用120%→OOM（Figure 18）。

从系统架构角度拆解术语：

Hazard-Aware Orchestration在SLINFER中的运转流程（Figure 19）：

1. **Optimistic Budget**：全局维护一个node-level memory budget。Scale-down请求到达时直接减少budget并立即发出scale-down操作——budget立即乐观更新但实际memory release在操作完成后才生效。

2. **Scale-up检查**：Scale-up请求到达时检查budget是否足够：(a) 足够→更新budget，发出操作；(b) 不足→拒绝，尝试compromise（降级为Mrequire而非Mrecommend）→仍不足→evict最长headroom请求。

3. **Pessimistic Reservation Station**：并行执行引入hazard——例如scale-up紧接在pending scale-down之后，若scale-up先于scale-down完成会导致OOM。为此，SLINFER维护pessimistic global memory tracking：pending scale-down的instance按其原始（较大）memory size记账。若pessimistic tracking显示OOM风险，scale-up操作不放行执行，而是放入reservation station等待。

4. **Reservation Station Wake-up**：当scale-down操作完成时通知reservation station→re-evaluate risk→若安全则执行pending scale-up操作。

5. **Error Handling**：极端情况下KV-cache underestimation→二次scale-up→若仍失败→evict最长headroom请求并reschedule。

术语一般如何实现？如何使用？

实现要点：
- **Budget管理**：per-node optimistic budget变量，atomic update保证一致性
- **异步操作追踪**：每个memory操作有唯一ID，记录expected memory delta和completion状态
- **与watermark scaling联动**：watermark决定scale-up/down的时机和目标size，orchestration决定这些操作何时实际执行
- **Pessimistic tracking**：维护一个"pre-operation" memory map，每instance记录其pre-scale-down的memory占用，用于OOM risk计算

涉及论文标题：
- Towards Resource-Efficient Serverless LLM Inference with SLINFER
