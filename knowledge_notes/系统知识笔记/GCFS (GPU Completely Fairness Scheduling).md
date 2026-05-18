## GCFS (GPU Completely Fairness Scheduling)

术语是什么？通过联网搜索让回答具体和精准。
GCFS (GPU Completely Fairness Scheduling) 是Infera针对normal（非实时）DNN inference任务设计的GPU调度算法，核心思想类比Linux CFS scheduler："Tasks execute the appropriate number of instructions according to their predefined priority"。每个scheduling cycle中，GCFS从gcfs_rq runqueues中选择多个任务组成VT，按nice value计算各任务的instruction budget，使不同优先级的任务按比例公平共享GPU执行时间。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GCFS在Infera中的运转流程：
1. **Runqueue组织**：64个runqueue中priority 40-63分配给gcfs_rq，数字越小优先级越高
2. **Task Selection**：每个调度周期初，GCFS从最高优先级非空gcfs_rq开始向下选择任务组成VT
3. **Budget Calculation**：按nice value为每个selected task计算instruction budget：
   - nice value越低（优先级越高）→ budget越大
   - 类似Linux CFS：budget ∝ weight / Σweights × total_cycle_budget
   - weight由nice value映射（nice 0→weight 1024, nice越高weight越小）
4. **Aging Mechanism**：长期未被选中的normal任务优先级逐渐提升（migrate到更高优先级gcfs_rq），防止starvation
5. **周期结束**：VTB中任务回Ready状态，下一周期重新计算budget

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera通过TSU中instruction-based budget tracking实现GCFS。与CPU CFS的vruntime不同，GCFS使用#inst作为fairness度量（因为GPU kernel执行时间与#inst正相关，而非固定）。TEU在每个调度周期内跟踪已执行指令数，通过GPU hardware performance counter或static kernel #inst分析估算。Aging mechanism将长期未选任务的priority值递减（数字变小=优先级升高），确保所有normal任务最终获得执行机会。当高优先级ddl/rt任务到达时，GCFS的VTB被抢占，任务上下文保存后切换。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
