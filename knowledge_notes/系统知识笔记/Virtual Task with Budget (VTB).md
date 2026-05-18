## Virtual Task with Budget (VTB)

术语是什么？通过联网搜索让回答具体和精准。
Virtual Task with Budget (VTB)是Infera task scheduler的核心抽象：在每个scheduling cycle开始时，TSU从各runqueue中选择一组任务VT={t_i}，并为每个任务分配instruction budget（本调度周期内允许执行的最大指令数）。VTB将任务选择（fairness问题）和任务执行（performance问题）解耦：TSU负责生成VTB，TEU负责执行VTB（SelectKernels→FuseKernels→LaunchKernel）。调度周期结束时执行被中断，VTB中的任务回到Ready状态。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
VTB在Infera中的运转流程：
```
// Algorithm 1: Select and Execute Tasks
while True:
    VTB = GenerateVirtualTask(T)  // §5.3 TSU
    while True:                    // §5.4 TEU
        ExecuteVirtualTask(VTB)
        K_set = SelectKernels(VTB)
        K = FuseKernels(K_set)
        LaunchKernel(K)
```
1. **GenerateVirtualTask(T)**：TSU根据调度策略选择任务并分配instruction budget：
   - SCHED_DEADLINE (ddl_rq)：EDF算法，VTB仅含一个任务，unlimited budget
   - SCHED_FIFO (rt_rq)：FIFO算法，VTB仅含一个任务，unlimited budget
   - SCHED_NORMAL (gcfs_rq)：GCFS算法，VTB含多个任务，每个任务按nice value分配instruction budget
2. **ExecuteVirtualTask(VTB)**：TEU在调度周期内循环执行SelectKernels→FuseKernels→LaunchKernel，直到所有任务耗尽instruction budget或无可执行data block
3. **调度周期结束**：执行被抢占，VTB中任务回Ready状态，下一个周期重新GenerateVirtualTask

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera实现中，VTB是TSU和TEU之间的接口契约。TSU在64-priority runqueue系统中按策略选取任务并计算budget（基于#inst估算），TEU在每个调度周期内跟踪已执行指令数并与budget比较。instruction budget机制使normal任务可以公平共享GPU：GCFS按nice value比例分配budget，类似Linux CFS的vruntime机制。deadline/real-time任务的unlimited budget确保紧急任务不被中断，但高优先级任务始终可抢占低优先级任务的VTB。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
