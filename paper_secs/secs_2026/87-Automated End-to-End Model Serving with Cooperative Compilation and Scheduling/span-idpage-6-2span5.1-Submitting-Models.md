# <span id="page-6-2"></span>5.1 Submitting Models

Models should be submitted to the inference server before they can be inferred. Static libraries in CUDA binary format and DNN weights are registered into a high-performance data structure (e.g., [\[53\]](#page-15-14)) by the inference server. They are marked as cudaHostAllocWriteCombined for fast host-device transfer and cudaHostAllocPortable for multi-device memory consistency.

![](_page_7_Figure_2.jpeg)

Figure 8: Inference task state.

## <span id="page-7-0"></span>5.2 Dispatching Jobs

User applications start by submitting inference jobs to the inference server, after which JDU adds them to the end of the inference job queue. If the system becomes overloaded which potentially leads to a long job queue wait, JDU throttles user jobs to regulate the queuing delay. As long as the queue is non-empty, JDU continues to dequeue jobs and assigns them to the available GPUs with enough free space and the least estimated remaining time, as computed with Eq. (1). While jobs are initially assigned to a GPU, they can be migrated between GPUs periodically due to inactivity at inference, rejoining the runqueue with the same priority.

## <span id="page-7-1"></span>5.3 Scheduling Tasks

When a job is dispatched to a GPU, it becomes an inference task managed by the GPU's dedicated task scheduler.

**Task state.** The task scheduling system defines 5 states for inference tasks:

- New. The task has been created but not yet admitted by the task scheduler. While the task control block exists, corresponding data structures are not prepared.
- Blocked. The task is unable to proceed due to incomplete data preparation in GPU memory, whether from unallocated space or unfinished data transfer.
- **Running.** The task has been selected by the task scheduler for execution in the current scheduling cycle.
- **Ready.** The task is able to be selected by the task scheduler for execution but has not yet been.
- Exit. The task has been released, either upon completion or due to an abort.

A job dispatched to a GPU begins in the "New" state and undergoes legitimacy checks such as memory requirement check. Tasks that pass these checks are admitted otherwise get rejected and exit. Once a task completes and releases its resources, it transitions to the "Exit" state. The "Blocked" state is highly relevant to memory management which is discussed in the memory management part. The transition between "Ready" and "Running" is managed by the task scheduling algorithm. The key modules of the task scheduler are as follows.

**Memory management.** During the execution of DNNs, large-scale data movement can lead to severe GPU perfor-

```
Algorithm 1: Select and Execute Tasks

Input: the set of all tasks T

1 while True do
2 | VTB = GenerateVirtualTask(T)
```

<span id="page-7-7"></span><span id="page-7-6"></span><span id="page-7-5"></span><span id="page-7-4"></span> $K = FuseKernels(K_{set})$ LaunchKernel(K)

GPU global memory.

<span id="page-7-2"></span>mance decay as the data transfer is too slow compared to computation. Therefore, an inference task is permitted to run by the task scheduler only when all its data (i.e., weights, input tensors, and intermediate tensors) is already ready in

At creation, a task is not allowed to demand more memory than the GPU's available memory. Additional memory constraints can be set to reduce memory pressure and enable more concurrent tasks. Once a task is admitted, it is added to mm\_wq where it waits for memory allocation. The waiting queue is composed of all tasks whose data is partially or entirely swapped out of GPU memory with state "Blocked". The queue is prioritized by the priority of the tasks, with tasks of the same priority being ordered by their enqueue sequence. Tasks in the waiting queue are swapped into GPU memory from the head of the queue whenever GPU space is available. In addition, the system reclaims memory of models with LRU policy in the background. The memory management system uses 4 independent GPU queues CUstream to handle asynchronous memory allocation, free, and bidirectional transfer respectively. Once a task's data is fully swapped into GPU memory, it is removed from mm\_wq and added to the corresponding runqueue based on its property (e.g., gcfs\_rq for normal tasks).

Scheduling algorithm. A primary function of the task scheduling system is to select and execute tasks. However, designing an optimal GPU task selector and task executor is non-trivial for three reasons: (1) GPU cores cannot be finely controlled via standard CUDA API. The minimum dispatching granularity is the whole GPU. (2) Concurrent kernels on a spatial-shared GPU can cause severe competition and uncertain dynamic resource occupation. (3) GPU performance fluctuates over time and varies with different kernels, unlike the constant speed assumption in CPU scheduling.

To address these challenges, we introduce an instruction-based task selector and task executor with Virtual Task (VT) abstraction (Alg. 1). It operates as follows. First of all, the task selection process is organized as consecutive scheduling cycles. At the beginning of each scheduling cycle, the TSU selects a set of tasks  $VT = \{t_i \mid i \in \{1, 2, ..., N\}\}$  from

the runqueues, where is the -th task and is the total number of tasks (Al[g1:](#page-7-2)[L2\)](#page-7-3). The tasks in VT are then promoted from "Ready" to "Running". Every task in VT is assigned an instruction budget which is the number of instructions allowed to be executed during the scheduling cycle. VT with instruction budgets is denoted as Virtual Task with Budget (VTB). During the scheduling cycle, TEU executes VTB by selecting, fusing, and launching kernels based on the tasks in it (Al[g1:](#page-7-2)[L4\)](#page-7-4). At the end of the scheduling cycle, the execution is interrupted and VTB's tasks revert back to the "Ready" state. Note that the task selection is a fairness problem and the task execution is a performance problem. The remainder of this section covers the task selection (GenerateVirtualTask(T), Al[g1:](#page-7-2)[L2\)](#page-7-3), and the task execution (ExecuteVirtualTask(VTB), Al[g1:](#page-7-2)[L4\)](#page-7-4) is described in § [5.4.](#page-8-0)

To meet different inference requirements such as realtime or deadline, several scheduling policies are designed to manage DNN inference tasks on GPUs as follows. Note that these scheduling policies (e.g., SCHED\_NORMAL) are named by analogy to Linux CPU scheduler policies, for intuitive understanding only.

We implement priority scheduling in the task scheduling system. Inference tasks are categorized into deadline tasks, realtime tasks, and normal (non-real-time) tasks. Deadline tasks (scheduled with the SCHED\_DEADLINE policy) are associated with a specific deadline by which they should complete their execution; real-time tasks (scheduled with the SCHED\_FIFO policy) require prompt and predictable responses; normal tasks (scheduled with the SCHED\_NORMAL policy) are expected to be executed with best effort.

To support priority scheduling, the task scheduler sets up 64 task runqueues with priority 0–63, where smaller number indicates higher priority, as shown in Fig. [7.](#page-6-1) Every runqueue is associated with a scheduling policy and only determines the scheduling order of tasks in it. Number 0 is assigned to a ddl\_rq runqueue for deadline tasks, number 1-39 are assigned to rt\_rq runqueues for real-time tasks, and number 40–63 are assigned to gcfs\_rq runqueues for normal tasks. Tasks in higher-priority runqueues preempt those in lowerpriority runqueues at all times. Once a runqueue exhausts its tasks, the task selector switches to a lower-priority runqueue. Additionally, the task scheduler employs an aging mechanism [\[3\]](#page-14-27) to progressively elevate the priority of normal tasks that have remained unselected for a prolonged period, by migrating tasks to higher-priority gcfs\_rq runqueues.

We implement deadline task scheduling in the task scheduling system. SCHED\_DEADLINE schedules real-time tasks, whose core is the EDF algorithm [\[11\]](#page-14-28). Tasks can be periodic (e.g., autonomous driving) or non-periodic (e.g., chatbot). The task selector selects the task with the nearest deadline to run, which means the generated VTB has only one task with unlimited budget. To avoid deadline miss, the task scheduler

estimates jobs' execution time and verifies GPU bandwidth availability prior to admitting deadline jobs. This can be accomplished via static analysis or one-shot profile of kernels.

We implement real-time task scheduling in the task scheduling system. SCHED\_FIFO schedules real-time tasks, whose core is the FIFO algorithm [\[6\]](#page-14-29). Real-time tasks run sequentially according to their order in the real-time task runqueues. Similar to deadline task scheduling, the VTB has only one task with unlimited budget.

We implement normal task scheduling in the task scheduling system. SCHED\_NORMAL schedules normal tasks, whose core is an original GPU Completely Fairness Scheduling (GCFS) algorithm. First of all, we define the fairness of normal tasks as "Tasks execute the appropriate number of instructions according to their predefined priority". At the beginning of a scheduling cycle, tasks in cfs\_rq are chosen as the VT, with their instruction budgets calculated based on the predefined nice values, similar to the Linux CFS scheduler [\[6\]](#page-14-29).

Preemption. Preemption in GPU scheduling is critical for real-time guarantees of inference tasks. The task scheduler implements all scheduling in a preemptive manner where tasks with higher priorities than the current tasks always preempt resources and run.

The preemption involves two phases: saving the context and switching the context. The arrival of a higher-priority task triggers the preemption routine, which sends a predefined preemption signal to the task scheduler. After receiving the signal, the task scheduler suspends its scheduling and saves the current task scheduling state, while the task executor responds to the preemption signal according to § [5.4.](#page-9-0) They synchronize after handling preemption and work on the new task runqueue with higher priority.

Soundness. The task scheduler is sound because it is free of deadlock and starvation, the two major liveness problems.

Deadlock (also known as circular wait) cannot occur between different priority runqueues since low-priority tasks are prohibited from requesting resources while high-priority tasks remain. Within the same runqueue, resource requirements are guaranteed to be met by TSU at task creation.

Starvation is impossible in the task scheduling system. The task migration between GPUs and the task aging between runqueues of different priorities guarantee that inactive tasks eventually get a chance to run. Therefore, we only need to prove that starvation does not exist within a single runqueue. The SCHED\_DEADLINE policy guarantees all tasks complete before their deadlines; the SCHED\_FIFO policy prevents starvation by limiting each kernel's runtime; the SCHED\_NORMAL policy lets tasks fairly share every scheduling cycle for running.

