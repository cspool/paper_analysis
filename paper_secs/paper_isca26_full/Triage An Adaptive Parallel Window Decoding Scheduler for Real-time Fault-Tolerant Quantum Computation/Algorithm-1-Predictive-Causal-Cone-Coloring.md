# Algorithm 1 Predictive Causal Cone Coloring

```
1: Input: Causal cone slice set C, current time t_{now}, decoder
    model D_{model}
 2: Output: An emergency plan P
 3: Initialize plan P \leftarrow \emptyset
 4: Initialize priority queue Q
 5: for slice s \in C do
         s.t_{start} \leftarrow \max(t_{now}, s.t_{syndrome\_ready})
         Push s to Q, prioritized by s.t_{start}
 7:
 8: end for
 9: while Q is not empty do
10:
         t_{sim} \leftarrow \text{NextEvent}(Q, D_{model})
         R \leftarrow \text{all slices from } Q \text{ where } s.t_{start} \leq t_{sim}
11:
         Sort R by degree
12:
         N_{free} \leftarrow D_{model}.num\_free(t_{sim})
13:
         D_{dispatch} \leftarrow \text{SelectConflictFree}(R, N_{free})
14:
         for slice s \in D_{dispatch} do
15:
16:
              Add (t_{sim}, s) to P
              t_{fin} \leftarrow t_{sim} + \text{CalculateDuration}(s.\text{degree})
17:
              for neighbor n \in Q of s do
18:
                  n.t_{start} \leftarrow \max(n.t_{start}, t_{finish})
19:
20:
                  n.\text{degree} \leftarrow n.\text{degree} - 1
21:
                  Update position of n in Q
22:
         end for
23:
         Re-insert non-dispatched slices from R back into Q
24:
25: end while
26: return P
```

neighbor updates, with each priority queue position update taking  $O(\log n)$ . Sorting the ready set R (Line 10) is also bounded by  $O(n\log n)$ . Thus, the overall worst-case complexity scales efficiently at  $O(n\log n)$ . We will empirically validate this overhead in Section V-E.

3) The Triage Trigger: The Triage Scheduler's adaptivity and efficiency is governed by the Triage trigger, the mechanism that decides precisely when and how to transition to the emergency mode. The trigger is activated whenever any PENDING slice's deadline reaches a predefined threshold  $\tau_{emergency}$  (e.g.,  $\tau_{emergency}=4$ ). To prevent the scheduling complexity from causing latency spikes on exceptionally large causal cones (which often accumulate near the end of highly entangled applications), we enforce a strict ScopeCap< 100. If an evaluated causal cone exceeds this size limit, the scheduler falls back to the steady mode.

To avoid thrashing, Triage re-plans only when all expansiondriven conditions hold:

- The set of urgent slices introduces a causal cone that is not fully contained within the currently emergency scope.
- The expansion is significant, exceeding a defined fraction (e.g., 30%) of the existing scope's size.
- A minimum time interval (e.g., 2) has passed since the last re-plan.

When expansion-driven conditions are met and there is overlap between the two scopes, the scheduler performs an

![](_page_7_Figure_0.jpeg)

Fig. 10. A 2-D snapshot of the Triage Trigger's operation. At T=k+1, a new critical slice  $C_2$  triggers a scope expansion.

incremental update. Figure 10 provides a 2-D simplified snapshot of this process. The scheduler operates on an evolving dependency graph. At time T=k, an emergency plan for a critical slice  $C_1$  is already active. As time goes to T=k+1, a new critical slice  $C_2$  becomes urgent, triggering a reevaluation. The new emergency scope for  $C_2$  excludes already COMPLETED slices, and the incremental planner will take into account the blocking effect of  $C_1$ 's plan on future slices.

Critical-path impact of scheduling. The part of Triage that can introduce noticeable latency overhead is emergency-mode causal-cone planning. Triage does not assume that every scheduling computation stalls the quantum processor. An emergency plan is cached and subsequently executed as a lightweight dispatch table. Therefore, only the portion of planning, dispatch, or interconnect latency that cannot be hidden behind ongoing decoding can affect the critical path. In Section V-E, we conservatively model this unhidden latency by delaying task start times.

4) Throughput Maximization via Opportunistic Backfilling: While the emergency mode is latency-optimal, its resource utilization can be inefficient. The parallelism of a causal cone often dictates a peak decoder requirement,  $M_{peak}$ , that is less than the total available decoders, M. As illustrated in Figure 11, this discrepancy creates idle decoders and wastes computational resources. To reclaim this lost throughput, we introduce an opportunistic backfilling mechanism. The scheduler first computes  $M_{peak}$ from the emergency plan, then derives the max usable decoders for backfilling at each pass as  $M_{usable}(t) =$  $\max(0, \min(M - M_{peak} - B_{bf}(t), F(t) - E(t))),$  $B_{bf}(t)$  is currently running backfill tasks, F(t) is physically free decoders, and E(t) is emergency tasks dispatched in the same pass. This budget is then used to dispatch non-critical, causally-disconnected tasks using the heuristic scheduler, thereby maximizing throughput without any risk of interfering with the critical emergency plan.

