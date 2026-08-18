# *B. Heterogeneous Resource Management*

Our dynamic tile scheduler operates over a distributed set of heterogeneous execution units, each managed by an independent, semantics-aware queue pair. The scheduling mechanism is organized into four cooperating steps (Figure 4), which together form a decentralized arbitration mechanism.

Step 1: Semantic Routing. Incoming TISA instructions are parsed for their OpType, UnitMap, and dependency metadata, then routed to the appropriate waiting queue (WQ) of each target unit. Each WQ preserves operator semantics and provides a local view of ready candidates per unit type.

Step 2: Dependency Resolution. The scheduler periodically selects a ready window W from each WQ and checks

![](_page_6_Figure_0.jpeg)

Fig. 4: Per-unit semantic scheduling. Decentralized queues localize dependency checking and prevent unrelated blocking across heterogeneous units. WQ: waiting queue; IQ: issue queue.

for semantic hazards against the unit's in-flight table Fu. Only instructions passing dependency and resource checks are promoted to the issue queue (IQ). This step enforces correctness while enabling out-of-order admission across units.

Step 3: Adaptive Issue. IQ entries are issued to hardware execution pipelines once their dependencies are cleared.

Step 4: Feedback. Upon completion, the corresponding F<sup>u</sup> entry is retired, dependent instructions are notified, and per-unit scheduling priorities are adaptively updated based on observed contention or latency. This feedback mechanism continuously tunes overlap and unit utilization at runtime.

These steps naturally form a five-stage microarchitecture of: (1) Reception(instruction) decoding; (2) Routing to per-unit WQs; (3) Dependency check matching the window; (4) Issue of conflict-free instruction from WQ to IQ; and (5) Dispatch IQ instruction to units.

Issued TISA instructions execute in a run-to-complete, nonpreemptive manner, with scheduling decisions made only at tile boundaries. Because these boundaries are coarse-grained (typically more than 10<sup>3</sup> operations), the control overhead remains low, i.e., around 7∼9 cycles per dispatch, as measured in our RTL synthesis.

## *C. The Dynamic Scheduling Algorithm*

Algorithm 1 formalizes this scheduling process. At a high level, the scheduler continuously receives semantically annotated TISA instructions, routes them to appropriate queues, performs dependency and resource checks, issues ready tiles out of order, and updates runtime states upon completion. Algorithm 2 details the semantic conflict detection routine that underpins this process. Through this mechanism, the scheduler achieves execution patterns illustrated in Figure 2(c,e).

The scheduling cycle complexity is O(U · W · |F|max), where U is the number of execution units, W the window size, and |F|max the maximum number of in-flight entries per unit. With typical settings (W ≤ 8, |Fu| ≤ 16), the effective complexity approaches O(U) per cycle with minimal constants. The conflict detection subroutine runs in O(|Fu|) time per candidate, with constant-space overlap checks. This design scales better than centralized ILP schedulers (e.g., Tomasulo [41]) that require O(N<sup>2</sup> ) global comparisons, while enabling finer-grained, semantics-driven parallelism.

Our synthesized RTL implementation integrates one scheduler per accelerator core (Table IV). At W = 8, the scheduler

```
Algorithm 1: Dynamic Tile Scheduling
 Input: Stream of semantically-annotated TISA instructions
 Output: Scheduled instruction sequences across
          heterogeneous units
 Initialize semantic tracking structures for all units u;
 while system running do
     // 1: Semantic Routing
     if Reception Buffer ̸= empty then
         I ← pop(Reception Buffer) ;
         extract semantic context(I) ;
         // Analyze OpType, dependencies, resource needs
         u ← adaptive unit selection(I) ;
         // Consider load balancing
         enqueue with priority(WQ[u], I) ;
     foreach u in Units do
         C ← select ready window(WQ[u]);
         // Semantic-aware selection
         foreach I ∈ C (by adaptive priority) do
             // 2: Dependency Resolution(call Algorithm 2)
             if !semantic conflict detection(I, Fu) and
              resources available(u) then
                 allocate resources adaptively(I, u);
                 update semantic tracking(Fu, I);
                 // 3: Adaptive Issue
                 issue out of order(I, u);
     foreach u in Units do
         foreach completed inst J from Exec[u] do
             // 4: Feedback
             update semantic state(Fu, J);
             trigger dependent instructions(WQ[u], J);
```

# *B. Heterogeneous Resource Management*

Our dynamic tile scheduler operates over a distributed set of heterogeneous execution units, each managed by an independent, semantics-aware queue pair. The scheduling mechanism is organized into four cooperating steps (Figure 4), which together form a decentralized arbitration mechanism.

Step 1: Semantic Routing. Incoming TISA instructions are parsed for their OpType, UnitMap, and dependency metadata, then routed to the appropriate waiting queue (WQ) of each target unit. Each WQ preserves operator semantics and provides a local view of ready candidates per unit type.

Step 2: Dependency Resolution. The scheduler periodically selects a ready window W from each WQ and checks

![](_page_6_Figure_0.jpeg)

Fig. 4: Per-unit semantic scheduling. Decentralized queues localize dependency checking and prevent unrelated blocking across heterogeneous units. WQ: waiting queue; IQ: issue queue.

for semantic hazards against the unit's in-flight table Fu. Only instructions passing dependency and resource checks are promoted to the issue queue (IQ). This step enforces correctness while enabling out-of-order admission across units.

Step 3: Adaptive Issue. IQ entries are issued to hardware execution pipelines once their dependencies are cleared.

Step 4: Feedback. Upon completion, the corresponding F<sup>u</sup> entry is retired, dependent instructions are notified, and per-unit scheduling priorities are adaptively updated based on observed contention or latency. This feedback mechanism continuously tunes overlap and unit utilization at runtime.

These steps naturally form a five-stage microarchitecture of: (1) Reception(instruction) decoding; (2) Routing to per-unit WQs; (3) Dependency check matching the window; (4) Issue of conflict-free instruction from WQ to IQ; and (5) Dispatch IQ instruction to units.

Issued TISA instructions execute in a run-to-complete, nonpreemptive manner, with scheduling decisions made only at tile boundaries. Because these boundaries are coarse-grained (typically more than 10<sup>3</sup> operations), the control overhead remains low, i.e., around 7∼9 cycles per dispatch, as measured in our RTL synthesis.

## *C. The Dynamic Scheduling Algorithm*

Algorithm 1 formalizes this scheduling process. At a high level, the scheduler continuously receives semantically annotated TISA instructions, routes them to appropriate queues, performs dependency and resource checks, issues ready tiles out of order, and updates runtime states upon completion. Algorithm 2 details the semantic conflict detection routine that underpins this process. Through this mechanism, the scheduler achieves execution patterns illustrated in Figure 2(c,e).

The scheduling cycle complexity is O(U · W · |F|max), where U is the number of execution units, W the window size, and |F|max the maximum number of in-flight entries per unit. With typical settings (W ≤ 8, |Fu| ≤ 16), the effective complexity approaches O(U) per cycle with minimal constants. The conflict detection subroutine runs in O(|Fu|) time per candidate, with constant-space overlap checks. This design scales better than centralized ILP schedulers (e.g., Tomasulo [41]) that require O(N<sup>2</sup> ) global comparisons, while enabling finer-grained, semantics-driven parallelism.

Our synthesized RTL implementation integrates one scheduler per accelerator core (Table IV). At W = 8, the scheduler

```
Algorithm 1: Dynamic Tile Scheduling
 Input: Stream of semantically-annotated TISA instructions
 Output: Scheduled instruction sequences across
          heterogeneous units
 Initialize semantic tracking structures for all units u;
 while system running do
     // 1: Semantic Routing
     if Reception Buffer ̸= empty then
         I ← pop(Reception Buffer) ;
         extract semantic context(I) ;
         // Analyze OpType, dependencies, resource needs
         u ← adaptive unit selection(I) ;
         // Consider load balancing
         enqueue with priority(WQ[u], I) ;
     foreach u in Units do
         C ← select ready window(WQ[u]);
         // Semantic-aware selection
         foreach I ∈ C (by adaptive priority) do
             // 2: Dependency Resolution(call Algorithm 2)
             if !semantic conflict detection(I, Fu) and
              resources available(u) then
                 allocate resources adaptively(I, u);
                 update semantic tracking(Fu, I);
                 // 3: Adaptive Issue
                 issue out of order(I, u);
     foreach u in Units do
         foreach completed inst J from Exec[u] do
             // 4: Feedback
             update semantic state(Fu, J);
             trigger dependent instructions(WQ[u], J);
```

