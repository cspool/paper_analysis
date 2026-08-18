# *B. The Dual-Mode Scheduling*

Triage combines a lightweight *steady mode* for average-case throughput with a priority-aware *emergency mode* that resolves causal cones for imminent critical operations.

*1) Steady Mode: Heuristic Scheduling:* At each syndromegeneration or decode-completion event, the scheduler selects up to Mavailable conflict-free PENDING slices using a priority function P(V ). We explore several heuristic policies: First-In-First-Out (FIFO) prioritizes slices with the oldest timestamp to clear backlogs chronologically; Earliest-Deadline-First (EDF) prioritizes slices with the smallest deadline to proactively service operations closest to becoming critical; and Min-Degree-First (MDF) prioritizes slices with the fewest neighbors to minimize decoding latency.

To balance these critical factors, we propose a unified priority function:

$$P(V) = w_u \cdot \text{Urgency}(V) + w_c \cdot \text{Cost-Efficiency}(V),$$
 (2)

where  $w_u$  and  $w_c$  are tunable weighting factors ( $w_u + w_c = 1$ ). The urgency term quantifies proximity to a critical deadline, defined as Urgency(V) = 1/Deadline(V). The costefficiency term favors slices that are computationally cheaper to decode, defined by the inverse of the slice's degree, Cost-Efficiency(V) = 1/(Degree(V) + 1).

![](_page_6_Figure_3.jpeg)

Fig. 9. Relative idle layers inserted by different heuristic policies. Heuristiconly scheduling leaves significant room for improvement.

Figure 9 provides a preliminary evaluation of these policies (detailed setups are deferred to Section IV). While our proposed weighted heuristic consistently outperforms the simple baselines, the sheer volume of idle layers remains substantial across all purely heuristic approaches. Pure heuristics inherently lack the foresight to guarantee low-latency Pauli frame updates for irregularly timed critical operations. This limitation motivates the emergency mode of our dual-mode architecture.

2) Emergency Mode: Predictive Causal Cone Coloring: When the Triage Trigger signals an imminent deadline, the scheduler transitions to the emergency mode. Its objective is to resolve the causal cone of the impending critical operations with maximum parallelism, ensuring the necessary Pauli frames are updated before the Clifford correction executes.

Rather than making step-by-step decisions, the emergency mode employs a *predictive coloring* algorithm, detailed in Algorithm 1. This algorithm runs a discrete event simulation. It initializes a priority queue with only the PENDING slices in the on-demand causal cone, ensuring the input size minimal. The main loop advances a simulated clock to the next event and then greedily selects an independent set of tasks (Lines 10-14). The algorithm then records each selected slice in the final plan, and updates the auxiliary information (Lines 18-22). The core intuition is once inside the emergency mode, all slices in the causal cone share the same urgency. The primary factor for throughput is therefore the computational cost, resolved by the *MDF* policy. The online scheduler then transitions to a simple executor, dispatching the pre-computed tasks from the plan at their scheduled start times.

**Complexity Analysis** Let n be the number of slices in the causal cone. The initialization (Lines 4-7) pushes n elements into the priority queue Q, taking  $O(n \log n)$  time. During the main loop, each slice is extracted and dispatched exactly once. Therefore, the inner loop (Lines 16-20) performs at most 6n

