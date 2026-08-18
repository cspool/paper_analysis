# *B. Pipeline Schedule Generation*

After constructing the communication pattern, the next step is to determine the precise execution step for each communication operation to efficiently implement collective communication. This scheduling process must account for both resource constraints, where network links serve as limited resources, and data dependencies, where nodes act as intermediaries between consecutive edges in the communication graph. This problem can be formulated as a conventional pipeline scheduling problem. Numerous modulo scheduling techniques [42], [47], [68] have been developed to address similar challenges by optimizing resource allocation and execution timing. However, in our case, the problem can be simplified due to the

## Algorithm 1: Pipeline Schedule Generation Input : R: Communication patterns, II: Initiation interval Output: step: Execution step for each data transfer <sup>1</sup> Init Heap with R; <sup>2</sup> while Heap is not empty do <sup>3</sup> (s, node, depth) ← Heap.pop(); <sup>4</sup> f inished ← true; <sup>5</sup> for each outgoing edge link of node in pattern s do <sup>6</sup> if (s, link) ∈/ step then <sup>7</sup> if RT[link] at depth is free then <sup>8</sup> f inished ← true; <sup>9</sup> step(p, link) ← depth; <sup>10</sup> for i ← 0 to w(link) − 1 do <sup>11</sup> RT[link][(depth + i) mod II] ← true; <sup>12</sup> end <sup>13</sup> if end(link) is ready then <sup>14</sup> Heap.push((s, end(link), depth + w(link)); <sup>15</sup> end <sup>16</sup> end <sup>17</sup> else <sup>18</sup> f inished ← f alse; <sup>19</sup> end <sup>20</sup> end <sup>21</sup> end <sup>22</sup> if *not* f inished then <sup>23</sup> Heap.push((s, node, depth + 1)); <sup>24</sup> end

absence of inter-iteration data dependencies, which eliminates the need for complex dependency resolution across iterations.

<sup>25</sup> end

*1) Algorithm Design:* We propose a heuristic scheduling strategy based on a Modulo-II Reservation Table (RT) to minimize the total number of pipeline stages (i.e., the pipeline latency depth). This method systematically assigns communication operations to physical time slots while ensuring conflict-free resource allocation across overlapping iterations, thereby enhancing overall efficiency. Algorithm 1 illustrates the pipeline generation procedure, which uses a heap-based scheduler to prioritize transfers. Here, the variable *depth* denotes the actual logical scheduled time offset, rather than just a topological level. The heap always extracts the target communication link with the smallest current depth to maintain a highly dense execution timeline.

Initially, all source nodes with no predecessors are inserted into a heap with a starting depth of zero. The algorithm iteratively pops out the node with the smallest depth and attempts to schedule its outgoing communication links (line 3). Each link is scheduled only if it has not been assigned a valid step. Crucially, the reservation table ensures conflictfree steady-state scheduling by checking the availability of bandwidth slots modulo the II (line 7). If a valid time slot is found, the communication is assigned to that step, and the Modulo-II reservation table is updated accordingly (line 9-12). A successor node is pushed into the heap only after its preceding transfer completes, strictly obeying structural dependencies similar to topological sorting (line 13-15).

If a resource conflict arises at the current depth (i.e., the target RT slot is already occupied), scheduling for that node is

![](_page_6_Figure_5.jpeg)

Fig. 5. Illustration of pipeline schedule: (a) A 6-step AllReduce communication schedule on a 3 × 3 2D mesh. Dotted arrows indicate idle time due to pending communications, while gray nodes represent partial nodes waiting for additional links to complete. (b) A pipeline schedule with an initiation interval of 2. For clarity, only three representative channels are shown.

![](_page_6_Figure_7.jpeg)

Fig. 6. Illustration of the AllReduce algorithm on a 2D mesh topology. The left side shows the code implementing the communication, while the right side depicts the corresponding patterns. Red arrows indicate the initial communication pattern on a sub-mesh, while blue arrows represent the hierarchical construction with pattern expansion.

deferred, and it is reinserted into the heap with an incremented depth ('depth + 1', lines 22-24). This deferral strategy resolves structural hazards by gracefully inflating pipeline latency. Because our MILP formulation already enforces that the aggregate number of chunk transmissions mapped to any link does not exceed the capacity of a single II, there are always sufficient slots available in the steady state. Furthermore, since there are no inter-iteration data dependencies, these latency deferrals only shift execution offsets without creating circular logical stalls, guaranteeing that the algorithm will eventually converge to a valid modulo schedule.

As shown in Figure 5(a), the pipeline construction produces a 6-step schedule for the full AllReduce operation. With an initiation interval of 2, the pipeline enables effective overlap across stages, keeping links active throughout most of the execution. Only during the prologue and epilogue exhibit partial underutilization, as illustrated in Figure 5(b). The initiation interval of 2 implies that each transfer exclusively occupies either the odd or even time slots, a constraint that must be explicitly incorporated into pipeline scheduling. The explicit mapping of depth to the Modulo-II reservation table ensures that each transfer is statically bound to either the odd or even operational slots, a constraint safely preserved by our overlapping strategy. Through careful pipeline scheduling, our approach ensures that reductions within a single pattern are forwarded to the subsequent broadcast phase as soon as they become available, rather than waiting for the entire reduction phase to complete.

*2) Performance Analysis:* For a pipeline schedule with data of size D in one node, partitioned into C chunks with an initiation interval II, and accumulated into R roots over S steps, each new iteration requires only II steps. Thus, the total number of steps is S + II ∗ (C − 1). Therefore, the total communication cost is:

$$Cost = (S + II * (C - 1)) * (\alpha + \frac{D}{RC}\beta)$$
$$= (S - II)\alpha + \frac{D * II\beta}{R} + (II * C\alpha + \frac{D(S - II)\beta}{RC})$$

To minimize cost, the optimal chunk size is:

$$C^* = \sqrt{\frac{D(S - II)\beta}{\alpha R * II}}$$

Thus, the lower bound on cost is:

$$Cost \ge (S - II)\alpha + \frac{D*II\beta}{R} + 2\sqrt{\frac{D(S - II)II\alpha\beta}{R}}$$

In Figure 5, given α = 200ns, 1/β = 50GB/s, D = 16MB, R = 3, S = 10, II = 2, we calculate the optimal chunk count as C <sup>∗</sup> ≈ 46. With an explicit model of pipeline behavior, the optimal partition can be determined analytically.

