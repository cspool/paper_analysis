# *C. Integration with Overlapping Techniques*

Computation-communication overlapping is a critical technique for improving distributed training performance and can be broadly categorized into two main approaches. The first is *operation decomposition* [2], [8], [61], which restructures the computation graph by decomposing operations and analyzing dependencies to identify parallelizable tasks. The second is *kernel fusion* [7], [21], which fuses computation and communication at a finer granularity during compilation, forming block-level pipelines without explicitly partitioning global operations.

Unlike approaches that focus on scheduling *when* to communicate [7]–[9], [21], PipeComm optimizes *how* to communicate efficiently over the network fabric. Our work is orthogonal and complementary to these overlapping techniques. Specifically, PipeComm can serve as the high-performance communication backend for kernel fusion frameworks: once a communication task is triggered, PipeComm ensures it completes with the minimal latency and maximal bandwidth utilization through pipelined link scheduling. This is particularly crucial in communication-bound scenarios, such as large language model training, where communication overhead cannot be fully hidden by computation. By reducing the absolute communication time, PipeComm effectively increases the portion of communication that can be overlapped, directly improving end-to-end performance.

TABLE II PROGRAMMABLE PRIMITIVES IN PIPECOMM

| Primitives                | Descriptions                                  |
|---------------------------|-----------------------------------------------|
| Communication Primitives  |                                               |
| broadcast from(n, d, b)   | Broadcast data d from node n with bubble b    |
| reduce at(n, d)           | Reduce data d at node n                       |
| tile(d, factor)           | Tile the data d into factor segments          |
| pipeline(II)              | Apply the pipeline transformation with II     |
| Construction Primitives   |                                               |
| interleave(s1, s2)        | Interleave two schedules s1, s2               |
| reverse(s)                | Reverse all the communication in schedule s   |
| constraint(ns)            | Constraint pattern in the node collection ns  |
| extend(s, ns, news, topo) | Extend schedule s on ns with news in topology |

