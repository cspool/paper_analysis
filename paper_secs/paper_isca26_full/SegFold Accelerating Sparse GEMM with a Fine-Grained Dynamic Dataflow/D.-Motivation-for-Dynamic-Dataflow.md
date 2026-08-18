# *D. Motivation for Dynamic Dataflow*

Static scheduling fixes the execution order and resource assignment ahead of time, leaving it unable to react to variations in nonzero distribution or rebalance work at runtime. As a result, no static dataflow can simultaneously maximize reuse on all three operands. We propose a dataflow that integrates *dynamic scheduling* to adapt work selection within a tile and *dynamic mapping* to redistribute partial sums across PEs at runtime, broadening the achievable reuse–utilization tradeoff.

