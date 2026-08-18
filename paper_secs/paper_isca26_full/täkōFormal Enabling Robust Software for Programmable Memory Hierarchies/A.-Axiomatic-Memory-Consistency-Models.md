# *A. Axiomatic Memory Consistency Models*

An axiomatic memory model is a formalization used to define the allowable executions of a program. A program execution is represented as a directed graph, where nodes represent instructions and labeled edges encode relations between instructions. The allowable executions of the program are given by axioms that enforce constraints on these defined relations. Various memory consistency models have been encoded using this approach [2, 4, 5, 29, 49, 65, 68].

As an example, consider Figure 5a's (non-tak¨ o) program. ¯ Under sequential consistency (SC) [29], the allowed executions are those corresponding to an interleaving of each core's instructions in program order. As such, the outcome of r1=1, r2=0 here is outlawed, as r1=1 would indicate (i2) has completed and r2=0 would indicate (i1) has not completed, violating program order on either core 0 or core 1.

Figure 5b shows an execution graph for the r1=1, r2=0 outcome of Figure 5a. All addresses are assumed to initially be 0, as enforced by the initialization writes (W) of 0 to [a] and [b]. The other W nodes are the write events ((i1) and (i2)), while the R nodes are the read events. Each node is annotated with its address and value read or written. The sb (sequencedbefore) edges connect a given instruction to instructions after it in program order. The rf (reads-from) edge denotes that the read at the target reads from the write at the source. The fr (from-reads) edge denotes that the write at the target occurs *after* the read at the source. The mo (modification order) relation establishes a total order on all writes to an address<sup>1</sup> .

The axiomatization of SC [5] forbids cycles comprised of the rf, fr, sb and mo relations. Formally, this is stated as acyclic(rf ∪fr∪sb∪mo). Figure 5b's execution graph has a cycle comprised of these relations. Thus, it is forbidden under SC, as we would expect for the outcome of r1=1, r2=0.

