# 5 Constraint-Aware Scheduling Algorithm

We assume that NLP applications have diverse latency constraints for generating a sequence within guaranteed time-frames. To maximize the resource utilization under these constraints, we formulate an optimization problem in this section and present an scheduling algorithm to address the problem.

Our optimization problem is to maximize the inference throughput under a given latency constraint by finding the optimal values of the control variables for the given scheduling policy. More formally the problem is stated as below.

```
\label{eq:argmax} \begin{split} \underset{B_{\{E,D,m\}},T_P,F_E,S}{\operatorname{Throughput}}(B_E,B_D,B_m,T_P,F_E,S,P_E,P_D) \\ \text{s.t. } Latency(B_E,B_D,B_m,T_P,F_E,S,P_E,P_D) < L_{Bound}, \quad \text{where} \end{split}
```

- $B_E$ ,  $B_D$ , and  $B_m$  are the sizes of encoder/decoder batches and decoder micro-batches,
- $T_P$  is the tensor-parallelism degree and applied GPU count,
- $F_E$  is the frequency of running encoders for RRA Scheduling,
- *S* is the given scheduling policy (RRA, WAA-C, or WAA-M),
- $P_E$  and  $P_D$  are the given distributions of input/output length,
- *Throughput()* and *Latency()* are the functions that give the throughput and latency for the given environment, and
- $L_{Bound}$  is the latency bound for the given sequence length.

The problem can be solved by applying black-box optimization techniques such as Bayesian optimization [20], which do not rely on any assumptions about the objective

functions. Alternatively, we can take advantage of the monotonicity property of the control variables for faster optimization by customizing existing algorithmic frameworks for monotonic optimization. We explain this approach in the following section.

