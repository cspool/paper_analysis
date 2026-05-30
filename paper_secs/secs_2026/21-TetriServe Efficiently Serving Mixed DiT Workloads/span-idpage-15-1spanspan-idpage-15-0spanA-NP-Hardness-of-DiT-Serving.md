# <span id="page-15-1"></span><span id="page-15-0"></span>A NP-Hardness of DiT Serving

We prove NP-hardness for the DiT serving problem defined in TetriServe, which maximizes the number of requests that complete by deadlines under GPU capacity constraints.

Let us first define the decision problem DiT-Serving-Decision: given an instance, and an integer target B, decide whether there exists a schedule in which at least B requests meet their deadlines. This is the natural decision version of TetriServe's objective max  $\sum_i I_i$ .

Bar-Noy et al. [5, 14] state that the following real-time (RT) scheduling feasibility decision problem (RT-Feasibility) is NP-hard in the strong sense: on a single machine, given jobs with release times  $r_i$ , deadlines  $d_i$ , and processing times  $l_i$ , decide whether all jobs can be scheduled within their time windows. Since RT-Feasibility is strongly NP-hard, it remains NP-hard even when all numeric parameters are bounded by a polynomial in the input size. Therefore,  $T_{\rm max} = {\rm max}_i \, d_i$  is polynomially bounded, and our time-indexed reduction is polynomial-time.

**Reduction to DiT serving with**  $K = \{1\}$ . Given a RT-FEASIBILITY instance [5] with jobs i = 1, ..., n and parameters  $(r_i, d_i, l_i)$ , let us construct a single-step DiT instance as follows:  $N := 1, R := n, S_i := 1, K := \{1\}$ , arrival\_time $(i) := r_i, D_i := d_i, T_i(1) := l_i$ . Set the throughput target B := n.

Equivalently, in TetriServe's single-step time-indexed formulation with variables  $x_{i,t,k}$  and constraints (1)–(5), we restrict to k = 1 and N = 1, and disallow infeasible start times by setting  $x_{i,t,1} = 0$  whenever  $t < r_i$  or  $t + l_i > d_i$ .

**Correctness.** ( $\Rightarrow$ ) If the RT-FEASIBILITY instance is feasible, let  $s_i$  be the start time of job i in a feasible single-machine schedule. Schedule each corresponding DiT request i to start at time  $s_i$  using one GPU. All requests meet deadlines, so  $\sum_i I_i = n \geq B$ .

(⇐) If the constructed DiT instance has a schedule with  $\sum_i I_i \ge n$ , then all n requests meet deadlines. Since N=1 and each request uses one GPU, the capacity constraint implies no two requests overlap. Thus the chosen start times form a feasible non-preemptive single-machine schedule for all jobs in the original RT-FEASIBILITY instance.

Therefore, we can convert any RT-Feasibility instance into a DtT-Serving-Decision instance in polynomial time such that a feasible schedule exists in the former iff one exists in the latter. DtT-Serving-Decision is NP-hard even for the restricted case  $S_i = 1$  and  $\mathcal{K} = \{1\}$ ; consequently, the general multi-step DiT serving problem is NP-hard.

## **B** Scheduling Overhead Analysis

To validate the necessity of TetriServe's heuristic approach, we quantify the computational cost of finding a globally optimal schedule via exhaustive search. As established in Appendix A, the underlying step-level scheduling problem is NP-hard.

<span id="page-15-2"></span>

| # Reqs | Time (s) | # Reqs | Time (s) |
|--------|----------|--------|----------|
| 1      | < 0.01   | 1      | 0.02     |
| 2      | 0.27     | 2      | 11.12    |
| 3      | 52.56    | 3      | >60.00   |
| 4      | >60.00   | 4      | >60.00   |
| (a) 4  | GPUs     | (b)    | 8 GPUs   |

**Table 6. Scheduling overhead of exhaustive search.** Control plane scheduling time under different GPU budgets and queue sizes. TetriServe remains lightweight: it takes <0.01 s compared to exhaustive search following the same settings, enabling online scheduling in practice.

**Experimental Setup.** We implement an exact baseline solver that enumerates the complete decision space to maximize SLO attainment. The solver explores two dimensions of complexity for each request: (1) all feasible sequence-parallel degrees per diffusion step (e.g.,  $k \in \{1, 2, 4, 8\}$ ), and (2) all valid permutations of physical GPU mapping for those degrees. The objective is to identify the schedule with the highest SLO attainment, using minimum total GPU hours as a tie-breaker. We measure the wall clock latency required to generate a single scheduling plan using an AMD EPYC 7513 32-Core CPU, varying the queue depth (R) under fixed GPU budgets of  $N \in \{4, 8\}$ .

**Results.** Table 6 presents the scheduling overhead. The baseline exhibits immediate combinatorial explosion: with a budget of 8 GPUs, optimally scheduling merely three requests exceeds a 60-second timeout. This intractability stems from the factorial growth of permutation possibilities as the number of available GPUs increases. In contrast, TetriServe maintains a decision latency of <10 ms. These results confirm that exhaustive optimization is prohibitive for online serving, necessitating the efficient round-based planning strategy employed by TetriServe.