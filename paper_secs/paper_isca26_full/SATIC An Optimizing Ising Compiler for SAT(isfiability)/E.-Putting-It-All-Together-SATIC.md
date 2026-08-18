# E. Putting It All Together: SATIC++

Fig.7 provides the SATIC flow with the entire bag of tricks, which we refer to as SATIC++, with the time complexity tabulated in Table II. Let |V| and |E| denote the number of nodes and edges in the VIG;  $L_s$  and  $m_s$ , the number of literals and clauses in a subproblem; and n, the total number of SAT variables. Overall, SATIC++ preserves the O(TL) time complexity of SATIC (linear in the number of CNF literals L per iteration, over T iterations), and adds a one-time  $O(|E|\log|V|)$  MST preprocessing step. The only hardware-dependent knob is the neighbor cap N in **Limited Neighbors**, which depends on the Ising machine capacity. As an example, for the 45-spin Ising chip (Section VI) we set N=10. The local-search interval  $T_{\rm LS}$  controls how frequently we restart the

process. We set TLS once per workload using a small profiling run.

We next quantitatively characterize the impact of each trick on overall performance and scalability, highlighting synergistic interactions between different tricks.

## VI. EVALUATION SETUP

## *A. Metrics*

*Batch* refers to a group of SAT problems that share the same configuration – the same clause width (k), number of variables (n), and number of clauses (m). We evaluate SATIC using multiple batches and consider each batch solved only if all of its instances are solved.

*Instance* denotes a single SAT problem within a batch. We use batches with at least 50 instances for proper evaluation. If a trick fails on even a few instances despite solving others quickly, that trick is treated as problem-specific rather than generally effective. We do not consider problem-specific tricks in this paper.

*Repeats* are independent runs per instance to guarantee statistical significance. We use at least 100 repeats and consider an instance solved if any repeat succeeds.

*Iteration Count* measures the number of hardware (Ising machine) calls made per run and serves as a proxy for the time overhead of SATIC.

*Time to Solution (TTS)* is a standard metric in evaluating stochastic solvers [59], [60] and refers to the expected time required to solve a given problem instance at a target confidence level; in this paper, we use a confidence level corresponding to solving at least 95 out of 100 independent repeats.

Ising solvers are stochastic and must be run many times per instance, so single wall-clock measurements can be noisy and implementation-dependent. We use the number of solver iterations (hardware calls) and TTS (in terms of iterations) as our primary cost metrics. We also report end-to-end runtime with detailed breakdown.

