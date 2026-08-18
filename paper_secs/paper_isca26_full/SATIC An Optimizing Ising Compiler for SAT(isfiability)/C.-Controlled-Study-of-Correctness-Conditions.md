# *C. Controlled Study of Correctness Conditions*

To isolate the impact of ancillary-awareness and clausecompleteness, we run a controlled SATLIB UF75 study with a 50K iteration budget and 100 instances on the Ising hardware. Table IV summarizes the results. Violating clause-completeness renders 0/100 solved instances; violating ancillary-awareness, 44/100 – confirming that both conditions are critical for convergence. In contrast, SATIC solves 87/100 instances without the full bag of tricks, while SATIC++

TABLE IV: Controlled study isolating ancillary-awareness and clause-completeness on SATLIB UF75.

| Variant           | Ancillary-<br>aware | Clause<br>complete | Solved  |
|-------------------|---------------------|--------------------|---------|
| Clause-incomplete | Yes                 | No                 | 0/100   |
| Ancillary-unaware | No                  | Yes                | 44/100  |
| SATIC             | Yes                 | Yes                | 87/100  |
| SATIC++           | Yes                 | Yes                | 100/100 |

solves all 100 – showing that ancillary-awareness and clausecompleteness form the correctness-preserving foundation for SATIC++.

## *D. Comparison to State of the Art*

Fig.11 provides a comparison with Cilasun et al. [46] – a recent SAT decomposer targeting similar Ising hardware, as well as D-Wave's Energy Impact Decomposer (D-Wave EID) – the most up-to-date version of qbsolv [50], which represents one of the best generic decomposers from the literature [62]. D-Wave EID++ is our modified version of D-Wave EID. D-Wave EID++ randomizes clause-specific ancillary variable values at every iteration before subproblem formation. This does not make D-Wave EID++ ancillary-aware by construction; but serves as a diagnostic heuristic. For a fair comparison, we report performance in terms of *average number of iterations to find a solution*, as time per iteration stays practically constant across runs.

![](_page_9_Figure_13.jpeg)

Fig. 11: Average number of iterations to find a solution for representative baselines (lower is better).

Cilasun et al. [46] only uses SATLIB UF20 benchmarks on a 49-spin all-to-all connected Ising chip. We stick to their reported data for our comparison. Our Ising chip has the same coefficient range, but has a slightly lower number of spins, which translates into an 8% reduction in hardware capacity.

The data points from Fig.11 for D-Wave EID, D-Wave EID++, as well as SATIC and SATIC++ come from our 45-spin Ising chip. Hence, Cilasun et al. [46] has an 8% higher hardware capacity in this comparison. Despite this difference, we observe that SATIC++ significantly outperforms the alternatives by solving all 10 benchmark instances in just 9.4 iterations on average. The closest baseline requires approximately 250 iterations in this case, with D-Wave EID often reaching the 500-iteration limit without any solution.

The main reason is that ancillary variables often have lower energy impact than problem variables. D-Wave EID tends to leave out ancillary variables in subproblem formation, breaking ancillary-awareness (Section III-A). D-Wave EID++ shows better performance than D-Wave EID because ancillary randomization can partially mitigate ancillary-unawareness and may yield better solutions – highlighting the importance of ancillary-awareness during subproblem formation. Even without the bag of tricks, SATIC achieves lower iteration counts for most of the benchmark problems. This comparison highlights SATIC's ability to extract more value from constrained hardware compared to representative alternatives.

