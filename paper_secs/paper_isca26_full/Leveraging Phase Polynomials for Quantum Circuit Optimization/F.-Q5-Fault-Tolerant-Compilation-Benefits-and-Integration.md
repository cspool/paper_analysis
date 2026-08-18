# *F. Q5: Fault-Tolerant Compilation Benefits and Integration*

Fault-tolerant quantum computing (FTQC) is limited by qubit overhead, runtime, and architectural constraints. Many logical/NISQ optimizations extend to FT compilation [\[47\]](#page-14-4). Traditional FT analyses focus on T-gate count/depth due to the high cost of magic-state distillation [\[24\]](#page-13-21). Recent advances in magic-state cultivation [\[25\]](#page-13-22)–[\[27\]](#page-13-23) substantially reduce this overhead, making Clifford costs increasingly important. Modern cost models suggest that CNOTs can be comparable in spacetime cost to T states of similar reliability [\[25\]](#page-13-22), with non-constant ancilla volume and operation depth [\[28\]](#page-13-24). Since *PhasePoly* reduces both CNOT and R<sup>z</sup> structure at the logical level, it can improve downstream FT resource costs.

FT resource estimation. Using the Azure Resource Estimator [\[48\]](#page-14-5), [\[49\]](#page-14-6) under a surface-code, nearest-neighbour architecture [\[50\]](#page-14-7), [\[51\]](#page-14-8), we perform end-to-end FT resource estimation. Fig. [17](#page-10-1) reports normalized wall-clock runtime relative to unoptimized circuits, excluding small-size and parameterized circuits. Quartz, QUESO, and *PhasePoly* achieve average reductions of 11.99%, 31.80%, and 44.62%, respectively, with *PhasePoly* providing the largest improvement.

<span id="page-10-1"></span>![](_page_10_Figure_12.jpeg)

Fig. 17: Normalized fault-tolerant circuit wall-clock runtime.

Integration with Clifford+T synthesis. Each arbitrary R<sup>z</sup> rotation must be synthesized into an FT instruction set such as Clifford+T (H, S, and T sequence). We therefore study how *PhasePoly* interacts with FT gate synthesis by combining it with GridSynth [\[52\]](#page-14-9) on 14 variational circuits: QAOA Max-Cut on 3-regular graphs (4–24 qubits, 2,150– 12,900 gates) and VQE circuits, including UCCSD ansatz with Jordan-Wigner (JW) [\[53\]](#page-14-10), Bravyi-Kitaev (BK) [\[54\]](#page-14-11), and parity (P) [\[55\]](#page-14-12) encodings, as well as the Hamming-weightpreserving ansatz (HW) [\[56\]](#page-14-13) (4–12 qubits, 2,641–231,780 gates). We compare two compilation orders: (A) GridSynth → *PhasePoly* and (B) *PhasePoly* → GridSynth. The purpose of investigating this is to see how to better apply *PhasePoly* into the overall compilation pipeline. Both pipelines apply the same commuting-rule simplification as the final pass.

Fig. [18](#page-11-0) reports normalized T count, two-qubit gate count, and circuit depth. Running *PhasePoly* before GridSynth produces the lowest depth on most circuits because large {CNOT,Rz} regions are simplified before GridSynth introduces additional H gates that split phase-polynomial blocks and limit rotation-merging opportunities.

Across benchmarks, we observe: (i) T-count changes are modest, noticeable mainly for HWPA circuits; (ii) two-qubit gate reductions are common using *PhasePoly*'s optimization and often translate to lower depth; and (iii) the benefit

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Fig. 18: Comparison of two compilation orders. Metrics are normalized to GridSynth  $\rightarrow$  *PhasePoly*.

depends on the regularity of circuit structure: circuits with structured phase interactions (e.g., parity and HWPA ansatz) obtain larger improvements ( $\sim \! 10\%$  depth reduction on average), while JW/BK encodings show < 1% change due to their already compact CNOT- $R_z$  structure.

**Q5 Summary:** PhasePoly's strong  $CNOT/R_z$  reductions, long-range optimization, and natural fit for Clifford+T circuits make it effective for fault-tolerant compilation. PhasePoly is most effective when applied before Clifford+T synthesis, which introduces additional H barriers.

#### G. Q6: Correctness and Robustness of PhasePoly

a) Equivalence checking for correctness: Phase polynomials faithfully model {CNOT,  $R_z$ } circuits and are widely used for verification [12], [13], [18]. Because *PhasePoly* introduces *cross-block IR and optimization* that merges multiple phase-polynomial blocks, we enforce additional constraints and pruning rules to guarantee that all intermediate states remain legal and synthesizable (Section III-C3).

We also perform end-to-end equivalence checking: for circuits with fewer than 8 qubits, we compare unitaries using Qiskit [6], and for all circuits we use the formal verification tool MQT QCEC [57]. All checked benchmarks pass verification; mod\_adder\_1024 is excluded because it exceeds our hardware limits.

b) Incremental block merging for robustness: Cross-block optimization can yield additional reductions—about one third of benchmarks (9 circuits) benefit from it. We use incremental block merging that expands the merge size gradually, optimizing step by step rather than merging all blocks at once.

Table I evaluates three representative circuits under the same parameters except for the merge size ("Group k", merging 1-7 adjacent blocks) and our *Incremental* strategy, which gradually increases k and keeps only beneficial steps. For barenco\_tof\_10, improvements emerge at Group 3 and stabilize, matching the *Incremental* result. For adder\_8,

<span id="page-11-1"></span>

| Circuit     | barenco_tof_10 |       | adder_8 |       | ham15_med |       |
|-------------|----------------|-------|---------|-------|-----------|-------|
|             | # Gates        | # CXs | # Gates | # CXs | # Gates   | # CXs |
| Org.        | 450            | 192   | 900     | 409   | 1272      | 534   |
| Group 1     | 262            | 128   | 557     | 274   | 696       | 353   |
| Group 3     | 248            | 114   | 542     | 259   | 695       | 352   |
| Group 5     | 248            | 114   | 540     | 257   | 693       | 350   |
| Group 7     | 248            | 114   | 540     | 257   | 694       | 351   |
| Incremental | 248            | 114   | 542     | 259   | 656       | 325   |

TABLE I: Effect of cross-block merge size on optimization quality for three typical circuits. "Group k" merges k adjacent blocks at a time,  $k \in \{1, 3, 5, 7\}$ ; "Incremental" increases k stepwise up to 7, retaining gains and pruning regressions. **Bold** numbers denote the best value in each column.

larger groups continue to help (CX:  $274 \rightarrow 257$ ), while *Incremental* remains close (259). For ham15\_med, performance peaks at Group 5 (353  $\rightarrow$  350) but degrades at Group 7; *Incremental* avoids this and achieves the best (325). Overall, *Incremental Block Merging* offers a robust approach that captures large gains while avoiding over-merging regressions.

**Q6 Summary:** Our *cross-block* IR and optimization preserve correctness (all verified except one timeout) and *Incremental block merging* strategy yields robust gains while avoiding over-merging side effects.

