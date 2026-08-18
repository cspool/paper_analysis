# *C. Q2: Necessity in General Circuit Optimization*

*PhasePoly* is orthogonal to subcircuit rewriting frameworks and can be composed with existing optimization passes. To study their interaction, we integrate *PhasePoly* with two stateof-the-art subcircuit rewriting frameworks—Quartz [\[17\]](#page-13-15) and QUESO [\[18\]](#page-13-25)—and evaluate them under their recommended settings (3-qubit / 6-gate subcircuits, 2-hour per circuit). Fig. [12](#page-8-0) summarizes their standalone performance; note that this is not an apples-to-apples comparison: although they optimize gates beyond phase polynomial structure, *PhasePoly* still delivers the strongest reductions: Quartz and QUESO reduce total gate by 22.17% and 27.83% (CNOTs by 16.88% and 20.70%), while *PhasePoly* achieves total reduction by 34.70% (CNOTs by 26.83%) on average.

*a) Effect of circuit scale:* On average, *PhasePoly* achieves greater reductions in both total and CNOT gates, though it is not always the best on small circuits. To analyze this trend, we group benchmarks by original gate count:

Small (<200 gates): Taking the best of Quartz and QUESO per circuit, *subcircuit rewriting* ties *PhasePoly* on 6 circuits and surpasses it on 2 of 10.

Medium (200–500 gates): Only 3 circuits tie or exceed *PhasePoly* among 10.

Large (>500 gates): Only one QAOA circuit ties *Phase-Poly*; all other circuits exhibit significant performance gaps.

These results are consistent with design intent: *PhasePoly* leverages the *phase polynomial* and *cross-block* IR to capture long-range structure across the circuit, while subcircuit rewriting—bounded by local equivalence patterns—loses effectiveness as it scales up despite covering more gate types.

*b) Integrating optimization passes:* Fig. [13](#page-8-1) analyzes how *PhasePoly* interacts with subcircuit rewriting in a combined pipeline. We denote sequential application as "A+B" (run A

<span id="page-8-1"></span>![](_page_8_Figure_10.jpeg)

Fig. 13: Integration of *PhasePoly* with Quartz and QUESO. Bars show average total-gate (left) and CNOT (right) reduction rates relative to original circuits. "A+B" applies A then B; solid and hatched bars indicate first and second passes.

first, then B). Applying subcircuit rewriting after *PhasePoly* yields modest additional gains (≈0.75–1.25% in total and CNOT reductions). Applying *PhasePoly* after rewriting provides substantially larger improvements (≈6–13%), revealing that *PhasePoly* identifies long-range opportunities left unexploited by local subcircuit rewriting.

Although QUESO slightly outperforms Quartz as a standalone pass—thanks to its phase modeling—both benefit when combined with *PhasePoly*. Quartz + *PhasePoly* performs better than QUESO + *PhasePoly*. Quartz does not consider any phase information, and therefore mainly removes local redundancy. This implies the benefit of having a dedicated phase polynomial pass, rather than having the phase polynomial optimization spread into different optimization passes.

Q2 Summary: *PhasePoly* complements local subcircuit rewriting: the former enables long-range CNOT/phase reductions, while the latter is effective on small circuits. Together, they unlock optimizations neither achieves alone.

Q2 shows that the gap between *PhasePoly* and subcircuit rewriting widens as circuits grow. Search-based *subcircuit rewriting* enumerates many candidate equivalent subcircuits before applying a rule. As the target pattern size grows, the search space explodes exponentially, so practical deployments restrict patterns to limited windows. This locality makes it difficult to realize long-range optimization opportunities that span large portions of a circuit.

We further stress-test scalability on three large-circuit families: *(i) MCX* (multi-controlled-NOT) circuits [\[20\]](#page-13-17), where qubit and gate counts grow roughly linearly (19–499 qubits; 480–14,880 gates); *(ii) Adder* circuits (23–383 qubits; 637– 12,637 gates); and *(iii) HWB* (Hamming coding functions) [\[29\]](#page-13-26) with a fixed count of 16 qubits but rapidly growing gates (345– 104,068). We compare *PhasePoly* with Quartz and QUESO under a 2-hour time budget; *PhasePoly* never exceeded 5,500 seconds even on the largest instance (hwb8\_113).

Fig. [14](#page-9-0) reports total-gate and CNOT reductions for adder and HWB circuits, complementing the MCX results in Fig. [5.](#page-3-1) As circuit size increases, Quartz and QUESO saturate or fail on the largest instances, whereas *PhasePoly* continues to achieve substantial reductions. These results show that structured parity-matrix reasoning and cross-block optimization expose long-range opportunities that fixed-window subcircuit rewriting struggles to capture.

Q3 Summary: On large circuits, *PhasePoly* sustains and widens its advantage by leveraging its global parity reasoning and cross-block optimization, capturing long-range reductions that *subcircuit rewriting* cannot.

<span id="page-9-0"></span>![](_page_9_Figure_5.jpeg)

(b) HWB circuits. X-axis: original gate count (log scale); Y-axis: total gates and CNOTs. Missing points denote optimization failures.

Fig. 14: Comparison of large circuits, *PhasePoly* scales effectively, while subcircuit rewriting saturates on large circuits.

