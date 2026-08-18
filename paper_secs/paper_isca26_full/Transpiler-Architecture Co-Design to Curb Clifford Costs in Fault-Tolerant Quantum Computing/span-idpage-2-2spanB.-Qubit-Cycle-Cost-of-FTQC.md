# <span id="page-2-2"></span>B. Qubit-Cycle Cost of FTQC

The overhead of FTQC execution is quantified using the *qubit-cycle volume* metric, where *qubit* denotes physical qubits and *cycle* is measured in QEC cycles. Each cycle represents one round of syndrome measurement and error correction, and the total volume is the product of these quantities. The FTQC circuit size determines the total number of logical qubits. In contrast, the number of physical qubits per logical qubit depends on the chosen code distance (*d*), set by the required logical error rate and total number of cycles.

The circuit's critical path determines the total number of QEC cycles. Within the Clifford+T gate set, the execution cost of each gate type differs. Pauli gates can be applied with no cost [30]. CNOT gates use lattice surgery and require 3d+4 cycles (where d is the code distance). Hadamard gates require patch-deformation techniques, requiring 3d+4 cycles. The S gate requires preparing and injecting a  $|Y\rangle$  state, with the total cost, including both preparation and consumption, taking 1.5d+3 cycles. For the T gate, once a magic state is prepared,

consuming it takes 2.5d+4 cycles. These cost estimates are based on the recent resource estimation study [7].

#### C. Fault Tolerant Quantum Computing: Clifford+T Gates

To execute an FTQC circuit, all gates must be expressed (transpiled) in the Clifford+T basis. Some gates, such as the Toffoli, can be exactly decomposed into Clifford+T gates. However, many gates, especially single-qubit rotations, do not have an exact representation and instead require approximate synthesis into sequences of Clifford+T gates.

The Solovay-Kitaev algorithm [24], [45] was one of the first methods proposed for this synthesis task, offering asymptotically efficient approximations for arbitrary unitary gates. The general algorithm can accept any unitary and generate approximations in any universal gate set that includes inverses for all its gates. It produces a sequence of length  $O(\log^c(1/\epsilon))$  to achieve an approximation error  $\epsilon$ , with practical  $c \approx 3.97$  [24], where c is a constant. Thus, the resulting gate sequences are prohibitively long even for moderate accuracy.

Recent number-theoretic approaches such as GridSynth [64] have dramatically improved synthesis efficiency by specifically targeting single-qubit Z-axis rotations (Rz gates). As shown in Figure 3(c), GridSynth achieves an error below  $10^{-10}$  with just 332 T gates, compared to over 50,000 gates using Solovay-Kitaev. Since any quantum algorithm can be decomposed into Rz plus Clifford gates, GridSynth is adopted as the default synthesis method in our framework, ensuring both accuracy and practical gate counts for large-scale FTQC circuits.

#### III. MOTIVATION

#### <span id="page-2-0"></span>A. The Decadal Shift: T-Gate Costs Down, Clifford Costs Up

Historically, T gates were the primary driver of FTQC overhead, with magic state preparation requiring over  $1000 \times$  more resources than Clifford operations [13], [30]. This motivated a decade of advances in T-gate and magic-state optimization [29], [52]. As shown in Figure 4, recent progress in magic-state cultivation [34] has reduced this overhead by more than  $100 \times$ , bringing T-gate costs close to those of logical CNOT operations. This shift also reflects an Amdahl's law effect: as the T-gate component is accelerated, the remaining Clifford component increasingly limits end-to-end improvement. Thus, once magic-state overhead becomes comparable to

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Fig. 4. Magic state preparation overhead (qubit-cycle volume) has decreased by more than  $100 \times$  since 2012 and is now comparable to logical CNOT operations. Data from [29], [30], [34], [52].

logical Clifford operations, further T-only optimizations yield diminishing returns unless Clifford overhead is also reduced.

For the benchmarks in Table I (see Section V-C for details), which cover key FTQC algorithms, Clifford gates now account for 58-65% of execution overhead, with an average of 60.5%. From an Amdahl's law perspective, even eliminating the T-attributable portion entirely would provide only a bounded speedup of roughly  $1.7\times-2.0\times$  in this regime. As T-gate costs continue to fall, Clifford overhead, once considered negligible, has emerged as a dominant challenge to scaling FTQC.

