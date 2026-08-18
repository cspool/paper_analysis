# Algorithm 2 Loose Scheduling Algorithm

```
Input: Pauli operator sequence S = \{P_1, P_2, \dots, P_l\}, qubit
    number n, Board B
Output: Executable operation sequence S'
 1: Initialize S' = \{ \}
 2: Build Pauli DAG \mathcal{G} from S
 3: while \mathcal{G} is not empty do
 4:
       for P_i \in \mathcal{G}.frontier and P_i is executable do
          Get bus patch list L_{p_i} for P_i
 5:
          Execute P_i on L_{p_i} and update S' \leftarrow S' + P_i
 6:
 7:
       Pop a Pauli operator P_i from \mathcal{G}.frontier
 8:
 9:
       while P_i is not executable do
          Get all possible patch operations O_B from B
10.
          Select o_b \in O_B with the best reward r(o_b, P_i)
11:
          Execute o_b on B and update S' \leftarrow S' + o_b
12:
       end while
13:
14: end while
15: return S'
```

**Determining bus patch.** In Step 5, the objective is to identify the minimal amount of routing space required to execute  $P_i$ , as reducing the overall routing path length not only increases opportunities for parallel execution of other operations but also has the potential to lower the overall logical error rate. To achieve this, we apply Dijkstra's algorithm to sequentially determine the shortest paths between the required patches. Previously identified paths are treated as nodes with zero cost. This process results in a bus patch list with a minimal number of patches used and the complexity of each executed Pauli operator  $P_i$  is  $\mathcal{O}(|B|^2)$ , where |B| denotes the size of board.

**Resolving unexecutable Pauli operator.** In Step 11, the reward function is decomposed into three components. First, it is defined as the number of data patches in  $B_o$  that enable the execution of  $P_i$ ; that is the number of data patches for

which a valid path exists to support the application of  $P_j$ , where  $B_o$  denotes the state of the board B after applying the patch operation o. Second, maintaining connectivity among all data patches is essential for enabling subsequent lattice surgery operations, so any patch operation that breaks this connectivity receives a reward of zero. Third, when multiple patch operations yield the same reward, preference is given to those incur lower time overhead.

Complexity Analysis. Since the number of candidate patch operations for each data patch is constant, the total number of candidates is  $\mathcal{O}(n)$ , where n denotes the number of data patches. Also, the complexity of evaluating the reward function is also  $\mathcal{O}(n)$ . Furthermore, the reward function is designed to guarantee that each patch operation increases the number of patches satisfying the execution requirements by at least one. Consequently, the overall complexity of a complete scheduling process is  $\mathcal{O}(n^2)$ .

## D. O3LS Module 4: Initial Mapping

**Edge-aware Initial Mapping.** Among all patch operations, qubit patch rotations are a major source of time overhead, particularly in squeeze layouts where such operations occur more frequently. To mitigate this overhead, we propose an edge-aware (EA) initial mapping strategy. The core idea is to analyze the PDAG to estimate the rotation demand for each qubit. Qubits with higher expected rotation frequencies are preferentially mapped to patches that are adjacent to both the X and Z-edges of the ancilla patch, thereby minimizing the need for costly patch rotations.

The effectiveness of the EA mapping method is most evident in layout boards that fall between compact and sparse configurations. For instance, in highly compact layouts, changes in qubit positions introduced by the mapping are less likely to expose additional edges, thereby limiting the potential benefits of the approach. In these cases, the performance of the EA strategy becomes highly dependent on the specific structure and gate distribution of the input circuit. Moreover, in sparse qubit patch layouts, the edge-aware component contributes less to performance improvement. This is because, in sparse configurations, each patch can typically expose both X and Z operators simultaneously, reducing the advantage gained from edge-aware placement.

**Complexity Analysis.** The rotation demand required for EA mapping can be efficiently extracted during the construction of PDAG. The process of counting the number of edges exposed by data patches has a computational complexity of  $\mathcal{O}(n)$ , where n denotes the number of data patches. The final mapping is achieved through two applications of quicksort, resulting in an overall complexity of  $\mathcal{O}(n \log n)$ .

## IV. EVALUATION

## A. Experiment Setup

**a) Metrics.** (1). *Logical error rate*. We simulate the logical error rate (LER) by parsing lattice surgery instructions and analyzing them at the time-slice level. We then compute the

![](_page_7_Figure_0.jpeg)

Fig. 10. Performance of executing applications on various data layout sizes using O3LS, including their corresponding time steps, ancilla patch lengths, and logical error rates for adder\_28 and ising\_26. The X and Y axes represent patch boards of size  $N \times M$ , where N and M range from 5 to 12.

layer-wise LER for each time slice and linearly accumulate them over the entire execution:

$$p_{\text{total}} \approx \sum_{t=1}^{T} p_{\text{layer}}^{(t)} \approx \sum_{t=1}^{T} (1 - (1 - P_{\text{PPM}}^{(t)})(1 - P_{\text{PR}}^{(t)})(1 - P_{\text{idle}}^{(t)})),$$

which follows the same method used in SPARO [28] under the assumption of rare failures and independent error events. Each  $p_{\rm layer}^{(t)}$  is estimated from the simulated LER in Pauli product measurement (PPM), patch rotation (PR), and idling memory errors during that layer. The PPM error rate is mainly determined by the routing space and the code distance. The rotation step is decomposed into three slices: patch deformation, corner movement, and patch movement [34], and they are simulated separately. (2). *Time cost.* Following prior work [25], [51], our evaluation also focuses on the time cost ( $\oplus$ ). We also record the size of data layout, which serves as an indicator of the space-time volume [47].

- b) Baseline Algorithms. We evaluate O3LS against the SPC algorithm [34], as implemented in [32], [52], and the recent locality-aware method LAPBC [25], which enhances circuit parallelism and outperforms prior compilers [4], [39]. We also compare against SPARO [28], another automated datalayout design method that aims to expand data layouts.
- c) Benchmarks. We benchmark using a representative set of FT quantum algorithms, following prior FTQC compiler studies [28], [38], [39], [48], [51]. These include circuits for Hamiltonian simulation, Quantum Fourier Transform, key components of Shor's algorithm (e.g., adders and multipliers), and SWAP tests for quantum machine learning, many of which serve as building blocks for larger algorithms. We source the

- QASM files from MQT Bench [41] and FTCircuitBench [21]. Some FTCircuitBench circuits were originally taken from QASMBench [33]. Unless otherwise specified in the circuit name, we assume a one-dimensional Hamiltonian. We evaluate O3LS across different layouts, including the compact design from [34] and standard layouts from [25].
- d) Experimental Setting. The benchmarks are decomposed into Clifford+T circuits using GridSynth [13], based on [42] with a synthesis error tolerance of  $10^{-5}$ . STIM simulations [18] are conducted to characterize atomic lattice surgery operations using a d = 9 surface code under a circuit-level depolarizing noise model with a physical error rate of  $p = 10^{-3}$ . Each atomic operation is independently compiled into a STIM circuit and simulated using Monte Carlo sampling with no less than  $10^6$  trials. Decoding is performed by PyMatching 2 [24]. For all experiments, we use a magicstate factory based on the 15-to-1 distillation protocol [8]. The factory is placed outside the designed layout, while ensuring at least one routing path connects it to the data region. The  $\pi/4$  and  $\pi/8$  Pauli-product measurements are implemented via standard gate teleportation protocol, following the implementation in [34] (Fig. 7 and 11(b)).
- e) Simulation Device. All simulations were performed on a device with an Intel Core i9-14900K 32-core processor and 188 GB of RAM using Python 3.10.

## B. Analysis of Data Layout Designs

a) Performance on Data Layout Designs. We begin by analyzing the performanc across different data layout sizes and their corresponding estimated logical error rates. In this

scenario, we consider patch boards of size  $N \times M$ , where N and M range from 5 to 12, and evaluate the performance of adder\_28 and Ising\_26 circuits across various patch board configurations. The experimental results are presented in Fig. 10, where the heatmaps illustrate the time steps, ancilla path lengths, and their associated RLER.

Our evaluation reveals that when the patch board size is too small, the primary performance bottleneck arises from the overhead associated with operation scheduling, such as patch rotations. As the board size increases, the number of time steps required for execution decreases and eventually converges. However, despite similar time step at larger board sizes, the ancilla patch length increases monotonically due to the availability of more routing space. This leads to an observation that both higher time step and longer ancilla path lengths contribute significantly to increased logical error rates. These results underscore a fundamental trade-off between time costs and ancilla patch distance, indicating that carefully designed, smaller data layouts can contribute to lower logical error rates.

On the other hand, the applications that are required to run on these fixed data layouts necessitate the use of  $10 \times 10$ and  $9 \times 15$  patch boards in the standard and sparse layouts. Compared to these scenarios, O3LS effectively reduces the required space overhead, achieving a board size reduction of up to 28.0% and 46.7%, while preserving the number of time steps. Furthermore, it achieves a reduction in logical error rates of up to 16.9% compared to larger data layouts (e.g.,  $12 \times 12$ ), due to the decreased length of ancilla patches. While larger data layouts can reduce the number of time steps, they often require longer routing paths. In contrast, smaller data layouts are more space-efficient but tend to suffer from higher time costs. Both scenarios can contribute to increased logical error rates. To balance this trade-off, O3LS generates more compact layout designs that minimize ancilla patch length while maintaining time costs comparable to those of sparse layouts, thereby achieving a sweet spot for reducing the overall logical error rate.

b) Sensitivity Analysis on Density Factor. We also perform a sensitivity analysis of the density factor  $\alpha_e$  in the layout design, as introduced in Section III-A, with results presented in Fig. 11. The experiments indicate that values of  $\alpha_e$  between 0.1 and 0.3 yield the best performance across most applications. Additionally, performance remains relatively stable for  $0 < \alpha_e < 0.5$ , but degrades noticeably when  $\alpha_e = 0$  or  $\alpha_e = 0.5$ . This behavior can be attributed to the trade-off governed by  $\alpha_e$ : smaller values prioritize placing patches with multiple edges to ancilla patches, promoting sparse layouts, while larger values emphasize compactness. Both extremes can lead to suboptimal layouts—either overly sparse and causing later qubit patches to obscure certain operators or overly compact and underutilizing available patches. Thus, an appropriate balance of  $\alpha_e$  is essential to optimize the tradeoff between compact and sparse placement in patch utilization. We recommend using  $\alpha_e \in [0.1, 0.3]$  to achieve this balance.

c) Comparison with SPARO. Moreover, other layout-

![](_page_8_Figure_5.jpeg)

Fig. 11. Sensitivity analysis of density factor on automatic layout design. The relative time step is defined as the ratio between the evaluated time step and the baseline time step corresponding to density factor  $\alpha_e = 0.1$ .

design methods such as SPARO [28] explore strategies to improve data-layout utilization. In Fig. 12, we compare O3LS with SPARO's layout-design approach. 'O3LS-1' uses an O3LS-generated data layout with SPARO's scheduling method and 'O3LS' uses the full O3LS stack for both layout generation and scheduling.

![](_page_8_Figure_8.jpeg)

Fig. 12. Layout design comparison with SPARO [28].

O3LS-generated data layouts achieve better performance than SPARO, improving LER by 3.05% on average when paired with SPARO's scheduling method. This gain comes primarily from smaller data layouts, which reduce average ancilla-routing space by 17.35%. In addition, O3LS's scheduler is particularly effective on the smaller data layouts produced by O3LS Module 1. It reduces time steps by 78.24% and average routing space by 27.17% on average, which together yield a further 77.1% reduction in LER. The routingspace savings in O3LS largely stem from its objective. O3LS searches for a sweet spot that minimizes space overhead in the data layout, whereas SPARO tends to allocate more datalayout resources. O3LS also includes more advanced synthesis and a looser scheduling strategy, which further reduces time steps. Overall, O3LS finds smaller layouts that shorten ancilla paths while reducing time steps, resulting in lower LER.

## C. Compilation Technique Comparison

a) Comparison with SPC. We begin by comparing our compiler pass with SPC across two data layout configurations: compact and standard. As shown in Fig. 13, our compiler pass O3LS achieves an average reduction of 36.07% in time steps compared to SPC under the compact layout, and an

![](_page_9_Figure_0.jpeg)

Fig. 13. Compilation technique comparison with SPC. (Left) Results with the compact layout. (Right) Results with the standard layout.

![](_page_9_Figure_2.jpeg)

Fig. 14. (Left) Comparison of time steps between the prior compiler pass LAPBC, which emphasizes parallelism, and our proposed O3LS. (Right) Logical error rate analysis indicates that O3LS achieves superior performance in most scenarios, highlighting the effectiveness of the proposed scheduling and Y-synthesis algorithms.

average reduction of 24.76% under the standard layout. These improvements achieved by O3LS are due to the fact that SPC does not incorporate any optimization techniques, such as scheduling, routing, or synthesis.

b) Comparison with LAPBC. Recent efforts have focused on improving circuit parallelism in lattice surgery [4], [5], [20], [25], with LAPBC [25] as the latest advancement. In Fig. 14 (left), we analyze the time steps required by both compilers. The results show that O3LS, achieves an average time-step reduction of 35.10% compared to LAPBC, with a maximum reduction of up to 80.6%. This results in an average LER reduction of 38.8%, with a maximum reduction of up to 82.3%, as shown in Fig. 14 (right). This highlights the effectiveness of O3LS in optimizing execution schedules with loose scheduling algorithm.

c) Incorporating parallelism into O3LS. While the highly parallel nature of LAPBC provides advantages for sparse data layouts, O3LS can also be integrated into LAPBC to further reduce time costs. In particular, we focus on several high-parallelism benchmarks where LAPBC is expected to perform well. By incorporating additional modules such as loose scheduling and advanced initial mapping algorithms, we further improve performance and demonstrate the effectiveness of the proposed methods. A detailed breakdown of the results is shown in Fig. 15, where our integrated approach achieves an average improvement of 9.31%.

![](_page_9_Figure_7.jpeg)

Fig. 15. Time step reduction by integrating O3LS with high-parallelism execution strategies.

## D. Initial Mapping Comparison

Furthermore, we analyze the initial mapping methods proposed in O3LS and compare them with the previous greedy mapping approach from [28]. We use the data layouts generated by O3LS to evaluate these initial mappings. As shown in Fig. 16, the edge-aware mapping outperforms the previous approach, achieving a time step reduction of 15.0% and a logical error rate reduction of 8.4%. These savings primarily result from the edge-aware mapping's tendency to place qubits with higher rotation demands in patches where both X and Z operators are adjacent to an ancilla patch, thereby reducing the need for costly rotations. This strategy is especially effective for the squeezed layouts produced by O3LS.

![](_page_10_Figure_0.jpeg)

Fig. 16. Initial mapping comparison.

#### E. Overall Performance Comparison

a) Overall Performance. After comparing individual components, we directly compare the overall performance of O3LS with compiler passes including SPC and LAPBC in terms of LER. The exact LER results are presented in Fig. 18. Compared to SPC, O3LS suppresses the logical error rate by 43.11% and 44.98% on compact and standard layouts, respectively (corresponding to a reduction by roughly half). Notably, in certain cases, O3LS achieves a maximum LER reduction of 93.95% compared to LAPBC, approaching an order of magnitude error suppression.

Since the simulated surface code distances and the level of improvement vary across applications, we also record the results and normalize them into relative LER values by setting the highest LER for each application as the baseline to clearly show the improvements. Overall, O3LS outperforms SPC, reducing relative LER by 35.9% and 21.8% on compact and standard layouts, respectively. Similarly, compared to LAPBC, O3LS achieves relative LER reductions of 50.9% and 31.1%.

Overall, O3LS outperforms previous compiler passes primarily due to all module design choices. First, it identifies the trade-off between scheduling overhead and ancilla path length during layout design, leading to more efficient data layouts. It also applies circuit synthesis techniques to reduce the total number of operations, thereby lowering time overhead associated with rotating data patches in the O3LS-generated layouts. Second, rather than relying on fixed scheduling schemes, O3LS introduces a flexible scheduling strategy that minimize scheduling costs. Third, it incorporates effective initial mapping techniques that improve both the routing efficiency. Together, these strategies not only reduce time overhead but also minimize the need for long ancilla paths, resulting in suppression of LER.

b) Sensitivity Analysis on Code Distance. Fig. 19 shows a sensitivity analysis over surface code distance  $d \in [3, 5, 7, 9]$ . The results show that O3LS consistently outperforms all previous compilers across all tested code distances. For the Ising\_n26 and swap\_test\_n25 benchmark, O3LS consistently achieves over 19.96% and 13.42% improvement in RLER compared to SPC and LAPBC, respectively. Crucially, this relative improvement remains stable as the code distance increases. This is because, under the layer-wise accumulation model that treats logical failures as independent rare events [28], O3LS optimizes only architectural factors, while

the distance-dependent exponential suppression from decoding applies equally to O3LS and the baseline. Consequently, our gains are not tied to any particular code distance.

c) Analysis on Resource Estimation. After demonstrating robustness across code distances and noise models in improving LER for all benchmarks, Fig. 20 (upper) reports resource estimates for overall time (number of syndrome-measurement cycles) and space (physical qubit count) savings achieved by O3LS relative to SPC. Overall, O3LS reduces space and time simultaneously, delivering an average 23.63% improvement compared with the prior compiler and thereby suppressing LER (Fig. 20 (lower right)) through automated layout design and scheduling.

Fig. 20 (lower left) further quantifies the space savings using the surface code with d=9 as an example. Because each tile corresponds to one surface-code logical qubit, reducing the tile count directly reduces the number of physical qubits. In our benchmarks, O3LS achieves up to a 44% space reduction, which corresponds to saving roughly 7000 physical qubits. This benefit becomes even more pronounced at larger surface-code distances. This highlights the usefulness of O3LS, which reduces not only time steps but also the physical-qubit requirements for executing fault-tolerant algorithms on hardware.

## F. Ablation Study of Compilation Techniques

Fig. 21 presents a comparative analysis of the individual modules in O3LS, using the data layout generated by the 'O3LS-1' configuration as the test case. 'O3LS-2' refers to the compiler pass that utilizes only the Module 2 Y-synthesis algorithm described in Sec. III-B. Compared with prior compiler passes, 'O3LS-2' improves time steps by 18.33% and LER by 18.30%, highlighting the potential of operator cancellation for optimizing circuit execution. With the integration of loose scheduling, 'O3LS-2+3' achieves an average improvement of 37.74% in time steps and 34.34% in LER, demonstrating the added benefit of loose scheduling. Finally, incorporating initial mapping technique in 'O3LS-2+3+4' further improves performance, yielding an average improvement of 38.62% in time steps and 35.17% in LER. The results highlight that combining Y-synthesis, loose scheduling, and edge-aware mapping reduces execution time and improves lattice surgery compilation efficiency.

## G. Compilation Time Analysis

We compare the compilation time across different compilers in Fig. 22 (left). O3LS achieves faster compilation times than SPC and delivers comparable performance to LAPBC. In some cases, O3LS is slightly slower than LAPBC, which benefits from maximizing parallelism and avoiding the overhead of absorbing Pauli operators into the final measurement. Although O3LS performs explicit Pauli operator transformations, it leverages the O3LS-IR to accelerate this process more effectively than SPC, resulting in compilation times that remain competitive with LAPBC. In Fig.22 (right), we also demonstrate the scalability of O3LS with respect to the

![](_page_11_Figure_0.jpeg)

Fig. 17. Relative logical error rate comparison (lower is better). O3LS vs. prior compilers with fixed compact (left) or standard (right) data layouts.

![](_page_11_Figure_2.jpeg)

Fig. 18. Logical error rate comparison with prior compilers for (left) compact and (right) standard layouts (lower is better).

![](_page_11_Figure_4.jpeg)

Fig. 19. Performance of O3LS scale with code distance  $d \in [3, 5, 7, 9]$ .

number of qubits, showing that its compilation time scales polynomially, as analyzed in Section III.

## H. Optimality Analysis

We further conduct an optimality analysis, with the results presented in Fig. 23. Due to the NP-hardness of the problem [23], our analysis focuses only on small cases, where the optimal LER can be determined through brute-force enumeration. Overall, O3LS achieves an average gap of just 4.20% from the optimal, demonstrating the effectiveness of O3LS.

## V. COMPARISON WITH PRIOR ART

a) Lattice Surgery Compilers. Prior work has explored quadratic assignment [30] and SAT formulations [39] for data layout assignment and scheduling. However, lattice surgery optimization is NP-hard [23], requiring scalable solutions.

![](_page_11_Figure_11.jpeg)

![](_page_11_Figure_12.jpeg)

Fig. 20. Analysis of resource estimation.

General compilers [32], [52] miss key optimization opportunities in synthesis, mapping, and scheduling. Our work addresses these gaps via improved loose scheduling methods and Y-synthesis algorithms. Meanwhile, methods to enhance parallelism [4], [5], [20], [25] show strong results in specific cases. O3LS achieves comparable performance and can integrate these techniques for further gains.

- b) Lattice Surgery Pipeline. [51] proposed TACO to reduce Clifford cost by minimizing Pauli-Z rotations. [12] introduced Q-Spellbook for selecting data block layouts and distillation protocols under various strategies. Our work focuses on a different but complementary aspect, and their techniques could be integrated into O3LS to further reduce overall cost.
- c) Data Layout Design. Designs with a similar setting. In the context of manually designed layouts, [11] proposed a 4/9 filling layout, while [4] and [5] introduced 1/4 and 1/2 filling layouts. Although these designs ensure that any logical operation on the target data patches can be executed, they often overlook opportunities for optimizing logical error rates. In the context of automated layout design, [28] automatically

![](_page_12_Figure_0.jpeg)

Fig. 21. Ablation study of compilation techniques. O3LS-2 refers to the use of Y-synthesis algorithm without additional scheduling methods. O3LS-2+3 incorporates loose scheduling in addition to Y-synthesis, while O3LS-2+3+4 integrates both initial mapping and loose scheduling alongside the Y-synthesis.

![](_page_12_Figure_2.jpeg)

Fig. 22. Compilation time analysis. (Left) Comparison with previous compilers. (Right) Scalability with respect to the number of logical qubits.

![](_page_12_Figure_4.jpeg)

Fig. 23. Optimality analysis.

enlarges the underlying data layout based on the analyzed bottleneck. LaSsynth [47] proposes a SAT-based solver that can optimally handle a limited number of qubits and operations. However, its scalability is limited. Our aim is to develop an automated and scalable compiler for finding squeezed layouts. Heterogeneous QEC designs. [45] introduces a toolbox for heterogeneous quantum architectures on superconducting devices, while [6], [46], [54] propose hybrid approaches that combine surface codes with qLDPC codes, leveraging their complementary strengths by assigning different codes to memory and computation regions. In all cases, surface-code architectures remain central, and our work can potentially offer improved pipelines to enhance their performance.

d) Mapping QEC codes into hardware. [53] presents a synthesis framework for surface codes on superconducting devices, while [57] extends this to stabilizer code mapping. [31], [56] explore surface code mapping on trapped-ion devices. These studies primarily focus on the lower layer of mapping QEC codes to physical hardware. Our work is orthogonal to these efforts, and integrating both layers has the potential to further reduce logical error rates.

#### VI. CONCLUSION

We present O3LS, a compiler that suppresses logical error rates by optimizing both space and time overhead for lattice surgery operations. It produces data layouts that minimize space overhead while maintaining time costs comparable to those of sparser layouts, thereby supporting the goal of reducing logical error rates. O3LS achieves this through loose scheduling, Y operator synthesis, and initial mapping tailored to the proposed layout architecture. The numerical results demonstrate that O3LS could outperform prior works in terms of error rates, time costs, and qubit resource overhead.

## ACKNOWLEDGMENT

We would like to thank the anonymous reviewers for their helpful feedback and suggestions.

## REFERENCES

- [1] R. Acharya, D. A. Abanin, L. Aghababaie-Beni, I. Aleiner, T. I. Andersen, M. Ansmann, F. Arute, K. Arya, A. Asfaw, N. Astrakhantsev *et al.*, "Quantum error correction below the surface code threshold," *Nature*, 2024.
- [2] G. Q. AI *et al.*, "Quantum error correction below the surface code threshold," *Nature*, vol. 638, no. 8052, p. 920, 2024.
- [3] F. Arute, K. Arya, R. Babbush, D. Bacon, J. C. Bardin, R. Barends, R. Biswas, S. Boixo, F. G. Brandao, D. A. Buell *et al.*, "Quantum supremacy using a programmable superconducting processor," *Nature*, vol. 574, no. 7779, pp. 505–510, 2019.
- [4] M. Beverland, V. Kliuchnikov, and E. Schoute, "Surface code compilation via edge-disjoint paths," *PRX Quantum*, vol. 3, no. 2, p. 020342, 2022.
- [5] M. E. Beverland, P. Murali, M. Troyer, K. M. Svore, T. Hoefler, V. Kliuchnikov, G. H. Low, M. Soeken, A. Sundaram, and A. Vaschillo, "Assessing requirements to scale to practical quantum advantage," *arXiv preprint arXiv:2211.07629*, 2022.
- [6] S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder, "High-threshold and low-overhead fault-tolerant quantum memory," *Nature*, vol. 627, no. 8005, pp. 778–782, 2024.
- [7] S. Bravyi and J. Haah, "Magic-state distillation with low overhead," *Physical Review A*, vol. 86, no. 5, Nov. 2012. [Online]. Available: http://dx.doi.org/10.1103/PhysRevA.86.052329
- [8] S. Bravyi and A. Kitaev, "Universal quantum computation with ideal clifford gates and noisy ancillas," *Physical Review A*, vol. 71, no. 2, Feb. 2005. [Online]. Available: http://dx.doi.org/10.1103/PhysRevA.71.022316
- [9] L. Caune, L. Skoric, N. S. Blunt, A. Ruban, J. McDaniel, J. A. Valery, A. D. Patterson, A. V. Gramolin, J. Majaniemi, K. M. Barnes *et al.*, "Demonstrating real-time and low-latency quantum error correction with superconducting qubits," *arXiv preprint arXiv:2410.05202*, 2024.
- [10] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, J. R. McClean, K. Mitarai, X. Yuan, L. Cincio *et al.*, "Variational quantum algorithms," *Nature Reviews Physics*, vol. 3, no. 9, pp. 625– 644, 2021.
- [11] C. Chamberland and E. T. Campbell, "Universal quantum computing with twist-free and temporally encoded lattice surgery," *PRX Quantum*, vol. 3, no. 1, p. 010331, 2022.
- [12] A. Chatterjee, A. Ghosh, and S. Ghosh, "The q-spellbook: Crafting surface code layouts and magic state protocols for large-scale quantum computing," 2025. [Online]. Available: https://arxiv.org/abs/2502.11253
- [13] Q. Community, "qiskit-gridsynth-plugin: A plugin for integrating gridsynth angle decomposition into qiskit," https://pypi.org/project/qiskit-gridsynth-plugin/, 2024, version 0.0.8, released August 29, 2024.
- [14] C. M. Dawson and M. A. Nielsen, "The solovay-kitaev algorithm," 2005. [Online]. Available: https://arxiv.org/abs/quant-ph/0505030
- [15] S. Ebadi, A. Keesling, M. Cain, T. T. Wang, H. Levine, D. Bluvstein, G. Semeghini, A. Omran, J.-G. Liu, R. Samajdar *et al.*, "Quantum optimization of maximum independent set using rydberg atom arrays," *Science*, vol. 376, no. 6598, pp. 1209–1215, 2022.
- [16] A. Eickbusch, M. McEwen, V. Sivak, A. Bourassa, J. Atalaya, J. Claes, D. Kafri, C. Gidney, C. W. Warren, J. Gross *et al.*, "Demonstrating dynamic surface codes," *arXiv preprint arXiv:2412.14360*, 2024.
- [17] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A—Atomic, Molecular, and Optical Physics*, vol. 86, no. 3, p. 032324, 2012.
- [18] C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum*, vol. 5, p. 497, 2021.
- [19] D. Gottesman, *Stabilizer codes and quantum error correction*. California Institute of Technology, 1997.
- [20] K. Hamada, Y. Suzuki, and Y. Tokunaga, "Efficient and highperformance routing of lattice-surgery paths on three-dimensional lattice," 2024. [Online]. Available: https://arxiv.org/abs/2401.15829
- [21] A. Harkness, S. Kan, C. Liu, M. Wang, J. M. Martyn, S. Xu, D. Chamaki, E. Decker, Y. Mao, L. F. Zuluaga, T. Terlaky, A. Li, and S. Stein, "Ftcircuitbench: A benchmark suite for fault-tolerant quantum compilation and architecture," 2026. [Online]. Available: https://arxiv.org/abs/2601.03185

- [22] A. W. Harrow, A. Hassidim, and S. Lloyd, "Quantum algorithm for linear systems of equations," *Physical review letters*, vol. 103, no. 15, p. 150502, 2009.
- [23] D. Herr, F. Nori, and S. J. Devitt, "Optimization of lattice surgery is np-hard," *Npj quantum information*, vol. 3, no. 1, p. 35, 2017.
- [24] O. Higgott and C. Gidney, "Sparse blossom: correcting a million errors per core second with minimum-weight matching," *Quantum*, vol. 9, p. 1600, 2025.
- [25] Y. Hirano and K. Fujii, "Locality-aware pauli-based computation for local magic state preparation," 2025. [Online]. Available: https://arxiv.org/abs/2504.12091
- [26] D. Horsman, A. G. Fowler, S. Devitt, and R. Van Meter, "Surface code quantum computing by lattice surgery," *New Journal of Physics*, vol. 14, no. 12, p. 123011, 2012.
- [27] C. Jones, "Multilevel distillation of magic states for quantum computing," *Physical Review A*, vol. 87, no. 4, Apr. 2013. [Online]. Available: http://dx.doi.org/10.1103/PhysRevA.87.042305
- [28] S. Kan, Z. Du, C. Liu, M. Wang, Y. Ding, A. Li, Y. Mao, and S. Stein, "Sparo: Surface-code pauli-based architectural resource optimization for fault-tolerant quantum computing," 2025. [Online]. Available: https://arxiv.org/abs/2504.21854
- [29] S. Krinner, N. Lacroix, A. Remm, A. Di Paolo, E. Genois, C. Leroux, C. Hellings, S. Lazar, F. Swiadek, J. Herrmann *et al.*, "Realizing repeated quantum error correction in a distance-three surface code," *Nature*, vol. 605, no. 7911, pp. 669–674, 2022.
- [30] L. Lao, B. van Wee, I. Ashraf, J. Van Someren, N. Khammassi, K. Bertels, and C. G. Almudever, "Mapping of lattice surgery-based quantum circuits on surface code architectures," *Quantum Science and Technology*, vol. 4, no. 1, p. 015005, 2018.
- [31] T. LeBlond, R. S. Bennink, J. G. Lietz, and C. M. Seck, "Tiscc: A surface code compiler and resource estimator for trapped-ion processors," in *Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis*, 2023, pp. 1426–1435.
- [32] T. LeBlond, C. Dean, G. Watkins, and R. Bennink, "Realistic cost to execute practical quantum circuits using direct clifford+ t lattice surgery compilation," *ACM Transactions on Quantum Computing*, 2023.
- [33] A. Li, S. Stein, S. Krishnamoorthy, and J. Ang, "Qasmbench: A lowlevel quantum benchmark suite for nisq evaluation and simulation," *ACM Transactions on Quantum Computing*, vol. 4, no. 2, pp. 1–26, 2023.
- [34] D. Litinski, "A game of surface codes: Large-scale quantum computing with lattice surgery," *Quantum*, vol. 3, p. 128, Mar. 2019. [Online]. Available: http://dx.doi.org/10.22331/q-2019-03-05-128
- [35] ——, "Magic state distillation: Not as costly as you think," *Quantum*, vol. 3, p. 205, Dec. 2019. [Online]. Available: http://dx.doi.org/10.22331/q-2019-12-02-205
- [36] M. Liu, R. Shaydulin, P. Niroula, M. DeCross, S.-H. Hung, W. Y. Kon, E. Cervero-Mart´ın, K. Chakraborty, O. Amer, S. Aaronson *et al.*, "Certified randomness using a trapped-ion quantum processor," *Nature*, pp. 1–6, 2025.
- [37] J. F. Marques, B. Varbanov, M. Moreira, H. Ali, N. Muthusubramanian, C. Zachariadis, F. Battistel, M. Beekman, N. Haider, W. Vlothuizen *et al.*, "Logical-qubit operations in an error-detecting surface code," *Nature Physics*, vol. 18, no. 1, pp. 80–86, 2022.
- [38] S. Maurya, A. Molavi, A. Albarghouthi, and S. Tannu, "Managing classical processing requirements for quantum error correction," *arXiv preprint arXiv:2406.17995*, 2024.
- [39] A. Molavi, A. Xu, S. Tannu, and A. Albarghouthi, "Dependency-aware compilation for surface code quantum architectures," *Proceedings of the ACM on Programming Languages*, vol. 9, no. OOPSLA1, pp. 57–84, 2025.
- [40] J. Preskill, "Quantum computing in the nisq era and beyond," *Quantum*, vol. 2, p. 79, 2018.
- [41] N. Quetschlich, L. Burgholzer, and R. Wille, "Mqt bench: Benchmarking software and design automation tools for quantum computing," *Quantum*, vol. 7, p. 1062, Jul. 2023. [Online]. Available: http://dx.doi.org/10.22331/q-2023-07-20-1062
- [42] N. J. Ross and P. Selinger, "Optimal ancilla-free clifford+t approximation of z-rotations," 2016. [Online]. Available: https://arxiv.org/abs/1403.2975
- [43] P. W. Shor, "Scheme for reducing decoherence in quantum computer memory," *Physical review A*, vol. 52, no. 4, p. R2493, 1995.

- [44] ——, "Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer," *SIAM review*, vol. 41, no. 2, pp. 303–332, 1999.
- [45] S. Stein, S. Sussman, T. Tomesh, C. Guinn, E. Tureci, S. F. Lin, W. Tang, J. Ang, S. Chakram, A. Li *et al.*, "Hetarch: Heterogeneous microarchitectures for superconducting quantum systems," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, 2023, pp. 539–554.
- [46] S. Stein, S. Xu, A. W. Cross, T. J. Yoder, A. Javadi-Abhari, C. Liu, K. Liu, Z. Zhou, C. Guinn, Y. Ding *et al.*, "Hetec: Architectures for heterogeneous quantum error correction codes," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, 2025, pp. 515–528.
- [47] D. B. Tan, M. Y. Niu, and C. Gidney, "A sat scalpel for lattice surgery: Representation and synthesis of subroutines for surface-code fault-tolerant quantum computing," in *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*. IEEE, Jun. 2024, p. 325–339. [Online]. Available: http://dx.doi.org/10.1109/ISCA59077.2024.00032
- [48] T. Trochatos, C. Kang, A. Wang, F. T. Chong, and J. Szefer, "Tracebased reconstruction of quantum circuit dataflow in surface codes," *arXiv preprint arXiv:2508.14533*, 2025.
- [49] Y. Ueno, T. Saito, T. Tanimoto, Y. Suzuki, Y. Tabuchi, S. Tamate, and H. Nakamura, "High-performance and scalable fault-tolerant quantum computation with lattice surgery on a 2.5 d architecture," *arXiv preprint arXiv:2411.17519*, 2024.
- [50] K. Wang, Z. Lu, C. Zhang, G. Liu, J. Chen, Y. Wang, Y. Wu, S. Xu, X. Zhu, F. Jin *et al.*, "Demonstration of low-overhead quantum error correction codes," *arXiv preprint arXiv:2505.09684*, 2025.
- [51] M. Wang, C. Liu, S. Stein, Y. Ding, P. Das, P. J. Nair, and A. Li, "Optimizing ftqc programs through qec transpiler and architecture codesign," *arXiv preprint arXiv:2412.15434*, 2024.
- [52] G. Watkins, H. M. Nguyen, K. Watkins, S. Pearce, H.-K. Lau, and A. Paler, "A high performance compiler for very large scale surface code computations," *Quantum*, vol. 8, p. 1354, 2024.
- [53] A. Wu, G. Li, H. Zhang, G. G. Guerreschi, Y. Ding, and Y. Xie, "A synthesis framework for stitching surface code with superconducting quantum devices," in *Proceedings of the 49th Annual International Symposium on Computer Architecture*, 2022, pp. 337–350.
- [54] Q. Xu, J. P. Bonilla Ataides, C. A. Pattison, N. Raveendran, D. Bluvstein, J. Wurtz, B. Vasic, M. D. Lukin, L. Jiang, and H. Zhou, "Constant- ´ overhead fault-tolerant quantum computation with reconfigurable atom arrays," *Nature Physics*, vol. 20, no. 7, pp. 1084–1090, 2024.
- [55] J. Yi, W. Ye, D. Gottesman, and Z.-W. Liu, "Complexity and order in approximate quantum error-correcting codes," *Nature Physics*, vol. 20, no. 11, pp. 1798–1803, 2024.
- [56] K. Yin, X. Fang, Z. Chen, A. Li, D. Hayes, E. Kaur, R. Nejabati, H. Haeffner, W. Campbell, E. Hudson *et al.*, "Flexion: Adaptive insitu encoding for on-demand qec in ion trap systems," *arXiv preprint arXiv:2504.16303*, 2025.
- [57] K. Yin, H. Zhang, X. Fang, Y. Shi, T. S. Humble, A. Li, and Y. Ding, "Qecc-synth: A layout synthesizer for quantum error correction codes on sparse architectures," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, 2025, pp. 876–890.
- [58] Y. Zhao, Y. Ye, H.-L. Huang, Y. Zhang, D. Wu, H. Guan, Q. Zhu, Z. Wei, T. He, S. Cao *et al.*, "Realization of an error-correcting surface code with superconducting qubits," *Physical Review Letters*, vol. 129, no. 3, p. 030501, 2022.