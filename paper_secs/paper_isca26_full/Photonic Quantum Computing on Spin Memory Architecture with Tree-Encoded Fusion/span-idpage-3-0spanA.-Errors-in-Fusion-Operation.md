# <span id="page-3-0"></span>*A. Errors in Fusion Operation*

In type-II fusion, the two dominant error sources are *fusion failure* and *fusion erasure*. Although both arise from the same imperfect fusion primitive, they differ in a key way: fusion failure leads to a *known* graph transformation, whereas fusion erasure leads to an *unknown* graph outcome.

As shown in Fig. [1,](#page-2-1) fusion failure is heralded when two fusion qubits are captured in the same detector, indicating that the desired entanglement is not created. In this case, the failed qubits are effectively measured in the Z basis and disconnected from the graph. Therefore, although the fusion attempt is unsuccessful, the resulting graph structure remains known to the compiler. This is the failure model considered in previous compilers such as OneAdapt [\[74\]](#page-15-2) and OnePerc [\[72\]](#page-15-1).

In contrast, fusion erasure is triggered by photon loss during fusion, where one fusion qubit cannot be captured by the detector, as shown in Fig. [1.](#page-2-1) The erased qubit is no longer accessible for computation, and its effect cannot be removed by a direct Z measurement. More importantly, the output graph state of the fusion becomes uncertain, since it is unknown whether the entanglement has been established or not. Such uncertainty is especially harmful to MBQC, because later measurements rely on the exact graph-state structure; therefore, the corrupted fusion output must be discarded unless additional protection is applied.

These two errors are also closely related in boosted-fusion design. A common way to suppress fusion failure is to introduce more fusion attempts, but each extra attempt also exposes more qubits to photon loss and thus increases the chance of fusion erasure. As a result, improving tolerance to fusion failure alone is insufficient under realistic photon-loss conditions. To demonstrate the impact of erasure, Fig. [3](#page-3-1) shows a simulation of a Max-Cut QAOA program under different erasure rates using the previous best fusion scheme. The results show that erasure undermines quantum programs in two aspects: (i) it increases the number of fusion attempts, leading to exponentially longer execution time; (ii) the longer execution time accumulates higher decoherence and CZ errors, resulting in lower-quality outputs and larger program-level overhead, such as more tuning iterations in QAOA.

## *B. Problems in Previous SOTA Compilers*

OneAdapt [\[74\]](#page-15-2) and RLGS [\[38\]](#page-14-3) are the SOTA compilers for all-photonic and emitter-based architectures, respectively. Though carefully designed, there are still several gaps from

![](_page_3_Figure_9.jpeg)

<span id="page-3-1"></span>Fig. 3. Optimizing a Max-Cut problem using 6-qubit QAOA program on PQC simulator [\[24\]](#page-13-20). We use the RUS boosted fusion method (m = 6), and simulate the fusion erasure at 0, 5% and 10% respectively, while fixing the fusion failure at 25%. Left: Optimization of QAOA expectation value. Right: Quantum circuit execution time per tuning iteration.

implementing the realistic error-tolerant MBQC, and we conclude their existing problems in Table. [I.](#page-4-1)

OneAdapt iteratively generates resource state layers (RSL) and normalizes them into effective 2D layers of lattice graph states to create the target graph state of a quantum program. This strategy resolves the fusion failure problem, but it overlooks the fusion erasure, which induces errors in the 2D graph state layers. Assuming an 1% erasure rate, generating the required 84 × 84 RSL will demand > 10<sup>5</sup> fusion operations, leading to an extremely low probability of not experiencing erasure in the whole RSL. Furthermore, the normalization method results in a low utilization rate of photons. For example, OneAdapt normalizes only a 4 × 4 2D layer from the 84 × 84 qubits RSL [\[74\]](#page-15-2).

The problem of RLGS primarily lies in the bottleneck of emitter-based architecture hardware. Up to now, only the generation of linear graph states with few qubits from a single quantum emitter has been experimentally demonstrated [\[13\]](#page-13-21), [\[59\]](#page-14-17). The hardware bottleneck arises from the inability to demonstrate high-quality CZ interactions between two emitters [\[26\]](#page-13-10), which is essential for generating MBQC graph states.

## *C. Potentials in Spin Memory Architecture*

We list our insights on addressing the above challenges in the MBQC compiler, leveraging the spin memory architecture:

- (1) The caterpillar state structure offers the chance to resist fusion failure by embedding specific graph state patterns, known as the boosted fusion scheme [\[25\]](#page-13-3). Since the above scheme failed to deal with fusion erasure, in this work, we explore the graph state pattern that tolerates both failure and erasure, while tailoring the pattern to the caterpillar state (Sec. [IV\)](#page-4-0).
- (2) Equipped with the error-tolerant graph state pattern, we can improve the fusion success rate. As a result, we have no need for excessive photon sources and apply normalization like OneAdapt. In contrast, we arrange the generation of caterpillar states from photon sources to be *program-agnostic*, while the structure of each caterpillar state is on demand and determined by the target graph state. Hence, we can improve the utilization rate of photon sources, with details in Sec. [V.](#page-6-0)

| Architecture  | Photonic Hardware       | Error Types           | Existing Problem                                    | Prior Compiler Framework |
|---------------|-------------------------|-----------------------|-----------------------------------------------------|--------------------------|
| all-photonic  | linear optics           | fusion failure        | <ul> <li>Low utilization rate of photons</li> </ul> | OneAdapt [74],           |
|               |                         | fusion erasure        | <ul> <li>Fusion erasure error unsolved</li> </ul>   | • FCM [46]               |
| emitter-based | quantum emitter         | emitter decohenrence  | Bottleneck of experimentally                        | • RLGS [38],             |
|               |                         | emitter-CZ infidelity | demonstrating the emitter-CZ                        | • GSDiv [55]             |
| spin memory   | quantum spin memory +   | fusion failure        | Fusion erasure error unsolved                       | None                     |
|               | linear optical hardware | fusion erasure        | rusion crasure error unsorved                       |                          |

TABLE I
COMPARISON BETWEEN PHOTONIC QUANTUM COMPUTING ARCHITECTURES.

<span id="page-4-1"></span>(3) Our compiler considers the hardware settings from real experiments, gaining more robustness and being more achievable in near-term PQC. Compared to the emitter-based architecture, which still has unsolved hardware barriers, the spin memory architecture is accessible on cloud platforms [2], [24].

