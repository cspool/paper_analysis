# Coset Ensemble Decoder for Quantum Error Correction with Algorithm–Hardware Co-Design

Shuang Liang\*, Jubo Xu\*, Giulio Bassanino\*, Qianzhou Wang\*, Yidong Zhou<sup>†</sup>, Yuncheng Lu\*, Zhiwen Mo\*, Paul H. J. Kelly\*, Bo Yuan<sup>†</sup>, Wayne Luk\*, and Hongxiang Fan\*

\*Imperial College London, United Kingdom <sup>†</sup>Rutgers University, United States {shuang.liang, jubo.xu20, p.kelly, w.luk, hongxiang.fan}@imperial.ac.uk

Abstract—Reliable large-scale quantum computation relies on fault-tolerant architectures, where quantum error correction (QEC) continuously extracts and decodes error syndromes in real time. A critical component in QEC is the decoder, a classical subsystem that must simultaneously deliver high logical accuracy and ultra-low latency. This paper presents a novel algorithmhardware co-design that improves the accuracy-latency trade-off over existing approaches such as vanilla Minimum-Weight Perfect Matching (MWPM) and Union-Find (UF) decoders. At the algorithmic level, we introduce coset ensemble decoding, which improves UF decoding by explicitly exploiting logically equivalent cosets. Our method performs ensemble forest exploration to generate multiple coset-consistent candidates and aggregates them to approximate coset-level maximum-likelihood decoding. We further reduce computational and memory complexity via reverseorder elimination and lossless graph compression, without sacrificing accuracy. At the hardware level, we design a domainspecific architecture that temporally reuses resources, avoiding the code-distance-proportional resource growth in prior spatial architectures. Several optimizations, such as multi-bank memory hashing and hierarchical ID mapping, are proposed to mitigate pipeline stalls and memory conflicts under highly concurrent access patterns. Under a circuit-level depolarizing noise model, our co-design approach achieves a better accuracy-latency tradeoff than prior MWPM- and UF-based decoders, while reducing FPGA LUT consumption by up to 8.2 times compared with reported UF-based decoder resources. The tunable candidate number further exposes a flexible design knob, enabling users to tailor decoding performance to the requirements of different fault-tolerant workloads. Our implementation is publicly available at https://github.com/IMSeonL/coset-ensemble-decoder.

Index Terms—Quantum error correction, coset ensemble decoding, Union-Find decoder, algorithm-hardware co-design, FPGA acceleration

#### I. INTRODUCTION

Recent advances in quantum computing have led to rapid growth in the number of physical qubits. However, existing quantum devices continue to suffer from gate errors, crosstalk, leakage, and limited coherence times [1], all of which hinder the execution of practical quantum circuits at scale. Achieving practical, large-scale quantum computation therefore requires fault-tolerant quantum computing (FTQC), which relies on quantum error correction (QEC) to encode logical qubits across multiple physical qubits. QEC continuously extracts

Accepted to appear in the 53rd Annual International Symposium on Computer Architecture (ISCA 2026).

error syndromes and corrects errors during runtime [2], suppressing the logical error rate below a desired threshold as guaranteed by the threshold theorem [3], [4].

At the core of QEC is the decoder, a classical subsystem that interprets error syndromes and generates corrections in real time. Designing such decoders is challenging due to the dual demands of high decoding accuracy and ultra-low latency. As quantum processors operate at extremely high speeds, decoders are typically required to process syndromes within less than 1 us in superconducting circuits, posing significant challenges for both decoding algorithm design and hardware implementation. Among various QEC codes, the surface code has emerged as a leading candidate due to its high threshold and local stabilizer structure [5]. Within the surface code framework, two prominent decoding strategies have been extensively adopted: Minimum-Weight Perfect Matching (MWPM) [6] and Union-Find (UF) [7]. MWPM achieves high accuracy by solving an optimal matching problem over the syndrome graph, but its high computational cost and inherent sequential nature of computation limit its processing speed. In contrast, the UF decoder achieves lower latency by clustering defects in parallel yet sacrifices accuracy under complex noise conditions.

To address these challenges, this work proposes an algorithm–hardware co-design for accurate, low-latency, and flexible QEC decoding. At the algorithmic level, we introduce coset ensemble decoding that enhances traditional UF decoding by considering logical-equivalent cosets, the group of physical errors with the same syndrome and logical effect. At the hardware level, we develop a customized architecture tailored to the proposed algorithm. In contrast to prior work [8], [9], which relies on spatial architectures whose hardware resources grow with code distance, our hardware features high generality and efficiency through temporal reuse and targeted pipeline optimizations. Together, these choices improve the accuracy–resource trade-off while enabling tunable decoding accuracy for different deployment needs.

This paper makes the following main contributions. The first improves decoding accuracy, while the latter two reduce decoding cost and latency, and improve hardware efficiency.

A novel coset ensemble decoding algorithm that leverages logically equivalent cosets to approximate coset-level maximum-likelihood decoding. This is achieved by en-

semble forest exploration, which effectively improves the algorithmic performance.

- Several algorithmic optimizations, including reverse-order elimination and lossless graph compression, together with an optimality analysis of the proposed algorithm, to further reduce computational complexity.
- A domain-specific hardware architecture tailored for coset ensemble decoding. It achieves high generality and efficiency through temporal reuse and targeted pipeline optimizations. Several hardware optimizations, such as multibank memory hashing and hierarchical ID mapping, are proposed to further improve hardware efficiency and reduce latency.

#### II. BACKGROUND AND MOTIVATION

#### A. Background

- 1) Quantum Error Correction (QEC): QEC employs protocols to protect quantum information from decoherence and noise [10]. An [[n,k,d]] QEC code encodes k logical qubits into n physical qubits (n>k), overcoming fundamental constraints, like the No-Cloning Theorem and projective measurements' destructive nature, through non-local encoding and indirect syndrome extraction using ancilla qubits. The code distance d, defined as the minimum number of physical qubits supporting an undetectable logical error, quantifies error resilience. The logical error rate can be exponentially suppressed by increasing d if the physical error rate is below a threshold.
- 2) Stabilizer Code: An [[n,k]] stabilizer code has stabilizer group  $\mathcal{S} \subset \mathcal{P}_n$  generated by m=n-k independent commuting Pauli operators. One may choose a canonical generating set  $\{S_g, T_g, \bar{X}_j, \bar{Z}_j\}$  [11], where  $S_g$  are stabilizer generators,  $\bar{X}_j, \bar{Z}_j$  are logical Paulis for the j-th logical qubit, and  $T_g$  are pure errors that generate the pure error group  $\mathcal{T}$  and satisfy  $\{T_g, S_g\} = 0$  and  $[T_g, S_h] = 0$  for  $h \neq g$ . Moreover, there exists an n-qubit Clifford unitary (an encoding circuit) U such that

$$S_g = U Z_g U^{\dagger}, \quad T_g = U X_g U^{\dagger}, \quad g = 1, \cdots, m, \quad (1)$$

$$\bar{Z}_j = UZ_{m+j}U^{\dagger}, \ \bar{X}_j = UX_{m+j}U^{\dagger}, \quad j = 1, \cdots, k,$$
 (2)

where  $Z_g$  and  $X_g$  represent Pauli-Z and Pauli-X applied on the g-th qubit. The commutation relation between  $T_g$  and  $S_{g'}$  is given by

$$S_g \cdot T_{g'} = U Z_g U^{\dagger} U X_{g'} U^{\dagger} = (-1)^{\delta_{g,g'}} T_{g'} \cdot S_g \qquad (3)$$

The logical operators satisfy the Pauli commutation relation  $\bar{X}_j \cdot \bar{Z}_{j'} = (-1)^{\delta_{j,j'}} \bar{Z}_{j'} \cdot \bar{X}_j$ , and a code word from a stabilizer code space is a state of the form  $U(|0\rangle^{\otimes m} \otimes |\psi\rangle)$  where  $|\psi\rangle \in (\mathbb{C}^2)^{\otimes k}$  [11].

3) Surface Code: Surface code is a promising stabilizer code known for its high error threshold and compatibility with 2D lattice architectures with nearest-neighbor interactions [6]. As illustrated in Fig. 1, a single logical qubit is encoded in a  $d \times d$  array of physical (data and ancilla) qubits. A key property is that a data qubit error anti-commutes with

![](_page_1_Picture_13.jpeg)

Fig. 1. Layout of surface code (left) and syndrome behavior with corresponding matching for X and Z errors on data qubits (right).

<span id="page-1-0"></span>adjacent stabilizers of the opposite type: an X error flips two neighboring Z stabilizers, and a Z error flips X stabilizers. This results in non-trivial syndrome signals only at error chain endpoints, crucial for decoding. To counter complex errors under circuit-level noise [12], decoding employs d syndrome rounds, XOR-ing consecutive syndrome outputs to form detectors for identifying measurement errors [5], isolating measurement errors from data qubit errors. Based on this, a 3D graph  $G(\mathcal{V}, \mathcal{E})$ , where vertices are detector events and edges are potential errors, can be constructed for decoding tasks [13].

4) Degeneracy: Stabilizer codes, like Classical Error Correction (CEC), have the property that multiple error patterns can produce the same syndrome measurement [14]. However, different QEC errors may be physically indistinguishable. In QEC, any error operator E can be decomposed into three components reflecting its stabilizer, pure error, and logical parts respectively as [11]:

<span id="page-1-2"></span>
$$E = s(E) \cdot t(s) \cdot l(E) = (s(E)l(E)) \cdot t(s) \tag{4}$$

where  $s(E) \in \mathcal{S}$ ,  $l(E) \in \mathcal{L}$ ,  $s(E)l(E) \in \mathcal{N}(\mathcal{S})$ . Pure error term t(s) depends only on syndrome s:  $t(s) = \prod_{g=1}^m T_g^{\frac{1-s_g}{2}}$ , where  $s_q \in \{+1, -1\}$  is the measured eigenvalue of the g-th syndrome. Thus, the same syndrome can correspond to distinct error patterns with same (differing only in s(E)) or different (differing in both s(E) and l(E)) logical errors. Consider the first case, where errors differ only in s(E). This happens if  $E_1 = E_2 \cdot S$  with  $S \in \mathcal{S}$ . Under this condition,  $E_1 | \psi \rangle =$  $E_2S|\psi\rangle = E_2|\psi\rangle$ , and the same correction R reverses both:  $RE_2 |\psi\rangle = |\psi\rangle \Rightarrow RE_1 |\psi\rangle = RE_2 S |\psi\rangle = RE_2 |\psi\rangle = |\psi\rangle.$ Such cases define indistinguishable degenerate errors, differing in physical error pattern but sharing both syndrome and logical effect, that can be grouped into the same logicalequivalent coset defined as  $\{E|E = S_g t(s)L, \forall S_g \in \mathcal{S}\},\$ which significantly impact decoding [15] (explained in Sec. II-A5).

<span id="page-1-1"></span>5) Optimal Decoding (Coset vs. Physical): Similar to CEC, it is straightforward to find the most probable physical error from syndrome s via Maximum-Likelihood (ML) decoding:

$$E^* = \underset{E \in \mathcal{P}_n}{\operatorname{argmax}} \{ p(E|s) \}$$
 (5)

![](_page_2_Picture_0.jpeg)

Fig. 2. Two matching examples: the blue (left) belongs to coset 1 with logical error Lb, the purple (right) to coset 2 with logical error Lp. Dashed lines indicate alternative error paths.

<span id="page-2-0"></span>However, degeneracy implies that the most likely physical error may not yield the most probable logical error, since multiple equally probable errors can have the same logical effect [\[16\]](#page-13-15). For example (just for illustration), as shown in Fig. [2,](#page-2-0) with non-trivial syndromes {1, 2, 3, 4}, error patterns can be matched as {1, 4} ∪ {2, 3} or {1, 2} ∪ {3, 4}, each implying a different logical error. Both matchings contain 6 errors (edges) and therefore have the same weight, which means they are equally probable. However, the second matching includes 9 combinations with equivalent probability, so the total probability of the second logical error would be higher than that of the first. Since the two matchings belong to different logical cosets, and the second has higher probability, we choose an error pattern from the second coset—even though a pattern in the first also has maximal individual likelihood. Thus, rather than identifying the most likely physical error, optimal decoding should find the most likely logical coset [\[11\]](#page-13-10):

<span id="page-2-2"></span>
$$L^* = \underset{L}{\operatorname{argmax}} \{ p(L|s) \}$$

$$p(L|s) \propto \sum_{E:l(E)=L} p(E) = \sum_{S \in \mathcal{S}} p(E = St(s)L)$$
(6)

*6) Prior Decoding Algorithms:* The two main surface code decoders are Minimum-Weight Perfect Matching (MWPM) and Union-Find (UF). MWPM addresses argmax{p(E|s)} by solving the minimum-weight perfect matching problem on the decoding graph via the Blossom algorithm [\[8\]](#page-13-7), which formulates the problem in a linear programming (LP) framework [\[17\]](#page-13-16). While this LP-based iterative procedure yields nearoptimal decoding accuracy by maintaining *Blossoms*, it incurs substantial algorithmic and implementation complexity [\[18\]](#page-13-17), as well as relatively large decoding latency for real-time scenarios. In contrast, UF, an approximation of MWPM, is a simplified and near-linear-time algorithm [\[7\]](#page-13-6) that clusters syndromes by growing and merging them until all clusters have even parity, then resolves matches via spanning tree generation and peeling. While UF offers much lower latency due to its simplicity and high parallelism [\[19\]](#page-13-18), [\[20\]](#page-13-19), this comes at the cost of suboptimal accuracy compared to MWPM.

#### *B. Motivation*

Prior hardware decoders (i) neglect the impact of degeneracy [\[7\]](#page-13-6), [\[13\]](#page-13-12), and (ii) often rely on highly customized designs to achieve ultra-low latency, resulting in poor scalability [\[20\]](#page-13-19). Our work revisits both the decoding theory and the hardware architecture, leading to a coset-ensemble algorithm and a stallresilient pipelined architecture.

*1) Algorithmic Challenge: Coset vs. Physical:* Prior decoders implicitly frame decoding as a physical ML problem: given a syndrome, they select the most likely physical error chain. In highly degenerate stabilizer codes, many distinct chains belong to the same logical coset, and the single most likely chain does not necessarily correspond to the most probable logical error. As a result, the decoder is optimized for recovering a particular localized configuration, while systemlevel reliability is determined by the logical error rate.

Insight and Our Approach. Given a syndrome, the ideal decision selects the coset with the highest posterior probability by summing over all compatible physical errors. This coset maximum-likelihood rule directly optimizes the logical error rate but may have exponential complexity in the code size [\[21\]](#page-13-20).

Inspired by the above insight, we approximate the intractable coset ML issue with our *Ensemble Forest Exploration*. By using UF-equivalent *Clustering* to partition intractable cosets and independently injecting random priority during the forest construction, the decoder generates a set of candidate corrections that implicitly "vote" for different logical cosets. Aggregating these outcomes at the logical level yields an approximate ranking of cosets in polynomial time.

*2) Architectural Challenge: Scalability vs. Latency:* Gridmapped hardware decoders achieve extremely low latency but suffer from poor scalability and efficiency. A more scalable Von Neumann-style organization naturally decomposes the decoder into two stages: a *pipelined clustering engine* that streams one vertex per cycle, followed by *post-clustering traversal modules* (e.g., spanning-tree construction and peeling in conventional UF) that run once clustering terminates. Profiling this two-stage baseline in Fig. [3](#page-3-0) reveals that the dominant bottleneck lies inside the clustering pipeline: stalls in the clustering pipeline alone account for 48%–58% of the total decoding latency.

Insight and Our Approach. Targeting an efficient decoder, we observe that the vast majority of stalls arise from two dominant patterns of concurrent memory accesses during cluster growth and merging stages.

Guided by this observation, we co-design a conflict-aware multi-bank hashed memory system and a hierarchical ID mapping scheme to resolve these two classes of conflicts and keep the pipeline highly utilized.

# Coset Ensemble Decoder for Quantum Error Correction with Algorithm–Hardware Co-Design

Shuang Liang\*, Jubo Xu\*, Giulio Bassanino\*, Qianzhou Wang\*, Yidong Zhou<sup>†</sup>, Yuncheng Lu\*, Zhiwen Mo\*, Paul H. J. Kelly\*, Bo Yuan<sup>†</sup>, Wayne Luk\*, and Hongxiang Fan\*

\*Imperial College London, United Kingdom <sup>†</sup>Rutgers University, United States {shuang.liang, jubo.xu20, p.kelly, w.luk, hongxiang.fan}@imperial.ac.uk

Abstract—Reliable large-scale quantum computation relies on fault-tolerant architectures, where quantum error correction (QEC) continuously extracts and decodes error syndromes in real time. A critical component in QEC is the decoder, a classical subsystem that must simultaneously deliver high logical accuracy and ultra-low latency. This paper presents a novel algorithmhardware co-design that improves the accuracy-latency trade-off over existing approaches such as vanilla Minimum-Weight Perfect Matching (MWPM) and Union-Find (UF) decoders. At the algorithmic level, we introduce coset ensemble decoding, which improves UF decoding by explicitly exploiting logically equivalent cosets. Our method performs ensemble forest exploration to generate multiple coset-consistent candidates and aggregates them to approximate coset-level maximum-likelihood decoding. We further reduce computational and memory complexity via reverseorder elimination and lossless graph compression, without sacrificing accuracy. At the hardware level, we design a domainspecific architecture that temporally reuses resources, avoiding the code-distance-proportional resource growth in prior spatial architectures. Several optimizations, such as multi-bank memory hashing and hierarchical ID mapping, are proposed to mitigate pipeline stalls and memory conflicts under highly concurrent access patterns. Under a circuit-level depolarizing noise model, our co-design approach achieves a better accuracy-latency tradeoff than prior MWPM- and UF-based decoders, while reducing FPGA LUT consumption by up to 8.2 times compared with reported UF-based decoder resources. The tunable candidate number further exposes a flexible design knob, enabling users to tailor decoding performance to the requirements of different fault-tolerant workloads. Our implementation is publicly available at https://github.com/IMSeonL/coset-ensemble-decoder.

Index Terms—Quantum error correction, coset ensemble decoding, Union-Find decoder, algorithm-hardware co-design, FPGA acceleration

#### I. INTRODUCTION

Recent advances in quantum computing have led to rapid growth in the number of physical qubits. However, existing quantum devices continue to suffer from gate errors, crosstalk, leakage, and limited coherence times [1], all of which hinder the execution of practical quantum circuits at scale. Achieving practical, large-scale quantum computation therefore requires fault-tolerant quantum computing (FTQC), which relies on quantum error correction (QEC) to encode logical qubits across multiple physical qubits. QEC continuously extracts

Accepted to appear in the 53rd Annual International Symposium on Computer Architecture (ISCA 2026).

error syndromes and corrects errors during runtime [2], suppressing the logical error rate below a desired threshold as guaranteed by the threshold theorem [3], [4].

At the core of QEC is the decoder, a classical subsystem that interprets error syndromes and generates corrections in real time. Designing such decoders is challenging due to the dual demands of high decoding accuracy and ultra-low latency. As quantum processors operate at extremely high speeds, decoders are typically required to process syndromes within less than 1 us in superconducting circuits, posing significant challenges for both decoding algorithm design and hardware implementation. Among various QEC codes, the surface code has emerged as a leading candidate due to its high threshold and local stabilizer structure [5]. Within the surface code framework, two prominent decoding strategies have been extensively adopted: Minimum-Weight Perfect Matching (MWPM) [6] and Union-Find (UF) [7]. MWPM achieves high accuracy by solving an optimal matching problem over the syndrome graph, but its high computational cost and inherent sequential nature of computation limit its processing speed. In contrast, the UF decoder achieves lower latency by clustering defects in parallel yet sacrifices accuracy under complex noise conditions.

To address these challenges, this work proposes an algorithm–hardware co-design for accurate, low-latency, and flexible QEC decoding. At the algorithmic level, we introduce coset ensemble decoding that enhances traditional UF decoding by considering logical-equivalent cosets, the group of physical errors with the same syndrome and logical effect. At the hardware level, we develop a customized architecture tailored to the proposed algorithm. In contrast to prior work [8], [9], which relies on spatial architectures whose hardware resources grow with code distance, our hardware features high generality and efficiency through temporal reuse and targeted pipeline optimizations. Together, these choices improve the accuracy–resource trade-off while enabling tunable decoding accuracy for different deployment needs.

This paper makes the following main contributions. The first improves decoding accuracy, while the latter two reduce decoding cost and latency, and improve hardware efficiency.

A novel coset ensemble decoding algorithm that leverages logically equivalent cosets to approximate coset-level maximum-likelihood decoding. This is achieved by en-

semble forest exploration, which effectively improves the algorithmic performance.

- Several algorithmic optimizations, including reverse-order elimination and lossless graph compression, together with an optimality analysis of the proposed algorithm, to further reduce computational complexity.
- A domain-specific hardware architecture tailored for coset ensemble decoding. It achieves high generality and efficiency through temporal reuse and targeted pipeline optimizations. Several hardware optimizations, such as multibank memory hashing and hierarchical ID mapping, are proposed to further improve hardware efficiency and reduce latency.

#### II. BACKGROUND AND MOTIVATION

#### A. Background

- 1) Quantum Error Correction (QEC): QEC employs protocols to protect quantum information from decoherence and noise [10]. An [[n,k,d]] QEC code encodes k logical qubits into n physical qubits (n>k), overcoming fundamental constraints, like the No-Cloning Theorem and projective measurements' destructive nature, through non-local encoding and indirect syndrome extraction using ancilla qubits. The code distance d, defined as the minimum number of physical qubits supporting an undetectable logical error, quantifies error resilience. The logical error rate can be exponentially suppressed by increasing d if the physical error rate is below a threshold.
- 2) Stabilizer Code: An [[n,k]] stabilizer code has stabilizer group  $\mathcal{S} \subset \mathcal{P}_n$  generated by m=n-k independent commuting Pauli operators. One may choose a canonical generating set  $\{S_g, T_g, \bar{X}_j, \bar{Z}_j\}$  [11], where  $S_g$  are stabilizer generators,  $\bar{X}_j, \bar{Z}_j$  are logical Paulis for the j-th logical qubit, and  $T_g$  are pure errors that generate the pure error group  $\mathcal{T}$  and satisfy  $\{T_g, S_g\} = 0$  and  $[T_g, S_h] = 0$  for  $h \neq g$ . Moreover, there exists an n-qubit Clifford unitary (an encoding circuit) U such that

$$S_g = U Z_g U^{\dagger}, \quad T_g = U X_g U^{\dagger}, \quad g = 1, \cdots, m, \quad (1)$$

$$\bar{Z}_j = UZ_{m+j}U^{\dagger}, \ \bar{X}_j = UX_{m+j}U^{\dagger}, \quad j = 1, \cdots, k,$$
 (2)

where  $Z_g$  and  $X_g$  represent Pauli-Z and Pauli-X applied on the g-th qubit. The commutation relation between  $T_g$  and  $S_{g'}$  is given by

$$S_g \cdot T_{g'} = U Z_g U^{\dagger} U X_{g'} U^{\dagger} = (-1)^{\delta_{g,g'}} T_{g'} \cdot S_g \qquad (3)$$

The logical operators satisfy the Pauli commutation relation  $\bar{X}_j \cdot \bar{Z}_{j'} = (-1)^{\delta_{j,j'}} \bar{Z}_{j'} \cdot \bar{X}_j$ , and a code word from a stabilizer code space is a state of the form  $U(|0\rangle^{\otimes m} \otimes |\psi\rangle)$  where  $|\psi\rangle \in (\mathbb{C}^2)^{\otimes k}$  [11].

3) Surface Code: Surface code is a promising stabilizer code known for its high error threshold and compatibility with 2D lattice architectures with nearest-neighbor interactions [6]. As illustrated in Fig. 1, a single logical qubit is encoded in a  $d \times d$  array of physical (data and ancilla) qubits. A key property is that a data qubit error anti-commutes with

![](_page_1_Picture_13.jpeg)

Fig. 1. Layout of surface code (left) and syndrome behavior with corresponding matching for X and Z errors on data qubits (right).

<span id="page-1-0"></span>adjacent stabilizers of the opposite type: an X error flips two neighboring Z stabilizers, and a Z error flips X stabilizers. This results in non-trivial syndrome signals only at error chain endpoints, crucial for decoding. To counter complex errors under circuit-level noise [12], decoding employs d syndrome rounds, XOR-ing consecutive syndrome outputs to form detectors for identifying measurement errors [5], isolating measurement errors from data qubit errors. Based on this, a 3D graph  $G(\mathcal{V}, \mathcal{E})$ , where vertices are detector events and edges are potential errors, can be constructed for decoding tasks [13].

4) Degeneracy: Stabilizer codes, like Classical Error Correction (CEC), have the property that multiple error patterns can produce the same syndrome measurement [14]. However, different QEC errors may be physically indistinguishable. In QEC, any error operator E can be decomposed into three components reflecting its stabilizer, pure error, and logical parts respectively as [11]:

<span id="page-1-2"></span>
$$E = s(E) \cdot t(s) \cdot l(E) = (s(E)l(E)) \cdot t(s) \tag{4}$$

where  $s(E) \in \mathcal{S}$ ,  $l(E) \in \mathcal{L}$ ,  $s(E)l(E) \in \mathcal{N}(\mathcal{S})$ . Pure error term t(s) depends only on syndrome s:  $t(s) = \prod_{g=1}^m T_g^{\frac{1-s_g}{2}}$ , where  $s_q \in \{+1, -1\}$  is the measured eigenvalue of the g-th syndrome. Thus, the same syndrome can correspond to distinct error patterns with same (differing only in s(E)) or different (differing in both s(E) and l(E)) logical errors. Consider the first case, where errors differ only in s(E). This happens if  $E_1 = E_2 \cdot S$  with  $S \in \mathcal{S}$ . Under this condition,  $E_1 | \psi \rangle =$  $E_2S|\psi\rangle = E_2|\psi\rangle$ , and the same correction R reverses both:  $RE_2 |\psi\rangle = |\psi\rangle \Rightarrow RE_1 |\psi\rangle = RE_2 S |\psi\rangle = RE_2 |\psi\rangle = |\psi\rangle.$ Such cases define indistinguishable degenerate errors, differing in physical error pattern but sharing both syndrome and logical effect, that can be grouped into the same logicalequivalent coset defined as  $\{E|E = S_g t(s)L, \forall S_g \in \mathcal{S}\},\$ which significantly impact decoding [15] (explained in Sec. II-A5).

<span id="page-1-1"></span>5) Optimal Decoding (Coset vs. Physical): Similar to CEC, it is straightforward to find the most probable physical error from syndrome s via Maximum-Likelihood (ML) decoding:

$$E^* = \underset{E \in \mathcal{P}_n}{\operatorname{argmax}} \{ p(E|s) \}$$
 (5)

![](_page_2_Picture_0.jpeg)

Fig. 2. Two matching examples: the blue (left) belongs to coset 1 with logical error Lb, the purple (right) to coset 2 with logical error Lp. Dashed lines indicate alternative error paths.

<span id="page-2-0"></span>However, degeneracy implies that the most likely physical error may not yield the most probable logical error, since multiple equally probable errors can have the same logical effect [\[16\]](#page-13-15). For example (just for illustration), as shown in Fig. [2,](#page-2-0) with non-trivial syndromes {1, 2, 3, 4}, error patterns can be matched as {1, 4} ∪ {2, 3} or {1, 2} ∪ {3, 4}, each implying a different logical error. Both matchings contain 6 errors (edges) and therefore have the same weight, which means they are equally probable. However, the second matching includes 9 combinations with equivalent probability, so the total probability of the second logical error would be higher than that of the first. Since the two matchings belong to different logical cosets, and the second has higher probability, we choose an error pattern from the second coset—even though a pattern in the first also has maximal individual likelihood. Thus, rather than identifying the most likely physical error, optimal decoding should find the most likely logical coset [\[11\]](#page-13-10):

<span id="page-2-2"></span>
$$L^* = \underset{L}{\operatorname{argmax}} \{ p(L|s) \}$$

$$p(L|s) \propto \sum_{E:l(E)=L} p(E) = \sum_{S \in \mathcal{S}} p(E = St(s)L)$$
(6)

*6) Prior Decoding Algorithms:* The two main surface code decoders are Minimum-Weight Perfect Matching (MWPM) and Union-Find (UF). MWPM addresses argmax{p(E|s)} by solving the minimum-weight perfect matching problem on the decoding graph via the Blossom algorithm [\[8\]](#page-13-7), which formulates the problem in a linear programming (LP) framework [\[17\]](#page-13-16). While this LP-based iterative procedure yields nearoptimal decoding accuracy by maintaining *Blossoms*, it incurs substantial algorithmic and implementation complexity [\[18\]](#page-13-17), as well as relatively large decoding latency for real-time scenarios. In contrast, UF, an approximation of MWPM, is a simplified and near-linear-time algorithm [\[7\]](#page-13-6) that clusters syndromes by growing and merging them until all clusters have even parity, then resolves matches via spanning tree generation and peeling. While UF offers much lower latency due to its simplicity and high parallelism [\[19\]](#page-13-18), [\[20\]](#page-13-19), this comes at the cost of suboptimal accuracy compared to MWPM.

#### *B. Motivation*

Prior hardware decoders (i) neglect the impact of degeneracy [\[7\]](#page-13-6), [\[13\]](#page-13-12), and (ii) often rely on highly customized designs to achieve ultra-low latency, resulting in poor scalability [\[20\]](#page-13-19). Our work revisits both the decoding theory and the hardware architecture, leading to a coset-ensemble algorithm and a stallresilient pipelined architecture.

*1) Algorithmic Challenge: Coset vs. Physical:* Prior decoders implicitly frame decoding as a physical ML problem: given a syndrome, they select the most likely physical error chain. In highly degenerate stabilizer codes, many distinct chains belong to the same logical coset, and the single most likely chain does not necessarily correspond to the most probable logical error. As a result, the decoder is optimized for recovering a particular localized configuration, while systemlevel reliability is determined by the logical error rate.

Insight and Our Approach. Given a syndrome, the ideal decision selects the coset with the highest posterior probability by summing over all compatible physical errors. This coset maximum-likelihood rule directly optimizes the logical error rate but may have exponential complexity in the code size [\[21\]](#page-13-20).

Inspired by the above insight, we approximate the intractable coset ML issue with our *Ensemble Forest Exploration*. By using UF-equivalent *Clustering* to partition intractable cosets and independently injecting random priority during the forest construction, the decoder generates a set of candidate corrections that implicitly "vote" for different logical cosets. Aggregating these outcomes at the logical level yields an approximate ranking of cosets in polynomial time.

*2) Architectural Challenge: Scalability vs. Latency:* Gridmapped hardware decoders achieve extremely low latency but suffer from poor scalability and efficiency. A more scalable Von Neumann-style organization naturally decomposes the decoder into two stages: a *pipelined clustering engine* that streams one vertex per cycle, followed by *post-clustering traversal modules* (e.g., spanning-tree construction and peeling in conventional UF) that run once clustering terminates. Profiling this two-stage baseline in Fig. [3](#page-3-0) reveals that the dominant bottleneck lies inside the clustering pipeline: stalls in the clustering pipeline alone account for 48%–58% of the total decoding latency.

Insight and Our Approach. Targeting an efficient decoder, we observe that the vast majority of stalls arise from two dominant patterns of concurrent memory accesses during cluster growth and merging stages.

Guided by this observation, we co-design a conflict-aware multi-bank hashed memory system and a hierarchical ID mapping scheme to resolve these two classes of conflicts and keep the pipeline highly utilized.

