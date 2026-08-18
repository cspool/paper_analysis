# Triage: An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

1<sup>st</sup> Jiahan Chen\*

The Hong Kong University of
Science and Technology (Guangzhou)
Guangzhou, China

2<sup>nd</sup> Chenghong Zhu\*

The Hong Kong University of Science
and Technology (Guangzhou)

Guangzhou, China

3rd Ge Bai<sup>†</sup>
The Hong Kong University of
Science and Technology (Guangzhou)
Guangzhou, China
gebai@hkust-gz.edu.cn

4th Xin Wang<sup>†</sup>
The Hong Kong University of Science
and Technology (Guangzhou)
Guangzhou, China
felixxinwang@hkust-gz.edu.cn

Abstract—Fault-tolerant quantum computation (FTQC) critically depends on real-time classical decoding, which is rapidly emerging as a system bottleneck. As quantum systems scale, decoding latency and throughput limitations lead to exponential syndrome backlogs and logical operation stalls. While hardware accelerators and parallel windowing offer pathways to speed up decoding, dynamically deploying a finite pool of decoders across a vast quantum error correction architecture remains an unresolved resource allocation problem.

To address this, we formulate FTQC decoding as a constrained dynamic scheduling problem by utilizing a spatio-temporal framework based on *slices*. We propose Triage, a dual-mode architecture that mitigates operation stalls by adaptively combining a cost-efficient heuristic scheduler with a priority-aware emergency mode to rapidly resolve the causal cone of critical operations. Our evaluation shows that Triage maintains low algorithm stalls and logical error rates even under scarce classical resource constraints. Across various benchmarks, Triage achieves an average logical error rate reduction of 52.6% compared to standard temporal parallelism, enabling an efficient classical control plane for scalable FTQC architectures.

Index Terms—Fault-tolerant quantum computing, quantum error correction, real-time decoding, parallel window decoding.

## I. Introduction

Quantum computers hold the potential to efficiently solve certain problems that are intractable for the best known classical algorithms [1]–[3]. However, current quantum hardware is highly error-prone [4], [5], requiring quantum error

This work has been partially supported by the National Key R&D Program of China (Grant No. 2024YFB4504004), the National Natural Science Foundation of China (Grant No. 12447107), the Guangdong Provincial Quantum Science Strategic Initiative (Grant Nos. GDZX2403008 and GDZX2503001), and the Guangdong Provincial Key Lab of Integrated Communication, Sensing and Computation for Ubiquitous Internet of Things (Grant No. 2023B1212010007).

![](_page_0_Figure_14.jpeg)

(a) Traditional vs Triage Decoding (b) Decoding Resource Space

Fig. 1. Navigating the FTQC Decoding Bottleneck. (a) Traditional decoding leads to large idle stalls, while Triage employs spatio-temporal windows and prioritizes the causal cone to effectively reduce the latency. (b) Triage achieves better performance in the near-term, resource-constrained landscape.

correction (QEC) to enable fault-tolerant quantum computation (FTQC) [6]. Encouragingly, recent experimental progress across various platforms and QEC codes [7]–[9] is rapidly validating this approach [10]–[14]. As these advances move FTQC from theory toward viability, the focus shifts from theoretical feasibility to the architectural challenges of implementation.

Towards a large scale FTQC, a significant architectural bottleneck arises from the classic control layer. At the heart of this layer is the *decoder*, whose function is to continuously process a massive stream of classical syndrome data from the quantum processor, and infer the most likely errors. Crucially, decoding must operate in *real-time*. This means the overall throughput of decoding must, on average, exceed the rate of syndrome generation. Otherwise, the system will accumulate an exponential backlog [15] of unprocessed syndromes, which will eventually overwhelm the computational resources.

To address this challenge, one line of research focuses on optimizing the *latency* of a single decoding operation. On

<sup>\*</sup>Co-first authors

<sup>†</sup>Co-corresponding authors

the software front, significant effort has gone into developing high-accuracy decoding algorithms with low computational complexity [16]–[19]. In parallel, hardware accelerators using FPGAs have demonstrated single-decode latencies below the demands of superconducting qubits for certain code distances [20]–[23]. However, these hardware demonstrations have been largely confined to memory experiments rather than integrated into logical computations. Meanwhile, the protocol that processes the syndrome stream in a *serial* fashion has been shown to be unscalable [24]. Therefore, beyond optimizations for latency, designing superior decoding protocols that enhance overall *throughput* is essential.

A second direction aims to improve decoding throughput via *parallelism*. Temporal parallelism, for instance, partitions the syndrome stream into time blocks, allowing concurrent processing of the non-adjacent blocks [24]–[26]. For multi-qubit logical operations such as lattice surgery, spatial parallelism can also be employed by partitioning the syndromes from the involved regions [27]. In principle, a spatio-temporal parallel approach would allow the system to scale its total throughput simply by adding more decoder units. Nevertheless, while temporal or spatial windowing techniques are known, a systematic scheduling framework integrating resource-aware temporal and spatial parallelism has not yet been demonstrated.

How to deploy a finite pool of decoders onto a FTQC application? First, there is an *asymmetry* between classical resources and logical qubits. Depending on hardware limitations or error targets, an application may require a few high-speed decoders for small codes, or a large pool of decoders collaborating in parallel for large codes. In a realistic large-scale architecture, this necessitates an M-for-N shared resource model where M < N decoders must be dynamically allocated. At any moment, determining *which* decoder to assign to *which* logical patch is a resource allocation problem [28].

This scheduling problem is further complicated by the operational logic of FTQC. The *Pauli frame* [29] stores the decoder's inference of accumulated errors. When the computation encounters a non-Clifford gate, the decoder must update all relevant Pauli frames, a process we term *synchronization*. A synchronization failure forces the logical operation to stall. During this idle period, qubits undergo additional error correction rounds, directly increasing the logical error rate (LER). To maximize fidelity via idle-reduction, decoding tasks relevant to the critical non-Clifford operation must be given higher priority. Combining this urgency with the spatio-temporal dependency constraints from parallel decoding, the problem is transformed into a dynamic constrained scheduling problem. As illustrated in Figure 1, traditional decoding approaches fail to navigate these dependencies efficiently, leading to severe resource waste and long idle stalls. Furthermore, as the hardware design space spans diverse decoder speeds and counts, a robust scheduling strategy becomes critical, especially in nearterm, resource-constrained environments where naive policies quickly fall into an unrecoverable regime of failure.

In this paper, we systematically address this challenge. First, we utilize a parallel spatio-temporal decoding framework using *slice* (a d × d patch over d rounds) as the basic scheduling unit. By modeling the lifecycle of each slice and identifying the *causal cone* of critical operations, we formulate the FTQC decoder scheduling problem. Second, to optimize performance under constrained resources, we propose Triage, a dual-mode scheduling architecture which combines a fast heuristic-based steady mode with a robust look-ahead emergency mode. Triage significantly reduces the logical operation stalls, leading to an average LER reduction of 52.6% compared to the standard temporal-parallel scheduling strategy.

In summary, we make the following contributions:

- We introduce an abstraction of the decoder scheduling problem based on a constraint graph of *slices*. This framework is hardware-agnostic and applicable to diverse quantum platforms utilizing surface codes.
- We propose the Triage scheduler, a dual-mode system that minimizes logical operation stalls by dynamically invoking an emergency mode to rapidly resolve the causal cone of prerequisite decodes.
- We demonstrate that by effectively scheduling parallel windows, it is possible to overcome the latency limitations of individual decoders, enabling FTQC even in the challenging slow-decoder regime (τdecode > τsyndrome).
- We quantify the impact of real-time scheduling on system-level fidelity, presenting a simulation framework that captures the interaction between syndrome generation and decoding.

