# 1 Introduction

D ata movement is the dominant bottleneck in parallel systems. As global communication gets increasingly expensive, architectures must keep most communication local to scale. This imperative motivates non-uniform memory access (NUMA) architectures, where data is distributed throughout the design so that at least some data is local to every processor.

Unfortunately, NUMA requires clever data placement to yield benefits, and good data placement is difficult or even impossible. NUMA architectures must identify data that are frequently accessed by particular threads and then move that data near those threads. Placing data in software is challenging, often requiring language support [\[19\]](#page-11-0). Placing data in hardware (e.g., via private caches [\[35\]](#page-12-0) or non-uniform cache access [\[40\]](#page-12-1) architectures) adds significant complexity. Moreover, NUMA does not benefit widely shared data, for which no single placement suffices, or large data sets, which do not fit in a single NUMA domain. Thus, although NUMA is the dominant approach to scale communication in modern systems, it is often ineffective at keeping communication local.

This paper explores scaling in the context of spatial dataflow architectures (SDAs), exploiting a new opportunity introduced by these architectures to scale memory access while avoiding the limitations of NUMA [\(Fig. 1\)](#page-0-0). SDAs are an emerging class of "generalpurpose accelerator" that comprise a fabric of simple processing elements (PEs) interconnected by a network on-chip (NoC). An SDA represents a program as a dataflow graph of instructions with edges for data dependencies. The compiler performs place-androute (PnR), placing instructions onto PEs and routing communication over the NoC.

SDAs are highly sensitive to the latency and energy of communication between PEs. As SDAs scale, the cost of communication across the fabric becomes highly heterogeneous: communication between adjacent PEs is easily 10× faster than across the full fabric. The compiler keeps communication local by placing frequently communicating instructions on nearby PEs.

Insight. Our key observation is that, for the same fundamental reasons that motivate NUMA and heterogeneous PE-to-PE latency, some processing elements are necessarily closer to memory than others. Although it is impossible to build an architecture where memory is close to every PE, it is possible to build one where memory is close to some PEs [\(Fig. 1\)](#page-0-0).

And unlike conventional multicores, it is natural in an SDA to place particular instructions at an advantageous location. In a multicore, each processor is time-multiplexed by many instructions, dynamically scheduled by the microarchitecture. There is thus little opportunity in a multicore to identify critical instructions and place them advantageously.

By contrast, SDA compilers already place instructions to optimize communication. It is a short leap to place critical loads — that is, loads on a program's critical path where any additional delay will slow down execution (e.g., on the recurrence of a loop) — closer to memory. Moreover, in contrast to the difficulty of identifying and partitioning data in NUMA systems, it is often easy for compilers to identify critical loads.

Our solution: Non-uniform processing-element access (NU-PEA). SDAs should architecturally expose the non-uniform latency to memory from different PEs. Doing so allows the compiler to place critical loads closer to memory, optimizing latency where it matters, while maintaining high memory bandwidth for latency-insensitive loads on other PEs.

We evaluate NUPEA on Monaco, a general-purpose SDA that features a 12×12 heterogeneous fabric of PEs [\(Fig. 1\)](#page-0-0). PEs are interconnected by statically routed, bufferless data NoC, and a subset of "load-store" (LS) PEs are connected to memory over a hierarchical, dynamically routed fabric-memory NoC. Memory is banked 32×, with a shared cache in front. The fabric-memory NoC exposes nonuniform latency, depending on which LS PE is accessing memory.

The Efficient C Compiler, or effcc, maps programs written in C onto Monaco. effcc splits programs into regions that fit on Monaco's fabric and automatically parallelizes loop nests to maximize performance. This paper focuses on effcc's optimization passes to

(i) identify critical loads, specifically those along loop recurrences, and (ii) preferentially place critical loads in fast NUPEA domains during place-and-route.

Monaco and effcc are both industry products. Monaco is implemented in a 22nm planar process and is based on a design taped out in Q1'25. effcc is implemented in MLIR [\[44\]](#page-12-2) and is general-purpose, supporting the full C language.

Contributions. This paper contributes the following:

- We introduce NUPEA, a new opportunity to scale communication in spatial dataflow architectures (SDAs) by placing instructions, not data.
- We present the design and implementation of NUPEA in the Monaco SDA.
- We present the design and implementation of NUPEA-aware optimization passes in the effcc compiler that identify critical loads and move them closer to memory during place-androute (PnR).
- We perform a design space exploration of NUPEA in SDAs to optimize the placement of load-store PEs within Monaco's dataflow fabric.

Results. This paper evaluates NUPEA in simulation on Monaco using effcc and our internal microarchitectural simulator. NUPEA outperforms practical alternatives. Monaco performs 28% better than an SDA with uniform, two-cycle access; 20% better than an SDA with distributed NUMA memory access; and within 21% of an idealized SDA with uniform, single-cycle access.

Road map. [Sec. 2](#page-1-0) discusses existing approaches to scale data movement and motivates NUPEA. [Sec. 3](#page-4-0) introduces NUPEA and illustrates it on sparse matrix-vector product. [Sec. 4](#page-5-0) describes Monaco, a representative NUPEA microarchitecture, and [Sec. 5](#page-6-0) describes the effcc NUPEA-aware compiler. [Sec. 6](#page-7-0) presents our evaluation methodology, and [Sec. 7](#page-8-0) evaluates NUPEA. [Sec. 8](#page-11-1) concludes.

