# <span id="page-1-2"></span>2.1 Data-centric scaling of data movement

<span id="page-1-1"></span>Non-uniform memory access (NUMA). NUMA architectures segment cores and main memory into domains (blue groups in [Fig. 2\)](#page-1-1).

![](_page_1_Picture_24.jpeg)

Figure 2: A classic NUMA system.

An access to a local domain (green arrow) is faster than to a remote domain (red arrow). The goal in a NUMA system is to place data in domains such that most accesses are local.

[Fig. 3](#page-2-0) shows sparse matrix-vector multiplication (spmspv, our running example in this paper). spmspv is application that parallelizes well onto a NUMA architecture: each thread performs an independent sparse inner product. spmspv's bottleneck is an intersection operation (∩, L.6) that finds matching locations of non-zero values in Ar and V, which are then multiplied and accumulated in a dot product (L.8-9).

```
1 parfor r = 0..A.numRows:
2 # Get non-zero (nz) indices of Ar
3 beg = A.rows[r], end = A.rows[r+1]
4 nzIdxA[] = A.nzIdx[beg..end]
5 # Ar ∩ V -> common nz indices
6 nzInBoth[] = nzIdxA ∩ V.nzIdx
7 # Dot-product, write to vector
8 D[r] = sum([A.val[i] * V.val[i] 
9 for i in nzInBoth])
```

Figure 3: Inner-product spmspv. A is in compressed sparse row format (CSR) and V is sparse. D is dense. ∩ (L.6) is its critical path.

For NUMA to be effective, threads 0–3 on domain D0 in [Fig. 2](#page-1-1) should have local access to A's CSR data structure, especially the data for nzIdxA (L.4-6). However, thread 2 incurs a remote access that slows down the intersection operation, hurting performance.

A NUMA system speeds up these remote accesses using information from a dynamic analysis and operating system (OS) support. An OS can migrate pages between NUMA domains [\[15\]](#page-11-2) or schedule a data-intensive thread close to a memory controller to reduce network traffic [\[24\]](#page-12-3). However, many systems simply interleave data across domains [\[27\]](#page-12-4). Interleaving utilizes all available memory capacity, but places most data remote, sacrificing most of the scaling potential of NUMA.

Non-uniform cache access (NUCA). NUCA [\[40\]](#page-12-1) systems distribute cache banks across the on-chip network, so that each core is closer to some cache banks than others. Like NUMA, the goal in NUCA is to place data so that most accesses are to local cache banks. Compared to NUMA, data migration is more practical in NUCA due to smaller data sizes and higher natural churn in caches.

Static NUCA [\[40\]](#page-12-1) simply interleaves addresses across cache accesses; like interleaved NUMA, Static NUCA incurs global communication for nearly all accesses and scales poorly. By contrast, Dynamic NUCA techniques respond to cache misses by migrating data towards the requesting core at the granularity of cache lines or pages [\[10,](#page-11-3) [12,](#page-11-4) [13,](#page-11-5) [18,](#page-11-6) [20,](#page-11-7) [83\]](#page-13-1). Dynamic NUCA techniques can be effective, but add significant complexity [\[83\]](#page-13-1) and/or rely on fragile heuristics [\[33,](#page-12-5) [55\]](#page-12-6). Due to these drawbacks, commercial multicores use Static NUCA despite its poor scalability.

Processing in-memory (PIM). Processing in- or near-memory architectures offload instructions to execute at or in the memory [\[1,](#page-11-8) [2,](#page-11-9) [5,](#page-11-10) [8,](#page-11-11) [26,](#page-12-7) [32,](#page-12-8) [34,](#page-12-9) [37,](#page-12-10) [41,](#page-12-11) [47,](#page-12-12) [62,](#page-13-2) [65,](#page-13-3) [69,](#page-13-4) [72,](#page-13-5) [73,](#page-13-6) [81,](#page-13-7) [87,](#page-13-8) [90,](#page-13-9) [91\]](#page-13-10). PIM systems generally target applications with poor data reuse, where caches are ineffective [\[47\]](#page-12-12). On these applications, PIM reduces the cost for instructions to fetch data, but increases the cost of communication between instructions. Thus, for PIM to be effective, most data accessed by a thread (or other unit of offloading) must be co-located at the same memory bank or vault, just like NUMA [\[88\]](#page-13-11).

NUPEA is similar to PIM in that both architectures reduce data movement by placing instructions closer to memory, however NU-PEA does this without increasing the communication cost between instructions. This opportunity is unique to SDAs.

Scratchpad memories. Some systems provide distributed scratchpads that enable extremely fast and cheap access to local data. Scratchpads are essentially an extreme version of NUMA: they shift the burden of data placement from the memory system to the user [\[43\]](#page-12-13) or compiler [\[54\]](#page-12-14), where statically reasoning about data access semantics is difficult. Moreover, since the scratchpad is not in the shared address space, data must be statically guaranteed to only be used by a single thread. For example, spmspv's accesses are difficult for a compiler to reason about because of pointer indirection through A and V. These drawbacks limit scratchpads to programs with regular accesses or language-level support [\[46,](#page-12-15) [66\]](#page-13-12), and have been largely unsuccessful on general-purpose processors.

Smart data placement is hard. The above techniques all rely, on some level, on smart data placement in the compiler to improve scalability. Data placement is extremely challenging for compilers on arbitrary code due to the difficulty of statically reasoning about memory aliasing. Moreover, data placement is generally ineffective when data are widely shared or too large to fit locally. Most generalpurpose systems instead prefer hardware-managed caches [\[55\]](#page-12-6).

Private caches offer the ability to place data near relevant compute based on the dynamic access pattern; this adaptability makes them extremely effective, and thus they are ubiquitous on multicores. However, private caches lower available capacity, risking a much higher miss rate than a shared cache. Moreover, private caches require a coherence protocol to arbitrate access to shared data, adding significant latency, network traffic, and design and verification complexity.

Private caches also require an abstraction for assigning work to each cache to effectively use the smaller capacity; this is typically exposed through a threading model and would be the responsibility of the programmer or an advanced compiler. On spmspv, private caches can achieve a low miss rate by splitting the rows of A across caches, provided that V is small enough to fit within each private cache. If work from the same row of A is spread to different caches, the same cache line may be loaded multiple times, increasing the miss rate. Worse, if D[r] is simultaneously written by multiple threads, coherence requires more global communication than a simple shared memory system.

Summary. Existing solutions to scale data movement are datacentric: all attempt to cleverly place data near threads that access it. To succeed, the data-centric approach requires data to be partitioned across threads, which systems have struggled to do automatically. Data-centric approaches are ineffective where data is widely shared or an individual thread's working set is large. NUPEA is a complementary way to scale data movement that exploits unique properties of SDAs by taking an instruction-centric approach.

