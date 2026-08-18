# VIII. RELATED WORK

**High-Level Language (HLL) and ISA MCMs:** There is a long line of work that models various ISA and HLL MCMs [2, 3, 5, 9, 34, 37, 42, 49, 52, 53]. Prior work on such models highlighted prefix-closure as a useful property [25, 47]. There have been a few papers on MCMs and coherence for novel hardware [6, 58, 68], but more research in this area is needed.

Hardware Models and Proofs: There has also been much work on formally modeling hardware designs using a variety of representations [10, 11, 33, 35, 57, 64, 68]. Work on formal

verification of hardware implementations includes bounded proofs [33, 35, 39, 40, 69] and complete (all-program) [10, 11, 30, 38, 62, 64] proofs. Some prior hardware verification work uses an intermediate state machine to decompose the proof [5, 64]. However, none of these prior decomposed proofs are completely machine-checked like ours. Refinement has been used to verify hardware [10, 11, 30] as well as distributed and operating systems [15, 17, 18, 23, 26, 32, 66]. Pensieve [67] uses uninterpreted functions to overapproximate microarchitectural security behavior, similar to how we overapproximate cache behavior using environmental transitions (§VI-D). Our environmental transitions are inspired by Wickerson et al. [64].

**Programmable Memory Hierarchies:** Programmable memory hierarchies give software greater control over data movement through the memory hierarchy [27, 43, 44, 54, 55], and can improve performance and energy efficiency.

