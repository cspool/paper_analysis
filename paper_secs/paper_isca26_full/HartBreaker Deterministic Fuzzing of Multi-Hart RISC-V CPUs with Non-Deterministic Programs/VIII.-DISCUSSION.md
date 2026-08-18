# VIII. DISCUSSION

We now discuss the scalability of data-flow verification and the challenges involved in reducing test cases.

Verification scalability. HARTBREAKER scales well due to the determinism anchors, enabling large program lengths. Yet, the number of memory operations that may be executed between two synchronization anchors remains bounded by the capacity of the memory consistency solver. Initially, we implemented our own custom memory consistency solver. This solver was inefficient for our verification scenario, and could only handle around ten memory operations before having to use a synchronization anchor due to extremely long solving times. We initially thought that the number of memory operations between two solving steps was a limiting factor of our implementation, and modified our approach to translate execution traces to litmus tests so we could benefit from the capacities of the state-of-the-art memory consistency solvers. When using Dartagnan [\[37\]](#page-14-11), we could scale HARTBREAKER to about 100 memory operations between synchronization anchors. However, we did not find any new bugs, and could only reproduce the ones found with our custom implementation. This suggests that further increasing the number of memory operations between synchronization anchors is unlikely to trigger more bugs since 100 memory operations already go beyond the size of most on-core data structures.

Program reduction. The non-deterministic nature of our test cases presents a challenge for debugging and analysis. When a test case triggers a bug, it cannot be reduced, as any modifications to the instruction stream might fail to reproduce the bug, even if the test case is far from minimal. Still, even without reduction, the memory consistency solvers used as backends offer visualization capabilities that we can use to pinpoint the incorrect instruction more easily.

Scaling to more harts. Our methodology uses two harts, as described in the examples. We added a third hart that generates independent memory operations to introduce additional noise into the system. We evaluated both configurations: the third hart yielded modest coverage improvement but did not uncover additional bugs. Scaling program generation beyond two harts would require engineering effort to extend the barrier's lock mechanism.

Size of re-ordering windows. The upper bound on the number of instructions between synchronization anchors is set by the capabilities of the backend we use. The bugs are not very sensitive to the window size; we discovered them with windows ranging from 10 to 100 memory operations. Larger windows may become relevant in future CPUs that may adopt wider store buffers and other memory-related optimizations.

## IX. RELATED WORK

Hardware fuzzing and memory consistency solvers are popular areas of research. We first discuss related hardware fuzzing research before discussing formal approaches to verifying memory consistency.

Hardware fuzzers. To the best of our knowledge, HART-BREAKER is the first general hardware fuzzer that is capable of generating multi-hart assembly programs and verifying their correctness. Some hardware fuzzers, such as INSTILLER [\[50\]](#page-15-0), DifuzzRTL [\[26\]](#page-14-7) and ProcessorFuzz [\[17\]](#page-14-6) support interrupt fuzzing, but unlike HARTBREAKER, they do not support IPIs or multi-hart capabilities. The most relevant fuzzer for multi-hart testing is RISCV-DV [\[7\]](#page-14-14), which cannot verify the correctness of test program execution. AXE [\[36\]](#page-14-32) is a tool for testing and validating the memory subsystem of sharedmemory multiprocessors. It records memory request/response traces, then checks those traces against a consistency model. Still, it only supports an older version of the RVWMO model, and requires a custom RTL-level generator such as Rocket's GroundTest.

Formal approaches. Formal approaches are not general, and mostly focus on memory consistency models, and struggle when scaling to larger CPU designs. There has been research on the interaction of interrupts with memory consistency models, but such work aims at defining a better model rather than testing hardware's correctness [\[38\]](#page-14-33). Other formal approaches formalize the microarchitecture of CPUs, and verify this model against different memory models [\[33\]](#page-14-0), [\[35\]](#page-14-1), [\[42\]](#page-14-2). Yet, unlike HARTBREAKER, the microarchitecture of these CPUs must be manually translated. Automatic translation of RTL into microarchitectural specifications is possible, but scaling to complex CPUs with speculation or caches remains a challenge [\[24\]](#page-14-3). Recent efforts to bridge this gap [\[25\]](#page-14-34) cannot yet verify memory consistency models. Other formal tools aim to synthesize litmus test suites from axiomatic memory model formalizations [\[34\]](#page-14-35), but they suffer from the same limitation as standard litmus tests: they are not complex enough to exert meaningful pressure on bare-metal hardware.

Software concurrency fuzzing. Previous work applies fuzzing techniques to detect concurrency bugs in multithreaded software, using strategies such as thread-aware input prioritization, automatic control of thread interleavings, and contextsensitive race detection [\[19\]](#page-14-36), [\[22\]](#page-14-37), [\[27\]](#page-14-38)–[\[30\]](#page-14-39), [\[45\]](#page-14-40). While effective at finding concurrency bugs at the software level, these approaches operate on compiled binaries or source code and rely on software-level oracles such as crash detection, sanitizers, or assertion violations. Even if such a tool happened to trigger a hardware memory ordering bug, it would lack the oracle to detect it: the observed outcomes are not checked against a formal memory model specification, and the microarchitectural state is not visible to the fuzzer. In contrast, our work generates bare-metal litmus tests and checks their outcomes against the RISC-V memory model, enabling the detection of hardware-level ordering violations that are invisible to software-level tools.

#### X. CONCLUSION

We presented HARTBREAKER, the first RISC-V fuzzer capable of systematically testing communication channels such as shared memory or inter-processor interrupts on multihart CPUs. To make this possible, HARTBREAKER needs to address the fundamental challenge of validating correct behavior in the presence of inherent non-determinism in multi-hart execution. HARTBREAKER achieves this through a mechanism that we call *determinism anchor*. Determinism anchors enable HARTBREAKER to generate test programs that exhibit arbitrary yet bounded non-deterministic behavior, enabling efficient program execution and scalable validation of correct CPU behavior. Our evaluation on five well-tested RISC-V designs discovered five previously unknown concurrency bugs, demonstrating that critical multi-hart interactions remain under-tested in practice.

```
5 # Write(x)
6 sw x2, 0(a1)
10 ...
                        1 la x1, ADDR_A
                        2 sd a0, 0(a1) # Write(x)
                        3 bltu a0, a4, ...
                        4 lb s2, 0(a1) # Load(x) id=0
                        5 xor s2, s2, s2
                        6 lbu s5, 0(a1) # Load(x) id=1
                        7 mulhu s5, s5, zero
                        8 bne s2, s2, ...
                        9 sd s0, 0(a7)
                       10 lwu sp, 0(a1) # Load(x) id=2
```

(a) Hart 0 instructions.

(b) Hart 1 instructions.

Fig. 18: Complete N1 test case excerpt.

```
13 # Write(x)
14 sd s1, 0(a1)
25 ...
                         1 lw s7,0(sp) # Load(x) id=0
                         2 sraiw zero,t1,0x0
                         3 remw s7,s7,s7
                         4 and s2,s2,t5
                         5 and s7,s7,t5
                         6 xor s2,s2,s7
                         7 lui tp,0x1cc5a
                         8 lwu t1,0(s2) # Load(x) id=1
                         9 addi tp,tp,-764
                        10 and tp,tp,t5
                        11 and a7,a7,t5
                        12 xor tp,tp,a7
                        13 lui a7,0x80005
                        14 mul t1,t1,zero
                        15 addi a7,a7,1496
                        16 and a7,a7,t5
                        17 and s7,s7,t5
                        18 xor a7,a7,s7
                        19 lui a5,0xd3f4d
                        20 addi a5,a5,-368
                        21 and a5,a5,t5
                        22 and t1,t1,t5
                        23 xor a5,a5,t1
                        24 lui t1,0x80005
                        25 lwu s0,0(tp) # Load(x) id=2
```

(a) Hart 0 instructions.

(b) Hart 1 instructions.

Fig. 19: Complete B1 test case excerpt.

#### ACKNOWLEDGMENTS

The authors would like to thank all reviewers for their valuable feedback and guidance during the review process. This work was supported in part by the Swiss State Secretariat for Education, Research and Innovation under contract number MB22.00057 (ERC-StG PROMISE).

## APPENDIX A TEST CASES

Figure [18](#page-12-1) and Figure [19](#page-12-0) show the entire assembly snippets relevant to the bugs described in Section [VII-E.](#page-10-0) The instructions prior to the snippets play an important role in priming the microarchitectural structures before these critical sections execute. The sections only show the memory operations relevant to the bugs, and their interleaving instructions, which play a crucial role in triggering the correct timings.

## APPENDIX B ARTIFACT APPENDIX

## *A. Abstract*

This artifact contains the HartBreaker fuzzing framework for RISC-V processor verification, together with pre-built simulator binaries, Docker containers, and scripts to reproduce all figures (Figures 8-14) from the paper. The artifact runs inside Docker containers to ensure a consistent environment. Reviewers can reproduce each figure independently using the provided scripts, which handle data collection, processing, and PDF figure generation. A Zenodo archive provides a persistent, citable snapshot of the full artifact.

#### *B. How to Access*

The artifact is archived on Zenodo at [https://doi.org/](https://doi.org/10.5281/zenodo.19417381) [10.5281/zenodo.19417381.](https://doi.org/10.5281/zenodo.19417381) We also make the source code available on GitHub at [https://github.com/comsec-group/](https://github.com/comsec-group/hartbreaker) [hartbreaker.](https://github.com/comsec-group/hartbreaker) We recommend using the GitHub version, as it will provide potential bugfixes, if any are found.

## *C. Hardware Dependencies*

To just run the fuzzer, a user will need a processor with at least 8 cores (more cores speed up parallel benchmark runs), at least 16 GB of RAM and 50 GB of free disk space. Note that the requirements may change depending on the requirements of the design under test. For full reproduction of the figures on the paper, a processor with 256 cores, 64GiB of RAM and 1TB of free disk space is recommended.

#### *D. Software Dependencies*

All dependencies are encapsulated in the provided Docker images. Locally, the following tools will be needed:

- Linux operating system (tested on Ubuntu 22.04).
- Docker (version ≥20.10).
- ModelSim to re-generate the Riscv-DV test corpus. (optional)

