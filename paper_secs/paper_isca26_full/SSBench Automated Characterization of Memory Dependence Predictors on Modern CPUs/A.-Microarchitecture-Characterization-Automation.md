# *A. Microarchitecture Characterization Automation*

Characterizing microarchitectural components. Several automated tools characterize microarchitectural components. X-Ray [\[75\]](#page-14-8) extracts cache/TLB parameters, Plumber [\[21\]](#page-13-12) analyzes Arm cache optimizations, FetchBench [\[55\]](#page-14-9) identifies data prefetchers, and StrideRE [\[20\]](#page-13-11) characterizes stride prefetchers. However, these tools target CPU memory subsystems (caches, TLBs, prefetchers) lacking complex state machines or multi-IP indexing mechanisms, making them unsuitable for MDP characterization.

Characterizing microarchitectural properties. Several studies automate the analysis of specific microarchitectural properties. For example, Gerlach et al. [\[15\]](#page-13-30) and Rainer et al. [\[52\]](#page-14-24) reverse-engineer nonlinear hash functions like LLC bank indexing; Abel et al. [\[1\]](#page-13-31) model cache replacement policies;

<span id="page-12-0"></span>TABLE VIII NEW FINDINGS VIA SSBENCH COMPARED TO PREVIOUS RESEARCH\*

| Research                                               | SM     | Organization     | Activation       | Isolation | Speculation |
|--------------------------------------------------------|--------|------------------|------------------|-----------|-------------|
| Intel [35], [51]<br>AMD [37]<br>Arm [36]<br>Apple [36] | △<br>△ | △<br>△<br>△<br>△ | △<br>△<br>△<br>△ | △         | △           |

\* SM means State Machine. Activation means the activate condition and the chaining effects of MDP. Isolation includes processes and privileges, Speculation means the transient execution behavior of the MDP.

and Xiao et al. [\[69\]](#page-14-37) characterize out-of-order execution after exceptions on x86. While these methods do not fully characterize predictors, they help analyze certain properties. SSBench integrates some techniques, with others incorporable as independent modules.

#### *B. Security of MDP on Modern CPUs*

Prior manual analyses have uncovered MDP security vulnerabilities. Ragab et al. [\[51\]](#page-14-5) found Intel's MDP lacks crossprocess isolation, enabling Spectre-V4 attacks. Liu et al. [\[37\]](#page-13-7) identified similar issues on AMD Zen 3 and demonstrated new Spectre-V4 variants, while also showing Intel's MDP lacks SGX isolation, enabling MDPeek attacks [\[35\]](#page-13-8). Another study [\[36\]](#page-13-9) discovered MDPs on Arm and Apple CPUs but, limited to L-nS types, achieved only low-precision attacks like website fingerprinting.

SSBench's systematic, automated characterization uncovers more MDP features and security issues across Intel, AMD, Arm, and Apple CPUs than manual efforts. Table [VIII](#page-12-0) summarizes SSBench's new findings over previous research.

