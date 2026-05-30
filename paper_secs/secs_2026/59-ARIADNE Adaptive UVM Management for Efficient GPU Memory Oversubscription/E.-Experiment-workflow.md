# *E. Experiment workflow*

Root privileges are recommended. Perform the below instructions on an environment where video kernel modules (nvidia.ko, nvidia-uvm.ko, nvidia-drm.ko, and nvidiamodeset.ko) are not in use.

- 1. In base directory, 'bash run ARIADNE AC.sh'
- 2. In base directory, 'bash run breakdown.sh'
- 3. In base directory, 'bash run sstv analysis.sh'
- 4. In results/, run all three Python scripts.

### *F. Evaluation and expected results*

Key experimental results corresponding to Figures 9, 11, and 13 can be found in the result/ directory as raw data and graph images.

Figure 9: Performance Comparison Comparison of the geometric mean execution time of ARIADNE against AC (and SUV) across various workloads and oversubscription levels.

Figure 11: Component breakdown Analysis Illustrates the relative contributions of ARIADNE 's key components.

Figure 13: Parameter Sensitivity Shows normalized geometric mean performance across different parameter settings.

Please note that while exact numerical values and ratios may vary due to differences in absolute execution times across realsystem configurations, the trends remain consistent.

### *G. Experiment customization*

If the GPU memory capacity differs from the experimental environment, the memory reservation amount in run bench.sh must be adjusted accordingly.

### *H. Note*

The comparison with SUV is provided as an optional component in this artifact, as it requires a different system configuration (specifically, different GPU driver and CUDA versions). If you wish to run the SUV experiments, please first configure your system environment accordingly and then execute run SUV.sh located in the base directory. We utilized the provided artifact of SUV [7]; for detailed information, please refer to its paper and artifact documentation [6].

Please note that ARIADNE is a research prototype developed for experimental validation, not a commercial product intended for general deployment. Consequently, rare fatal bugs may occur. In such cases, please attempt to re-execute the experiment or reboot the system before retrying. Furthermore, while the design and concepts of ARIADNE are capable of supporting multi-GPU environments, the research prototype provided in this artifact has not been implemented or tested for such configurations.

### *I. Methodology*

Submission, reviewing and badging methodology:

- https://www.acm.org/publications/policies/artifactreview-and-badging-current
- https://cTuning.org/ae

### REFERENCES

- [1] N. Agarwal, D. Nellans, M. Stephenson, M. O'Connor, and S. W. Keckler, "Page placement strategies for gpus within heterogeneous memory systems," *SIGPLAN Not.*, vol. 50, no. 4, p. 607–618, Mar. 2015. [Online]. Available: https://doi.org/10.1145/2775054.2694381
- [2] T. Allen, B. Cooper, and R. Ge, "Fine-grain quantitative analysis of demand paging in unified virtual memory," *ACM Trans. Archit. Code Optim.*, vol. 21, no. 1, Jan. 2024. [Online]. Available: https://doi.org/10.1145/3632953
- [3] T. Allen and R. Ge, "Demystifying gpu uvm cost with deep runtime and workload analysis," in *2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2021, pp. 141–150.
- [4] ——, "In-depth analyses of unified virtual memory system for gpu accelerated computing," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, ser. SC '21. New York, NY, USA: Association for Computing Machinery, 2021. [Online]. Available: https://doi.org/10.1145/3458817. 3480855
- [5] R. Ausavarungnirun, J. Landgraf, V. Miller, S. Ghose, J. Gandhi, C. J. Rossbach, and O. Mutlu, "Mosaic: a gpu memory manager with application-transparent support for multiple page sizes," in *Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO-50 '17. New York, NY, USA: Association for Computing Machinery, 2017, p. 136–150. [Online]. Available: https://doi.org/10.1145/3123939.3123975
- [6] P. B, "Suv-micro2024," Sep. 2024. [Online]. Available: https: //doi.org/10.5281/zenodo.13743206
- [7] P. B, G. Cox, J. Vesely, and A. Basu, "Suv: Static analysis guided unified virtual memory," in *2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2024, pp. 293–308.
- [8] C.-H. Chang, J. Han, A. Sivasubramaniam, V. Sharma Mailthody, Z. Qureshi, and W.-M. Hwu, "Gmt: Gpu orchestrated memory tiering for the big data era," in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, ser. ASPLOS '24. New York, NY, USA: Association for Computing Machinery, 2024, p. 464–478. [Online]. Available: https://doi.org/10.1145/3620666.3651353

- [9] C.-H. Chang, A. Kumar, and A. Sivasubramaniam, "To move or not to move? page migration for irregular applications in over-subscribed gpu memory systems with dynamap," in *Proceedings of the 14th ACM International Conference on Systems and Storage*, ser. SYSTOR '21. New York, NY, USA: Association for Computing Machinery, 2021.
- [10] S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, S.-H. Lee, and K. Skadron, "Rodinia: A benchmark suite for heterogeneous computing," in *2009 IEEE International Symposium on Workload Characterization (IISWC)*, 2009, pp. 44–54.
- [11] S. Chien, I. Peng, and S. Markidis, "Performance evaluation of advanced features in cuda unified memory," in *2019 IEEE/ACM Workshop on Memory Centric High Performance Computing (MCHPC)*, 2019, pp. 50–57.
- [12] S. Choi, T. Kim, J. Jeong, R. Ausavarungnirun, M. Jeon, Y. Kwon, and J. Ahn, "Memory harvesting in Multi-GPU systems with hierarchical unified virtual memory," in *2022 USENIX Annual Technical Conference (USENIX ATC 22)*. Carlsbad, CA: USENIX Association, Jul. 2022, pp. 625–638. [Online]. Available: https: //www.usenix.org/conference/atc22/presentation/choi-sangjin
- [13] E. Choukse, M. B. Sullivan, M. O'Connor, M. Erez, J. Pool, D. Nellans, and S. W. Keckler, "Buddy compression: Enabling larger memory for deep learning and hpc workloads on gpus," in *2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2020, pp. 926–939.
- [14] I. D. Dio Lavore, D. Maffi, M. Arnaboldi, A. Delamare, D. Bonetta, and M. D. Santambrogio, "Grout: Transparent scale-out to overcome uvm's oversubscription slowdowns," in *2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, 2024, pp. 696–705.
- [15] D. Ganguly, R. Melhem, and J. Yang, "An adaptive framework for oversubscription management in cpu-gpu unified memory," in *2021 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, 2021, pp. 1212–1217.
- [16] D. Ganguly, Z. Zhang, J. Yang, and R. Melhem, "Interplay between hardware prefetcher and page eviction policy in cpu-gpu unified virtual memory," in *2019 ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)*, 2019, pp. 224–235.
- [17] ——, "Adaptive page migration for irregular data-intensive applications under gpu memory oversubscription," in *2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2020, pp. 451–461.
- [18] R. Garg, A. Mohan, M. Sullivan, and G. Cooperman, "Crum: Checkpoint-restart support for cuda's unified memory," in *2018 IEEE International Conference on Cluster Computing (CLUSTER)*, 2018, pp. 302–313.
- [19] R. Gayatri, K. Gott, and J. Deslippe, "Comparing managed memory and ats with and without prefetching on nvidia volta gpus," in *2019 IEEE/ACM Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS)*, 2019, pp. 41–46.
- [20] S. Go, H. Lee, J. Kim, J. Lee, M. K. Yoon, and W. W. Ro, "Earlyadaptor: An adaptive framework forproactive uvm memory management," in *2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2023, pp. 248–258.
- [21] S. Grauer-Gray, L. Xu, R. Searles, S. Ayalasomayajula, and J. Cavazos, "Auto-tuning a high-level language targeted to gpu codes," in *2012 Innovative Parallel Computing (InPar)*, 2012, pp. 1–10.
- [22] Y. Gu, W. Wu, Y. Li, and L. Chen, "Uvmbench: A comprehensive benchmark suite for researching unified virtual memory in gpus," *arXiv preprint arXiv:2007.09822*, 2020.
- [23] P. Harish and P. J. Narayanan, "Accelerating large graph algorithms on the gpu using cuda," in *Proceedings of the 14th International Conference on High Performance Computing*, ser. HiPC'07. Berlin, Heidelberg: Springer-Verlag, 2007, p. 197–208.
- [24] C.-C. Huang, G. Jin, and J. Li, "Swapadvisor: Pushing deep learning beyond the gpu memory limit via smart swapping," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 1341–1355. [Online]. Available: https://doi.org/10.1145/3373376.3378530
- [25] T. B. Jablin, J. A. Jablin, P. Prabhu, F. Liu, and D. I. August, "Dynamically managed data for cpu-gpu architectures," in *Proceedings of the Tenth International Symposium on Code Generation and Optimization*, ser. CGO '12. New York, NY, USA: Association

- for Computing Machinery, 2012, p. 165–174. [Online]. Available: https://doi.org/10.1145/2259016.2259038
- [26] T. B. Jablin, P. Prabhu, J. A. Jablin, N. P. Johnson, S. R. Beard, and D. I. August, "Automatic cpu-gpu communication management and optimization," in *Proceedings of the 32nd ACM SIGPLAN Conference on Programming Language Design and Implementation*, ser. PLDI '11. New York, NY, USA: Association for Computing Machinery, 2011, p. 142–151. [Online]. Available: https://doi.org/10.1145/1993498.1993516
- [27] Z. Jin and J. S. Vetter, "A benchmark suite for improving performance portability of the sycl programming model," in *2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2023, pp. 325–327.
- [28] J. Jung, J. Kim, and J. Lee, "Deepum: Tensor migration and prefetching in unified memory," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, ser. ASPLOS 2023. New York, NY, USA: Association for Computing Machinery, 2023, p. 207–221. [Online]. Available: https://doi.org/10.1145/3575693.3575736
- [29] M. Khairy, V. Nikiforov, D. Nellans, and T. G. Rogers, "Localitycentric data and threadblock management for massive gpus," in *2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2020, pp. 1022–1036.
- [30] H. Kim, J. Sim, P. Gera, R. Hadidi, and H. Kim, "Batch-aware unified memory management in gpus for irregular workloads," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 1357–1370. [Online]. Available: https://doi.org/10.1145/3373376.3378529
- [31] H. Kim and H. Han, "GPU thread throttling for page-level thrashing reduction via static analysis," *The Journal of Supercomputing*, Dec. 2023. [Online]. Available: https://link.springer.com/10.1007/s11227- 023-05787-y
- [32] C. Li, R. Ausavarungnirun, C. J. Rossbach, Y. Zhang, O. Mutlu, Y. Guo, and J. Yang, "A framework for memory oversubscription management in graphics processing units," in *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '19. New York, NY, USA: Association for Computing Machinery, 2019, p. 49–63. [Online]. Available: https://doi.org/10.1145/3297858.3304044
- [33] L. Li and B. Chapman, "Compiler assisted hybrid implicit and explicit gpu memory management under unified address space," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, ser. SC '19. New York, NY, USA: Association for Computing Machinery, 2019. [Online]. Available: https://doi.org/10.1145/3295500.3356141
- [34] M. Lin, Y. Feng, G. Cox, and H. Jeon, "Forest: Access-aware gpu uvm management," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, ser. ISCA '25. New York, NY, USA: Association for Computing Machinery, 2025, p. 137–152. [Online]. Available: https://doi.org/10.1145/3695053.3731047
- [35] M. Lin, K. Zhou, and P. Su, "Drgpum: Guiding memory optimization for gpu-accelerated applications," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, ser. ASPLOS 2023. New York, NY, USA: Association for Computing Machinery, 2023, p. 164–178. [Online]. Available: https://doi.org/10.1145/3582016.3582044
- [36] X. Long, X. Gong, B. Zhang, and H. Zhou, "An intelligent framework for oversubscription management in cpu-gpu unified memory," *J. Grid Comput.*, vol. 21, no. 1, Feb. 2023. [Online]. Available: https://doi.org/10.1007/s10723-023-09646-1
- [37] S. W. Min, V. S. Mailthody, Z. Qureshi, J. Xiong, E. Ebrahimi, and W.-m. Hwu, "Emogi: efficient memory-access for out-of-memory graphtraversal in gpus," *Proc. VLDB Endow.*, vol. 14, no. 2, p. 114–127, Oct. 2020. [Online]. Available: https://doi.org/10.14778/3425879.3425883
- [38] A. Nihaal and M. Mutyam, "Selective memory compression for gpu memory oversubscription management," in *Proceedings of the 53rd International Conference on Parallel Processing*, ser. ICPP '24. New York, NY, USA: Association for Computing Machinery, 2024, p. 189–198. [Online]. Available: https://doi.org/10.1145/3673038.3673058
- [39] NVIDIA, "Nvidia blackwell architecture whitepaper," https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidiartx-blackwell-gpu-architecture.pdf, accessed: August 10, 2025.

- [40] NVIDIA, "NVIDIA UVM," https://github.com/NVIDIA/open-gpukernel-modules/tree/main/kernel-open/nvidia-uvm, accessed: August 10, 2025.
- [41] ——, "OpenGPU-Kernel-Modules," https://github.com/NVIDIA/opengpu-kernel-modules, accessed: August 10, 2025.
- [42] Z. Qureshi, V. S. Mailthody, I. Gelado, S. Min, A. Masood, J. Park, J. Xiong, C. J. Newburn, D. Vainbrand, I.-H. Chung, M. Garland, W. Dally, and W.-m. Hwu, "Gpu-initiated on-demand high-throughput storage access in the bam system architecture," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, ser. ASPLOS 2023. New York, NY, USA: Association for Computing Machinery, 2023, p. 325–339. [Online]. Available: https://doi.org/10.1145/3575693.3575748
- [43] J. Ren, J. Luo, K. Wu, M. Zhang, H. Jeon, and D. Li, "Sentinel: Efficient tensor migration and allocation on heterogeneous memory systems for deep learning," in *2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, 2021, pp. 598–611.
- [44] M. Rhu, M. Sullivan, J. Leng, and M. Erez, "A localityaware memory hierarchy for energy-efficient gpu architectures," in *Proceedings of the 46th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO-46. New York, NY, USA: Association for Computing Machinery, 2013, p. 86–98. [Online]. Available: https://doi.org/10.1145/2540708.2540717
- [45] G. Schieffer, J. Wahlgren, J. Ren, J. Faj, and I. Peng, "Harnessing integrated cpu-gpu system memory for hpc: a first look into grace hopper," in *Proceedings of the 53rd International Conference on Parallel Processing*, ser. ICPP '24. New York, NY, USA: Association for Computing Machinery, 2024, p. 199–209. [Online]. Available: https://doi.org/10.1145/3673038.3673110
- [46] C. Shao, J. Guo, P. Wang, J. Wang, C. Li, and M. Guo, "Oversubscribing gpu unified virtual memory: Implications and suggestions," in *Proceedings of the 2022 ACM/SPEC on International Conference on Performance Engineering*, ser. ICPE '22. New York, NY, USA: Association for Computing Machinery, 2022, p. 67–75. [Online]. Available: https://doi.org/10.1145/3489525.3511691
- [47] H. Shin, S. Bang, H. Park, and D. Kim, "Safe: Sharing-aware prefetching for efficient gpu memory management with unified virtual memory," *IEEE Computer Architecture Letters*, vol. 24, no. 1, pp. 117–120, 2025.
- [48] T. Sultana, B. Allen, and A. Qasem, "Intelligent data placement on discrete gpu nodes with unified memory," in *Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques*, ser. PACT '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 139–151. [Online]. Available: https://doi.org/10.1145/3410463.3414651
- [49] J. R. Tramm and A. R. Siegel, "Memory bottlenecks and memory contention in multi-core monte carlo transport codes," *Annals of Nuclear Energy*, vol. 82, pp. 195–202, 2015, joint International Conference on Supercomputing in Nuclear Applications and Monte Carlo 2013, SNA + MC 2013. Pluri- and Trans-disciplinarity, Towards New Modeling and Numerical Simulation Paradigms. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0306454914004332
- [50] N. Vijaykumar, E. Ebrahimi, K. Hsieh, P. B. Gibbons, and O. Mutlu, "The locality descriptor: A holistic cross-layer abstraction to express data locality in gpus," in *2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA)*, 2018, pp. 829–842.
- [51] L. Wang, M. Jahre, A. Adileho, and L. Eeckhout, "Mdm: The gpu memory divergence model," in *2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2020, pp. 1009–1021.
- [52] Y. Wang, B. Li, M. T. I. Ziad, L. Eeckhout, J. Yang, A. Jaleel, and X. Tang, "Oasis: Object-aware page management for multi-gpu systems," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, 2025, pp. 1678–1692.
- [53] T. T. Yeh, R. N. Green, and T. G. Rogers, "Dimensionalityaware redundant simt instruction elimination," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 1327–1340. [Online]. Available: https://doi.org/10.1145/3373376. 3378520
- [54] Q. Yu, B. Childers, L. Huang, C. Qian, H. Guo, and Z. Wang, "Coordinated page prefetch and eviction for memory oversubscription management in gpus," in *2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, 2020, pp. 472–482.

- [55] Y. Yu, S. Kang, and Y. Park, "A compiler-based approach for gpgpu performance calibration using tlp modulation (wip paper)," in *Proceedings of the 20th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems*, ser. LCTES 2019. New York, NY, USA: Association for Computing Machinery, 2019, p. 193–197. [Online]. Available: https://doi.org/10.1145/3316482.3326343
- [56] H. Zhang, Y. Zhou, Y. Xue, Y. Liu, and J. Huang, "G10: Enabling an efficient unified gpu memory and storage architecture with smart tensor migrations," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 395–410. [Online]. Available: https://doi.org/10.1145/3613424.3614309
- [57] Z. Zhang, T. Allen, F. Yao, X. Gao, and R. Ge, "Tunnels for bootlegging: Fully reverse-engineering gpu tlbs for challenging isolation guarantees of nvidia mig," in *Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security*, ser. CCS '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 960–974. [Online]. Available: https://doi.org/10.1145/3576915.3616672
- [58] T. Zheng, D. Nellans, A. Zulfiqar, M. Stephenson, and S. W. Keckler, "Towards high performance paged memory for gpus," in *2016 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, 2016, pp. 345–357.
- [59] W. Zhu, G. Cox, J. Vesely, M. Hairgrove, A. L. Cox, and S. Rixner, "Uvm discard: Eliminating redundant memory transfers for accelerators," in *2022 IEEE International Symposium on Workload Characterization (IISWC)*, 2022, pp. 27–38.