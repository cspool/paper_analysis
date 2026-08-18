# VIII. CONCLUSION

Tracing frameworks introduce asymmetric delays when persisting traced data, affecting application behavior relative to untraced executions. We formally define correctness notions for traced executions and propose a novel *time dilation* approach to achieve them. We have realized time dilation in BULLETTIME, a tracing framework built atop Pin. Our evaluations of memory contiguity and synchronization studies on real-world workloads show that existing tracing approaches can cause application behavior to deviate by up to 20× relative to untraced execution. In contrast, BULLETTIME deviates by at most 10% for all evaluated workloads.

#### IX. ACKNOWLEDGEMENTS

We would like to thank our shepherd and anonymous ISCA reviewers for their valuable comments and insightful feedback. This work is supported in part by NSF awards 2112562, 2047220, an NSF Graduate Research Fellowship, a Meta research gift, and a NetApp Faculty Fellowship. Generative AI was used to assist in implementing BULLETTIME and in editing paper text.

#### REFERENCES

- [1] "Google workload traces version 2," https://console.cloud.google.com/ storage/browser/external-traces-v2, accessed: 2026-02-27.
- [2] M. Accetta, R. Baron, W. Bolosky, D. Golub, R. Rashid, A. Tevanian, and M. Young, "Mach: A new kernel foundation for unix development," 1986.
- [3] G. Altekar and I. Stoica, "Odr: Output-deterministic replay for multicore debugging," in *Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles*, 2009, pp. 193–206.
- [4] A. Arcangeli, "Transparent hugepage support," in *KVM forum*, vol. 9, 2010.
- [5] B. Atikoglu, Y. Xu, E. Frachtenberg, S. Jiang, and M. Paleczny, "Workload analysis of a large-scale key-value store," in *Proceedings of the 12th ACM SIGMETRICS/PERFORMANCE joint international conference on Measurement and Modeling of Computer Systems*, 2012, pp. 53–64.
- [6] K. Banker, D. Garrett, P. Bakkum, and S. Verch, *MongoDB in action: covers MongoDB version 3.0*. Simon and Schuster, 2016.
- [7] T. W. Barr, A. L. Cox, and S. Rixner, "Translation caching: skip, don't walk (the page table)," *ACM SIGARCH Computer Architecture News*, vol. 38, no. 3, pp. 48–59, 2010.
- [8] S. Beamer, K. Asanovic, and D. Patterson, "The gap benchmark suite," ´ *arXiv preprint arXiv:1508.03619*, 2015.
- [9] R. Bera, K. Kanellopoulos, A. Nori, T. Shahroodi, S. Subramoney, and O. Mutlu, "Pythia: A customizable hardware prefetching framework using online reinforcement learning," in *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*, 2021, pp. 1121–1137.
- [10] T. Bergan, O. Anderson, J. Devietti, L. Ceze, and D. Grossman, "Coredet: A compiler and runtime system for deterministic multithreaded execution," in *Proceedings of the fifteenth International Conference on Architectural support for programming languages and operating systems*, 2010, pp. 53–64.
- [11] T. Bergan, N. Hunt, L. Ceze, and S. D. Gribble, "Deterministic process groups in {dOS}," in *9th USENIX Symposium on Operating Systems Design and Implementation (OSDI 10)*, 2010.
- [12] E. D. Berger, T. Yang, T. Liu, and G. Novark, "Grace: Safe multithreaded programming for c/c++," in *Proceedings of the 24th ACM SIGPLAN conference on Object oriented programming systems languages and applications*, 2009, pp. 81–96.
- [13] A. Bhattacharjee, "Translation-triggered prefetching," in *Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems*, 2017, pp. 63–76.
- [14] N. L. Binkert, R. G. Dreslinski, L. R. Hsu, K. T. Lim, A. G. Saidi, and S. K. Reinhardt, "The m5 simulator: Modeling networked systems," *Ieee micro*, vol. 26, no. 4, pp. 52–60, 2006.
- [15] D. Bruening and S. Amarasinghe, "Efficient, transparent, and comprehensive runtime code manipulation," 2004.
- [16] J. Carlson, *Redis in action*. Simon and Schuster, 2013.

- [17] J. B. Chen, D. W. Wall, and A. Borg, "Software methods for system address tracing: implementation and validation," *WRL Research Report 94/6*, 1994.
- [18] D. W. Clark and J. S. Emer, "Performance of the vax-11/780 translation buffer: Simulation and measurement," *ACM Transactions on Computer Systems (TOCS)*, vol. 3, no. 1, pp. 31–62, 1985.
- [19] Y. Collet, "LZ4: Extremely fast compression algorithm," https://github. com/lz4/lz4, 2011, accessed: 2026-02-27.
- [20] Y. Collet and M. Kucherawy, "Zstandard compression and the application/zstd media type," Tech. Rep., 2018.
- [21] B. F. Cooper, A. Silberstein, E. Tam, R. Ramakrishnan, and R. Sears, "Benchmarking cloud serving systems with YCSB," in *Proceedings of the 1st ACM symposium on Cloud computing*, 2010, pp. 143–154.
- [22] J. Corbet, "Flexible-order anonymous folios," LWN.net, 2023, accessed: 2026-03-04. [Online]. Available: https://lwn.net/Articles/932386/
- [23] ——, "Large folios for anonymous memory," LWN.net, 2023, accessed: 2026-03-04. [Online]. Available: https://lwn.net/Articles/937239/
- [24] G. Cox and A. Bhattacharjee, "Efficient address translation for architectures with multiple page sizes," *ACM SIGPLAN Notices*, vol. 52, no. 4, pp. 435–448, 2017.
- [25] H. Cui, J. Simsa, Y.-H. Lin, H. Li, B. Blum, X. Xu, J. Yang, G. A. Gibson, and R. E. Bryant, "Parrot: A practical runtime for deterministic, stable, and reliable threads," in *Proceedings of the Twenty-Fourth ACM Symposium on Operating Systems Principles*, 2013, pp. 388–405.
- [26] C. Curtsinger and E. D. Berger, "Coz: Finding code that counts with causal profiling," in *Proceedings of the 25th Symposium on Operating Systems Principles*, 2015, pp. 184–197.
- [27] J. Devietti, B. Lucia, L. Ceze, and M. Oskin, "Dmp: Deterministic shared memory multiprocessing," in *Proceedings of the 14th international conference on Architectural support for programming languages and operating systems*, 2009, pp. 85–96.
- [28] J. Devietti, B. P. Wood, K. Strauss, L. Ceze, D. Grossman, and S. Qadeer, "Radish: always-on sound and complete ra d etection in s oftware and h ardware," *ACM SIGARCH Computer Architecture News*, vol. 40, no. 3, pp. 201–212, 2012.
- [29] J. Evans, M. Andersch, V. Sethi, G. Brito, and V. Mehta, "Nvidia grace hopper superchip architecture in-depth," Jul 2025. [Online]. Available: https://developer.nvidia.com/blog/nvidia-gracehopper-superchip-architecture-in-depth
- [30] B. Fitzpatrick, "Distributed caching with memcached," *Linux journal*, vol. 2004, no. 124, p. 5, 2004.
- [31] G. Gerganov, "llama.cpp," https://github.com/ggml-org/llama.cpp, 2025.
- [32] S. R. Goldschmidt and J. L. Hennessy, "The accuracy of trace-driven simulations of multiprocessors," *ACM SIGMETRICS Performance Evaluation Review*, vol. 21, no. 1, pp. 146–157, 1993.
- [33] F. Guvenilir and Y. N. Patt, "Tailored page sizes," in *2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2020, pp. 900–912.
- [34] D. R. Hower, P. Dudnik, M. D. Hill, and D. A. Wood, "Calvin: Deterministic or not? free will to choose," in *2011 IEEE 17th International Symposium on High Performance Computer Architecture*. IEEE, 2011, pp. 333–334.
- [35] D. R. Hower and M. D. Hill, "Rerun: Exploiting episodes for lightweight memory race recording," *ACM SIGARCH computer architecture news*, vol. 36, no. 3, pp. 265–276, 2008.
- [36] Y. Huang, L. Chen, Z. Cui, Y. Ruan, Y. Bao, M. Chen, and N. Sun, "Hmtt: A hybrid hardware/software tracing system for bridging the dram access trace's semantic gap," *ACM Transactions on Architecture and Code Optimization (TACO)*, vol. 11, no. 1, pp. 1–25, 2014.
- [37] Intel, "Intel® processor trace," https://edc.intel.com/content/www/us/ en/design/products/platforms/processor-and-core-i3-n-series-datasheetvolume-1-of-2/001/intel-processor-trace/, 2015.
- [38] R. Jagtap, S. Diestelhorst, and A. Hansson, "Elastic traces for fast and accurate system performance exploration," in *2016 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*. IEEE, 2016, pp. 147–148.
- [39] A. Jaleel, R. S. Cohn, C.-K. Luk, and B. Jacob, "Cmp\$im: A pin-based on-the-fly multi-core cache simulator," in *Proceedings of the Fourth Annual Workshop on Modeling, Benchmarking and Simulation (MoBS), co-located with ISCA*, 2008, pp. 28–36.
- [40] A. Khandelwal, Y. Tang, R. Agarwal, A. Akella, and I. Stoica, "Jiffy: elastic far-memory for stateful serverless analytics," in *Proceedings of the Seventeenth European Conference on Computer Systems*, ser. EuroSys '22. New York, NY, USA: Association

- for Computing Machinery, 2022, p. 697–713. [Online]. Available: https://doi.org/10.1145/3492321.3527539
- [41] A. Klimovic, Y. Wang, P. Stuedi, A. Trivedi, J. Pfefferle, and C. Kozyrakis, "Pocket: Elastic ephemeral storage for serverless analytics," in *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*. Carlsbad, CA: USENIX Association, Oct. 2018, pp. 427–444. [Online]. Available: https: //www.usenix.org/conference/osdi18/presentation/klimovic
- [42] U. Laghi, S. Manoni, E. Parisi, and A. Bartolini, "Efficient trace for risc-v: Design, evaluation, and integration in cva6," *arXiv preprint arXiv:2504.01972*, 2025.
- [43] D. Lee, B. Wester, K. Veeraraghavan, S. Narayanasamy, P. M. Chen, and J. Flinn, "Respec: efficient online multiprocessor replayvia speculation and external determinism," *ACM Sigplan Notices*, vol. 45, no. 3, pp. 77–90, 2010.
- [44] S.-s. Lee, Y. Yu, Y. Tang, A. Khandelwal, L. Zhong, and A. Bhattacharjee, "Mind: In-network memory management for disaggregated data centers," in *Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles*, 2021, pp. 488–504.
- [45] T. Liu, C. Curtsinger, and E. D. Berger, "Dthreads: efficient deterministic multithreading," in *Proceedings of the Twenty-Third ACM Symposium on Operating Systems Principles*, 2011, pp. 327–336.
- [46] H. Lu, K. Doshi, R. Seth, and J. Tran, "Using hugetlbfs for mapping application text regions," in *Proceedings of the Linux Symposium*, vol. 2, 2006, pp. 75–82.
- [47] J. M. Lucas and M. S. Saccucci, "Exponentially weighted moving average control schemes: properties and enhancements," *Technometrics*, vol. 32, no. 1, pp. 1–12, 1990.
- [48] C.-K. Luk, R. Cohn, R. Muth, H. Patil, A. Klauser, G. Lowney, S. Wallace, V. J. Reddi, and K. Hazelwood, "Pin: building customized program analysis tools with dynamic instrumentation," *Acm sigplan notices*, vol. 40, no. 6, pp. 190–200, 2005.
- [49] W. Luo, R. Fan, Z. Li, D. Du, Q. Wang, and X. Chu, "Benchmarking and dissecting the nvidia hopper gpu architecture," in *2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*. IEEE, 2024, pp. 656–667.
- [50] P. S. Magnusson, M. Christensson, J. Eskilson, D. Forsgren, G. Hallberg, J. Hogberg, F. Larsson, A. Moestedt, and B. Werner, "Simics: A full system simulation platform," *Computer*, vol. 35, no. 2, pp. 50–58, 2002.
- [51] M. Martonosi, A. Gupta, and T. Anderson, "Memspy: Analyzing memory system bottlenecks in programs," *ACM SIGMETRICS Performance Evaluation Review*, vol. 20, no. 1, pp. 1–12, 1992.
- [52] ——, "Effectiveness of trace sampling for performance debugging tools," *ACM SIGMETRICS Performance Evaluation Review*, vol. 21, no. 1, pp. 248–259, 1993.
- [53] T. Merrifield, S. Roghanchi, J. Devietti, and J. Eriksson, "Lazy determinism for faster deterministic multithreading," in *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2019, pp. 879–891.
- [54] P. Michaud, "Online compression of cache-filtered address traces," in *2009 IEEE International Symposium on Performance Analysis of Systems and Software*. IEEE, 2009, pp. 185–194.
- [55] S. Mirbagher-Ajorpaz, E. Garza, G. Pokam, and D. A. Jimenez, "Chirp: ´ Control-flow history reuse prediction," in *2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, 2020, pp. 131–145.
- [56] O. S. Navarro Leija, K. Shiptoski, R. G. Scott, B. Wang, N. Renner, R. R. Newton, and J. Devietti, "Reproducible containers," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 167–182.
- [57] M. Olszewski, J. Ansel, and S. Amarasinghe, "Kendo: efficient deterministic multithreading in software," in *Proceedings of the 14th international conference on Architectural support for programming languages and operating systems*, 2009, pp. 97–108.
- [58] N.-C. Papadopoulos, S. Psomadakis, V. Karakostas, N. Koziris, and D. N. Pnevmatikatos, "Design, implementation and evaluation of the svnapot extension on a risc-v processor," *arXiv preprint arXiv:2406.17802*, 2024.
- [59] H. Patil, R. Cohn, M. Charney, R. Kapoor, A. Sun, and A. Karunanidhi, "Pinpointing representative portions of large intel® itanium® programs with dynamic instrumentation," in *37th International Symposium on Microarchitecture (MICRO-37'04)*. IEEE, 2004, pp. 81–92.
- [60] H. Patil, C. Pereira, M. Stallcup, G. Lueck, and J. Cownie, "Pinplay: a framework for deterministic replay and reproducible analysis of parallel

- programs," in *Proceedings of the 8th annual IEEE/ACM international symposium on Code generation and optimization*, 2010, pp. 2–11.
- [61] E. Perelman, G. Hamerly, M. Van Biesbrouck, T. Sherwood, and B. Calder, "Using simpoint for accurate and efficient simulation," *ACM SIGMETRICS Performance Evaluation Review*, vol. 31, no. 1, pp. 318– 319, 2003.
- [62] B. Pham, A. Bhattacharjee, Y. Eckert, and G. H. Loh, "Increasing tlb reach by exploiting clustering in page translations," in *2014 IEEE 20th International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2014, pp. 558–567.
- [63] B. Pham, V. Vaidyanathan, A. Jaleel, and A. Bhattacharjee, "Colt: Coalesced large-reach tlbs," in *2012 45th Annual IEEE/ACM International Symposium on Microarchitecture*. IEEE, 2012, pp. 258–269.
- [64] B. Pham, J. Vesely, G. H. Loh, and A. Bhattacharjee, "Large pages and ` lightweight memory management in virtualized environments: Can you have it both ways?" in *Proceedings of the 48th International Symposium on Microarchitecture*, 2015, pp. 1–12.
- [65] Q. Pu, S. Venkataraman, and I. Stoica, "Shuffling, fast and slow: Scalable analytics on serverless infrastructure," in *16th USENIX Symposium on Networked Systems Design and Implementation (NSDI 19)*. Boston, MA: USENIX Association, Feb. 2019, pp. 193–206. [Online]. Available: https://www.usenix.org/conference/nsdi19/presentation/pu
- [66] P. Ranganathan and V. Lee, "Advancing systems research with open-source Google workload traces." [Online]. Available: https://cloud.google.com/blog/topics/systems/workload-traces-forgoogle-warehouse-scale-computers
- [67] D. Sanchez and C. Kozyrakis, "Zsim: Fast and accurate microarchitectural simulation of thousand-core systems," *ACM SIGARCH Computer architecture news*, vol. 41, no. 3, pp. 475–486, 2013.
- [68] Y. Shan, S.-Y. Tsai, and Y. Zhang, "Distributed shared persistent memory," in *Proceedings of the 2017 Symposium on Cloud Computing*, 2017, pp. 323–337.
- [69] Z. Shi, A. Jain, K. Swersky, M. Hashemi, P. Ranganathan, and C. Lin, "A hierarchical neural model of data prefetching," in *Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2021, pp. 861–873.
- [70] R. T. Short and H. M. Levy, "A simulation study of two-level caches," *ACM SIGARCH Computer Architecture News*, vol. 16, no. 2, pp. 81–88, 1988.
- [71] K. Sriram, I. Karageorgos, X. Wen, J. Vesely, N. Lindsay, M. Wu, ` L. Khazan, R. P. Pothukuchi, R. Manohar, and A. Bhattacharjee, "Halo: A hardware–software co-designed processor for brain–computer interfaces," *Ieee micro*, vol. 43, no. 3, pp. 64–72, 2023.
- [72] A. P. Su, J. Kuo, K.-J. Lee, J. Huang, G.-A. Jian, C.-A. Chien, J.-I. Guo, and C.-H. Chen, "Multi-core software/hardware co-debug platform with arm coresight™, on-chip test architecture and axi/ahb bus monitor," in *Proceedings of 2011 International Symposium on VLSI Design, Automation and Test*. IEEE, 2011, pp. 1–6.
- [73] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
- [74] R. Uhlig, D. Nagle, T. Mudge, S. Sechrest, and J. Emer, "Instruction fetching: Coping with code bloat," *ACM SIGARCH Computer Architecture News*, vol. 23, no. 2, pp. 345–356, 1995.
- [75] C. Wang, P. Balaji, and M. Snir, "Pilgrim: scalable and (near) lossless mpi tracing," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2021, pp. 1–14.
- [76] C.-J. Wu, A. Jaleel, W. Hasenplaugh, M. Martonosi, S. C. Steely Jr, and J. Emer, "Ship: Signature-based hit predictor for high performance caching," in *Proceedings of the 44th Annual IEEE/ACM International Symposium on Microarchitecture*, 2011, pp. 430–441.
- [77] J. Xu, M. Dong, Q. Tian, Z. Tian, T. Xin, and H. Chen, "Asyncfs: Metadata updates made asynchronous for distributed filesystems with in-network coordination," *arXiv preprint arXiv:2410.08618*, 2024.
- [78] Z. Yan, D. Lustig, D. Nellans, and A. Bhattacharjee, "Translation ranger: Operating system support for contiguity-aware tlbs," in *Proceedings of the 46th International Symposium on Computer Architecture*, 2019, pp. 698–710.
- [79] J. Yang, H. Cui, J. Wu, Y. Tang, and G. Hu, "Making parallel programs reliable with stable multithreading," *Communications of the ACM*, vol. 57, no. 3, pp. 58–69, 2014.

- [80] J. J. Yi, D. J. Lilja, and D. M. Hawkins, "A statistically rigorous approach for improving simulation methodology," in *The Ninth International Symposium on High-Performance Computer Architecture, 2003. HPCA-9 2003. Proceedings.* IEEE, 2003, pp. 281–291.
- [81] R. Zhang, S. Biswas, V. Balaji, M. D. Bond, and B. Lucia, "Peacenik: Architecture support for not failing under fail-stop memory consistency," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 317–333.
- [82] K. Zhao, K. Xue, Z. Wang, D. Schatzberg, L. Yang, A. Manousis, J. Weiner, R. Van Riel, B. Sharma, C. Tang *et al.*, "Contiguitas: The pursuit of physical memory contiguity in datacenters," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, 2023, pp. 1–15.# VIII. CONCLUSION

Tracing frameworks introduce asymmetric delays when persisting traced data, affecting application behavior relative to untraced executions. We formally define correctness notions for traced executions and propose a novel *time dilation* approach to achieve them. We have realized time dilation in BULLETTIME, a tracing framework built atop Pin. Our evaluations of memory contiguity and synchronization studies on real-world workloads show that existing tracing approaches can cause application behavior to deviate by up to 20× relative to untraced execution. In contrast, BULLETTIME deviates by at most 10% for all evaluated workloads.

#### IX. ACKNOWLEDGEMENTS

We would like to thank our shepherd and anonymous ISCA reviewers for their valuable comments and insightful feedback. This work is supported in part by NSF awards 2112562, 2047220, an NSF Graduate Research Fellowship, a Meta research gift, and a NetApp Faculty Fellowship. Generative AI was used to assist in implementing BULLETTIME and in editing paper text.

#### REFERENCES

- [1] "Google workload traces version 2," https://console.cloud.google.com/ storage/browser/external-traces-v2, accessed: 2026-02-27.
- [2] M. Accetta, R. Baron, W. Bolosky, D. Golub, R. Rashid, A. Tevanian, and M. Young, "Mach: A new kernel foundation for unix development," 1986.
- [3] G. Altekar and I. Stoica, "Odr: Output-deterministic replay for multicore debugging," in *Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles*, 2009, pp. 193–206.
- [4] A. Arcangeli, "Transparent hugepage support," in *KVM forum*, vol. 9, 2010.
- [5] B. Atikoglu, Y. Xu, E. Frachtenberg, S. Jiang, and M. Paleczny, "Workload analysis of a large-scale key-value store," in *Proceedings of the 12th ACM SIGMETRICS/PERFORMANCE joint international conference on Measurement and Modeling of Computer Systems*, 2012, pp. 53–64.
- [6] K. Banker, D. Garrett, P. Bakkum, and S. Verch, *MongoDB in action: covers MongoDB version 3.0*. Simon and Schuster, 2016.
- [7] T. W. Barr, A. L. Cox, and S. Rixner, "Translation caching: skip, don't walk (the page table)," *ACM SIGARCH Computer Architecture News*, vol. 38, no. 3, pp. 48–59, 2010.
- [8] S. Beamer, K. Asanovic, and D. Patterson, "The gap benchmark suite," ´ *arXiv preprint arXiv:1508.03619*, 2015.
- [9] R. Bera, K. Kanellopoulos, A. Nori, T. Shahroodi, S. Subramoney, and O. Mutlu, "Pythia: A customizable hardware prefetching framework using online reinforcement learning," in *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*, 2021, pp. 1121–1137.
- [10] T. Bergan, O. Anderson, J. Devietti, L. Ceze, and D. Grossman, "Coredet: A compiler and runtime system for deterministic multithreaded execution," in *Proceedings of the fifteenth International Conference on Architectural support for programming languages and operating systems*, 2010, pp. 53–64.
- [11] T. Bergan, N. Hunt, L. Ceze, and S. D. Gribble, "Deterministic process groups in {dOS}," in *9th USENIX Symposium on Operating Systems Design and Implementation (OSDI 10)*, 2010.
- [12] E. D. Berger, T. Yang, T. Liu, and G. Novark, "Grace: Safe multithreaded programming for c/c++," in *Proceedings of the 24th ACM SIGPLAN conference on Object oriented programming systems languages and applications*, 2009, pp. 81–96.
- [13] A. Bhattacharjee, "Translation-triggered prefetching," in *Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems*, 2017, pp. 63–76.
- [14] N. L. Binkert, R. G. Dreslinski, L. R. Hsu, K. T. Lim, A. G. Saidi, and S. K. Reinhardt, "The m5 simulator: Modeling networked systems," *Ieee micro*, vol. 26, no. 4, pp. 52–60, 2006.
- [15] D. Bruening and S. Amarasinghe, "Efficient, transparent, and comprehensive runtime code manipulation," 2004.
- [16] J. Carlson, *Redis in action*. Simon and Schuster, 2013.

- [17] J. B. Chen, D. W. Wall, and A. Borg, "Software methods for system address tracing: implementation and validation," *WRL Research Report 94/6*, 1994.
- [18] D. W. Clark and J. S. Emer, "Performance of the vax-11/780 translation buffer: Simulation and measurement," *ACM Transactions on Computer Systems (TOCS)*, vol. 3, no. 1, pp. 31–62, 1985.
- [19] Y. Collet, "LZ4: Extremely fast compression algorithm," https://github. com/lz4/lz4, 2011, accessed: 2026-02-27.
- [20] Y. Collet and M. Kucherawy, "Zstandard compression and the application/zstd media type," Tech. Rep., 2018.
- [21] B. F. Cooper, A. Silberstein, E. Tam, R. Ramakrishnan, and R. Sears, "Benchmarking cloud serving systems with YCSB," in *Proceedings of the 1st ACM symposium on Cloud computing*, 2010, pp. 143–154.
- [22] J. Corbet, "Flexible-order anonymous folios," LWN.net, 2023, accessed: 2026-03-04. [Online]. Available: https://lwn.net/Articles/932386/
- [23] ——, "Large folios for anonymous memory," LWN.net, 2023, accessed: 2026-03-04. [Online]. Available: https://lwn.net/Articles/937239/
- [24] G. Cox and A. Bhattacharjee, "Efficient address translation for architectures with multiple page sizes," *ACM SIGPLAN Notices*, vol. 52, no. 4, pp. 435–448, 2017.
- [25] H. Cui, J. Simsa, Y.-H. Lin, H. Li, B. Blum, X. Xu, J. Yang, G. A. Gibson, and R. E. Bryant, "Parrot: A practical runtime for deterministic, stable, and reliable threads," in *Proceedings of the Twenty-Fourth ACM Symposium on Operating Systems Principles*, 2013, pp. 388–405.
- [26] C. Curtsinger and E. D. Berger, "Coz: Finding code that counts with causal profiling," in *Proceedings of the 25th Symposium on Operating Systems Principles*, 2015, pp. 184–197.
- [27] J. Devietti, B. Lucia, L. Ceze, and M. Oskin, "Dmp: Deterministic shared memory multiprocessing," in *Proceedings of the 14th international conference on Architectural support for programming languages and operating systems*, 2009, pp. 85–96.
- [28] J. Devietti, B. P. Wood, K. Strauss, L. Ceze, D. Grossman, and S. Qadeer, "Radish: always-on sound and complete ra d etection in s oftware and h ardware," *ACM SIGARCH Computer Architecture News*, vol. 40, no. 3, pp. 201–212, 2012.
- [29] J. Evans, M. Andersch, V. Sethi, G. Brito, and V. Mehta, "Nvidia grace hopper superchip architecture in-depth," Jul 2025. [Online]. Available: https://developer.nvidia.com/blog/nvidia-gracehopper-superchip-architecture-in-depth
- [30] B. Fitzpatrick, "Distributed caching with memcached," *Linux journal*, vol. 2004, no. 124, p. 5, 2004.
- [31] G. Gerganov, "llama.cpp," https://github.com/ggml-org/llama.cpp, 2025.
- [32] S. R. Goldschmidt and J. L. Hennessy, "The accuracy of trace-driven simulations of multiprocessors," *ACM SIGMETRICS Performance Evaluation Review*, vol. 21, no. 1, pp. 146–157, 1993.
- [33] F. Guvenilir and Y. N. Patt, "Tailored page sizes," in *2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2020, pp. 900–912.
- [34] D. R. Hower, P. Dudnik, M. D. Hill, and D. A. Wood, "Calvin: Deterministic or not? free will to choose," in *2011 IEEE 17th International Symposium on High Performance Computer Architecture*. IEEE, 2011, pp. 333–334.
- [35] D. R. Hower and M. D. Hill, "Rerun: Exploiting episodes for lightweight memory race recording," *ACM SIGARCH computer architecture news*, vol. 36, no. 3, pp. 265–276, 2008.
- [36] Y. Huang, L. Chen, Z. Cui, Y. Ruan, Y. Bao, M. Chen, and N. Sun, "Hmtt: A hybrid hardware/software tracing system for bridging the dram access trace's semantic gap," *ACM Transactions on Architecture and Code Optimization (TACO)*, vol. 11, no. 1, pp. 1–25, 2014.
- [37] Intel, "Intel® processor trace," https://edc.intel.com/content/www/us/ en/design/products/platforms/processor-and-core-i3-n-series-datasheetvolume-1-of-2/001/intel-processor-trace/, 2015.
- [38] R. Jagtap, S. Diestelhorst, and A. Hansson, "Elastic traces for fast and accurate system performance exploration," in *2016 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*. IEEE, 2016, pp. 147–148.
- [39] A. Jaleel, R. S. Cohn, C.-K. Luk, and B. Jacob, "Cmp\$im: A pin-based on-the-fly multi-core cache simulator," in *Proceedings of the Fourth Annual Workshop on Modeling, Benchmarking and Simulation (MoBS), co-located with ISCA*, 2008, pp. 28–36.
- [40] A. Khandelwal, Y. Tang, R. Agarwal, A. Akella, and I. Stoica, "Jiffy: elastic far-memory for stateful serverless analytics," in *Proceedings of the Seventeenth European Conference on Computer Systems*, ser. EuroSys '22. New York, NY, USA: Association

- for Computing Machinery, 2022, p. 697–713. [Online]. Available: https://doi.org/10.1145/3492321.3527539
- [41] A. Klimovic, Y. Wang, P. Stuedi, A. Trivedi, J. Pfefferle, and C. Kozyrakis, "Pocket: Elastic ephemeral storage for serverless analytics," in *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*. Carlsbad, CA: USENIX Association, Oct. 2018, pp. 427–444. [Online]. Available: https: //www.usenix.org/conference/osdi18/presentation/klimovic
- [42] U. Laghi, S. Manoni, E. Parisi, and A. Bartolini, "Efficient trace for risc-v: Design, evaluation, and integration in cva6," *arXiv preprint arXiv:2504.01972*, 2025.
- [43] D. Lee, B. Wester, K. Veeraraghavan, S. Narayanasamy, P. M. Chen, and J. Flinn, "Respec: efficient online multiprocessor replayvia speculation and external determinism," *ACM Sigplan Notices*, vol. 45, no. 3, pp. 77–90, 2010.
- [44] S.-s. Lee, Y. Yu, Y. Tang, A. Khandelwal, L. Zhong, and A. Bhattacharjee, "Mind: In-network memory management for disaggregated data centers," in *Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles*, 2021, pp. 488–504.
- [45] T. Liu, C. Curtsinger, and E. D. Berger, "Dthreads: efficient deterministic multithreading," in *Proceedings of the Twenty-Third ACM Symposium on Operating Systems Principles*, 2011, pp. 327–336.
- [46] H. Lu, K. Doshi, R. Seth, and J. Tran, "Using hugetlbfs for mapping application text regions," in *Proceedings of the Linux Symposium*, vol. 2, 2006, pp. 75–82.
- [47] J. M. Lucas and M. S. Saccucci, "Exponentially weighted moving average control schemes: properties and enhancements," *Technometrics*, vol. 32, no. 1, pp. 1–12, 1990.
- [48] C.-K. Luk, R. Cohn, R. Muth, H. Patil, A. Klauser, G. Lowney, S. Wallace, V. J. Reddi, and K. Hazelwood, "Pin: building customized program analysis tools with dynamic instrumentation," *Acm sigplan notices*, vol. 40, no. 6, pp. 190–200, 2005.
- [49] W. Luo, R. Fan, Z. Li, D. Du, Q. Wang, and X. Chu, "Benchmarking and dissecting the nvidia hopper gpu architecture," in *2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*. IEEE, 2024, pp. 656–667.
- [50] P. S. Magnusson, M. Christensson, J. Eskilson, D. Forsgren, G. Hallberg, J. Hogberg, F. Larsson, A. Moestedt, and B. Werner, "Simics: A full system simulation platform," *Computer*, vol. 35, no. 2, pp. 50–58, 2002.
- [51] M. Martonosi, A. Gupta, and T. Anderson, "Memspy: Analyzing memory system bottlenecks in programs," *ACM SIGMETRICS Performance Evaluation Review*, vol. 20, no. 1, pp. 1–12, 1992.
- [52] ——, "Effectiveness of trace sampling for performance debugging tools," *ACM SIGMETRICS Performance Evaluation Review*, vol. 21, no. 1, pp. 248–259, 1993.
- [53] T. Merrifield, S. Roghanchi, J. Devietti, and J. Eriksson, "Lazy determinism for faster deterministic multithreading," in *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2019, pp. 879–891.
- [54] P. Michaud, "Online compression of cache-filtered address traces," in *2009 IEEE International Symposium on Performance Analysis of Systems and Software*. IEEE, 2009, pp. 185–194.
- [55] S. Mirbagher-Ajorpaz, E. Garza, G. Pokam, and D. A. Jimenez, "Chirp: ´ Control-flow history reuse prediction," in *2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, 2020, pp. 131–145.
- [56] O. S. Navarro Leija, K. Shiptoski, R. G. Scott, B. Wang, N. Renner, R. R. Newton, and J. Devietti, "Reproducible containers," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 167–182.
- [57] M. Olszewski, J. Ansel, and S. Amarasinghe, "Kendo: efficient deterministic multithreading in software," in *Proceedings of the 14th international conference on Architectural support for programming languages and operating systems*, 2009, pp. 97–108.
- [58] N.-C. Papadopoulos, S. Psomadakis, V. Karakostas, N. Koziris, and D. N. Pnevmatikatos, "Design, implementation and evaluation of the svnapot extension on a risc-v processor," *arXiv preprint arXiv:2406.17802*, 2024.
- [59] H. Patil, R. Cohn, M. Charney, R. Kapoor, A. Sun, and A. Karunanidhi, "Pinpointing representative portions of large intel® itanium® programs with dynamic instrumentation," in *37th International Symposium on Microarchitecture (MICRO-37'04)*. IEEE, 2004, pp. 81–92.
- [60] H. Patil, C. Pereira, M. Stallcup, G. Lueck, and J. Cownie, "Pinplay: a framework for deterministic replay and reproducible analysis of parallel

- programs," in *Proceedings of the 8th annual IEEE/ACM international symposium on Code generation and optimization*, 2010, pp. 2–11.
- [61] E. Perelman, G. Hamerly, M. Van Biesbrouck, T. Sherwood, and B. Calder, "Using simpoint for accurate and efficient simulation," *ACM SIGMETRICS Performance Evaluation Review*, vol. 31, no. 1, pp. 318– 319, 2003.
- [62] B. Pham, A. Bhattacharjee, Y. Eckert, and G. H. Loh, "Increasing tlb reach by exploiting clustering in page translations," in *2014 IEEE 20th International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2014, pp. 558–567.
- [63] B. Pham, V. Vaidyanathan, A. Jaleel, and A. Bhattacharjee, "Colt: Coalesced large-reach tlbs," in *2012 45th Annual IEEE/ACM International Symposium on Microarchitecture*. IEEE, 2012, pp. 258–269.
- [64] B. Pham, J. Vesely, G. H. Loh, and A. Bhattacharjee, "Large pages and ` lightweight memory management in virtualized environments: Can you have it both ways?" in *Proceedings of the 48th International Symposium on Microarchitecture*, 2015, pp. 1–12.
- [65] Q. Pu, S. Venkataraman, and I. Stoica, "Shuffling, fast and slow: Scalable analytics on serverless infrastructure," in *16th USENIX Symposium on Networked Systems Design and Implementation (NSDI 19)*. Boston, MA: USENIX Association, Feb. 2019, pp. 193–206. [Online]. Available: https://www.usenix.org/conference/nsdi19/presentation/pu
- [66] P. Ranganathan and V. Lee, "Advancing systems research with open-source Google workload traces." [Online]. Available: https://cloud.google.com/blog/topics/systems/workload-traces-forgoogle-warehouse-scale-computers
- [67] D. Sanchez and C. Kozyrakis, "Zsim: Fast and accurate microarchitectural simulation of thousand-core systems," *ACM SIGARCH Computer architecture news*, vol. 41, no. 3, pp. 475–486, 2013.
- [68] Y. Shan, S.-Y. Tsai, and Y. Zhang, "Distributed shared persistent memory," in *Proceedings of the 2017 Symposium on Cloud Computing*, 2017, pp. 323–337.
- [69] Z. Shi, A. Jain, K. Swersky, M. Hashemi, P. Ranganathan, and C. Lin, "A hierarchical neural model of data prefetching," in *Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2021, pp. 861–873.
- [70] R. T. Short and H. M. Levy, "A simulation study of two-level caches," *ACM SIGARCH Computer Architecture News*, vol. 16, no. 2, pp. 81–88, 1988.
- [71] K. Sriram, I. Karageorgos, X. Wen, J. Vesely, N. Lindsay, M. Wu, ` L. Khazan, R. P. Pothukuchi, R. Manohar, and A. Bhattacharjee, "Halo: A hardware–software co-designed processor for brain–computer interfaces," *Ieee micro*, vol. 43, no. 3, pp. 64–72, 2023.
- [72] A. P. Su, J. Kuo, K.-J. Lee, J. Huang, G.-A. Jian, C.-A. Chien, J.-I. Guo, and C.-H. Chen, "Multi-core software/hardware co-debug platform with arm coresight™, on-chip test architecture and axi/ahb bus monitor," in *Proceedings of 2011 International Symposium on VLSI Design, Automation and Test*. IEEE, 2011, pp. 1–6.
- [73] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
- [74] R. Uhlig, D. Nagle, T. Mudge, S. Sechrest, and J. Emer, "Instruction fetching: Coping with code bloat," *ACM SIGARCH Computer Architecture News*, vol. 23, no. 2, pp. 345–356, 1995.
- [75] C. Wang, P. Balaji, and M. Snir, "Pilgrim: scalable and (near) lossless mpi tracing," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2021, pp. 1–14.
- [76] C.-J. Wu, A. Jaleel, W. Hasenplaugh, M. Martonosi, S. C. Steely Jr, and J. Emer, "Ship: Signature-based hit predictor for high performance caching," in *Proceedings of the 44th Annual IEEE/ACM International Symposium on Microarchitecture*, 2011, pp. 430–441.
- [77] J. Xu, M. Dong, Q. Tian, Z. Tian, T. Xin, and H. Chen, "Asyncfs: Metadata updates made asynchronous for distributed filesystems with in-network coordination," *arXiv preprint arXiv:2410.08618*, 2024.
- [78] Z. Yan, D. Lustig, D. Nellans, and A. Bhattacharjee, "Translation ranger: Operating system support for contiguity-aware tlbs," in *Proceedings of the 46th International Symposium on Computer Architecture*, 2019, pp. 698–710.
- [79] J. Yang, H. Cui, J. Wu, Y. Tang, and G. Hu, "Making parallel programs reliable with stable multithreading," *Communications of the ACM*, vol. 57, no. 3, pp. 58–69, 2014.

- [80] J. J. Yi, D. J. Lilja, and D. M. Hawkins, "A statistically rigorous approach for improving simulation methodology," in *The Ninth International Symposium on High-Performance Computer Architecture, 2003. HPCA-9 2003. Proceedings.* IEEE, 2003, pp. 281–291.
- [81] R. Zhang, S. Biswas, V. Balaji, M. D. Bond, and B. Lucia, "Peacenik: Architecture support for not failing under fail-stop memory consistency," in *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2020, pp. 317–333.
- [82] K. Zhao, K. Xue, Z. Wang, D. Schatzberg, L. Yang, A. Manousis, J. Weiner, R. Van Riel, B. Sharma, C. Tang *et al.*, "Contiguitas: The pursuit of physical memory contiguity in datacenters," in *Proceedings of the 50th Annual International Symposium on Computer Architecture*, 2023, pp. 1–15.