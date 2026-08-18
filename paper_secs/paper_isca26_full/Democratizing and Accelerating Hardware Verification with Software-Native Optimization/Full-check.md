# Full check

./scripts/A with XS.sh

./scripts/B with XS.sh

#### *F. Evaluation and expected results*

The performance of UCV may be affected by several platform-dependent factors, including last-level cache capacity, memory frequency, and CPU affinity. To account for such variation, our scripts include repeated sampling with mean and variance analysis to show the stability of the measurements.

In all environments we tested, the overall trends and relative orderings are consistent with those reported in Fig. 11 and Fig. 13. Successful reproduction should preserve the relative ordering among compared approaches even if absolute values vary across platforms. In particular, UCV shows a clear advantage over cocotb in the Python interface experiments, while UCV's multi-language wrappers introduce only limited overhead relative to raw Verilator.

For evaluators using the SSH-based environment, the original data and binary artifacts used in this AE are available under \${HOME}/original.

#### REFERENCES

- [1] "Ieee standard for universal verification methodology language reference manual," *IEEE Std 1800.2-2020 (Revision of IEEE Std 1800.2-2017)*, pp. 1–458, 2020.
- [2] "Ieee standard for standard systemc® language reference manual," *IEEE Std 1666-2023 (Revision of IEEE Std 1666-2011)*, pp. 1–618, 2023.
- [3] "Ieee standard for systemverilog–unified hardware design, specification, and verification language," *IEEE Std 1800-2023 (Revision of IEEE Std 1800-2017)*, pp. 1–1354, 2024.
- [4] S. Alyahya, "Crowdsourced software testing: A systematic literature review," *Information and Software Technology*, vol. 127, p. 106363, 2020. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0950584920301312
- [5] S. Anand, E. K. Burke, T. Y. Chen, J. Clark, M. B. Cohen, W. Grieskamp, M. Harman, M. J. Harrold, P. McMinn, A. Bertolino *et al.*, "An orchestrated survey of methodologies for automated software test case generation," *Journal of systems and software*, vol. 86, no. 8, pp. 1978–2001, 2013.
- [6] AnthonyVH, "Mix and match with asyncio · issue #3994 · cocotb/cocotb," 2024, accessed: 2025 Apr 09. [Online]. Available: https://github.com/cocotb/cocotb/discussions/3994
- [7] K. Asanovic, R. Avizienis, J. Bachrach, S. Beamer, D. Biancolin, ´ C. Celio, H. Cook, D. Dabbelt, J. Hauser, A. Izraelevitz, S. Karandikar, B. Keller, D. Kim, J. Koenig, Y. Lee, E. Love, M. Maas, A. Magyar, H. Mao, M. Moreto, A. Ou, D. A. Patterson, B. Richards, C. Schmidt, S. Twigg, H. Vo, and A. Waterman, "The rocket chip generator," Tech. Rep. UCB/EECS-2016-17, Apr 2016. [Online]. Available: http: //www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-17.html
- [8] S. Beamer and D. Donofrio, "Efficiently exploiting low activity factors to accelerate rtl simulation," in *Proceedings of the 57th ACM/EDAC/IEEE Design Automation Conference*, ser. DAC '20. IEEE Press, 2020.
- [9] D. M. Beazley, "Swig: an easy to use tool for integrating scripting languages with c and c++," in *Proceedings of the 4th Conference on USENIX Tcl/Tk Workshop, 1996 - Volume 4*, ser. TCLTK'96. USA: USENIX Association, 1996, p. 15.
- [10] J. Bergeron, *Writing Testbenches using SystemVerilog*, 1st ed. Springer Publishing Company, Incorporated, 2010.
- [11] N. Binkert, B. Beckmann, G. Black, S. K. Reinhardt, A. Saidi, A. Basu, J. Hestness, D. R. Hower, T. Krishna, S. Sardashti, R. Sen, K. Sewell, M. Shoaib, N. Vaish, M. D. Hill, and D. A. Wood, "The gem5 simulator," vol. 39, no. 2, p. 1–7, Aug. 2011. [Online]. Available: https://doi.org/10.1145/2024716.2024718
- [12] L. Chen, D. Zhao, Z. Yu, N. Sun, and Y. Bao, "Gsim: Accelerating rtl simulation for large-scale designs," in *Proceedings of the 62nd Design Automation Conference*, ser. DAC '25, 2025.
- [13] cocotb, "Timing model · cocotb/cocotb wiki," 2023, accessed: 2025 Apr 09. [Online]. Available: https://github.com/cocotb/cocotb/wiki/Timing-Model
- [14] O. Community. Openstack bug tracker. OpenStack Foundation. Accessed: 11 Apr 2025. [Online]. Available: https://bugs.launchpad.net/ openstack
- [15] E. Daka and G. Fraser, "A survey on unit testing practices and problems," in *2014 IEEE 25th International Symposium on Software Reliability Engineering*, 2014, pp. 201–211.
- [16] A. Dobis, T. Petersen, H. J. Damsgaard, K. J. H. Rasmussen, E. Tolotto, S. T. Andersen, R. Lin, and M. Schoeberl, "Chiselverify: An open-source hardware verification library for chisel and scala," in *Proceedings of the 2021 IEEE Nordic Circuits and Systems Conference (NORCAS): NORCHIP and International Symposium on System-on-Chip (SoC)*, 2021, pp. 1–8.
- [17] J. Francesconi, J. A. Rodriguez, and P. M. Julian, "Uvm based testbench architecture for unit verification," in *2014 Argentine Conference on Micro-Nanoelectronics, Technology and Applications (EAMTA)*. IEEE, 2014, pp. 89–94.

- [18] T. Fulcini, R. Coppola, L. Ardito, and M. Torchiano, "A review on tools, mechanics, benefits, and challenges of gamified software testing," *ACM Computing Surveys*, vol. 55, no. 14s, pp. 1–37, 2023.
- [19] D. N. Gadde, S. Kumari, and A. Kumar, "Effective design verification–constrained random with python and cocotb," *arXiv preprint arXiv:2407.10312*, 2024.
- [20] gruvw, "Signals not updated after rising edge · issue #3110 · cocotb/cocotb," 2022, accessed: 2025 Apr 09. [Online]. Available: https://github.com/cocotb/cocotb/issues/3110
- [21] N. B. Harshitha, Y. G. Praveen Kumar, and M. Z. Kurian, "An introduction to universal verification methodology for the digital design of integrated circuits (ic's): A review," in *2021 International Conference on Artificial Intelligence and Smart Systems (ICAIS)*, 2021, pp. 1710–1713.
- [22] B.-Y. Huang, H. Zhang, P. Subramanyan, Y. Vizel, A. Gupta, and S. Malik, "Instruction-level abstraction (ila): A uniform specification for system-on-chip (soc) verification," *ACM Trans. Des. Autom. Electron. Syst.*, vol. 24, no. 1, Dec. 2018. [Online]. Available: https://doi.org/10.1145/3282444
- [23] K. Incki, I. Ari, and H. Sozer, "A survey of software testing in the cloud," ¨ in *2012 IEEE Sixth International Conference on Software Security and Reliability Companion*. IEEE, 2012, pp. 18–23.
- [24] S. Jiang, Y. Ou, P. Pan, and C. Batten, "Umoc: Unified modular ordering constraints to unify cycle- and register-transfer-level modeling," in *2021 58th ACM/IEEE Design Automation Conference (DAC)*, 2021, pp. 883– 888.
- [25] Jonathan Corbet, "Bugs and fixes in the kernel history," https://lwn.net/ Articles/914632/, December 2022, online.
- [26] C. June, "Open-source hardware: a growing movement to democratize ic design," https://ece.engin.umich.edu/stories/open-source-hardware-agrowing-movement-to-democratize-ic-design, 2022, online.
- [27] S. Kalantari, E. Nazemi, and B. Masoumi, "Emergence phenomena in self-organizing systems: a systematic literature review of concepts, researches, and future prospects," *Journal of organizational computing and electronic commerce*, vol. 30, no. 3, pp. 224–265, 2020.
- [28] L. Liu, S. Zhao, B. Li, H. Ren, Z. Xu, M. Wang, X. Li, Y. Han, and Y. Wang, "Make llm inference affordable to everyone: Augmenting gpu memory with ndp-dimm," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, 2025, pp. 1751– 1765.
- [29] D. Lockhart, G. Zibrat, and C. Batten, "Pymtl: A unified framework for vertically integrated computer architecture research," in *Proceedings of the 47th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO-47. USA: IEEE Computer Society, 2014, p. 280–292. [Online]. Available: https://doi.org/10.1109/MICRO.2014.50
- [30] G. Lopez-Parad ´ ´ıs, A. Armejach, and M. Moreto, "gem5 + rtl: A ´ framework to enable rtl models inside a full-system simulator," ser. ICPP '21. New York, NY, USA: Association for Computing Machinery, 2021. [Online]. Available: https://doi.org/10.1145/3472456.3472461
- [31] A. Mahmoud, R. Venkatagiri, K. Ahmed, S. Misailovic, D. Marinov, C. W. Fletcher, and S. V. Adve, "Minotaur: Adapting software testing techniques for hardware errors," in *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '19. New York, NY, USA: Association for Computing Machinery, 2019, p. 1087–1103. [Online]. Available: https://doi.org/10.1145/3297858.3304050
- [32] Mark Horowitz, "Life post moore's law: The new cad frontiers," https: //aha.stanford.edu/life-post-moores-law-new-cad-frontier, 2023, online.
- [33] V. Melikyan, S. Harutyunyan, A. Kirakosyan, and T. Kaplanyan, "Uvm verification ip for axi," in *2021 IEEE East-West Design & Test Symposium (EWDTS)*, 2021, pp. 1–4.
- [34] K. Namjoshi, A. Zeller, and A. Ziv, Eds., *Hardware and Software: Verification and Testing*, ser. Lecture Notes in Computer Science. Springer Berlin Heidelberg, 2011, vol. 6405.
- [35] K. Namjoshi, A. Zeller, and A. Ziv, "Chiseltest: The batteriesincluded testing and formal verification library for chisel-based rtl designs," 2024, accessed: 2024 Oct 13. [Online]. Available: https://github.com/ucb-bar/chiseltest
- [36] Y. Qin, Y. Wang, Z. Zhao, X. Yang, Y. Zhou, S. Wei, Y. Hu, and S. Yin, "Mecla: Memory-compute-efficient llm accelerator with scaling sub-matrix partition," in *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, 2024, pp. 1032–1047.

- [37] B. J. Rosser, "Cocotb: a python-based digital logic verification framework," in *Micro-electronics Section seminar. CERN, Geneva, Switzerland*, 2018.
- [38] K. Selvakkani and K. Venkatesan, "High speed uvm based verification ip for gigabit ethernet protocol," *Intenational Journal of Engineering Research & Technology (IJERT)*, vol. 2, no. 12, pp. 2278–0181, 2013.
- [39] K. Shi, S. Xu, Y. Diao, D. Boland, and Y. Bao, "Encore: Efficient architecture verification framework with fpga acceleration," in *Proceedings of the 2023 ACM/SIGDA International Symposium on Field Programmable Gate Arrays*, ser. FPGA '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 209–219. [Online]. Available: https://doi.org/10.1145/3543622.3573187
- [40] Siemens EDA, "2024 wilson research groupic/asic functional verificationtrend report," https://resources.sw.siemens.com/en-US/whitepaper-2024-wilson-research-group-ic-asic-functional-verificationtrend-report/, 2025, online.
- [41] H. Skinner, R. Trapani Possignolo, S.-H. Wang, and J. Renau, "Livesim: A fast hot reload simulator for hdls," in *2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2020, pp. 126–135.
- [42] W. Snyder, "Verilator 4.0: open simulation goes multithreaded," in *Open Source Digital Design Conference (ORConf)*, 2018.
- [43] A. S. I. V. T. Sub-Committee, "Verification intellectual property (vip) recommended practices, version 1.0," Technical Report, 2009, standardized practices for VIP interoperability across methodologies. [Online]. Available: https://www.accellera.org/images/ downloads/standards/uvm/VIP 1.0.pdf
- [44] L. Truong, S. Herbst, R. Setaluri, M. Mann, R. Daly, K. Zhang, C. Donovick, D. Stanley, M. Horowitz, C. Barrett, and P. Hanrahan, "fault: A python embedded domain-specific language for metaprogramming portable hardware verification components," in *Computer Aided Verification: 32nd International Conference, CAV 2020, Los Angeles, CA, USA, July 21–24, 2020, Proceedings, Part I*. Berlin, Heidelberg: Springer-Verlag, 2020, p. 403–414. [Online]. Available: https://doi.org/10.1007/978-3-030-53288-8 19
- [45] S. Vagaggini, D. Davalle, P. Nannipieri, and L. Fanucci, "Integration of twin models in uvm verification ips for space telecommunication systems," *IEEE Access*, 2024.
- [46] H. Wang and S. Beamer, "Repcut: Superlinear parallel rtl simulation with replication-aided partitioning," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, ser. ASPLOS 2023. New York, NY, USA: Association for Computing Machinery, 2023, p. 572–585. [Online]. Available: https://doi.org/10.1145/3582016.3582034
- [47] J. Wang, Y. Huang, C. Chen, Z. Liu, S. Wang, and Q. Wang, "Software testing with large language models: Survey, landscape, and vision," *IEEE Transactions on Software Engineering*, 2024.
- [48] K. Wang, J. Chen, Y. Xu, Z. Yu, Z. Zhang, G. Chen, X. Hu, L. Zhang, X. Chen, W. He, D. Tang, N. Sun, and Y. Bao, " XiangShan: An Open-Source Project for High-Performance RISC-V Processors Meeting Industrial-Grade Standards ," in *2024 IEEE Hot Chips 36 Symposium (HCS)*. Los Alamitos, CA, USA: IEEE Computer Society, Aug. 2024, pp. 1–25. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/HCS61935.2024.10665293
- [49] S.-H. Wang, R. T. Possignolo, H. B. Skinner, and J. Renau, "Livehd: A productive live hardware development flow," *IEEE Micro*, vol. 40, no. 4, pp. 67–75, 2020.
- [50] Y. Xu, S. Wang, D. Tang, N. Sun, and Y. Bao, "Pathfuzz: Broadening fuzzing horizons with footprint memory for cpus," in *In 61st ACM/IEEE Design Automation Conference (DAC '24)*. ACM, June 2024. [Online]. Available: https://doi.org/10.1145/3649329.3655911
- [51] J. Yun, F. Rustamov, J. Kim, and Y. Shin, "Fuzzing of embedded systems: A survey," *ACM Comput. Surv.*, vol. 55, no. 7, Dec. 2022. [Online]. Available: https://doi.org/10.1145/3538644
- [52] Z. Zhang, W. Weng, Y. Li, L. Cai, H. Wang, D. Boland, Y. Bao, and K. Shi, "Hassert: Hardware assertion-based verification framework with fpga acceleration," in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 4*, ser. ASPLOS '24. New York, NY, USA: Association for Computing Machinery, 2025, p. 142–154. [Online]. Available: https://doi.org/10.1145/3622781.3698899
- [53] K. Zhou, Y. Liang, Y. Lin, R. Wang, and R. Huang, "Khronos: Fusing memory access for improved hardware rtl simulation," in *2023 56th*

- *IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2023, pp. 180–193.
- [54] X. Zhu, S. Wen, S. Camtepe, and Y. Xiang, "Fuzzing: A survey for roadmap," vol. 54, no. 11s, Sep. 2022. [Online]. Available: https://doi.org/10.1145/3512345# Full check

./scripts/A with XS.sh

./scripts/B with XS.sh

#### *F. Evaluation and expected results*

The performance of UCV may be affected by several platform-dependent factors, including last-level cache capacity, memory frequency, and CPU affinity. To account for such variation, our scripts include repeated sampling with mean and variance analysis to show the stability of the measurements.

In all environments we tested, the overall trends and relative orderings are consistent with those reported in Fig. 11 and Fig. 13. Successful reproduction should preserve the relative ordering among compared approaches even if absolute values vary across platforms. In particular, UCV shows a clear advantage over cocotb in the Python interface experiments, while UCV's multi-language wrappers introduce only limited overhead relative to raw Verilator.

For evaluators using the SSH-based environment, the original data and binary artifacts used in this AE are available under \${HOME}/original.

#### REFERENCES

- [1] "Ieee standard for universal verification methodology language reference manual," *IEEE Std 1800.2-2020 (Revision of IEEE Std 1800.2-2017)*, pp. 1–458, 2020.
- [2] "Ieee standard for standard systemc® language reference manual," *IEEE Std 1666-2023 (Revision of IEEE Std 1666-2011)*, pp. 1–618, 2023.
- [3] "Ieee standard for systemverilog–unified hardware design, specification, and verification language," *IEEE Std 1800-2023 (Revision of IEEE Std 1800-2017)*, pp. 1–1354, 2024.
- [4] S. Alyahya, "Crowdsourced software testing: A systematic literature review," *Information and Software Technology*, vol. 127, p. 106363, 2020. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0950584920301312
- [5] S. Anand, E. K. Burke, T. Y. Chen, J. Clark, M. B. Cohen, W. Grieskamp, M. Harman, M. J. Harrold, P. McMinn, A. Bertolino *et al.*, "An orchestrated survey of methodologies for automated software test case generation," *Journal of systems and software*, vol. 86, no. 8, pp. 1978–2001, 2013.
- [6] AnthonyVH, "Mix and match with asyncio · issue #3994 · cocotb/cocotb," 2024, accessed: 2025 Apr 09. [Online]. Available: https://github.com/cocotb/cocotb/discussions/3994
- [7] K. Asanovic, R. Avizienis, J. Bachrach, S. Beamer, D. Biancolin, ´ C. Celio, H. Cook, D. Dabbelt, J. Hauser, A. Izraelevitz, S. Karandikar, B. Keller, D. Kim, J. Koenig, Y. Lee, E. Love, M. Maas, A. Magyar, H. Mao, M. Moreto, A. Ou, D. A. Patterson, B. Richards, C. Schmidt, S. Twigg, H. Vo, and A. Waterman, "The rocket chip generator," Tech. Rep. UCB/EECS-2016-17, Apr 2016. [Online]. Available: http: //www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-17.html
- [8] S. Beamer and D. Donofrio, "Efficiently exploiting low activity factors to accelerate rtl simulation," in *Proceedings of the 57th ACM/EDAC/IEEE Design Automation Conference*, ser. DAC '20. IEEE Press, 2020.
- [9] D. M. Beazley, "Swig: an easy to use tool for integrating scripting languages with c and c++," in *Proceedings of the 4th Conference on USENIX Tcl/Tk Workshop, 1996 - Volume 4*, ser. TCLTK'96. USA: USENIX Association, 1996, p. 15.
- [10] J. Bergeron, *Writing Testbenches using SystemVerilog*, 1st ed. Springer Publishing Company, Incorporated, 2010.
- [11] N. Binkert, B. Beckmann, G. Black, S. K. Reinhardt, A. Saidi, A. Basu, J. Hestness, D. R. Hower, T. Krishna, S. Sardashti, R. Sen, K. Sewell, M. Shoaib, N. Vaish, M. D. Hill, and D. A. Wood, "The gem5 simulator," vol. 39, no. 2, p. 1–7, Aug. 2011. [Online]. Available: https://doi.org/10.1145/2024716.2024718
- [12] L. Chen, D. Zhao, Z. Yu, N. Sun, and Y. Bao, "Gsim: Accelerating rtl simulation for large-scale designs," in *Proceedings of the 62nd Design Automation Conference*, ser. DAC '25, 2025.
- [13] cocotb, "Timing model · cocotb/cocotb wiki," 2023, accessed: 2025 Apr 09. [Online]. Available: https://github.com/cocotb/cocotb/wiki/Timing-Model
- [14] O. Community. Openstack bug tracker. OpenStack Foundation. Accessed: 11 Apr 2025. [Online]. Available: https://bugs.launchpad.net/ openstack
- [15] E. Daka and G. Fraser, "A survey on unit testing practices and problems," in *2014 IEEE 25th International Symposium on Software Reliability Engineering*, 2014, pp. 201–211.
- [16] A. Dobis, T. Petersen, H. J. Damsgaard, K. J. H. Rasmussen, E. Tolotto, S. T. Andersen, R. Lin, and M. Schoeberl, "Chiselverify: An open-source hardware verification library for chisel and scala," in *Proceedings of the 2021 IEEE Nordic Circuits and Systems Conference (NORCAS): NORCHIP and International Symposium on System-on-Chip (SoC)*, 2021, pp. 1–8.
- [17] J. Francesconi, J. A. Rodriguez, and P. M. Julian, "Uvm based testbench architecture for unit verification," in *2014 Argentine Conference on Micro-Nanoelectronics, Technology and Applications (EAMTA)*. IEEE, 2014, pp. 89–94.

- [18] T. Fulcini, R. Coppola, L. Ardito, and M. Torchiano, "A review on tools, mechanics, benefits, and challenges of gamified software testing," *ACM Computing Surveys*, vol. 55, no. 14s, pp. 1–37, 2023.
- [19] D. N. Gadde, S. Kumari, and A. Kumar, "Effective design verification–constrained random with python and cocotb," *arXiv preprint arXiv:2407.10312*, 2024.
- [20] gruvw, "Signals not updated after rising edge · issue #3110 · cocotb/cocotb," 2022, accessed: 2025 Apr 09. [Online]. Available: https://github.com/cocotb/cocotb/issues/3110
- [21] N. B. Harshitha, Y. G. Praveen Kumar, and M. Z. Kurian, "An introduction to universal verification methodology for the digital design of integrated circuits (ic's): A review," in *2021 International Conference on Artificial Intelligence and Smart Systems (ICAIS)*, 2021, pp. 1710–1713.
- [22] B.-Y. Huang, H. Zhang, P. Subramanyan, Y. Vizel, A. Gupta, and S. Malik, "Instruction-level abstraction (ila): A uniform specification for system-on-chip (soc) verification," *ACM Trans. Des. Autom. Electron. Syst.*, vol. 24, no. 1, Dec. 2018. [Online]. Available: https://doi.org/10.1145/3282444
- [23] K. Incki, I. Ari, and H. Sozer, "A survey of software testing in the cloud," ¨ in *2012 IEEE Sixth International Conference on Software Security and Reliability Companion*. IEEE, 2012, pp. 18–23.
- [24] S. Jiang, Y. Ou, P. Pan, and C. Batten, "Umoc: Unified modular ordering constraints to unify cycle- and register-transfer-level modeling," in *2021 58th ACM/IEEE Design Automation Conference (DAC)*, 2021, pp. 883– 888.
- [25] Jonathan Corbet, "Bugs and fixes in the kernel history," https://lwn.net/ Articles/914632/, December 2022, online.
- [26] C. June, "Open-source hardware: a growing movement to democratize ic design," https://ece.engin.umich.edu/stories/open-source-hardware-agrowing-movement-to-democratize-ic-design, 2022, online.
- [27] S. Kalantari, E. Nazemi, and B. Masoumi, "Emergence phenomena in self-organizing systems: a systematic literature review of concepts, researches, and future prospects," *Journal of organizational computing and electronic commerce*, vol. 30, no. 3, pp. 224–265, 2020.
- [28] L. Liu, S. Zhao, B. Li, H. Ren, Z. Xu, M. Wang, X. Li, Y. Han, and Y. Wang, "Make llm inference affordable to everyone: Augmenting gpu memory with ndp-dimm," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, 2025, pp. 1751– 1765.
- [29] D. Lockhart, G. Zibrat, and C. Batten, "Pymtl: A unified framework for vertically integrated computer architecture research," in *Proceedings of the 47th Annual IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO-47. USA: IEEE Computer Society, 2014, p. 280–292. [Online]. Available: https://doi.org/10.1109/MICRO.2014.50
- [30] G. Lopez-Parad ´ ´ıs, A. Armejach, and M. Moreto, "gem5 + rtl: A ´ framework to enable rtl models inside a full-system simulator," ser. ICPP '21. New York, NY, USA: Association for Computing Machinery, 2021. [Online]. Available: https://doi.org/10.1145/3472456.3472461
- [31] A. Mahmoud, R. Venkatagiri, K. Ahmed, S. Misailovic, D. Marinov, C. W. Fletcher, and S. V. Adve, "Minotaur: Adapting software testing techniques for hardware errors," in *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '19. New York, NY, USA: Association for Computing Machinery, 2019, p. 1087–1103. [Online]. Available: https://doi.org/10.1145/3297858.3304050
- [32] Mark Horowitz, "Life post moore's law: The new cad frontiers," https: //aha.stanford.edu/life-post-moores-law-new-cad-frontier, 2023, online.
- [33] V. Melikyan, S. Harutyunyan, A. Kirakosyan, and T. Kaplanyan, "Uvm verification ip for axi," in *2021 IEEE East-West Design & Test Symposium (EWDTS)*, 2021, pp. 1–4.
- [34] K. Namjoshi, A. Zeller, and A. Ziv, Eds., *Hardware and Software: Verification and Testing*, ser. Lecture Notes in Computer Science. Springer Berlin Heidelberg, 2011, vol. 6405.
- [35] K. Namjoshi, A. Zeller, and A. Ziv, "Chiseltest: The batteriesincluded testing and formal verification library for chisel-based rtl designs," 2024, accessed: 2024 Oct 13. [Online]. Available: https://github.com/ucb-bar/chiseltest
- [36] Y. Qin, Y. Wang, Z. Zhao, X. Yang, Y. Zhou, S. Wei, Y. Hu, and S. Yin, "Mecla: Memory-compute-efficient llm accelerator with scaling sub-matrix partition," in *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, 2024, pp. 1032–1047.

- [37] B. J. Rosser, "Cocotb: a python-based digital logic verification framework," in *Micro-electronics Section seminar. CERN, Geneva, Switzerland*, 2018.
- [38] K. Selvakkani and K. Venkatesan, "High speed uvm based verification ip for gigabit ethernet protocol," *Intenational Journal of Engineering Research & Technology (IJERT)*, vol. 2, no. 12, pp. 2278–0181, 2013.
- [39] K. Shi, S. Xu, Y. Diao, D. Boland, and Y. Bao, "Encore: Efficient architecture verification framework with fpga acceleration," in *Proceedings of the 2023 ACM/SIGDA International Symposium on Field Programmable Gate Arrays*, ser. FPGA '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 209–219. [Online]. Available: https://doi.org/10.1145/3543622.3573187
- [40] Siemens EDA, "2024 wilson research groupic/asic functional verificationtrend report," https://resources.sw.siemens.com/en-US/whitepaper-2024-wilson-research-group-ic-asic-functional-verificationtrend-report/, 2025, online.
- [41] H. Skinner, R. Trapani Possignolo, S.-H. Wang, and J. Renau, "Livesim: A fast hot reload simulator for hdls," in *2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2020, pp. 126–135.
- [42] W. Snyder, "Verilator 4.0: open simulation goes multithreaded," in *Open Source Digital Design Conference (ORConf)*, 2018.
- [43] A. S. I. V. T. Sub-Committee, "Verification intellectual property (vip) recommended practices, version 1.0," Technical Report, 2009, standardized practices for VIP interoperability across methodologies. [Online]. Available: https://www.accellera.org/images/ downloads/standards/uvm/VIP 1.0.pdf
- [44] L. Truong, S. Herbst, R. Setaluri, M. Mann, R. Daly, K. Zhang, C. Donovick, D. Stanley, M. Horowitz, C. Barrett, and P. Hanrahan, "fault: A python embedded domain-specific language for metaprogramming portable hardware verification components," in *Computer Aided Verification: 32nd International Conference, CAV 2020, Los Angeles, CA, USA, July 21–24, 2020, Proceedings, Part I*. Berlin, Heidelberg: Springer-Verlag, 2020, p. 403–414. [Online]. Available: https://doi.org/10.1007/978-3-030-53288-8 19
- [45] S. Vagaggini, D. Davalle, P. Nannipieri, and L. Fanucci, "Integration of twin models in uvm verification ips for space telecommunication systems," *IEEE Access*, 2024.
- [46] H. Wang and S. Beamer, "Repcut: Superlinear parallel rtl simulation with replication-aided partitioning," in *Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, ser. ASPLOS 2023. New York, NY, USA: Association for Computing Machinery, 2023, p. 572–585. [Online]. Available: https://doi.org/10.1145/3582016.3582034
- [47] J. Wang, Y. Huang, C. Chen, Z. Liu, S. Wang, and Q. Wang, "Software testing with large language models: Survey, landscape, and vision," *IEEE Transactions on Software Engineering*, 2024.
- [48] K. Wang, J. Chen, Y. Xu, Z. Yu, Z. Zhang, G. Chen, X. Hu, L. Zhang, X. Chen, W. He, D. Tang, N. Sun, and Y. Bao, " XiangShan: An Open-Source Project for High-Performance RISC-V Processors Meeting Industrial-Grade Standards ," in *2024 IEEE Hot Chips 36 Symposium (HCS)*. Los Alamitos, CA, USA: IEEE Computer Society, Aug. 2024, pp. 1–25. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/HCS61935.2024.10665293
- [49] S.-H. Wang, R. T. Possignolo, H. B. Skinner, and J. Renau, "Livehd: A productive live hardware development flow," *IEEE Micro*, vol. 40, no. 4, pp. 67–75, 2020.
- [50] Y. Xu, S. Wang, D. Tang, N. Sun, and Y. Bao, "Pathfuzz: Broadening fuzzing horizons with footprint memory for cpus," in *In 61st ACM/IEEE Design Automation Conference (DAC '24)*. ACM, June 2024. [Online]. Available: https://doi.org/10.1145/3649329.3655911
- [51] J. Yun, F. Rustamov, J. Kim, and Y. Shin, "Fuzzing of embedded systems: A survey," *ACM Comput. Surv.*, vol. 55, no. 7, Dec. 2022. [Online]. Available: https://doi.org/10.1145/3538644
- [52] Z. Zhang, W. Weng, Y. Li, L. Cai, H. Wang, D. Boland, Y. Bao, and K. Shi, "Hassert: Hardware assertion-based verification framework with fpga acceleration," in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 4*, ser. ASPLOS '24. New York, NY, USA: Association for Computing Machinery, 2025, p. 142–154. [Online]. Available: https://doi.org/10.1145/3622781.3698899
- [53] K. Zhou, Y. Liang, Y. Lin, R. Wang, and R. Huang, "Khronos: Fusing memory access for improved hardware rtl simulation," in *2023 56th*

- *IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2023, pp. 180–193.
- [54] X. Zhu, S. Wen, S. Camtepe, and Y. Xiang, "Fuzzing: A survey for roadmap," vol. 54, no. 11s, Sep. 2022. [Online]. Available: https://doi.org/10.1145/3512345