# *C. Description*

*1) How to access:* The artifact can be accessed here https://doi.org/10.5281/zenodo.19901476 . It can also be accessed through the public GitHub repository by cloning the repository https://github.com/bearhw/LIPPEN. The README file in the repository contains the full instructions for building and running the artifact.

- *2) Hardware dependencies:* The artifact supports multiple workflows with different hardware requirements. For simulation and software compilation, a standard Linux server or workstation is sufficient. For FPGA-based evaluation, the artifact requires a Xilinx VCU118 FPGA board. For the ARM64 microbenchmark experiments, the artifact requires an Apple M1 system, and it should also work on other Apple M-series processors.
- *3) Software dependencies:* The artifact requires a Linux environment with the dependencies needed to build the modified Chipyard design, the LLVM-based compiler toolchain, and the provided benchmarks. Running the FPGA workflow additionally requires the tools needed to generate the bitstream, build the Linux image, prepare the SD card, and execute binaries on the prototype. For the ARM64 microbenchmark workflow, the artifact requires macOS with the appropriate build tools installed. Because Apple restricts the use of PAC instructions in user-space programs, System Integrity Protection must be disabled before running those experiments.

#### *D. Installation*

Clone the GitHub repository. Then follow the instructions in the README file to set up the build environment, compile the required components, and prepare the selected workflow. The README describes separate steps for simulation, FPGAbased evaluation, and ARM64 microbenchmark experiments.

## *E. Experiment workflow*

- 1) Clone the repository and initialize the required submodules and dependencies.
- 2) Build the modified Chipyard hardware design and the LLVM-based compiler toolchain by following the provided scripts.
- 3) Run the Verilator-based simulation flow to verify the design and execute the included small test programs.
- 4) If FPGA evaluation is desired, generate the FPGA bitstream and prepare the Linux image for the VCU118 platform.
- 5) Load the required binaries and files onto the SD card and boot the system on the FPGA prototype.
- 6) Execute the provided microbenchmarks and collect the performance results.
- 7) For the ARM64 workflow, compile and run the microbenchmarks on an Apple M1 or another Apple Mseries processor after disabling System Integrity Protection.

#### *F. Evaluation and expected results*

Successful execution of the artifact is demonstrated by:

- 1) Correctly building the modified Chipyard design and LLVM-based toolchain.
- 2) Running the provided test programs in the Verilator simulation.
- 3) Reproducing the experimental results from Figure 4 of the paper.

The FPGA-based experiments require access to a Xilinx VCU118 board and involve significant synthesis and setup time. Similarly, the ARM64 microbenchmark experiments require Apple M-series hardware and additional system configuration (e.g., disabling System Integrity Protection).

#### REFERENCES

- [1] "arm64: ptrauth: add pointer authentication Armv8.6 enhanced feature," http://www.spinics.net/lists/arm-kernel/msg814954.html. Accessed nov 2025.
- [2] M. Abadi, M. Budiu, U. Erlingsson, and J. Ligatti, "Control-flow integrity principles, implementations, and applications," *ACM Transactions on Information and System Security (TISSEC)*, vol. 13, no. 1, pp. 1–40, 2009.
- [3] F. Abed, E. List, and S. Lucks, "On the security of the core of PRINCE against biclique and differential cryptanalysis," Cryptology ePrint Archive, Paper 2012/712, 2012. [Online]. Available: https://eprint.iacr.org/2012/712
- [4] A. Amid, D. Biancolin, A. Gonzalez, D. Grubb, S. Karandikar, H. Liew, A. Magyar, H. Mao, A. Ou, N. Pemberton *et al.*, "Chipyard: Integrated design, simulation, and implementation framework for custom socs," *Ieee Micro*, vol. 40, no. 4, pp. 10–21, 2020.
- [5] Apple, "Preparing your app to work with pointer authentication," https://developer.apple.com/documentation/security/preparingyour-app-to-work-with-pointer-authentication. Accessed nov 2025.
- [6] ——, "Operating system integrity," 2021, https://support.apple.com/enhk/guide/security/sec8b776536b/1/web.
- [7] Apple Inc. (2023) Pointer authentication in dyld. Accessed: 2025-02-05. [Online]. Available: https://deepwiki.com/apple-oss-distributions/dyld/ 6.2-pointer-authentication
- [8] ARM, "Rop," ARM, Tech. Rep., 2018, available at: https://developer. arm.com/documentation/102433/0200/Return-oriented-programming.
- [9] Arm Limited, *Arm Architecture Reference Manual for A-profile architecture*, rev. k.a ed., Arm Limited, 2023, document ARM DDI 0487. [Online]. Available: https://developer.arm.com/documentation/ ddi0487/latest/
- [10] *Architecture Extensions before 2020: Speculative behavior of pointer authentication instructions (FEAT FPACC SPEC)*, Arm Limited, 2024, version 1.2. [Online]. Available: https://developer.arm.com/ documentation/110389/1-2/
- [11] Arm Ltd., "Arm architecture reference manual supplement: Memory tagging extension," https://developer.arm.com/-/media/Arm% 20Developer%20Community/PDF/Arm Memory Tagging Extension Whitepaper.pdf, 2019, accessed: 2025-11-03.
- [12] ——, "Hardware-assisted checking using silicon secured memory (ssm)," https://docs.oracle.com/cd/E37069 01/html/E37085/gphwb. html, 2019, accessed: 2025-11-03.
- [13] *Arm® Architecture Reference Manual Armv8, for Armv8-A architecture profile*, Version i.a ed., Arm Ltd., 2024, [Online; accessed Nov. 16, 2025]. [Online]. Available: https://developer.arm.com/documentation/ 110389/latest/
- [14] K. Asanovic, R. Avizienis, J. Bachrach, S. Beamer, D. Biancolin, C. Celio, H. Cook, D. Dabbelt, J. Hauser, A. Izraelevitz *et al.*, "The rocket chip generator," *EECS Department, University of California, Berkeley, Tech. Rep. UCB/EECS-2016-17*, vol. 4, pp. 6–2, 2016.
- [15] C. Auth, A. Aliyarukunju, M. Asoro, D. Bergstrom, V. Bhagwat, J. Birdsall, N. Bisnik, M. Buehler, V. Chikarmane, G. Ding, Q. Fu, H. Gomez, W. Han, D. Hanken, M. Haran, M. Hattendorf, R. Heussner, H. Hiramatsu, B. Ho, S. Jaloviar, I. Jin, S. Joshi, S. Kirby, S. Kosaraju, H. Kothari, G. Leatherman, K. Lee, J. Leib, A. Madhavan, K. Marla, H. Meyer, T. Mule, C. Parker, S. Parthasarathy, C. Pelto, L. Pipes, I. Post, M. Prince, A. Rahman, S. Rajamani, A. Saha, J. D. Santos, M. Sharma, V. Sharma, J. Shin, P. Sinha, P. Smith, M. Sprinkle, A. S. Amour, C. Staus, R. Suri, D. Towner, A. Tripathi, A. Tura, C. Ward, and A. Yeoh, "A 10nm high performance and low-power CMOS technology featuring 3rd generation FinFET transistors, Self-Aligned Quad Patterning, contact over active gate and cobalt local interconnects," in *2017 IEEE International Electron Devices Meeting (IEDM)*, 2017, pp. 29.1.1–29.1.4.

- [16] R. Avanzi, "The qarma block cipher family. almost mds matrices over rings with zero divisors, nearly symmetric even-mansour constructions with non-involutory central rounds, and search heuristics for low-latency s-boxes," *IACR Transactions on Symmetric Cryptology*, pp. 4–44, 2017.
- [17] R. Avanzi, S. Banik, O. Dunkelman, M. Eichlseder, S. Ghosh, M. Nageler, and F. Regazzoni, "The QARMAv2 Family of Tweakable Block Ciphers," *IACR Transactions on Symmetric Cryptology*, vol. 2023, no. 3, p. 25–73, Sep. 2023. [Online]. Available: https: //tosc.iacr.org/index.php/ToSC/article/view/11184
- [18] R. Beaulieu, D. Shors, J. Smith, S. Treatman-Clark, B. Weeks, and L. Wingers, "The SIMON and SPECK Lightweight Block Ciphers," in *Proceedings of the 52nd Annual Design Automation Conference*, ser. DAC '15. New York, NY, USA: Association for Computing Machinery, 2015. [Online]. Available: https://doi.org/10.1145/2744769.2747946
- [19] Y. Belkheyar, J. Daemen, C. Dobraunig, S. Ghosh, and S. Rasoolzadeh, "BipBip: A Low-Latency Tweakable Block Cipher with Small Dimensions," *IACR Transactions on Cryptographic Hardware and Embedded Systems*, vol. 2023, no. 1, p. 326–368, Nov. 2022. [Online]. Available: https://tches.iacr.org/index.php/TCHES/article/view/9955
- [20] B. Bierbaumer, J. Kirsch, T. Kittel, A. Francillon, and A. Zarras, "Smashing the stack protector for fun and profit," in *IFIP International Conference on ICT Systems Security and Privacy Protection*. Springer, 2018, pp. 293–306.
- [21] T. Bletsch, X. Jiang, V. W. Freeh, and Z. Liang, "Jump-oriented programming: a new class of code-reuse attack," in *Proceedings of the 6th ACM symposium on information, computer and communications security*, 2011, pp. 30–40.
- [22] A. Bogdanov, L. R. Knudsen, G. Leander, C. Paar, A. Poschmann, M. J. B. Robshaw, Y. Seurin, and C. Vikkelsoe, "PRESENT: An Ultra-Lightweight Block Cipher," in *Cryptographic Hardware and Embedded Systems - CHES 2007*, P. Paillier and I. Verbauwhede, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007, pp. 450–466.
- [23] J. Borghoff, A. Canteaut, T. Guneysu, E. B. Kavun, M. Knezevic, ¨ L. R. Knudsen, G. Leander, V. Nikov, C. Paar, C. Rechberger, P. Rombouts, S. S. Thomsen, and T. Yalc¸ın, "Prince: a low-latency block cipher for pervasive computing applications," in *Proceedings of the 18th International Conference on The Theory and Application of Cryptology and Information Security*, ser. ASIACRYPT'12. Berlin, Heidelberg: Springer-Verlag, 2012, p. 208–225. [Online]. Available: https://doi.org/10.1007/978-3-642-34961-4 14
- [24] D. Bozilov, M. Eichlseder, M. Kne ˇ zevi ˇ c, B. Lambin, G. Leander, ´ T. Moos, V. Nikov, S. Rasoolzadeh, Y. Todo, and F. Wiemer, "Princev2: More security for (almost) no overhead," in *Selected Areas in Cryptography: 27th International Conference, Halifax, NS, Canada (Virtual Event), October 21-23, 2020, Revised Selected Papers*. Berlin, Heidelberg: Springer-Verlag, 2020, p. 483–511. [Online]. Available: https://doi.org/10.1007/978-3-030-81652-0 19
- [25] D. Brash, "Armv8-a architecture 2016 additions," ARM Community Blog, 2016, https://community.arm.com/arm-communityblogs/b/architectures-and-processors-blog/posts/armv8-a-architecture-2016-additions.
- [26] N. Burow, D. McKee, S. A. Carr, and M. Payer, "Cup: Comprehensive user-space protection for c/c++," in *Proceedings of the 2018 on Asia Conference on Computer and Communications Security*, 2018, pp. 381– 392.
- [27] Z. Cai, J. Zhu, W. Shen, Y. Yang, R. Chang, Y. Wang, J. Li, and K. Ren, "Demystifying pointer authentication on apple m1," in *32nd USENIX Security Symposium (USENIX Security 23)*, 2023, pp. 2833–2848.
- [28] C. Canniere, O. Dunkelman, and M. Kne ` zevi ˇ c, "KATAN and ´ KTANTAN – A Family of Small and Efficient Hardware-Oriented Block Ciphers," in *Proceedings of the 11th International Workshop on Cryptographic Hardware and Embedded Systems*, ser. CHES '09. Berlin, Heidelberg: Springer-Verlag, 2009, p. 272–288. [Online]. Available: https://doi.org/10.1007/978-3-642-04138-9 20
- [29] N. Carlini, A. Barresi, M. Payer, D. Wagner, and T. R. Gross, "{Control-Flow} bending: On the effectiveness of {Control-Flow} integrity," in *24th USENIX Security Symposium (USENIX Security 15)*, 2015, pp. 161–176.
- [30] N. Carlini and D. Wagner, "{ROP} is still dangerous: Breaking modern defenses," in *23rd USENIX Security Symposium (USENIX Security 14)*, 2014, pp. 385–399.
- [31] M. S. R. Center. (2019) A proactive approach to more secure code. Reports 70% of Microsoft CVEs are memory-safety issues. ˜

- [Online]. Available: https://www.microsoft.com/en-us/msrc/blog/2019/ 07/a-proactive-approach-to-more-secure-code
- [32] S. Chen, J. Xu, E. C. Sezer, P. Gauriar, and R. K. Iyer, "Non-controldata attacks are realistic threats." in *USENIX security symposium*, vol. 5, 2005, p. 146.
- [33] C. Cowan, S. Beattie, J. Johansen, and P. Wagle, "{PointGuard™}: Protecting pointers from buffer overflow vulnerabilities," in *12th USENIX Security Symposium (USENIX Security 03)*, 2003.
- [34] L. Davi, A.-R. Sadeghi, D. Lehmann, and F. Monrose, "Stitching the gadgets: On the ineffectiveness of Coarse-Grained Control-Flow integrity protection," in *23rd USENIX Security Symposium (USENIX Security 14)*. San Diego, CA: USENIX Association, Aug. 2014, pp. 401–416. [Online]. Available: https://www.usenix.org/conference/ usenixsecurity14/technical-sessions/presentation/davi
- [35] P. Derbez and L. Perrin, "Meet-in-the-Middle Attacks and Structural Analysis of Round-Reduced PRINCE," *J. Cryptol.*, vol. 33, no. 3, p. 1184–1215, 2020.
- [36] J. Devietti, C. Blundell, M. M. Martin, and S. Zdancewic, "Hardbound: Architectural support for spatial safety of the c programming language," *ACM SIGOPS Operating Systems Review*, vol. 42, no. 2, pp. 103–114, 2008.
- [37] U. Dhawan, C. Hritcu, R. Rubin, N. Vasilakis, S. Chiricescu, J. M. Smith, T. F. Knight Jr, B. C. Pierce, and A. DeHon, "Architectural support for software-defined metadata processing," in *Proceedings of the Twentieth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2015, pp. 487–502.
- [38] C. Dobraunig, M. Eichlseder, F. Mendel, and M. Schlaffer, "Ascon v1.2: ¨ Lightweight Authenticated Encryption and Hashing," *J. Cryptol.*, vol. 34, no. 3, Jul. 2021. [Online]. Available: https://doi.org/10.1007/s00145- 021-09398-9
- [39] I. Evans, F. Long, U. Otgonbaatar, H. Shrobe, M. Rinard, H. Okhravi, and S. Sidiroglou-Douskos, "Control jujutsu: On the weaknesses of finegrained control flow integrity," in *Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security*, 2015, pp. 901– 913.
- [40] R. M. Farkhani, M. Ahmadi, and L. Lu, "{PTAuth}: temporal memory safety via robust points-to authentication," in *30th USENIX Security Symposium (USENIX Security 21)*, 2021, pp. 1037–1054.
- [41] M. Gallagher, L. Biernacki, S. Chen, Z. B. Aweke, S. F. Yitbarek, M. T. Aga, A. Harris, Z. Xu, B. Kasikci, V. Bertacco *et al.*, "Morpheus: A vulnerability-tolerant secure architecture based on ensembles of moving target defenses with churn," in *Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems*, 2019, pp. 469–484.
- [42] E. Goktas, K. Razavi, G. Portokalidis, H. Bos, and C. Giuffrida, "Spec- ¨ ulative probing: Hacking blind in the spectre era," in *Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security*, 2020, pp. 1871–1885.
- [43] R. T. Gollapudi, G. Yuksek, D. Demicco, M. Cole, G. Kothari, R. Kulkarni, X. Zhang, K. Ghose, A. Prakash, and Z. Umrigar, "Control flow and pointer integrity enforcement in a secure tagged architecture," in *2023 IEEE Symposium on Security and Privacy (SP)*. IEEE, 2023, pp. 2974– 2989.
- [44] Z. Gong, S. Nikova, and Y. W. Law, "KLEIN: A New Family of Lightweight Block Ciphers," in *RFID. Security and Privacy*, A. Juels and C. Paar, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012, pp. 1–18.
- [45] J. Guo, T. Peyrin, A. Poschmann, and M. Robshaw, "The LED Block Cipher," in *Cryptographic Hardware and Embedded Systems – CHES 2011*, B. Preneel and T. Takagi, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011, pp. 326–341.
- [46] H. Hu, S. Shinde, S. Adrian, Z. L. Chua, P. Saxena, and Z. Liang, "Data-oriented programming: On the expressiveness of non-control data attacks," in *2016 IEEE Symposium on Security and Privacy (SP)*. IEEE, 2016, pp. 969–986.
- [47] Institute of Applied Information Processing and Communications (IAIK), TU Graz, "memsec: Hardware security primitives for memory protection," https://github.com/isec-tugraz/memsec/tree/develop/hdl/ crypto, 2025, contains VHDL implementations qarma.vhd and prince.vhd accessed on November 17, 2025.
- [48] M. Ismail, C. Jelesnianski, Y. Jang, C. Min, and W. Xiong, "Enforcing c/c++ type and scope at runtime for control-flow and data-flow integrity," in *Proceedings of the 29th ACM International Conference on Archi-*

- *tectural Support for Programming Languages and Operating Systems, Volume 3*, 2024, pp. 283–300.
- [49] M. Ismail, A. Quach, C. Jelesnianski, Y. Jang, and C. Min, "Tightly seal your sensitive pointers with {PACTight}," in *31st USENIX Security Symposium (USENIX Security 22)*, 2022, pp. 3717–3734.
- [50] D. Jang, Z. Tatlock, and S. Lerner, "Safedispatch: Securing c++ virtual calls from memory corruption attacks." in *NDSS*, 2014.
- [51] J. Jean, I. Nikolic, T. Peyrin, L. Wang, and S. Wu, "Security analysis ´ of prince," in *Fast Software Encryption (FSE)*, ser. Lecture Notes in Computer Science, vol. 8424. Springer, 2014, pp. 92–111.
- [52] J. Kim, J. Park, S. Roh, J. Chung, Y. Lee, T. Kim, and B. Lee, "Tiktag: Breaking arm's memory tagging extension with speculative execution," in *2025 IEEE Symposium on Security and Privacy (SP)*. IEEE, 2025, pp. 4063–4081.
- [53] Y. Kim, J. Lee, and H. Kim, "Hardware-based always-on heap memory safety," in *2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, 2020, pp. 1153–1166.
- [54] V. Kiriansky and C. Waldspurger, "Speculative buffer overflows: Attacks and defenses," *arXiv preprint arXiv:1807.03757*, 2018.
- [55] M. Kounavis, S. Deutsch, S. Ghosh, and D. Durham, "K-Cipher: A Low Latency, Bit Length Parameterizable Cipher," in *2020 IEEE Symposium on Computers and Communications (ISCC)*, 2020, pp. 1–7.
- [56] C. Lattner and V. Adve, "Llvm: A compilation framework for lifelong program analysis & transformation," in *International symposium on code generation and optimization, 2004. CGO 2004.* IEEE, 2004, pp. 75–86.
- [57] T. Lelegard, "Arm system registers: PAC format," https://github. com/lelegard/arm-cpusysregs/blob/main/docs/pac-format.md, 2024, accessed: May 2024.
- [58] M. LeMay, J. Rakshit, S. Deutsch, D. M. Durham, S. Ghosh, A. Nori, J. Gaur, A. Weiler, S. Sultana, K. Grewal *et al.*, "Cryptographic capability computing," in *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*, 2021, pp. 253–267.
- [59] Y. Li, W. Tan, Z. Lv, S. Yang, M. Payer, Y. Liu, and C. Zhang, "Pacmem: Enforcing spatial and temporal memory safety via arm pointer authentication," in *Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security*, 2022, pp. 1901–1915.
- [60] ——, "Pacsan: Enforcing memory safety based on arm pa," *arXiv preprint arXiv:2202.03950*, 2022.
- [61] H. Liljestrand, T. Nyman, L. J. Gunn, J.-E. Ekberg, and N. Asokan, "PACStack: an authenticated call stack," in *30th USENIX Security Symposium (USENIX Security 21)*. USENIX Association, Aug. 2021, pp. 357–374. [Online]. Available: https://www.usenix.org/conference/ usenixsecurity21/presentation/liljestrand
- [62] H. Liljestrand, T. Nyman, K. Wang, C. C. Perez, J.-E. Ekberg, and N. Asokan, "{PAC} it up: Towards pointer integrity using {ARM} pointer authentication," in *28th USENIX Security Symposium (USENIX Security 19)*, 2019, pp. 177–194.
- [63] Z. Lin, Z. Yu, Z. Guo, S. Campanoni, P. Dinda, and X. Xing, "{CAMP}: Compiler and allocator-based heap memory protection," in *33rd USENIX Security Symposium (USENIX Security 24)*, 2024, pp. 4015–4032.
- [64] LLVM Project. (2024) Pointer authentication in clang/llvm. Accessed: 2025-02-05. [Online]. Available: https://clang.llvm.org/ docs/PointerAuthentication.html
- [65] B. B. Madan, S. Phoha, and K. S. Trivedi, "Stackoffence: a technique for defending against buffer overflow attacks," in *International Conference on Information Technology: Coding and Computing (ITCC'05)-Volume II*, vol. 1. IEEE, 2005, pp. 656–661.
- [66] M. Mahzoun, L. Kraleva, R. Posteuca, and T. Ashur, "Differential Cryptanalysis of K-Cipher," in *2022 IEEE Symposium on Computers and Communications (ISCC)*, 2022, pp. 1–7.
- [67] A. J. Mashtizadeh, A. Bittau, D. Boneh, and D. Mazieres, "Ccfi: ` Cryptographically enforced control flow integrity," in *Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security*, 2015, pp. 941–951.
- [68] C. McGarr. (2023) Windows on arm64 pointer authentication codes (pac). Accessed: 2025-02-05. [Online]. Available: https://connormcgarr. github.io/windows-pac-arm64/
- [69] P. Morawiecki, "Practical attacks on the round-reduced PRINCE," Cryptology ePrint Archive, Paper 2015/245, 2015. [Online]. Available: https://eprint.iacr.org/2015/245
- [70] W. T. Na, J. S. Emer, and M. Yan, "Penetrating shields: A systematic analysis of memory corruption mitigations in the spectre era," *arXiv preprint arXiv:2309.04119*, 2023.

- [71] NIST, "Submission requirements and evaluation criteria for the lightweight cryptography standardization process," 2018. [Online]. Available: https://csrc.nist.gov/CSRC/media/Projects/Lightweight-Cryptography/documents/final-lwc-submission-requirementsaugust2018.pdf
- [72] N. Pemberton and A. Amid, "Firemarshal: Making hw/sw co-design reproducible and reliable," in *2021 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*. IEEE, 2021, pp. 299–309.
- [73] S. D. Phaye, G. J. Duck, R. H. Yap, and T. E. Carlson, "Fully randomized pointers," in *Proceedings of the 2025 ACM SIGPLAN International Symposium on Memory Management*, 2025, pp. 94–108.
- [74] T. C. Project. (2020) Memory safety. 70% of high/critical Chrome ˜ bugs are memory-safety. [Online]. Available: https://www.chromium. org/Home/chromium-security/memory-safety/
- [75] I. Qualcomm Technologies, "Pointer authentication on armv8.3 -a," Qualcomm Technologies, Inc., Tech. Rep. v7, Jan 2017. [Online]. Available: https://www.qualcomm.com/content/dam/qcommmartech/dm-assets/documents/pointer-auth-v7.pdf
- [76] J. Ravichandran, W. T. Na, J. Lang, and M. Yan, "Pacman: attacking arm pointer authentication with speculative execution," in *Proceedings of the 49th Annual International Symposium on Computer Architecture*, ser. ISCA '22. New York, NY, USA: Association for Computing Machinery, 2022, p. 685–698. [Online]. Available: https://doi.org/10.1145/3470496.3527429
- [77] R. Rudd, R. Skowyra, D. Bigelow, V. Dedhia, T. Hobson, S. Crane, C. Liebchen, P. Larsen, L. Davi, M. Franz *et al.*, "Address oblivious code reuse: On the effectiveness of leakage resilient diversity." in *NDSS*, 2017.
- [78] H. Sasaki, M. A. Arroyo, M. T. I. Ziad, K. Bhat, K. Sinha, and S. Sethumadhavan, "Practical byte-granular memory blacklisting using califorms," in *Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture*, 2019, pp. 558–571.
- [79] D. Schrammel, S. Sultana, K. Grewal, M. LeMay, D. Durham, M. Unterguggenberger, P. Nasahl, and S. Mangard, "Memes: Memory encryptionbased memory safety on commodity hardware," in *20th International Conference on Security and Cryptography: SECRYPT 2023*. SciTePress, 2023, pp. 25–36.
- [80] F. Schuster, T. Tendyck, C. Liebchen, L. Davi, A.-R. Sadeghi, and T. Holz, "Counterfeit object-oriented programming: On the difficulty of preventing code reuse attacks in c++ applications," in *2015 IEEE Symposium on Security and Privacy*. IEEE, 2015, pp. 745–762.
- [81] K. Serebryany, E. Stepanov, A. Shlyapnikov, V. Tsyrklevich, and D. Vyukov, "Memory tagging and how it improves c/c++ memory safety," *arXiv preprint arXiv:1802.09517*, 2018.
- [82] H. Shacham, "The geometry of innocent flesh on the bone: Return-intolibc without function calls (on the x86)," in *Proceedings of the 14th ACM conference on Computer and communications security*, 2007, pp. 552–561.
- [83] H. Shacham, M. Page, B. Pfaff, E.-J. Goh, N. Modadugu, and D. Boneh, "On the effectiveness of address-space randomization," in *Proceedings of the 11th ACM conference on Computer and communications security*, 2004, pp. 298–307.
- [84] ——, "On the effectiveness of address-space randomization," in *Proceedings of the 11th ACM conference on Computer and communications security*, 2004, pp. 298–307.
- [85] K. Shibutani, T. Isobe, H. Hiwatari, A. Mitsuda, T. Akishita, and T. Shirai, "Piccolo: An Ultra-Lightweight Blockcipher," in *Cryptographic Hardware and Embedded Systems – CHES 2011*, B. Preneel and T. Takagi, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011, pp. 342–357.
- [86] H. Soleimany, C. Blondeau, X. Yu, W. Wu, K. Nyberg, H. Zhang, L. Zhang, and Y. Wang, "Reflection cryptanalysis of prince-like ciphers," *Journal of Cryptology)*, vol. 28, pp. 718–744, 2015.
- [87] L. Song and L. Hu, "Differential fault attack on the prince block cipher," in *Lightweight Cryptography for Security and Privacy*, ser. Lecture Notes in Computer Science, vol. 8162. Springer, 2013, pp. 43–54.
- [88] T. Suzaki, K. Minematsu, S. Morioka, and E. Kobayashi, "TWINE: A Lightweight Block Cipher for Multiple Platforms," in *Selected Areas in Cryptography*, L. R. Knudsen and H. Wu, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, pp. 339–354.
- [89] The Linux Kernel Community. (2024) Pointer authentication on arm64. Accessed: 2025-02-05. [Online]. Available: https://docs.kernel.org/arch/ arm64/pointer-authentication.html

- [90] V. van der Veen, D. Andriesse, M. Stamatogiannakis, X. Chen, H. Bos, and C. Giuffrdia, "The dynamics of innocent flesh on the bone: Code reuse ten years later," in *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security*, 2017, pp. 1675–1689.
- [91] Y. Wang, J. Wu, T. Yue, Z. Ning, and F. Zhang, "Rettag: Hardwareassisted return address integrity on risc-v," in *Proceedings of the 15th European Workshop on Systems Security*, 2022, pp. 50–56.
- [92] J. Woodruff, R. N. Watson, D. Chisnall, S. W. Moore, J. Anderson, B. Davis, B. Laurie, P. G. Neumann, R. Norton, and M. Roe, "The cheri capability model: Revisiting risc in an age of risk," *ACM SIGARCH Computer Architecture News*, vol. 42, no. 3, pp. 457–468, 2014.
- [93] M. T. I. Ziad, M. A. Arroyo, E. Manzhosov, R. Piersma, and S. Sethumadhavan, "No-fat: Architectural support for low overhead memory safety checks," in *2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2021, pp. 916–929.
- [94] M. T. I. Ziad, M. A. Arroyo, E. Manzhosov, and S. Sethumadhavan, "Zerø: Zero-overhead resilient operation under pointer integrity attacks," in *2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2021, pp. 999–1012.