# Takeaway VII:

LLM models generate incoherent as well as factually incorrect text upon inducing a single bit-flip.

### *H. Effectiveness of PRowhammer*

Success rate. We evaluate the effectiveness of PRowhammer in terms of success rate under two configurations: (i) with system noise and (ii) without system noise. In our evaluation, we treat any program running alongside the Rowhammer attack as noise. Since Rowhammer depends on repeatedly accessing specific rows in memory, the most meaningful type of noise is one that also heavily accesses memory. We therefore run multiple copies of the memory-intensive workload PageRank from the GAP benchmark suite [\[16\]](#page-12-24) on all cores along with the PRowhammer attack. This ensures continuous contention for memory and CPU scheduling resources throughout the experiment. We perform the PRowhammer attack in the presence

<span id="page-10-2"></span>TABLE VII: Success rate of PRowhammer with and without system noise.

| Memory type | Without noise | With noise |
|-------------|---------------|------------|
| DDR3        | 50%           | 30%        |
| DDR4        | 80%           | 73%        |

<span id="page-10-3"></span>![](_page_10_Figure_12.jpeg)

Fig. 9: Distribution of time taken in seconds for (a) time to get first exploitable flip (b) time to get first flip for DDR3 and DDR4.

of this noise, and Table [VII](#page-10-2) presents the empirical success rate of our attack in the presence of this noise.

Time. We also present the cost of PRowhammer in terms of time taken for (a) time to get the first exploitable flip (bitflip at the correct offset) and (b) time to get the first flip, as shown in Fig. [9.](#page-10-3) We observe that the time taken by our attack with DDR3 is lower than DDR4. However, in absolute numbers, the time to get the first flip takes few minutes, and the time to get an exploitable flip is under an hour for both DDR3 and DDR4. Please note that the time taken by memory massaging is in microseconds since we use vmtouch for pagecache eviction [\[34\]](#page-13-18).

Applicability. The effectiveness of our attack is not limited to frameworks like PyTorch and Llama.cpp. Production LLM servers such as the NVIDIA Triton Inference Server [\[5\]](#page-12-25) and TensorRT-LLM [\[4\]](#page-12-26) primarily rely on cuBLASLt for GEMM and tensor-core operations, depending on the selected backend and model configuration. Our attack targets these underlying GPU libraries rather than the high-level framework. Specifically, it requires profiling the kernels selected at runtime and performing SASS-level analysis. Because kernel implementations differ across library versions and GPU architectures, profiling must be repeated for each (library version, GPU architecture) pair. Although autotuning may alter kernel selection across matrix shapes and precision modes, the overall attack methodology remains unchanged.

### VI. POTENTIAL COUNTERMEASURES

GPU-specific countermeasures. We posit that effective mitigation of our attack requires protection mechanisms explicitly designed for the GPU execution path. A key enabler of our attack is that bit flips within the compressed SASS code can yield syntactically valid yet semantically corrupted instructions. Consequently, we advocate augmenting the GPU toolchain with error detection and correction integrated into the compression/decompression pipeline, ensuring that corruption in the compressed binary is detected before execution. Integrity verification should be performed after code transfer into GPU memory, as Rowhammer-induced corruptions arise in hDRAM accessible to the GPU. We further propose the use of cryptographic hash-based validation immediately before kernel dispatch, providing end-to-end assurance of code integrity. Finally, lightweight ECC or CRC protection within GPU instruction caches and decompression units would provide runtime resilience to single-bit corruptions, thereby complementing system-level Rowhammer defenses. Instruction-corruption defenses for ML workloads [\[21\]](#page-12-27) have been proposed, but are tailored for CPUs.

Rowhammer countermeasures. The proposed attack primarily targets the hDRAM, and therefore, conventional CPUoriented Rowhammer defenses remain relevant. These include DRAM-level and memory-controller-level techniques [\[45\]](#page-13-4), [\[46\]](#page-13-5) that rely on hardware modifications. DDR5's Per-Row Activation Counter (PRAC) [\[6\]](#page-12-13) offers robust protection against Rowhammer, though it introduces timing-channel vulnerabilities [\[65\]](#page-13-6); the subsequent Timing-Safe PRAC (TPRAC) design eliminates these channels without compromising effectiveness. Error-Correcting Codes (ECC), implemented at hardware or software layers, mitigate bit-flip rates but are still susceptible to single-bit corruption [\[23\]](#page-12-2) [\[33\]](#page-12-4).

Targeted software defenses like CATT [\[18\]](#page-12-28) and Soft-TRR [\[71\]](#page-14-1) protect specific privileged CPU data structures such as page tables and kernel memory, effectively preventing privilege escalation, but do not prevent bit-flips in DRAM outside those protected structures. Guard-row-based approaches like ZebRAM [\[36\]](#page-13-23) stripe memory into alternating guard rows (rows never allocated to any process) and data rows to absorb bit flips, but incur a DRAM capacity loss of 50-67%, which is impractical. Coarse-grained isolation schemes like Siloz [\[44\]](#page-13-24) assign entire DRAM subarrays (physically isolated structures comprising hundreds of rows each) to individual domains, but limit the number of supported domains and introduce severe memory stranding (unused reserved memory that cannot be used by other domains). The most recent work, Citadel [\[55\]](#page-13-25), addresses most of these scalability limitations by supporting thousands of variably-sized domains, but still does not provide protection against shared-library-based attacks. More recently, MOAT [\[50\]](#page-13-26) shows that Panopticon [\[17\]](#page-12-29), which inspired JEDEC's PRAC+ABO framework in DDR5, is vulnerable, indicating that PRAC-based mitigations are not inherently secure. Rowhammer remains an unsolved problem as existing mitigations continue to be circumvented [\[48\]](#page-13-7) [\[31\]](#page-12-14).

PRowhammer can only be mitigated if DRAM bit-flips are prevented altogether, or if CPU-to-GPU communication includes integrity verification that detects corruption.

### VII. RELATED WORK

Early research demonstrates that Rowhammer enables OS privilege escalation [\[56\]](#page-13-27). Researchers have extended these techniques to create sophisticated cross-virtual-machine attacks, such as Flip Feng Shui [\[54\]](#page-13-17), and kernel-manipulation attacks, like Go Go Gadget [\[60\]](#page-13-28). Ultimately, these attacks compromise hypervisors using tools like HyperHammer [\[20\]](#page-12-30). Attackers expand the trigger mechanisms beyond traditional CPU-based exploitation to include integrated GPUs that induce bit-flips in shared hDRAM affecting CPU processes [\[25\]](#page-12-31), remote network-based vectors that exploit Network Interface Cards [\[59\]](#page-13-29), FPGA-driven bit-flips in hDRAM [\[64\]](#page-13-30), browserbased exploits [\[27\]](#page-12-32) [\[34\]](#page-13-18), and ARM/Android mobile platforms [\[63\]](#page-13-31). Rowhammer vulnerabilities persist across successive DRAM generations. Researchers first demonstrated attacks on DDR3 modules on Intel processors [\[35\]](#page-13-1). They then exploited DDR4 systems with TRRespass [\[26\]](#page-12-1) and LPDDR4 devices with Blacksmith [\[30\]](#page-12-3). Recently, Zenhammer showed susceptibility in DDR5 modules on AMD platforms [\[31\]](#page-12-14), and finally, GDDR6 modules used in GPUs were also targeted [\[43\]](#page-13-3). Experimental results also show that DRAM modules with error-correcting code (ECC), including DDR3, DDR4, and DDR5, remain vulnerable to targeted bit-flips despite on-die mitigation mechanisms [\[23\]](#page-12-2), [\[33\]](#page-12-4), [\[48\]](#page-13-7), [\[69\]](#page-14-2). Accuracy degradation attacks. A large body of research investigates accuracy degradation attacks on ML models [\[29\]](#page-12-5), [\[41\]](#page-13-2), [\[52\]](#page-13-8), [\[53\]](#page-13-9), [\[68\]](#page-13-10). Most studies target model weights and therefore require multiple bit-flips. Several works demonstrate the practical feasibility of such attacks using Rowhammer [\[29\]](#page-12-5), [\[41\]](#page-13-2), [\[68\]](#page-13-10). The work that most closely relates to ours is [\[41\]](#page-13-2), which also uses Rowhammer to corrupt library code targeting ML models. However, Li et al. [\[41\]](#page-13-2) leave attacks on closedsource GPU libraries as an open problem, which we address in this paper. In doing so, we advance the state of the art in accuracy degradation attacks. Existing fault injection defenses for DNNs, such as DeepDyve [\[42\]](#page-13-32), fail against both [\[41\]](#page-13-2) and our attack, since both target the library code rather than the model parameters.

### VIII. FUTURE WORK

While PRowhammer has been successful in accuracy degradation attacks, we note that more stealthy attacks exist, such as backdoor injection [\[61\]](#page-13-33) and weight stealing [\[51\]](#page-13-34). Expanding our PRowhammer attack for such scenarios remains an open problem, particularly due to the compressed nature of the libraries. Another interesting direction of research could be combining PRowhammer with GPUHammer [\[43\]](#page-13-3) and investigating the potential exploits in the presence of both attacks. Attacks on cryptographic algorithms are another open problem, as such attacks require a more informed choice of bits to be flipped.

### IX. CONCLUSION

We demonstrated PRowhammer, a novel attack exploiting the architectural coupling between CPUs and GPUs through hDRAM. We leveraged CPU-based Rowhammer to induce bit-flips in hDRAM, corrupting GPU kernel code before execution. We overcame significant challenges inherent to this attack, including the massive size of GPU shared libraries and proprietary code compression. Our automated techniques successfully identified exploitable bit-flip locations despite these obstacles. We validated PRowhammer's effectiveness on stateof-the-art ML models in realistic black-box settings. A single bit-flip in GPU shared libraries degraded image classification accuracy to random guessing and caused LLMs to produce incoherent text. Overall, this work exposed an entirely new class of GPU vulnerability rooted in the architectural coupling between CPUs and GPUs, underscoring the need for holistic security approaches to heterogeneous computing systems.

### X. ACKNOWLEDGEMENTS

We would like to thank members of the CASPER research group and Pratheek B for their valuable feedback. This work is supported by the Trust Lab Grant 2024.

### REFERENCES

- <span id="page-12-9"></span>[1] "cuBLAS 13.0 documentation — docs.nvidia.com," [https://docs.nvidia.](https://docs.nvidia.com/cuda/cublas/) [com/cuda/cublas/,](https://docs.nvidia.com/cuda/cublas/) [Accessed 15-08-2025].
- <span id="page-12-22"></span>[2] "GitHub - pytorch/pytorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration — github.com," [https://github.c](https://github.com/pytorch/pytorch) [om/pytorch/pytorch,](https://github.com/pytorch/pytorch) [Accessed 18-08-2025].
- <span id="page-12-19"></span>[3] "NVIDIA cuDNN docs.nvidia.com," [https://docs.nvidia.com/deeplearni](https://docs.nvidia.com/deeplearning/cudnn/latest/) [ng/cudnn/latest/,](https://docs.nvidia.com/deeplearning/cudnn/latest/) [Accessed 04-11-2025].
- <span id="page-12-26"></span>[4] "NVIDIA TensorRT-LLM — docs.nvidia.com," [https://docs.nvidia.co](https://docs.nvidia.com/tensorrt-llm/) [m/tensorrt-llm/,](https://docs.nvidia.com/tensorrt-llm/) [Accessed 27-02-2026].
- <span id="page-12-25"></span>[5] "NVIDIA Triton Inference Server — NVIDIA Triton Inference Server — docs.nvidia.com," [https://docs.nvidia.com/deeplearning/triton](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)[inference-server/user-guide/docs/,](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/) [Accessed 27-02-2026].
- <span id="page-12-13"></span>[6] "JEDEC Updates JESD79-5C DDR5 SDRAM Standard: Elevating Performance and Security for Next-Gen Technologies JEDEC jedec.org," [https://www.jedec.org/news/pressreleases/jedec-updates-jesd79-](https://www.jedec.org/news/pressreleases/jedec-updates-jesd79-5c-ddr5-sdram-standard-elevating-performance-and-security) [5c-ddr5-sdram-standard-elevating-performance-and-security,](https://www.jedec.org/news/pressreleases/jedec-updates-jesd79-5c-ddr5-sdram-standard-elevating-performance-and-security) 2024, [Accessed 20-08-2025].
- <span id="page-12-12"></span>[7] "GitHub - ggml-org/ggml: Tensor library for machine learning github.com," [https://github.com/ggml-org/ggml,](https://github.com/ggml-org/ggml) 2025.
- <span id="page-12-18"></span>[8] "GitHub - ggml-org/llama.cpp: LLM inference in C/C++ github.com," [https://github.com/ggml-org/llama.cpp,](https://github.com/ggml-org/llama.cpp) 2025.
- <span id="page-12-20"></span>[9] "nvfatbin," [https://pdfs.semanticscholar.org/5096/25785304410039297b](https://pdfs.semanticscholar.org/5096/25785304410039297b741ad2007e7ce0636b.pdf) [741ad2007e7ce0636b.pdf,](https://pdfs.semanticscholar.org/5096/25785304410039297b741ad2007e7ce0636b.pdf) 2025.
- <span id="page-12-6"></span>[10] "NVIDIA CUDA Compiler Driver 13.0 documentation docs.nvidia.com," [https://docs.nvidia.com/cuda/cuda- compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)[driver-nvcc/,](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) 2025.
- <span id="page-12-17"></span>[11] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, ´ C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viegas, O. Vinyals, ´ P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, "TensorFlow: Large-scale machine learning on heterogeneous systems," 2015, software available from tensorflow.org. [Online]. Available: <https://www.tensorflow.org/>
- <span id="page-12-11"></span>[12] E. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cappelli, R. Cojocaru, M. Debbah, E. Goffinet, D. Hesslow, J. Launay, Q. Malartic ´ *et al.*, "The falcon series of open language models," *arXiv preprint arXiv:2311.16867*, 2023.
- <span id="page-12-15"></span>[13] S. Amer, Y. Wang, H. Kippen, T. Dang, D. Genkin, A. Kwong, A. Nelson, and A. Yerukhimovich, "Pq-hammer: End-to-end key recovery attacks on post-quantum cryptography using rowhammer," in *2025 IEEE Symposium on Security and Privacy (SP)*. IEEE Computer Society, 2024, pp. 48–48.
- <span id="page-12-16"></span>[14] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski *et al.*, "Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation," in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, 2024, pp. 929–947.
- <span id="page-12-0"></span>[15] Z. B. Aweke, S. F. Yitbarek, R. Qiao, R. Das, M. Hicks, Y. Oren, and T. Austin, "Anvil: Software-based protection against next-generation rowhammer attacks," in *Proceedings of the Twenty-First International Conference on Architectural Support for Programming Languages and Operating Systems*, ser. ASPLOS '16. New York, NY, USA: Association for Computing Machinery, 2016, p. 743–755. [Online]. Available:<https://doi.org/10.1145/2872362.2872390>

- <span id="page-12-24"></span>[16] S. Beamer, K. Asanovic, and D. Patterson, "The gap benchmark suite," ´ 2017. [Online]. Available:<https://arxiv.org/abs/1508.03619>
- <span id="page-12-29"></span>[17] T. Bennett, S. Saroiu, A. Wolman, and L. Cojocar, "Panopticon: A complete in-dram rowhammer mitigation," in *1st Workshop on DRAM Security (DRAMSec)*, June 2021. [Online]. Available: [https://www.microsoft.com/en-us/research/publication/panopticon-a](https://www.microsoft.com/en-us/research/publication/panopticon-a-complete-in-dram-rowhammer-mitigation/)[complete-in-dram-rowhammer-mitigation/](https://www.microsoft.com/en-us/research/publication/panopticon-a-complete-in-dram-rowhammer-mitigation/)
- <span id="page-12-28"></span>[18] F. Brasser, L. Davi, D. Gens, C. Liebchen, and A.-R. Sadeghi, "CAn't touch this: Software-only mitigation against rowhammer attacks targeting kernel memory," in *26th USENIX Security Symposium (USENIX Security 17)*. Vancouver, BC: USENIX Association, Aug. 2017, pp. 117–130. [Online]. Available: [https://www.usenix.org/confere](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/brasser) [nce/usenixsecurity17/technical-sessions/presentation/brasser](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/brasser)
- <span id="page-12-23"></span>[19] Canonical, "Ubuntu Manpage: vmtouch - the Virtual Memory Toucher — manpages.ubuntu.com," [https://manpages.ubuntu.com/manpages/que](https://manpages.ubuntu.com/manpages/questing/en/man8/vmtouch.8.html) [sting/en/man8/vmtouch.8.html,](https://manpages.ubuntu.com/manpages/questing/en/man8/vmtouch.8.html) [Accessed 06-09-2025].
- <span id="page-12-30"></span>[20] W. Chen, Z. Zhang, X. Zhang, Q. Shen, Y. Yarom, D. Genkin, C. Yan, and Z. Wang, "Hyperhammer: Breaking free from kvm-enforced isolation," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, 2025, pp. 545–559.
- <span id="page-12-27"></span>[21] Y. Chen, Y. Yuan, Z. Liu, S. Hu, T. Li, and S. Wang, "Bitshield: Defending against bit-flip attacks on DNN executables," in *32nd Annual Network and Distributed System Security Symposium, NDSS 2025, San Diego, California, USA, February 24-28, 2025*. The Internet Society, 2025. [Online]. Available: [https://www.ndss-symposium.org/ndss](https://www.ndss-symposium.org/ndss-paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/)[paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/](https://www.ndss-symposium.org/ndss-paper/bitshield-defending-against-bit-flip-attacks-on-dnn-executables/)
- <span id="page-12-21"></span>[22] L. Cojocar, J. Kim, M. Patel, L. Tsai, S. Saroiu, A. Wolman, and O. Mutlu, "Are we susceptible to rowhammer? an end-to-end methodology for cloud providers," in *2020 IEEE symposium on security and privacy (SP)*. IEEE, 2020, pp. 712–728.
- <span id="page-12-2"></span>[23] L. Cojocar, K. Razavi, C. Giuffrida, and H. Bos, "Exploiting correcting codes: On the effectiveness of ecc memory against rowhammer attacks," in *2019 IEEE Symposium on Security and Privacy (SP)*. IEEE, 2019, pp. 55–71.
- <span id="page-12-8"></span>[24] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "Imagenet: A large-scale hierarchical image database," in *2009 IEEE conference on computer vision and pattern recognition*. Ieee, 2009, pp. 248–255.
- <span id="page-12-31"></span>[25] P. Frigo, C. Giuffrida, H. Bos, and K. Razavi, "Grand pwning unit: Accelerating microarchitectural attacks with the gpu," in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018, pp. 195–210.
- <span id="page-12-1"></span>[26] P. Frigo, E. Vannacci, H. Hassan, V. van der Veen, O. Mutlu, C. Giuffrida, H. Bos, and K. Razavi, "Trrespass: Exploiting the many sides of target row refresh," *CoRR*, vol. abs/2004.01807, 2020. [Online]. Available:<https://arxiv.org/abs/2004.01807>
- <span id="page-12-32"></span>[27] D. Gruss, C. Maurice, and S. Mangard, "Rowhammer. js: A remote software-induced fault attack in javascript," in *Detection of Intrusions and Malware, and Vulnerability Assessment: 13th International Conference, DIMVA 2016, San Sebastian, Spain, July 7-8, 2016, Proceedings ´ 13*. Springer, 2016, pp. 300–321.
- <span id="page-12-7"></span>[28] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, pp. 770–778.
- <span id="page-12-5"></span>[29] S. Hong, P. Frigo, Y. Kaya, C. Giuffrida, and T. Dumitras, , "Terminal brain damage: Exposing the graceless degradation in deep neural networks under hardware fault attacks," in *28th USENIX Security Symposium (USENIX Security 19)*, 2019, pp. 497–514.
- <span id="page-12-3"></span>[30] P. Jattke, V. van der Veen, P. Frigo, S. Gunter, and K. Razavi, "BLACK-SMITH: Scalable Rowhammering in the Frequency Domain," 2022-05, [https://github.com/comsec-group/blacksmith.](https://github.com/comsec-group/blacksmith)
- <span id="page-12-14"></span>[31] P. Jattke, M. Wipfli, F. Solt, M. Marazzi, M. Bolcskei, and ¨ K. Razavi, "ZenHammer: Rowhammer attacks on AMD zen-based platforms," in *33rd USENIX Security Symposium (USENIX Security 24)*. Philadelphia, PA: USENIX Association, Aug. 2024, pp. 1615–1633. [Online]. Available: [https://www.usenix.org/conference/usenixsecurity](https://www.usenix.org/conference/usenixsecurity24/presentation/jattke) [24/presentation/jattke](https://www.usenix.org/conference/usenixsecurity24/presentation/jattke)
- <span id="page-12-10"></span>[32] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, "Mistral 7b," 2023. [Online]. Available: <https://arxiv.org/abs/2310.06825>
- <span id="page-12-4"></span>[33] N. Kamadan, W. Wang, S. van Schaik, C. Garman, D. Genkin, and Y. Yarom, "Ecc.fail: Mounting rowhammer attacks on ddr4 servers with ecc memory," in *34th USENIX Security Symposium (USENIX*

- *Security 25)*. USENIX Association, 2025. [Online]. Available: [https:](https://www.usenix.org/conference/usenixsecurity25/presentation/kamadan) [//www.usenix.org/conference/usenixsecurity25/presentation/kamadan](https://www.usenix.org/conference/usenixsecurity25/presentation/kamadan)
- <span id="page-13-18"></span>[34] I. Kang, W. Wang, J. Kim, S. van Schaik, Y. Tobah, D. Genkin, A. Kwong, and Y. Yarom, "SledgeHammer: Amplifying rowhammer via bank-level parallelism," in *33rd USENIX Security Symposium (USENIX Security 24)*. Philadelphia, PA: USENIX Association, Aug. 2024, pp. 1597–1614. [Online]. Available: [https://www.usenix.org/con](https://www.usenix.org/conference/usenixsecurity24/presentation/kang) [ference/usenixsecurity24/presentation/kang](https://www.usenix.org/conference/usenixsecurity24/presentation/kang)
- <span id="page-13-1"></span>[35] Y. Kim, R. Daly, J. S. Kim, C. Fallin, J. Lee, D. Lee, C. Wilkerson, K. Lai, and O. Mutlu, "Flipping bits in memory without accessing them: An experimental study of DRAM disturbance errors," in *ACM/IEEE 41st International Symposium on Computer Architecture, ISCA 2014, Minneapolis, MN, USA, June 14-18, 2014*. IEEE Computer Society, 2014, pp. 361–372. [Online]. Available: <https://doi.org/10.1109/ISCA.2014.6853210>
- <span id="page-13-23"></span>[36] R. K. Konoth, M. Oliverio, A. Tatar, D. Andriesse, H. Bos, C. Giuffrida, and K. Razavi, "ZebRAM: Comprehensive and compatible software protection against rowhammer attacks," in *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*. Carlsbad, CA: USENIX Association, Oct. 2018, pp. 697–710. [Online]. Available: <https://www.usenix.org/conference/osdi18/presentation/konoth>
- <span id="page-13-14"></span>[37] A. Krizhevsky, G. Hinton *et al.*, "Learning multiple layers of features from tiny images.(2009)," 2009.
- <span id="page-13-22"></span>[38] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee *et al.*, "Natural questions: a benchmark for question answering research," *Transactions of the Association for Computational Linguistics*, vol. 7, pp. 453–466, 2019.
- <span id="page-13-16"></span>[39] A. Kwong, D. Genkin, D. Gruss, and Y. Yarom, "Rambleed: Reading bits in memory without accessing them," in *41st IEEE Symposium on Security and Privacy (S&P)*, 2020.
- <span id="page-13-12"></span>[40] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 2002.
- <span id="page-13-2"></span>[41] S. Li, X. Wang, M. Xue, H. Zhu, Z. Zhang, Y. Gao, W. Wu, and X. S. Shen, "Yes, One-Bit-Flip matters! universal DNN model inference depletion with runtime code fault injection," in *33rd USENIX Security Symposium (USENIX Security 24)*. Philadelphia, PA: USENIX Association, Aug. 2024, pp. 1315–1330. [Online]. Available: [https:](https://www.usenix.org/conference/usenixsecurity24/presentation/li-shaofeng) [//www.usenix.org/conference/usenixsecurity24/presentation/li-shaofeng](https://www.usenix.org/conference/usenixsecurity24/presentation/li-shaofeng)
- <span id="page-13-32"></span>[42] Y. Li, M. Li, B. Luo, Y. Tian, and Q. Xu, "Deepdyve: Dynamic verification for deep neural networks," in *Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security*, ser. CCS '20. New York, NY, USA: Association for Computing Machinery, 2020, p. 101–112. [Online]. Available: [https:](https://doi.org/10.1145/3372297.3423338) [//doi.org/10.1145/3372297.3423338](https://doi.org/10.1145/3372297.3423338)
- <span id="page-13-3"></span>[43] C. S. Lin, J. Qu, and G. Saileshwar, "Gpuhammer: Rowhammer attacks on gpu memories are practical," *arXiv preprint arXiv:2507.08166*, 2025.
- <span id="page-13-24"></span>[44] K. Loughlin, J. Rosenblum, S. Saroiu, A. Wolman, D. Skarlatos, and B. Kasikci, "Siloz: Leveraging dram isolation domains to prevent inter-vm rowhammer," in *Proceedings of the 29th Symposium on Operating Systems Principles*, ser. SOSP '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 417–433. [Online]. Available:<https://doi.org/10.1145/3600006.3613143>
- <span id="page-13-4"></span>[45] M. Marazzi, P. Jattke, F. Solt, and K. Razavi, "Protrr: Principled yet optimal in-dram target row refresh," in *2022 IEEE Symposium on Security and Privacy (SP)*, 2022, pp. 735–753.
- <span id="page-13-5"></span>[46] M. Marazzi, F. Solt, P. Jattke, K. Takashi, and K. Razavi, "Rega: Scalable rowhammer mitigation with refresh-generating activations," in *2023 IEEE Symposium on Security and Privacy (SP)*, 2023, pp. 1684– 1701.
- <span id="page-13-0"></span>[47] R. Merritt, "Why GPUs Are Great for AI — blogs.nvidia.com," [https:](https://blogs.nvidia.com/blog/why-gpus-are-great-for-ai/) [//blogs.nvidia.com/blog/why-gpus-are-great-for-ai/,](https://blogs.nvidia.com/blog/why-gpus-are-great-for-ai/) 2023, [Accessed 13-08-2025].
- <span id="page-13-7"></span>[48] D. Meyer, P. Jattke, M. Marazzi, S. Qazi, D. Moghimi, and K. Razavi, "Phoenix: Rowhammer Attacks on DDR5 with Self-Correcting Synchronization," in *Proceedings of the 2026 IEEE Symposium on Security and Privacy (SP)*. San Francisco, CA, USA: IEEE, May 2026.
- <span id="page-13-20"></span>[49] P. Pessl, D. Gruss, C. Maurice, M. Schwarz, and S. Mangard, "{DRAMA}: Exploiting {DRAM} addressing for {Cross-CPU} attacks," in *25th USENIX security symposium (USENIX security 16)*, 2016, pp. 565–581.
- <span id="page-13-26"></span>[50] M. Qureshi and S. Qazi, "Moat: Securely mitigating rowhammer with per-row activation counters," in *Proceedings of the 30th ACM*

- *International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, ser. ASPLOS '25. New York, NY, USA: Association for Computing Machinery, 2025, p. 698–714. [Online]. Available:<https://doi.org/10.1145/3669940.3707278>
- <span id="page-13-34"></span>[51] A. S. Rakin, M. H. I. Chowdhuryy, F. Yao, and D. Fan, "Deepsteal: Advanced model extractions leveraging efficient weight stealing in memories," in *2022 IEEE symposium on security and privacy (SP)*. IEEE, 2022, pp. 1157–1174.
- <span id="page-13-8"></span>[52] A. S. Rakin, Z. He, and D. Fan, "Bit-flip attack: Crushing neural network with progressive bit search," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2019, pp. 1211–1220.
- <span id="page-13-9"></span>[53] A. S. Rakin, Z. He, J. Li, F. Yao, C. Chakrabarti, and D. Fan, "Tbfa: Targeted bit-flip adversarial weight attack," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 11, pp. 7928– 7939, 2021.
- <span id="page-13-17"></span>[54] K. Razavi, B. Gras, E. Bosman, B. Preneel, C. Giuffrida, and H. Bos, "Flip feng shui: Hammering a needle in the software stack," in *25th USENIX Security Symposium (USENIX Security 16)*, 2016, pp. 1–18.
- <span id="page-13-25"></span>[55] A. Saxena, W. Wang, and A. Daglis, *Citadel: Rethinking Memory Allocation to Safeguard Against Inter-Domain Rowhammer Exploits*. New York, NY, USA: Association for Computing Machinery, 2025, p. 1117–1131. [Online]. Available: [https://doi.org/10.1145/3725843.3756](https://doi.org/10.1145/3725843.3756098) [098](https://doi.org/10.1145/3725843.3756098)
- <span id="page-13-27"></span>[56] M. Seaborn and T. Dullien, "Exploiting the dram rowhammer bug to gain kernel privileges," *Black Hat*, vol. 15, no. 71, p. 2, 2015.
- <span id="page-13-11"></span>[57] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings*, Y. Bengio and Y. LeCun, Eds., 2015. [Online]. Available:<http://arxiv.org/abs/1409.1556>
- <span id="page-13-21"></span>[58] A. Tatar, C. Giuffrida, H. Bos, and K. Razavi, "Defeating software mitigations against rowhammer: a surgical precision hammer," in *International Symposium on Research in Attacks, Intrusions, and Defenses*. Springer, 2018, pp. 47–66.
- <span id="page-13-29"></span>[59] A. Tatar, R. K. Konoth, E. Athanasopoulos, C. Giuffrida, H. Bos, and K. Razavi, "Throwhammer: Rowhammer attacks over the network and defenses," in *2018 USENIX Annual Technical Conference (USENIX ATC 18)*, 2018, pp. 213–226.
- <span id="page-13-28"></span>[60] Y. Tobah, A. Kwong, I. Kang, D. Genkin, and K. G. Shin, "Go go gadget hammer: Flipping nested pointers for arbitrary data leakage," in *33rd USENIX Security Symposium (USENIX Security 24)*, 2024, pp. 1635–1650.
- <span id="page-13-33"></span>[61] M. C. Tol, S. Islam, A. J. Adiletta, B. Sunar, and Z. Zhang, "Don't knock! rowhammer at the backdoor of dnn models," in *2023 53rd Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)*. IEEE, 2023, pp. 109–122.
- <span id="page-13-15"></span>[62] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-13-31"></span>[63] V. Van Der Veen, Y. Fratantonio, M. Lindorfer, D. Gruss, C. Maurice, G. Vigna, H. Bos, K. Razavi, and C. Giuffrida, "Drammer: Deterministic rowhammer attacks on mobile platforms," in *Proceedings of the 2016 ACM SIGSAC conference on computer and communications security*, 2016, pp. 1675–1689.
- <span id="page-13-30"></span>[64] Z. Weissman, T. Tiemann, D. Moghimi, E. Custodio, T. Eisenbarth, and B. Sunar, "Jackhammer: Efficient rowhammer on heterogeneous fpgacpu platforms," *arXiv preprint arXiv:1912.11523*, 2019.
- <span id="page-13-6"></span>[65] J. Woo, J. Qu, G. Saileshwar, and P. J. Nair, "When mitigations backfire: Timing channel attacks and defense for prac-based rowhammer mitigations," ser. ISCA '25. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: [https://doi.org/10.1](https://doi.org/10.1145/3695053.3731007) [145/3695053.3731007](https://doi.org/10.1145/3695053.3731007)
- <span id="page-13-13"></span>[66] H. Xiao, K. Rasul, and R. Vollgraf, "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms," *arXiv preprint arXiv:1708.07747*, 2017.
- <span id="page-13-19"></span>[67] M. Yan, C. W. Fletcher, and J. Torrellas, "Cache telepathy: Leveraging shared resource attacks to learn {DNN} architectures," in *29th USENIX Security Symposium (USENIX Security 20)*, 2020, pp. 2003–2020.
- <span id="page-13-10"></span>[68] F. Yao, A. S. Rakin, and D. Fan, "DeepHammer: Depleting the intelligence of deep neural networks through targeted chain of bit flips," in *29th USENIX Security Symposium (USENIX Security 20)*. USENIX Association, Aug. 2020, pp. 1463–1480. [Online]. Available: <https://www.usenix.org/conference/usenixsecurity20/presentation/yao>

- <span id="page-14-2"></span>[69] I. E. Yuksel, A. Olgun, N. Bostanci, H. Luo, A. G. Yaglikci, and O. Mutlu, "Columndisturb: Understanding column-based read disturbance in real dram chips and implications for future systems," in *Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO '25. New York, NY, USA: Association for Computing Machinery, 2025, p. 975–994. [Online]. Available: <https://doi.org/10.1145/3725843.3756022>
- <span id="page-14-0"></span>[70] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "Bertscore: Evaluating text generation with bert," *arXiv preprint arXiv:1904.09675*, 2019.
- <span id="page-14-1"></span>[71] Z. Zhang, Y. Cheng, M. Wang, W. He, W. Wang, S. Nepal, Y. Gao, K. Li, Z. Wang, and C. Wu, "SoftTRR: Protect page tables against rowhammer attacks using software-only target row refresh," in *2022 USENIX Annual Technical Conference (USENIX ATC 22)*. Carlsbad, CA: USENIX Association, Jul. 2022, pp. 399–414. [Online]. Available: <https://www.usenix.org/conference/atc22/presentation/zhang-zhi>

### APPENDIX

### *A. Abstract*

*This artifact demonstrates the core mechanism of PRowhammer, showing how a single Rowhammer-induced bitflip in host DRAM can alter GPU-executed code. It provides a minimal, reproducible setup that illustrates how bit-flips in deduplicated shared library pages propagate to GPUs and modify low-level instructions while preserving syntactic validity. The artifact includes tools to identify vulnerable bit locations, induce controlled bit-flips, and observe their impact on GPU kernel execution. Through simplified examples, users can reproduce how a single bit-flip in a shared library can lead to semantically altered GPU computation, highlighting the security implications of PRowhammer in heterogeneous CPU–GPU systems.*

### *B. Artifact check-list (meta-information)*

- Model: ResNet-18, ResNet-34, ResNet-50 and VGG-16
- Data set: MNIST, FMNIST, CIFAR-10 and ImageNet
- Run-time environment: Anaconda
- How much disk space is required?: Around 70GB.
- How much time is needed to prepare workflow?: 2-3 hours
- How much time is needed to complete experiments?: 6-9 hours
- Publicly available?: Yes
- Archived (provide DOI)?: The DOI for the artifact is [https:](https://doi.org/10.5281/zenodo.19326669) [//doi.org/10.5281/zenodo.19326669.](https://doi.org/10.5281/zenodo.19326669)

### *C. Description*

### *1) How to access:*

• We have provided our artifact in Zenodo repository [https://doi.org/10.5281/zenodo.19326669.](https://doi.org/10.5281/zenodo.19326669) Download all the files and follow the README.md file for further instruction.

============================================== \$ tar -xvzf prowhammer-artifact.tar.gz ==============================================

### *2) Hardware Dependencies:*

- GPU: NVIDIA GPUs any one of RTX A6000, RTX 4090, or RTX 5060
- CPU: Minimum 8-core processor
- Memory: At least 16 GB of DRAM

### *3) Software Dependencies:*

- Operating System: Ubuntu 24.04 LTS
- CUDA Toolkit: Version 12.8
- Python Environment: Python 3.12 with Anaconda
- Tools: GNU Make 4.3, CMake 3.28.3 and md5sum (GNU coreutils) 9.4

### *D. Installation*

- 1) To install CUDA toolkit go to [https://developer.nvidia](https://developer.nvidia.com/cuda-downloads) [.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and install CUDA Toolkit 12.8. Choose the installer type as deb(local).
- 2) Navigate to the repository and enter the main experiment directory:

============================================== \$ cd prowhammer-artifact \$ cd image\_classification\_and\_llm\_attack ==============================================

3) Install the conda environment (installs Python 3.12 with PyTorch, NumPy, pandas, matplotlib, and all other dependencies):

```
==============================================
$ bash setup_env.sh
$ conda activate prowhammer
==============================================
```

### *E. Experiment Workflow*

The README.md provides a stepwise workflow on how to replicate the attack on your machine. Most of the steps are the same for all the GPUs mentioned. Follow the same steps to get the results for each GPU.

### 1) Step A: Model Training (Optional)

Model download and training is optional; pretrained models and datasets are already included in the repository. To train from scratch, refer to model\_training/README.md and run:

============================================ \$ cd model\_training \$ python download\_dataset.py \$ python run.py --model-dir trained\_models/\ | tee out/golden\_out\_1000.txt

After a successful run, a summary table reporting the baseline accuracy of each model is printed. The pre-

============================================

trained models can also be used directly by specifying the model path in the attack scripts.

### 2) Step B: Profiling Bit-Flip Locations (Optional)

Profiling is only necessary if you are not using the provided conda environment. The profiling step takes approximately 2–3 hours. If the same conda environment is used, the pre-computed bit-flip locations in the .csv files can be used directly.

Refer to profiling\_bit\_flip\_location/README.md and follow the steps below to obtain bit-flip locations for the target libraries.

a) Obtaining the target library.

### Copy libcublasLt.so.12 to the lib/ directory:

============================================

- \$ cd profiling\_bit\_flip\_location
- \$ bash get\_golden\_lib.sh prowhammer ============================================

b) Profile each target library using the corresponding script:

============================================ \$ bash run\_profile\_custom.sh # custom lib \$ bash run\_profile\_cublas.sh # cuBLAS lib \$ bash run\_profile\_ggml.sh # GGML lib ============================================

Each script executes the following five-stage profiling pipeline:

Kernel locater (kernel\_locater\_<target\_lib>.sh): Identifies which kernels inside the target library are invoked by the application. Produces a *regions file* containing memory offsets of interest.

Target-region selection (choose\_target\_region.sh): Selects large, contiguous memory chunks from the regions file that are likely to be Rowhammer-flippable. Produces a list of *candidate regions*.

Bit-flip experiments (run\_flipper\_watchdog\_<target \_lib>.sh): Performs precise bit-flip experiments on the candidate regions and records raw flip results.

Segregation (segregate.sh): Organises the raw flip results into structured per-run outputs.

Useful-flip extraction (extract\_useful\_flips.sh): Filters and summarises the organised results to produce bitflip\_data.csv, which contains the usable flip locations.

After profiling, a .csv file listing all possible flip locations for the target library is produced. To verify the validity of the SASS instructions after flipping, use cuobjdump and diff to inspect changes in the SASS code.

The most critical bit-flip locations identified for the prowhammer environment are already provided in:

- most\_critical\_bit\_flip.csv for mnist, fmnist, and cifar10 (bit 4 at offset 0x95c787a in cublasLt).
- most\_critical\_bit\_flip\_imagenet.csv for imagenet (bit 8 at offset 0xc56745c).

### 3) Step C: Image Classification Attack

*a) Datasets* mnist*,* fmnist*, and* cifar10*:* Download the datasets and run the attack:

============================================ \$ python download\_dataset.py \$ bash scripts/flip\_analysis.sh \ most\_critical\_bit\_flip.csv \$ bash get\_output.sh ============================================

*b) Dataset* imagenet*:* Follow the instructions in imagenet\_dataset.md to download the dataset, then

### run:

============================================

- \$ bash scripts/flip\_analysis\_imagenet.sh \ most\_critical\_bit\_flip\_imagenet.csv \$ bash get\_output\_imagenet.sh
- ============================================
  - *c) Generating plots and tables.:*
- Table III: Follow the steps in attacking\_image\_classification\_model/README .md for each of the four datasets (mnist, fmnist, cifar10, imagenet).
- Figure 7, Table IV, and Table V: Pre-computed exploitable bit-flip locations are provided. Run the following script to reproduce the data:

```
============================================
$ cd plotting_tabulation/scripts/
$ python plot_figure7_table_4_5.py
============================================
```

### • Figure 8: Run the following script:

```
============================================
$ cd plotting_tabulation/scripts/
$ python plot_figure8_table_4_5.py
============================================
```

• Figure 4: Profile the bit-flip locations for each GPU and library. Copy the output from bit-flip-data/bit-flip-stats to plotting\_tabulation/data/figure4/bitflip\_st ats.csv, then run:

```
============================================
$ cd plotting_tabulation/scripts/
$ python plot_figure4.py
============================================
```

### 4) LLM Attack

Run the profiling script to inspect the exploitable bit-flip locations for the <lib\_name> library. The profiling output is stored in bitflip\_data\_<lib\_name>.csv inside the profiling\_bit\_flip\_location/bit-flip-data/ directory. The first column of this file corresponds to the index of the associated log file located under the outs\_<lib\_name>/ subdirectory. The corrupted output for a given run can be inspected in outs\_<lib\_name>/stdout/out\_err\_<index>.log.

### *F. Evaluation and Expected Results*

The expected results from this artifact are to recreate the key results presented in the paper:

- Figures Figure 4, Figure 7, and Figure 8.
- Tables Table III, Table VI, and Table V.

### *G. Notes*

1) While profiling target libraries, if no exploitable bit-flips are found, increase the value of the --iterations pa-

- rameter in the run\_profile\_<target\_lib>.sh scripts.
- 2) During profiling, ensure that the GPUs are adequately cooled and operating within safe thermal limits. Overheating may cause GPU driver crashes, interrupting or invalidating experimental runs.