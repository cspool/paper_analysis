# C. Accelerator/Cache Space Management

In PipeIMC, SRAM arrays can be configured as either accelerators or caches. To evaluate the impact of this partitioning, we vary the fraction of SRAM allocated to accelerators and measure performance across problem sizes for matmul and backprop. The results are presented in Fig. 11.

For the compute-bound workload matmul, allocating more SRAM to accelerators improves performance at small problem sizes, but the optimal point shifts toward fewer accelerator arrays as size increases. In contrast, for the memory-bound backprop, configurations with more accelerator arrays increasingly outperform others as the problem size grows. These results indicate that the optimal partitioning is both workload- and size-dependent. In our evaluation, we use a 25% accelerator configuration with 256 KB of SRAM for acceleration, 512 KB

![](_page_11_Figure_8.jpeg)

Fig. 11. Performance of PipeIMC with different accelerator array ratios under different benchmark sizes. In matmul, the matrices have sizes 128, 256, and 512. In backprop, the hidden layers have sizes 4096, 8192, and 16384.

![](_page_11_Figure_10.jpeg)

Fig. 12. Normalized performance with different memory configurations. for caching, and 256 KB for shared memory. This configuration provides the best overall performance across the evaluated workloads. Future work will explore optimal static partitioning strategies and dynamic reconfiguration mechanisms to further improve PipeIMC performance.

#### D. Sensitivity to DRAM Bandwidth

Given that several workloads exhibit memory-bound behavior, we evaluate the target architectures under varying DRAM configurations to quantify their sensitivity to memory bandwidth. Specifically, we employ DDR4-2400 with 1, 2, and 4 memory channels. Our evaluation includes two memory-bound workloads, backprop and bfs from the Rodinia benchmark suite, and two compute-bound workloads, matvec and attention from transformer kernels. The results are shown in Fig. 12.

For memory-bound workloads, all architectures achieve substantial performance improvements as memory bandwidth increases, attributable to higher effective DRAM throughput. In contrast, compute-bound workloads exhibit only marginal performance gains, as execution time is dominated by computation rather than memory accesses. Importantly, the performance trends remain consistent across DRAM configurations, indicating that the performance benefits of PipeIMC are robust across a wide range of memory bandwidth provisions.

## E. Area, Energy and Efficiency Analysis

Table X shows the estimated area breakdown of SIMT-EVE [2], Duality Cache [11], and PipeIMC. We synthesize the operation tables, schedulers, renaming units and the peripheral control circuits using Synopsis Design Compiler in TSMC 40nm process. The area of computing SRAMs and peripheral circuits are from Cadence Virtuoso. The area of data transpose units is from the original paper [2], [11] and scaled to 40nm process. And the area of shared memory arrays and memory

 $TABLE\ IX \\ Computing\ SRAM\ Array\ Utilization\ of\ SIMT-EVE\ [2],\ Duality\ Cache\ [11]\ and\ PipeIMC$ 

| Benchmark(%) | backprop | bfs  | kmeans | matmul | pathfinder | stencil3d | matvec | layernorm | attention | ffn   |
|--------------|----------|------|--------|--------|------------|-----------|--------|-----------|-----------|-------|
| SIMT-EVE     | 4.91     | 1.48 | 8.86   | 36.72  | 3.33       | 4.45      | 58.06  | 8.03      | 37.35     | 58.86 |
| DualityCache | 3.75     | 4.79 | 13.60  | 23.61  | 6.47       | 7.89      | 32.11  | 13.49     | 27.28     | 31.60 |
| Pipe-1       | 10.16    | 3.03 | 17.13  | 50.44  | 6.69       | 9.76      | 67.42  | 17.04     | 51.83     | 68.41 |
| Pipe-2       | 11.59    | 2.76 | 17.69  | 56.02  | 5.89       | 8.67      | 68.61  | 16.95     | 58.07     | 73.95 |
| Pipe-1r      | 16.50    | 5.18 | 25.74  | 74.56  | 11.37      | 16.64     | 90.24  | 23.03     | 72.04     | 86.65 |
| Pipe-2r      | 17.88    | 5.40 | 29.78  | 79.10  | 12.41      | 17.63     | 92.28  | 25.88     | 76.12     | 87.13 |

TABLE X
ESTIMATED AREA BREAKDOWN OF SIMT-EVE [2], DUALITY CACHE
[11] AND PIPEIMC

| Components                      | SIMT-E                 | VE [2]     | Duality | / Cache [11] | PipeIMC-2r |            |  |
|---------------------------------|------------------------|------------|---------|--------------|------------|------------|--|
| <u>-</u>                        | Area(mm <sup>2</sup> ) | Percentage | Area    | Percentage   | Area       | Percentage |  |
| Warp Scheduler                  | 0.60                   | 4.03%      | /       | /            | 0.60       | 3.42%      |  |
| Decoder & Buffer                | 0.82                   | 5.51%      | 1.35    | 9.75%        | 0.82       | 4.68%      |  |
| Instruction Buffer              | 2.20                   | 14.78%     | 1.76    | 12.72%       | /          | /          |  |
| Operation Table<br>& Dispatcher | /                      | /          | /       | /            | 2.46       | 14.04%     |  |
| Rename Unit                     | /                      | /          | /       | /            | 0.78       | 4.45%      |  |
| Peripheral<br>Control Total     | =3.62                  | 24.31%     | =3.11   | 22.47%       | =4.66      | 26.60%     |  |
| Computing SRAM<br>Peripherals   | 0.71                   | 4.77%      | 0.71    | 5.13%        | 1.44       | 8.22%      |  |
| Computing SRAM<br>Arrays        | 2.64                   | 17.73%     | 2.64    | 19.08%       | 3.50       | 19.98%     |  |
| Computing SRAM<br>Total         | =3.35                  | 22.50%     | =3.35   | 24.21%       | =4.94      | 28.20%     |  |
| Memory Arrays                   | 2.76                   | 18.54%     | 2.76    | 19.94%       | 2.76       | 15.75%     |  |
| Data Transpose<br>Unit          | 2.95                   | 22.30%     | 2.41    | 17.41%       | 2.95       | 16.84%     |  |
| Memory<br>Controller            | 2.21                   | 14.84%     | 2.21    | 15.97%       | 2.21       | 12.61%     |  |
| Total                           | 14.89                  |            | 13.84   | -7.0%        | 17.52      | +17.7%     |  |

controller are from GPUWattch and McPat [20], [22]. We did not count the 512KB array that remained as cache in the breakdown. From the table, PipeIMC-2r has an area overhead of 17.7% and 26.6% compared to SIMT-EVE [2] and Duality Cache [11], respectively. The extra overhead is caused by the renaming units, the operation tables, and the additional compute port and peripherals in the computing SRAMs. Assuming the architectures have abundant cache and memory bandwidth, PipeIMC achieves an area efficiency (throughput per  $mm^2$ ) of 2.17x and 1.68x over SIMT-EVE [2] and Duality Cache [11].

Fig. 10 shows the estimated energy breakdown of SIMT-EVE [2], Duality Cache [11], and PipeIMC. The energy consumption is normalized to SIMT-EVE [2] and includes a breakdown of peripherals, computing SRAMs, caches, and memory controllers. We report runtime average power as the total energy consumed during kernel execution divided by execution time. Across the benchmarks, Pipe-2r has 1.33x and 1.34x average power compared to SIMT-EVE and Duality Cache. Despite this increase in power, Pipe-2r reduces execution time, resulting in higher throughput per watt, computed as performance divided by average power. Overall, Pipe-2r achieves 1.92x and 1.60x energy efficiency improvements compared to SIMT-EVE [2] and Duality Cache [11].

#### VI. FLOATING POINT SUPPORT DISCUSSION

Unlike integer and fixed-point formats, floating-point numbers comprise sign, exponent, and mantissa fields, each requiring distinct operations during arithmetic execution. Mapping them onto a bit-hybrid layout breaks bitline symmetry and increases peripheral overhead.

TABLE XI
ESTIMATED FLOATING-POINT OPERATION LATENCIES ON PIPEIMC

| Operations           | Cycles                          | Operations   | Cycles               |
|----------------------|---------------------------------|--------------|----------------------|
| fadd<br>fsub<br>fcvt | 342-1635<br>450-1926<br>120-265 | fmul<br>fdiv | 288-1047<br>350-1375 |

Assuming that PipeIMC supports fine-grained control over bit-hybrid data layouts, and leveraging the floating-point algorithms proposed in Duality Cache [11], we estimate the latency of floating-point operations on PipeIMC, as summarized in Table XI. The results indicate that floating-point operations incur higher latency than integer and fixed-point counterparts and will shift such workloads toward compute-bound behavior. In floating-point transformer kernels, compute operations require 37% more cycles than in the fixed-point implementations. Given PipeIMC's strong performance on compute-bound workloads (e.g., matmul and ffn), we expect similar benefits for floating-point applications. Future work will focus on efficient floating-point support and further optimizations.

#### VII. CONCLUSION

This paper proposes PipeIMC, a pipelined in-SRAM computing architecture. We discovered the potential to improve the performance of in-SRAM architectures by pipelining data-independent in-SRAM computing operations. We further identified inefficiencies in the pipelines and adapted explicit register renaming, out-of-order operation scheduling, and a fine-grained issue mechanism to solve these inefficiencies. Evaluation results show that PipeIMC achieves 2.15x to 3.96x and 1.13x to 4.77x utilization, compared to EVE [2] and Duality Cache [11], two state-of-the-art in-SRAM computing architectures. This improvement in utilization yields a performance of 2.17x and 1.68x per area, and an energy efficiency of 1.92x and 1.60x, on average, over EVE [2] and Duality Cache [11] on the Rodinia GPU benchmarks [3]. Our proposed architecture can satisfy the demand for a high-performance, power-efficient in-SRAM computing architecture.

#### ACKNOWLEDGMENT

This work was supported by Beijing Science and Technology Plan Project (Z241100004824002) and National Natural Science Foundation of China (NSFC) under Grant 92373103.

## REFERENCES

- [1] S. Aga, S. Jeloka, A. Subramaniyan, S. Narayanasamy, D. Blaauw, and R. Das, "Compute caches," in *2017 IEEE International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2017, pp. 481–492.
- [2] K. Al-Hawaj, T. Ta, N. Cebry, S. Agwa, O. Afuye, E. Hall, C. Golden, A. B. Apsel, and C. Batten, "Eve: Ephemeral vector engines," in *2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. IEEE, 2023, pp. 691–704.
- [3] S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, S.-H. Lee, and K. Skadron, "Rodinia: A benchmark suite for heterogeneous computing," in *2009 IEEE international symposium on workload characterization (IISWC)*. Ieee, 2009, pp. 44–54.
- [4] M. Cheng, L. Xia, Z. Zhu, Y. Cai, Y. Xie, Y. Wang, and H. Yang, "Time: A training-in-memory architecture for memristor-based deep neural networks," in *Proceedings of the 54th Annual Design Automation Conference 2017*, 2017, pp. 1–6.
- [5] P. Chi, S. Li, C. Xu, T. Zhang, J. Zhao, Y. Liu, Y. Wang, and Y. Xie, "Prime: A novel processing-in-memory architecture for neural network computation in reram-based main memory," *ACM SIGARCH Computer Architecture News*, vol. 44, no. 3, pp. 27–39, 2016.
- [6] C. Eckert, X. Wang, J. Wang, A. Subramaniyan, R. Iyer, D. Sylvester, D. Blaaauw, and R. Das, "Neural cache: Bit-serial in-cache acceleration of deep neural networks," in *2018 ACM/IEEE 45Th annual international symposium on computer architecture (ISCA)*. IEEE, 2018, pp. 383–396.
- [7] S. Fakhoury, A. Naik, G. Sakkas, S. Chakraborty, and S. K. Lahiri, "Llmbased test-driven interactive code generation: User study and empirical evaluation," *IEEE Transactions on Software Engineering*, 2024.
- [8] R. Fan, Y. Cui, Q. Chen, M. Wang, Y. Zhang, W. Zheng, and Z. Li, "Maicc: A lightweight many-core architecture with in-cache computing for multi-dnn parallel inference," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, 2023, pp. 411–423.
- [9] R. Fan, Y. Cui, W. Li, M. Wang, and Z. Li, "Magicache: A virtual in-cache computing engine," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, 2025, pp. 1806–1818.
- [10] T. Finkbeiner, G. Hush, T. Larsen, P. Lea, J. Leidel, and T. Manning, "In-memory intelligence," *IEEE micro*, vol. 37, no. 4, pp. 30–38, 2017.
- [11] D. Fujiki, S. Mahlke, and R. Das, "Duality cache for data parallel acceleration," in *Proceedings of the 46th International Symposium on Computer Architecture*, 2019, pp. 397–410.
- [12] F. Gao, G. Tziantzioulis, and D. Wentzlaff, "Computedram: In-memory compute using off-the-shelf drams," in *Proceedings of the 52nd annual IEEE/ACM international symposium on microarchitecture*, 2019, pp. 100–113.
- [13] Github, "Github copilot," 2023. [Online]. Available: https://github.com/ features/copilot
- [14] Google, "Gemini," 2023. [Online]. Available: https://gemini.google. com/app
- [15] X. Guo, F. M. Bayat, M. Bavandpour, M. Klachko, M. Mahmoodi, M. Prezioso, K. Likharev, and D. Strukov, "Fast, energy-efficient, robust, and reproducible mixed-signal neuromorphic classifier based on embedded nor flash memory technology," in *2017 IEEE International Electron Devices Meeting (IEDM)*. IEEE, 2017, pp. 6–5.
- [16] S. Jain, L. Lin, and M. Alioto, "±cim sram for signed in-memory broadpurpose computing from dsp to neural processing," *IEEE Journal of Solid-State Circuits*, vol. 56, no. 10, pp. 2981–2992, 2021.
- [17] S. Jeloka, N. B. Akesh, D. Sylvester, and D. Blaauw, "A 28 nm configurable memory (tcam/bcam/sram) using push-rule 6t bit cell enabling logic-in-memory," *IEEE Journal of Solid-State Circuits*, vol. 51, no. 4, pp. 1009–1021, 2016.
- [18] Y. S. Kim, J. An, J. B. Jeon, M. W. Son, S. Son, W. Park, Y. Lee, J. Park, G. Y. Kim, G. Kim *et al.*, "Ternary logic with stateful neural networks using a bilayered taox-based memristor exhibiting ternary states," *Advanced Science*, vol. 9, no. 5, p. 2104107, 2022.
- [19] S. Kvatinsky, D. Belousov, S. Liman, G. Satat, N. Wald, E. G. Friedman, A. Kolodny, and U. C. Weiser, "Magic—memristor-aided logic," *IEEE Transactions on Circuits and Systems II: Express Briefs*, vol. 61, no. 11, pp. 895–899, 2014.
- [20] J. Leng, T. Hetherington, A. ElTantawy, S. Gilani, N. S. Kim, T. M. Aamodt, and V. J. Reddi, "Gpuwattch: Enabling energy optimizations in gpgpus," *ACM SIGARCH computer architecture news*, vol. 41, no. 3, pp. 487–498, 2013.

- [21] G. Li, G. Dai, S. Li, Y. Wang, and Y. Xie, "Graphia: An in-situ accelerator for large-scale graph processing," in *Proceedings of the International Symposium on Memory Systems*, 2018, pp. 79–84.
- [22] S. Li, J. H. Ahn, R. D. Strong, J. B. Brockman, D. M. Tullsen, and N. P. Jouppi, "Mcpat: An integrated power, area, and timing modeling framework for multicore and manycore architectures," in *Proceedings of the 42nd annual ieee/acm international symposium on microarchitecture*, 2009, pp. 469–480.
- [23] S. Li, X. Ning, L. Wang, T. Liu, X. Shi, S. Yan, G. Dai, H. Yang, and Y. Wang, "Evaluating quantized large language models," in *Proceedings of the 41st International Conference on Machine Learning*, ser. ICML'24. JMLR.org, 2024.
- [24] S. Li, D. Niu, K. T. Malladi, H. Zheng, B. Brennan, and Y. Xie, "Drisa: A dram-based reconfigurable in-situ accelerator," in *Proceedings of the 50th annual ieee/acm international symposium on microarchitecture*, 2017, pp. 288–301.
- [25] Y.-Y. Lin, F.-M. Lee, M.-H. Lee, W.-C. Chen, H.-L. Lung, K.-C. Wang, and C.-Y. Lu, "A novel voltage-accumulation vector-matrix multiplication architecture using resistor-shunted floating gate flash memory device for low-power and high-density neural network applications," in *2018 IEEE International Electron Devices Meeting (IEDM)*. IEEE, 2018, pp. 2–4.
- [26] Z. Lin, Z. Tong, F. Wang, J. Zhang, Y. Zhao, P. Sun, T. Xu, C. Zhang, X. Li, X. Wu *et al.*, "In situ storing 8t sram-cim macro for fullarray boolean logic and copy operations," *IEEE Journal of Solid-State Circuits*, vol. 58, no. 5, pp. 1472–1486, 2022.
- [27] F. Merrikh-Bayat, X. Guo, M. Klachko, M. Prezioso, K. K. Likharev, and D. B. Strukov, "High-performance mixed-signal neurocomputing with nanoscale floating-gate memory cell arrays," *IEEE transactions on neural networks and learning systems*, vol. 29, no. 10, pp. 4782–4790, 2017.
- [28] Y. Moslem, R. Haque, J. D. Kelleher, and A. Way, "Adaptive machine translation with large language models," *arXiv preprint arXiv:2301.13294*, 2023.
- [29] A. Munshi, "The opencl specification," in *2009 IEEE Hot Chips 21 Symposium (HCS)*. IEEE, 2009, pp. 1–314.
- [30] OpenAI, "Chatgpt," 2023. [Online]. Available: https://openai.com/ chatgpt/overview/
- [31] A. Sebastian, M. Le Gallo, R. Khaddam-Aljameh, and E. Eleftheriou, "Memory devices and applications for in-memory computing," *Nature nanotechnology*, vol. 15, no. 7, pp. 529–544, 2020.
- [32] V. Seshadri, D. Lee, T. Mullins, H. Hassan, A. Boroumand, J. Kim, M. A. Kozuch, O. Mutlu, P. B. Gibbons, and T. C. Mowry, "Ambit: Inmemory accelerator for bulk bitwise operations using commodity dram technology," in *Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture*, 2017, pp. 273–287.
- [33] W. A. Simon, Y. M. Qureshi, M. Rios, A. Levisse, M. Zapater, and D. Atienza, "Blade: An in-cache computing architecture for edge devices," *IEEE Transactions on Computers*, vol. 69, no. 9, pp. 1349– 1363, 2020.
- [34] Z. Sun, E. Ambrosi, A. Bricalli, and D. Ielmini, "Logic computing with stateful neural networks of resistive switches," *Advanced Materials*, vol. 30, no. 38, p. 1802554, 2018.
- [35] B. Tine, K. P. Yalamarthy, F. Elsabbagh, and K. Hyesoon, "Vortex: Extending the risc-v isa for gpgpu and 3d-graphics," in *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*, 2021, pp. 754–766.
- [36] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.
- [37] N. Verma, H. Jia, H. Valavi, Y. Tang, M. Ozatay, L.-Y. Chen, B. Zhang, and P. Deaville, "In-memory computing: Advances and prospects," *IEEE solid-state circuits magazine*, vol. 11, no. 3, pp. 43–55, 2019.
- [38] Vicuna Team, "Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality," 2023.
- [39] L. Wang, C. Lyu, T. Ji, Z. Zhang, D. Yu, S. Shi, and Z. Tu, "Documentlevel machine translation with large language models," *arXiv preprint arXiv:2304.02210*, 2023.