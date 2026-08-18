# F. Area and Power Overheads

The only die-area overhead for Sigries comes from the counters implemented in the memory controller. Due to their compact design, the counters occupy less than 10% of the memory controller's area. The power overhead is similarly minimal. The counters are built using low-power flip-flops, resulting in low power consumption.

We used these numbers for a quick estimate of the counter storage required for a full Misra–Gries implementation. It would inflate the memory controller by roughly  $40\times$ , with the defense logic alone accounting for 96% of die-area. These numbers are hypothetical because a CAM with the required width cannot be built and operated at DDR5 speeds.

**Takeaway:** The SRAM Sigries uses for its counter tables accounts for less than 10% of the memory controller's total area.

<span id="page-12-9"></span>

| Rowhammer<br>Defense | DRAM bandwidth overhead | Stalls   |
|----------------------|-------------------------|----------|
| BlockHammer          | <b>√</b>                | ×        |
| DREAM-R              | ×                       | 1        |
| Hydra                | ✓                       | ×        |
| PARA/PRA             | ×                       | 1        |
| RRS                  | ✓                       | ×        |
| Graphene             | <b>√</b>                | <b>V</b> |
| PRAC                 | ✓                       | 1        |
| Sigries              | ✓                       | /        |

TABLE VI: Summary of performance comparison to prior defenses. Graphene and PRAC are the only two prior defenses that have performance comparable to Sigries. Graphene uses a configuration so large it is impractical, and PRAC is an optional DDR5 Rowhammer defense not yet implemented in silicon.

#### VII. CONCLUSIONS

This paper presents Sigries, a production-level Row-hammer defense for cloud SoCs. Sigries combines an under-provisioned version of Misra-Gries with row-sampling to achieve a balanced trade-off between performance, cost, and security that meets practical industry requirements. The paper also presents the results of verifying the algorithm, configuring it, and designing the telemetry to meet confidential computing requirements. Our detailed evaluation demonstrates that Sigries maintains minimal DRAM bandwidth overhead and ensures robust performance. We hope Sigries will guide future research in Rowhammer protection schemes to consider our design requirements.

#### ACKNOWLEDGMENT

We would like to thank everyone who contributed to Sigries from its inception to turning it into a production-ready security feature including: Selim Bilgin, Michael Borgens, Tim Cowles, Brett Dodds, Adam Grenzebach, Rob Huryn, Scott Kivitz, Darren Lasko, Arthur Leung, Maurice Ma, Dorica Munell, Phyllis Ng, Charles Patisaul, Andy Rushing, Vignesh Saravanan, Brian Sutton, John Tien, and Kushagra Vaid. We are also grateful to Victor Bahl for his constant encouragement and support. Finally, we would like to thank the anonymous reviewers for their feedback on the submission.

## REFERENCES

- <span id="page-12-1"></span> Amazon, "AWS Graviton Processors," https://aws.amazon.com/ec2/ graviton/, 2025.
- <span id="page-12-4"></span>[2] Anonymized, "Internal Communication," 2025.
- <span id="page-12-6"></span>[3] Z. B. Aweke, S. F. Yitbarek, R. Qiao, R. Das, M. Hicks, Y. ORren, and T. Austin, "ANVIL: Software-Based Protection Against Next-Generation Rowhammer Attacks," in ASPLOS, 2016.
- <span id="page-12-0"></span>[4] S. Baek, M. Wi, S. Park, H. Nam, M. J. Kim, N. S. Kim, and J. Ahn, "Marionette: A rowhammer attack via row coupling," in ASPLOS, 2024.
- <span id="page-12-2"></span>[5] F. Bellard, "Qemu, a fast and portable dynamic translator," in USENIX ATC, FREENIX Track, 2005.
- <span id="page-12-3"></span>[6] T. Bennett, S. Saroiu, A. Wolman, and L. Cojocar, "Panopticon: A Complete In-DRAM Rowhammer Mitigation," in *DRAMSec*, 2021.
- <span id="page-12-7"></span>[7] C. Bienia, S. Kumar, J. P. Singh, and K. Li, "The PARSEC Benchmark Suite: Characterization and Architectural Implications," in *PACT*, 2008.
- <span id="page-12-5"></span>[8] C. Bock, F. Brasser, D. Gens, C. Liebchen, and A.-R. Sadeghi, "RIP-RH: Preventing Rowhammer-Based Inter-Process Attacks," in ASIA-CCS, 2019.

- <span id="page-13-42"></span>[9] F. N. Bostanci, I. E. Yuksel, A. Olgun, K. Kanellopoulos, Y. C. Tugrul, A. G. Yaglikci, M. Sadrosadati, and O. Mutlu, "CoMeT: Count-Min-Sketch-based Row Tracking to Mitigate RowHammer at Low Cost," in *HPCA*, 2024.
- <span id="page-13-36"></span>[10] F. Brasser, L. Davi, D. Gens, C. Liebchen, and A.-R. Sadeghi, "CAn't Touch This: Practical and Generic Software-only Defenses Against RowHammer Attacks," in *USENIX Security*, 2017.
- <span id="page-13-43"></span>[11] O. Canpolat, A. G. Yaglıkc¸ı, A. Olgun, I. E. Yuksel, Y. C. Tu ¨ grul, ˘ K. Kanellopoulos, O. Ergin, and O. Mutlu, "BlockHammer: Preventing RowHammer at Low Cost by Blacklisting Rapidly-Accessed DRAM Rows," in *MICRO*, 2024.
- <span id="page-13-40"></span>[12] A. Chakraborty, M. Alam, and D. Mukhopadhyay, "Deep Learning based Diagnostics for Rowhammer Protection of DRAM Chips," in *ATS*, 2019.
- <span id="page-13-45"></span>[13] CMU-SAFARI, "Ramulator Source Code," [https://github](https://github.com/CMU-SAFARI/ramulator).com/CMU-[SAFARI/ramulator,](https://github.com/CMU-SAFARI/ramulator) 2025.
- <span id="page-13-22"></span>[14] L. Cojocar, K. Razavi, C. Giuffrida, and H. Bos, "Exploiting Correcting Codes: On the Effectiveness of ECC Memory Against Rowhammer Attacks," in *IEEE S&P*, 2019.
- <span id="page-13-19"></span>[15] J. Emer, P. Ahuja, E. Borch, A. Klauser, C.-K. Luk, S. Manne, S. S. Mukherjee, H. Patil, S. Wallace, N. Binkert, R. Espasa, and T. Juan, "Asim: A performance model framework," *IEEE Computer*, vol. 35, no. 2, pp. 68–76, 2002.
- <span id="page-13-26"></span>[16] A. Fakhrzadehgan, Y. N. Patt, P. J. Nair, and M. K. Qureshi, "SafeGuard: Reducing the Security Risk from Row-Hammer via Low-Cost Integrity Protection," in *HPCA*, 2022.
- <span id="page-13-3"></span>[17] P. Frigo, E. Vannacci, H. Hassan, V. van der Veen, O. Mutlu, C. Giuffrida, H. Bos, and K. Razavi, "TRRespass: Exploiting the Many Sides of Target Row Refresh," in *S&P*, 2020.
- <span id="page-13-27"></span>[18] S. K. Gautam, S. K. Manhas, A. Kumar, M. Pakala, and E. Yieh, "Row Hammering Mitigation Using Metal Nanowire in Saddle Fin DRAM," *IEEE T-ED*, vol. 66, 2019.
- [19] M. Ghasempour, M. Lujan, and J. Garside, "ARMOR: A Run-Time Memory Hot-Row Detector," http://apt.cs.manchester.ac.[uk/projects/](http://apt.cs.manchester.ac.uk/projects/ARMOR/RowHammer/armor.html) [ARMOR/RowHammer/armor](http://apt.cs.manchester.ac.uk/projects/ARMOR/RowHammer/armor.html).html, 2015.
- <span id="page-13-28"></span>[20] H. Gomez, A. Amaya, and E. Roa, "DRAM Row-hammer Attack Reduction using Dummy Cells," in *NORCAS*, 2016.
- <span id="page-13-24"></span>[21] Google, "Introducing Half-Double: New hammering technique for DRAM Rowhammer bug," [https://security](https://security.googleblog.com/2021/05/introducing-half-double-new-hammering.html).googleblog.com/2021/05/ [introducing-half-double-new-hammering](https://security.googleblog.com/2021/05/introducing-half-double-new-hammering.html).html, 2021.
- <span id="page-13-12"></span>[22] ——, "Introducing Google Axion Processors, our new Arm-based CPUs," https://cloud.google.[com/blog/products/compute/introducing](https://cloud.google.com/blog/products/compute/introducing-googles-new-arm-based-cpu)[googles-new-arm-based-cpu,](https://cloud.google.com/blog/products/compute/introducing-googles-new-arm-based-cpu) 2024.
- <span id="page-13-6"></span>[23] M. Greenberg, "Row Hammering: What it is, and how hackers could use it to gain access to your system," [https://blogs](https://blogs.synopsys.com/committedtomemory/2015/03/09/row-hammering-what-it-is-and-how-hackers-could-use-it-to-gain-access-to-your-system/).synopsys.com/ [committedtomemory/2015/03/09/row-hammering-what-it-is-and-how](https://blogs.synopsys.com/committedtomemory/2015/03/09/row-hammering-what-it-is-and-how-hackers-could-use-it-to-gain-access-to-your-system/)[hackers-could-use-it-to-gain-access-to-your-system/,](https://blogs.synopsys.com/committedtomemory/2015/03/09/row-hammering-what-it-is-and-how-hackers-could-use-it-to-gain-access-to-your-system/) 2015.
- <span id="page-13-29"></span>[24] Z. Greenfield, J. B. Halbert, and K. S. Bains, "Method, apparatus and system for determining a count of accesses to a row of memory," Patent No. US 2014/0085995, 2014.
- <span id="page-13-15"></span>[25] S. Hong, D. Kim, J. Lee, R. Oh, C. Yoo, S. Hwang, and J. Lee, "DSAC: Low-Cost Rowhammer Mitigation Using In-DRAM Stochastic and Approximate Counting Algorithm," arXiv preprint arXiv:2302.03591, 2023.
- <span id="page-13-30"></span>[26] A. Jaleel, G. Saileshwar, S. W. Keckler, and M. Qureshi, "PrIDE: Achieving Secure Rowhammer Mitigation with Low-Cost In-DRAM Trackers," in *ISCA*, 2024.
- <span id="page-13-4"></span>[27] P. Jattke, V. Veen, P. Frigo, S. Gunter, and K. Razavi, "Blacksmith: Scalable Rowhammering in the Frequency Domain," in *IEEE S&P*, 2022.
- <span id="page-13-5"></span>[28] P. Jattke, M. Wipfli, F. Solt, M. Marazzi, M. Bolcskei, and K. Razavi, ¨ "ZenHammer: Rowhammer Attacks on AMD Zen-based Platforms," in *USENIX Security*, 2024.
- <span id="page-13-8"></span>[29] JEDEC, *Near-Term DRAM Level Rowhammer Mitigation (JEP300-1)*, 2021.
- <span id="page-13-9"></span>[30] ——, *System Level Rowhammer Mitigation (JEP301-1)*, 2021.
- <span id="page-13-20"></span>[31] ——, *Double Data Rate 5 (DDR5) SDRAM Standard Version 1.31 JESD79-5C.01*, 2024.
- <span id="page-13-21"></span>[32] ——, *Low Power Double Data Rate 6 (LPDDR6) SDRAM Standard*, 2025.
- <span id="page-13-18"></span>[33] M. Kaczmarski, "Thoughts on Intel Xeon E5-2600 v2 Product Performance Optimisation," 2014.
- <span id="page-13-23"></span>[34] N. Kamadan, W. Wang, S. van Schaik, C. Garman, D. Genkin, and R. U. B. Yuval Yarom, "ECC.fail: Mounting Rowhammer Attacks on DDR4 Servers with ECC Memory," in *USENIX Security*, 2025.

- <span id="page-13-17"></span>[35] D.-H. Kim, P. J. Nair, and M. K. Qureshi, "Architectural Support for Mitigating Row Hammering in DRAM Memories," *CAL*, vol. 14, pp. 9–12, 2015.
- <span id="page-13-1"></span>[36] J. Kim, M. Patel, A. G. Yaglikci, H. Hassan, R. Azizi, L. Orosa, and O. Mutlu, "Revisiting RowHammer: An Experimental Analysis of Modern DRAM Devices and Mitigation Techniques," in *ISCA*, 2020.
- <span id="page-13-25"></span>[37] M. J. Kim, S. Baek, J. Kim, H. Nam, N. S. Kim, and J. Ahn, "SoK: Systematizing a Decade of Architectural RowHammer Defenses Through the Lens of Streaming Algorithms," in *IEEE S&P*, 2025.
- <span id="page-13-31"></span>[38] M. J. Kim, J. Park, Y. Park, W. Doh, N. Kim, T. J. Ham, J. W. Less, and J. Ahn, "Mithril: Cooperative Row Hammer Protection on Commodity DRAM Leveraging Managed Refresh," in *HPCA*, 2022.
- <span id="page-13-32"></span>[39] M. Kim, J. Choi, H. Kim, and H.-J. Lee, "An Effective DRAM Address Remapping for Mitigating Rowhammer Errors," in *TC*, 2019.
- <span id="page-13-16"></span>[40] W. Kim, C. Jung, S. Yoo, D. Hong, J. Hwang, J. Yoon, O. Jung, J. Choi, S. Hyun, M. Kang, S. Lee, D. Kim, S. Ku, D. Choi, N. Joo, S. Yoon, J. Noh, B. Go, C. Kim, S. Hwang, M. Hwang, S.-M. Yi, H. Kim, S. Heo, Y. Jang, K. Jang, S. Chu, Y. Oh, K. Kim, J. Kim, S. Kim, J. Hwang, S. Park, J. Lee, I. J. an Joohwan Cho, and J. Kim, "A 1.1V 16Gb DDR5 DRAM with Probabilistic-Aggressor Tracking, Refresh-Management Functionality, Per-Row Hammer Tracking, a Multi-Step Precharge, and Core-Bias Modulation for Security and Reliability Enhancement," in *ISSCC*, 2023.
- <span id="page-13-0"></span>[41] Y. Kim, R. Daly, J. Kim, C. Fallin, J. H. Lee, D. Lee, C. Wilkerson, K. Lai, and O. Mutlu, "Flipping Bits in Memory Without Accessing Them: An Experimental Study of DRAM Disturbance Errors," in *ISCA*, 2014.
- <span id="page-13-46"></span>[42] Y. Kim, W. Yang, and O. Mutlu, "Ramulator: A Fast and Extensible DRAM Simulator," in *CAL*, 2016.
- <span id="page-13-37"></span>[43] R. K. Konoth, M. Oliverio, A. Tatar, D. Andriesse, H. Bos, C. Giuffrida, and K. Razavi, "ZebRAM: Comprehensive and Compatible Software Protection Against Rowhammer Attacks," in *OSDI*, 2018.
- <span id="page-13-7"></span>[44] M. Lanteigne, "How Rowhammer Could be Used to Exploit Weaknesses in Computer Hardware," http://www.thirdio.[com/rowhammer](http://www.thirdio.com/rowhammer.pdf).pdf, 2016.
- <span id="page-13-33"></span>[45] E. Lee, S. Lee, G. E. Suh, and J. Ahn, "TWiCe: Time Window Counter Based Row Refresh to Prevent Row-Hammering," *CAL*, vol. 17, pp. 96–99, 2018.
- <span id="page-13-34"></span>[46] ——, "TWiCe: Preventing Row-hammering by Exploiting Time Window Counters," in *ISCA*, 2019.
- <span id="page-13-44"></span>[47] K. R. M. Leino, "Dafny: An automatic program verifier for functional correctness," in *Proceedings of the Conference on Logic for Programming, Artificial Intelligence, and Reasoning (LPAR)*, 2010, pp. 348–370.
- <span id="page-13-41"></span>[48] C. Li and J.-L. Gaudiot, "Detecting Malicious Attacks Exploiting Hardware Vulnerabilities Using Performance Counters," in *COMPSAC*, 2019.
- <span id="page-13-47"></span>[49] S. Li, Z. Yang, D. Reddy, A. Srivastava, and B. Jacob, "Dramsim3: A cycle-accurate, thermal-capable dram simulator," vol. 19, no. 2, p. 106–109, 2020.
- <span id="page-13-10"></span>[50] Linux Documentation, "sysfs-memory-page-offline," [https:](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-memory-page-offline) //www.kernel.[org/doc/Documentation/ABI/testing/sysfs-memory](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-memory-page-offline)[page-offline,](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-memory-page-offline) 2009.
- <span id="page-13-38"></span>[51] K. Loughlin, J. Rosenblum, S. Saroiu, A. Wolman, D. Skarlatos, and B. Kasikci, "Siloz: Leveraging DRAM Isolation Domains to Prevent Inter-VM Rowhammer," in *SOSP*, 2023.
- <span id="page-13-39"></span>[52] K. Loughlin, S. Saroiu, A. Wolman, and B. Kasikci, "Stop! Hammer Time: Rethinking Our Approach To Rowhammer Mitigations," in *HotOS*, 2021.
- <span id="page-13-2"></span>[53] K. Loughlin, S. Saroiu, A. Wolman, Y. A. Manerkar, and B. Kasikci, "MOESI-prime: Preventing Coherence-Induced Hammering in Commodity Workloads," in *ISCA*, 2022.
- <span id="page-13-11"></span>[54] H. Luo, A. Olgun, A. G. Yaglıkc¸ı, Y. C. Tu ˘ grul, S. Rhyner, M. B. Cavlak, ˘ J. Lindegger, M. Sadrosadati, and O. Mutlu, "Rowpress: Amplifying read disturbance in modern dram chips," in *ISCA*, 2023.
- <span id="page-13-35"></span>[55] M. Marazzi, P. Jattke, F. Solt, and K. Razavi, "PROTRR: Principled yet Optimal In-DRAM Target Row Refresh," in *IEEE S&P*, 2022.
- <span id="page-13-14"></span>[56] M. Marazzi, F. Solt, P. Jattke, K. Takashi, and K. Razavi, "REGA: Scalable Rowhammer Mitigation with Refresh-Generating Activations," in *IEEE S&P*, 2023.
- <span id="page-13-13"></span>[57] Microsoft, "With a systems approach to chips, Microsoft aims to tailor everything 'from silicon to service' to meet AI demand," https://news.microsoft.[com/source/features/ai/in-house-chips-silicon-to](https://news.microsoft.com/source/features/ai/in-house-chips-silicon-to-service-to-meet-ai-demand/)[service-to-meet-ai-demand/,](https://news.microsoft.com/source/features/ai/in-house-chips-silicon-to-service-to-meet-ai-demand/) 2023.

- <span id="page-14-2"></span>[58] Microsoft Learn, "Predictive Failure Analysis (PFA)," https://learn.microsoft.[com/en-us/windows-hardware/drivers/whea/](https://learn.microsoft.com/en-us/windows-hardware/drivers/whea/predictive-failure-analysis--pfa-) [predictive-failure-analysis--pfa-,](https://learn.microsoft.com/en-us/windows-hardware/drivers/whea/predictive-failure-analysis--pfa-) 2023.
- <span id="page-14-3"></span>[59] J. Misra and D. Gries, "Finding repeated elements," *Science of Computer Programming*, vol. 2, no. 2, pp. 143–152, 1982.
- <span id="page-14-24"></span>[60] O. Mutlu, "The RowHammer Problem and Other Issues We May Face as Memory Becomes Denser," in *DATE*, 2017.
- <span id="page-14-31"></span>[61] A. Olgun, Y. C. Tugrul, N. Bostanci, ˘ ˙Ismail Emir Yuksel, H. Luo, ¨ S. R. A. G. Yaglıkc¸ı, G. F. Oliveira, and O. Mutlu, "ABACuS: All-Bank Activation Counters for Scalable and Low Overhead RowHammer Mitigation," in *USENIX Security*, 2024.
- <span id="page-14-4"></span>[62] Y. Park, W. Kwon, E. Lee, T. J. Han, J. Ahn, and J. W. Lee, "Graphene: Strong yet Lightweight Row Hammer Protection," in *MICRO*, 2020.
- <span id="page-14-6"></span>[63] M. Qureshi, "AutoRFM: Scaling Low-Cost In-DRAM Trackers to Ultra-Low Rowhammer Thresholds," in *HPCA*, 2025.
- [64] ——, "SALT: Track-and-Mitigate Subarrays, Not Rows, for Blast-Radius-Free Rowhammer Defense," in *HPCA*, 2026.
- [65] M. Qureshi and S. Q. Jaleel, "MOAT: Securely Mitigating Rowhammer with Per-Row Activation Counters," in *ASPLOS*, 2025.
- <span id="page-14-7"></span>[66] M. Qureshi, S. Qazi, and A. Jaleel, "MINT: Securely Mitigating Rowhammer with a Minimalist In-DRAM Tracker," in *MICRO*, 2024.
- <span id="page-14-29"></span>[67] M. Qureshi, A. Rohan, G. Saileshwar, and P. J. Nair, "Hydra: Enabling Low-Overhead Mitigation of Row-Hammer at Ultra-Low Thresholds via Hybrid Tracking," in *ISCA*, 2022.
- <span id="page-14-1"></span>[68] F. Ridder, P. Frigo, E. Vannacci, H. Bos, C. Giuffrida, and K. Razavi, "SMASH: Synchronized Many-sided Rowhammer Attacks from JavaScript," in *USENIX Security*, 2021.
- <span id="page-14-34"></span>[69] P. Rosenfeld, E. Cooper-Balis, and B. Jacob, "Dramsim2: A cycle accurate memory system simulator," vol. 10, no. 1, p. 16–19, 2011.
- <span id="page-14-8"></span>[70] S. Ryu, K. Min, J. Shin, H. Kwon, D. Nam, T. Oh, T. Jang, M. Yoo, Y. Kim, and S. Hong, "Overcoming the reliability limitation in the ultimately scaled DRAM using silicon migration technique by hydrogen annealing," in *IEEE IEDM*, 2017.
- <span id="page-14-25"></span>[71] G. Saleshwar, B. Wang, M. Qureshi, and P. J. Nair, "Randomized Row-Swap: Mitigating Row Hammer by Breaking Spatial Correlation between Aggressor and Victim Rows," in *ASPLOS*, 2022.
- <span id="page-14-33"></span>[72] S. Saroiu and A. Wolman, "How to Configure Row-Sampling-Based Rowhammer Defenses," in *DRAMSec*, 2022.
- <span id="page-14-0"></span>[73] S. Saroiu, A. Wolman, and L. Cojocar, "The Price of Secrecy: How Hiding Internal DRAM Topologies Hurts Rowhammer Defenses," in *IPRS*, 2022.
- <span id="page-14-27"></span>[74] A. Saxena and M. Qureshi, "START: Scalable Tracking for Any Rowhammer Threshold," in *HPCA*, 2024.
- <span id="page-14-30"></span>[75] A. Saxena, G. Saileshwar, P. J. Nair, and M. Qureshi, "AQUA: Scalable Rowhammer Mitigation by Quarantining Aggressor Rows at Runtime," in *Micro*, 2022.
- <span id="page-14-20"></span>[76] A. Saxena, W. Wang, and A. Daglis, "Citadel: Rethinking Memory Allocations to Safeguard Against Inter-Domain Rowhammer Exploits," 2025.
- <span id="page-14-9"></span>[77] S. M. Seyedzadeh, A. K. Jones, and R. Melhem, "Counter-based Tree Structure for Row Hammering Mitigation in DRAM," *CAL*, vol. 16, pp. 18–21, 2017.
- [78] ——, "Mitigating Wordline Crosstalk Using Adaptive Trees of Counters," in *ISCA*, 2018.
- <span id="page-14-10"></span>[79] M. Son, H. Park, J. Ahn, and S. Yoo, "Making DRAM Stronger Against Row Hammering," in *DAC*, 2017.
- <span id="page-14-36"></span>[80] Standard Performance Evaluation Corporation, "SPEC CPU 2017 Benchmark Suite," https://www.spec.[org/cpu2017/,](https://www.spec.org/cpu2017/) 2017.
- <span id="page-14-11"></span>[81] H. Taneja, A. Hajiabadi, M. Marazzi, K. Razavi, and M. Qureshi, "MIRZA: Efficiently Mitigating Rowhammer with Randomization and ALERT," in *HPCA*, 2026.
- <span id="page-14-32"></span>[82] H. Taneja and M. Qureshi, "DREAM: Enabling Low-Overhead Rowhammer Mitigation via Directed Refresh Management," in *ISCA*, 2025.
- <span id="page-14-21"></span>[83] V. Van Der Veen, M. Lindorfer, Y. Fratantonio, H. P. Pillai, G. Vigna, C. Kruegel, H. Bos, and K. Razavi, "GuardION: Practical Mitigation of DMA-Based Rowhammer Attacks on ARM," in *DIMVA*, 2018.
- <span id="page-14-12"></span>[84] S. Vittal, S. Qazi, P. Das, and M. Qureshi, "MoPAC: Efficiently Mitigating Rowhammer with Probabilistic Activation Counting," in *ISCA*, 2025.
- <span id="page-14-13"></span>[85] A. J. Walker, S. Lee, and D. Beery, "On DRAM Rowhammer and the Physics of Insecurity," *IEEE T-ED*, vol. 68, 2021.

- <span id="page-14-35"></span>[86] D. Wang, B. Ganesh, N. Tuaycharoen, K. Baynes, A. Jaleel, and B. Jacob, "DRAMsim: a memory system simulator," *SIGARCH Comput. Archit. News*, vol. 33, no. 4, p. 100–107, 2005.
- <span id="page-14-19"></span>[87] Y. Wang, Y. Liu, P. Wu, and Z. Zhang, "Detect DRAM Disturbance Error by Using Disturbance Bin Counters," in *CAL*, 2019.
- <span id="page-14-14"></span>[88] ——, "Reinforce Memory Error Protection by Breaking DRAM Disturbance Correlation Within ECC Words," in *ICCD*, 2019.
- <span id="page-14-23"></span>[89] ——, "Discreet-PARA: Rowhammer Defense with Low Cost and High Efficiency," in *ICCD*, 2021.
- <span id="page-14-15"></span>[90] M. Wi, J. Park, S. Ko, M. J. Kim, N. S. Kim, E. Lee, and J. Ahn, "SHADOW: Preventing Row Hammer in DRAM with Intra-Subarray Row Shuffling," in *HPCA*, 2023.
- [91] M. Wi, Y. Yoo, Y. Kim, J. Shin, J. Kim, Y. Ryu, S. Gorgin, J. Ahn, and J. Kim, "RowArmor: Efficient and Comprehensive Protection Against DRAM Disturbance Attacks," in *ASPLOS*, 2026.
- <span id="page-14-16"></span>[92] J. Woo, S. C. Lin, P. J. Nair, A. Jaleel, and G. Saileshwar, "QPRAC: Towards Secure and Practical PRAC-based Rowhammer Mitigation using Priority Queues," in *HPCA*, 2024.
- <span id="page-14-28"></span>[93] J. Woo, G. Saileshwar, and P. J. Nair, "Scalable and Secure Row-Swap: Efficient and Safe Row Hammer Mitigation in Memory Systems," in *HPCA*, 2023.
- <span id="page-14-22"></span>[94] X.-C. Wu, T. Sherwood, F. T. Chong, and Y. Li, "Protecting Page Tables from RowHammer Attacks using Monotonic Pointers in DRAM True-Cells," in *ASPLOS*, 2019.
- <span id="page-14-26"></span>[95] A. G. Yaglıkc¸ı, M. Patel, J. S. Kim, R. Azizi, A. Olgun, L. Orosa, H. Hassan, J. Park, K. Kanellopoulos, T. Shahroodi, S. Ghose, and O. Mutlu, "BlockHammer: Preventing RowHammer at Low Cost by Blacklisting Rapidly-Accessed DRAM Rows," in *HPCA*, 2021.
- <span id="page-14-17"></span>[96] C.-M. Yang, C.-K. Wei, Y. J. Chang, T.-C. Wu, H.-P. Chen, and C.-S. Lai, "Suppression of Row Hammer Effect by Doping Profile Modification in Saddle-Fin Array Devices for Sub-30-nm DRAM Technology," *IEEE T-DMR*, vol. 16, 2016.
- <span id="page-14-18"></span>[97] J. M. You and J.-S. Yang, "MRLoc: Mitigating Row-hammering based on memory Locality," in *DAC*, 2019.
- <span id="page-14-5"></span>[98] I. E. Yuksel, A. Olgun, N. Bostanci, H. Luo, A. G. Yaglikci, and O. Mutlu, "ColumnDisturb: Understanding Column-based Read Disturbance in Real DRAM Chips and Implications for Future Systems," in *MICRO*, 2025.