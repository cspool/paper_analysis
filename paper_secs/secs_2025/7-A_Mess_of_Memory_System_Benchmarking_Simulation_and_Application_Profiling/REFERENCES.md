# REFERENCES

- [1] W. A. Wulf and S. A. McKee. Hitting the memory wall: Implications of the obvious. In *SIGARCH*, pages 20–24. Comput. Archit. News, March 1995.
- [2] Richard Sites. It's the memory, stupid! *Microprocessor Report*, pages 2–3, August 1996.
- [3] A. Saulsbury, F. Pong, and A. Nowatzyk. Missing the memory wall: The case for processor/memory integration. In *Proceedings of the 23rd Annual International Symposium on Computer Architecture, ISCA*, pages pages 90–101, New York, 1996. ACM.
- [4] Intel. Intel Advisor. https://www.intel.com/content/www/us/en/ developer/tools/oneapi/advisor.html, 2021. [Online; accessed 27-June-2023].
- [5] John D McCalpin. Memory bandwidth and machine balance in current high performance computers. *IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter*, 1995.

#### DRAMsim3

```
measurement_rdRatio_Pause →raw simulation
results
   dramsim3.json →results in json format
   dramsim3.txt →results in txt format
   dramsim3epoch.json →result per time epoch
   output_jobID.err →simulation error print
   output_jobID.out →simulation output print
   submit.batch →run a single experiment
DRAMsim3_mn5 →DRAMsim3 simulator directory
traceInput →input trace used in our
experiments
main.py →parser of raw simulation results
results.csv →final processed outputs
results_original.csv →original data to
validate against
runner.sh →the workflow to generate all raw
simulation results
submit.batch →template to generate a single
simulation result
replicate.sh →the main bash file to replicate
Figure 6.b
convert.py →generate bandwidth--latency
curves.
output.pdf →final bandwidth--latency curves.
```

Fig. 19: Directory structure to replicate DRAMsim3 experiments.

- [6] Wenqing Peng and Evgenij Belikov. Camp: a synthetic microbenchmark for assessing deep memory hierarchies. In *2022 IEEE/ACM International Workshop on Hierarchical Parallelism for Exascale Computing (HiPar)*, 2022.
- [7] j. Dongarra, M. Heroux, and P. Luszczek. "The HPCG Benchmark," http://www.hpcg-benchmark.org, 2016.
- [8] LMbench. http://lmbench.sourceforge.net, 12 2005.
- [9] Google. Multichase. https://github.com/google/multichase, 2021.
- [10] R. S. Verdejo and P. Radojkovic. Microbenchmarks for Detailed Validation and Tuning of Hardware Simulators. In *International Conference on High Performance Computing Simulation (HPCS)*, 2017.
- [11] Intel Corporation. Intel memory latency checker v3.5. https://software. intel.com/en-us/articles/intelr-memory-latency-checker, 2023.
- [12] Mark Gottscho, Sriram Govindan, Bikash Sharma, Mohammed Shoaib, and Puneet Gupta. X-mem: A cross-platform and extensible memory characterization tool for the cloud. In *2016 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 2016.
- [13] Trevor E. Carlson, Wim Heirman, Stijn Eyerman, Ibrahim Hur, and Lieven Eeckhout. An Evaluation of High-Level Mechanistic Core Models. *ACM Trans. Archit. Code Optim.*, 11(3), August 2014.
- [14] Trevor E. Carlson, Wim Heirman, and Lieven Eeckhout. Sniper: Exploring the level of abstraction for scalable and accurate parallel multi-core simulation. In *SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis*, pages 1–12, November 2011.
- [15] Daniel Sanchez and Christos Kozyrakis. ZSim: fast and accurate microarchitectural simulation of thousand-core systems. In *ISCA' 13 Proceedings of the 40th Annual International Symposium on Computer Architecture*, pages 475–486, 2013.
- [16] Jason Lowe-Power, Abdul Mutaal Ahmad, Ayaz Akram, Mohammad Alian, Rico Amslinger, Matteo Andreozzi, Adria Armejach, ` Nils Asmussen, Brad Beckmann, Srikant Bharadwaj, Gabe Black, Gedare Bloom, Bobby R. Bruce, Daniel Rodrigues Carvalho, Jeronimo Castrillon, Lizhong Chen, Nicolas Derumigny, Stephan Diestelhorst,

- Wendy Elsasser, Carlos Escuin, Marjan Fariborz, Amin Farmahini-Farahani, Pouya Fotouhi, Ryan Gambord, Jayneel Gandhi, Dibakar Gope, Thomas Grass, Anthony Gutierrez, Bagus Hanindhito, Andreas Hansson, Swapnil Haria, Austin Harris, Timothy Hayes, Adrian Herrera, Matthew Horsnell, Syed Ali Raza Jafri, Radhika Jagtap, Hanhwi Jang, Reiley Jeyapaul, Timothy M. Jones, Matthias Jung, Subash Kannoth, Hamidreza Khaleghzadeh, Yuetsu Kodama, Tushar Krishna, Tommaso Marinelli, Christian Menard, Andrea Mondelli, Miquel Moreto, Tiago Muck, Omar Naji, Krishnendra Nathella, Hoa Nguyen, Nikos Nikoleris, ¨ Lena E. Olson, Marc Orr, Binh Pham, Pablo Prieto, Trivikram Reddy, Alec Roelke, Mahyar Samani, Andreas Sandberg, Javier Setoain, Boris Shingarov, Matthew D. Sinclair, Tuan Ta, Rahul Thakur, Giacomo Travaglini, Michael Upton, Nilay Vaish, Ilias Vougioukas, William Wang, Zhengrong Wang, Norbert Wehn, Christian Weis, David A. Wood, Hongil Yoon, and Eder F. Zulian. The gem5 Simulator: Version ´ 20.0+, 2020.
- [17] Nagendra Gulur, Mahesh Mehendale, Raman Manikantan, and Ramaswamy Govindarajan. Anatomy: An analytical model of memory system performance. In *The 2014 ACM International Conference on Measurement and Modeling of Computer Systems*, SIGMETRICS '14, 2014.
- [18] Jason E. Miller, Harshad Kasture, George Kurian, Charles Gruenwald, Nathan Beckmann, Christopher Celio, Jonathan Eastep, and Anant Agarwal. Graphite: A distributed parallel simulator for multicores. In *HPCA - 16 The Sixteenth International Symposium on High-Performance Computer Architecture*, pages 1–12, 2010.
- [19] Sadagopan Srinivasan Li Zhao Brinda Ganesh, Bruce Jacob, and Mike Espig Ravi Iyer. CMP Memory Modeling: How Much Does Accuracy Matter? In *Fifth Annual Workshop on Modeling, Benchmarking and Simulation*, 2009.
- [20] Niladrish Chatterjee, Rajeev Balasubramonian, Manjunath Shevgoor, Seth Pugsley, Aniruddha Udipi, Ali Shafiee, Kshitij Sudan, Manu Awasthi, and Zeshan Chishti. Usimm: the utah simulated memory module. *University of Utah and Intel, Tech. Rep*, 2012.
- [21] David Wang, Brinda Ganesh, Nuengwong Tuaycharoen, Katie Baynes, Aamer Jaleel, and Bruce Jacob. DRAMsim: A memory-system simulator, November 2005.
- [22] S. Li, Z. Yang, D. Reddy, A. Srivastava, and B. Jacob. DRAMsim3: A Cycle-Accurate, Thermal-Capable DRAM Simulator. *IEEE Computer Architecture Letters*, 19(2):106–109, 2020.
- [23] Yoongu Kim, Weikun Yang, and Onur Mutlu. Ramulator: A Fast and Extensible DRAM Simulator. In *IEEE Computer Architecture Letters*, volume 15, pages 45–49, 2016.
- [24] Haocong Luo, Yahya Can Tu, F Nisa Bostancı, Ataberk Olgun, A Giray Ya, and Onur Mutlu. Ramulator 2.0: A Modern, Modular, and Extensible DRAM Simulator. *IEEE Computer Architecture Letters*, pages 1–4, November 2023.
- [25] Lukas Steiner, Matthias Jung, Felipe S Prado, Kirill Bykov, and Norbert Wehn. DRAMSys4. 0: An open-source simulation framework for indepth DRAM Analyses. volume 50, pages 217–242. Springer, 2022.
- [26] Jeffrey Dean, James E Hicks, Carl A Waldspurger, William E Weihl, and George Chrysos. ProfileMe: Hardware support for instruction-level profiling on out-of-order processors. In *Proceedings of 30th Annual International Symposium on Microarchitecture*, pages 292–302. IEEE, 1997.
- [27] Christian Helm and Kenjiro Taura. Perfmemplus: A tool for automatic discovery of memory performance problems. In *International Conference on High Performance Computing*, pages 209–226. Springer, 2019.
- [28] Intel Corporation. *Intel® 64 and IA-32 Architectures Software Developer's Manual, Volume 3B: System Programming Guide, Part 2*, November 2009.
- [29] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4):65–76, 2009.
- [30] Aleksandar Ilic, Frederico Pratas, and Leonel Sousa. Cache-aware roofline model: Upgrading the loft. *IEEE Computer Architecture Letters*, 13(1):21–24, 2013.
- [31] Ahmad Yasin. A top-down method for performance analysis and counters architecture. In *IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pages 35–44. IEEE, 2014.
- [32] Mess benchmark. https://github.com/bsc-mem/Mess-benchmark, 2024. [33] Guillem Lopez-Parad ´ ´ıs, Brian Li, Adria Armejach, Stefan Wallentowitz, ` Miquel Moreto, and Jonathan Balkind. Fast Behavioural RTL Simulation ´

- of 10B Transistor SoC Designs with Metro-Mpi. In *2023 Design, Automation & Test in Europe Conference & Exhibition (DATE)*, pages 1–6. IEEE, 2023.
- [34] CXL Consortium. Compute express link (cxl), 2020.
- [35] Thomas F Wenisch, Roland E Wunderlich, Michael Ferdman, Anastassia Ailamaki, Babak Falsafi, and James C Hoe. SimFlex: Statistical Sampling of Computer System Simulation. *IEEE Micro*, 26(4):18– 31, 2006.
- [36] Barcelona Supercomputing Center Performance Tools. BSC Tools, 2023, Dec.
- [37] Bruce Jacob. *The Memory System: You Can't Avoid It, You Can't Ignore It, You Can't Fake It*. Morgan and Claypool Publishers, 2009.
- [38] Christian Helm and Kenjiro Taura. Measurement of Main Memory Bandwidth and Memory Access Latency in Intel Processors. Technical report, 2019.
- [39] Jian Yang, Juno Kim, Morteza Hoseinzadeh, Joseph Izraelevitz, and Steve Swanson. An Empirical Guide to the Behavior and Use of Scalable Persistent Memory. In *18th USENIX Conference on File and Storage Technologies (FAST 20)*, pages 169–182, Santa Clara, CA, February 2020. USENIX Association.
- [40] Joseph Izraelevitz, Jian Yang, Lu Zhang, Juno Kim, Xiao Liu, Amirsaman Memaripour, Yun Joon Soh, Zixuan Wang, Yi Xu, Subramanya R. Dulloor, Jishen Zhao, and Steven Swanson. Basic Performance Measurements of the Intel Optane DC Persistent Memory Module, 2019.
- [41] Mess simulator. https://github.com/bsc-mem/Mess-simulator, 2024.
- [42] Mess extension of paraver. https://github.com/bsc-performance-tools/ wxparaver, 2024.
- [43] Norman P. Jouppi. Cache write policies and performance. In *Proceedings of the 20th Annual International Symposium on Computer Architecture*, ISCA '93, page 191–201, New York, NY, USA, 1993. Association for Computing Machinery.
- [44] Linux. perf: Linux profiling with performance counters. https://perf. wiki.kernel.org/index.php/Main Page, 2023.
- [45] Dan Terpstra, Heike Jagode, Haihang You, and Jack Dongarra. Collecting performance data with PAPI-C. In *Tools for High Performance Computing 2009: Proceedings of the 3rd International Workshop on Parallel Tools for High Performance Computing, September 2009, ZIH, Dresden*, pages 157–173. Springer, 2010.
- [46] Jan Treibig, Georg Hager, and Gerhard Wellein. LIKWID: A Lightweight Performance-Oriented Tool Suite for x86 Multicore Environments. In *39th International Conference on Parallel Processing Workshops*, pages 207–216, 2010.
- [47] CUDA CUPTI: Cuda profiling tools interface. https://docs.nvidia.com/ cupti, 2024.
- [48] Rommel Sanchez Verdejo and Petar Radojkovic. Microbenchmarks for ´ Detailed Validation and Tuning of Hardware Simulators. In *International Conference on High-Performance Computing & Simulation (HPCS)*, pages 881–883, 2017.
- [49] Rommel Sanchez Verdejo, Kazi Asifuzzaman, Milan Radulovic, Petar ´ Radojkovic, Eduard Ayguade, and Bruce Jacob. Main memory latency ´ simulation: the missing link. In *Proceedings of the International Symposium on Memory Systems*, MEMSYS '18, page 107–116. Association for Computing Machinery, 2018.
- [50] D. W. Westcott and V. White. Instruction sampling instrumentation. US Patent #5,151,981, September 1992.
- [51] Paul Drongowski, Lei Yu, Frank Swehosky, Suravee Suthikulpanit, and Robert Richter. Incorporating Instruction-Based Sampling into AMD CodeAnalyst. In *IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pages 119–120, 2010.
- [52] M. Srinivas, B. Sinharoy, R. J. Eickemeyer, R. Raghavan, S. Kunkel, T. Chen, W. Maron, D. Flemming, A. Blanchard, P. Seshadri, J. W. Kellington, A. Mericas, A. E. Petruski, V. R. Indukuru, and S. Reyes. IBM POWER7 performance modeling, verification, and evaluation. *IBM Journal of Research and Development*, pages 1–4, 2011.
- [53] Xinxin Mei and Xiaowen Chu. Dissecting GPU Memory Hierarchy Through Microbenchmarking. *IEEE Transactions on Parallel and Distributed Systems*, 28(1):72–86, 2017.
- [54] Intel. *Intel 64 and IA-32 Architectures Optimization Reference Manual*, November 2009.
- [55] Milan Radulovic, Rommel Sanchez Verdejo, Paul Carpenter, Petar Rado- ´ jkovic, Bruce Jacob, and Eduard Ayguad ´ e. PROFET: Modeling System ´ Performance and Energy Without Simulating the CPU. *SIGMETRICS Perform. Eval. Rev.*, 47(1):71–72, December 2019.

- [56] Russell Clapp, Martin Dimitrov, Karthik Kumar, Vish Viswanathan, and Thomas Willhalm. Quantifying the Performance Impact of Memory Latency and Bandwidth for Big Data Workloads. In *Proceedings of the 2015 IEEE International Symposium on Workload Characterization*, IISWC '15, pages 213–224. IEEE Computer Society, 2015.
- [57] JEDEC STANDARD. *DDR4 SDRAM*. JEDEC Solid State Technology Association, 2017.
- [58] Tom Deakin, James Price, Matt Martineau, and Simon McIntosh-Smith. GPU-STREAM v2.0: Benchmarking the Achievable Memory Bandwidth of Many-Core Processors Across Diverse Parallel Programming Models. In *High Performance Computing*, pages 489–507. Springer International Publishing, 2016.
- [59] Joel Hestness, Stephen W. Keckler, and David A. Wood. A comparative analysis of microarchitecture effects on cpu and gpu memory system behavior. In *IEEE International Symposium on Workload Characterization (IISWC)*, pages 150–160, 2014.
- [60] A. Kashyap. *High Performance Computing: Tuning guide for AMD EPYC 7002 Series Processor*, 2020.
- [61] Satish Kumar Sadasivam, Brian W. Thompto, Ron Kalla, and William J. Starke. IBM Power9 Processor Architecture. *IEEE Micro*, 37(2):40–51, 2017.
- [62] B. Wheeler. Graviton3 Debuts Neoverse V1. Technical report, Linley Group Microprocessor, 2022.
- [63] Arijit Biswas. Sapphire Rapids. In *IEEE Hot Chips 33 Symposium (HCS)*, pages 1–22, 2021.
- [64] Fujitsu. A64FX Microarchitecture Manual. Technical report, 2019.
- [65] Jack Choquette. NVIDIA Hopper H100 GPU: Scaling Performance. *IEEE Micro*, 43(3):9–17, 2023.
- [66] Stijn Eyerman, Wim Heirman, and Ibrahim Hur. DRAM Bandwidth and Latency Stacks: Visualizing DRAM Bottlenecks. In *IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pages 322–331, 2022.
- [67] Markus Velten, Robert Schone, Thomas Ilsche, and Daniel Hackenberg. ¨ Memory Performance of AMD EPYC Rome and Intel Cascade Lake SP Server Processors. In *Proceedings of the 2022 ACM/SPEC on International Conference on Performance Engineering*, ICPE '22, 2022.
- [68] Jan Laukemann, Thomas Gruber, Georg Hager, Dossay Oryspayev, and Gerhard Wellein. CloverLeaf on Intel Multi-Core CPUs: A Case Study in Write-Allocate Evasion. *arXiv preprint arXiv:2311.04797*, 2023.
- [69] Irma Esmer Papazian. New 3rd gen intel® xeon® scalable processor (codename: Ice lake-sp). In *Hot Chips Symposium*, pages 1–22, 2020.
- [70] Zsim+dramsim3 simulation infrastructure for process in memory. https: //github.com/bsc-mem/zsim/tree/master, 2022.
- [71] Pouya Esmaili-Dokht, Miquel Guiot, Petar Radojkovic, Xavier Martorell, Eduard Ayguade, Jesus Labarta, Jason Adlard, Paolo Amato, and Marco Sforzin. O(n) Key–Value Sort With Active Compute Memory. *IEEE Transactions on Computers*, 73(05):1341–1356, may 2024.
- [72] Dramsim3. https://github.com/umd-memsys/DRAMsim3, 2024.
- [73] Geraldo F Oliveira, Juan Gomez-Luna, Lois Orosa, Saugata Ghose, ´ Nandita Vijaykumar, Ivan Fernandez, Mohammad Sadrosadati, and Onur Mutlu. DAMOV: A New Methodology and Benchmark Suite for Evaluating Data Movement Bottlenecks. *IEEE Access*, 2021.
- [74] JEDEC STANDARD. *DDR3 SDRAM*. JEDEC Solid State Technology Association, 2009.
- [75] Jonathan Balkind, Michael McKeown, Yaosheng Fu, Tri Nguyen, Yanqi Zhou, Alexey Lavrov, Mohammad Shahrad, Adi Fuchs, Samuel Payne, Xiaohua Liang, Matthew Matl, and David Wentzlaff. OpenPiton: An Open Source Manycore Research Framework. In *Proceedings of the Twenty-First International Conference on Architectural Support for Programming Languages and Operating Systems*, ASPLOS '16, page 217–232, New York, NY, USA, 2016. Association for Computing Machinery.
- [76] Jonathan Balkind, Katie Lim, Michael Schaffner, Fei Gao, Grigory Chirkov, Ang Li, Alexey Lavrov, Tri M. Nguyen, Yaosheng Fu, Florian Zaruba, Kunal Gulati, Luca Benini, and David Wentzlaff. BYOC: A "Bring Your Own Core" Framework for Heterogeneous-ISA Research. ASPLOS '20, page 699–714, New York, NY, USA, 2020. Association for Computing Machinery.
- [77] Neiel Leyva, Alireza Monemi, Noelia Oliete-Escu´ın, Guillem Lopez- ´ Parad´ıs, Xabier Abancens, Jonathan Balkind, Enrique Vallejo, Miquel Moreto, and Lluc Alvarez. OpenPiton Optimizations Towards High ´ Performance Manycores. In *Proceedings of the 16th International Workshop on Network on Chip Architectures*, NoCArc '23, page 27–33, New York, NY, USA, 2023. Association for Computing Machinery.

- [78] Jonathan Balkind, Katie Lim, Fei Gao, Jinzheng Tu, David Wentzlaff, Michael Schaffner, Florian Zaruba, and Luca Benini. OpenPiton+ Ariane: The first open-source, SMP Linux-booting RISC-V system scaling from one to many cores. In *Workshop on Computer Architecture Research with RISC-V (CARRV)*, pages 1–6, 2019.
- [79] Wilson Snyder. Verilator and systemperl. In *North American SystemC Users' Group, Design Automation Conference*, 2004.
- [80] Shang Li, Rommel Sanchez Verdejo, Petar Radojkovi ´ c, and Bruce Jacob. ´ Rethinking Cycle Accurate DRAM Simulation. In *Proceedings of the International Symposium on Memory Systems*, MEMSYS '19, page 184–191, 2019.
- [81] Stijn Eyerman, Wim Heirman, and Ibrahim Hur. Modeling dram timing in parallel simulators with immediate-response memory model. *IEEE Computer Architecture Letters*, 20(2):90–93, July 2021.
- [82] Piotr R Luszczek, David H Bailey, Jack J Dongarra, Jeremy Kepner, Robert F Lucas, Rolf Rabenseifner, and Daisuke Takahashi. The HPC Challenge (HPCC) benchmark suite. In *Proceedings of the 2006 ACM/IEEE conference on Supercomputing*, volume 213, page 1, 2006.
- [83] G Franklin, J.D. Powell, and Abbas Emami-Naeini. *Feedback Control Of Dynamic Systems*. 1994.
- [84] Graham C. Goodwin, Stefan F. Graebe, and Mario E. Salgado. *Control System Design*. 2000.
- [85] Ayaz Akram and Lina Sawalha. A Survey of Computer Architecture Simulation Techniques and Tools. *IEEE Access*, 7:78120–78145, 2019.
- [86] Anastasiia Butko, Rafael Garibotti, Luciano Ost, and Gilles Sassatelli. Accuracy evaluation of GEM5 simulator system. In *7th International Workshop on Reconfigurable and Communication-Centric Systems-on-Chip (ReCoSoC)*, pages 1–7, 2012.
- [87] Anthony Gutierrez, Joseph Pusdesris, Ronald G. Dreslinski, Trevor Mudge, Chander Sudanthi, Christopher D. Emmons, Mitchell Hayenga, and Nigel Paver. Sources of error in full-system simulation. In *IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, pages 13–22, 2014.
- [88] A. Akram and L. Sawalha. x86 computer architecture simulators: A comparative study. In *IEEE 34th International Conference on Computer Design (ICCD)*, pages 638–645, 2016.
- [89] Frank Ghenassia. *Transaction level modeling with SystemC*. Springer, 2005.
- [90] Debendra Das Sharma. Compute express link (cxl): Enabling heterogeneous data-centric computing with heterogeneous memory hierarchy. *IEEE Micro*, 43(2):99–109, 2023.
- [91] Barcelona Supercomputing Center Performance Tools. Paraver data browser, 2023, December.
- [92] Vincent Pillet, Jes Us Labarta, Toni Cortes, Sergi Girona, and Jesus´ Labarta. PARAVER : A Tool to Visualizeand Analyze Parallel. 1995.
- [93] Barcelona Supercomputing Center Performance Tools. Paraver Tracefile description. Technical report, 2023, December.
- [94] Barcelona Supercomputing Center Performance Tools. Extrae tracing framework, 2023, December.
- [95] Michael Allen Heroux, Jack Dongarra, and Piotr Luszczek. HPCG benchmark technical specification. Technical report, Sandia National Lab.(SNL-NM), Albuquerque, NM (United States), 2013.
- [96] Intel. Intel Xeon CPU Max Series Configuration and Tuning Guide. https://www.intel.com/content/www/us/en/content-details/769060/ intel-xeon-cpu-max-series-configuration-and-tuning-guide.html, 2023.
- [97] John D McCalpin. Bandwidth Limits in the Intel Xeon Max (Sapphire Rapids with HBM) Processors. In *International Conference on High Performance Computing*, pages 403–413. Springer, 2023.
- [98] John D McCalpin. STREAM: Sustainable Memory Bandwidth in High Performance Computers. https://www.cs.virginia.edu/stream/., 2022.
- [99] John D McCalpin. The STREAM 2 Benchmark. https://www.cs.virginia. edu/stream/stream2/., 2022.
- [100] Alif Ahmed and Kevin Skadron. Hopscotch: a micro-benchmark suite for memory performance evaluation. In *Proceedings of the International Symposium on Memory Systems*, MEMSYS '19, page 167–172, New York, NY, USA, 2019. Association for Computing Machinery.
- [101] PROFET. https://github.com/bsc-mem/PROFET, 2024.