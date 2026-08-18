# VI. OTHER RELATED WORK

Beyond spatial footprint prefetchers discussed in Section II-A, there are also other important mechanisms.

Streaming prefetchers target simple sequential access streams by fetching the next consecutive cache lines [21]. Stride prefetchers extend this idea to fixed address deltas, typically learned per PC [16]. More flexible stride-like designs, such as BOP and Berti, capture recurring offset-based progressions that are not strictly sequential but still exhibit regularity [28], [31].

Runahead execution allows the core to speculatively execute beyond long-latency misses so as to expose future memory accesses earlier [29], [30]. These schemes effectively act as a demand-driven prefetch engine embedded in the pipeline, and are largely orthogonal to STEP: Speculative threads are generated to run ahead of the main computation to generate prefetches, and could in principle incorporate STEP-like eventbased triggering on their own miss streams.

Machine-learning-based prefetchers have recently attracted interest for their potential to capture complex patterns. Some designs train sequence models on address or delta streams, targeting long-range correlations that are difficult to encode with handcrafted history tables [15], [33], [37]. Others formulate prefetching as a reinforcement-learning problem, where an agent learns when and how aggressively to prefetch based on performance feedback and resource usage [10], [11]. Compared to these approaches, STEP deliberately uses lightweight statistics over observed footprints, avoiding complex models while still enabling adaptive, multi-point decisions.

On the memory side, prefetch-aware DRAM controllers and bandwidth management schemes arbitrate between demand and prefetch traffic to avoid excessive queuing and interference [18], [26]. STEP's event-based confidence evaluation is complementary to such techniques: our design concentrates on issuing more selective and timely prefetches, while cache and memory policies can provide an additional layer of protection against pollution and bandwidth contention.

#### VII. CONCLUSION

This paper revisits a core design choice in spatial footprint prefetchers: the reliance on a single fixed trigger point for prefetching. We propose STEP, which replaces a one-shot trigger with a sequence of temporal decision points and thereby mitigates the trade-off among opportunity, accuracy, and storage. Our evaluation across diverse benchmark suites shows that STEP consistently outperforms strong single-point trigger spatial footprint baselines. In single-core systems, STEP achieves a geometric-mean speedup of 1.28× over no prefetching and outperforms the strengthened ISO-storage baseline eBingo. Across multicore, cache-hierarchy, systemparameter, and storage-sensitivity studies, STEP continues to perform strongly while maintaining a favorable performance–storage operating point.

Looking forward, several promising directions remain. First, the current multi-point trigger decisions rely primarily on a lightweight confidence evaluator, which could be further enhanced with additional runtime signals such as memory bandwidth utilization or prefetch feedback. Second, exploring alternative triggering events beyond offset-based ones—or extending the STEP-style staged trigger-decision framework to other prefetcher families—could further broaden its applicability and unlock additional performance potential.

### ACKNOWLEDGEMENTS

This work was supported by the Federal Ministry of Education and Research (BMBF), Germany, in the framework of the project MANNHEIM-CeCaS (Grant No. 16ME0800K).

#### REFERENCES

- [1] "2nd cache replacement championship (crc2)," https://crc2.ece.tamu. edu/.
- [2] "2nd data prefetching championship (dpc2)," https://comparch-conf. gatech.edu/dpc2/.
- [3] "3rd data prefetching championship (dpc3)," https://dpc3.compas.cs. stonybrook.edu/.
- [4] "Champsim," https://github.com/ChampSim/ChampSim.
- [5] "The POWER4 processor introduction and tuning guide," IBM Corporation, IBM Redbooks SG247041, 2002.
- [6] *Software Optimization Guide for the AMD Zen 4 Microarchitecture*, Advanced Micro Devices, Inc., 2023, revision 1.00.

- [7] S. Ainsworth and L. Mukhanov, "Triangel: A High-Performance, Accurate, Timely On-Chip Temporal Prefetcher," in *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, Jun. 2024, pp. 1202–1216.
- [8] M. Bakhshalipour, P. Lotfi-Kamran, and H. Sarbazi-Azad, "Domino temporal data prefetcher," in *2018 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, Feb. 2018, pp. 131– 142.
- [9] M. Bakhshalipour, M. Shakerinava, P. Lotfi-Kamran, and H. Sarbazi-Azad, "Bingo spatial data prefetcher," in *2019 IEEE International Symposium on High Performance Computer Architecture (HPCA)*, Feb. 2019, pp. 399–411.
- [10] R. Bera, K. Kanellopoulos, S. Balachandran, D. Novo, A. Olgun, M. Sadrosadati, and O. Mutlu, "Hermes: Accelerating long-latency load requests via perceptron-based off-chip load prediction," in *55th IEEE/ACM International Symposium on Microarchitecture, MICRO 2022, Chicago, IL, USA, October 1-5, 2022*. IEEE, 2022, pp. 1–18. [Online]. Available: https://doi.org/10.1109/MICRO56248.2022.00015
- [11] R. Bera, K. Kanellopoulos, A. Nori, T. Shahroodi, S. Subramoney, and O. Mutlu, "Pythia: A customizable hardware prefetching framework using online reinforcement learning," in *MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture*, 2021, pp. 1121–1137.
- [12] R. Bera, A. V. Nori, O. Mutlu, and S. Subramoney, "Dspatch: Dual spatial pattern prefetcher," in *Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture*, 2019, pp. 531–544.
- [13] E. Bhatia, G. Chacon, S. Pugsley, E. Teran, P. V. Gratz, and D. A. Jimenez, "Perceptron-based prefetch filtering," in ´ *Proceedings of the 46th International Symposium on Computer Architecture*, 2019, pp. 1– 13.
- [14] Z. Chen, C. Wu, Y. Gu, R. Jia, J. Li, and M. Guo, "Gaze into the pattern: Characterizing spatial patterns with internal temporal correlations for hardware prefetching," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2025, pp. 173– 187.
- [15] Q. Duong, A. Jain, and C. Lin, "A New Formulation of Neural Data Prefetching," in *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, Jun. 2024, pp. 1173–1187.
- [16] J. W. Fu, J. H. Patel, and B. L. Janssens, "Stride directed prefetching in scalar processors," *ACM SIGMICRO Newsletter*, vol. 23, no. 1-2, pp. 102–110, 1992.
- [17] J. L. Hennessy and D. A. Patterson, *Computer architecture: a quantitative approach*. Elsevier, 2011.
- [18] C. Huang, V. Nagarajan, and A. Joshi, "DCA: a dram-cache-aware DRAM controller," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2016, Salt Lake City, UT, USA, November 13-18, 2016*, J. West and C. M. Pancake, Eds. IEEE Computer Society, 2016, pp. 887–897. [Online]. Available: https://doi.org/10.1109/SC.2016.75
- [19] *Intel 64 and IA-32 Architectures Optimization Reference Manual*, Intel Corporation, 2024, order Number 248966.
- [20] S. Jiang, Q. Yang, and Y. Ci, "Merging similar patterns for hardware prefetching," in *2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, 2022, pp. 1012–1026.
- [21] N. P. Jouppi, "Improving direct-mapped cache performance by the addition of a small fully-associative cache and prefetch buffers," in *Proceedings of the 17th Annual International Symposium on Computer Architecture (ISCA)*, 1990.
- [22] J. Kim, S. H. Pugsley, P. V. Gratz, A. N. Reddy, C. Wilkerson, and Z. Chishti, "Path confidence based lookahead prefetching," in *2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, 2016, pp. 1–12.
- [23] P. M. Kogge, "Memory intensive computing, the third wall, and the need for innovation in architecture," Univ. of Notre Dame white paper, 2017, available: https://memsys.io/wp-content/uploads/2017/12/The Wall.pdf.
- [24] A. Labrinidis and H. V. Jagadish, "Challenges and opportunities with big data," *Proceedings of the VLDB Endowment*, vol. 5, no. 12, pp. 2032–2033, 2012.
- [25] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *nature*, vol. 521, no. 7553, pp. 436–444, 2015.
- [26] C. J. Lee, O. Mutlu, V. Narasiman, and Y. N. Patt, "Prefetch-aware DRAM controllers," in *41st Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-41 2008), November 8-12, 2008, Lake Como, Italy*. IEEE Computer Society, 2008, pp. 200–209.

- [27] Y. Liu and M. Chen, "Planaria: Pattern directed cross-page composite prefetcher," in *Proceedings of the 61st ACM/IEEE Design Automation Conference*, 2024, pp. 1–6.
- [28] P. Michaud, "A best-offset prefetcher," in *2nd Data Prefetching Championship*, 2015.
- [29] O. Mutlu, H. Kim, and Y. N. Patt, "Techniques for efficient processing in runahead execution engines," in *32nd International Symposium on Computer Architecture (ISCA'05)*. IEEE, 2005, pp. 370–381.
- [30] O. Mutlu, J. Stark, C. Wilkerson, and Y. N. Patt, "Runahead execution: An alternative to very large instruction windows for out-of-order processors," in *The Ninth International Symposium on High-Performance Computer Architecture, 2003. HPCA-9 2003. Proceedings.* IEEE, 2003, pp. 129–140.
- [31] A. Navarro-Torres, B. Panda, J. Alastruey-Benede, P. Ib ´ a´nez, V. Vi ˜ nals- ˜ Yufera, and A. Ros, "Berti: an accurate local-delta data prefetcher," in ´ *2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)*. IEEE, 2022, pp. 975–991.
- [32] S. Pakalapati and B. Panda, "Bouquet of instruction pointers: Instruction pointer classifier-based spatial hardware prefetching," in *2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2020, pp. 118–131.
- [33] L. Peled, U. Weiser, and Y. Etsion, "A neural network prefetcher for arbitrary memory access patterns," *ACM Transactions on Architecture and Code Optimization*, vol. 16, no. 4, pp. 1–27, Dec. 2019.
- [34] S. Somogyi, T. F. Wenisch, A. Ailamaki, B. Falsafi, and A. Moshovos, "Spatial memory streaming," *ACM SIGARCH Computer Architecture News*, vol. 34, no. 2, pp. 252–263, 2006.
- [35] T. F. Wenisch, M. Ferdman, A. Ailamaki, B. Falsafi, and A. Moshovos, "Practical off-chip meta-data for temporal memory streaming," in *2009 IEEE 15th International Symposium on High Performance Computer Architecture*, Feb. 2009, pp. 79–90.
- [36] W. A. Wulf and S. A. McKee, "Hitting the memory wall: Implications of the obvious," *ACM SIGARCH computer architecture news*, vol. 23, no. 1, pp. 20–24, 1995.
- [37] P. Zhang, N. Gupta, R. Kannan, and V. K. Prasanna, "Attention, distillation, and tabularization: Towards practical neural network-based prefetching," in *2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, May 2024, pp. 876–888.