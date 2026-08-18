# V. EVALUATION METHODOLOGY

Simulator: We conduct experiments using the industryvalidated MGPUsim simulator [55], following prior work on multi-GPU systems [11], [32], [34], [38], [60]. We target a 4-GPU system, where each GPU maintains its own local page table and GMMU. Configurations are summarized in Table III.

TABLE III BASELINE MULTI-GPU CONFIGURATION

| Module            | Configuration                                  |  |  |
|-------------------|------------------------------------------------|--|--|
| SM                | 1.0 GHz, 108 per GPU                           |  |  |
| L1 D-Cache        | 64 KB, 4-way                                   |  |  |
| L1 I-Cache        | 32 KB, 4-way                                   |  |  |
| L2 Cache          | 2 MB, 8-way                                    |  |  |
| DRAM              | Configured to 70% of application's memory      |  |  |
|                   | footprint [11], [60]                           |  |  |
| L1 TLB            | 16 entries, 16-way, 1-cycle lookup latency,    |  |  |
|                   | TPC shared, LRU replacement policy             |  |  |
|                   | 128 entries, 8-way, 16 sub-entries per entry,  |  |  |
| L2 TLB            | 10-cycle lookup latency, GPC shared,           |  |  |
|                   | LRU replacement policy                         |  |  |
|                   | 1024 entries, 8-way, 16 sub-entries per entry, |  |  |
| L3 TLB            | 40-cycle lookup latency, GPU shared,           |  |  |
|                   | LRU replacement policy                         |  |  |
|                   | GMMU 8 shared page table walkers,              |  |  |
| Page Table Walk   | 100-cycle latency per level                    |  |  |
| Inter-GPU Network | 300 GB/s NVLink 3.0                            |  |  |
| CPU-GPU Network   | 32 GB/s PCIe-v4                                |  |  |

Applications and Workloads: Following prior work on multi-GPU page migration [11], [60], we use 23 applications with various multi-GPU memory access and page sharing patterns from AMDAPPSDK [8], Hetero-Mark [56], SHOC [15], and DNN-MARK [17] benchmark suites as listed in Table IV. We use the default input sets of these applications for evaluation. Compared Related Work: We compare seven state-of-the-art methods: (1) TBNP-O [21]: NVIDIA's TBNP with on-touch migration, migrating accessed pages immediately to local GPU memory; (2) TBNP-F [21]: TBNP with first-touch migration, migrating a page only once upon initial access; (3) TBNP-AT [22]: Adaptive Threshold adjusts between remote zero-copy access and migration using hardware counters; (4) TBNP-EA [23]: Early Adapter dynamically tunes prefetch thresholds based on page fault variations; (5) Forest [38]: Modifies prefetch size for blocks and trees based on access sequences;

TABLE IV BENCHMARK APPLICATIONS

| Abbr.   | Application                         | Memory<br>Footprint<br>(per GPU) | Access<br>Pattern |
|---------|-------------------------------------|----------------------------------|-------------------|
| SC      | Simple Convolution                  | 32 MB                            | Adjacent          |
| C2D     | Convolution 2D                      | 23 MB                            | Adjacent          |
| MM      | Matrix Multiplication               | 8 MB                             | Scatter-Gather    |
| MT      | Matrix Transpose                    | 16 MB                            | Scatter-Gather    |
| FIR     | Finite Impulse Resp.                | 38 MB                            | Adjacent          |
| ST      | Stencil 2D                          | 8 MB                             | Adjacent          |
| IM2COL  | Image To Column                     | 20 MB                            | Scatter-Gather    |
| FFT     | Fast Fourier Transform              | 12 MB                            | Scatter-Gather    |
| LeNet   | LeNet                               | 6 MB                             | Mixed             |
| VGG     | Visual Geometry Group 16-layer      | 55 MB                            | Mixed             |
| BS      | Bitonic Sort                        | 7 MB                             | Random            |
| BERT-T  | BERT Tiny                           | 68 MB                            | Mixed             |
| BERT-M  | BERT Mini                           | 136 MB                           | Mixed             |
| BERT-ME | BERT Medium                         | 272 MB                           | Mixed             |
| BERT-B  | BERT Base                           | 544 MB                           | Mixed             |
| GPT2-M  | GPT-2 Mini                          | 65 MB                            | Mixed             |
| GPT2    | GPT-2                               | 196 MB                           | Mixed             |
| BFS     | Breadth-First Search                | 8 MB                             | Random            |
| PR      | Page Rank                           | 8 MB                             | Random            |
| MIS     | Max. Independent Set                | 4 MB                             | Random            |
| SSSP    | Single Source Shortest Path         | 14 MB                            | Random            |
| SPMV    | Sparse Matrix Vector Multiplication | 14 MB                            | Random            |
| KM      | K-Means Clustering Algorithm        | 33 MB                            | Random            |

(6) HOPP [36]: Prefetches pages based on categorized access patterns in disaggregated memory; (7) GRIT [60]: Reactively selects migration strategies among on-touch, counter-based, and duplication methods.

## VI. EVALUATION

## *A. End-to-End Performance*

Figure 14 displays the normalized end-to-end performance across seven related works. For regular benchmarks, LIBRA yields performance improvements of 44%, 37%, 31%, and 29% over TBNP-EA, Forest, HOPP, and GRIT, respectively. For irregular benchmarks, the numbers are 30%, 29%, 38%, and 36%, overall, the numbers are 40%, 35%, 32%, and 30%. These results underscore our method's effectiveness in optimizing page migration strategies and improving overall performance.

These performance gains primarily stem from LIBRA's capability to prefetch pages based on access patterns and to make cost-benefit aware, coordinated migration decisions. For example, in the FIR benchmark, LIBRA achieves performance improvements of 48% and 53% over GRIT and Forest, respectively. FIR has a significant proportion of pages that are infrequently accessed. In such scenarios, the cost of remote access is lower than that of migrating pages. Other approaches that lack cost-aware designs may still migrate pages frequently. In the ST benchmark, LIBRA outperforms Forest and GRIT by 31% and 40%, respectively. This gain stems from ST's access pattern, where GPUs access certain pages intensively but briefly, causing excessive migrations under spatial locality prefetchers (TBNP-O, TBNP-F, TBNP-EA, Forest) and GRIT's on-demand migration.

