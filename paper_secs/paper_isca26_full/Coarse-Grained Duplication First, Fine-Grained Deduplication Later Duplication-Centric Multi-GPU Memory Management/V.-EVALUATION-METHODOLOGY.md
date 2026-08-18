# V. EVALUATION METHODOLOGY

Simulator: We conduct experiments using the industryvalidated MGPUsim simulator [38], following prior work on multi-GPU systems [8], [23], [24], [42]. We target a 4-GPU system, where each GPU maintains its own local page table and GMMU. Configurations are summarized in Table II.

| TABLE II                         |
|----------------------------------|
| BASELINE MULTI-GPU CONFIGURATION |

| Module            | Configuration                                                                                                    |
|-------------------|------------------------------------------------------------------------------------------------------------------|
| SM                | 1.0 GHz, 108 per GPU                                                                                             |
| L1 D-Cache        | 64 KB, 4-way                                                                                                     |
| L1 I-Cache        | 32 KB, 4-way                                                                                                     |
| L2 Cache          | 2 MB, 8-way                                                                                                      |
| DRAM              | 2 GB                                                                                                             |
| L1 TLB            | 16 entries, 16-way, 1-cycle lookup latency,<br>TPC shared, LRU replacement policy                                |
| L2 TLB            | 128 entries, 8-way, 16 sub-entries per entry,<br>10-cycle lookup latency, GPC shared,<br>LRU replacement policy  |
| L3 TLB            | 1024 entries, 8-way, 16 sub-entries per entry,<br>40-cycle lookup latency, GPU shared,<br>LRU replacement policy |
| Page Table Walk   | GMMU 8 shared page table walkers,<br>100-cycle latency per level                                                 |
| Inter-GPU Network | NVLink 3.0 latency                                                                                               |
|                   | across different transfer sizes in Section III-A                                                                 |
| CPU-GPU Network   | 128 GB/s PCIe-v5                                                                                                 |

Applications and Workloads: Following prior work on multi-GPU page migration [8], [42], we use thirteen applications with various multi-GPU memory access and page sharing patterns from AMDAPPSDK [6], Hetero-Mark [39], SHOC [11], and DNN-MARK [13] benchmark suites as listed in Table III. We use the default input sets of these applications for evaluation.

TABLE III BENCHMARK INFORMATION

| Abbr.  | Application            | Memory Footprint<br>(per GPU) | Access<br>Pattern |
|--------|------------------------|-------------------------------|-------------------|
| SC     | Simple Convolution     | 1024 MB                       | Adjacent          |
| C2D    | Convolution 2D         | 736 MB                        | Adjacent          |
| MM     | Matrix Multiplication  | 512 MB                        | Scatter-Gather    |
| MT     | Matrix Transpose       | 1024 MB                       | Scatter-Gather    |
| FIR    | Finite Impulse Resp.   | 1216 MB                       | Adjacent          |
| ST     | Stencil 2D             | 512 MB                        | Adjacent          |
| IM2COL | Image To Column        | 640 MB                        | Scatter-Gather    |
| FFT    | Fast Fourier Transform | 384 MB                        | Scatter-Gather    |
| PR     | Page Rank              | 1024 MB                       | Random            |
| BERT-M | BERT Mini              | 2176 MB                       | Mixed             |
| BERT-B | BERT Base              | 8704 MB                       | Mixed             |
| GPT2-M | GPT-2 Mini             | 2080 MB                       | Mixed             |
| GPT2   | GPT-2                  | 6272 MB                       | Mixed             |

Compared Related Work: We compare two state-of-the-art methods along with a coarse-grained duplication baseline: (1) GPS [30], which employs a subscription-based model for page duplication with batched remote write updates; (2) GRIT [42], which dynamically selects among on-touch migration, counter-based migration, and duplication strategies; and (3) CoarseDup, our method that only employs coarsegrained duplication with remote updates, leveraging insights from our NVLink 3.0 performance characterization.

# V. EVALUATION METHODOLOGY

Simulator: We conduct experiments using the industryvalidated MGPUsim simulator [38], following prior work on multi-GPU systems [8], [23], [24], [42]. We target a 4-GPU system, where each GPU maintains its own local page table and GMMU. Configurations are summarized in Table II.

| TABLE II                         |
|----------------------------------|
| BASELINE MULTI-GPU CONFIGURATION |

| Module            | Configuration                                                                                                    |
|-------------------|------------------------------------------------------------------------------------------------------------------|
| SM                | 1.0 GHz, 108 per GPU                                                                                             |
| L1 D-Cache        | 64 KB, 4-way                                                                                                     |
| L1 I-Cache        | 32 KB, 4-way                                                                                                     |
| L2 Cache          | 2 MB, 8-way                                                                                                      |
| DRAM              | 2 GB                                                                                                             |
| L1 TLB            | 16 entries, 16-way, 1-cycle lookup latency,<br>TPC shared, LRU replacement policy                                |
| L2 TLB            | 128 entries, 8-way, 16 sub-entries per entry,<br>10-cycle lookup latency, GPC shared,<br>LRU replacement policy  |
| L3 TLB            | 1024 entries, 8-way, 16 sub-entries per entry,<br>40-cycle lookup latency, GPU shared,<br>LRU replacement policy |
| Page Table Walk   | GMMU 8 shared page table walkers,<br>100-cycle latency per level                                                 |
| Inter-GPU Network | NVLink 3.0 latency                                                                                               |
|                   | across different transfer sizes in Section III-A                                                                 |
| CPU-GPU Network   | 128 GB/s PCIe-v5                                                                                                 |

Applications and Workloads: Following prior work on multi-GPU page migration [8], [42], we use thirteen applications with various multi-GPU memory access and page sharing patterns from AMDAPPSDK [6], Hetero-Mark [39], SHOC [11], and DNN-MARK [13] benchmark suites as listed in Table III. We use the default input sets of these applications for evaluation.

TABLE III BENCHMARK INFORMATION

| Abbr.  | Application            | Memory Footprint<br>(per GPU) | Access<br>Pattern |
|--------|------------------------|-------------------------------|-------------------|
| SC     | Simple Convolution     | 1024 MB                       | Adjacent          |
| C2D    | Convolution 2D         | 736 MB                        | Adjacent          |
| MM     | Matrix Multiplication  | 512 MB                        | Scatter-Gather    |
| MT     | Matrix Transpose       | 1024 MB                       | Scatter-Gather    |
| FIR    | Finite Impulse Resp.   | 1216 MB                       | Adjacent          |
| ST     | Stencil 2D             | 512 MB                        | Adjacent          |
| IM2COL | Image To Column        | 640 MB                        | Scatter-Gather    |
| FFT    | Fast Fourier Transform | 384 MB                        | Scatter-Gather    |
| PR     | Page Rank              | 1024 MB                       | Random            |
| BERT-M | BERT Mini              | 2176 MB                       | Mixed             |
| BERT-B | BERT Base              | 8704 MB                       | Mixed             |
| GPT2-M | GPT-2 Mini             | 2080 MB                       | Mixed             |
| GPT2   | GPT-2                  | 6272 MB                       | Mixed             |

Compared Related Work: We compare two state-of-the-art methods along with a coarse-grained duplication baseline: (1) GPS [30], which employs a subscription-based model for page duplication with batched remote write updates; (2) GRIT [42], which dynamically selects among on-touch migration, counter-based migration, and duplication strategies; and (3) CoarseDup, our method that only employs coarsegrained duplication with remote updates, leveraging insights from our NVLink 3.0 performance characterization.

