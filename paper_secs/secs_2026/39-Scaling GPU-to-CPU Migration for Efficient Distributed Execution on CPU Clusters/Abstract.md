# Abstract

The growing demand for GPU resources has led to widespread shortages in data centers, prompting the exploration of CPUs as an alternative for executing GPU programs. While prior research supports executing GPU programs on single CPUs, these approaches struggle to achieve competitive performance due to the computational capacity gap between GPUs and CPUs.

To further improve performance, we introduce CuCC, a framework that scales GPU-to-CPU migration to CPU clusters and utilizes distributed CPU nodes to execute GPU programs. Compared to single-CPU execution, CPU cluster execution requires cross-node communication to maintain data consistency. We present the CuCC execution workflow and communication optimizations, which aim to reduce network overhead. Evaluations demonstrate that CuCC achieves high scalability on large-scale CPU clusters and delivers runtimes approaching those of GPUs. In terms of cluster-wide throughput, CuCC enables CPUs to achieve an average of 2.59× higher throughput than GPUs.

## CCS Concepts: • Computing methodologies→Distributed computing methodologies.

Keywords: compiler optimization, GPU-to-CPU migration, CPU cluster

#### ACM Reference Format:

Ruobing Han and Hyesoon Kim. 2026. Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters. In Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '26), January 31 – February 4, 2026, Sydney, NSW, Australia. ACM, New York, NY, USA, [14](#page-13-0) pages. <https://doi.org/10.1145/3774934.3786435>

