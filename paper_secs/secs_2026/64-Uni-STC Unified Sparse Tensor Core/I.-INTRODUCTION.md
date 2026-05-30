# I. INTRODUCTION

In the past decade, tensor cores may be the most innovative data-level parallelism technology on modern processors. Compared to classic vector SIMD units, tensor cores can complete matrix-matrix multiplication (GEMM) far more efficiently in both throughput and energy. Driven by such demand from high performance scientific and AI workloads, modern mainstream GPUs [9], [71], CPUs [4] and TPUs [35], [36] are already equipped with tensor cores of various precisions, sizes, and structured sparsity capabilities.

As sparse matrix computations are one of the major parallel computing patterns [1], designing sparsity-aware architectures received much attention [59], [66], [87]. Domain-specific architectures (DSAs) accelerating sparse computations, as well as sparse tensor cores (STCs) able to replace tensor cores in GPUs (the focus of our work), are representative directions. However, despite these advances, they still face significant limitations in terms of generality and performance.

From the perspective of generality, modern scientific computing and AI applications are exhibiting an increasing demand for diverse sparse computation patterns [25], [53], [56], [69], with the main operations covering combinatorial

TABLE I: A brief comparison of DS-STC [78], [92], RM-STC [30] and Uni-STC (our work proposed in this paper).

| STC                    | Sparse kernel                  | Dataflow                                       | Task of one cycle                                              |
|------------------------|--------------------------------|------------------------------------------------|----------------------------------------------------------------|
| DS-STC                 | SpGEMM                         | Outer-product                                  | Vector mul. vector<br>to update a matrix                       |
| RM-STC                 | SpGEMM                         | Row-row                                        | Scalars mul. vectors<br>to update vectors                      |
| Uni-STC<br>(this work) | SpMV, SpMSpV,<br>SpMM & SpGEMM | Outer-product<br>plus segmented<br>dot-product | A group of parallel<br>vector mul. vector<br>to update scalars |

applications of multiple sparse kernels. Unfortunately, the limited functional support of existing sparsity-aware architectures constrains their use in wider real-world applications.

From the perspective of performance, the existing architectures utilising outer-product [63], [78], [92] and row-row [30], [87], [93] dataflows often adopt coarse task partitioning, which results in suboptimal MAC utilisation. These architectures also continuously transmit intermediate products over large-scale networks, leading to high energy consumption.

Although the goals are explicitly specified, simultaneously improving generality and performance remains challenging. Software-only interface expansion may address generality, but often leaves hardware capabilities underutilised, highlighting the need for hardware-software co-design [64], [66]. First, it is essential to devise a single sparse format that can efficiently support a variety of sparse kernels. Second, a unified architecture must be able to generate fine-grained tasks to utilise hardware resources, schedule tasks in parallel to increase data reuse, and manage data movement to reduce energy consumption. Finally, the architectural design requires rigorous validation using a large number of sparse matrices, various sparse kernels and real-world applications.

In this paper, we propose Uni-STC, a unified sparse tensor core that brings high performance to complete sparse kernels, including sparse matrix-vector multiplication (SpMV), sparse matrix-sparse vector multiplication (SpMSpV), sparse matrixmultiple vector multiplication (SpMM), and sparse general matrix-matrix multiplication (SpGEMM). Uni-STC works on a fundamental sparse format called Bitmap-Bitmap-CSR (BBC) that combines compressed sparse row (CSR) arrays and two-level bitmap information. In addition, Uni-STC includes three newly designed functional units: tile multiply scheduler (TMS), dot-product generator (DPG), and segmented dotproduct unit (SDPU). These units take sparse tiles from the BBC format as input, split and recombine them into small dot-product tasks, schedule them for data reuse, execute the dot-products with fewer data movements, and finally save the output in the BBC format.

Compared to two state-of-the-art STC studies dual-side sparse tensor core (DS-STC) [78], [92] and row-merge sparse tensor core (RM-STC) [30], the Uni-STC emphasizes (1) the support of more complete sparse kernels, (2) the combination of various dataflows for generating fine-grained tasks, and (3) the increase of data-level parallelism in a single cycle. Table I gives a brief comparison of DS-STC, RM-STC and Uni-STC.

We evaluate Uni-STC with all 2,893 SuiteSparse matrices across the four sparse kernels (SpMV, SpMSpV, SpMM, SpGEMM), 302 DLMC matrices for DNN inference, and an Algebraic MultiGrid (AMG) solver for application-level testing. Simulation results show Uni-STC achieves geometric mean speedups of 3.35× and 2.21× over DS-STC and RM-STC at the kernel level, accompanied by energy reductions of 1.97× and 1.27×, leading to energy efficiency gains of 7.05× and 2.96×. Despite an 18% area overhead in its dedicated modules compared to the state-of-the-art RM-STC, Uni-STC retains application-level speedups of 1.43× on DNNs and 1.92× on the AMG solver, enabled by its kernel performance. This work makes the following contributions:

- We propose BBC, a unified format that supports softwarehardware collaborative computing for the four sparse kernels, while reducing storage overhead and mitigating complex hardware decoding.
- We design the Uni-STC architecture to support the four sparse kernels, optimizing resource utilisation, data reuse, and energy efficiency by featuring three novel functional units: TMS, DPG and SDPU.
- We conduct evaluation covering the performance, energy, and area of Uni-STC. Results demonstrate performance improvement and energy reduction over state-of-the-art designs with acceptable area overhead.

