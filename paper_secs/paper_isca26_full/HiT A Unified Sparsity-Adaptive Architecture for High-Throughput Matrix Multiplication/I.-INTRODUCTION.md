# I. INTRODUCTION

Matrix multiplication is a critical computational kernel powering numerous important application domains, including deep learning [1]–[4], scientific computing [5], [6], tensor algebra [7], and graph analytics [8], [9]. With the ever-increasing demand and computational intensity of these applications, enhancing the performance of matrix multiplications becomes one of the key focuses in many commercial modern processors, such as GPUs [10] and TPUs [11], [12]. However, these processors increasingly execute workloads spanning a wide range of sparsity levels. While these processors excel on dense matrices, they waste energy and cycles on the zero elements prevalent in real-world sparse workloads [11], [13], [14].

Efficiently accelerating sparse matrix multiplication poses significant challenges due to irregular data access patterns, low

![](_page_0_Figure_9.jpeg)

Fig. 1: Modern applications span a broad sparsity spectrum, while specialized accelerators typically target a specific sparsity range. Trapezoid supports the full sparsity range but underperforms on HS workloads. This work, HiT, achieves high performance across all sparsity levels.

data reuse, low computational intensity, and load imbalance. Furthermore, sparse matrices exhibit substantial variability in sparsity within and across application domains. For instance, non-zero elements commonly constitute less than 1% of the data in graph analytics (highly sparse (HS)) [14], while sparse convolutional neural networks have 1%-90% non-zero elements (moderately sparse (MS)) [15], and fully dense (D) matrices are prevalent in neural network training [2]–[4], [16], as shown in Fig. 1. This wide range of sparsity levels demands accelerator designs that can efficiently support diverse sparsity patterns without sacrificing performance.

While many specialized accelerators have been proposed to accelerate the sparse matrix multiplication [17]–[40], they suffer from two key limitations. First, these accelerators are typically optimized for limited sparsity ranges, leading to significant inefficiencies when workloads deviate from their targeted density. Second, many of these accelerators have limited scalability, offering low peak throughput that cannot scale to meet the demands of modern, large-scale matrix computations. Prior works [22]–[25] have been designed with only 16 or 64 MACs and are limited to scaling to only a few hundred MACs. This scalability challenge is particularly significant for accelerators targeting highly sparse matrix multiplication, where low data reuse and excessive data movement remain major challenges. These limitations highlight the need for a unified, high-throughput accelerator capable of efficiently handling matrices across the entire sparsity range. Recent work, Trapezoid [41], is the first large-scale unified matrix

![](_page_1_Figure_0.jpeg)

Fig. 2: Input and output density of 7 SuiteSparse Datasets.

multiplication accelerator. Although effective for D and MS workloads, it only obtains 3.125% (512 MACs) of its peak throughput in HS×HS multiplications, limiting its efficiency on crucial workloads in domains such as graph analytics and scientific computing.

Ideally, high-performance accelerators should adapt to the available sparsity; however, prior designs continue to face key challenges in accelerating HS workloads. We observe that most accelerators targeting HS workloads employ a Gustavson-based dataflow [19], [22], [28], [41]. Although effective at a small scale, these accelerators struggle to maintain high throughput as the system scales. We find that the main reason for this limitation is the large number of irregular memory accesses to the input matrices introduced by the Gustavson dataflow. As parallelism increases, the number of *random accesses to memory grows. Scaling up parallel accesses to a single cache can become both challenging and costly.* Outer-product dataflows, on the other hand, eliminate random accesses by reading the second input matrix sequentially. However, they introduce a new challenge: a large number of unmerged partial sums. This highlights the need for dataflows that retain the outer-product's sequential-access benefits while effectively handling partial sum growth.

In this work, we present HiT, a unified sparsity-adaptive accelerator that achieves high performance across the full sparsity spectrum. To efficiently support sparse computations, we propose two novel outer-product-based dataflows: HSparse for HS workloads and MSparse for MS workloads. The dataflows eliminate the gather-induced memory contention that limits Gustavson-style accelerators such as Trapezoid and enables throughput to scale with compute parallelism. For D matrices, we adopt the inner-product dataflow to maximize data reuse, parallelism, and overall computational efficiency.

Concretely, HiT includes the following contributions:

• HSparse and MSparse dataflow, deliver three key advantages: (1) higher valid matches and higher MAC utilization by exploiting two levels of spatial parallelism to match a larger number of non-zero inputs; (2) regular memory access that avoids the memory read conflicts and stalls inherent in Gustavson-based designs and (3) efficient on-chip accumulation of partial sums, addressing a core limitation of prior outer-product accelerators: OuterSPACE [18] requires costly off-chip accumulation, and SpArch [26] relies on a large merge tree, limiting scalability.

![](_page_1_Figure_7.jpeg)

Fig. 3: Matrix multiplication dataflows and data reuse comparison.

- Exploiting output sparsity. In HS×HS computation, HSparse exploits high output sparsity (as shown in Fig. 2) by storing partial sums on-chip in a compressed format, increasing the buffer utilization. This also enables the processing of larger tiles, which contain more non-zero elements and therefore increase compute density and throughput.
- Unified sparsity-adaptive architecture. HiT supports HSparse, MSparse, and inner-product dataflows with a unified architecture. Its Parallel Intersection & Distribution Unit enables large-scale, low-overhead index matching without the power-intensive intersection units and routing networks used in prior work [41]. The Dual-mode Accumulator efficiently handles sparse partial sums. Together, these components allow HiT to deliver state-ofthe-art performance across diverse sparsity levels to meet the demands of modern large-scale workloads.

In our evaluation, HiT delivers 3.24× geomean performance/area gain on HS×HS workloads, effectively overcoming the severe throughput bottlenecks of prior HS accelerators and Trapezoid. Across all evaluated workloads, HiT achieves 1.93× higher geomean performance/area and consumes 1.64× less energy compared to Trapezoid, demonstrating its advantage across the full sparsity spectrum.

For the rest of the paper, we use the term *intersection / intersect* to denote the process of matching the column index of a non-zero element in sparse Matrix A with the row index of a non-zero element in sparse Matrix B. We use *intersection rate* to quantify the ratio of valid matches between A and B elements to the total number of index comparisons evaluated.

