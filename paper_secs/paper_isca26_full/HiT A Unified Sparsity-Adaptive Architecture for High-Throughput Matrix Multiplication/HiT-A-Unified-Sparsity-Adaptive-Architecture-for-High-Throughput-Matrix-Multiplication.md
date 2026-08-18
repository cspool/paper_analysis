# HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication

Tingting Xiang<sup>1</sup> , Xiaochen Wang<sup>2</sup> , Miao Yu<sup>1</sup>, Trevor E. Carlson<sup>1</sup> <sup>1</sup>National University of Singapore, {tingtingxiang, miao.yu}@u.nus.edu, tcarlson@comp.nus.edu.sg <sup>2</sup>Zhejiang University, xiaochenwang@zju.edu.cn

*Abstract*—Accelerating matrix operations has become increasingly critical as AI models and scientific workloads continue to scale. These applications involve matrices spanning sparsity levels from <0.0001% to fully dense, demanding accelerators that maintain high performance across this full range. However, prior designs either target a narrow sparsity range, resulting in inefficiencies outside their specialization, or support broad sparsity at the cost of throughput, with the state-of-the-art accelerator achieving less than 3.125% of peak performance on highly sparse matrices.

We present HiT, a unified sparsity-adaptive architecture that delivers consistently high throughput and efficiency across the entire sparsity spectrum. HiT introduces two novel outer-productbased dataflows, HSparse and MSparse, supported by a Parallel Intersection & Distribution Unit and a Dual-mode Accumulator, targeting highly and moderately sparse workloads, respectively. These dataflows enable regular memory access to sparse data and on-chip accumulation of partial sums while exploiting two levels of spatial parallelism. As a result, HiT achieves high intersection rates (more valid non-zero matches per cycle) and data reuse, leading to high throughput. For dense workloads, HiT employs an inner-product dataflow to maximize compute efficiency.

We benchmark HiT against Trapezoid, a state-of-the-art accelerator for full-spectrum sparsity. Specifically, it delivers a 3.24× geomean performance/area improvement on highly sparse× highly sparse multiplications, 2.18× across all highly sparse workloads, and 1.99× on moderately sparse workloads. Across a comprehensive set of dense and sparse workloads, HiT achieves 1.93× higher geomean performance/area and reduces energy consumption by 1.64× compared to Trapezoid.

*Index Terms*—Sparse matrices, Matrix multiplication, Dataflow, Accelerator architectures

