# Uni-STC: Unified Sparse Tensor Core

Haocheng Lian<sup>1</sup> , Qiyue Zhang<sup>1</sup> , Xinran Zhao<sup>1</sup> , Meichen Dong<sup>1</sup> , Yijie Nie<sup>1</sup> , Zhengyi Zhao<sup>1</sup> , Junzhong Shen<sup>2</sup> , Wei Guo<sup>2</sup> , Chun Huang<sup>2</sup> , Bingcai Sui<sup>2</sup> and Weifeng Liu<sup>1</sup>

1. Super Scientific Software Laboratory, Department of CST, China University of Petroleum-Beijing, Beijing, China 2. National University of Defense Technology, Changsha, China

{haocheng.lian, qiyue.zhang, xr.zhao, meichen.dong, yijie.nie, zhengyi.zhao}@student.cup.edu.cn {shenjunzhong, wineer guowei, chunhuang, bingcaisui}@nudt.edu.cn and weifeng.liu@cup.edu.cn

*Abstract*—Modern processors are increasingly adopting tensor cores as key computational units. Compared to existing designs for dense and structured sparsity, recent dual-side sparse tensor cores have evolved to support general sparsity. However, existing methods still face limitations on generality (incomplete sparse kernel support prevents broad applicability) and performance (outer-product/row-row schemes yield unsatisfactory hardware utilisation, data reuse, and energy efficiency).

In this paper, we propose Uni-STC, a unified sparse tensor core that delivers high-performance dataflows for four key sparse kernels: sparse matrix-vector multiplication (SpMV), sparse matrixsparse vector multiplication (SpMSpV), sparse matrix-multiple vector multiplication (SpMM), and sparse general matrix-matrix multiplication (SpGEMM). To efficiently support these diverse sparse workloads, we first introduce BBC, a unified sparse format co-designed with Uni-STC's dataflow. We then design Uni-STC's architecture supporting (1) fine-grained task partitioning to improve resource utilisation, (2) parallel sparse-tile processing to enhance data reuse, and (3) a dynamic network to reduce intermediate data movement and energy consumption. Evaluated across 2893 SuiteSparse and 302 DLMC matrices, Uni-STC demonstrates significant improvements, outperforming the stateof-the-art RM-STC with a 2.21× geomean speedup and 2.96× higher energy efficiency.

