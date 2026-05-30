# Trapezoid: A Versatile Accelerator for Dense and Sparse Matrix Multiplications

Yifan Yang *MIT CSAIL*yifany@csail.mit.edu

Joel S. Emer

MIT CSAIL / NVIDIA
\nemer@csail.mit.edu

Daniel Sanchez

MIT CSAIL
sanchez@csail.mit.edu

Abstract—Accelerating matrix multiplication is crucial to achieve high performance in many application domains, including neural networks, graph analytics, and scientific computing. These applications process matrices with a wide range of sparsities, from completely dense to highly sparse. Ideally, a single accelerator should handle matrices of all sparsity levels well. However, prior matrix multiplication accelerators each target a limited range of sparsity levels.

We present Trapezoid, a versatile accelerator that performs matrix multiplication across all sparsity levels effectively. Trapezoid builds on a 2D spatial array design, which excels at dense matrix multiplication, and extends it with new hardware mechanisms that let it handle sparse inputs. We present a novel inner-product-based dataflow with a multi-fiber intersection unit that handles mildly sparse matrices. Furthermore, novel Gustavson-based dataflows and a multi-level memory hierarchy enable high performance on highly sparse matrices. Trapezoid's hardware extensions are reused across dataflows to minimize area overheads.

We evaluate Trapezoid on a broad range of dense and sparse matrix multiplication workloads. Trapezoid has gmean  $19.7\times$ ,  $4.3\times$ , and  $2.9\times$  better performance/area than TPU, SIGMA, and Flexagon, prior state-of-the-art accelerators that target dense, mildly sparse, and highly sparse matrices, respectively.

Index Terms—Sparsity, Matrix Multiplication, Accelerator, Dataflow

