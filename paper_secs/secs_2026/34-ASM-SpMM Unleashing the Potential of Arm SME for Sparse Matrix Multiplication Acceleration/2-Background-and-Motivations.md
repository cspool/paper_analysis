# 2 Background and Motivations

#### 2.1 Sparse Matrix Multiplication

SpMM computes the product of a sparse matrix A of size  $M \times K$  and a dense matrix B of size  $K \times N$ , producing a dense output matrix C of size  $M \times N$ , where C = AB. The sparse matrix A contains NNZ nonzero elements, while both B and C are fully dense. NNZ denotes the total number of nonzero entries in sparse matrix. A substantial body of research have been conducted to improve SpMM performance on CPUs [30], encompassing a variety of optimization techniques such as sparse storage formats, reordering algorithms, parallel strategies, and memory access optimizations. Most existing SpMM optimizations are grounded in exploiting the general-purpose capabilities of modern processors, utilizing multicore architectures in conjunction with vectorized instruction sets such as NEON to accelerate SpMM computations.

#### 2.2 ARM CPU and SME Matrix Unit

The rising computational demands of AI and scientific computing have driven the development of specialized hardware accelerators for matrix operations. Broadly, matrix acceleration architectures follow two paths: (i) dedicated matrix-multiplication units supporting diverse matrix sizes, exemplified by Intel AMX, NVIDIA Tensor Cores, and Google TPUs; and (ii) lightweight vector outer-product units, such as IBM's Math Matrix Accelerator and ARM's Scalable Matrix Extension (SME). ARM CPUs, now pervasive from mobile to servers, are increasingly required to deliver high-performance matrix computation for workloads ranging from edge AI inference to large-scale simulations. This trend underscores the need to embed matrix-centric acceleration directly into general-purpose CPUs.

ARM's Scalable Vector Extension (SVE) introduces variable-length vector registers and vector-length-agnostic programming, providing flexibility and scalability for SIMD work-loads. ARM SME with ARMv9 enhances the CPU architecture's support for matrix operations. SME works with the existing SVE and provides a dedicated two-dimensional matrix register array (ZA) storage and outer product instructions using two SVE Z registers as input vectors, enabling efficient construction of matrix multiplication via outer-product accumulation as shown in Figure 2. The ZA register is architecturally defined for matrix tiles, with supporting instructions for data transfer between registers and memory. Each input

source vector can be independently predicated by its corresponding predicate register. For example, input vector Z0 is predicated by P0, and Z1 is predicated by P1.

<span id="page-2-0"></span>![](_page_2_Figure_3.jpeg)

Figure 2. Registers in SME.

Apple's M4 chip is the frst and currently the only publicly available processor to support SME. This represents a pivotal step toward the mainstream adoption of matrix acceleration on general-purpose CPUs[\[17\]](#page-12-20). Developers can leverage SME's C-level intrinsics to efciently implement matrix operations while abstracting away low-level register management. For instance, data tiles can be loaded from memory into vector registers using intrinsics such as svld1\_f32, and outer product accumulation can be performed with svmopa\_za32\_f32\_m, directly accumulating results into specifed ZA tiles. Results are then written back to memory using store intrinsics such as svst1\_hor\_za32.

Listing 1. SME Intrinsics for Matrix Multiplication.

```
1 svbool_t pg = svptrue_b32();
2 //Load a tile from global memory into a vector register
3 svfloat32_t a_vec = svld1_f32(pg, A_block);
4 svfloat32_t b_vec = svld1_f32(pg, B_block);
5 //Perform outer product and accumulate into ZA[0].
6 svmopa_za32_f32_m(0, pg, pg, a_vec, b_vec);
7 //Store the result from ZA[0] to global memory.
8 svst1_hor_za32(0, row_idx, pg, C_result);
```

