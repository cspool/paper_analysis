# 4 NO-TOKEN-LEFT-BEHIND WITH BLOCK SPARSITY<sup>5</sup>

This section describes how we formulate MoE layer computation in terms of block-sparse computation in order to avoid dropping tokens. The motivation for using block-sparse primitives to express MoE computation is manifold. First, as we show below, block-sparse matrices are a natural and flexible way of describing the dynamic and load-imbalanced computation in MoEs. Second, block sparsity maps efficiently to hardware accelerators built around systolic array matrix multipliers like GPUs and TPUs. Since MoE experts have coarse granularity, we can select a block size for our implementation that is large enough to enable the computation to realize high fractions of peak device throughput. Last, block-sparse kernels like matrix multiplication and convolution are general-purpose primitives that are useful across a range of applications [\(Narang et al.,](#page-12-0) [2017;](#page-12-0) [Gray](#page-11-0) [et al.,](#page-11-0) [2017;](#page-11-0) [Child et al.,](#page-11-0) [2019;](#page-11-0) [Elsen et al.,](#page-11-0) [2020\)](#page-11-0). This makes investment in high-performance kernels more practi-

<sup>5</sup>The name No-Token-Left-Behind references the technique briefly discussed by [Fedus et al.](#page-11-0) [\(2022\)](#page-11-0), which was an unsuccessful attempt to regain the quality lost from dropping tokens.

<span id="page-4-0"></span>cal, as work can be amortized across target tasks. We could similarly invest in variable sized batched matrix multiplication kernels, but the utility of this would be limited to MoE architectures as they are designed today.

In addition to these considerations, the block-sparse formulation of MoEs exposes a new perspective on these algorithms as a form of dynamic, structured, activation sparsity. This perspective draws parallels to much of the literature on sparse training algorithms and opens up the opportunity to further improve MoEs with insights from this adjacent field.

**Preliminaries: Sparse Matrix Product Notation.** In this paper we often refer to matrix multiplication where one of the three matrices (the two inputs and one output) is sparse and the others are dense. We borrow the notation from Triton Blocksparse (Tillet et al., 2019) to describe these different operations. Each operation is described with a three character string where each character is either "S" for sparse or "D" for dense. The order of characters is output, followed by the left input and then the right input. For example, the product of two dense matrices with a sparse output is "SDD", which is also referred to as sampled dense-dense matrix multiplication (SDDMM). This notation is useful to distinguish operations like DSD and DDS, which are different forms of sparse matrix-dense matrix multiplication (SpMM). Superscript "T" indicates transposition of the input arguments. For example, SDD<sup>T</sup> indicates an SDD where the right-hand input matrix is transposed.

#### 4.1 Expert Computation With Block Sparsity

The key insight behind our method is shown in Figure 3. Rather than the prevailing approach of computing the experts within an MoE layer using batched matrix multiplication, we can equivalently compute the experts as an SDD product where the output sparse matrix has block diagonal structure, as shown in Figure 3B. In this formulation, allowing for a load-imbalanced assignment of tokens to experts is analogous to allowing the blocks in the block diagonal matrix to have a variable number of rows. To achieve this, we propose to compute each block as many smaller fixed size blocks using block-sparse matrix multiplication, as shown in Figure 3C. To construct multi-layer experts, we can iterate between SDD and DSD operations (see Figure 4).

In this formulation, we can also relax the constraint on the number of columns in each block to build MoE layers with variable sized experts, as is shown in Figure 3C. While this is an interesting direction for future work, we did not explore these configurations as more research is needed to identify how this capability can be used to increase efficiency.

With sufficiently large blocks, block-sparse matrix multiplication is capable of reaching high fractions of peak throughput on modern GPUs (Gray et al., 2017; NVIDIA, 2021).

```
# x.shape: (num_tokens, hidden_size)
   def dmoe_forward(self, x):
    # (1) Assign tokens to experts.
 3
 4
 567
       indices.shape: (num_tokens)
       weights.shape: (num_tokens)
     indices, weights = router(x)
 8
 9
       (2) Create the sparse matrix topology.
10
11
12
      # This describes the matrix in Figure 3C.
     topology = make_topology(indices)
13
14
      # (3) Permute the tokens to group by expert.
15
     x = padded_gather(x, indices)
16
17
       (4): Compute the expert layers.
18
19
      # inner_dim = ffn_hidden_size * num_experts
2.0
       self.w1.shape: (hidden_size, inner_dim)
21
22
       self.w2.shape: (inner_dim, hidden_size)
     x = sdd(x, self.wl, topology)
23
     x = dsd(x, self.w2)
24
25
      # (5) Un-permute the tokens and scale.
26
     x = padded_scatter(x, indices)
     return x * weights
```

Figure 4. **Pseudo-Code for a dMoE.** The code follows Figure 1 with three changes. First, we construct the sparse matrix topology from Figure 3C from expert assignments (line 12). Second, we pad each expert batch to a multiple of the block size during permutation (line 15, §5.2). Last, we compute the experts in parallel by iterating between SDD and DSD operations (lines 22-23, §4.1).

The coarse-grained sparsity in MoEs lends itself to this requirement – in Transformer models using MoE FFN layers, the number of columns in the blocks shown in Figure 3B corresponds to *ffin\_hidden\_size*, which is commonly between 1024 and 8192 (Vaswani et al., 2017; Radford et al., 2019; Brown et al., 2020). The number of rows in these blocks corresponds to the number of tokens assigned to each expert, which is expected to be equal to the number of tokens divided by the number of experts under a uniform distribution. This can range from a few thousand to tens of thousands of tokens per expert (Lepikhin et al., 2020; Artetxe et al., 2021; Fedus et al., 2022). These coarse-grained blocks are many times larger than the largest tile dimensions used for dense matrix multiplication kernels, which give us the flexibility to select a block size that can match their throughput.

### 4.2 Dropless Mixture-of-Experts Layers

We use this formulation of expert computation as block-sparse operations to implement *dropless-MoE* (dMoE) layers. Figure 4 highlights the key differences in dMoE implementation relative to standard MoEs. Steps 1, 3 and 5 are identical to a standard MoE implementation. dMoE introduces two changes. First, in step 2, we construct the sparse matrix shown in Figure 3C. Second, in step 4, we replace calls to batched matrix multiplication with block-sparse matrix multiplication. We describe the implementation of these two changes in detail in §5.2 and §5.1, respectively.

