# C COMPARISON TO TRITON BLOCKSPARSE

Figure 13 compares our block-sparse matrix multiplication kernels to the kernels available in Triton Blocksparse on the problems shown in Figure 9. We benchmarked with the same setup described in  $\S6.3$  with Triton 2.1. For these benchmarks, we included the time spent preprocessing the sparse matrix topology on each invocation since the topology of the sparse matrix in dMoE layers changes every iteration of training. If this preprocessing cost is excluded, our kernels outperform Triton Blocksparse by  $1.17\times$  on average. This advantage, despite no preprocessing of the sparse matrix, highlights the efficiency of our kernels for this workload.

## D ARTIFACT APPENDIX

#### D.1 Abstract

MegaBlocks is available on GitHub. This appendix explains how to run the test suite for the MegaBlocks dMoE layer.

#### D.2 Artifact check-list (meta-information)

- Program: megablocks/layers/dmoe\_test.py.
- Run-time environment: Docker.
- Hardware: Nvidia A100 GPU.
- Execution: python megablocks/layers/dmoe\_test.py.
- How much time is needed to prepare workflow (approximately)?: 5 minutes.
- How much time is needed to complete experiments (approximately)?: 30 seconds.
- Publicly available?: github.com/stanfordfuturedata/megablocks.
- Code licenses (if publicly available)?: Apache-2.0.

### **D.3** Description

#### D.3.1 How delivered

The repository is hosted on GitHub at github.com/stanford-futuredata/megablocks. An archival version of the respository is available at doi.org/10.5281/zenodo.7883726.

#### D.3.2 Hardware dependencies

Nvidia A100 GPU.

#### D.3.3 Software dependencies

Docker. All dependencies handled by Dockerfile.

#### <span id="page-15-0"></span>**D.4** Installation

Follow "Training Models with Megatron-LM".

#### D.5 Evaluation and expected result

Once inside the Docker container with MegaBlocks installed (§D.4), run 'python megablocks/layers/dmoe\_test.py'. After passing, the test suite will print "OK".

```
1
   /** Sparse = Dense x Dense.
 3
    * Arguments:
       a: Dense, left input with shape (m, k).
 5
       b: Dense, right input with shape (k, n).
 6
       c: Sparse, output with shape (m, n).
 8
     _global_
             void sdd (Matrix a, Matrix b,
       SparseMatrix c) {
10
        (1) Load row and column indices.
11
12
     // These indicate the location of the non-
13
     // zero block computed by this threadblock.
     int row = c.row_idxs[blockIdx.x];
14
15
     int column = c.column_idxs[blockIdx.x];
16
17
     // (2) Zero accumulator tile.
Tile<128, 128> tile_c(/*init_to=*/0);
18
19
20
     // (3) Main loop.
2.1
      // Load tiles from a & b and compute.
2.2.
2.3
     int k = a.shape[1];
     for (int i = 0; i < k; i += 128) {</pre>
2.4
       Tile<128, 128> tile_a = LoadTile(
2.5
         a, /*row=*/row, /*column=*/i);
26
       Tile<128, 128> tile_b = LoadTile(
2.7
2.8
         b, /*row=*/i, /*column=*/column);
29
       tile_c += tile_a * tile_b;
30
31
32
      // (4) Write output for this non-zero block.
33
     StoreTile(tile_c, c);
34
```

Figure 11. CUDA Pseudo-Code for Our SDD Kernel. We launch one threadblock per non-zero block in the sparse output. (1) On startup, each threadblock loads the row and column indices of its non-zero block in the output. As described in §5.1.3, this step is made trivial by the availability of row indices in our hybrid blocked-CSR-COO encoding. (2) Next, each threadblock sets its accumulator tile to zero. (3) The threadblock steps through the *a* and *b* matrices, computing the product of two-dimensional tiles loaded from the input matrices and accumulating the results. (4) Last, we store the final accumulated result to the output sparse matrix.

```
1
   /** Dense = Sparse x Dense.
 2
 3
      Arguments:
 4
       a: Sparse, left input with shape (m, k).
 5
       b: Dense, right input with shape (k, n).
 6
       c: Dense, output with shape (m, n).
             _ void dsd(SparseMatrix a, Matrix b,
 8
   __global_
       Matrix c)
 a
10
         (1) Calculate row and column indices.
11
     // Each threadblock computes one tile of 'c'.
12
13
     int row = blockIdx.x;
     int column = blockIdx.y;
15
         (2) Load offset into 'a' and calculate
      // the non-zeros in this threadblock's row.
17
18
     int offset_a = a.row_offsets[blockIdx.x];
19
     int nnz = a.row_offsets[blockIdx.x + 1]
20
       offset a:
21
22
23
         (3) Zero accumulator tile.
     Tile<128, 128> tile_c(/*init_to=*/0);
25
        (4) Main loop.
26
      // Load tiles from a & b and compute.
     for (int i = 0; i < nnz; ++i)</pre>
29
       Tile<128, 128> tile_a = LoadTile(
30
         a, /*row=*/offset_a, /*column=*/i);
31
32
          (5) Load the column index from 'a'
33
       // for this non-zero block.
34
35
       // This indicates which row we need to
        // load from 'b'
36
37
       int row_b = a.column_idxs[offset_a + i];
       Tile<128, 128> tile_b = LoadTile(
  b, /*row=*/row_b, /*column=*/column);
38
39
40
       tile_c += tile_a * tile_b;
41
42
      // (4) Write output
43
44
     StoreTile(tile_c, c);
45 }
```

Figure 12. CUDA Pseudo-Code for Our DSD Kernel. We launch one threadblock per tile in the dense output. (1) On startup, each threadblock calculates the row and column indices of it's tile in the output. (2) Next, each threadblock loads the offset of the nonzero blocks for its row of a and calculates the number of non-zero blocks in it. (3) Next, each threadblock sets its accumulator tile to zero. (4) The threadblock steps through the a and b matrices, computing the product of two-dimensional tiles loaded from the input matrices and accumulating the results. (5) The row index of the tile to load from matrix b depends on the column index of the non-zero block loaded from a, which we load prior to loading from b. (6) Last, we store the final accumulated result to the output sparse matrix.

<span id="page-16-0"></span>![](_page_16_Figure_1.jpeg)

Figure 13. Block-Sparse Matrix Multiplication Throughput Compared to Triton Blocksparse. Benchmarked for the problem configurations used in training MoE-XS, MoE-Small and MoE-Medium models. For these problems, our block-sparse matrix multiplication kernels realize over 9× the throughput achieved by Blocksparse on average due to the overhead of sparse matrix preprocessing.