# D. The Swift SpMM Algorithms

The Swift method is composed of two algorithms: the first one computes the SpMM operation concerning the regular blocks of the sparse matrix A, and the second one takes care of the irregular blocks. Algorithm 2 presents the pseudocode of the regular part of the Swift SpMM algorithm. A thread block (configured as 32×8 threads) is assigned to compute 8 blocks of the sparse matrix and the corresponding 32 columns in the dense matrix. Swift allocates three shared memories per thread block:  $rowIdx\_sh$ ,  $val\_offset\_sh$ , and  $val\_sh$ . The variable tid represents the local thread index within the thread block (Line 4), and rid serves as the index into blkPtr, with every 32 threads corresponding to one of the blocks in the regular part. Swift first uses blkPtr and tid to assign each thread a specific value of position (Line 7) and the corresponding column index in the dense matrix (Lines 9–10). Then, the row indices and offset indices are loaded into shared memory (Lines 11-12). The for loop (Lines 13-31) iterates over the 32 assigned columns of the dense matrix. In each iteration, every thread computes a partial multiplication result and stores it into shared memory based on the position information (Lines 14–17). Next, segments with same row indices are identified using the offset array, and their corresponding values are locally reduced the number of atomic operation by adding them together (Lines 19-23). Finally, the accumulated results are written back to the output matrix using atomic add operations to ensure correctness (Lines 25–29).

Algorithm 3 presents the pseudocode of the irregular part of the Swift SpMM algorithm. To properly balance the load, Algorithm 3 divides the columns that are longer than the warpSize into several sub-columns. Mapping one warp to each column belonging to an irregular block brings significant load imbalance, as the example in Figure 6 shows. If the first warp handles the rightmost column of elements of irregular blocks (the last column in sortedMatrxA) and the second

**Algorithm 2:** A pseudocode of parallel Swift SpMM (regular part).

```
Input: M; N; K; blockNum; blkPtr[];
          blkCoIdx[]; value[]; rowIdx[]; matrixB[];
          positionIdx[]; offsetIdx[];
   Output: matrixC[]
1 extern __shared__ rowIdx_sh[];
2 \ extern \_shared\_ \ val\_offset\_sh[];
3 extern \_shared\_val\_sh[];
4 tid \leftarrow threadIdx.y << 5 + threadIdx.x;
5 \ rid \leftarrow blockDim.y * blockIdx.x + threadIdx.y;
\mathbf{6} if rid < blockNum then
      ptr \leftarrow blkPtr[rid] + threadIdx.x;
7
      position \leftarrow positionIdx[ptr];
      colIdxStart \leftarrow blkColIdx[rid];
      dnIndex \leftarrow blockIdx.y << 5;
10
      rowIdx\_sh[tid] \leftarrow rowIdx[ptr];
11
      val\ of\ fset\ sh[tid] \leftarrow of\ fsetIdx[ptr];
12
      for kk \leftarrow 0; kk < 32; kk + + do
13
           dnIdx \leftarrow (dnIndex + kk) * K +
14
            colIdxStart + threadIdx.x;
          dnVal \leftarrow matrixB[dnIdx];
15
          spVal = value[ptr];
16
          val\_sh[position] = spVal * dnVal;
17
           \_syncthreads();
18
          if val\_offset\_sh[tid] > 0 then
19
              for i = 1; i \le val\_offset\_sh[tid]; i + +
20
                  val\_sh[tid] + = val\_sh[tid + i];
21
              end
22
          end
23
             _syncthreads();
24
          if val of fset sh[tid]! = 0 then
25
              rowIndex \leftarrow rowIdx \ sh[tid];
26
              resultIdx \leftarrow
27
               rowIndex + (dnIndex + kk) * M;
              atomicAdd(matrixC[resultIdx], val\_sh[tid]);
28
          end
29
30
            \_syncthreads();
31
      end
32 end
```

warp handles the elements of the second column from the right, the number of irregular elements handled by the two warps is different (6 and 3 for the first and the second warps).

In Algorithm 3, irrNumBlk denotes the total number of irregular blocks, derived by summing the number of columns smaller than the warpSize and the ceiling of the quotient of larger columns divided by the warpSize. For example, in Figure 6, the number of irrNumBlk is 4 (assume warpSize is 4). The colIdxIndex array records whether the block is an independent column (where NNZ is less than warpSize) or a part of a column (where NNZ is greater than warpSize). The

**Algorithm 3:** A pseudocode of parallel Swift SpMM (irregular part).

```
Input: irrNumBlk; colIdxIndex[]; blkStart[]; blkStop[];
           N; K; irrPtr[];
           irrRowIdx[]; irrValue[]; matrixB[];
   Output: matrixC[]
1 \ globalId \leftarrow blockIdx.x * blockDim.x + threadIdx.x;
\textbf{2} \ warpId \leftarrow globalId >> 5;
3 laneId \leftarrow 31 \& threadIdx.x;
4 if warpId < irrNumBlk then
       colIdx \leftarrow colIdxIndex[warpId];
5
       signBit \leftarrow (colIdx >> 31) \& 0x1;
       colIdx\_1 \leftarrow signBit ==
7
        1? colIdx \& 0x7FFFFFFFF: colIdx;
       realColIdx \leftarrow K - colIdx_1 - 1;
8
       start \leftarrow signBit == 1 ? blkStart[warpId] :
        irrPtr[colIdx\_1];
       stop \leftarrow signBit == 1 ? blkStop[warpId] :
10
        irrPtr[colIdx\_1+1];
11
       for i \leftarrow start + laneId; i < stop; i+=32 do
          rowIndex \leftarrow irrRowIdx[i];
12
           spValue \leftarrow irrValue[i];
13
          for j \leftarrow 0; j < N; j + + do
14
              dnIdx \leftarrow j * K + realColIdx;
15
              resultIdx \leftarrow rowIndex * N + j;
16
              atomicAdd(matrixC[resultIdx], spValue \times
17
                matrixB[dnIdx]);
          end
18
19
       end
20 end
```

blkStart and blkStop array records the position of the block (where in the larger column). We use colIdxIndex to get the column index of each irregular block (Lines 5-8). Then we use blkStart, blkStop, and irrPtr array to get the position of each irregular block in irrValue and irrRowIdx (Lines 9-10). The computing part of Algorithm 3 (Lines 11-19) is similar to that of Algorithm 2.

#### V. PERFORMANCE EVALUATION

#### A. Experimental Setup

Our experimental platforms for evaluation of Swift are the NVIDIA GeForce RTX 4080s (default), GeForce RTX 3090Ti, Tesla V100, and A100 GPUs. The consider 2757 sparse matrices from SuiteSparse Matrix Collection [20]. The dense matrix is generated randomly considering two different number of columns (N=32 and N=128). Our evaluation of Swift includes both single (FP32) and double (FP64) precision SpMM algorithms. We compare Swift with four SOTA methods, ASpT [17], cuSPARSE v12.2 kernel [21], RoDe [10], and Sputnik [18]. To compare the performance of these four previous methods and Swift, we report the time they spend running SpMM on the GPU devices.

|  | TABLE I: Overall speedup of Swift over the four SOTA |  |  |  |
|--|------------------------------------------------------|--|--|--|
|  | baselines in terms of the geometric mean of time.    |  |  |  |

| Platform   | Configuration | ASpT | cuSPARSE | RoDe | Sputnik |
|------------|---------------|------|----------|------|---------|
|            | FP64, N=32    | 2.22 | 59.19    | 5.16 | 10.92   |
|            | FP64, N=128   | 1.79 | 27.02    | 3.62 | 6.53    |
| RTX 4080s  | FP32, N=32    | 1.74 | 61.59    | 3.99 | 8.80    |
|            | FP32, N=128   | 1.19 | 28.09    | 2.46 | 4.83    |
|            | FP64, N=32    | 2.43 | 51.35    | 4.96 | 9.89    |
|            | FP64, N=128   | 1.57 | 19.42    | 2.92 | 5.05    |
| RTX 3090Ti | FP32, N=32    | 1.78 | 48.89    | 3.55 | 7.44    |
|            | FP32, N=128   | 1.04 | 19.11    | 1.83 | 3.58    |
|            | FP64, N=32    | 2.79 | 92.82    | 7.41 | 18.38   |
|            | FP64, N=128   | 1.85 | 55.48    | 4.68 | 10.29   |
| A100       | FP32, N=32    | 2.45 | 93.06    | 6.03 | 13.96   |
|            | FP32, N=128   | 1.66 | 51.87    | 3.72 | 8.55    |
|            | FP64, N=32    | 2.51 | 85.35    | 5.82 | 13.92   |
|            | FP64, N=128   | 1.62 | 39.60    | 3.82 | 7.42    |
| V100       | FP32, N=32    | 2.39 | 91.14    | 6.25 | 14.94   |
|            | FP32, N=128   | 1.54 | 46.03    | 3.55 | 7.40    |

![](_page_7_Figure_2.jpeg)

Fig. 8: Time comparison of Swift with SOTA methods on RTX 4080s (FP64, N=128).

