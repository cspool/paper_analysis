# <span id="page-5-0"></span>5.1 Sampling and Estimation

To accurately assess the performance of each tiling scheme, we need to know the computation amount and the data sizes. Besides the easily known non-zero sizes of the input tensors, we mainly need to estimate two critical metrics: effMAC as the expected effectual (i.e., non-zero) MAC number, and nnzCTk as the non-zero access traffic size of the output tensor C under each tiling factor  $T_k$  of dimension k when repetitive accesses are accounted for.

Algorithm 2: Estimating effMAC and nnzCTk.

```
1 function ESTEFFMAC(A, B, S_I, S_J, sp):
         sum \leftarrow 0
2
         for k \leftarrow 0 to K do
 3
               szA \leftarrow \text{Number of non-zeros in } A[S_I][k];
 4
               szB \leftarrow \text{Number of non-zeros in } B[k][S_I];
 5
               sum += szA \times szB;
         return \frac{sum}{sp^2}
8 function EstNnzCT\kappa(A, B, S_I, S_J, \text{sp, sk}):
         function EstNnzC(N_k, A, B, S_I, S_I, sp, sk):
               R[N_k] \leftarrow A list of empty min-heaps, each of size sk;
               for k \leftarrow 0 to K do
11
                     t \leftarrow \frac{k}{K/N_L};
12
                     a \leftarrow \text{Non-zero indices of } A[S_I][k] \text{ sorted by } h_1();
13
                     b \leftarrow \text{Non-zero indices of } B[k][S_I] \text{ sorted by } h_2();
14
                     foreach i, j in a, b do
15
                          v \leftarrow (h_1(i) - h_2(j)) \mod 1;
16
                          Add ((i, j), v) to R[t], if not already existing;
17
               R_{\text{all}} \leftarrow \text{Deduplicate} \text{ and merge sort } R[N_k];
18
               v_{sk} \leftarrow The sk-th smallest value in R_{all};
19
               sum \leftarrow \sum |\{v \leq v_{sk}|v \in R[t]\}|;
20
              return \frac{3um}{v_{\rm sk} \cdot {\rm sp}^2}
21
         \mathsf{nnzCTk} \leftarrow [\,];
22
23
           nnzCTk.Append(EstNnzC(K/T_k, A, B, S_I, S_J, sp, sk));
24
         return nnzCTk;
```

Note that both the effectual MACs and the non-zero size of C depend not only on the sparsities of the two inputs A and B, but more importantly also on the correlation between their non-zero distributions. Due to the complex relationship, we adopt sampling-based approaches to efficiently estimate these metrics. We sample a small fraction (denoted as sp) of rows and columns from A and B (Algorithm 1 Lines 1 to 2). If the compressed formats are compatible, e.g., getting rows/columns from CSR/CSC, the sampling is straightforward. If the formats mismatch, e.g., extracting a column from CSR, we sample a set of points uniformly from the entire matrix, and re-group these points into the desired rows/columns.

With the sampled matrices, our estimation methods are summarized in Algorithm 2. For effMAC, we iterate over the contracted dimension k, and multiply the non-zero sizes of the sampled A column and B row pair, to get the MAC number of their outer-product. The total estimated MAC number is the sum scaled by  $\operatorname{sp}^2$ .

To estimate nnzCTk, we borrow from previous techniques [2, 3]. We first consider nnzCTk for  $T_k = K$ , i.e., no tiling. As in Algorithm 2 Lines 15 to 17, for each non-zero element pair A[i][k] and B[k][j] in the sampled matrices, we add (i,j) into a min-heap with the value  $h(i,j) = (h_1(i) - h_2(j))$  mod 1, where  $h_1$  and  $h_2$  are two hash functions to fixed-point numbers, and "mod 1" extracts the fractional part of the result. The same (i,j) pairs for different k are deduplicated in the heap. Then  $\operatorname{nnzCTk}_{T_k=K}$  can be estimated as  $\operatorname{sk}/(v_{\operatorname{sk}} \cdot \operatorname{sp}^2)$ , where  $v_{\operatorname{sk}}$  is the sk-th smallest h(i,j) value.

The estimation of nnzCTk for other values of  $T_k$  can be done similarly as in Algorithm 2. However, naively doing so would incur  $O(\log K)$  invocations of the above procedure (Lines 23 to 24). We apply two optimizations. First, we use the  $v_{\rm sk}$  value obtained in the  $T_k=K$  case to filter out most of the (i,j) pairs, and only process the pairs within the sk-th smallest. These pairs are kept in a queue during the  $T_k=K$  invocation. Second, we only iterate these (i,j) pairs once, to build  $N_{k,\rm max}=4096$  deduplicated heaps  $R[N_{k,\rm max}]$  for  $T_{k,\rm min}=K/N_{k,\rm max}$ . For  $T_{k\times 2}$ , we merge each pair of adjacent heaps of  $T_k$  with deduplication. We do this recursively to calculate nnzCTk for all  $T_k$  values. With these optimizations, the extra cost of estimating for other  $T_k$  values can be reduced to 10% of that for  $T_k=K$  for most matrices.

The above estimation involves two hyperparameters, whose default values are chosen as sp =  $1/\sqrt{N}$  and sk =  $\sqrt{N}$  (where N represents the corresponding dimension I, J, or K), to strike a balance between estimation accuracy and computational cost. This follows the theoretically proved suggestion of sp · sk  $\leq$  1 [2]. The empirical time cost is further evaluated in Section 8.4.

