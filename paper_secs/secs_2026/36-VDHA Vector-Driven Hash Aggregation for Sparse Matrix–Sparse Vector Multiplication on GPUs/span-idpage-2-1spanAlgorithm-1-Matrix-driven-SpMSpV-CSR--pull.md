# <span id="page-2-1"></span>Algorithm 1 Matrix-driven SpMSpV (CSR / pull)

**Input:** *A* in CSR: (row\_ptr, indices, values) with *N* rows; dense vector *x* with value array *x* and bitmask *bm*;

**Output:** Result vector *y* 

```
1: for all r \leftarrow 0 to N in parallel do
2: start \leftarrow row\_ptr[r], end \leftarrow row\_ptr[r+1]
3: res \leftarrow 0
4: for j \leftarrow start to end do
5: col \leftarrow indices[j]
6: mask \leftarrow bm[col]
7: if mask then
8: res \leftarrow res \oplus (values[j] \otimes x[col])
9: y[r] \leftarrow res
```

In contrast, the vector-driven paradigm iterates only over the nonzeros of  $\mathbf{x}$ . Algorithm 2 illustrates the vector-driven paradigm of SpMSpV. For each active entry in  $\mathbf{x}$ , the corresponding column of the CSC matrix is fetched (line 4, the fetch step), and the partial products (i.e.,  $mat\_val \otimes v\_val$ ) are generated toward the result vector according to the row indices of the column. These partial results are then written back to  $\mathbf{y}$  (line 7, the write\_back step). Representative column-major SpMSpV approaches include FastSpMSpV [40] and GPU graph frameworks such as Gunrock [34]. Hybrid methods such as Adaptive SpMSpV [20] combine both paradigms depending on vector sparsity.

