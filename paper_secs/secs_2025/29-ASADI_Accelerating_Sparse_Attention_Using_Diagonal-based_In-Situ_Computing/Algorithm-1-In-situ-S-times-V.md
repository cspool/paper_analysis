# **Algorithm 1** In-situ $S \times V$

**Require:** DIA format of  $S \in \mathbb{R}^{n \times \omega}$ , Matrix  $V \in \mathbb{R}^{n \times d}$ . **Ensure:** Output matrix  $Z \in \mathbb{R}^{n \times d}$ .

- 1: Mapping matrix V to  $d \times \text{ReRAM}$  arrays from  $\text{Arr}_0$  to  $\text{Arr}_{d-1} \in \mathbb{R}^n$ . Mapping DIA format of S from  $\text{Arr}_0$  to  $\text{Arr}_{d-1}$ , each array has  $\frac{\omega}{d}$  vectors  $\in \mathbb{R}^n$  (Figure 10 (a)).
- 2: **while**  $i < d \ (i = 0)$  **do**
- 3: For all arrays, performing vec-vec multi.  $I = S \times V$ .
- 4: Transfer  $S_0$  to  $Arr_i$ ,  $S_1$  to  $Arr_{i+1}$  and so on, i++.
- 5: end while (Figure 10 (b) and (c))
- 6: For all arrays, decompress matrices I (Fig. 10 (d)).
- 7: For all arrays, vec-vec accum.  $Z = \sum I$  (Figure 10 (e)).

**Details of computation paradigm.** In Figure 10 (a), each dimension of the V matrix is stored in a separate ReRAM array, totaling d = 2 arrays. The DIA format of the S matrix is evenly distributed on d = 2 ReRAM arrays, with each array storing  $\frac{\omega}{d}$  diagonals (line 1 of Alg. 1). Specifically, Arr<sub>0</sub> stores  $DI_{-1}$  and  $V_{d0}$ , while Arr<sub>1</sub> stores  $DI_0$  and  $V_{d1}$ . In Figure 10 (b), the in-situ vector-vector multiplication is performed on each array (line 3 of Alg. 1). In Figure 10 (c), we transfer the DIA vectors to different arrays, i.e.,  $DI_{-1} \rightarrow Arr_1$  and  $DI_0 \rightarrow Arr_0$ (line 4 of Alg. 1). The in-situ vector-vector multiplication is again performed on each array, and the results are stored in the fourth columns of Figure 10 (c). The decompression of the intermediate result matrices in each array follows the process shown in Figure 7 (d) and (e), while the decompression results are illustrated in Figure 10 (d) (line 6 of Alg. 1). Figure 10 (e) performs the in-situ vector-vector addition in each array to obtain all dimensions of the output Z matrix (line 7 of

![](_page_4_Figure_14.jpeg)

Fig. 11. (a) SDDMM between dense Q, K matrices, (b) In-situ computing with CSR format of matrix M, (c) In-situ computing with DIA format of matrix M

![](_page_4_Figure_16.jpeg)

Fig. 12. (a) Matrices Q and K in two ReRAM arrays, (b) Vector-vector multiplication of  $\mathrm{DI}_0$ , (c) Vector-vector multiplication of  $\mathrm{DI}_{-1}$ 

Alg. 1). To conserve memory space, we opt to decompress each diagonal sequentially. Specifically, we decompress two diagonals, sum them, and then proceed to decompress and add the remaining diagonals one by one.

