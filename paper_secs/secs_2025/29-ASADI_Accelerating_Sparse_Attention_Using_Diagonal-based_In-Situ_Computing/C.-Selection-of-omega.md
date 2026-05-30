# C. Selection of $\omega$

We refer to the central  $\omega$  diagonal region as  $\omega$  region while referring the rest as other region. Figure 8 presents the definition of *NAE* (number of all elements), *NZ* (number of zeros), and *NNZ* (number of non-zeros). Our compression method will move non-zero values in other region to the bubbles of  $\omega$  region. Therefore, it is necessary to ensure that  $NZ_{\omega}$  (*NZ* of  $\omega$  region) is slightly bigger than  $NNZ_{o}$  (*NNZ* of other region). For quantitative analysis, we assume that the sparsity of the  $n \times n$  attention matrix is 10% (common scenario in Sanger [22]). From Figure 5, we get  $\frac{NNZ_{\omega}}{NNZ} = 40\%, 50\%, 70\%$ 

![](_page_3_Figure_8.jpeg)

Fig. 9. (a) Matrix multiplication between sparse S matrix and dense V matrix, (b) In-situ computing with CSR format of matrix S, (c) In-situ computing with DIA format of matrix S

when  $\omega = \frac{n}{16}, \frac{n}{8}, \frac{n}{4}$ , respectively. Figure 8 presents how we calculate  $NZ_{\omega}$  and  $NNZ_{o}$  when  $\omega = \frac{n}{16}, \frac{n}{8}, \frac{n}{4}$ . If  $NZ_{\omega}$  is much bigger than  $NNZ_{o}$ , it means we have too many bubbles in the  $\omega$  region. If  $NNZ_{o}$  is much bigger than  $NZ_{\omega}$ , it means we do not have enough bubbles for all grey cells. Therefore, we choose  $\frac{n}{8}$  as the default configuration of  $\omega$ .

## IV. DIA-BASED IN-SITU COMPUTING

Figure 1 depicts the attention mechanism, comprising of four operations: linear layer,  $Q \times K^{\mathsf{T}}$ , softmax, and  $S \times V$ . In this section, we will explain how we implement in-situ computing to process these four operations. For exposition purposes, we assume a sequence length of n=6 and a model dimension of d=2 for the Q, K, and V matrices. Additionally, we assume a diagonal window size of  $\omega=2$  for both the sparse S matrix and the sparse mask matrix M. We utilize the diagonal index (DI) to mark the diagonals of the DIA format, where DI<sub>0</sub> represents the center-most diagonal.

## A. In-situ $S \times V$

**High-level motivations.** Figure 9 (a) illustrates the sparse S matrix and dense V matrix. For brevity, we select two diagonals  $\mathrm{DI}_{-1}$  and  $\mathrm{DI}_0$  from Figure 7 (c). Figure 9 (b) presents the in-situ computation paradigm of the CSR format, where we assume that the CSR format of matrix S and the dense matrix V are stored in the same ReRAM array. The fundamental concept of matrix multiplication involves coordinates alignment, aligning the column coordinates of the left matrix with the row coordinates of the right matrix. However, the CSR storage format breaks the column coordinates of the left S matrix, preventing its direct use for in-situ matrix multiplication. Consequently, a row-wise remapping phase is necessary to align the coordinates of the left and right matrices.

In the  $\bigcirc$  iteration (red arrow), the first row of the CSR format will be remapped for coordinates alignment. Subsequently, the column marked by the red arrow will be insitu computed with matrix V to generate the first row of the output matrix (two valid computing with non-zeros while four invalid computing with zeros). In the  $\bigcirc$  iteration (green arrow), the second row of the CSR format will be remapped and computed with matrix V (three valid and three invalid). After five iterations, output matrix Z is generated. Figure 9 (c) presents the mapping of the DIA storage format of matrix S and the dense matrix V. As the DIA format does not break the column coordinates of the left matrix, it enables direct use for matrix multiplication without remapping. In our example,

![](_page_4_Figure_0.jpeg)

Fig. 10. (a) Mapping matrices S and V to two ReRAM arrays, (b) Intermediate results of the first iteration of vector-vector multiplication, (c) Intermediate results of the second iteration of vector-vector multiplication, (d) Decompressed intermediate results, (e) Output Z matrix

two DIA iterations (five valid computing and one invalid computing) are sufficient to complete the calculation.

**Quantitative analysis.** In the above example, each diagonal has an average of five elements, while each row has an average of two elements. This means that the number of DIA's valid computing is  $2.5\times$  greater than the CSR format. Thus, the number of CSR iterations is exactly  $2.5\times$  that of DIA iterations. given that the diagonal locality of Longformer is  $7.5\times$  greater than the row locality (as observed in Section III), the DIA-based computation paradigm can save  $7.5\times$  iterations than the CSR computation paradigm, due to the poor row locality and the presence of numerous bubbles in the ReRAM arrays at each CSR iteration.

