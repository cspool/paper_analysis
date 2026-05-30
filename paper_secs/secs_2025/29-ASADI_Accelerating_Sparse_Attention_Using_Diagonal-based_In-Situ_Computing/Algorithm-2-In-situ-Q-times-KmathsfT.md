# **Algorithm 2** In-situ $Q \times K^{\mathsf{T}}$

**Require:** DIA format of  $M \in \mathbb{R}^{n \times \omega}$ , Matrix  $Q, K \in \mathbb{R}^{n \times d}$ . **Ensure:** Output matrix  $S \in \mathbb{R}^{n \times \omega}$ .

- 1: Mapping matrix Q and K to  $d \times \text{ReRAM}$  arrays from  $\text{Arr}_0$  (stored  $Q_0$  and  $K_0$ ) to  $\text{Arr}_{d-1}$  (stored  $Q_{d-1}$  and  $K_{d-1}$ )  $\in \mathbb{R}^n$ . Matrix M is stored on-chip (Figure 12 (a)).
- 2: **while**  $i < \omega$  (i = 0) **do**
- 3: Transfer  $M_i$  to the Row Selector of all arrays.
- 4: For all arrays, shift up/down Q using  $M_i$ 's DIA index.
- 5: For all arrays, copy  $Q_{Ro}$  to  $Q_{Rd}$  using  $M_i$ 's  $(R_d, R_o)$ .
- 6: For all arrays, vec-vec multi.  $Slices_{Si} = Q \times K$ .
- 7: Restore  $Q_{Rd}$  for next iteration, i + +.
- 8: end while (Figure 12 (b), (c) and Figure 13 (a))
- 9: Transfer all *Slices<sub>Si</sub>* to Arr<sub>i</sub>. (Figure 13 (b) and (c)).
- 10: For Arr<sub>i</sub>, vec-vec accum.  $S_i = \sum Slices_{Si}$  (Figure 13 (d)).

As Figure 12 (b) shows, the DI<sub>0</sub> vector of matrix M ( $M_0$ ), along with the ( $R_d$ ,  $R_o$ ) list of this vector, will be transmitted to the row selector (line 3 of Alg. 2). Then, the memory controller shifts the  $Q_0$  up/down according to DI (diagonal index) of  $M_0$  (line 4 of Alg. 2). If DI = DI<sub>0</sub>, nothing is done; if DI = DI<sub>j</sub>,  $Q_0$  is shifted down  $j \times$ ; if DI = DI<sub>-j</sub>,  $Q_0$  is shifted up  $j \times$ . Next, the memory controller executes a memory copy operation based on the ( $R_d$ ,  $R_o$ ) list of grey cells in  $M_0$  (line 5 of Alg. 2). Specifically, the memory controller copies  $Q_{00}$  ( $Q_{Ro}$ ) to  $Q_{30}$  ( $Q_{Rd}$ ). Afterward, each array performs the in-situ vector-vector multiplication to derive one slice  $Slices_{S0}$  of the S vector (line 6 of Alg. 2).

Figure 12 (c) presents the calculation of  $DI_{-1}$  of the mask matrix M ( $M_1$ ). The memory controller will restore  $Q_{30}$  of  $Q_0$  (line 7 of Alg. 2), which are modified in the previous iteration. First,  $M_1$  with  $DI_{-1}$  is sent to the row selector (line 3).  $DI = DI_{-1}$ , so  $Q_0$  will shift up  $1 \times$  (line 4). Then, the memory controller will perform a memory copy according to the ( $R_d$ ,  $R_o$ ) list of grey cells in  $M_{-1}$  (line 5). Specifically,  $Q_{10}$  will be copied to  $Q_{50}$ . Finally, ReRAM arrays will perform vector-vector multiplication  $Slices_{S1} = Q \times K$  to get one slice of the S vector (line 6).

Figure 13 (a) depicts the results of the two dimensions of  $Slices_{Si} = Q \times K$ , which are stored in  $Arr_0$  and  $Arr_1$ , each holding  $Slices_{S0}$  and  $Slices_{S1}$  of the DIA-based S matrix. We then transmit all  $Slices_{S0}$  to  $Arr_0$  and all  $Slices_{S1}$  to  $Arr_1$ , as

![](_page_5_Figure_18.jpeg)

Fig. 14. (a) Using analog in-situ computing for the linear layer, (b) Different dimensions of matrices Q, K, and V are stored in different ReRAM arrays

shown in Figure 13 (b) and (c) (line 9 of Alg. 2). Finally, we perform in-situ vector-vector addition  $S_i = Slices_{Si} + Slices_{Si}$  on all ReRAM arrays (line 10 of Alg. 2), obtaining the DIA format of matrix S illustrated in Figure 13 (d).

