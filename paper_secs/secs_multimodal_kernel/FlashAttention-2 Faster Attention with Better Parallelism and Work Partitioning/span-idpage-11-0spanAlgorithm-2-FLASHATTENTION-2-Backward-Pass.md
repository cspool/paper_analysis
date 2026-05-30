# <span id="page-11-0"></span>Algorithm 2 FLASHATTENTION-2 Backward Pass

**Require:** Matrices  $Q,K,V,O,dO \in \mathbb{R}^{N \times d}$  in HBM, vector  $L \in \mathbb{R}^N$  in HBM, block sizes  $B_c, B_r$ .

- 1: Divide **Q** into  $T_r = \left\lceil \frac{N}{B_r} \right\rceil$  blocks  $\mathbf{Q}_1, ..., \mathbf{Q}_{T_r}$  of size  $B_r \times d$  each, and divide  $\mathbf{K}, \mathbf{V}$  in to  $T_c = \left\lceil \frac{N}{B_c} \right\rceil$  blocks  $\mathbf{K}_1, ..., \mathbf{K}_{T_c}$  and  $\mathbf{V}_1, ..., \mathbf{V}_{T_c}$ , of size  $B_c \times d$  each.
- 2: Divide **O** into  $T_r$  blocks  $\mathbf{O}_i,...,\mathbf{O}_{T_r}$  of size  $B_r \times d$  each, divide  $\mathbf{dO}$  into  $T_r$  blocks  $\mathbf{dO}_i,...,\mathbf{dO}_{T_r}$  of size  $B_r \times d$  each, and divide L into  $T_r$  blocks  $L_i,...,L_{T_r}$  of size  $B_r$  each.
- 3: Initialize  $d\mathbf{Q} = (0)_{N \times d}$  in HBM and divide it into  $T_r$  blocks  $d\mathbf{Q}_1, ..., d\mathbf{Q}_{T_r}$  of size  $B_r \times d$  each. Divide  $d\mathbf{K}, d\mathbf{V} \in \mathbb{R}^{N \times d}$  in to  $T_c$  blocks  $d\mathbf{K}_1, ..., d\mathbf{K}_{T_c}$  and  $d\mathbf{V}_1, ..., d\mathbf{V}_{T_c}$ , of size  $B_c \times d$  each.
- 4: Compute  $D = \text{rowsum}(\mathbf{dO} \circ \mathbf{O}) \in \mathbb{R}^d$  (pointwise multiply), write D to HBM and divide it into  $T_r$  blocks  $D_1,...,D_{T_r}$  of size  $B_r$  each.
- 5: **for**  $1 \le j \le T_c$  **do**
- 6: Load  $\mathbf{K}_i$ ,  $\mathbf{V}_i$  from HBM to on-chip SRAM.
- 7: Initialize  $d\mathbf{K}_i = (0)_{B_c \times d}$ ,  $d\mathbf{V}_i = (0)_{B_c \times d}$  on SRAM.
- 8: **for**  $1 \le i \le T_r$  **do**
- 9: Load  $\mathbf{Q}_i$ ,  $\mathbf{d}\mathbf{Q}_i$ ,  $\mathbf{d}\mathbf{Q}_i$ ,  $L_i$ ,  $D_i$  from HBM to on-chip SRAM.
- 10: On chip, compute  $\mathbf{S}_{i}^{(j)} = \mathbf{Q}_{i} \mathbf{K}_{i}^{T} \in \mathbb{R}^{B_{r} \times B_{c}}$ .
- 11: On chip, compute  $\mathbf{P}_{i}^{(j)} = \exp(\mathbf{S}_{ij} L_{i}) \in \mathbb{R}^{B_r \times B_c}$ .
- 12: On chip, compute  $d\mathbf{V}_{i} \leftarrow d\mathbf{V}_{i} + (\mathbf{P}_{i}^{(j)})^{\mathsf{T}} d\mathbf{O}_{i} \in \mathbb{R}^{B_{c} \times d}$ .
- 13: On chip, compute  $d\mathbf{P}_{i}^{(j)} = d\mathbf{O}_{i} \mathbf{V}_{j}^{\mathsf{T}} \in \mathbb{R}^{B_{r} \times B_{c}}$ .
- 14: On chip, compute  $d\mathbf{S}_{i}^{(j)} = \mathbf{P}_{i}^{(j)} \circ (d\mathbf{P}_{i}^{(j)} D_{i}) \in \mathbb{R}^{B_{r} \times B_{c}}$ .
- 15: Load  $d\mathbf{Q}_i$  from HBM to SRAM, then on chip, update  $d\mathbf{Q}_i \leftarrow d\mathbf{Q}_i + d\mathbf{S}_i^{(j)}\mathbf{K}_j \in \mathbb{R}^{B_r \times d}$ , and write back to HBM.
- 16: On chip, compute  $\mathbf{dK}_i \leftarrow \mathbf{dK}_i + \mathbf{dS}_i^{(j)^{\top}} \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$ .
- 17: **end for**
- 18: Write  $d\mathbf{K}_j$ ,  $d\mathbf{V}_j$  to HBM.
- 19: **end for**
- 20: Return dQ,dK,dV.

