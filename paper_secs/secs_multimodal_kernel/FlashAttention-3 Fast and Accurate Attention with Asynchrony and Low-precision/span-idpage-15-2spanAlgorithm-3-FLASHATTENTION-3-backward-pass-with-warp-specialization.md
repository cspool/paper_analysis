# <span id="page-15-2"></span>Algorithm 3 FLASHATTENTION-3 backward pass with warp specialization

```
Require: Matrices \mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO} \in \mathbb{R}^{N \times d} in HBM, logsum exp vector L \in \mathbb{R}^N in HBM, block sizes
      B_c, B_r.
```

- 1: In a preprocessing kernel, compute  $D = \text{rowsum}(\mathbf{dO} \circ \mathbf{O}) \in \mathbb{R}^d$  (pointwise multiply), write D to HBM and divide it into  $T_r$  blocks  $D_1,...,D_{T_r}$  of size  $B_r$  each.
- 2: Divide **Q** into  $T_r = \left\lceil \frac{N}{B_r} \right\rceil$  blocks  $\mathbf{Q}_1, ..., \mathbf{Q}_{T_r}$  of size  $B_r \times d$  each, and divide  $\mathbf{K}, \mathbf{V}$  in to  $T_c = \left\lceil \frac{N}{B_c} \right\rceil$  blocks  $\mathbf{K}_1, ..., \mathbf{K}_{T_c}$  and  $\mathbf{V}_1, ..., \mathbf{V}_{T_c}$ , of size  $B_c \times d$  each.
- 3: Divide **dO** into  $T_r$  blocks **dO**<sub>i</sub>,...,**dO**<sub>T<sub>r</sub></sub> of size  $B_r \times d$  each, and divide L into  $T_r$  blocks  $L_i,...,L_{T_r}$ of size  $B_r$  each.
- 4: Initialize pipeline object to manage barrier synchronization with s-stage circular SMEM buffer.
- 5: **if** in producer warpgroup **then**
- Deallocate predetermined number of registers.
- Issue load  $\mathbf{K}_i$  and  $\mathbf{V}_i$  from HBM to shared memory. 7:
- Upon completion, commit to notify consumer of the load of  $\mathbf{K}_i$  and  $\mathbf{V}_i$ . 8:
- 9: for  $1 \le i \le T_r$  do
- 10: Wait for the (i%s)th stage of the buffer to be consumed.
- 11: Issue loads of  $\mathbf{Q}_i$ ,  $\mathbf{dQ}_i$  from HBM to shared memory at the (i%s) th stage of the buffer.
- 12: Upon completion, commit to notify consumers of the loads of  $\mathbf{Q}_i$ ,  $\mathbf{dO}_i$ .
- 13: end for
- 14: else if in consumer warpgroups then
- Reallocate predetermined number of registers as function of number of consumer warps. 15:
- On-chip, Initialize  $\mathbf{dK}_i = (0)_{B_c \times d}, \mathbf{dV}_i = (0)_{B_c \times d}$ . 16:
- Wait for  $\mathbf{K}_i$  and  $\mathbf{V}_i$  to be loaded in shared memory. 17:
- 18: for  $1 \le i \le T_r$  do
- 19: Wait for  $\mathbf{Q}_i$  to be loaded in shared memory.
- Load  $L_i$ , $D_i$  from HBM to on-chip SRAM. 20:
- On chip, compute  $\mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c}$  (SS-GEMM). Commit. Wait for  $\mathbf{dO}_i$  to be loaded in shared memory. 21:
- 22:
- On chip, compute  $\mathbf{dP}_i^{(j)} = \mathbf{dO}_i \mathbf{V}_i^{\mathsf{T}} \in \mathbb{R}^{B_r \times B_c}$  (SS-GEMM). Commit. 23:
- 24:
- On chip, wait for  $\mathbf{S}_i^{(j)}$ , then compute  $\mathbf{P}_i^{(j)} = \exp(\mathbf{S}_{ij} L_i) \in \mathbb{R}^{B_r \times B_c}$ . On chip, wait for  $\mathbf{dP}_i^{(j)}$ , then compute  $\mathbf{dS}_i^{(j)} = \mathbf{P}_i^{(j)} \circ (\mathbf{dP}_i^{(j)} D_i) \in \mathbb{R}^{B_r \times B_c}$ . 25:
- On chip, compute  $\mathbf{dV}_j \leftarrow \mathbf{dV}_j + (\mathbf{P}_i^{(j)})^\top \mathbf{dO}_i \in \mathbb{R}^{B_c \times d}$  (RS-GEMM). Commit. 26:
- On chip, compute  $\mathbf{dK}_i \leftarrow \mathbf{dK}_i + \mathbf{dS}_i^{(j)^{\top}} \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$  (RS-GEMM). Commit and wait for both 27:
- On chip, compute  $\mathbf{dQ}_i^{(local)} = \mathbf{dS}_i^{(j)} \mathbf{K}_i \in \mathbb{R}^{B_r \times d}$  (SS-GEMM), and write  $\mathbf{dQ}_i^{(local)}$  to smem. 28: Notify the **dQ**-writer.
- 29: end for
- 30: **else if** in **dQ**-writer warp **then**
- 31: for  $1 \le i \le T_r$  do
- Wait for  $\mathbf{dQ}_i^{(local)}$  to be ready in smem. 32:
- Using a semaphore, atomically add  $d\mathbf{Q}_i^{(local)}$  to  $d\mathbf{Q}_i$  in global memory. 33:
- 34: end for
- <span id="page-15-1"></span>35: **end if**

### B.2 2-Stage Pipelining SASS Analysis

We give simplified SASS code for the inside of the consumer warpgroup mainloop.

```
// Compute row_max
FMNMX.FTZ R0, R24, R6, !PT ;
SHFL.BFLY PT, R185, R2, 0x2, 0x1f ;
... FMNMX and SHFL.BFLY ...
// Apply exp2 and row_sum. Rescale O.
FMUL.FTZ R2, R4, UR9 ;
MUFU.EX2 R185, R184 ;
FFMA.FTZ R24, R24, UR9, -R6.reuse ;
FADD.FTZ R24, R211, R24 ;
... FMUL, FFMA, FMUL, MUFU.EX2, FADD ...
// FP32 -> FP16 conversion are interleaved with exp2, row_sum and O rescaling.
F2FP.F16.F32.PACK_AB R231, R25, R231 ;
... F2FP, FMUL, MUFU, FFMA, FADD ...
// Start the first WGMMA. Broken down into 8 HGMMAs.
// The first 7 HGMMAs are packed together.
WARPGROUP.ARRIVE ;
HGMMA.64x192x16.F32 R24, gdesc[UR44], RZ, !UPT ;
... HGMMA x 6 ...
// FP32->FP16, exp2, row_sum, O rescaling are interleaved with HGMMA.
F2FP.F16.F32.PACK_AB R214, R214, R187 ;
MUFU.EX2 R234, R5 ;
FADD.FTZ R237, R187, R2 ;
... F2FP, MUFU, FADD ...
// The last HGMMA is issued here. No need to wait.
HGMMA.64x192x16.F32 R24, gdesc[UR44], R24, gsb0 ;
// Start the second WGMMA. Broken down into 12 HGMMAs.
// All 12 HGMMAs are packed together. Not interleaved with other instructions.
WARPGROUP.ARRIVE ;
HGMMA.64x128x16.F32 R120, R228, gdesc[UR8].tnspB, R120 ;
... HGMMA x 10 ...
HGMMA.64x128x16.F32 R120, R184, gdesc[UR8].tnspB, R120, gsb0 ;
// wgmma.wait_group at the end.
WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
```

We make the following observations:

- 1. Softmax is reordered to the very beginning, even before the first WGMMA.
- 2. The first WGMMA is interleaved with softmax and FP32 → FP16 datatype conversion of **S**. This indicates that WGMMA and non-WGMMAs are executed in parallel.
- 3. exp2, row\\_sum, O rescaling and FP32 → FP16 conversions are interleaved together.
- 4. The second WGMMA is not overlapped with other instructions, as expected.

<span id="page-16-0"></span>Overall, SASS shows that the 2-stage pipelining idea works as expected.

