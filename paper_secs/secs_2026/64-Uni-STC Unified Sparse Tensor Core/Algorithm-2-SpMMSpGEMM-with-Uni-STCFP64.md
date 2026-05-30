# Algorithm 2: SpMM/SpGEMM with Uni-STC@FP64

```
1: warpid ← threadIdx.x >> 5
2: laneid ← threadIdx.x&31
3: row ← warpRowId[warpid]
4: for j ← Arow ptr[row] → Arow ptr[row + 1] do
5: Acol ← Aci[j]
6: A16b ← A16b ptr[j]
7: Av[8] ← load v(row, Acol, laneid)
8: Abi ← load bi(row, Acol, laneid)
9: % stc.load.a Av[0 ∼ 7]// Load 16 × 16 block data of matrix A
10: for Bj ← Brow ptr[Acol] → Brow ptr[Acol + 1] do
11: Bcol ← Bcol idx[Bj]
12: B16b ← B16b ptr[Bj]
13: if A16b × B16b and bf ind(Bcol) then
14: C16b ← Ccol idx[bf ind result]
15: Bbi ← load bi(Acol, Bcol, laneid)
16: Cbi ← load bi(row, Bcol, laneid)
17: % stc.load.meta mm A16b, B16b, C16b, Abi, Bbi, Cbi
18: % stc.task gen.mm// TMS and DPG generate T3 and T4 tasks
19: Bv[8] ← load v(Acol, Bcol, laneid)
20: Cv[8] ← load v(row, Bcol, laneid)
21: % stc.numeric.mm Bv[0 ∼ 7], Cv[0 ∼ 7]// SDPU execution
22: accumulate c(row, Bcol, laneid, Cv)
23: end if
24: end for
25: end for
```

