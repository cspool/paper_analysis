# Algorithm 1: SpMV/SpMSpV with Uni-STC@FP64

```
1: laneid ← threadIdx.x&31
2: row ← warpRowId[warpid]
3: start ← warpIndex[warpid]
4: end ← warpIndex[warpid + 1]
5: ry ← 0
6: j ← start
7: while j < end do
8: a4b ← load bitmap(laneid)
9: a4i ← load of f set(laneid)
10: rxb ← load bitmapx(laneid)
11: % stc.load.meta mv A16b[j], A16b[j + 1], rxb, a4b, a4i
12: % stc.task gen.mv// TMS and DPG generate T3 and T4 tasks
13: for i ← 0 → 15 do
14: rA[i] ← load value A(A val + a4i, a4b, laneid, i)
15: end for
16: % stc.load.a rA[0 ∼ 7]// Load 16 × 16 block data of matrix A
17: rx ← load value x(laneid)
18: % stc.numeric.mv rA[8 ∼ 15], rx, ry// SDPU execute T4 tasks
19: j+ = 2
20: end while
21: shf l gather(ry)
22: write back(ry, row, laneid)
```

