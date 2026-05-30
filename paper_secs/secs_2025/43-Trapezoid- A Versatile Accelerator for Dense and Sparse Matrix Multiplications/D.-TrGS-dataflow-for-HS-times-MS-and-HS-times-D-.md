# D. TrGS dataflow (for $HS \times MS$ and $HS \times D$ )

While TrGT minimizes traffic, which is the key for HS matrices, it has low peak arithmetic intensity. Our final dataflow, TrGS, is a novel Gustavson-based dataflow that processes rows of B *spatially*. TrGS leverages our spatial fabric's multipliers and cache bandwidth, and is useful for HS×MS and HS×D, which have higher arithmetic intensity than HS×HS.

Fig. 16 shows TrGS's loop nest. TrGS uses a PE row (not subrow) to compute a single row of C, by linearly combining rows of B. TrGS spatially maps A's rows (i.e., the  $M_0$  dimension in Fig. 16) across PE rows. TrGS's key feature is that it also

```
A = Matrix(shape=[M1,M0,K])
B = Matrix(shape=[N2,K,N1,N0])
C = Matrix(shape=[N2,M1,M0,N1,N0])

for n2 = [0, N2):
    for m1 = [0, M1):
```

<span id="page-8-2"></span>![](_page_8_Figure_1.jpeg)

Fig. 17: Example of Trapezoid running TrGS for HS $\times$ MS. spatially maps elements of each row of B (i.e., the  $N_0$  dimension

in Fig. 16) within each PE row. TrGS reuses existing hardware.

Each multiplier is responsible for producing a single element in the final C row; 128 elements of C's row will be produced after merging all the relevant B rows. The 16-word wide cache is able to provide 16 contiguous nonzeros of B per cycle to the PE row; the B distribution network routes these nonzeros to the corresponding multipliers using their n coordinates. In this way, the PE row running TrGS conducts 16 MACs/cycle, which is  $4 \times$  higher than TrGT (1 MAC/cycle/subrow). TrGS works well for HS×MS and HS×D, but not for MS×MS, because it leverages the fact that the B row is MS or D so that we can treat the N dimension as a dense dimension with low overhead. Walkthrough example: Fig. 17 shows a 4-multiplier PE row running TrGS to generate a row of C  $(C_{20}, C_{21}, C_{22}, C_{23})$  with a 2-word wide cache. Every multiplier is responsible for producing one final output element of C's row. According to the k coordinate of the nonzeros in A  $(A_{20}, A_{22})$ , the PE row needs to scale and accumulate B row 0  $(B_{00}, B_{02}, B_{03})$  and B row 2  $(B_{21}, B_{22}, B_{23})$ . The nonzeros of these two rows of B are streamed in from the cache in order. In this example we are currently working on B row 0, so the corresponding A value  $(A_{20})$  is broadcast to all multipliers using the A distribution network. 

• Because the cache is 2-word wide, the first two

nonzeros of B row 0 ( $B_{00}$ ,  $B_{02}$ ) are fetched. 2 The bitmasks of

the two elements are read in from cache and used to route the B distribution network. 3 The B distribution network routes these B values to the corresponding multipliers using their n coordinates:  $B_{00}$  is routed to multiplier 0;  $B_{02}$  is routed to multiplier 2. 4 A and B values are multiplied to produce partial products. 5 The reduction tree accumulates partial products  $(A_{20}B_{00}, A_{20}B_{02})$  with the partial results of  $C_{20}$  and  $C_{22}$ , respectively. 6 Later, the remaining nonzeros of B row 0  $(B_{03})$  and B row 2  $(B_{21}, B_{22}...)$  are fetched in wide 2-word accesses from the cache and accumulated. 7 Finally, when all the accumulation of B row 0 and B row 2 are done, the final result row of C  $(C_{20}, C_{21}, C_{22}, C_{23})$  is written to the cache.

### IV. METHODOLOGY

<span id="page-8-0"></span>**System:** We built a cycle-level simulator to evaluate Trapezoid, using the configuration shown in Table I. This configuration provides 32 TFLOP/s, using 128 PE rows, each with 128 FP32 multipliers and adders, running at 1 GHz. The 17 MB of onchip SRAM is organized as a 16MB cache (4MB/cluster); local buffers take an additional 1 MB. The system has 2TB/s HBM main memory, representative of modern GPUs and TPUs. We model the activities of all hardware components cycle by cycle, including MAC units, merge-reduction tree, distribution networks, multi-fiber intersection unit, local buffers, global cache, and HBM. We model contention and stalls faithfully.

**Baselines:** We compare Trapezoid against three state-of-the-art accelerators designed for matrix multiplication with D, MS, and HS inputs: TPU [33], SIGMA [60], and Flexagon [50]. Since TPU and SIGMA are also designed on top of a 2D spatial array, we size them with the same 128×128 spatial array as Trapezoid and a 16MB global scratchpad. SIGMA is also equipped with the same 1MB local buffer as Trapezoid.

The original Flexagon design has 64 MACs and a 1MB cache, which provide limited compute throughput (this is the case for other HS accelerators). We carefully scale it up to match Trapezoid's area by replicating 67 Flexagon instances without establishing all-to-all connections among instances (otherwise, the crossbar would completely dominate area). The scaled-up Flexagon system has  $67 \times 64$  MACs, and 67 MB of cache.

We model the baselines using the same simulation infrastructure described above. Our simulation results closely follow the performance numbers reported in the original papers.

**Area and energy:** We implement Trapezoid and baseline components in RTL and synthesize them in 45 nm using the FreePDK library [52]. We use CACTI7 [5] to estimate SRAM area in 45 nm. We then scale the area to 16 nm [59]. We present detailed area analysis in Sec. V-A. We obtain component energies using FreePDK15 [6] and Synopsys Design Compiler, and estimate HBM energy from prior work [18, 61].

**Workloads:** We evaluate 128 standalone matrix multiplication workloads (15 D×D, 15 MS×D, 38 MS×MS, 12 HS×D, 36 HS×MS, 12 HS×HS) and 8 DNNs (4 Llama, 2 ResNet, 2 VGG) with widely varying sparsity levels. Table III and Table IV list the matrices we use and their densities.

D and MS combinations use DNN workloads. For  $\mathbf{D} \times \mathbf{D}$ , we select 15 projection layers from the Llama-2-7B [67] large

TABLE I CONFIGURATION AND AREA BREAKDOWN OF TRAPEZOID.

<span id="page-9-0"></span>

| Component                     | Config                                          | $Area(mm^2)$ |
|-------------------------------|-------------------------------------------------|--------------|
| Vector multiplier             | 128× FP32 multiplier                            | 0.17         |
| Merge-reduction tree          | radix-128, FP32 adder                           | 0.13         |
| Distribution network          | 32b 128×128 Benes                               | 0.10         |
| Multi-fiber intersection unit | 4 rows & 4 columns                              | 0.12         |
| Local Buffer                  | 8KB, 4 banks, 16B-wide                          | 0.03         |
| PE row                        |                                                 | 0.54         |
| Compute overall               | 128×PE row                                      | 69.7         |
| Cache                         | 16MB, 128 banks, 16-waset-associative, 64B line | 10.2         |
| NoC                           | 4 64B 32×32 crossbar (cache banks ↔ 32 PE re    | 2.0          |
| Trapezoid Overall             | 1GHz, 128×128 MACs,<br>17MB SRAM, 2TB/s HE      | 81.9         |

