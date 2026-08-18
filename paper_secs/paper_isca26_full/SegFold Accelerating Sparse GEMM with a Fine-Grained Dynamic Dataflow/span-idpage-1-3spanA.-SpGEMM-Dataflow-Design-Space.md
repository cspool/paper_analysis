# <span id="page-1-3"></span>*A. SpGEMM Dataflow Design Space*

We use A, B to denote the input of SpGEMM, and C for the output. The arithmetic representation for SpGEMM is:

$$C_{m,n} = \sum_{k} A_{m,k} \times B_{k,n}, \tag{1}$$

where both A and B are sparse. The iteration space

$$S = \{ (m, n, k) \mid 0 \le m < M, \ 0 \le n < N, \ 0 \le k < K \},\$$

can be broken into multiplications and additions as:

$$T_{m,n,k} = A_{m,k} \times B_{k,n},\tag{2}$$

$$C_{m,n} = \sum_{k} T_{m,n,k}.$$
 (3)

T is the product of A and B elements paired by the shared index k; C is the reduction of T along the K dimension.

The major aspects of SpGEMM's dataflow design are:

- 1) Tiling strategy for M, N and K.
- 2) Scheduling of all dimensions (M, N, K plus tiling subdimensions) onto spatial and temporal axes.
- 3) Fusion of the computation in Eq. [2](#page-1-0) and Eq. [3.](#page-1-1)

Our work focuses on the second and third aspects – i.e., the scheduling and fusion of work at a fine-grain within a tile. In this paper, we discuss three representative dataflows—inner product, outer product, and Gustavson. We next explain how each dataflow uses different scheduling and fusion strategies.

TABLE I: Taxonomy of SpGEMM accelerators.

<span id="page-1-2"></span>

| Accelerator     | Scheduling                   | Reconfigurability to<br>Balance Load & Comp.                  |  |  |  |
|-----------------|------------------------------|---------------------------------------------------------------|--|--|--|
| TPU [21]        | –                            | –                                                             |  |  |  |
| ExTensor [17]   | IP                           | –                                                             |  |  |  |
| SIGMA [39]      | IP                           | Reconfigurable reduction network                              |  |  |  |
| OuterSpace [33] | OP                           | None                                                          |  |  |  |
| SpArch [54]     | OP                           | Data-dependent comparator arrays                              |  |  |  |
| MatRaptor [43]  | Gust                         | Comparator queues                                             |  |  |  |
| Gamma [51]      | Gust                         | None                                                          |  |  |  |
| Flexagon [28]   | IP/OP/Gust                   | Reconfigurable distribution, merge,<br>and reduction networks |  |  |  |
| Trapezoid [49]  | IP/Gust                      | Reconfigurable distribution, merge,                           |  |  |  |
|                 | (spatial/temporal)           | and reduction networks                                        |  |  |  |
| Spada [24]      | Window/Tile<br>size Adaptive | Neighboring-lane work stealing                                |  |  |  |
| SegFold (ours)  | Sub-tile Dynamic<br>dataflow | Sub-tile scheduling & Reconfig<br>urable merge network        |  |  |  |

Static Dataflow Definitions. *Inner product* (order M ✮N ✮K) performs dot products between rows of A and columns of B. *Outer product* (order K ✮ M ✮ N) performs cross products between column vectors from A and row vectors from B. *Gustavson* (order M ✮K ✮N) performs row products between elements Am,k and the corresponding k row from B. Figure [1](#page-2-0) shows an example of these dataflows, which we discuss further in §[II-C](#page-2-1) to elucidate their limitations.

