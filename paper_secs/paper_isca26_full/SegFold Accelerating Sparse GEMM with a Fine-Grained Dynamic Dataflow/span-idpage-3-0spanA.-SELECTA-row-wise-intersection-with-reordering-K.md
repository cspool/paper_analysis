# <span id="page-3-0"></span>*A.* SELECTA*: row-wise intersection with reordering* K

Because matrix B is processed at row-wise granularity, dimension N is placed in the innermost loop, as shown in Fig. [2\(](#page-4-1)d). This inherently maximizes the reuse of A. The nonzero positions of A determine which rows of B are accessed in each iteration. SELECTA therefore picks a set of A

