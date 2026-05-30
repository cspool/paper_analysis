# <span id="page-36-2"></span>D.4 Routing cost

In Fig. 5, we visualize the router computation cost (Eq. 6) relative to that of the experts' layer propagation (Eq. 3), corresponding to the best-performing models in Table 2. The x-axis denotes the different S'MoRE structures (in Table 2, we do not include the results corresponding to the "4-2" and "8-8" S'MoRE architectures, due to space limit). The costs are measured by the total number of arithmetic operations performed by the routers versus by the experts. In general, when the residual rank  $r_{\ell}$  is lower, the cost of routing becomes *relatively* higher (since the router operation is independent of the ranks). However, in all cases, the routing cost is insignificant compared to the cost of expert propagation (at most 26%). This is consistent with our theoretical complexity analysis in §3.4.

<span id="page-36-3"></span><span id="page-36-0"></span><sup>&</sup>lt;sup>11</sup>Same as above, the fanouts are only set for sparse gates. For dense gates, the fanout of layer  $\ell$  equals the total number of experts in layer  $\ell$ 

<span id="page-37-0"></span>![](_page_37_Figure_0.jpeg)

Figure 5: Cost of router (Eq. [6\)](#page-4-1) relative to expert propagation (Eq. [3\)](#page-3-0), measured by their number of arithmetic operations. Here we consider S'MoRE with noisy top-k gate on LLaMA 3.2-1B

