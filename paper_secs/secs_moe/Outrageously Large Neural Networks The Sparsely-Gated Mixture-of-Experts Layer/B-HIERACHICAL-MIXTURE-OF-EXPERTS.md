# B HIERACHICAL MIXTURE OF EXPERTS

If the number of experts is very large, we can reduce the branching factor by using a two-level hierarchical MoE. In a hierarchical MoE, a primary gating network chooses a sparse weighted combination of "experts", each of which is itself a secondary mixture-of-experts with its own gating network.<sup>3</sup> If the hierarchical MoE consists of a groups of b experts each, we denote the primary gating network by Gprimary, the secondary gating networks by (G1, G2..Ga), and the expert networks by (E0,0, E0,1..Ea,b). The output of the MoE is given by:

$$y_H = \sum_{i=1}^{a} \sum_{j=1}^{b} G_{primary}(x)_i \cdot G_i(x)_j \cdot E_{i,j}(x)$$
 (12)

Our metrics of expert utilization change to the following:

$$Importance_{H}(X)_{i,j} = \sum_{x \in X} G_{primary}(x)_{i} \cdot G_{i}(x)_{j}$$
(13)

$$Load_{H}(X)_{i,j} = \frac{Load_{primary}(X)_{i} \cdot Load_{i}(X^{(i)})_{j}}{|X^{(i)}|}$$
(14)

Loadprimary and Load<sup>i</sup> deonte the Load functions for the primary gating network and i th secondary gating network respectively. X(i) denotes the subset of X for which Gprimary(x)<sup>i</sup> > 0.

It would seem simpler to let LoadH(X)i,j = Loadi(Xi)<sup>j</sup> , but this would not have a gradient with respect to the primary gating network, so we use the formulation above.

## C 1 BILLION WORD LANGUAGE MODELING BENCHMARK - EXPERIMENTAL DETAILS

