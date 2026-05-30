# <span id="page-24-0"></span>C.1 Derivation of parameter & computation costs

Here we provide additional algorithmic details for the parameter and computation efficiency calculation in [§3.4.](#page-5-4)

Parameter efficiency. From Eq. [8,](#page-5-5) we have

$$P_{\ell+1} = s_{\ell} \cdot d \cdot r_{\ell} + d_{\ell+1}^2 \tag{12}$$

Then Eq. [9](#page-5-6) is derived as

$$P_{\text{proj}} + \sum_{\ell=1}^{L} P_{\ell} = d \cdot d_{L} + \sum_{\ell=0}^{L-1} P_{\ell+1}$$

$$= d \cdot d_{L} + \sum_{\ell=0}^{L-1} s_{\ell} \cdot d \cdot r_{\ell} + \sum_{\ell=0}^{L-1} d_{\ell+1}^{2}$$

$$= d \cdot d_{L} + d \cdot \left(\sum_{\ell=0}^{L-1} s_{\ell} \cdot r_{\ell}\right) + \sum_{\ell=0}^{L-1} d_{\ell+1}^{2}$$

$$= d \cdot d_{L} + d \cdot d_{L} + \sum_{\ell=0}^{L-1} d_{\ell+1}^{2}$$

$$= 2 \cdot d \cdot d_{L} + \Delta \qquad (13)$$
where
$$\Delta = \sum_{\ell=0}^{L-1} d_{\ell+1}^{2} \ll 2 \cdot d \cdot d_{L}$$

**Computation cost.** In Eq. 3, each  $B^n_\ell \cdot A^n_\ell \cdot x$  requires  $C' = d \cdot r_\ell + r_\ell \cdot d_{\ell+1}$  operations. Each  $W_\ell \cdot x^n_\ell$  requires  $C'' = d_\ell \cdot d_{\ell+1}$ . Consider all activated experts i in layer  $\ell+1$ , there can be at most  $N' = \min\{s_\ell, F_\ell\}$  distinct  $B^n_\ell \cdot A^n_\ell$  terms, incurring C' cost N' times. There are  $F_\ell$  different  $x^n_\ell$  inputs, each incurring C'' cost. Ignoring the element-wise addition " $\sum_{n \in \mathcal{N}^n_\ell}$ " and multiplication  $\alpha^{i,n}_\ell$ , the total cost of layer  $\ell+1$  equals (where  $d_\ell$  and  $F_\ell$  follow Eq. 4 and Eq. 5):

$$C_{\ell+1} \leq \min\{s_{\ell}, F_{\ell}\} \cdot r_{\ell} \cdot (d + d_{\ell+1}) + F_{\ell} \cdot d_{\ell} \cdot d_{\ell+1}$$

$$= \min\{s_{\ell}, F_{\ell}\} \cdot r_{\ell} \cdot d + \min\{s_{\ell}, F_{\ell}\} \cdot r_{\ell} \cdot d_{\ell+1} + F_{\ell} \cdot d_{\ell} \cdot d_{\ell+1}$$

$$\leq s_{\ell} \cdot r_{\ell} \cdot d + F_{\ell} \cdot r_{\ell} \cdot d_{\ell+1} + F_{\ell} \cdot d_{\ell} \cdot d_{\ell+1}$$

$$= s_{\ell} \cdot r_{\ell} \cdot d + F_{\ell} \cdot d_{\ell+1} \cdot (d_{\ell} + r_{\ell})$$
(14)

The cost of the final projection equals  $C_{\text{proj}} = d \cdot d_L$ . So the overall computation cost is:

$$C_{\text{proj}} + \sum_{\ell=1}^{L} C_{\ell} = d \cdot d_{L} + d \cdot \left(\sum_{\ell=0}^{L-1} s_{\ell} \cdot r_{\ell}\right) + \Delta' \stackrel{\text{(b)}}{=} 2 \cdot d \cdot d_{L} + \Delta' \stackrel{\text{(c)}}{\approx} 2 \cdot d \cdot d_{L} \tag{15}$$

where  $\Delta' \leq \sum_{\ell=0}^{L-1} F_{\ell} \cdot d_{\ell+1} \cdot (d_{\ell} + r_{\ell})$ . Steps "(b)" and "(c)" follow similar reasoning to Eq. 9.

Under practical values of  $d_\ell+r_\ell \leq d_{\ell+1} \ll d$ , the overhead term  $\Delta'$  is small or negligible compared to the main cost  $2\cdot d\cdot d_L$ . In Table 6, we empirically calculate the value of  $2\cdot d\cdot d_L$ , the overhead  $\Delta'$ , and their ratio. We take a representative configuration with  $f_\ell=2$ ,  $s_\ell=4$ , and  $r_\ell=8$  or 16 for all layers  $\ell$  (consistent with the experiments in §4). For 2 layers, the overhead  $\Delta'$  is just **1.2%** (or **2.3%**) of the Table 6: Overhead  $\Delta'$  compared with

cost of vanilla LoRA with rank  $d_L = 64$  (or  $d_L = 128$ ).

<span id="page-25-2"></span>Table 6: Overhead  $\Delta'$  compared with the main computation  $\cos 2 \cdot d \cdot d_L$ 

Similar to the analysis in the "Parameter efficiency" paragraph, the gating MLPs are lightweight compared to the main cost  $2 \cdot d \cdot d_L$ , due to the small dimensionalities.  $\begin{array}{c|ccccccccccccccccccccccccccccccccccc$ 

| $T\ell$ | L | $a_L$ | $z \cdot a \cdot a_L$ | Δ      | Overnead ratio |
|---------|---|-------|-----------------------|--------|----------------|
|         | 2 | 64    | 0.5M                  | 0.006M | 1.2%           |
| 8       | 3 | 96    | 0.8M                  | 0.026M | 3.3%           |
|         | 4 | 128   | 1.0M                  | 0.079M | 7.5%           |
|         | 2 | 128   | 1.0M                  | 0.025M | 2.3%           |
| 16      | 3 | 192   | 1.6M                  | 0.104M | 6.6%           |
|         | 4 | 256   | 2.1M                  | 0.315M | 15.0%          |

Thus, the total computation cost of S'MoRE is approximately  $2 \cdot d \cdot d_L$ , which is the same as the cost of a vanilla LoRA

with rank  $d_L$ . This proves the both parameter and the computation efficiency of S'MoRE.

#### <span id="page-25-0"></span>C.2 Proof of model capacity

#### <span id="page-25-1"></span>C.2.1 Proof for two special S'MoRE configurations

**Proposition C.1.** (Proposition 3.1) S'MoRE can express MoLRE, when L=1 and  $\sigma\left(\cdot\right)$  is the identity mapping.

*Proof.* When L=1, there is only a single layer propagation. When we set  $\sigma$  as the identity mapping, Eq. 3 becomes

$$\boldsymbol{x}_1 = \sum_{n \in \mathcal{N}_0} \alpha_\ell^n \cdot \boldsymbol{B}_0^n \cdot \boldsymbol{A}_0^n \cdot \boldsymbol{x} \tag{16}$$

where we omit the superscript i since there is just one parent node (the root of all all experts in a flat layer).

Combined with the final projection (see end of §3.2), the final output is computed by

<span id="page-26-0"></span>
$$\mathbf{x}' = \mathbf{W}_{\text{proj}} \cdot \sum_{n \in \mathcal{N}_0} \alpha_{\ell}^n \cdot \mathbf{B}_0^n \cdot \mathbf{A}_0^n \cdot \mathbf{x}$$

$$= \sum_{n \in \mathcal{N}_0} \alpha_{\ell}^n \cdot (\mathbf{W}_{\text{proj}} \cdot \mathbf{B}_0^n) \cdot \mathbf{A}_0^n \cdot \mathbf{x}$$
(17)

where  $\bm{A}_0^n \in \mathbb{R}^{r_0 \times d}$ ,  $\bm{B}_0^n \in \mathbb{R}^{(s_0 \cdot r_0) \times r_0}$  and  $\bm{W}_{\text{proj}} \in \mathbb{R}^{d \times (s_0 \cdot r_0)}$ .

For MoLRE with  $s_0$  rank- $r_0$  experts, according to the definition in §3.1, we can express its layer operation as

<span id="page-26-1"></span>
$$\bar{x}' = \sum_{n \in \bar{N}} \text{ROUTE}(x)^n \cdot \bar{B}^n \cdot \bar{A}^n \cdot x$$
 (18)

where we use overhead "bar" to distinguish variables of MoLRE from those of 1-layer S'MoRE. Here  $\bar{A}^n \in \mathbb{R}^{r_0 \times d}$  and  $\bar{B}^n \in \mathbb{R}^{d \times r_0}$ .

To make Eq. 17 and Eq. 18 equivalent, we can have

- S'MoRE's router implementing as ROUTE  $(x)^n$
- $A_0^n = \bar{A}^n$  (by definition, both matrices have the same shape)

• 
$$\bm{B}_0^n = \begin{bmatrix} \bm{0}_{r_0} \\ \vdots \\ \bm{0}_{r_0} \\ \bm{0}_{r_0} \\ \vdots \\ \bm{0}_{r_0} \end{bmatrix}$$
 , which is a binary matrix by vertically stacking  $s_0$  square blocks of  $r_0 \times r_0$ 

sub-matrices. The n-th block is a  $r_0 \times r_0$  identity matrix,  $I_{r_0}$ , while all the other blocks are 0 (denoted as  $\mathbf{0}_{r_0}$ ).

• 
$$\boldsymbol{W}_{\text{proj}} = \big[\bar{\boldsymbol{B}}^1, \dots, \bar{\boldsymbol{B}}^{s_0}\big].$$

Then  $W_{\text{proj}} \cdot B_0^n = \bar{B}_0^n$ . And Eq. 17 becomes identical to Eq. 18, completing the proof.

**Proposition C.2.** (Proposition 3.2) S'MoRE can express MoMOR, when setting  $\sigma(\cdot)$  as the identity mapping.

*Proof.* Without  $\sigma$ , we can collapse a multi-layer S'MoRE into a single-layer equivalent. For L=2, following Eq. 3, we have

$$x_{2} = \sum_{n \in \mathcal{N}_{1}} \alpha_{1}^{n} \cdot (\boldsymbol{B}_{1}^{n} \cdot \boldsymbol{A}_{1}^{n} \cdot \boldsymbol{x} + \boldsymbol{W}_{1} \cdot \boldsymbol{x}_{1}^{n})$$

$$= \sum_{n \in \mathcal{N}_{1}} \alpha_{1}^{n} \cdot \boldsymbol{B}_{1}^{n} \cdot \boldsymbol{A}_{1}^{n} \cdot \boldsymbol{x} + \boldsymbol{W}_{1} \sum_{n \in \mathcal{N}_{1}} \alpha_{1}^{n} \cdot \left( \sum_{m \in \mathcal{N}_{0}^{n}} \alpha_{0}^{n,m} \cdot \boldsymbol{B}_{0}^{m} \cdot \boldsymbol{A}_{0}^{m} \cdot \boldsymbol{x} \right)$$

$$= \sum_{n \in \mathcal{N}_{1}} \hat{\alpha}_{1}^{n} \cdot \boldsymbol{B}_{1}^{n} \cdot \boldsymbol{A}_{1}^{n} \cdot \boldsymbol{x} + \sum_{m \in \mathcal{N}_{0}} \hat{\alpha}_{0}^{m} \cdot (\boldsymbol{W}_{1} \cdot \boldsymbol{B}_{0}^{m} \cdot \boldsymbol{A}_{0}^{m}) \cdot \boldsymbol{x}$$

$$(19)$$

where we define  $\hat{\alpha}_1^n = \alpha_1^n$  and  $\hat{\alpha}_0^m = \sum_{n \in \mathcal{N}_1 \text{ and } m \in \mathcal{N}_0^n} (\alpha_1^n \cdot \alpha_0^{n,m})$ .

In general, for L layers and with the final projection step  ${\bm W}_{\rm proj}$ , we can summarize the propagation equation as

<span id="page-27-2"></span><span id="page-27-1"></span>
$$\boldsymbol{x}' = \sum_{\ell=0}^{L-1} \sum_{i=1}^{s_{\ell}} \hat{\alpha}_{\ell}^{i} \cdot \left( \prod_{k=\ell+1}^{L} \boldsymbol{W}_{k} \right) \cdot \boldsymbol{B}_{\ell}^{i} \cdot \boldsymbol{A}_{\ell}^{i} \cdot \boldsymbol{x}$$
 (20)

where we define  $W_L = W_{\text{proj}} \in \mathbb{R}^{d \times d_L}$  and  $\hat{\alpha}^i_\ell$  is a scalar coefficient by aggregating the router weights along all paths that end at the layer- $(\ell+1)$  expert  $i^4$ . In other words,  $\hat{\alpha}^i_\ell$  generalizes the definition of  $\hat{\alpha}^m_0$  above. The "path" here refers to the "ancestral path" (Definition C.5) ending at i. See more discussion on the routing tree in Appendix C.2.3. Also, if an expert is never selected, we let its  $\hat{\alpha}^i_\ell = 0$ . This way, we can replace the summation over  $\mathcal{N}_\ell$  in Eq. 19 with the summation over  $1 \leq i \leq s_\ell$  in Eq. 20.

For MoMOR model, following Eq. 2, we write its layer propagation as

$$x' = \sum_{\ell=1}^{L-1} \sum_{i=1}^{s_{\ell}} \text{ROUTE}_{\ell} \left( \boldsymbol{x} \right)^{i} \cdot \bar{\boldsymbol{B}}_{\ell}^{i} \cdot \bar{\boldsymbol{A}}_{\ell}^{i} \cdot \boldsymbol{x} \tag{21}$$

We can make Eq. 20 and Eq. 21 equivalent by a similar construction as the proof for Proposition 3.1. First, define a special binary projection matrix  $P_{a \times b} \in \{0,1\}^{a \times b}$  (where a > b) as

<span id="page-27-3"></span>
$$\boldsymbol{P}_{a\times b} = \begin{bmatrix} \mathbf{0}_{(a-b)\times b} \\ \boldsymbol{I}_{b\times b} \end{bmatrix} \tag{22}$$

meaning that the first a-b rows of  $P_{a\times b}$  are all 0, and the bottom b rows are an identity matrix. It is easy to verify that for a>b>c:

$$P_{a \times b} \cdot P_{b \times c} = P_{a \times c} \tag{23}$$

Then we can set all parameters of S'MoRE as follows:

- Let the S'MoRE router implement ROUTE $_{\ell}\left(\boldsymbol{x}\right)^{i}$ .
- Let  $A^i_{\ell} = \bar{A}^i_{\ell}$ .
- Let  $B_{\ell}^i$  be a  $d_{\ell+1} \times r_{\ell}$  binary matrix, where its row  $(i-1) \cdot r_{\ell} + 1$  to row  $i \cdot r_{\ell}$  is a  $r_{\ell} \times r_{\ell}$  identity matrix, and its all other rows are all 0. Here we let both i and the row index start from 1.
- Let  $\pmb{W}_L = \pmb{W}_{\text{proj}} = \left[\bar{\pmb{B}}_0^1, \dots, \bar{\pmb{B}}_0^{s_0}, \dots, \bar{\pmb{B}}_{L-1}^1, \dots, \bar{\pmb{B}}_{L-1}^{s_{L-1}}\right]$  as the horizontal concatenation of all MoMOR's up-projection matrices  $\bar{\pmb{B}}_\ell^i$ .

<span id="page-27-0"></span><sup>&</sup>lt;sup>4</sup>The same expert i of layer  $\ell+1$  may be selected multiple times, corresponding to different parents or ancestors. Thus, there can be multiple paths ending at the layer- $(\ell+1)$  expert i.

• Each  $W_k$  has shape  $d_{k+1} \times d_k$  where  $d_{k+1} = d_k + s_k \cdot r_k$ . We set it as  $W_k = P_{d_{k+1} \times d_k}$ . Then it follows that

$$\prod_{k=\ell+1}^{L-1} \boldsymbol{W}_{k} = \boldsymbol{P}_{d_{L} \times d_{L-1}} \cdot \boldsymbol{P}_{d_{L-1} \times d_{L-2}} \dots \boldsymbol{P}_{d_{\ell+2} \times d_{\ell+1}} = \boldsymbol{P}_{d_{L} \times d_{\ell+1}}$$

$$\Rightarrow \left(\prod_{k=\ell+1}^{L} \boldsymbol{W}_{k}\right) \cdot \boldsymbol{B}_{\ell}^{i} = \boldsymbol{W}_{\text{proj}} \cdot \left(\prod_{k=\ell+1}^{L-1} \boldsymbol{W}_{k}\right) \cdot \boldsymbol{B}_{\ell}^{i}$$

$$= \boldsymbol{W}_{\text{proj}} \cdot \boldsymbol{P}_{d_{L} \times d_{\ell+1}} \cdot \boldsymbol{B}_{\ell}^{i}$$

$$= \bar{\boldsymbol{B}}_{\ell}^{i} \tag{25}$$

Under the above construction, it is clear that Eq. 21 and Eq. 20 are exactly the same. Thus, S'MoRE can express MoMOR, concluding the proof.

**Remark.** Note that the equivalence between S'MoRE and MoMOR can only be established when we set the layer dimension  $d_{\ell}$  according to Eq. 4. This can be seen from the "minimum dimensionality" discussion in §3.2.

#### <span id="page-28-0"></span>C.2.2 Proof of Theorem 3.3

**Theorem C.3.** (Theorem 3.3) The structural flexibility of MoMOR is upper-bounded by  $\Gamma_{MoMOR} = \max_{\boldsymbol{x},\Theta} dist(\boldsymbol{x};\Theta) \leq \binom{s_{L-1}}{f_{L-1}} \cdot \prod_{\ell=0}^{L-2} \left(\sum_{i=f_{\ell}}^{\min\{F_{\ell},s_{\ell}\}} \binom{s_{\ell}}{i}\right)$ .

*Proof.* The upper bound of  $\Gamma_{\text{MoMOR}}$  basically quantifies the total number of combinations to select experts from each residual pool.

**Assumption.** We first simplify Eq. 2 that the router-generated coefficient  $\mathtt{ROUTE}_{\ell}\left(\boldsymbol{x}\right)^{i}$  is just a binary mask. i.e., for a selected expert i, we have  $\mathtt{ROUTE}_{\ell}\left(\boldsymbol{x}\right)^{i}=1$ . Otherwise,  $\mathtt{ROUTE}_{\ell}\left(\boldsymbol{x}\right)^{i}=0$ . Such an assumption is just to ease the calculation of  $\Gamma_{\mathtt{MOMOR}}$  and  $\Gamma_{\mathtt{S'MoRE}}$ . It does not affect our fundamental conclusion that  $\mathtt{S'MoRE}$  yields exponentially higher structural flexibility than MoMOR.

Based on Eq. 2, the MoMOR output is generated by a flat summation of different-order residues. Given any input x, the number of distinct outputs cannot exceed the number of distinct ways to select residues from the pools  $\mathcal{R}_0, \cdots, \mathcal{R}_{L-1}$ . Here we show some examples to illustrate the meaning of "distinct expert selection".

- "Selecting experts 1,2,3 from  $\mathcal{R}_0$ " and "selecting experts 1,3,4 from  $\mathcal{R}_0$ " correspond to 2 distinct ways.
- "Selecting experts 1,2,3 from  $\mathcal{R}_0$ " and "selecting experts 3,2,1 from  $\mathcal{R}_0$ " correspond to the same way, because there is no ordering among the selected experts<sup>5</sup>.
- "Selecting experts 1,1,3 from  $\mathcal{R}_0$ " and "selecting experts 1,3,3 from  $\mathcal{R}_0$ " orrespond to the same way due to our assumption of making  $\mathtt{ROUTE}_\ell\left(\boldsymbol{x}\right)^i$  a binary mask. Basically we only care about whether an expert is selected or not. It does not matter how many times an expert is selected.

**Remark.** Distinct expert selections do not guarantee distinct outputs. For example, consider "selecting 1,2,3 from  $\mathcal{R}_0$ " and "selecting 1,3,4" from  $\mathcal{R}_0$ . Following the notation of Eq. 2, if the experts' weights satisfy  $\Delta W_0^1 + \Delta W_0^2 + \Delta W_0^3 = \Delta W_0^1 + \Delta W_0^3 + \Delta W_0^4$ , then the two case generates the same output for all input  $\boldsymbol{x}$ :

<span id="page-28-1"></span><sup>&</sup>lt;sup>5</sup>The order among selected experts does not matter because the sum aggregation of Eq. 2 is *permutation invariant* 

<span id="page-28-2"></span><sup>&</sup>lt;sup>6</sup>If we follow S'MoRE's recursive expert selection process described in §3.3, the same expert of higher-order may be selected multiple times, from different lower-order parents.

$$\sum_{i \in \{1,2,3\}} \Delta \mathbf{W}_0^i \cdot \mathbf{x} = \sum_{j \in \{1,3,4\}} \Delta \mathbf{W}_0^j \cdot \mathbf{x}$$
 (26)

Hence, counting the number of distinct ways of expert selection just gives an upper bound of  $\Gamma_{\text{MoMOR}}$ , because  $\Gamma_{\text{MoMOR}}$  is defined on the number of distinct outputs.

Counting the combinations. For the  $\mathcal{R}_{L-1}$  pool with size  $s_{L-1}$ , there are  $\binom{s_{L-1}}{f_{L-1}}$  ways to pick  $f_{L-1}$  residues. For  $\mathcal{R}_{L-2}$  with  $\ell \leq L-2$ , there are  $F_{\ell+1}$  parents, each picking  $f_{\ell}$  children in the pool. Different parents can pick the same children. The number of distinct children selected by all parents ranges from  $f_{\ell}$  to  $\min\{F_{\ell}, s_{\ell}\}$ . This makes the total count  $\sum_{i=f_{\ell}}^{\min\{F_{\ell}, s_{\ell}\}} \binom{s_{\ell}}{i}$ . From basic Combinatorics, each layer  $\ell$  contributes to a multiplicative factor in the total count. Thus, the final upper bound is:

$$\Gamma_{\text{MoMOR}} \le \binom{s_{L-1}}{f_{L-1}} \cdot \prod_{\ell=0}^{L-2} \left( \sum_{i=f_{\ell}}^{\min\{F_{\ell}, s_{\ell}\}} \binom{s_{\ell}}{i} \right)$$
(27)

#### <span id="page-29-0"></span>C.2.3 Proof of Theorem 3.4

**Theorem C.4.** (Theorem 3.4) Setting  $\sigma(\cdot)$  as an MLP, there exists some  $\Theta'$  such that the structural flexibility of S'MoRE is  $\Gamma_{S'MoRE} = \min_{\boldsymbol{x}} \operatorname{dist}(\boldsymbol{x}; \Theta') = \prod_{\ell=0}^{L-1} \binom{s_{\ell}}{f_{\ell}}^{F_{\ell+1}}$ , where  $F_L := 1$ .

*Proof.* We prove in two stages:

- 1. We show that following the routing process of S'MoRE, there can be  $\Gamma_{\text{S'MoRE}}$  non-isomorphic depth-L trees, where each tree node is an expert residue.
- 2. We construct a S'MoRE instance where its L-layer propagation (Eq. 3) generates distinct outputs for all non-isomorphic trees above, regardless of input token embedding x.

Both can be proven by induction.

**Assumption.** Similar to Theorem 3.3, we make simplification to the layer propagation Eq. 3, that the coefficient  $\alpha_{\ell}^{i,n}$  is just a binary mask. i.e., for a selected children n, we have  $\alpha_{\ell}^{i,n}=1$ . Otherwise,  $\alpha_{\ell}^{i,n}=0$ .

Stage 1: Number of non-isomorphic trees. Recall the expert selection / tree construction process in §3.3: each active parent expert of layer  $\ell+1$  (in  $\mathcal{R}_{\ell}$ ) selects  $f_{\ell-1}$  children out of all the  $s_{\ell-1}$  experts of layer  $\ell$ . So by traversing all the L layers, the router builds a depth-L balanced tree (which has  $\prod_{\ell=0}^{L-1} f_{\ell}$  leaf nodes in total). Note that

- 1. For each parent, its  $f_{\ell}$  selected children are distinct (i.e., the same parent cannot select the same child twice).
- 2. However, the same expert may appear in the same tree-level multiple times, corresponding to different parents or ancestors.
- 3. There is **no ordering** among the selected children, since Eq. 3 performs "sum" aggregation which is *permutation invariant*. e.g., it is equivalent to say that a parent of layer  $\ell$  selects "children 1,3,4" and "children 4,3,1".

Due to Point 2 above, we cannot uniquely identify a tree node by the its corresponding expert's layer index and expert index. Yet, Points 1 and 3 ensure that any tree node n is *uniquely identifiable* by n's ancestral path  $\mathcal{P}_n$ .

<span id="page-30-0"></span>**Definition C.5.** (Ancestral path  $\mathcal{P}_n$ ) Let  $(\ell,i)$  denote expert i of layer  $\ell$ . Suppose a tree-node n at tree-level t corresponds to expert (L-t+1,i). Then n's ancestral path,  $\mathcal{P}_n = ((L-t+1,i),(L-t+2,i'),\ldots,(L,i'^{\dots'}))$ , defines the unique path to traverse from n up to the tree root (where we treat the root as a *virtual* node that is the parent of all  $(L,i'^{\dots'})$ , and we omit the root in the path).

**Definition C.6.** (Leaves' ancestral paths  $\mathcal{T}$ ) Given a tree, define  $\mathcal{T} = \{\mathcal{P}_n \mid n \text{ is a leaf node}\}$  as the set of ancestral paths of all leaf nodes, where there are  $\prod_{\ell=0}^{L-1} f_\ell$  leaves, all at tree-level L.

Two trees are *isomorphic* if their structures are equivalent. That means, we can permute or swap the children (together with their corresponding descendant sub-tree) of some parent nodes to make the two trees look exactly the same.  $\mathcal{T}$  enables us to define isomorphism. In our construction, the children are not ordered (Point 3 above), and so permuting or swapping children does not change  $\mathcal{T}$ . Thus, isomorphic trees have the same  $\mathcal{T}$ . On the other hand, we can show trees of the same  $\mathcal{T}$  can be made equivalent by permutation or swapping, and thus are isomorphic. In sum, we can define tree isomorphism by  $\mathcal{T}$  as follows:

<span id="page-30-1"></span>**Definition C.7.** (Isomorphism) Given two trees, let their leaves' ancestral paths be  $\mathcal{T}$  and  $\mathcal{T}'$ . The two trees are isomorphic if and only if  $\mathcal{T} = \mathcal{T}'$ .

We next derive the number of depth-L non-isomorphic trees by induction.

Imagine that we apply the top-down expert selection from layer  $\ell$  down to layer 1 (with  $\ell \geq 1$ ): at layer  $\ell$ , we select  $f_{\ell-1}$  experts from  $s_{\ell-1}$  experts; at layer  $\ell-1$ , for each of the selected parent of layer  $\ell$ , we select  $f_{\ell-2}$  from  $s_{\ell-2}$  experts, and so on.

<u>Induction hypothesis</u>: the number of non-isomorphic trees yielded by such an expert-selection process equals:

$$\Gamma_{\text{S,MORE}}^{\ell} = \prod_{k=0}^{\ell-1} {s_k \choose f_k}^{F_{k+1}/F_{\ell}}$$
(28)

for some  $1 \le \ell < L$ .

<u>Base case  $\ell = 1$ </u>: we are just sampling a single level. So the number of non-isomorphic trees equals the number of total ways to select  $f_0$  experts from  $s_0$ , which is  $\binom{s_0}{f_0}$ .

And

$$\Gamma_{S,MORE}^{1} = \prod_{k=0}^{1-1} {s_k \choose f_k}^{F_{k+1}/F_1}$$

$$= {s_0 \choose f_0}$$
(29)

So the base case holds.

<u>Induction from  $\ell$  to  $\ell+1$ </u>: To construct a tree by selecting experts from layer  $\ell+1$  to 1, we follow two steps:

- 1. We select  $f_{\ell}$  out of  $s_{\ell}$  experts. Denote them as  $\mathcal{E}_{\ell} = \{(\ell+1, i_1), \dots, (\ell+1, i_{f_{\ell}})\}$ , where  $i_a \neq i_b$  for all  $a \neq b$ .
- 2. We start from each  $(\ell+1,i_m)$  and recursively activate experts from layer  $\ell$  down to 1 (where  $1 \leq m \leq f_\ell$ ), following the procedure described above. Denote each such tree by its leaves' ancestral paths,  $\mathcal{T}_{\ell,i_m}$ .

Let  $\mathbb{T}_{\ell}$  be the set of all possible  $\mathcal{T}_{\ell,i_m}$  — note that  $\mathbb{T}_{\ell}$  does not have subscript  $i_m$ , since an ancestral path ends at a virtual root node independent of  $i_m$  (see Definition C.5), and thus  $\mathbb{T}_{\ell}$  is the same for all  $i_m$ . Based on the induction hypothesis,  $|\mathbb{T}_{\ell}| = \Gamma_{S,MORE}^{\ell}$ .

For such a tree constructed by the two steps above, let  $\mathcal{T}_{\ell+1}$  be its leaves' ancestral paths:

$$\mathcal{T}_{\ell+1} = \bigcup_{k=1}^{f_{\ell}} \{ p \oplus (\ell+1, i_k) \mid p \in \mathcal{T}_{\ell, i_k} \}$$
(30)

where " $\oplus$ " means appending  $(\ell+1, i_k)$  to the end of the path p. By Definition C.7, the total number of non-isomorphic trees equals the number of distinct  $\mathcal{T}_{\ell+1}$ , which can be calculated with the following reasoning:

- There are  $\binom{s_{\ell}}{f_{\ell}}$  distinct ways to choose  $\mathcal{E}_{\ell}$  of Step 1.
- For each choice of  $\mathcal{E}_{\ell}$ , there are  $|\mathbb{T}_{\ell}|$  choices of  $\mathcal{T}_{\ell,i_k}$  for each  $i_k$  of  $\mathcal{E}_{\ell}$ , leading to  $|\mathbb{T}_{\ell}|^{f_{\ell}}$  distinct combinations.

So the number of distinct  $\mathcal{T}_{\ell+1}$  equals:

$$|\mathbb{T}_{\ell+1}| = \begin{pmatrix} s_{\ell} \\ f_{\ell} \end{pmatrix} \cdot |\mathbb{T}_{\ell}|^{f_{\ell}}$$

$$= \begin{pmatrix} s_{\ell} \\ f_{\ell} \end{pmatrix} \cdot \left(\Gamma_{S,MORE}^{\ell}\right)^{f_{\ell}}$$

$$= \begin{pmatrix} s_{\ell} \\ f_{\ell} \end{pmatrix} \cdot \left(\prod_{k=0}^{\ell-1} \begin{pmatrix} s_{k} \\ f_{k} \end{pmatrix}^{F_{k+1}/F_{\ell}} \right)^{f_{\ell}}$$

$$= \begin{pmatrix} s_{\ell} \\ f_{\ell} \end{pmatrix} \cdot \prod_{k=0}^{\ell-1} \begin{pmatrix} s_{k} \\ f_{k} \end{pmatrix}^{F_{k+1} \cdot \frac{f_{\ell}}{F_{\ell}}}$$

$$= \begin{pmatrix} s_{\ell} \\ f_{\ell} \end{pmatrix}^{F_{\ell+1}/F_{\ell+1}} \cdot \prod_{k=0}^{\ell-1} \begin{pmatrix} s_{k} \\ f_{k} \end{pmatrix}^{F_{k+1}/F_{\ell+1}}$$

$$= \prod_{k=0}^{\ell} \begin{pmatrix} s_{k} \\ f_{k} \end{pmatrix}^{F_{k+1}/F_{\ell+1}}$$

$$= \Gamma_{S,MORE}^{\ell+1}$$

$$(31)$$

This completes the induction step. Thus, the total number of non-isomorphic trees for all L layers equals  $\Gamma_{\text{S'MoRE}}^L = \prod_{\ell=0}^{L-1} \binom{s_\ell}{f_\ell}^{F_{\ell+1}/F_L} = \prod_{\ell=0}^{L-1} \binom{s_\ell}{f_\ell}^{F_{\ell+1}}$  where  $F_L := 1$ .

Stage 2: Distinguishing non-isomorphic trees. We next show that there exists some parameters  $\Theta'$  such that the layer propagation following Eq. 3 generates distinct output for non-isomorphic trees.

Notational correction to Eq. 3: In §3.2, we use  $\boldsymbol{x}_{\ell}^{i}$  to denote the output embedding where i is the expert index. This notation is not precise since the same expert can appear as multiple tree nodes, as discussed in the Stage 1 proof above. To make the correction, we instead let  $\boldsymbol{x}_{\ell}^{i}$  denote the embedding of node index  $i^{7}$  for tree-level  $L-\ell$ , meaning that there can be  $\boldsymbol{x}_{\ell}^{i}$  and  $\boldsymbol{x}_{\ell}^{i'}$  mapped to the same expert, where  $i \neq i'$ .

Including the bias term: Our proof requires a minor modification of Eq. 3 to add a bias term  $b_k^n \in \mathbb{R}^{d_{k+1}}$  associated with each expert n. So the updated layer propagation equation, adapted from Eq. 3 now becomes:

<span id="page-31-0"></span>
$$\boldsymbol{x}_{\ell+1}^{i} = \sum_{n \in \mathcal{N}_{\ell}^{i}} \sigma \left( \boldsymbol{B}_{\ell}^{n} \cdot \boldsymbol{A}_{\ell}^{n} \cdot \boldsymbol{x} + \boldsymbol{W}_{\ell} \cdot \boldsymbol{x}_{\ell}^{i \to n} + \boldsymbol{b}_{\ell}^{n} \right)$$
(32)

<span id="page-31-1"></span><sup>&</sup>lt;sup>7</sup>In our terminology above, this means that each  $(\ell, i)$  now corresponds to a *distinct* ancestral path.

where  $\ell$  is the *layer* index; i is the index of a *tree node*, while n is still the index of an *expert*.  $\mathcal{N}^i_\ell$  denotes the set of indices of the children experts selected by node i. Note, " $i \to n$ " means that a tree node i picks a previous-layer expert n as its child. So with a slight abuse of notation, we use superscript " $i \to n$ " to index such a child tree node.  $\alpha^{i,n}_\ell$  of Eq. 3 is omitted since we simplify the expert weight as binary mask, as stated above.

We are now ready for the proof.

First, note that since the operations by Eq. 3 are permutation invariant, S'MoRE will generate the same output for all isomorphic trees.

Next, we consider non-isomorphic trees. Again we prove by induction.

Similar to the Stage 1 setting, we consider an expert-selection process from layer  $\ell$  down to layer 1. After building such an  $\ell$ -level tree, the model propagates the input token x from layer 1 up to layer  $\ell$  to generate the output  $x_{\ell}$ . Note, since in the induction step, the propagation terminates at  $x_{\ell}$ , we do not need to superscript  $x_{\ell}$  with an additional node index i. In other words,  $x_{\ell}$  here is analogous to the *final* embedding  $x_{\ell}$  described in §3.2.

<u>Induction hypothesis</u>: For any  $\ell$ -level non-isomorphic trees  $\mathcal{T}_{\ell} \neq \mathcal{T}'_{\ell}$ , we can set the layer 1 to  $\ell$  parameters of S'MoRE such that  $x_{\ell} \neq x'_{\ell}$ .

Base case  $\ell = 1$ : For a single layer, the propagation simplifies to

$$x_1 = \sum_{n \in \mathcal{N}_0} \sigma \left( \mathbf{B}_0^n \cdot \mathbf{A}_0^n \cdot \mathbf{x} + \mathbf{b}_0^n \right)$$
 (33)

where non-isomorphic trees under  $\ell = 0$  degrades to distinct neighbor sets  $\mathcal{N}_0$ .

We want distinct outputs  $x_{\ell} \neq x'_{\ell}$  for *all* inputs x. So we have the following simple way to construct the parameters:

- $B_0^n = \mathbf{0}$  and  $A_0^n = \mathbf{0}$ , which leads to  $B_0^n \cdot A_0^n \cdot x + b_0^n = b_0^n$  for all input x;
- Let the first element of  $b_0^n$  store the expert index (an integer from 1 to  $s_0$ ), and the rest of the elements be 0.

We reuse the following lemma from Xu et al. [2019]:

<span id="page-32-0"></span>**Lemma C.8.** (see Lemma 5 of Xu et al. [2019]) Assume a countable input feature space  $\mathcal{X}$ . There exists a function  $f: \mathcal{X} \to \mathbb{R}^d$  so that  $h(X) = \sum_{x \in X} f(x)$  is unique for each set  $X \subset \mathcal{X}$  of bounded size.

In our case,  $\sigma$  of Eq. 32 corresponds to function f of Lemma C.8, and we treat  $\boldsymbol{b}_0^n$  as the function's input features. The "feature space" consisting of all possible  $\boldsymbol{b}_0^n$  is clearly countable (since each element of  $\boldsymbol{b}_0^n$  is either 0 or a bounded integer). The neighbor set  $\mathcal{N}_0$  corresponds to X of Lemma C.8, which can be an arbitrary combination of the children experts.

Thus, due to the universal approximation theorem [Hornik et al., 1989], we can instantiate  $\sigma$  as an MLP to implement such a function f, to guarantee that all non-isomorphic trees get a unique output  $x_1$ . This proves the base case.

Induction from  $\ell$  to  $\ell+1$ : Consider two trees constructed by recursive expert selection from layer  $\ell+1$  to 1. We use "prime" to denote quantities of the second tree. For example, their leaves' ancestral paths are  $\mathcal{T}_{\ell+1}$  and  $\mathcal{T}'_{\ell+1}$ . According to the analysis in the Stage 1 proof above, there are two possibilities to make the two trees non-isomorphic. i.e.,  $\mathcal{T}_{\ell+1} \neq \mathcal{T}'_{\ell+1}$ :

- 1. The sets of level-1 nodes are different:  $\mathcal{E}_{\ell} \neq \mathcal{E}'_{\ell}$ ;
- 2. Otherwise, let  $\mathcal{E}_{\ell} = \mathcal{E}'_{\ell} = \{(\ell+1,i_1),\ldots,(\ell+1,i_{f_{\ell}})\}$ . There exists  $i_m$  such that  $\mathcal{T}_{\ell,i_m} \neq \mathcal{T}'_{\ell,i_m}$  for some  $1 \leq m \leq f_{\ell}$ .

Our goal is to show that for each of the above cases, Eq. 32 can generate distinct outputs for  $\mathcal{T}_{\ell+1}$  and  $\mathcal{T}'_{\ell+1}$ .

Similar to the construction in the  $\ell=1$  case, we set  $\boldsymbol{B}_{\ell}^n=\mathbf{0}$  and  $\boldsymbol{A}_{\ell}^n=\mathbf{0}$ . And  $\boldsymbol{b}_{\ell}^n$  is a one-hot vector with the first element being the expert index (ranging from 1 to  $s_{\ell}$ ). Recall that  $\boldsymbol{W}_{\ell}\in\mathbb{R}^{d_{\ell+1}\times d_{\ell}}$  where  $d_{\ell+1}=s_{\ell}\cdot r_{\ell}+d_{\ell}$  (see Eq. 4). We set

<span id="page-33-1"></span>
$$W_{\ell} = \begin{bmatrix} \mathbf{0}_{(s_{\ell} \cdot r_{\ell}) \times d_{\ell}} \\ \mathbf{I}_{d_{\ell} \times d_{\ell}} \end{bmatrix}$$
(34)

where  $\mathbf{0}_{(s_{\ell} \cdot r_{\ell}) \times d_{\ell}}$  is a  $(s_{\ell} \cdot r_{\ell}) \times d_{\ell}$  all-0 matrix and  $\mathbf{I}_{d_{\ell} \times d_{\ell}}$  is a  $d_{\ell} \times d_{\ell}$  identity matrix.

So Eq. 32 now becomes

$$\boldsymbol{x}_{\ell+1}^{i} = \sum_{n \in \mathcal{N}_{\ell}^{i}} \sigma\left(\begin{bmatrix} \hat{\boldsymbol{b}}_{\ell}^{n} \\ \boldsymbol{x}_{\ell}^{i \to n} \end{bmatrix}\right)$$
(35)

where  $\hat{b}_{\ell}^{n}$  is a length- $(s_{\ell} \cdot r_{\ell})$  vector by discarding the trailing 0s of  $b_{\ell}^{n}$ .

Since the layer- $(\ell+1)$  output corresponds to the tree root, we can ignore the index i. Also note that  $i \to n$  is essentially  $i_m$  of  $\mathcal{E}_{\ell}$  above.

So we have

$$x_{\ell+1} = \sum_{n \in \mathcal{N}_{\ell}} \sigma\left(\begin{bmatrix} \hat{\boldsymbol{b}}_{\ell}^{n} \\ x_{\ell}^{i_{m}} \end{bmatrix}\right) \tag{36}$$

Finally, we go back to the two cases above that makes two trees non-isomorphic. Clearly, for either case, the two non-isomorphic trees will have different sets of  $\begin{bmatrix} \hat{\boldsymbol{b}}_{\ell}^n \\ \boldsymbol{x}_{\ell}^{i_m} \end{bmatrix}$ . This allows us to apply Lemma C.8, and conclude that the outputs  $\boldsymbol{x}_{\ell+1}$  will also be different for the two non-isomorphic trees.

Note that 1. we are still dealing with a countable feature space, since there are finite number (i.e.,  $\Gamma^{\ell}_{\text{S'MoRE}}$ ) of distinct  $\boldsymbol{x}^{i_m}_{\ell}$ ; 2. Different sets of  $\begin{bmatrix} \hat{\boldsymbol{b}}^n_{\ell} \\ \boldsymbol{x}^{i_m}_{\ell} \end{bmatrix}$  means different input "X" to function f in Lemma C.8.

This completes the induction step from  $\ell$  to  $\ell + 1$ .

In sum, our layer propagation function in Eq. 32 ensures that we can find some S'MoRE parameters  $\Theta'$  such that all depth-L non-isomorphic trees will lead to distinct outputs  $x_L$ .

Combining the proof for the two stages, we have shown that the "structural flexibility" of S'MoRE equals

$$\Gamma_{\text{S'MORE}} = \prod_{\ell=0}^{L-1} {s_{\ell} \choose f_{\ell}}^{F_{\ell+1}}.$$
(37)

Final remark. In the proof, we require  $\sigma$  to be an MLP. In practice, we can implement  $\sigma$  simply as non-linear activation (e.g., ReLU). It is easy to see that setting  $\sigma$  as "an MLP with a *single* hidden layer of dimension  $d_{\ell+1}$ " is equivalent to setting  $\sigma$  simply as an activation function — For the single-layer MLP, the transformation matrix before the activation can be merged with  $B_{\ell}^n \cdot A_{\ell}^n$  and  $W_{\ell}$  of the S'MoRE layer. The transformation matrix after the activation can be merged with the next layer  $W_{\ell+1}$ .

<span id="page-33-0"></span>Even if we implement  $\sigma$  as an MLP of at least 2 layers, it is still computation and parameter efficient. The input dimension to the MLP is  $d_{\ell}$ , which is small (compared with the dimension of the token embeddings). Thus, it is reasonable to set the hidden dimension of the MLP layers also small. This makes the overall MLP very compact. We can follow similar reasoning as §3.4.

#### C.2.4 Proof of Corollary 3.5

**Corollary C.9.** (Corollary 3.5) Let  $\Gamma_{S'MORE*}^{\ell}$  be the structural flexibility of  $\ell$ -layer S'MORE variant under Eq. 38. It satisfies the following recursion:  $\Gamma_{S'MORE*}^{\ell} = \binom{s_{\ell-1}}{f_{\ell-1}} \cdot \binom{\Gamma_{S'MORE*}^{\ell-1} + f_{\ell-1} - 1}{f_{\ell-1}}$ , where  $\Gamma_{S'MORE*}^{0} := 1$ .

*Proof.* This proof utilizes the construction in proving Theorem 3.4.

First, we decompose Eq. 38 as (like before, we ignore router weight  $\alpha$  for brevity):

$$\boldsymbol{x}_{\ell+1}^{i} = \sum_{n \in \mathcal{N}_{\ell}^{i}} \left( \boldsymbol{B}_{\ell}^{n} \cdot \boldsymbol{A}_{\ell}^{n} \cdot \boldsymbol{x} + \boldsymbol{W}_{\ell} \cdot \boldsymbol{\sigma} \left( \boldsymbol{x}_{\ell}^{n} \right) \right)$$
(38)

<span id="page-34-0"></span>
$$= \underbrace{\left(\sum_{n \in \mathcal{N}_{\ell}^{i}} \boldsymbol{B}_{\ell}^{n} \cdot \boldsymbol{A}_{\ell}^{n} \cdot \boldsymbol{x}\right)}_{(a)} + \underbrace{\boldsymbol{W}_{\ell} \cdot \left(\sum_{n \in \mathcal{N}_{\ell}^{i}} \sigma\left(\boldsymbol{x}_{\ell}^{n}\right)\right)}_{(b)}$$
(39)

We consider how many distinct values (a) and (b) can take.

**Term (b).** Suppose an  $\ell$ -layer S'MoRE \* can generate  $\Gamma^{\ell}_{S'MoRE}$  \* distinct outputs, meaning that  $\boldsymbol{x}^n_{\ell}$  can take  $\Gamma^{\ell}_{S'MoRE}$  \* different values – This is as if we have a pool of  $\Gamma^{\ell}_{S'MoRE}$  \* distinct elements.

The <u>first question</u> is, if we take  $f_{\ell} = \left| \mathcal{N}_{\ell}^{i} \right|$  elements from this pool (where the same element can be taken <u>multiple time</u>, since different children n can have the same descendant sub-tree), how many unique multisets can we obtain. This is a classic "combination with replacement" problem, and the solution is  $\binom{\Gamma_{\text{S'MoRE}}^{\ell} * + f_{\ell} - 1}{f_{\ell}}$ .

The second question is, can we encode each distinct multiset into distinct outputs via the form of  $\sum \sigma(\cdot)$ . Reusing Lemma C.8<sup>9</sup>, the answer is affirmative.

So term (b) can take  $\Gamma_{\mathtt{S},\mathtt{MoRE}\,*}^{\ell}$  distinct values.

**Term (a).** Since the router takes top- $f_\ell$  experts, there are in total  $\binom{s_\ell}{f_\ell}$  distinct  $\mathcal{N}_\ell^i$ . The key problem is if we perform the simple summation  $\sum_{n\in\mathcal{N}_\ell^i}$  without the mapping  $\sigma$ , can we ensure distinct output for each distinct  $\mathcal{N}_\ell^i$  (we cannot apply Lemma C.8 without  $\sigma$ )? i.e., for any  $\mathcal{N}_\ell^i\neq\mathcal{N}_\ell^{i'}$ , how can we ensure  $\sum_{n\in\mathcal{N}_\ell^i}B_\ell^n\cdot A_\ell^n\cdot x\neq\sum_{n'\in\mathcal{N}_\ell^{i'}}B_\ell^{n'}\cdot A_\ell^{n'}\cdot x$ . Setting a bias term encoding the expert index i, following Appendix C.2.3, does not work. A failure case is that  $\mathcal{N}_\ell^i$  contains experts 1, 4 and  $\mathcal{N}_\ell^{i'}$  contains experts 2, 3: 1+4=2+3 even through  $\{1,4\}\neq\{2,3\}$ . Fortunately, there are existing encoding schemes that satisfies our requirement. For example, we can encode the  $s_\ell$  experts into a "superincreasing sequence" where expert i is encoded into  $2^i$ . In this case, it is guaranteed that  $\sum_{n\in\mathcal{N}_\ell^i}2^n\neq\sum_{n'\in\mathcal{N}_\ell^{i'}}2^{n'}$  for any  $\mathcal{N}_\ell^i\neq\mathcal{N}_\ell^{i'}$ .

**Combining (a) and (b).** Finally, when we set  $W_{\ell}$  according to Eq. 34, we are guaranteed that any two different pairs of (a) and (b) will have different values of "(a) + (b)". This means the total number of distinct  $x_{\ell+1}^i$  we can obtain from Eq. 38 equals:

<span id="page-34-3"></span>
$$\Gamma_{\mathsf{S},\mathsf{MORE}}^{\ell+1} = \begin{pmatrix} s_{\ell} \\ f_{\ell} \end{pmatrix} \cdot \begin{pmatrix} \Gamma_{\mathsf{S},\mathsf{MORE}}^{\ell} + f_{\ell} - 1 \\ f_{\ell} \end{pmatrix} \tag{40}$$

Lastly, when  $\ell=1$ , it is obvious that  $\Gamma^1_{S'MoRE*}$  should be  $\binom{s_0}{f_0}$ . If we define  $\Gamma^0_{S'MoRE*} := 1$  and let  $\ell=0$ , Eq. 40 becomes  $\Gamma^1_{S'MoRE*} = \binom{s_0}{f_0} \cdot \binom{1+f_0-1}{f_0} = \binom{s_0}{f_0}$ , which satisfies the initial condition.

This completes the proof.

<span id="page-34-1"></span><sup>&</sup>lt;sup>8</sup>A multiset is a set where an element can appear multiple times.

<span id="page-34-2"></span><sup>&</sup>lt;sup>9</sup>The original Lemma in Xu et al. [2019] is indeed derived on multisets.

#### <span id="page-35-2"></span>C.2.5 Proof of Corollary 3.6

**Corollary C.10.** (Corollary 3.6) The structural flexibility of S'MoRE# equals  $\prod_{\ell=0}^{L-1} {s \choose \ell}^{F_{\ell+1}}$  where  $F_L := 1$ .

*Proof.* This proof follows almost exactly as the the proof of Theorem 3.4 in Appendix C.2.3. The only difference is that now every layer has the same dimension d, rather than d being increased with larger layer index  $\ell$ .

This just requires the following minor modification to the proof in Appendix C.2.3:

- When applying Lemma C.8, instead of constructing the mapping  $\mathcal{X} \to \mathbb{R}^d$ , we instead do the mapping  $\mathcal{X} \to \mathbb{R}^{d'}$ , with any d' < d (Note that the Lemma does not have constraint on the output dimension d). So the output of  $\sum \sigma(\cdot)$  is in a d'-dimensional subspace of  $\mathbb{R}^d$ .
- Updating Eq. 34, we set  $W_{\ell}$  to be a projection matrix with the first d-d' rows being 0, and the rest d' rows being a projection from  $\mathbb{R}^d$  to the  $\mathbb{R}^{d'}$  that  $\sum \sigma(\cdot)$  spans.

