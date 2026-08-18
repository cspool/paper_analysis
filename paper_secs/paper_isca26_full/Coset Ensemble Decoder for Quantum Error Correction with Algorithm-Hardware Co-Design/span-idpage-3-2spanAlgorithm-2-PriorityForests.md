# <span id="page-3-2"></span>Algorithm 2 PriorityForests

```
Require: Graph G = (V, E); Priorities ϕ : V, E → (0, 1)
Ensure: Array parent: V → V ∪ {NIL}; discovery order σ
1: visited[v] ← false, parent[v] ← NIL for all v ∈ V; σ ←
   [ ]; Q ← QUEUE
2: ΠV ← V sorted by ascending ϕ
3: for all u ∈ ΠV do
4: if not visited[u] then
5: ENQUEUE(Q, u); visited[u] ← true; PUSH(σ, u)
6: while not EMPTY(Q) do
7: x ← DEQUEUE(Q)
8: Adj ← ADJ(G, x) sorted by ascending
9: for all y ∈ Adj do
10: if not visited[y] then
11: parent[y] ← x; visited[y] ← true;
   ENQUEUE(Q, y); PUSH(σ, y)
12: return (parent, σ)
```

### <span id="page-3-3"></span>Algorithm 3 Reverse-Order Elimination (ROE)

```
Require: Array parent; discovery order σ; parity s ∈ {0, 1}
                                                      V
Ensure: Correction Ei ⊆ E
 1: Ei ← ∅; p ← s
 2: for t = |σ| down to 1 do
 3: x ← σt; r ← parent[x]
 4: if r ̸= NIL and p[x] = 1 then
 5: Ei ← Ei ∪ {(x, r)}
 6: p[x] ← p[x] ⊕ 1; p[r] ← p[r] ⊕ 1
 7: Li = DECODELOGICAL(Ei)
 8: return {Ei, Li}
```

*be partitioned into* V<sup>t</sup> *and* Vnt *such that* V<sup>t</sup> = {v|s(v) = +1} *is called the set of trivial syndromes,* Vnt = {v|s(v) = −1} *the set of non-trivial syndromes, and the size of non-trivial syndromes is even. If the size of non-trivial syndromes is zero, we call the syndrome graph trivial. The input decoding graph of our algorithm is a non-trivial syndrome graph.*

Definition 2 (Clustering). *Clustering takes in the input decoding graph* G(V, E) *and outputs a partition of sub-graphs* C = {Gi(V<sup>i</sup> , Ei)} *such that each* G<sup>i</sup> *is a syndrome graph and precisely one* G<sup>i</sup> *is trivial.*

Premise 1. *After clustering, each non-trivial syndrome graph is sent to ensemble-forest-exploration. The* K *independent priority samples induce* K *error ensembles* {E<sup>i</sup> , Li} K <sup>i</sup>=1*, where* E<sup>i</sup> ∈ P<sup>n</sup> *and* L<sup>i</sup> *is the corresponding logical error.*

We now present two lemmas to substantiate the claim that our algorithm solves a sub-optimal coset ML problem. The first lemma demonstrates that the errors of candidate error ensembles are degenerate, forming logically equivalent cosets. The second lemma analyzes the algorithm's asymptotic optimality within the partitioned solution space after clustering.

Lemma 1. *For* K *error ensembles* {E<sup>i</sup> , Li}*, the* E<sup>i</sup> *with equal* L<sup>i</sup> *are degenerate errors and belong to the same logical equivalent coset of the stabilizer group* S*.*

*Proof.* As shown in Eq. [4,](#page-1-2) s(E) is a stabilizer term that could deform the error chain, or, equivalently, affect the matching in surface code. t(s) corresponds to the pure error operator, which only depends on the syndrome measurement s, and l(E) is the operator of the E's logical error. Therefore, for any two error ensembles  $\{E_1, L\}$  and  $\{E_2, L\}$  with the same logical error, they both derive from the same syndrome measurement pattern, so their error expansions form the same t(s) and l(E). Because L, S, and t(s) are all Hermitian and unitary,

$$E_1 E_2^{\dagger} = (S_1 t(s) L) (S_2 t(s) L)^{\dagger} = S_1 S_2 \tag{7}$$

Therefore, it's proved that for all error ensembles with the same logical error, their errors  $E_i$  are degenerate and belong to the same logical-equivalent coset.

**Lemma 2.** The clustering C reformulates the global optimal coset decoding problem  $\operatorname{argmax}_L\{p(L|s)\}$  by solving a locally optimal Maximum-Likelihood problem

$$\underset{L}{\operatorname{argmax}} \left\{ \sum_{b \in \mathcal{B}_{-}} p(E = S(b)t(s)L) \right\}$$
 (8)

where for index c running over all non-trivial syndrome graphs in the cluster,

<span id="page-4-1"></span>
$$\mathcal{B}_c = \{ b \in \mathbb{F}_2^m | b_g = 0 \text{ for all } g \text{ with } s_g \notin G_c \}$$
 (9)

is the set of all m-bit strings whose support is restricted to the indices in the cluster.

*Proof.* Given a stabilizer code and the syndrome measurement result s, the optimal decoding is to find the logical-equivalent coset with the highest probability by solving Eq. 6. Since the operator of each stabilizer generator  $S_g$  can be constructed as  $S_g = UZ_gU^{\dagger}$ , and each stabilizer in  $\mathcal S$  is the product of several stabilizer generators, the stabilizer S can be written as:

$$S = U Z_m^{b_m} U^{\dagger} U Z_{m-1}^{b_{m-1}} U^{\dagger} \cdots U Z_1^{b_1} U^{\dagger}$$

$$= U \left( Z_m^{b_m} Z_{m-1}^{b_{m-1}} \cdots Z_1^{b_1} \right) U^{\dagger}, \quad b_g \in \{0, 1\}$$
(10)

If  $b_g=1$ , the g-th stabilizer generator is multiplied by the pure error term, which would cause a local deformation on the error chain. Therefore, if an m-bit bitstring b is constructed by concatenating  $b_g$  as  $b=\bigoplus_{g=1}^m b_g\cdot (1\ll (g-1))$ , the error operator will only depend on the value of this bitstring and can be rewritten as E=S(b)t(s)L. The activation of g-th bit of b represents the contribution of  $S_g$  on deforming the final error chain. In this case, the original coset probability becomes:

$$\sum_{S \in \mathcal{S}} p(E = St(s)L) \equiv \sum_{b \in \mathbb{F}_2^m} p(E = S(b)t(s)L) \tag{11}$$

After clustering  $\mathcal{C}$ , the error chain could only be modified locally within each cluster. This implies that only the nontrivial (activated) syndromes within a cluster contribute to its error deformation. Given the one-to-one correspondence between each bit  $b_g$  of b, a syndrome bit  $s_g$ , and a stabilizer  $S_g$ ,  $b_g$  can vary (0 or 1) or deactivate (set to 0) if its corresponding syndrome  $s_g$  is inside or outside any cluster  $\mathcal{G}_c$ . Each cluster thus defines a local configuration. The space of valid bitstrings is a subset of  $\mathbb{F}_2^m$ , which is the union of the spaces  $\mathcal{B}_c$  as given

in (9). In this framework, the clustering  $\mathcal{C}$  approximates the original global optimization problem from a locally optimal version by partitioning the stabilizer space into activated and deactivated regions.

Based on the preceding lemmas, the error ensembles produced contain degenerate errors that can be grouped into logically equivalent cosets based on their logical errors. Under the priority-sampling distribution, the sample frequency of a logical outcome estimates its probability mass within the partitioned candidate space, which can be represented as  $\tilde{p}(L_i|s) = \frac{n_{L_i}}{K}$ , where  $n_{L_i}$  is the number of ensembles with logical error  $L_i$ . The MAJORVOTE thus identifies the most frequently sampled coset and approximates the sub-optimal coset ML problem as

<span id="page-4-2"></span>
$$\underset{L_i}{\operatorname{argmax}} \{ \tilde{p}(L_i|s) \} = \underset{L_i}{\operatorname{argmax}} \left( \frac{n_{L_i}}{K} \right)$$
 (12)

A final correction can then be chosen arbitrarily from this coset due to the degeneracy among candidates. Moreover, as the candidate number  $K \to \infty$ , the sampling estimate converges within this partitioned candidate space, though performance remains bounded below the original optimal coset ML due to the clustering constraint. In practice, this vote is restricted to the candidates with the smallest correction size  $|E_i|$ . This empirical refinement improves accuracy and reduces to Eq. (12) when all candidates share the same correction size.

3) Relationship to UF and MWPM: Our coset ensemble decoder typically sits between UF and MWPM. It leverages UF's efficient clustering but critically advances it by introducing a coset-decoding step, which produces multiple error ensembles to identify the most probable logical coset and thus outperforms UF. Directly contrasting with MWPM reveals the impact of the clustering stage. MWPM performs maximum-likelihood (ML) decoding on physical errors, whereas our coset ensemble decoder solves a constrained version of the coset maximum-likelihood problem, and the solution space, a subset of all cosets defined by the clustering, is sub-optimal. Consequently, our decoder may exceed MWPM's accuracy only when the cluster structure aligns with the error structure that MWPM's Blossom would capture.

# <span id="page-3-2"></span>Algorithm 2 PriorityForests

```
Require: Graph G = (V, E); Priorities ϕ : V, E → (0, 1)
Ensure: Array parent: V → V ∪ {NIL}; discovery order σ
1: visited[v] ← false, parent[v] ← NIL for all v ∈ V; σ ←
   [ ]; Q ← QUEUE
2: ΠV ← V sorted by ascending ϕ
3: for all u ∈ ΠV do
4: if not visited[u] then
5: ENQUEUE(Q, u); visited[u] ← true; PUSH(σ, u)
6: while not EMPTY(Q) do
7: x ← DEQUEUE(Q)
8: Adj ← ADJ(G, x) sorted by ascending
9: for all y ∈ Adj do
10: if not visited[y] then
11: parent[y] ← x; visited[y] ← true;
   ENQUEUE(Q, y); PUSH(σ, y)
12: return (parent, σ)
```

### <span id="page-3-3"></span>Algorithm 3 Reverse-Order Elimination (ROE)

```
Require: Array parent; discovery order σ; parity s ∈ {0, 1}
                                                      V
Ensure: Correction Ei ⊆ E
 1: Ei ← ∅; p ← s
 2: for t = |σ| down to 1 do
 3: x ← σt; r ← parent[x]
 4: if r ̸= NIL and p[x] = 1 then
 5: Ei ← Ei ∪ {(x, r)}
 6: p[x] ← p[x] ⊕ 1; p[r] ← p[r] ⊕ 1
 7: Li = DECODELOGICAL(Ei)
 8: return {Ei, Li}
```

*be partitioned into* V<sup>t</sup> *and* Vnt *such that* V<sup>t</sup> = {v|s(v) = +1} *is called the set of trivial syndromes,* Vnt = {v|s(v) = −1} *the set of non-trivial syndromes, and the size of non-trivial syndromes is even. If the size of non-trivial syndromes is zero, we call the syndrome graph trivial. The input decoding graph of our algorithm is a non-trivial syndrome graph.*

Definition 2 (Clustering). *Clustering takes in the input decoding graph* G(V, E) *and outputs a partition of sub-graphs* C = {Gi(V<sup>i</sup> , Ei)} *such that each* G<sup>i</sup> *is a syndrome graph and precisely one* G<sup>i</sup> *is trivial.*

Premise 1. *After clustering, each non-trivial syndrome graph is sent to ensemble-forest-exploration. The* K *independent priority samples induce* K *error ensembles* {E<sup>i</sup> , Li} K <sup>i</sup>=1*, where* E<sup>i</sup> ∈ P<sup>n</sup> *and* L<sup>i</sup> *is the corresponding logical error.*

We now present two lemmas to substantiate the claim that our algorithm solves a sub-optimal coset ML problem. The first lemma demonstrates that the errors of candidate error ensembles are degenerate, forming logically equivalent cosets. The second lemma analyzes the algorithm's asymptotic optimality within the partitioned solution space after clustering.

Lemma 1. *For* K *error ensembles* {E<sup>i</sup> , Li}*, the* E<sup>i</sup> *with equal* L<sup>i</sup> *are degenerate errors and belong to the same logical equivalent coset of the stabilizer group* S*.*

*Proof.* As shown in Eq. [4,](#page-1-2) s(E) is a stabilizer term that could deform the error chain, or, equivalently, affect the matching in surface code. t(s) corresponds to the pure error operator, which only depends on the syndrome measurement s, and l(E) is the operator of the E's logical error. Therefore, for any two error ensembles  $\{E_1, L\}$  and  $\{E_2, L\}$  with the same logical error, they both derive from the same syndrome measurement pattern, so their error expansions form the same t(s) and l(E). Because L, S, and t(s) are all Hermitian and unitary,

$$E_1 E_2^{\dagger} = (S_1 t(s) L) (S_2 t(s) L)^{\dagger} = S_1 S_2 \tag{7}$$

Therefore, it's proved that for all error ensembles with the same logical error, their errors  $E_i$  are degenerate and belong to the same logical-equivalent coset.

**Lemma 2.** The clustering C reformulates the global optimal coset decoding problem  $\operatorname{argmax}_L\{p(L|s)\}$  by solving a locally optimal Maximum-Likelihood problem

$$\underset{L}{\operatorname{argmax}} \left\{ \sum_{b \in \mathcal{B}_{-}} p(E = S(b)t(s)L) \right\}$$
 (8)

where for index c running over all non-trivial syndrome graphs in the cluster,

<span id="page-4-1"></span>
$$\mathcal{B}_c = \{ b \in \mathbb{F}_2^m | b_g = 0 \text{ for all } g \text{ with } s_g \notin G_c \}$$
 (9)

is the set of all m-bit strings whose support is restricted to the indices in the cluster.

*Proof.* Given a stabilizer code and the syndrome measurement result s, the optimal decoding is to find the logical-equivalent coset with the highest probability by solving Eq. 6. Since the operator of each stabilizer generator  $S_g$  can be constructed as  $S_g = UZ_gU^{\dagger}$ , and each stabilizer in  $\mathcal S$  is the product of several stabilizer generators, the stabilizer S can be written as:

$$S = U Z_m^{b_m} U^{\dagger} U Z_{m-1}^{b_{m-1}} U^{\dagger} \cdots U Z_1^{b_1} U^{\dagger}$$

$$= U \left( Z_m^{b_m} Z_{m-1}^{b_{m-1}} \cdots Z_1^{b_1} \right) U^{\dagger}, \quad b_g \in \{0, 1\}$$
(10)

If  $b_g=1$ , the g-th stabilizer generator is multiplied by the pure error term, which would cause a local deformation on the error chain. Therefore, if an m-bit bitstring b is constructed by concatenating  $b_g$  as  $b=\bigoplus_{g=1}^m b_g\cdot (1\ll (g-1))$ , the error operator will only depend on the value of this bitstring and can be rewritten as E=S(b)t(s)L. The activation of g-th bit of b represents the contribution of  $S_g$  on deforming the final error chain. In this case, the original coset probability becomes:

$$\sum_{S \in \mathcal{S}} p(E = St(s)L) \equiv \sum_{b \in \mathbb{F}_2^m} p(E = S(b)t(s)L) \tag{11}$$

After clustering  $\mathcal{C}$ , the error chain could only be modified locally within each cluster. This implies that only the nontrivial (activated) syndromes within a cluster contribute to its error deformation. Given the one-to-one correspondence between each bit  $b_g$  of b, a syndrome bit  $s_g$ , and a stabilizer  $S_g$ ,  $b_g$  can vary (0 or 1) or deactivate (set to 0) if its corresponding syndrome  $s_g$  is inside or outside any cluster  $\mathcal{G}_c$ . Each cluster thus defines a local configuration. The space of valid bitstrings is a subset of  $\mathbb{F}_2^m$ , which is the union of the spaces  $\mathcal{B}_c$  as given

in (9). In this framework, the clustering  $\mathcal{C}$  approximates the original global optimization problem from a locally optimal version by partitioning the stabilizer space into activated and deactivated regions.

Based on the preceding lemmas, the error ensembles produced contain degenerate errors that can be grouped into logically equivalent cosets based on their logical errors. Under the priority-sampling distribution, the sample frequency of a logical outcome estimates its probability mass within the partitioned candidate space, which can be represented as  $\tilde{p}(L_i|s) = \frac{n_{L_i}}{K}$ , where  $n_{L_i}$  is the number of ensembles with logical error  $L_i$ . The MAJORVOTE thus identifies the most frequently sampled coset and approximates the sub-optimal coset ML problem as

<span id="page-4-2"></span>
$$\underset{L_i}{\operatorname{argmax}} \{ \tilde{p}(L_i|s) \} = \underset{L_i}{\operatorname{argmax}} \left( \frac{n_{L_i}}{K} \right)$$
 (12)

A final correction can then be chosen arbitrarily from this coset due to the degeneracy among candidates. Moreover, as the candidate number  $K \to \infty$ , the sampling estimate converges within this partitioned candidate space, though performance remains bounded below the original optimal coset ML due to the clustering constraint. In practice, this vote is restricted to the candidates with the smallest correction size  $|E_i|$ . This empirical refinement improves accuracy and reduces to Eq. (12) when all candidates share the same correction size.

3) Relationship to UF and MWPM: Our coset ensemble decoder typically sits between UF and MWPM. It leverages UF's efficient clustering but critically advances it by introducing a coset-decoding step, which produces multiple error ensembles to identify the most probable logical coset and thus outperforms UF. Directly contrasting with MWPM reveals the impact of the clustering stage. MWPM performs maximum-likelihood (ML) decoding on physical errors, whereas our coset ensemble decoder solves a constrained version of the coset maximum-likelihood problem, and the solution space, a subset of all cosets defined by the clustering, is sub-optimal. Consequently, our decoder may exceed MWPM's accuracy only when the cluster structure aligns with the error structure that MWPM's Blossom would capture.

