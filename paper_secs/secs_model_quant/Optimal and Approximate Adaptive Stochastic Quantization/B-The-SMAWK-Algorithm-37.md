# B The SMAWK Algorithm [37]

Here, we provide some intuition into how SMAWK operates and achieves its efficiency. The SMAWK algorithm has four main steps:

- **Pruning Phase:** Remove columns that cannot possibly contain a row maximum. This is done by comparing each column with its neighbors and discarding those that cannot be maxima based on the totally monotone property. At the end of this phase, the number of columns can be no larger than the number of rows.
- **Recursive Reduction:** The algorithm reduces the problem size by considering a subset of the rows and columns. It selects every other row and recursively solves the reduced problem.
- Candidate Set: After solving the smaller problem, the solution provides candidate columns for the original problem. The algorithm only needs to consider these columns to find the maxima for the skipped rows.
- Merge Phase: Combine the results from the reduced problem with the candidate set to find the maximum for each original row.

Regarding efficiency, the SMAWK algorithm achieves a time complexity of O(d) for a  $d \times d$  matrix. This efficiency is due to the recursive reduction of the problem size and the properties of totally monotone matrices that limit the number of comparisons needed. Namely, the pruning step takes O(#cols), where #cols is the number of columns still being considered. The crux is that the recursive step happens after the pruning, which means that the recursive invocation happens with a number of columns that is, at most, double the number of rows (as the number of rows is halved). This means that the overall complexity of each recursive step is proportional to the number of rows, yielding the recursion: T(n) = T(n/2) + O(n) = O(n). A simple example Python implementation (by David Eppstein) appears here [38]. Our implementation is in optimized C++ [30].

### <span id="page-13-1"></span>C Proof of Lemma 5.1

**Lemma 5.1.**  $C^2$  satisfies the quadrangle inequality.

*Proof.* The lemma claims that, for any  $a \le b \le c \le d$ :

$$C^{2}[\mathbf{a}, \mathbf{c}] + C^{2}[\mathbf{b}, \mathbf{d}] \leq C^{2}[\mathbf{a}, \mathbf{d}] + C^{2}[\mathbf{b}, \mathbf{c}].$$

Recall that for any  $a \le c \in \{1, \dots, d\}$ , we denote

$$b_{a,c}^* = \underset{b \in \{a,...,c\}}{\operatorname{argmin}} C[a,b] + C[b,c].$$

We prove the lemma by a case analysis:

• Case  $b_{b,c}^* \leq b_{a,d}^*$ . In this case, we have that:

$$\begin{split} C^2(\mathbf{a},\mathbf{c}) + C^2(\mathbf{b},\mathbf{d}) &= C(\mathbf{a},b^*_{\mathbf{a},\mathbf{c}}) + C(b^*_{\mathbf{a},\mathbf{c}},\mathbf{c}) + C(\mathbf{b},b^*_{\mathbf{b},\mathbf{d}}) + C(b^*_{\mathbf{b},\mathbf{d}},\mathbf{d}) \\ &\leq C(\mathbf{a},b^*_{\mathbf{b},\mathbf{c}}) + C(b^*_{\mathbf{b},\mathbf{c}},\mathbf{c}) + C(\mathbf{b},b^*_{\mathbf{a},\mathbf{d}}) + C(b^*_{\mathbf{a},\mathbf{d}},\mathbf{d}) \\ &\leq C(\mathbf{b},b^*_{\mathbf{b},\mathbf{c}}) + C(b^*_{\mathbf{b},\mathbf{c}},\mathbf{c}) + C(\mathbf{a},b^*_{\mathbf{a},\mathbf{d}}) + C(b^*_{\mathbf{a},\mathbf{d}},\mathbf{d}) \\ &= C^2(\mathbf{b},\mathbf{c}) + C^2(\mathbf{a},\mathbf{d}). \end{split}$$

Here, the Inequality (i) follows from the definition of  $b_{a,c}^*$  that minimizes the MSE over the interval  $[x_{\mathbf{a}}, x_{\mathbf{c}}]$  and  $b_{b,d}^*$  that minimizes it over  $[x_{\mathbf{b}}, x_{\mathbf{d}}]$ . Inequality (ii) follows from the quadrangle inequality of C (Lemma 4.2), as  $\mathbf{a} \leq \mathbf{b} \leq b_{\mathbf{b},\mathbf{c}}^* \leq b_{\mathbf{a},\mathbf{d}}^*$ , and thus

$$C(\mathtt{a},b^*_{\mathtt{b},\mathtt{c}}) + C(\mathtt{b},b^*_{\mathtt{a},\mathtt{d}}) \leq C(\mathtt{b},b^*_{\mathtt{b},\mathtt{c}}) + C(\mathtt{a},b^*_{\mathtt{a},\mathtt{d}}).$$

• Case  $b_{b,c}^* > b_{a,d}^*$ . In this case, we have that:

$$\begin{split} C^2(\mathbf{a},\mathbf{c}) + C^2(\mathbf{b},\mathbf{d}) &= C(\mathbf{a},b^*_{\mathbf{a},\mathbf{c}}) + C(b^*_{\mathbf{a},\mathbf{c}},\mathbf{c}) + C(\mathbf{b},b^*_{\mathbf{b},\mathbf{d}}) + C(b^*_{\mathbf{b},\mathbf{d}},\mathbf{d}) \\ &\leq C(\mathbf{a},b^*_{\mathbf{a},\mathbf{d}}) + C(b^*_{\mathbf{a},\mathbf{d}},\mathbf{c}) + C(\mathbf{b},b^*_{\mathbf{b},\mathbf{c}}) + C(b^*_{\mathbf{b},\mathbf{c}},\mathbf{d}) \\ &\leq C(\mathbf{b},b^*_{\mathbf{b},\mathbf{c}}) + C(b^*_{\mathbf{b},\mathbf{c}},\mathbf{c}) + C(\mathbf{a},b^*_{\mathbf{a},\mathbf{d}}) + C(b^*_{\mathbf{a},\mathbf{d}},\mathbf{d}) \\ &= C^2(\mathbf{b},\mathbf{c}) + C^2(\mathbf{a},\mathbf{d}). \end{split}$$

Here, the Inequality (i) follows again from  $b_{a,c}^*$  and  $b_{b,d}^*$  being optimal for  $[x_a, x_c]$  and  $[x_b, x_d]$ . Inequality (ii) follows from the quadrangle inequality of C, as  $b_{a,d}^* \leq b_{b,c}^* \leq c \leq d$  and, therefore,

$$C(b_{a,d}^*, c) + C(b_{b,c}^*, d) \le C(b_{a,d}^*, d) + C(b_{b,c}^*, c).$$

Together, this concludes the proof.

