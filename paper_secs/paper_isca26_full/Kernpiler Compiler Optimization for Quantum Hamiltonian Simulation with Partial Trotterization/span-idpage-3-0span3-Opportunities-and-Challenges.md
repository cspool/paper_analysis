# <span id="page-3-0"></span>**3 Opportunities and Challenges**

**Opportunity**: Our optimization opportunities come from fine-grained analysis of the error terms in the approximation. The error between the Trotter product formula and exact Hamiltonian time evolution can be shown through the BCH formula [\[20\]](#page-13-6). The formula states:

$$\log\left(e^{\Delta t H_i} e^{\Delta t H_j}\right) = \Delta t H_i + \Delta t H_j + \frac{(\Delta t)^2}{2} [H_i, H_j] + \cdots \quad (2)$$

When approximating log *e* ∆*t*(*Hi*+*H<sup>j</sup>* ) with ∆*tH<sup>i</sup>* + ∆*tH<sup>j</sup>* , the dominant error term is (∆*t*) 2 [*H<sup>i</sup>* , *H<sup>j</sup>* ] + · · · . The higherorder nested commutators are of order (∆*t*) <sup>3</sup> and beyond. The primary optimization opportunity identified in this work is to reduce the effect of these commutators. As a small example, consider the following Hamiltonian with 4 terms where none commute with each other:

$$H = H_i + H_j + H_k + H_l$$
, where  $H_i = X_1Y_2Z_3$ ,  $H_j = Y_1Z_2X_3$   $H_k = Z_1X_2Y_3$ ,  $H_l = X_1Z_2X_3$ .

Now, naive Trotterization would give an error of the form:

$$\epsilon_{\text{full Trotter}} \propto [H_i, H_j] + [H_i, H_k] + [H_i, H_l] + [H_j, H_k]$$

$$+ [H_j, H_l] + [H_k, H_l]$$
(3)

However, if we did not fully Trotterize the Hamiltonian and instead kept *H<sup>i</sup>* + *H<sup>j</sup>* and *H<sup>k</sup>* + *H<sup>l</sup>* in the exponentials (see Figure [1\)](#page-1-0), there would be a smaller bound on the error term:

$$\epsilon_{\text{partial Trotter}} \propto [H_i, H_k] + [H_i, H_l] + [H_j, H_k] + [H_j, H_l]$$

This motivates us to consider grouping terms to contract the additive errors that arise from Trotterization. By strategically partitioning non-commuting operators into commuting partitions, we can potentially reduce the commutator error between terms, leading to lower overall Trotterization error and step counts. However, partitioning the Hamiltonian terms will immediately bring two challenges listed as follows.

**Challenge 1:** The first question is how we can partition the terms effectively. The objective of partitioning the Hamiltonian terms is to let the partitions be as dense as possible so that the follow-up compilation has more potential to rewrite the circuit with more gate count reduction. Without dense partitions, our rewrites would be very similar to the naive CNOT

tree decomposition of the Hamiltonian simulation compilation due to the lack of opportunity for gate cancellations in the rewrite. Existing quantum program partitioning mostly focus on gate-level circuit partitioning for circuit resynthesis [\[12\]](#page-12-15), [\[24\]](#page-13-11) which only collects adjacent gates. Other partitionings for specific Hamiltonians have also been explored [\[33\]](#page-13-12), however, existing partitioning techniques have not been generalized to other Hamiltonians of interest, and often require pre-processing circuits to allow for partitions to be analytically decomposed. Therefore, we believe that there is improvement to be made for Hamiltonian partitioning on the axes of generality and efficiency.

**Challenge 2:** Suppose we make a partition of Hamiltonian terms *H<sup>i</sup>* , *H<sup>j</sup>* , and *H<sup>k</sup>* . The second challenge is how to efficiently compile and optimize the unitary *e it*(*Hi*+*Hj*+*H<sup>k</sup>* ) as there is no established approach for the complicated exponentials. Previous approaches mostly focused on implementing the exponential of individual terms [\[27\]](#page-13-3), [\[14\]](#page-12-9), [\[23\]](#page-13-13). If we implement the exponential of these terms one by one, we naturally resort to the vanilla Trotterization and lose all the benefits of error reduction from partitioning. Additionally, there exists general unitary decompositions [\[26\]](#page-13-14), [\[41\]](#page-14-5), however the gate counts of these methods are very high and can hurt complexity savings from the partitions. Consequently, exploring more efficient approaches for decomposing unitaries is motivated by the hypothesis that using high level Hamiltonian structure and learning algorithms will allow for more efficient circuits.

We now summarize the opportunities and challenges. For conventional full Trotterization, the error at each step is relatively high, leading to a high Trotter step count while implementing the circuit of the exponential of individual Hamiltonian terms is easy. On the other hand, implementing partial Trotterization by partitioning the Hamiltonian terms will reduce the error and thus yield a low Trotter step count while the lack of efficient unitary decomposition methods may negate gates saved through less steps. Overall, our objective is to use partial Trotterization with a new term partitioning method and a new unitary decomposition method for the exponential of many Hamiltonian terms, achieving low Trotterization step count and low gate count in unitary decomposition simultaneously.

