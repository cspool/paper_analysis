# A The Hash Algorithm for Simple Cycle Structure

Algorithm 1 presents the procedure for hashing FSMs in a simple cycle structure. In this setting, all FSMs referenced by those in the cycle are first hashed using Algorithm 1. Consequently, for each FSM in the cycle, exactly one referenced FSM remains unhashed, namely the next FSM in the cycle. We therefore assign a shared placeholder constant X to these unresolved references and compute a hash for each FSM using Algorithm 1. This yields a **local** hash value for each FSM, which captures only the individual FSM but not the overall cycle structure. Finally, we combine the local hash values of all FSMs in the cycle to derive the final hash for each FSM. Since the hash function is non-commutative, the resulting final hash values are unique.

#### <span id="page-10-2"></span>**Algorithm 2** Handle Simple Cycle Structure in FSM Reference

```
Input: a series of local hash values of simple-cycle FSMs L_0, L_1, ..., L_n
Output: a series of final hash values of simple-cycle FSMs H_0, H_1, ..., H_n
for i in range(n + 1) do
H_i \leftarrow 0
for j in range(n + 1) do
H_i \leftarrow \mathcal{H}(H_i, L_{\{(i+j) \bmod |L|\}})\nend for\nend for
return H_0, H_1, ..., H_n
```

