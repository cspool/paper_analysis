# <span id="page-14-0"></span>**D** No apparent closed-form solution for s > 3

We explain why our acceleration method from Section 4 fails for s > 3. Consider computing the location of two additional quantization values  $b \le u$  between  $x_a$  and  $x_c$ .

Similarly to the above analysis, we define by Q(b, u) the resulting sum of variances for all entries in  $[x_a, x_c]$ . Then:

$$Q(b,u) = \sum_{x \in [x_a,b]} (b-x)(x-x_a) + \sum_{x \in (b,u]} (u-x)(x-b) + \sum_{x \in (u,x_c]} (x_c-x)(x-u).$$

Computing the partial derivatives, we then get:

$$\frac{\partial Q(b, u)}{\partial b} = \sum_{x \in [x_a, b]} (x - x_a) - \sum_{x \in (b, u]} (u - x).$$
$$\frac{\partial Q(b, u)}{\partial u} = \sum_{x \in (b, u]} (x - b) - \sum_{x \in (u, x_c]} (x_c - x).$$

The challenge now is that both derivatives are non-continuous, and there are multiple indices i, j such that  $Q(x_i, x_j) < 0$  but  $Q(x_{i+1}, x_j) \ge 0$  or  $Q(x_i, x_{j+1}) \ge 0$ . Accordingly, it seems unlikely that a closed-form solution that is computable in constant time follows from this approach.

## Algorithm 4 Apx. QUIVER

```
1: Input: X \in \mathbb{R}^d, s, m \in \mathbb{N}.

2: S = \left\{ x_1 + \ell \cdot \frac{x_d - x_1}{m} \mid \ell \in \{0, \dots, m\} \right\}

3: Preprocess(X, m) \triangleright Enables computing C_m[k, j] in constant time (Appendix E).

4: for j = 2 to m do

5: MSE[2, j] = C_m[1, j]

6: for i = 3 to s do

7: K[i, \cdot] = \text{SMAWK}(Z) \triangleright Where Z[k, j] \triangleq MSE[i - 1, k] + C_m[k, j] \quad \forall k, j.

8: MSE[i, j] = MSE[i - 1, K[i, j]] + C_m[K[i, j], j] for all j \in \{i, \dots, m\}.

9: Q = \{s_0, s_m\}

10: j = m

11: for i = s down to 3 do

12: j = K[i, j]

13: Q = Q \cup \{s_j\}
```

