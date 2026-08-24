# **B.** Algorithm of Distribution Fitting

<span id="page-11-1"></span>**Algorithm 1** Estimating Token Budget via Distribution Fitting

- 1: **Input:** Token sequence unpacking from sorted clusters  $\{x_1, x_2, \ldots, x_n\}$ , query Q, weight percentage threshold P, initial token count N, head dimension d
- 2: **Output:** Token budget K
- 3:
- 4: Compute  $\mu_1$  and  $\mu_2$  as the means of  $\exp(x_i \cdot Q/\sqrt{d})$  within fixed windows around  $p_1$  and  $p_2$ . Solve for parameters a and b in y = a/x + b the two data points.
- 5:
- 6: Initialize array  $w_i$  to store the simulated attention scores for all tokens
- 7: **for** i = 1 to n **do**
- 8: If  $i \leq N$ ,  $w_i = \exp(x_i \cdot Q/\sqrt{d})$
- 9: Else,  $w_i = a/i + b$
- 10: **end for**
- 11: Compute the minimal k such that the cumulative sum  $\sum_{1}^{k} w_i \ge P \cdot \sum_{1}^{n} w_i$ .
- 12:
- 13: **return** *k*