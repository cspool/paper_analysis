# <span id="page-4-3"></span>4 The QUIVER Algorithm

To derive a faster algorithm, we observe that C satisfies the quadrangle inequality, defined below:

**Definition 4.1.** A function  $w: \{1, \dots, d\} \times \{1, \dots, d\} \to \mathbb{R}$  satisfies the quadrangle inequality if for any  $a \le b \le c \le d$ :  $w[a, c] + w[b, d] \le w[a, d] + w[b, c]$ .

<span id="page-4-2"></span>**Lemma 4.2.** C satisfies the quadrangle inequality.

*Proof.* We first observe that for any  $x \in [x_a, x_b]$ :

$$(x_{c} - x)(x - x_{a}) = (x_{d} - x)(x - x_{a}) + (x_{c} - x_{d})(x - x_{a}) \le (x_{d} - x)(x - x_{a}).$$
(1)

For any  $x \in [x_c, x_d]$ , we similarly get:

$$(x_{d} - x)(x - x_{b}) = (x_{d} - x)(x - x_{a}) + (x_{d} - x)(x_{a} - x_{b}) \le (x_{d} - x)(x - x_{a}).$$
(2)

Similarly, for  $x \in [x_b, x_c]$ , we have that:

$$(x_{c} - x)(x - x_{a}) + (x_{d} - x)(x - x_{b}) = (x_{c} - x)(x - x_{b}) + (x_{d} - x)(x - x_{a}) + (x_{a} - x_{b})(x_{d} - x_{c})$$

$$\leq (x_{c} - x)(x - x_{b}) + (x_{d} - x)(x - x_{a}). \tag{3}$$

Therefore, we get:

$$\begin{split} C[\mathbf{a},\mathbf{c}] + C[\mathbf{b},\mathbf{d}] &= \sum_{x \in [x_{\mathbf{a}},x_{\mathbf{c}}]} (x_{\mathbf{c}} - x)(x - x_{\mathbf{a}}) + \sum_{x \in [x_{\mathbf{b}},x_{\mathbf{d}}]} (x_{\mathbf{d}} - x)(x - x_{\mathbf{b}}) \\ &= \sum_{x \in [x_{\mathbf{a}},x_{\mathbf{b}}]} (x_{\mathbf{c}} - x)(x - x_{\mathbf{a}}) + \sum_{x \in [x_{\mathbf{c}},x_{\mathbf{d}}]} (x_{\mathbf{d}} - x)(x - x_{\mathbf{b}}) + \sum_{x \in [x_{\mathbf{b}},x_{\mathbf{c}}]} (x_{\mathbf{c}} - x)(x - x_{\mathbf{a}}) + (x_{\mathbf{d}} - x)(x - x_{\mathbf{b}}) \\ &\leq \sum_{x \in [x_{\mathbf{a}},x_{\mathbf{b}}]} (x_{\mathbf{d}} - x)(x - x_{\mathbf{a}}) + \sum_{x \in [x_{\mathbf{c}},x_{\mathbf{d}}]} (x_{\mathbf{d}} - x)(x - x_{\mathbf{a}}) + \sum_{x \in [x_{\mathbf{b}},x_{\mathbf{c}}]} (x_{\mathbf{c}} - x)(x - x_{\mathbf{b}}) + (x_{\mathbf{d}} - x)(x - x_{\mathbf{a}}). \\ &= \sum_{x \in [x_{\mathbf{a}},x_{\mathbf{d}}]} (x_{\mathbf{d}} - x)(x - x_{\mathbf{a}}) + \sum_{x \in [x_{\mathbf{b}},x_{\mathbf{c}}]} (x_{\mathbf{c}} - x)(x - x_{\mathbf{b}}) \\ &= C[\mathbf{a},\mathbf{d}] + C[\mathbf{b},\mathbf{c}]. \end{split}$$

Here, the inequality follows from equations (1)-(3).

Next, let us implicitly define a matrix  $A \in \mathbb{R}^{d \times d}$  such that A[k,j] = MSE[i-1,k] + C[k,j]. Importantly, A is not stored in memory but admits constant time lookups as  $MSE[i-1,\cdot]$  is stored and C is efficiently computable (Section 3). Also, C satisfies the quadrangle inequality and thus A is a totally monotone matrix [36], i.e., for any a < b and c < d:  $(A[a,c] > A[b,c]) \Longrightarrow (A[a,d] > A[b,d])$ . By applying the SMAWK algorithm [37], which finds the row minimas of an implicitly defined totally monotone matrix, on  $A^T$ , we obtain in O(d) time and space the indices  $k_j = \operatorname{argmin}_{k \in \{1, \dots, d\}} A[k,j]$  for all  $j \in \{1, \dots, d\}$ . This immediately gives the next row of the dynamic program, as  $MSE[i,j] = MSE[i-1,k_j] + C[k_j,j]$ .

<span id="page-4-1"></span><span id="page-4-0"></span>

The resulting solution, which we call QUIVER, is given in Algorithm 1 and requires just  $O(s \cdot d)$  time and space to compute the optimal quantization values.

## 5 The Accelerated QUIVER Algorithm

To accelerate QUIVER, we rely on the observation that while the problem is non-convex for s > 3, it admits a closed-form solution when s = 3.

Denoting by  $C^2[k,j] = \min_{b \in \{k,\dots,j\}} (C[k,b] + C[b,j])$  the optimal MSE of quantizing the range  $[x_k,x_j]$  using three quantization values (at  $x_k,x_b,x_j$ ), we show how to compute  $C^2$  in constant time.

Namely, consider adding a quantization value  $q \in [x_k, x_j]$  (not necessarily in X) between two existing quantization values  $x_k$  and  $x_j$ . Let us define the sum of variances of all input entries in  $[x_k, x_j]$  as a function of q:  $Q(q) = \sum_{x \in [x_k, q]} (q - x)(x - x_k) + \sum_{x \in (q, x_j]} (x_j - x)(x - q)$ . This function is differentiable in  $[x_k, x_j] \setminus X$ , and we get:  $\frac{dQ(q)}{dq} = \sum_{x \in [x_k, q]} (x - x_k) - \sum_{x \in (q, x_j]} (x_j - x)$ .

## **Algorithm 1** QUIVER

```
1: Input: X \in \mathbb{R}^d, s \in \mathbb{N}.
                                                                                                      \triangleright X is sorted.
 2: Preprocess(X)
                                                  \triangleright Enables computing C[k, j] in constant time (Section 3).
 3: for j = 2 to d do
         MSE[2, j] = C[1, j]
 5: for i = 3 to s do
                                                            \triangleright Where A[k,j] \triangleq MSE[i-1,k] + C[k,j] \quad \forall k, j.
         K[i,\cdot] = SMAWK(A)
         MSE[i, j] = MSE[i - 1, K[i, j]] + C[K[i, j], j] for all j \in \{i, ..., d\}.
 7:
 8: Q = \{x_1, x_d\}
 9: i = d
10: for i = s down to 3 do
11:
         j = K[i, j]
         Q = Q \cup \{x_i\}
12:
13: return Q
```

## <span id="page-5-16"></span><span id="page-5-9"></span><span id="page-5-8"></span><span id="page-5-5"></span><span id="page-5-0"></span>Algorithm 2 Accelerated QUIVER

```
1: Input: X \in \mathbb{R}^d, s \in \mathbb{N}.
                                                                                                                                   \triangleright X is sorted.
                                                             \triangleright Enables computing C[k, j] and C^2[k, j] in constant time.
 2: Preprocess(X)
 3: s' = (s \mod 2)
 4: if s' = 0 then
           for j = 2 to d do
                 MSE[2, j] = C[1, j]
 6:
 7: else
           for j = 3 to d do
 8:
                 MSE[3, j] = C^{2}[1, j]
10: for i = 2 to |s/2| do
           \begin{split} K[i,\cdot] &= \text{SMAWK}(B) & \qquad \text{$\triangleright$ Where } B[k,j] \triangleq MSE[2\cdot(i-1)+s',k] + C^2[k,j] \quad \forall k,j. \\ MSE[2\cdot i+s',j] &= MSE[2\cdot(i-1)+s',K[i,j]] + C^2[K[i,j],j] \quad \forall j \in \{i,\dots,d\}. \end{split}
11:
12:
13: Q = \{x_1, x_d\}
14: j = d
15: for i = |s/2| down to 2 do
           b^* = \mathop{\rm argmin}_{b \in \{K[i,j],...,j\}} \left( C[K[i,j],b] + C[b,j] \right)
                                                                                                                          \triangleright Takes O(1) time.
           j = K[i, j]
Q = Q \cup \{x_j, x_{b^*}\}
17:
19: if s' = 1 then
           b^* = \operatorname{argmin}_{b \in \{0, \dots, j\}} (C[0, b] + C[b, j])
20:
                                                                                                                          \triangleright Takes O(1) time.
           Q = Q \cup \{x_{b^*}\}
22: return Q
```

<span id="page-5-14"></span><span id="page-5-10"></span><span id="page-5-7"></span><span id="page-5-6"></span><span id="page-5-1"></span>Notice that the derivative is monotonically non-decreasing and for any  $\ell \in \{k, k+1, \ldots, j-1\}$  the derivative is fixed (independent of q) over any interval  $(x_\ell, x_{\ell+1})$ . This means that Q(q) is minimized at  $u = \inf_q (\frac{dQ(q)}{dq} \geq 0)$ , where  $u \in X$ . Denote by  $b_{k,j}^* \in \{k, \ldots, j\}$  the value such that  $x_{b_{k,j}^*} = u$ . Notice that while  $\frac{dQ(u)}{dq}$  may not be defined, we have that  $\lim_{h \to 0^+} \frac{dQ(u+h)}{dq} \geq 0$  is well-defined.

We thus require  $\sum_{i=k+1}^{b_{k,j}^*}(x_i-x_k)-\sum_{i=b_{k,j}^*+1}^j(x_j-x_i)\geq 0$ . With some simplifications, this is equivalent to:  $\sum_{i=k+1}^j x_i-(b_{k,j}^*-k)x_k-(j-b_{k,j}^*)x_j\geq 0$ , yielding  $b_{k,j}^*\geq \frac{jx_j-kx_k-\sum_{i=k+1}^j x_i}{x_i-x_k}$ .

As  $b_{k,j}^*$  is an integer, we get a formula for  $C^2[k,j]$  that can be computed in constant time using:  $b_{k,j}^* = \lceil \frac{jx_j - kx_k - \sum_{i=k+1}^j x_i}{x_j - x_k} \rceil = \lceil \frac{jx_j - kx_k - (\beta_j - \beta_k)}{x_j - x_k} \rceil$ . That is, for any  $1 \le k \le j \le d$  we have that  $C^2[k,j] = C[k,b_{k,j}^*] + C[b_{k,j}^*,j]$  is the sum of the variances in quantizing the entries in  $[x_k,x_j]$  using the quantization values  $\{x_k,x_{b_{k,j}^*},x_j\}$ .

We can then use this method to halve the required number of invocations of SMAWK by always using it to pick the *second-next* quantization value and computing the optimal quantization value in between directly. Our accelerated dynamic program is then given by:

$$MSE[i,j] = \begin{cases} \min_{k \in \{i, \dots, j\}} & MSE[i-2, k] + C^{2}[k, j] & i > 3 \\ C^{2}[1, j] & i = 3 \\ C[1, j] & i = 2 \end{cases},$$

and the resulting pseudo-code for Accelerated QUIVER is given by Algorithm 2. Similarly to QUIVER, we start by initializing the first row of MSE. Importantly, we now separate the even s case (lines 5-6), in which we initialize the row using C, and the odd case, where we use  $C^2$  (lines 8-9). That is, the odd s case 'skips' a quantization value that we later determine separately (lines 19-21). Next, denoting  $s' = (s \mod 2)$ , we proceed with  $\lfloor s/2 \rfloor -1$  invocations of the SMAWK algorithm (lines 10-12), applied on the implicitly defined matrix  $B[k,j] \triangleq MSE[2 \cdot (i-1) + s', K[i,j]] + C^2[K[i,j],j]$ . The output yields the minimizers of  $MSE[2 \cdot i + s',j]$  used for reconstruction. In the reconstruction step (lines 15-21), we fill in the missing quantization values by finding the optimal value between every two outputs from the dynamic program minimizers K.

Overall, the Accelerated QUIVER algorithm requires at most half of the number of SMAWK invocations compared to QUIVER and at most half of the memory to store K and MSE.

To establish correctness, we state the following lemma, whose proof appears in Appendix C.

<span id="page-6-0"></span>**Lemma 5.1.**  $C^2$  satisfies the quadrangle inequality.

In Appendix D, we discuss why this approach is not suitable for further acceleration by placing more than one quantization value in  $[x_a, x_c]$ .

