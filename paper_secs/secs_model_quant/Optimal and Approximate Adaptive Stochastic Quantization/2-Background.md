# 2 Background

#### 2.1 Motivation

We now briefly explain the benefits of ASQ compared to alternative methods.

The benefits of adaptivity Unbiased solutions such as QSGD [14] and NUQSGD [29] rely only on global properties (e.g., the input's norm) when selecting Q. Figure 1(a) shows the benefit of adaptivity by illustrating the potential MSE reduction from selecting Q optimally for the specific input. A similar behavior is observed for biased methods where the non-adaptive Round-To-Nearest (RTN) has a higher error than the optimal adaptive biased scalar quantizer, k-means. As shown, this can translate to orders of magnitude lower error, depending on the data's skew.

The benefits of unbiasedness In many cases, it is beneficial for the quantization to be unbiased. For example, when there are n senders (e.g., when doing distributed mean estimation [1, 2, 4, 17, 18]), having unbiased and independent estimates of the vectors allows the mean estimation's MSE to decay proportionally to  $\frac{1}{n}$ ; with biased quantization, the MSE may not decay with respect to n since the errors may be correlated [17] (e.g., when all clients have the same vector). This benefit is demonstrated in Figure 1(b), which shows that while biased adaptive solutions have lower error for a small number of vectors (1-2), having unbiased quantization is critical to lowering the error for a large n.

As another example, it was recently shown that compressing large language model parameters with biased techniques such as RTN may result in inferior performance than uniform stochastic quantization [35]. This outcome arises because the LLM layers' parameters are used to compute inner products with their inputs. Having these inner products themselves be unbiased leads to smaller errors in layers' outputs, which in turn leads to better performance.

### 2.2 Preliminaries

Given two quantization values a,b and a number  $x\in[a,b]$ , Stochastic Quantization (SQ) is a procedure that rounds x to  $\widehat{x}$  where  $\widehat{x}\in\{a,b\}$ . Specifically,  $\widehat{x}$  obtains the value a with probability  $p_a=\frac{b-x}{b-a}$  and the value b otherwise, i.e., with probability  $p_b=1-p_a=\frac{x-a}{b-a}$ . An important property of SQ is that the expected rounded value is a unbiased, i.e.,  $\mathbb{E}\left[\widehat{x}\right]=a\cdot p_a+b\cdot p_b=x$ . The variance of stochastically quantizing a is then given by  $\mathbb{E}\left[(x-\widehat{x})^2\right]=(x-a)^2\cdot p_a+(x-b)^2\cdot p_b=(b-x)(x-a)$ . Given a vector a0 undiagraph and an integer a1 undiagraph 2, the Adaptive Stochastic Quantization (ASQ) problem [26, 27, 28] looks for a set of quantization values a2 where a3 undiagraph 3 undiagraph 4 undiagraph 5 undiagraph 6 undiagraph 6 undiagraph 7 undiagraph 8 undiagraph 8 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 undiagraph 9 und

Formally, ASQ seeks to minimize the MSE, given by  $\mathbb{E}[\|X - \widehat{X}\|_2^2] = \sum_{x \in X} (b_x - x)(x - a_x)$ , where  $\mathbb{E}[\widehat{X}] = X$  holds by construction.

## 2.3 Existing ASQ methods

Leveraging the fact that there exists an optimal solution in which  $Q \subseteq X$  [26] (i.e., the quantization values are a subset of the input), one can naively solve the problem in  $d^{\Theta(s)}$  time by going over all choices for the quantization values. Instead, the following dynamic program (DP) allows us to solve it optimally and in polynomial time for any s [26]. Given a *sorted* vector  $X = \langle x_1, \ldots, x_d \rangle$ , we denote by MSE[i,j] the optimal MSE of quantizing the prefix vector  $X_j = \langle x_1, \ldots, x_j \rangle$  using i quantization values that include  $x_j$ , that is:

$$MSE[i, j] = \min_{Q:|Q| \le i, x_j \in Q} \sum_{x \in X_j} (b_x - x)(x - a_x).$$

Our goal is to compute a set of quantization values Q that results in an optimal MSE of MSE[s,d]. Accordingly, we express the dynamic program as follows. We first define C[k,j] as the sum of variances of all vector entries in the range  $[x_k,x_j]$  where  $x_k,x_j\in Q$  are two consecutive quantization values, i.e.,  $C[k,j]=\sum_{x\in [x_k,x_j]}(x_j-x)(x-x_k)$ . Here and when clear from context, to simplify notation, we write  $\sum_x$  to denote  $\sum_{x\in X}$ .

For  $i \in \{2, ..., s\}$ ,  $j \in \{i, ..., d\}$ , we set  $MSE[2, j] = C[1, j] \ \forall j$  and use the recurrence

$$MSE[i, j] = \min_{k \in \{i, \dots, j\}} MSE[i - 1, k] + C[k, j].$$

Here, the index k denotes the entry in X,  $x_k$ , of the rightmost quantization value to the left of  $x_j$ . A naive solution for the above DP is first to compute the matrix C (which takes  $O(d^3)$  time and  $O(d^2)$  space) and then calculate MSE[i,j] for all i,j, and thus Q, in  $O(s \cdot d^2)$  time and  $O(s \cdot d)$  space. In Appendix A, we describe a simple algorithm that implements this dynamic program.

An improved solution, ZipML [26], uses  $O(s \cdot d^2)$  time and  $O(d^2)$  space, but it remains infeasible even for moderate (e.g.,  $d=10^5$ ) dimensions. Accordingly, we next design novel techniques to asymptotically improve both the space and time complexities.

## <span id="page-3-0"></span>3 Optimization Using Pre-processing

The first ingredient in our solution is the usage of preprocessed arrays that allow us to efficiently compute C[k,j] in constant time, at the cost of only O(d) additional space. We define the following arrays,  $\beta, \gamma \in \mathbb{R}^d$ , that store the cumulative sums of the vector and its squared entries:

$$\beta_j = \sum_{x \in X_j} x$$
 ,  $\gamma_j = \sum_{x \in X_j} x^2$   $\forall j \in \{1, \dots, d\}$ .

Denoting  $\beta_0 = \gamma_0 = 0$ , both are computable in O(d) time as  $\beta_j = \beta_{j-1} + x_j$  and  $\gamma_j = \gamma_{j-1} + x_j^2$ . We can then express C[k,j] as follows:

$$C[k,j] = \sum_{x \in [x_k, x_j]} (x_j - x)(x - x_k) = \sum_{x \in (x_k, x_j]} (x_j - x)(x - x_k)$$

$$= -x_j \cdot x_k \cdot \sum_{x \in (x_k, x_j]} 1 + (x_j + x_k) \cdot \sum_{x \in (x_k, x_j]} x - \sum_{x \in (x_k, x_j]} x^2$$

$$= -x_j \cdot x_k \cdot (j - k) + (x_j + x_k) \cdot (\beta_j - \beta_k) - (\gamma_j - \gamma_k).$$

With this optimization, we can evaluate C[k,j] in constant time, yielding a solution that uses  $O(s \cdot d)$  memory instead of  $O(d^2)$ . Next, we show how to improve the runtime.

