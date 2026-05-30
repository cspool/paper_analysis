# 6 The Approximate QUIVER Algorithm

We now show how the usage of quantization value discretization gives a controllable tradeoff between accuracy and speed. Intuitively, by allowing the quantization values to be placed only on a uniform grid of controllable size  $m+1 \geq s$  (for some  $m \in \mathbb{N}^+$ ), we can accelerate the computation at the cost of a small additional error. Importantly, while the quantization values are from a discretized set of possibilities, we compute the *optimal* subset of discretized values for the *original input vector*.

To that end, consider the discrete set  $S=\left\{x_1+\ell\cdot\frac{x_d-x_1}{m}\mid \ell\in\{0,\dots,m\}\right\}$ . Our goal is then to find  $Q\in\binom{S}{s}$  that minimizes the sum of variances for the original input. Denoting  $s_\ell=x_1+\ell\cdot\frac{x_d-x_1}{m}$ , we modify our preprocessing scheme to consider the discretization:

$$\alpha_{\ell} = \sum_{x \in [s_0, s_{\ell}]} 1$$
 ,  $\beta_{\ell} = \sum_{x \in [s_0, s_{\ell}]} x$  ,  $\gamma_{\ell} = \sum_{x \in [s_0, s_{\ell}]} x^2$   $\forall \ell \in \{1, \dots, m\}$  .

As we explain in Appendix E, we can compute these values in O(d) time and space.

Using these arrays, we can express the sum of variances of all input entries between two quantization values  $s_k$ ,  $s_j$  as follows:

$$C_m[k,j] = \sum_{x \in [s_k, s_j]} (s_j - x)(x - s_k) = \sum_{x \in (s_k, s_j]} (s_j - x)(x - s_k)$$

$$= -s_j \cdot s_k \cdot \sum_{x \in (s_k, s_j]} 1 + (s_j + s_k) \cdot \sum_{x \in (s_k, s_j]} x - \sum_{x \in (s_k, s_j]} x^2$$

$$= -s_j \cdot s_k \cdot (\alpha_j - \alpha_k) + (s_j + s_k) \cdot (\beta_j - \beta_k) - (\gamma_j - \gamma_k).$$

Note that the quadrangle inequality trivially holds for this extension. The resulting algorithm, termed Approximate QUIVER (or in short, Apx. QUIVER), proceeds as QUIVER with  $C_m$  instead of C, except for the reconstruction stage where we pick Q from S instead of the input X. Apx. QUIVER, whose pseudo-code is given in Appendix F, runs in space and time complexities of  $O(d + m \cdot s)$ .

We next analyze the approximation guarantee of Apx. QUIVER. Denote by  $\operatorname{opt}_{X,s}$  the optimal MSE attainable for X using s quantization values, and by  $\operatorname{AQ}_{X,2s-2}$  the MSE of Apx. QUIVER with 2s-2 values. We prove that the MSE of Apx. QUIVER with 2s-2 quantization values is close to the optimal algorithm with s values. In practice, we generally find Apx. QUIVER does better than the bound below, and for moderate s, it is nearly optimal.

<span id="page-7-1"></span>
$$\text{Lemma 6.1. } \textit{For any } X, s, m \textit{ we have } \mathsf{AQ}_{X,2s-2} \leq \mathsf{opt}_{X,s} + \tfrac{d \cdot (x_d - x_1)^2}{4m^2} \leq \mathsf{opt}_{X,s} + \tfrac{d \cdot \|X\|_2^2}{2m^2}.$$

*Proof.* Let  $Q^*\subseteq X$  be the optimal solution with  $|Q^*|\le s$ . For any  $q\in Q^*$ , denote by  $\underline{q}=\max\{s_\ell\in S\mid s_\ell\le q\}$  and  $\overline{q}=\min\{s_\ell\in S\mid s_\ell\ge q\}$ . Consider the solution  $\widetilde{Q}=\left\{\underline{q},\overline{q}\mid q\in Q^*\right\}$ . Note that  $|\widetilde{Q}|\le 2s-2$  as  $x_1,x_d\in Q^*$  and  $\overline{x_1}=\underline{x_1}$  and  $\overline{x_d}=\underline{x_d}$ . Also,  $\widetilde{Q}\subseteq S$  and is thus a valid solution of Apx. QUIVER. Thus,  $\mathrm{AQ}_{X,2s-2}$  is upper bounded by the MSE when using  $\widetilde{Q}$ .

Next, consider  $x \in X$  and let  $a_x = \max\{q \in Q^* \mid q \le x\}$  and  $b_x = \min\{q \in Q^* \mid q \ge x\}$  be the values between which x is stochastically quantized in  $Q^*$ . We consider two cases:

- $x \in [\underline{a_x}, \overline{a_x}) \cup (\underline{b_x}, \overline{b_x}]$ . In this case, when using  $\widetilde{Q}$ , we have that x is quantized in an interval of size  $(x_d x_1)/m$  and thus its variance is bounded by  $(x_d x_1)^2/4m^2$ .
- $x \in [\overline{a_x}, \underline{b_x}]$ , in this case, using  $\widetilde{Q}$ , x is quantized between  $\overline{a_x}$  and  $\underline{b_x}$ , yielding a variance of  $(\underline{b_x} x)(\overline{x} \overline{a_x}) \leq (b_x x)(x a_x)$ , i.e., lower than the variance under  $Q^*$ .

As the two cases capture all options, summing the variances over all  $x \in X$  yields the result.  $\square$ 

In terms of the *vector normalized MSE* (vNMSE), which is a normalized MSE measure given by  $\frac{\mathbb{E}\left[\left\|X-\widehat{X}\right\|_{2}^{2}\right]}{\|X\|_{2}^{2}}$ , Apx. QUIVER with 2s-2 quantization values achieves an additive  $\frac{d}{2m^{2}}$  term to the optimal vNMSE when using s quantization values.

However, the first inequality of Lemma 6.1 is generally much tighter than the second that uses the squared norm. For example, if the entries of X were i.i.d. U[a,b] random variables, for some constants a < b then  $(x_d - x_1)^2 = O(1)$  while  $\|X\|_2^2 = \Theta(d)$ . Similarly, for i.i.d  $\mathcal{N}(\mu, \sigma^2)$  entries for constants  $\mu, \sigma$  we have  $(x_d - x_1)^2 = O(\log d)$  while  $\|X\|_2^2 = \Theta(d)$  (both with high probability).

## 7 Evaluation

We evaluate our algorithms' empirical vNMSE and runtime against SOTA ASQ solutions.

**Setup.** We implement all algorithms in C++. Unless stated otherwise, we use a g4dn.4xlarge AWS EC2 server with custom Intel Cascade Lake CPUs with 64 GB RAM and Ubuntu 22.04 OS and average all results over 5 seeds.

**Acceleration Speedup** Appendix G shows the speedup attainable by Accelerated QUIVER. As we show, Accelerated QUIVER is consistently faster than QUIVER, providing up to 5.4× speedup.

**Distributions.** All experiments are done with vectors whose entries are independent and identically distributed. We present results for the LogNormal distribution and defer to Appendix H results for Normal, Exponential, TruncNorm, and Weibull distributions. As mentioned, these distributions are of interest as they are reported to capture gradients, model weights and activations (see Section 1).

<span id="page-7-0"></span><sup>&</sup>lt;sup>1</sup>This metric is standard in quantization works (e.g., see [17] and the references therein). It enables us to reason about the results among different dimensions and distributions.

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

<span id="page-8-3"></span><span id="page-8-2"></span><span id="page-8-1"></span>Figure 2: Comparing exact solutions with LogNormal(0, 1) distributed input.

<span id="page-8-4"></span>![](_page_8_Figure_2.jpeg)

<span id="page-8-7"></span><span id="page-8-6"></span><span id="page-8-5"></span>Figure 3: Comparing approximate solutions with LogNormal(0, 1) distributed input.

Baselines. We evaluate Accelerated QUIVER and compare its runtime to ZipML [26]. For the approximate variants, we evaluate Apx. QUIVER and compare it with three approximation variants of ZipML proposed in [26], namely ZipML-CP Quantiles, ZipML-CP Uniform, and ZipML 2-Approximation. ZipML-CP is an algorithm that runs the exact ZipML algorithm on a subset of the points called 'Candidate Points'. Since ZipML runs in  $O(d^2s)$  time, here we use M candidate points to get  $O(d+M^2s)$  time. ZipML 2-Apx is an algorithm that computes an approximate solution in  $O(d \log d + s^3)$  time. It guarantees that its sum of variances is at most twice that of an optimal solution with |s/2| quantization values. We also compare with the recently proposed ALQ [28], which is an algorithm that finds good quantization values for a truncated normal distribution. It samples several gradients (by computing the gradient of several random batches) to fit the truncated normal parameters. To be fair to ALQ, since we evaluate a single-shot quantization scenario, we calculate the exact mean, variance, and support parameters for the input vector. This then runs for several (we used 10, as in their released code) iterations, so in total, they compute  $\approx 10s$  integrals. While theoretically requiring O(d) time, in a model where such integral calculation takes constant time, this is markedly slower than other approaches. We note that it is possible that with low-precision integral calculations, one may improve the runtime, but the error (which is already not competitive) will degrade further. We further discuss these approximation algorithms in Appendix I.

**Exact algorithms experiments.** The results are presented in Figure 2. Figure 2(a) shows the runtime for optimally solving the ASQ problem for different dimensions and s. As shown, all our solutions are markedly faster than ZipML, which we are unable to run for dimensions  $d \ge 2^{17}$  due to its prohibitively large memory requirements. The asymptotic difference  $(O(s \cdot d^2))$  for ZipML and  $O(s \cdot d)$  for Accelerated QUIVER) is clearly visible in the different slopes on the log-log plot. As

a result, Accelerated QUIVER can efficiently quantize vectors. For example, Acc. QUIVER can compute the optimal 4-bit (s=16) quantization values for a 1M-sized vector in under a second.

Next, Figure 2(b) and Figure 2(c) show the vNMSE and runtime with respect to the number of quantization values s for  $d=2^{12}$  and  $d=2^{16}$ . As shown, the vNMSE decays linearly with s while the runtime increases linearly. Even for these small dimensions, our algorithms are orders of magnitude faster than ZipML.

**Approximate algorithms experiments.** The comparison results are presented in Figure 3. It is evident in Figure 3(a) that approximate solutions are significantly faster than exact ones. Also, Apx. QUIVER offers both near-optimal vNMSE and the fastest runtime as the dimension increases. As shown in Figures 3(b) and 3(c), Apx. QUIVER offers these advantages for different s, m values.

Notably, on a commodity PC, Apx. QUIVER can compute near-optimal 4-bit quantization values (s=16) for a vector with  $d=2^{20}$  entries in just six milliseconds, and about 70ms for  $d=2^{24}$ , potentially enabling quantizing vectors on the fly for many applications.

### 8 Discussion

In this paper, we presented algorithms for the Adaptive Stochastic Quantization (ASQ) problem with improved space and time complexities compared to the state of the art. For parameters of interest, our exact algorithms are up to four orders of magnitude faster compared to the alternatives while using markedly less memory. To potentially enable on-the-fly adaptive quantization of vectors, we also introduce an approximate algorithm with strong guarantees that runs faster while being significantly more accurate than other approximate solutions.

**Limitations:** QUIVER is not GPU friendly, and it remains an interesting future work to design GPU-friendly ASQ algorithms. Also, similarly to previous works (e.g., [26]), our exact solution assumes that the input vector is sorted. Otherwise, the runtime is increased to  $O(d \cdot \log d + s \cdot d)$ . We note that Apx. QUIVER does not require the vector to be sorted and the time complexity remains  $O(d + s \cdot m)$  even for non-sorted inputs, making it even more appealing compared to the exact solutions.

Offloading Computation to a GPU: For exact algorithms, one can sort the input vector on a GPU, bringing the CPU solution complexity to  $O(s \cdot d)$  which is faster for large vectors. In practice, GPU sorting is rarely the bottleneck; indeed, in Appendix J we measure the time it takes to sort the vector on a T4 GPU, and also to quantize the vector after an ASQ outputs the optimal quantization values. For example, the sorting and quantization time for a 1M-sized vector sums up to only 4ms where the runtime of Accelerated QUIVER is about one second.

Generalizing the algorithms for weighted inputs: An interesting generalization of the ASQ problem is the weighted variant, where each entry  $x_i \in X$  is associated with a weight  $w_i \in \mathbb{R}$  and the goal is to minimize the weighted sum of variances  $\sum_{i=1}^d (x_i - \widehat{x_i})^2 \cdot w_i$ . This variant is useful when, instead of getting an input vector, one wishes to solve ASQ for an empirical distribution. In Appendix K we explain how our algorithms and their analyses generalize to the weighted case, while maintaining the  $O(d \cdot s)$  and  $O(d + M \cdot s)$  runtime and space complexities for QUIVER and Apx. QUIVER accordingly. Our measurements indicate that the weighted variants are only 10-20% slower than their unweighted counterparts.

**Reproducability:** All our results are reproducible and our code is open sourced [30].

## Acknowledgments and Disclosure of Funding

We thank Wenchen Han for his insightful comments and suggestions. Michael Mitzenmacher was supported in part by NSF grants CCF-2101140, CNS-2107078, and DMS-2023528.

