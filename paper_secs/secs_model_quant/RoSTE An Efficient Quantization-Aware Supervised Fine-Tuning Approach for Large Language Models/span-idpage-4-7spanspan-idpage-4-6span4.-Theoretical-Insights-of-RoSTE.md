# <span id="page-4-7"></span><span id="page-4-6"></span>4. Theoretical Insights of RoSTE

This section aims at providing theoretical insights on the RoSTE algorithm that tackles the bilevel problem (11). In particular, we show that the quantization error (12) is a

suitable surrogate loss for optimizing the rotation matrices, provided that the weight matrices are optimized using the STE method as in Algorithm 1. We remark that the SFT objective on quantized LLMs is complicated and possibly untractable for analysis. To concentrate on the insights pertaining to using rotation in the quantized LLMs, we shall introduce a few approximations. We will use  $\langle \cdot \mid \cdot \rangle$  to denote inner products of vectors, and  $\|\mathbf{x}\|_{\mathbf{K}}^2 = \langle \mathbf{x} \mid \mathbf{K}\mathbf{x} \rangle$  to denote a K-weighted squared norm of vector  $\mathbf{x}$  for any square matrix  $\mathbf{K}$ .

Our setup follows from the literature on analyzing the convergence of SGD for neural networks under the interpolation regime (Ma et al., 2018; Vaswani et al., 2019). To describe it, let us fix the rotation matrices  $\{\mathbf{R}_i\}_{i=0}^{\ell-1}$  and consider the QAT stage (cf. line 7–9) in the RoSTE algorithm. Instead of analyzing  $\mathcal{L}_{\mathrm{SFT}}(\mathbf{m}_Q(\cdot))$  directly, we consider the quadratic loss function as a simplified objective to draw insights for RoSTE. Moreover, the training dataset consists of samples  $(\mathbf{x}_\xi, \mathbf{y}_\xi)$  with a target output token in  $\mathbb R$  such that  $\mathbf{y}_\xi \in \mathcal{Y} \equiv \mathbb R$ .

For any  $\mathbf{m}:\overline{\mathcal{X}}\to\mathbb{R}^{|\mathcal{T}|}$ , we now consider the squared prediction error:

$$\widehat{\mathcal{L}}(\mathbf{m}(\cdot)) := \frac{1}{2} \mathbb{E}_{\xi} \left[ (o(\mathbf{m}(\mathbf{x}_{\xi})) - \mathbf{y}_{\xi})^{2} \right], \tag{13}$$

in lieu of  $\mathcal{L}_{SFT}(\cdot)$ , where  $o : \mathbb{R}^{|\mathcal{T}|} \to \mathcal{Y}$  maps the probability distribution over  $\mathcal{T}$  to a token.

We further assume that the composite map  $o(\mathbf{m}_Q(\cdot))$  is a linear activation-weight quantized model given by

<span id="page-5-2"></span>
$$o(\mathbf{m}_Q(\mathbf{x}; \mathbf{w}, \mathbf{R})) = \langle Q_x(\mathbf{R}\mathbf{x}) \mid Q_w(\mathbf{R}\mathbf{w}) \rangle,$$
 (14)

where **R** is a rotation matrix satisfying  $\mathbf{R}\mathbf{R}^{\top} = \mathbf{I}$  and  $Q_x, Q_w : \mathbb{R}^d \to \mathbb{R}^d$  are the quantization functions [see Sec. 2.1]. Let  $\mathbf{x}_t, \mathbf{y}_t$  be the sample drawn at iteration t in the inner loop update of line 8, Algorithm 1, we have

$$\mathbf{w}^{t+1} = \mathbf{w}^t - \eta \, \mathbf{g}_{\text{s.t.e}}^t \tag{15}$$

$$\boldsymbol{g}_{\text{s.t.e.}}^t = (\left\langle Q_x(\mathbf{R}\mathbf{x}_t) \mid Q_w(\mathbf{R}\mathbf{w}^t) \right\rangle - \mathbf{y}_t) \mathbf{R}^\top Q_x(\mathbf{R}\mathbf{x}_t),$$

where  $\eta > 0$  is the step size and we have used the STE approximation  $\partial (Q_w(\mathbf{R}\mathbf{w}))/\partial(\mathbf{w}) \approx \mathbf{R}$  when computing the stochastic gradient  $g_{\mathrm{s.t.e.}}^t$  at  $\mathbf{w}^t$ .

Our next endeavor is to study an upper bound on the loss value of quantized model,  $\widehat{\mathcal{L}}(\mathbf{m}_Q(\,\cdot\,;\mathbf{w}^T,\mathbf{R}))$ , after running the recursion (15) for  $T\geq 1$  steps. Define the Gram matrix of the quantized-rotated features by

$$\mathbf{G} := \mathbb{E}\left[Q_x(\mathbf{R}\mathbf{x}_{\xi})Q_x(\mathbf{R}\mathbf{x}_{\xi})^{\top}\right]$$
 (16)

and make the following assumptions accordingly:

<span id="page-5-3"></span>**Assumption 4.1** (Gram Matrix). There exists constants  $\lambda_{\min}$ ,  $\rho > 0$  such that

<span id="page-5-6"></span>
$$\mathbf{G}^2 \succeq \lambda_{\min} \mathbf{G}, \quad \sup_{0 \le t \le T-1} \|Q_x(\mathbf{R} \mathbf{x}_t)\|_{\mathbf{G}}^2 \le \rho. \quad (17)$$

The above conditions are mild as  $\lambda_{\min}$  is only the smallest non-zero eigenvalue of the Gram matrix G and  $\rho$  exists when the input prompts  $\mathbf{x}_t$  are bounded.

<span id="page-5-1"></span>**Assumption 4.2** (Interpolation). For any orthogonal matrix  $\mathbf{R}$ , there exists  $\mathbf{w}_{\mathbf{R}}^{\star} \in \mathbb{R}^d$  such that  $\mathbf{y}_{\xi} = \langle Q_x(\mathbf{R}\mathbf{x}_{\xi}) \mid \mathbf{w}_{\mathbf{R}}^{\star} \rangle$  for any  $\xi$ .

The above assumption requires that the quantized-rotated features  $(Q_x(\mathbf{R}\mathbf{x}_\xi),\mathbf{y}_\xi)$  are interpolatable by a full-precision model  $\mathbf{w}_\mathbf{R}^*$ . This assumption is closely related to the standard interpolation assumption that appeared in the literature on training over-parameterized models (Ma et al., 2018; Vaswani et al., 2019). It is worth noticing that Assumption 4.2 does not require the interpolator  $\mathbf{w}_\mathbf{R}^*$  to be in the quantized model parameter space (14).

Define the shorthand notation  $\mathbf{m}_{Q,\mathbf{R}}^t := \mathbf{m}_Q(\cdot; \mathbf{w}^t, \mathbf{R})$ , we observe the following convergence results for the QAT stage during the RoSTE algorithm:

<span id="page-5-4"></span>**Theorem 4.3.** Under Assumptions 4.1, 4.2 and the step size  $\eta = \lambda_{\min}/(6\rho)$ , the objective value of the quantized model produced by the recursion (15) is bounded by

<span id="page-5-5"></span>
$$\mathbb{E}[\widehat{\mathcal{L}}(\mathbf{m}_{Q,\mathbf{R}}^{t+1})] \le (1-\mu)^{t+1} \widehat{\mathcal{L}}(\mathbf{m}_{Q,\mathbf{R}}^{0})$$

$$+ (6+2\mu^{-1}) \sum_{s=0}^{t+1} (1-\mu)^{t-s} \mathbb{E}\left[\|\mathbf{e}(\mathbf{R}\mathbf{w}^{s})\|_{\mathbf{G}}^{2}\right]$$
(18)

for any 
$$t \geq 0$$
, where  $\mu = \frac{\lambda_{\min}^2}{12\rho}$  and  $\mathbf{e}(\mathbf{x}) := Q_w(\mathbf{x}) - \mathbf{x}$ .

See Appendix A for the proof. Our result shows that STE only converges to an inexact solution, which is consistent with previous findings on STE training. For instance, when training models with activation-only quantization, (Yin et al., 2019, Lemma 10) proved that the STE gradient is non-vanishing near local minima. For models with weight-only quantization, (Li et al., 2017, Corollary 1) only showed a convergence guarantee for the full-precision weights but not the quantized weights. In comparison to the prior findings, our result demonstrates the convergence of prediction error with quantized model.

<span id="page-5-0"></span>More specifically, suppose the QAT stage of RoSTE is run with  $T\gg 1$  inner-loop iterations. Applying the theorem shows that given  $\overline{\mathbf{R}}$ , the resultant prediction error of model  $\mathbf{w}^T$  will be bounded by  $\mathcal{O}(\sum_{s=0}^T (1-\mu)^{T-s} \mathbb{E}\left[\|Q_w(\overline{\mathbf{R}}\mathbf{w}^s) - \overline{\mathbf{R}}\mathbf{w}^s\|_{\mathbf{G}}^2\right])$ , i.e., a weighted sum of the weight quantization errors during the QAT process. Due to the exponential weighting  $(1-\mu)^{T-s}$ , the prediction error is dominated by the weight quantization error of recent iterates. Crucially, the above analysis shows that the rotation matrices play a pivoting role in the performance of QAT. This inspires us to apply  $\mathcal{E}(\cdot)$  in (12) to guide us in the selection for optimal rotation matrices, covering the weight quantization error of the rotated weight matrices  $\|\mathbf{e}(\mathbf{R}_i^{\top}\mathbf{W}_i)\|^2$ .

Randomized Rotation Matrices. Now as we demonstrated that the quantization error is crucial to the prediction performance with the quantized model, we turn our focus to tackling the lower-level subproblem in (11). Notice that minimizing  $\mathcal{E}(\cdot)$  w.r.t. the rotation matrix remains challenging. Instead of directly tackling the manifold optimization, our strategy is to apply the random Walsh-Hadamard matrix (Tseng et al., 2024) design as an approximate-yet-universal solution. Consider the random rotation matrix:

<span id="page-6-0"></span>
$$\mathbf{R}(\zeta) = \mathbf{H}\mathrm{Diag}(\mathbf{r}(\zeta)) \tag{19}$$

where  $\mathbf{H} \in \mathbb{R}^{d \times d}$  is a Walsh-Hadamard matrix (Fino & Algazi, 1976) and  $\mathbf{r}(\zeta) \in \{-1,1\}^d$  is a random sign vector. Notice that  $\mathbf{R}(\zeta)$  is a binary matrix which favors efficient implementation on GPUs.

We observe the following proposition adapted from (Tseng et al., 2024, Lemma 3.1):

<span id="page-6-3"></span>**Proposition 4.4.** Consider a  $b_w$ -bits symmetric quantizer  $Q_w : \mathbb{R}^d \to \mathbb{R}^d$  [cf. (2), (3) with c = 1]. For any  $\mathbf{w} \in \mathbb{R}^d$ ,

• with  $\mathbf{R} = \mathbf{I}$ , it holds that

<span id="page-6-4"></span>
$$||Q_w(\mathbf{w}) - \mathbf{w}||^2 \le \frac{d \max_i \mathbf{w}_i^2}{4(2^{b_w - 1} - 1)^2}.$$
 (20)

• with  $\mathbf{R} = \mathbf{R}(\zeta)$  from (19), with probability  $1 - \delta$  we have

<span id="page-6-2"></span>
$$\|Q_w(\mathbf{R}(\zeta)\mathbf{w}) - \mathbf{R}(\zeta)\mathbf{w}\|^2 \le \frac{\log(4d/\delta)}{2(2^{b_w-1}-1)^2} \|\mathbf{w}\|^2.$$
 (21)

See Appendix B for the proof.

Observe that the quantization error is  $\mathcal{O}(d \max_i \mathbf{w}_i^2)$  without rotation, and is  $\mathcal{O}(\|\mathbf{w}\|^2)$  with rotation. Note that the former bound is more sensitive to weight vectors with outliers. In particular, the worst case prediction error in the QAT stage with  $\mathbf{R}$  chosen as (19) is strictly better than that for the case with  $\mathbf{R} = \mathbf{I}$  (no rotation) if

<span id="page-6-1"></span>
$$\frac{\log(4d/\delta)}{2} \|\overline{\mathbf{w}}_{\mathbf{R}}\|^2 \le \frac{\max_i(\overline{\mathbf{w}}_{\mathbf{I}i})^2 d}{4}, \tag{22}$$

where  $\overline{\mathbf{w}}_{\mathbf{R}}$ ,  $\overline{\mathbf{w}}_{\mathbf{I}}$  are the respective converged solutions of (15). It demonstrates that applying the random rotation matrix in (19) suffices to reduce the quantization error of weight matrices that contain outlier values. To obtain the best performance, we design the RoSTE algorithm such that at the outer loop, it chooses between  $\mathbf{H}$  or  $\mathbf{I}$  (i.e., no rotation) according to the current weight matrices.

Remark 4.5. The analysis in Theorem 4.3 and (22) enables a novel interpretation of the bit-widths in  $Q_x$  and  $Q_w$  during STE training. On one hand, it is beneficial to increase the bit-width of activation quantization  $Q_x$  until Assumption 4.2 is satisfied, and further increasing its bit-width would

not improve the prediction performance as the bound (18) only depends on weight quantization error. On the other hand, increasing the bit-width of weight quantization always reduces the prediction error as seen in (18), (21). It is also interesting to see that despite adopting low-bit activation quantizers, increasing the dimension d may still allow us to satisfy the interpolation condition Assumption 4.2, under the intuition that kernelized high dimensional features are more likely to be separable (Liang & Rakhlin, 2020). In other words, a neural network with high-dimensional hidden representations can tolerate low-bit quantized activations because the information about  $\mathbf{x}_{\xi}$  retains in the high-dimensional discrete vector  $Q_x(\mathbf{R}\mathbf{x}_{\xi})$ .

### 5. Experiments

We evaluate the performance of the proposed RoSTE algorithm for QA-SFT on two standard sets of open-source models and datasets. For the first experiment (Exp. 1), we fine-tune the pre-trained Pythia 1B/6.9B models (Biderman et al., 2023) and Qwen2.5 0.5B/7B models (Yang et al., 2024) on the Reddit TL;DR Summarization dataset (Huang et al., 2024) with evaluation on the TL;DR test dataset using the ROUGE metric (Lin, 2004). For the second experiment (Exp. 2), we fine-tune the pre-trained Llama 3.1 8B model (Dubey et al., 2024) on the Tulu 3 SFT mixture dataset (Lambert et al., 2024) with real-world downstream task evaluations (Gao et al., 2021). These tasks include TruthfulQA (Lin et al., 2021), MMLU-Pro (Wang et al., 2024b), Big-BenchHard (Suzgun et al., 2022), AGIEval (Zhong et al., 2023), GSM8K (Cobbe et al., 2021), and MATH (Hendrycks et al., 2020).

For the RoSTE algorithm, while we relaxed the lower level as a  $\ell$ -variable binary combinatorial problem (9), solving this sub-problem has a complexity of  $\mathcal{O}(2^{\ell})$  which is still intractable for models like Llama 3.1 8B with  $\ell=3\times32+1$ . As a remedy, we estimate the solution of (9) by computing only  $\mathcal{E}(\mathcal{W}^{kT}, \{\mathbf{I}\}_{i=0}^{\ell-1})$  and  $\mathcal{E}(\mathcal{W}^{kT}, \{\mathbf{H}\}_{i=0}^{\ell-1})$ , then we determine each layer's  $\mathbf{R}_i$  by comparing the quantization error layer-wise. Lastly, we set K=1 where a one-shot rotation configuration adaptation by pre-trained model is found to perform well. We anticipate the performance to further improve with larger K on larger datasets. More implementation details can be found in Appendices C, D.

**Baselines.** Besides the proposed RoSTE algorithm, we compare the performances of LLMs with quantized weight and activation obtained by two streams of baseline approaches. The first stream consists of applying PTQ methods on open-source supervised fine-tuned models in (Huang et al., 2024; Lambert et al., 2024). We reproduce the PTQ benchmarks using round-to-nearest (RTN) quantization, GPTQ (Frantar et al., 2022), QuaRot (Ashkboos et al., 2024b) and Spin-Quant (Liu et al., 2024). The second set consists of QAT

<span id="page-7-0"></span>Table 1. Results on **Exp.1**. Accuracies of the 4-bit quantized Pythia 6.9B and Qwen2.5 7B models fine-tuned using the Reddit TL;DR dataset. FP16 and BF16 refer to using 16-bit half-precision floating points and 16-bit brain floating points formats, respectively, and W4A4KV4 refers to using 4-bit quantizations on weights, activation, and KV cache.

| Bit-width | Method      | ROUGE-1 | ROUGE-2 | ROUGE-L    | ROUGE-LSum | ROUGE (Avg.) |  |
|-----------|-------------|---------|---------|------------|------------|--------------|--|
|           | Pythia-6.9B |         |         |            |            |              |  |
|           | Base        | 28.81   | 9.45    | 22.29      | 22.91      | 20.87        |  |
| FP16      | SFT         | 33.69   | 12.60   | 26.27      | 26.31      | 24.72        |  |
|           | RTN         | 7.42    | 0.06    | 6.53       | 6.56       | 5.14         |  |
|           | GPTQ        | 8.16    | 0.08    | 7.06       | 7.60       | 5.73         |  |
|           | QuaRot      | 11.70   | 0.23    | 8.52       | 9.39       | 7.46         |  |
| W4A4KV4   | SpinQuant   | 8.61    | 0.10    | 8.10       | 8.07       | 6.22         |  |
|           | STE         | 28.91   | 9.07    | 22.30      | 22.33      | 20.65        |  |
|           | RoSTE       | 32.60   | 11.54   | 25.25      | 25.25      | 23.66        |  |
|           |             |         |         | Qwen2.5-7B |            |              |  |
|           | Base        | 32.72   | 11.82   | 25.18      | 25.42      | 23.79        |  |
| BF16      | SFT         | 34.75   | 13.59   | 27.56      | 27.58      | 25.87        |  |
|           | RTN         | 1.07    | 0.00    | 1.01       | 1.01       | 0.77         |  |
|           | GPTQ        | 0.72    | 0.00    | 0.69       | 0.69       | 0.53         |  |
|           | QuaRot      | 7.21    | 0.10    | 5.93       | 5.93       | 4.79         |  |
| W4A4KV4   | SpinQuant   | 6.87    | 0.29    | 5.97       | 6.12       | 4.81         |  |
|           | STE         | 30.86   | 10.16   | 23.73      | 23.73      | 22.12        |  |
|           | RoSTE       | 34.01   | 12.89   | 26.74      | 26.74      | 25.10        |  |

methods applied on the SFT objective, including STE and RoSTE. The hyperparameters for reproducing our experiment results can be found in the Appendix at Table [4](#page-13-1) and [5.](#page-13-2)

All experiments are conducted on a cluster of 8 NVIDIA A100 GPUs. Details of the training and evaluation settings can be found in Appendix [C.](#page-13-0) Statistics of the training cost (time, memory) can be found in Appendix [G.](#page-19-0)

