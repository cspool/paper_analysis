# <span id="page-14-0"></span>A Proofs and Definitions

#### <span id="page-14-1"></span>A.1 Proof of Theorem [1](#page-3-4)

Theorem 1. *Consider a sequence of representations* h1, h2, . . . , h<sup>T</sup> *during an LLM's reasoning process, where* T *denotes the number of total reasoning steps. Let* y*,* yˆ *denote the golden answer and the LLM's prediction answer, respectively. Define* p<sup>e</sup> " Prpyˆ ‰ yq *as the LLM's prediction error probability. Then the following inequality holds:*

$$p_e \ge \frac{1}{\log(|\mathcal{Y}| - 1)} \Big[ H(y) - \sum_{j=1}^T I(y; \mathbf{h}_j \mid \mathbf{h}_{< j}) - H_b(p_e) \Big],$$
 (1)

*where* |Y| *is the size of the support of* y*, and* Hbppeq *denote the binary entropy of* p<sup>e</sup> *that defined by*

$$H_b(p_e) = -p_e \log p_e - (1 - p_e) \log(1 - p_e). \tag{2}$$

*Proof.* We first define an indicator random variable E " 1tyˆ ‰ yu, where E " 1 if yˆ ‰ y, and E " 0 otherwise.

By the chain rule of entropy, we have:

$$H(y \mid \hat{y}) = H(E \mid \hat{y}) + H(y \mid \hat{y}, E)$$
  
=  $H(E \mid \hat{y}) + H(y \mid \hat{y}, E = 0) \Pr(E = 0) + H(y \mid \hat{y}, E = 1) \Pr(E = 1).$  (3)

Since E " 0 indicates yˆ " y, we have Hpy | y, E ˆ " 0q " 0. And for HpE | yˆq, we have:

$$H(E \mid \hat{y}) \leqslant H(E) := H_b(p_e). \tag{4}$$

Thus, we can derive:

$$H(y \mid \hat{y}) \le H_b(p_e) + p_e H(y \mid \hat{y}, E = 1).$$
 (5)

Since E " 1 indicates yˆ ‰ y, the random variable y can take at most |Y| ´ 1 values given yˆ as condition. Hence, we have [\[12\]](#page-9-3):

<span id="page-14-2"></span>
$$H(y \mid \hat{y}) \leqslant H_b(p_e) + p_e \log(|\mathcal{Y}| - 1). \tag{6}$$

Based on the definition of mutual information, we have:

<span id="page-14-3"></span>
$$I(y; \hat{y}) = H(y) - H(y \mid \hat{y}).$$
 (7)

Combining Eq. [\(6\)](#page-14-2) and Eq. [\(7\)](#page-14-3) derives:

<span id="page-14-4"></span>
$$p_e \geqslant \frac{1}{\log(|\mathcal{Y}| - 1)} \Big[ H(y) - I(y; \hat{y}) - H_b(p_e) \Big]. \tag{8}$$

Consider an LLM's reasoning process, given the intermediate representations h1:<sup>T</sup> " ph1, h2, . . . , h<sup>T</sup> q, the output yˆ is computed as a function of these representations yˆ " fph1:<sup>T</sup> q. Thus, based on the Data Processing Inequality (DPI), we have:

<span id="page-14-5"></span>
$$I(y; \hat{y}) \leqslant I(y; \boldsymbol{h}_{1:T}). \tag{9}$$

Combining Eq. [\(8\)](#page-14-4) and Eq. [\(9\)](#page-14-5), and applying the chain rule of mutual information, we have:

$$p_e \geqslant \frac{1}{\log(|\mathcal{Y}| - 1)} \Big[ H(y) - \sum_{j=1}^{T} I(y; \, \boldsymbol{h}_j \mid \boldsymbol{h}_{< j}) - H_b(p_e) \Big],$$
 (10)

which completes the proof.

#### <span id="page-15-0"></span>A.2 Proof of Theorem [2](#page-4-1)

Theorem 2. *Following the notations in Theorem [1,](#page-3-4) the following inequality holds:*

$$p_e \leqslant \frac{1}{2} \Big[ H(y) - \sum_{j=1}^{T} I(y; \mathbf{h}_j \mid \mathbf{h}_{< j}) \Big].$$
 (11)

*Proof.* The output of a reasoning model yˆ can be formulated as a multi-class classification task with predicted probabilities p<sup>i</sup> " Prpyˆ " i | h1:<sup>T</sup> q. According to Bayesian decision theory[\[4\]](#page-9-13) [\[58\]](#page-12-7), the conditional error probability is given by:

<span id="page-15-6"></span>
$$p_e = 1 - \max_{i} \{ \Pr(y = i \mid \mathbf{h}_{1:T}) \}.$$
 (12)

For binary classification (|Y| " 2), we have:

<span id="page-15-2"></span>
$$min\{p, 1-p\} \le \frac{1}{2} [-p\log p - (1-p)\log (1-p)].$$
 (13)

Then take an expectation over p:

$$p_e = \mathbb{E}_p[\min\{p, 1-p\}] \leqslant \frac{1}{2} \mathbb{E}_p[-p\log p - (1-p)\log(1-p)].$$
 (14)

So we derive:

$$p_e \leqslant \frac{1}{2} \mathbb{E}_{h_{1:T}} [H(y \mid \mathbf{h}_{1:T})] = \frac{1}{2} H(y \mid \mathbf{h}_{1:T}).$$
 (15)

This extends to multiclass problems through a recursive application (see Eq. [\(16\)](#page-15-1)).

We prove the following inequality by mathematical induction that for any m-class discrete probability distribution tp1, . . . , pmu:

<span id="page-15-1"></span>
$$p_e = 1 - \max_i \{p_i\} \leqslant \frac{1}{2} H(p_1, \dots, p_m).$$
 (16)

*Base case* (m " 2): Direct verification using binary entropy function Eq. [\(13\)](#page-15-2).

*Inductive step*: Assume validity for m classes. For m ` 1 classes, assume without loss of generality pm`<sup>1</sup> " maxitpiu. Consider the merged distribution tp1, . . . , pm´1, p<sup>m</sup> ` pm`1u and apply:

1. The induction hypothesis:

<span id="page-15-3"></span>
$$1 - (p_m + p_{m+1}) \le \frac{1}{2} H(p_1, \dots, p_{m-1}, p_m + p_{m+1}).$$
(17)

2. The grouping axiom [\[3\]](#page-9-1):

<span id="page-15-4"></span>
$$H(p_1, \dots, p_{m+1}) = H(p_1, \dots, p_m + p_{m+1}) + (p_m + p_{m+1})H\left(\frac{p_m}{p_m + p_{m+1}}, \frac{p_{m+1}}{p_m + p_{m+1}}\right). (18)$$

3. Binary entropy bound for the final term:

<span id="page-15-5"></span>
$$1 - \frac{p_{m+1}}{p_m + p_{m+1}} \le \frac{1}{2} H\left(\frac{p_m}{p_m + p_{m+1}}, \frac{p_{m+1}}{p_m + p_{m+1}}\right). \tag{19}$$

Combining Eq. [\(17\)](#page-15-3), Eq. [\(18\)](#page-15-4) and Eq. [\(19\)](#page-15-5) completes the induction:

$$\frac{1}{2}H(p_1,\ldots,p_{m+1}) = \frac{1}{2}H(p_1,\ldots,p_m+p_{m+1}) + \frac{1}{2}(p_m+p_{m+1})H\left(\frac{p_m}{p_m+p_{m+1}},\frac{p_{m+1}}{p_m+p_{m+1}}\right)$$

$$\geqslant 1 - (p_m+p_{m+1}) + (p_m+p_{m+1})(1 - \frac{p_{m+1}}{p_m+p_{m+1}})$$

$$= 1 - p_{m+1}$$

$$= 1 - \max_{i} \{p_i\}.$$

Thus, we have proved the Eq. [\(16\)](#page-15-1).

Taking expectation over  $h_{1:T}$  in Eq. (12) and applying the Eq. (16), we have

$$p_{e} = \mathbb{E}_{h_{1:T}} [1 - \max_{i} \{ \Pr(y = i | h_{1:T}) \} ].$$

$$\leq \frac{1}{2} \mathbb{E}_{h_{1:T}} [H(y | h_{1:T})]$$

$$= \frac{1}{2} H(y | h_{1:T})$$

$$= \frac{1}{2} \left[ H(y) - \sum_{i=1}^{T} I(y; h_{j} | h_{< j}) \right],$$

which completes the proof.

#### <span id="page-16-2"></span>A.3 Definitions

**Definition 3** (Mutual Information [3, 24]). Given two continuous random variables X and Y, the mutual information is defined as:

$$I(X;Y) = \int_{Y} \int_{X} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} dx dy,$$
(20)

where p(x,y) denotes the joint probability density function of X and Y; p(x), p(y) denotes the marginal probability density functions of X and Y, respectively.

<span id="page-16-0"></span>**Definition 4** (Hilbert-Schmidt Independence Criterion (HSIC) [17]). HSIC is the Hilbert-Schmidt norm of the cross-covariance operator between the distributions in Reproducing Kernel Hilbert Space (RKHS). Formally:

$$\operatorname{HSIC}(X,Y) = \mathbb{E}_{XYX'Y'} \left[ k_X \left( X, X' \right) k_Y \left( Y, Y' \right) \right] + \mathbb{E}_{XX'} \left[ k_X \left( X, X' \right) \right] \mathbb{E}_{YY'} \left[ k_Y \left( Y, Y' \right) \right]$$

$$-2 \mathbb{E}_{XY} \left[ \mathbb{E}_{X'} \left[ k_X \left( X, X' \right) \right] \mathbb{E}_{Y'} \left[ k_Y \left( Y, Y' \right) \right] \right],$$
(21)

where X', Y' are independent copies of X, Y, respectively, and  $k_X$ ,  $k_Y$  are kernel functions.

#### <span id="page-16-1"></span>**B** Experimental Implementation Details

**Practical implementation of HSIC.** Due to the difficulty of accurately computing MI in high-dimensional spaces [24, 32, 12], we employ the HSIC to estimate MI. Following [29, 35, 12], the empirical HSIC from Definition 4 is computed as

$$HSIC(X,Y) = \frac{1}{(n-1)^2} \operatorname{tr}(K_X H K_Y H), \tag{22}$$

where  $K_X$  and  $K_Y$  are kernel matrices with entries

$$K_{X_{ij}} = k_X(x_i, x_j), \quad K_{Y_{ij}} = k_Y(y_i, y_j),$$

and  $H = I - \frac{1}{n} \mathbf{1} \mathbf{1}^{\top}$  is the centering matrix. Consistent with [29, 35, 12], we adopt the Gaussian kernel to implement the kernel:

$$k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}\right),\tag{23}$$

where the bandwidth  $\sigma$  is selected by grid search over the range [50, 400].

**Datasets.** 1) Evaluation of LRMs' reasoning performance. We select three widely-used math reasoning benchmarks to evaluate the reasoning capabilities of LRMs, ordering from easy to hard: GSM8K [9], MATH500 [25], and AIME24 [1]. We adopt the evaluation framework provided by Qwen2.5-Math [53]. To ensure the reproducibility of our results, we fix the temperature to 0 in all experiments. 2) Observing the MI trajectories during LRMs' reasoning process. We use the training set of the MATH dataset [19]. Specifically, we randomly sample 100 instances to compute MI along the reasoning trajectories.

Models. We conduct experiments on DeepSeek's R1 series models [\[18\]](#page-10-3) and QwQ-32B [\[42\]](#page-11-4). For DeepSeek's R1 series models, we pair each LRM with its corresponding non-reasoning LLM counterpart as follows: DeepSeek-R1-Distill-Qwen-7B and Qwen2.5-Math-7B [\[53\]](#page-12-8), DeepSeek-R1- Distill-Llama-8B and Llama-3.1-8B [\[16\]](#page-9-2), DeepSeek-R1-Distill-Qwen-14B and Qwen2.5-14B [\[52\]](#page-12-9), DeepSeek-R1-Distill-Qwen-32B and Qwen2.5-32B [\[52\]](#page-12-9), DeepSeek-R1-Distill-Llama-70B and Llama-3.3-70B-Instruct [\[16\]](#page-9-2). As observed, all LRMs in the R1 series are trained from foundation LLMs, except for DeepSeek-R1-Distill-Qwen-7B, which is trained from a math-specialized LLM. As for QwQ-32B, existing public report [\[42\]](#page-11-4) has not disclosed which specific LLM it was trained from. All experiments are conducted on four NVIDIA A100 GPUs.

More implementation details. For all experiments involving MI computation, we extract the representation from the *last layer* of the model. We concentrate on the *last layer* since higher layers have been shown to encode more semantic content [\[59,](#page-12-4) [37\]](#page-11-16) and the *last layer* directly influence the model's output text [\[34\]](#page-11-17). For TTTS in Section 4.2, to ensure that the model begins continuation with semantically meaningful tokens, we filter out tokens with little semantic information, such as punctuation, single characters, etc. In this way, the resulting token list is: [So, Let, Hmm, I, Okay, First, Wait, But, Now, Then, Since, Therefore, If, Maybe, To]. All experiments are conducted on four NVIDIA A100 GPUs.

