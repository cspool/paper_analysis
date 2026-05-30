# **A. Derivations and Theoretical Analysis**

This section consolidates the mathematical foundations underpinning ResAdapt. We first derive the joint RL formulation and then formalize the computational bounds.

### <span id="page-24-0"></span>**A.1. Derivation of Joint RL Formulation**

This section details derivations omitted from Sec. [3](#page-3-2) and clarifies how the one-step contextual MDP (Contextual Bandit) introduced in Sec. [2.2](#page-3-3) motivates our practical surrogate objectives. We state all derivations for a single context (video and query). The full objective requires taking the expectation over the dataset D.

*Notation.* Let *x* = (*q*, V) denote the prompt context. The Allocator samples latent actions *a* from a Beta policy *q<sup>θ</sup>* (*a* | *x*) (Sec. [3.2\)](#page-5-2). The continuous allocation *s* is the deterministic mapping of *a* via Eq. [\(10\)](#page-5-0). Let *π<sup>θ</sup>* (*s* | *x*) denote the induced density (pushforward). A deterministic transformation yields the operator-transformed input  $\tilde{x} = (q, \{\mathcal{O}(f_t, s_t)\}_{t=1}^T)$ . In our experiments,  $\mathcal{O}$  implements bilinear resizing. The MLLM backbone policy  $\pi_{\phi}(y \mid \tilde{x})$  then samples a complete response rollout y = (r, o), comprising the reasoning trace r and the final answer o.

One-Step Contextual MDP and the Joint Objective. Sec. 2.2 defines the system as a one-step contextual MDP. Here, sequential state transitions across time steps t do not exist. The episode terminates immediately after the system samples the allocation s and generates the corresponding rollout y. Consequently, value functions collapse to immediate rewards. The Policy Gradient Theorem simplifies drastically, eliminating the need for temporal discount factors or complex credit assignment across Markov states.

The joint distribution of the allocation and the rollout factorizes conditionally:

$$p_{\theta,\phi}(s,y\mid x) = \pi_{\theta}(s\mid x)\,\pi_{\phi}(y\mid \tilde{x}). \tag{26}$$

For a single context with ground-truth answer  $o^*$ , the marginal answer probability under the transformed input is:

<span id="page-25-0"></span>
$$p_{\theta,\phi}(\boldsymbol{o}^{\star} \mid \boldsymbol{x}) = \mathbb{E}_{\pi_{\theta}(\boldsymbol{s}|\boldsymbol{x})} \Big[ \mathbb{E}_{\pi_{\phi}(\boldsymbol{r}|\tilde{\boldsymbol{x}})} \Big[ \pi_{\phi}(\boldsymbol{o}^{\star} \mid \tilde{\boldsymbol{x}}, \boldsymbol{r}) \Big] \Big].$$
 (27)

Equation (27) expresses the law of total expectation under an autoregressive factorization  $\pi_{\phi}(y \mid \tilde{x}) = \pi_{\phi}(r \mid \tilde{x}) \, \pi_{\phi}(o \mid \tilde{x}, r)$ . The inner term represents the conditional probability of the ground-truth answer  $o^*$  given the reasoning prefix r. Integrating over r yields the marginal  $\mathbb{P}(o^* \mid x)$  strictly under this generative ordering. The subsequent RL objective does not require a closed-form evaluation of Eq. (27).

Since  $\log(\cdot)$  is monotonically increasing, maximizing  $\log p_{\theta,\phi}(o^* \mid x)$  serves as an equivalent objective. However, the RL derivation below avoids introducing the logarithm directly. It merely requires evaluating a scalar utility after sampling (s,y). We abstract the answer-quality term as a rollout utility Q(x,y), where y=(r,o), and treat it as parameter-independent post-sampling. This represents a modeling abstraction rather than an exact reformulation. When we define Q as an answer-aligned task score, the resulting RL problem acts as a surrogate for likelihood maximization. This formulation allows us to define the ideal rollout reward:

$$R_{s,y}^{\text{ideal}} = Q(x,y) - \lambda C(s), \tag{28}$$

and optimize the one-step expected return:

$$\max_{\theta, \phi} \ \mathcal{J}(\theta, \phi) = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{\pi_{\theta}(s|x)} \Big[ \mathbb{E}_{\pi_{\phi}(y|\tilde{x})} \Big[ R_{s,y}^{\text{ideal}} \Big] \Big]. \tag{29}$$

**Policy Gradient and Alternating Optimization.** Because the objective involves two distinct parameterized policies, its gradients follow the score-function estimator (the REINFORCE identity). This establishes the underlying policy-gradient structure. GRPO/PPO retains this structure but replaces the raw reward with normalized advantages and clipped surrogates to stabilize optimization. Taking the gradient of  $\mathcal{J}(\theta,\phi)$  with respect to the backbone parameters  $\phi$ :

$$\nabla_{\phi} \mathcal{J}(\theta, \phi) = \mathbb{E}_{x} \mathbb{E}_{\pi_{\theta}(s|x)} \left[ \nabla_{\phi} \int \pi_{\phi}(y \mid \tilde{x}) R_{s,y}^{\text{ideal}} dy \right]$$

$$= \mathbb{E}_{x} \mathbb{E}_{\pi_{\theta}(s|x)} \mathbb{E}_{\pi_{\phi}(y|\tilde{x})} \left[ R_{s,y}^{\text{ideal}} \nabla_{\phi} \log \pi_{\phi}(y \mid \tilde{x}) \right]. \tag{30}$$

Similarly, the gradient with respect to the Allocator parameters  $\theta$  relies on the marginalized reward  $R_s^{\text{ideal}} = \mathbb{E}_{\pi_{\theta}(y|\tilde{x})}[R_{s,y}^{\text{ideal}}]$ :

$$\nabla_{\theta} \mathcal{J}(\theta, \phi) = \mathbb{E}_{x} \mathbb{E}_{\pi_{\theta}(s|x)} \left[ R_{s}^{\text{ideal}} \nabla_{\theta} \log \pi_{\theta}(s \mid x) \right]. \tag{31}$$

To optimize this objective via GRPO/PPO, we introduce importance sampling from behavior policies  $\pi_{\theta_{\text{old}}}$  and  $\pi_{\phi_{\text{old}}}$ . A naive joint importance weight  $\frac{\pi_{\theta}\pi_{\phi}}{\pi_{\theta_{\text{old}}}\pi_{\phi_{\text{old}}}}$  suffers from compounded variance. We mitigate this using an **alternating block-coordinate ascent** approximation. When updating the MLLM ( $\phi$ ), we fix the Allocator to its behavior policy ( $\pi_{\theta} = \pi_{\theta_{\text{old}}}$ ), yielding an importance ratio of exactly 1. The off-policy surrogate gradient for  $\phi$  becomes:

$$\nabla_{\phi} \mathcal{J}_{\text{surr}}(\phi) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \mathbb{E}_{\pi_{\phi_{\text{old}}}} \left[ \frac{\pi_{\phi}(\boldsymbol{y} \mid \tilde{\boldsymbol{x}})}{\pi_{\phi_{\text{old}}}(\boldsymbol{y} \mid \tilde{\boldsymbol{x}})} R_{s,\boldsymbol{y}}^{\text{ideal}} \nabla_{\phi} \log \pi_{\phi}(\boldsymbol{y} \mid \tilde{\boldsymbol{x}}) \right]. \tag{32}$$

Applying the log-derivative identity  $\nabla_{\phi} r_{\phi} = r_{\phi} \nabla_{\phi} \log \pi_{\phi}$  (where  $r_{\phi} = \pi_{\phi} / \pi_{\phi_{\text{old}}}$ ), we derive the surrogate objective:

$$\mathcal{L}_{\phi}^{\text{ideal}} = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \mathbb{E}_{\pi_{\phi_{\text{old}}}} \left[ r_{\phi}(\boldsymbol{y} \mid \tilde{\boldsymbol{x}}) R_{s,\boldsymbol{y}}^{\text{ideal}} \right]. \tag{33}$$

Policy-gradient ascent on  $\phi$  maximizes  $\mathcal{L}_{\phi}^{\text{ideal}}$ . Sec. 3.4 implements the clipped PPO surrogate, substituting  $R^{\text{ideal}}$  with advantages.

Conversely, when updating the Allocator ( $\theta$ ), we fix the backbone to its behavior policy ( $\pi_{\phi} = \pi_{\phi_{\text{old}}}$ ). The corresponding ideal allocator surrogate is:

$$\mathcal{L}_{\theta}^{\text{ideal}} = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ r_{\theta}(\mathbf{s} \mid \mathbf{x}) \, R_{\mathbf{s}}^{\text{ideal}} \right], \qquad r_{\theta}(\mathbf{s} \mid \mathbf{x}) = \frac{\pi_{\theta}(\mathbf{s} \mid \mathbf{x})}{\pi_{\theta_{\text{old}}}(\mathbf{s} \mid \mathbf{x})}, \tag{34}$$

where  $R_s^{\text{ideal}} = \mathbb{E}_{\pi_{\phi_{\text{old}}}(y|\tilde{x})}[R_{s,y}^{\text{ideal}}]$ . In practice, we approximate this expectation using Monte Carlo rollouts under the frozen backbone.

**Sequential allocator–backbone updates within one iteration.** The alternating derivation above fixes one policy while updating the other, rendering the inactive policy's importance ratio unity. In implementations that first update the Allocator from  $\theta_{\text{old}}$  to  $\theta'$  and subsequently update the MLLM on the *same* rollout batch, trajectories originate from the behavior pair  $(\theta_{\text{old}}, \phi_{\text{old}})$ . The MLLM gradient evaluation occurs under  $\phi$  at fixed (x, a, y). The importance weight  $\omega_{\theta} = q_{\theta'}(a \mid x)/q_{\theta_{\text{old}}}(a \mid x) = \pi_{\theta'}(s \mid x)/\pi_{\theta_{\text{old}}}(s \mid x)$  corrects the shift in the marginal allocation distribution. Multiplying rollout-level advantages by  $\omega_{\theta}$  prior to the token-level PPO surrogate for  $\phi$  implements the standard importance-sampling correction. This matches the practical "ispred" path in our codebase.

Advantage Shaping and Monte Carlo Surrogates. The ideal linear penalty  $-\lambda C(s)$  inside  $R^{\text{ideal}}$  frequently triggers catastrophic collapse to minimum budgets. CAPO mitigates this by replacing the raw reward with a cost-shaped, group-normalized advantage  $A_{s,y}$  (denoted  $A_{m,n}$  in the main text). This substitution is *not* an unbiased baseline transformation of  $R_{s,y}^{\text{ideal}}$ . Instead, it constitutes a deliberately biased surrogate objective that sacrifices exact fidelity to the Lagrangian reward in exchange for reduced variance and robust budget control.

Applying PPO clipping to the exact joint ratios would entangle all frame- and token-level factors, yielding prohibitive noise. We therefore adopt decoupled objectives. For a batch of M allocations and N rollouts per allocation, the MLLM sequence-level surrogate is:

$$\mathcal{L}_{\phi}^{\text{seq}} = -\frac{1}{MN} \sum_{m=1}^{M} \sum_{n=1}^{N} \min \left( r_{\phi}^{(m,n)} A_{m,n}, \operatorname{clip}(r_{\phi}^{(m,n)}, 1 - \varepsilon, 1 + \varepsilon) A_{m,n} \right). \tag{35}$$

This sequence-level loss remains approximate by substituting the CAPO-shaped advantage for the ideal reward. To enable finer credit assignment for the autoregressive MLLM, we factorize  $\pi_{\phi}(y \mid \tilde{x})$  into token-level probabilities, distribute the rollout-level advantage  $A_{m,n}$  across all tokens, and average over the sequence

length  $L_{m,n}$ . Equation (24) represents the standard token-level PPO approximation to this sequence-level surrogate.

When updating the Allocator ( $\theta$ ), we fix the MLLM ( $\pi_{\phi} = \pi_{\phi_{\text{old}}}$ ) and apply the aggregated advantage  $A_m^{\text{CAPO}} = \frac{1}{N} \sum_n A_{m,n}$ . Since the Allocator's output distribution factorizes conditionally across frames (Eq. 11), its score function decomposes additively:

$$\nabla_{\theta} \log \pi_{\theta}(\mathbf{s}^{(m)} \mid \mathbf{x}) = \sum_{t=1}^{T} \nabla_{\theta} \log \operatorname{Beta}(a_{t}^{(m)}; \alpha_{t}, \beta_{t}). \tag{36}$$

This additive log-probability structure facilitates low-variance frame-level credit assignment. Equation (21) provides a practical approximation to a trajectory-level clipped objective. We deploy this per-frame surrogate to guarantee stability in large-scale training.

### A.2. Complexity Analysis

We derive formal computational bounds for ResAdapt to establish when Allocator overhead becomes negligible compared to backbone savings. For clarity, we assume a standard Transformer backbone with quadratic self-attention and a uniform native resolution  $H \times W$  over T frames. Replacing HW with per-frame products  $H_tW_t$  extends this immediately to heterogeneous resolutions.

**Baseline cost.** Let *P* denote the ViT patch size. A vanilla MLLM encoding *T* frames at full resolution generates a visual token count of:

$$N_0 = T \cdot \left\lceil \frac{H}{P} \right\rceil \left\lceil \frac{W}{P} \right\rceil \approx \frac{THW}{P^2}. \tag{37}$$

**Adaptive cost and token retention ratio.** In our resize instantiation, the operator rescales frame  $f_t$  by factor  $s_t \in [s_{\min}, s_{\max}]$ , producing  $n_t(s_t) = \lceil s_t H/P \rceil \lceil s_t W/P \rceil \approx s_t^2 \cdot HW/P^2$  tokens. Summing across the sequence and normalizing by  $N_0$  yields the *token retention ratio*:

$$N^{\text{adapt}} = \sum_{t=1}^{T} n_t(s_t) \approx \frac{HW}{P^2} \sum_{t=1}^{T} s_t^2, \qquad \rho \triangleq \frac{N^{\text{adapt}}}{N_0} = \frac{1}{T} \sum_{t=1}^{T} s_t^2.$$
 (38)

Because the learned Beta policy concentrates redundant frames near  $s_{min}$  (Figure 5),  $\rho$  remains substantially smaller than 1. Across our evaluation suite,  $\rho \in [0.06, 0.16]$ .

**Quadratic FLOPs reduction.** For an  $L_{\text{mllm}}$ -layer MLLM with hidden dimension  $D_{\text{mllm}}$ , self-attention cost scales quadratically with visual sequence length:  $\Phi(N) = O(L_{\text{mllm}}N^2D_{\text{mllm}})$ . Substituting  $N^{\text{adapt}} = \rho \cdot N_0$  gives:

$$\Phi_{\text{mllm}}^{\text{adapt}} = O(L_{\text{mllm}} \cdot \rho^2 N_0^2 \cdot D_{\text{mllm}}). \tag{39}$$

This reflects a reduction by a factor of  $\rho^2$  relative to full-resolution processing. At a representative operating point of  $\rho = 0.11$ , we achieve  $\rho^2 \approx 0.012$ , eliminating roughly 83× of the backbone attention FLOPs.

**Allocator overhead.** The Allocator processes  $N_c = T \cdot \lceil H/P_c \rceil \lceil W/P_c \rceil$  coarsely pooled tokens across  $L_{\text{pred}}$  layers with dimension  $D_{\text{pred}}$ , utilizing a coarse spatial stride  $P_c \gg P$ . Its computational cost and relative overhead are:

$$\Phi_{\text{pred}} = O(L_{\text{pred}} \cdot N_c^2 \cdot D_{\text{pred}}), \qquad \frac{\Phi_{\text{pred}}}{\Phi_{\text{mllm}}^{\text{base}}} = O\left(\frac{L_{\text{pred}} D_{\text{pred}}}{L_{\text{mllm}} D_{\text{mllm}}} \cdot \left(\frac{P}{P_c}\right)^4\right) \ll 1.$$
 (40)

Inserting our implementation parameters (*Pc*=14, *L*pred=4, *D*pred=1,024 versus *L*mllm=28, *D*mllm=3,584), the Allocator consumes less than 3% of total inference FLOPs. The decision stage overhead remains trivial compared to the backbone computation it bypasses.

**Net speedup.** Combining these bounds under the first-order approximation Φbase mllm ≫ Φpred:

Speedup 
$$\approx \frac{\Phi_{\text{mllm}}^{\text{base}}}{\Phi_{\text{mllm}}^{\text{adapt}} + \Phi_{\text{pred}}} \approx \frac{N_0^2}{(N^{\text{adapt}})^2} = \frac{1}{\rho^2}.$$
 (41)

At *ρ* = 0.11, this dictates a theoretical 83× acceleration in backbone attention.

**Temporal context scaling.** These savings translate directly into expanded *temporal coverage*. Given a strict token budget *B*, a vanilla MLLM accommodates only *T*<sup>0</sup> = *BP*2/(*HW*) full-resolution frames. ResAdapt processes *T*0/*ρ* adaptively resized frames within the identical budget. This yields a 1/*ρ* ≈ 6–16× expansion in temporal horizon, unlocking the long-context performance detailed in Sec. [4.2.](#page-10-2)

**Acceleration transparency.** Input-side adaptation guarantees the backbone receives a standard, albeit shorter, visual-token sequence. Consequently, ResAdapt integrates seamlessly with optimized attention kernels like FlashAttention, vLLM [\(Kwon et al.,](#page-19-2) [2023\)](#page-19-2), and SGLang [\(Zheng et al.,](#page-23-0) [2024\)](#page-23-0) without necessitating low-level modifications. Conversely, model-side pruning and merging strategies introduce irregular token layouts that disrupt these kernels, demanding bespoke engineering fallbacks.

