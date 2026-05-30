# ResAdapt: Adaptive Resolution for Efficient Multimodal Reasoning

**Huanxuan Liao***τ***,** *<sup>µ</sup>* **, Zhongtao Jiang, Yupu Hao***τ***,** *<sup>µ</sup>* **, Yuqiao Tan***τ***,** *<sup>µ</sup>* **, Shizhu He***τ***,** *<sup>µ</sup>* **, Ben Wang, Jun Zhao***τ***,** *<sup>µ</sup>* **, Kun Xu**† **, Kang Liu***τ***,** *<sup>µ</sup>***,** <sup>∗</sup>

Institute of Automation, Chinese Academy of Sciences , *<sup>µ</sup>*University of Chinese Academy of Sciences ,

†Project Leader <sup>∗</sup>Corresponding author: [kliu@nlpr.ia.ac.cn](mailto:kliu@nlpr.ia.ac.cn)

Scaling both spatial resolution and temporal coverage in video reasoning demands visual-token budgets that grow prohibitively for Multimodal Large Language Models (MLLMs). Existing efficiency strategies intervene too late: model-side token pruning discards fine-grained evidence after the encoder has already paid the full computational cost, while output-side iterative retrieval introduces multi-turn latency. We propose **ResAdapt**, a framework that reallocates visual budget *before* encoding. A lightweight, query-aware Allocator predicts a per-frame resolution scale, adjusting the pixels the backbone receives while preserving its native token interface and compatibility with optimized inference engines. To train this non-differentiable pipeline, we introduce **Cost-Aware Policy Optimization (CAPO)**, which combines a dynamic cost pivot with asymmetric reward shaping to jointly maximize reasoning accuracy under strict visual budgets—preventing the policy collapse that plagues direct cost penalties. The resulting Allocator concentrates pixels on information-dense frames, exhibiting content-adaptive active perception learned entirely from task reward. Across video QA and temporal grounding benchmarks, ResAdapt matches or exceeds uncompressed baselines while eliminating over 90% of visual tokens. Crucially, the saved spatial budget is reinvested into temporal coverage: under equivalent compute, ResAdapt processes 16× more frames, yielding > 15% relative gains on complex long-video reasoning tasks.

**Project Page**: <https://xnhyacinth.github.io/projects/ResAdapt> **Code Repository**: <https://github.com/Xnhyacinth/ResAdapt>

**Contact**: [liaohuanxuan2023@ia.ac.cn](mailto:liaohuanxuan2023@ia.ac.cn)

# **1. Introduction**

Multimodal Large Language Models (MLLMs) achieve stronger visual understanding by scaling input fidelity, yet the resulting visual-token growth makes jointly sustaining high spatial resolution and long temporal context prohibitive [\(Guo et al.,](#page-19-0) [2025a,](#page-19-0) [Bai et al.,](#page-17-0) [2025a,](#page-17-0) [Liu et al.,](#page-20-0) [2025a,](#page-20-0) [Shu et al.,](#page-21-0) [2025,](#page-21-0) [Shao et al.,](#page-20-1) [2025b\)](#page-20-1). In practice, this trade-off is central to video reasoning: reducing resolution risks losing the small visual cues that determine the answer, whereas shortening the clip removes the temporal context needed for long-horizon inference. Even architecturally efficient encoders [\(Zhang et al.,](#page-22-0) [2026,](#page-22-0) [Liu et al.,](#page-20-2) [2025b\)](#page-20-2) do not remove this tension; they merely shift where it becomes painful.

Mainstream efficiency methods largely fall into two paradigms (Figure [1a](#page-1-0)), both of which intervene too late and share a common root: *they accept the encoder's full-resolution input as a fixed cost and attempt to recover efficiency downstream*. *Model-side* approaches prune or merge tokens after visual encoding [\(Khaki et al.,](#page-19-1) [2025,](#page-19-1) [Xu et al.,](#page-22-1) [2025,](#page-22-1) [Bolya et al.,](#page-18-0) [2022,](#page-18-0) [Tao et al.,](#page-21-1) [2025\)](#page-21-1). Once fine-grained evidence is discarded, it cannot be recovered, and the irregular token layouts that result from pruning or merging disrupt optimized attention

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

**Figure 1: Input-side Adaptation improves the visual-token efficiency frontier. (a)** Three efficiency paradigms for video reasoning. Model-side methods compress tokens after encoding; output-side methods iteratively retrieve or zoom; ResAdapt reallocates per-frame visual budget before encoding, preserving the backbone's native token interface and compatibility with optimized inference engines. **(b)** Qwen2.5-VL-7B results with 32 frames at ∼10% visual retention, where ResAdapt lies on or near the Pareto frontier and shows its largest gain on the reasoning-heavy benchmark.

kernels and inference engines [\(Dao,](#page-18-1) [2024,](#page-18-1) [Kwon et al.,](#page-19-2) [2023,](#page-19-2) [Zheng et al.,](#page-23-0) [2024\)](#page-23-0). Conversely, *output-side* agentic reasoning introduces iterative retrieval or zoom steps [\(Zhang et al.,](#page-23-1) [2025b,](#page-23-1) [Yang et al.,](#page-22-2) [2025d,](#page-22-2) [Shen et al.,](#page-21-2) [2025b,](#page-21-2) [Zheng et al.,](#page-23-2) [2025b\)](#page-23-2). While this strategy recovers coverage, it multiplies inference cost: each retrieval step demands a separate backbone call, and the initial coarse view that triggers refinement frequently undersamples the very cues it seeks to recover.

We argue that the intervention point itself is the problem. Rather than compressing representations *after* encoding or retrieving them *after* reasoning, an efficient system should optimize the pixel volume the encoder receives in the first place. Our framework, **ResAdapt**, instantiates this *input-side adaptation* principle: a lightweight Allocator predicts a per-frame visual allocation from coarse features and the query, then realizes that allocation through a visual budget operator, such as resolution resizing or frame selection. The backbone therefore processes a standard—albeit shorter—visual-token sequence in a single call, preserving full compatibility with FlashAttention, vLLM [\(Kwon et al.,](#page-19-2) [2023\)](#page-19-2), and SGLang [\(Zheng et al.,](#page-23-0) [2024\)](#page-23-0) without bespoke kernel engineering. Compared with prior slow–fast pipelines [\(Yang et al.,](#page-22-3) [2025a,](#page-22-3) [Zhang et al.,](#page-22-0) [2026\)](#page-22-0), which route frames using query-agnostic heuristics or fixed resolution tiers, ResAdapt learns a query-aware allocation policy directly from task reward.

Optimizing this pre-encoding allocation presents severe reinforcement learning challenges: the action space is continuous, the visual operator is non-differentiable, and naive accuracy–cost penalties catastrophically collapse the policy toward minimum budgets. We overcome these optimization hurdles with **Cost-Aware Policy Optimization (CAPO)**, which converts sparse rollout feedback into a stable asymmetric learning signal, and a temporal-similarity regularizer that suppresses redundant high-budget allocations on adjacent similar frames. Together, these components transform Input-side adaptation into a trainable, content-aware policy rather than a handcrafted compression rule.

Extensive empirical evaluations across video QA and temporal grounding benchmarks demonstrate that ResAdapt decisively advances the efficiency–accuracy Pareto frontier. ResAdapt matches or surpasses state-of-the-art token economy methods while discarding over 90% of visual tokens (Figure 1b). Crucially, this spatial compression unlocks massive temporal expansion: under equivalent computational budgets, ResAdapt processes  $16 \times$  more frames, yielding > 15% relative performance gains. Furthermore, the learned policy exhibits *active perception*—autonomously concentrating visual budget on decisive frames in a single forward pass, without requiring explicit saliency supervision.

Our main contributions are:

- 1. We introduce **ResAdapt**, an *input-side adaptation* framework that formulates dynamic per-frame visual budgeting as a contextual bandit problem, fully preserving the native architecture and hardware optimizations of MLLMs.
- 2. We propose **CAPO** with a temporal similarity regularizer, providing a stable, asymmetric learning signal that jointly optimizes accuracy and cost without hand-crafted heuristics.
- 3. Extensive experiments and ablations demonstrate that ResAdapt achieves a superior efficiency–accuracy Pareto frontier across video QA and temporal grounding tasks, with the learned policy exhibiting content-adaptive active perception.

### 2. Background and Problem Formulation

#### 2.1. Preliminaries

Given a text query q and a video  $\mathcal{V} = \{f_t\}_{t=1}^T$ , let  $x = (q, \mathcal{V})$  denote the full input. A backbone policy  $\pi_{\phi}$  encodes every frame at fixed fidelity and autoregressively generates a rollout  $y = (y_1, \dots, y_L)$ :

$$\pi_{\phi}(y \mid x) = \prod_{j=1}^{L} \pi_{\phi}(y_j \mid y_{< j}, x).$$
 (1)

When useful, we write y = (r, o) for a reasoning trace r and a final answer o. The computational inefficiency of this paradigm is stark: visual encoding cost scales quadratically with pixel volume, yet the evidence required to answer complex queries remains remarkably sparse in time.

To control pre-encoding cost, we introduce an Allocator policy  $\pi_{\theta}$  that emits a per-frame allocation vector

$$\mathbf{s} = (s_1, \dots, s_T) \sim \pi_{\theta}(\cdot \mid \mathbf{x}), \qquad s_t \in [s_{\min}, s_{\max}], \tag{2}$$

and applies a *visual budget operator*  $\mathcal{O}$  to each frame:  $\tilde{f}_t = \mathcal{O}(f_t, s_t)$ . The backbone then generates from the transformed input  $\tilde{x} = (q, \{\tilde{f}_t\}_{t=1}^T)$ :

$$\pi_{\phi}(\boldsymbol{y} \mid \tilde{\boldsymbol{x}}) = \prod_{j=1}^{L} \pi_{\phi}(y_j \mid y_{< j}, \tilde{\boldsymbol{x}}). \tag{3}$$

We keep  $\mathcal{O}$  abstract only to state the decision problem cleanly. The framework is operator-agnostic:  $\mathcal{O}$  may implement resizing, frame selection, or other pre-encoding budget controls.

#### <span id="page-3-3"></span>2.2. Problem Formulation

Because the Allocator acts once before decoding, the outer problem is a *Contextual Bandit* (equivalently, a one-step contextual MDP). The context is the raw input  $x \in \mathcal{X}$ , and the action is the continuous allocation vector  $s \in [s_{\min}, s_{\max}]^T$ . For joint training, it is convenient to write the induced two-stage policy as

$$p_{\theta,\phi}(s,y\mid x) = \pi_{\theta}(s\mid x)\,\pi_{\phi}(y\mid \tilde{x}),\tag{4}$$

where  $\tilde{x} = (q, \{\mathcal{O}(f_t, s_t)\}_{t=1}^T)$  is the deterministically transformed input. The immediate reward is response quality r(x, s, y) = Q(x, y).

Let C(s) denote the visual cost induced by allocation s. The ideal budgeted objective is

<span id="page-3-0"></span>
$$\max_{\theta,\phi} \quad \mathbb{E}_{x \sim \mathcal{D}, s \sim \pi_{\theta}(\cdot|x), y \sim \pi_{\phi}(\cdot|\tilde{x})}[Q(x, y)]$$
s.t. 
$$\mathbb{E}_{x \sim \mathcal{D}, s \sim \pi_{\theta}(\cdot|x)}[C(s)] \leq \tau,$$
(5)

where  $\tau$  is the target budget. Lagrangian relaxation yields the unconstrained utility

<span id="page-3-1"></span>
$$\max_{\theta, \phi} \mathbb{E}_{x, s, y}[\mathcal{U}(x, s, y)],$$

$$\mathcal{U}(x, s, y) = Q(x, y) - \lambda C(s),$$
(6)

for trade-off coefficient  $\lambda \geq 0$ .

Equations (5)–(6) define the target trade-off but not yet a stable optimizer. Section 3 instantiates this objective with an Input-side adaptation policy, CAPO, temporal regularization, and PPO-style surrogate losses; the experiments use resize as the concrete operator. Detailed derivations are deferred to Appendix A.1.

### <span id="page-3-2"></span>3. Method

Figure 2 illustrates ResAdapt. At inference, the Allocator predicts a scalar allocation per frame and applies a pre-encoding operator before the video reaches the backbone. In our primary instantiation, the operator  $\mathcal{O}$  performs bilinear resizing: the allocation determines a per-frame resize factor  $s_t$ , yielding  $\tilde{f}_t = \mathcal{R}(f_t, s_t)$ . At training time, rollout feedback from the backbone updates the Allocator and, optionally, the backbone itself.

### 3.1. Joint RL Optimization Framework

As formulated in Section 2.2, we cast pre-encoding allocation as a contextual bandit. Starting from the marginal probability of generating the correct answer under the transformed input (see Appendix A.1), we derive a one-step expected-reward objective. Abstracting the answer-quality term as a rollout utility Q(x,y)—treated as parameter-free once y has been sampled—the joint policy factorizes as

$$p_{\theta,\phi}(s,y\mid x) = \pi_{\theta}(s\mid x)\,\pi_{\phi}(y\mid \tilde{x}). \tag{7}$$

Here  $\pi_{\theta}(s \mid x)$  is the density induced by the latent Beta policy  $q_{\theta}(a \mid x)$  through the affine map in Eq. (10). Because this map has a  $\theta$ -independent Jacobian,  $\nabla_{\theta} \log \pi_{\theta}(s \mid x)$  coincides with  $\nabla_{\theta} \log q_{\theta}(a \mid x)$ , so all PPO ratios can be evaluated directly on the latent actions  $a_t$  (Eq. 11). The corresponding ideal rollout reward combines task quality and visual cost:

$$R_{s,y}^{\text{ideal}} = Q(x,y) - \lambda C(s), \tag{8}$$

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 2: ResAdapt framework. (a) At inference, a lightweight Allocator  $\pi_{\theta}$  maps coarse visual features and the query to latent actions  $a_t \sim \text{Beta}(\alpha_t, \beta_t)$ , which parameterize per-frame input allocations. In the resize instantiation used in our experiments, these allocations are realized as scales  $s_t \in [s_{\min}, s_{\max}]$ , and the resized frames are processed by the MLLM in a single call. (b) During training, CAPO reshapes group-relative advantages with a dynamic cost pivot  $\tau_{\text{dyn}}$ , while temporal-similarity regularization suppresses redundant high-budget allocation on adjacent similar frames.

and the optimization target becomes

<span id="page-4-1"></span>
$$\max_{\theta, \phi} \ \mathcal{J}(\theta, \phi) = \mathbb{E}_{\pi_{\theta}(s|x)} \Big[ \mathbb{E}_{\pi_{\phi}(y|\tilde{x})} \Big[ R_{s,y}^{\text{ideal}} \Big] \Big] . \tag{9}$$

Equation (9) defines the expected return for a single context x; training marginalizes over  $x \sim \mathcal{D}$ . While the policy gradients follow the standard score-function estimator (Appendix A.1), directly optimizing this objective is brittle in practice due to three challenges:

- 1. **Policy parameterization.**  $\pi_{\theta}$  must emit a *T*-dimensional continuous action with negligible overhead relative to the backbone.
- 2. **Credit assignment.** The raw Lagrangian reward  $Q(x,y) \lambda C(s)$  exhibits extreme variance and frequently collapses the policy to the minimum allowable budget, since every reduction in C is unconditionally rewarded regardless of answer quality.
- 3. **Temporal structure.** Rollout-level rewards carry no frame-level granularity, permitting redundant high-budget allocations on visually near-duplicate neighbors.

The remainder of this section resolves each bottleneck in turn.

#### <span id="page-5-2"></span>3.2. Allocator Architecture

Each frame  $f_t \in \mathbb{R}^{3 \times H_t \times W_t}$  is encoded by a frozen lightweight visual encoder; the query is encoded separately. Both representations are projected to a shared dimension D. A shallow Transformer decoder alternates temporal self-attention over  $\{f_t\}_{t=1}^T$  with gated cross-attention to the query, producing per-frame hidden states  $\{h_t\}_{t=1}^T$ . This architecture exposes both temporal redundancy and query dependence at low cost.

We parameterize each latent action with a Beta distribution, whose bounded support maps naturally to  $[s_{\min}, s_{\max}]$ :

<span id="page-5-0"></span>
$$a_t \sim \text{Beta}(\alpha_t, \beta_t), \qquad s_t = s_{\min} + a_t (s_{\max} - s_{\min}).$$
 (10)

Since  $a_t \in (0,1)$ , the allocation satisfies  $s_t \in (s_{\min}, s_{\max})$  almost surely; setting  $0 < s_{\min} < 1 < s_{\max}$  permits both downscaling and selective upscaling. Let  $q_{\theta}(\boldsymbol{a} \mid \boldsymbol{x})$  denote the joint latent policy over  $\boldsymbol{a} = (a_1, \dots, a_T)$ . Conditioned on  $\{h_t\}$ , the log-density factorizes across frames:

<span id="page-5-1"></span>
$$\log q_{\theta}(\mathbf{a} \mid \mathbf{x}) = \sum_{t=1}^{T} \log \operatorname{Beta}(a_{t}; \alpha_{t}, \beta_{t}). \tag{11}$$

The affine map  $a \mapsto s$  induces the allocation policy  $\pi_{\theta}(s \mid x)$ ; change-of-variables details are deferred to Appendix A.1.

### <span id="page-5-3"></span>3.3. Cost-Aware Policy Optimization (CAPO)

A flat penalty on C(s) drives the policy toward uniformly minimal budgets regardless of question difficulty: any cost reduction is rewarded identically whether it preserves or destroys the answer. CAPO replaces this raw penalty with a shaped signal that couples cost awareness to answer correctness.

**Compute metric.** For the resize operator, if frame  $f_t \in \mathbb{R}^{3 \times H_t \times W_t}$  is rescaled by  $s_t$ , its visual token count satisfies  $n_t(s_t) \propto \lceil s_t H_t / P \rceil \lceil s_t W_t / P \rceil$  for patch size P. We measure physical compute by the *token retention ratio* 

$$\rho(s) = \frac{\sum_{t=1}^{T} n_t(s_t)}{\sum_{t=1}^{T} n_t(1)} \approx \frac{\sum_{t=1}^{T} s_t^2 H_t W_t}{\sum_{t=1}^{T} H_t W_t}.$$
 (12)

Because frames are normalized to a common base resolution before allocation,  $\rho(s)$  reduces to the mean quadratic scale.

**Proxy cost.** The quadratic dependence of  $\rho$  on  $s_t$  amplifies a few large allocations and inflates gradient variance. We therefore optimize against a smoother proxy

$$c(s) = \frac{\bar{s} - s_{\min}}{s_{\max} - s_{\min}}, \qquad \bar{s} = \frac{1}{T} \sum_{t=1}^{T} s_t, \tag{13}$$

used only inside the optimizer; the quadratic  $\rho(s)$  remains the efficiency metric reported in all experiments.

**Base advantage.** For each prompt x, let  $R_{m,n}^{\text{task}}$  denote the scalar task reward of rollout (m,n) (defined in Appendix B.3),  $A_{m,n}^{\text{base}}$  the corresponding GRPO group-normalized advantage,  $c_m = c(s_m)$  the proxy cost of allocation m, and  $u_{m,n} \in \{0,1\}$  a binary correctness indicator (exact-match for QA; thresholded success for continuous metrics).

**Dynamic cost pivot.** CAPO's key ingredient is a decision boundary that determines whether a sampled cost  $c_m$  should be rewarded for efficiency or penalized for being expensive. A fixed target budget  $\tau_{\text{fix}}$  ignores the

policy's current state, causing unstable updates when the model operates far from this target. Conversely, using only the prompt-local mean  $\bar{c}_{\text{group}} = \frac{1}{M} \sum_{m=1}^{M} c_m$  encourages relative efficiency but cannot anchor the policy to the absolute compression goal. CAPO interpolates between both via a dynamic pivot:

$$\tau_{\rm dyn} = \kappa_{\rm mix} \, \bar{c}_{\rm group} + \left(1 - \kappa_{\rm mix}\right) \tau_{\rm fix},\tag{14}$$

where  $\kappa_{mix} \in [0,1]$ . The group mean provides a state-aware baseline for local cost comparisons, while  $\tau_{fix}$  continuously steers the policy toward the global compression target.

**Asymmetric shaping.** With  $\tau_{dvn}$  as pivot, CAPO applies a correctness-dependent cost signal:

<span id="page-6-0"></span>
$$S_{m,n} = \begin{cases} \lambda_{+} \sigma \left( \frac{\tau_{\text{dyn}} - c_{m}}{\tau_{\text{s}}} \right) & \text{if } u_{m,n} = 1, \\ -\lambda_{-} \sigma \left( \frac{c_{m} - \tau_{\text{dyn}}}{\tau_{\text{s}}} \right) & \text{if } u_{m,n} = 0, \end{cases}$$

$$(15)$$

with  $\lambda_- > \lambda_+ > 0$ . A correct rollout at below-pivot cost receives a moderate bonus; an incorrect rollout at above-pivot cost receives a stronger penalty. The sigmoid temperature  $\tau_s$  smooths the transition near the boundary. This asymmetry is the mechanism that prevents collapse: reducing cost on correct answers is encouraged, but reducing cost at the expense of correctness is strictly penalized.

Final CAPO advantage. The shaped signal is combined with the base advantage:

$$\tilde{A}_{m,n} = A_{m,n}^{\text{base}} + \lambda_{\text{capo}} S_{m,n} - \gamma c_m, \tag{16}$$

where  $\lambda_{\text{capo}} > 0$  scales the shaping term and  $\gamma \geq 0$  applies a residual global cost pressure. The final advantage applies a floor on correct rollouts:

<span id="page-6-1"></span>
$$A_{m,n} = \begin{cases} \max(\tilde{A}_{m,n}, \varepsilon_+) & \text{if } u_{m,n} = 1, \\ \tilde{A}_{m,n} & \text{if } u_{m,n} = 0, \end{cases}$$

$$(17)$$

ensuring that correct, low-cost rollouts always retain a positive learning signal ( $\varepsilon_+ > 0$ ).

### <span id="page-6-2"></span>3.4. Regularization and Training Objective

CAPO stabilizes the global accuracy—cost trade-off but does not break the symmetry among visually redundant neighbors: the optimizer can assign identical scales to adjacent near-duplicate frames without penalty. We introduce two regularizers to resolve this.

**Temporal similarity loss** ( $\mathcal{L}_{sim}$ ). Reusing the coarse features  $f_t$  from Sec. 3.2, we penalize redundant joint high-budget allocation on similar adjacent pairs:

$$\mathcal{L}_{\text{sim}} = \frac{1}{T - 1} \sum_{t=1}^{T - 1} w_t \cdot \max(0, \log s_t + \log s_{t+1} + \eta_{\text{sim}}),$$
 (18)

where the similarity-gated weight

$$w_t = \sigma \left( \frac{\cos(f_t, f_{t+1}) - \tau_{\text{sim}}}{\gamma_{\text{sim}}} \right) \tag{19}$$

activates only when adjacent frames exceed a cosine-similarity threshold  $\tau_{\text{sim}} \in (0,1)$ , with temperature  $\gamma_{\text{sim}}$ . No penalty is incurred when  $s_t s_{t+1} \leq e^{-\eta_{\text{sim}}}$ .

**Concentration loss** ( $\mathcal{L}_{con}$ ). To prevent the Beta distributions from collapsing to near-deterministic spikes, we softly cap the total concentration at  $\kappa_{max} > 0$ :

$$\mathcal{L}_{\text{con}} = \frac{1}{T} \sum_{t=1}^{T} \max(0, \alpha_t + \beta_t - \kappa_{\text{max}}).$$
 (20)

Together,  $\mathcal{L}_{sim}$  forces differentiated allocation across redundant neighbors, while  $\mathcal{L}_{con}$  preserves sufficient stochasticity for continued exploration.

**Training procedure.** We optimize both policies in a single GRPO-style loop (Zheng et al., 2025a, Yu et al., 2025). For each prompt x, the Allocator draws M allocation trajectories  $s_{1:M}$ ; each transformed input  $\tilde{x}^{(m)}$  produces N response rollouts from the backbone. CAPO computes per-rollout advantages  $A_{m,n}$ , which serve as the shared learning signal for both policies (Appendix A.1).

**Allocator objective.** Rollout advantages are aggregated per allocation,  $A_m^{\text{CAPO}} = \frac{1}{N} \sum_n A_{m,n}$ , and used in a per-frame PPO surrogate:

<span id="page-7-2"></span>
$$\mathcal{L}_{\theta} = -\frac{1}{MT} \sum_{m=1}^{M} \sum_{t=1}^{T} \min \left( r_{\theta,t}^{(m)} A_{m}^{\text{CAPO}}, \operatorname{clip}\left(r_{\theta,t}^{(m)}, 1 - \varepsilon, 1 + \varepsilon\right) A_{m}^{\text{CAPO}} \right), \tag{21}$$

where the per-frame importance ratio is

$$r_{\theta,t}^{(m)} = \frac{q_{\theta}(a_t^{(m)} \mid \mathbf{x})}{q_{\theta_{\text{old}}}(a_t^{(m)} \mid \mathbf{x})}.$$
 (22)

The full Allocator loss combines the policy gradient with both regularizers:

<span id="page-7-0"></span>
$$\mathcal{L}_{\text{alloc}} = \mathcal{L}_{\theta} + \lambda_{\text{sim}} \mathcal{L}_{\text{sim}} + \lambda_{\text{con}} \mathcal{L}_{\text{con}}.$$
 (23)

**Backbone update.** Conditioned on the sampled allocations, the backbone is updated with the standard token-level PPO surrogate:

<span id="page-7-1"></span>
$$\mathcal{L}_{\phi} = -\frac{1}{MN} \sum_{m=1}^{M} \sum_{n=1}^{N} \frac{1}{L_{m,n}} \sum_{j=1}^{L_{m,n}} \min \left( r_{\phi,j}^{(m,n)} A_{m,n}, \operatorname{clip} \left( r_{\phi,j}^{(m,n)}, 1 - \varepsilon, 1 + \varepsilon \right) A_{m,n} \right), \tag{24}$$

where  $L_{m,n}$  is the rollout length and

$$r_{\phi,j}^{(m,n)} = \frac{\pi_{\phi}(y_j^{(m,n)} \mid y_{< j}^{(m,n)}, \tilde{\mathbf{x}}^{(m)})}{\pi_{\phi_{\text{old}}}(y_j^{(m,n)} \mid y_{< j}^{(m,n)}, \tilde{\mathbf{x}}^{(m)})}.$$
(25)

The two objectives are fully decoupled:  $\mathcal{L}_{alloc}$  updates only  $\theta$  while  $\mathcal{L}_{\phi}$  updates only  $\phi$ , so either component can be frozen or activated independently. When the backbone is held fixed, only the Allocator is trained; when both are active, the two losses are optimized alternately within the same training loop. Algorithm 1 summarizes one iteration.

<span id="page-8-0"></span>**Table 1: Video QA results** across two backbones (Qwen2.5-VL-7B, Qwen3-VL-8B) and two temporal horizons (32/128 frames). Retention ratio R reflects visual token count; *Reasoning* ( $\checkmark/x$ ) indicates chain-of-thought use; **bold** marks the best result per group.

| Backbone      | Method                                       | Retention                           | Reasoning      | Vic                                   | leo Perception Ben                   | Video Reasoning Benchmark |                      |                                 |                        |  |  |  |
|---------------|----------------------------------------------|-------------------------------------|----------------|---------------------------------------|--------------------------------------|---------------------------|----------------------|---------------------------------|------------------------|--|--|--|
| Бискропс      | Method                                       | Ratio R                             | reasoning      | VideoMME                              | LongVideoBench                       | MMVU                      | MLVU                 | VideoMMMU                       | LVBench                |  |  |  |
|               |                                              |                                     |                | 32 Fra                                | mes                                  |                           |                      |                                 |                        |  |  |  |
| -             | Vanilla                                      | 100%                                | X              | 62.0                                  | 58.9                                 | 52.7                      | 63.1                 | 49.6                            | 38.6                   |  |  |  |
|               | Random Drop                                  | 25.0%                               | x              | 58.9                                  | 57.8                                 | 49.6                      | 58.3                 | 45.3                            | 36.7                   |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 25.0%                               | X              | 58.7                                  | 58.0                                 | 51.0                      | 58.7                 | 41.8                            | 37.7                   |  |  |  |
|               | VisionZip (Yang et al., 2025c)               | 25.0%                               | X              | 59.4                                  | 57.1                                 | 49.8                      | 57.9                 | 42.4                            | 36.5                   |  |  |  |
|               | FlashVid (Fan et al., 2026)                  | 29.3%                               | X              | 60.2                                  | 58.6                                 | 51.1                      | 59.2                 | 46.3                            | 36.9                   |  |  |  |
|               | FixedScale                                   | 25.0%                               | Х              | 60.0                                  | 56.8                                 | 51.2                      | 59.8                 | 46.7                            | 37.3                   |  |  |  |
|               | ResAdapt (Ours)                              | 23.8%                               | <mark>X</mark> | 60.3                                  | 58.2                                 | 51.9                      | 60.1                 | 48.8                            | 37.9                   |  |  |  |
|               | Random Drop                                  | 10.0%                               | X              | 56.1                                  | 55.6                                 | 47.1                      | 56.5                 | 39.8                            | 35.2                   |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 10.0%                               | X              | 56.4                                  | 55.2                                 | 48.9                      | 58.0                 | 39.2                            | 33.6                   |  |  |  |
|               | VisionZip (Yang et al., 2025c)               | 10.0%                               | X              | 55.5                                  | 54.5                                 | 47.6                      | 57.3                 | 39.1                            | 35.3                   |  |  |  |
| 7B            | FlashVid (Fan et al., 2026)                  | 10.4%                               | X              | 57.9                                  | 56.8                                 | 47.9                      | 57.7                 | 39.4                            | 36.5                   |  |  |  |
| Qwen2.5-VL-7B | FixedScale ResAdapt (Ours)                   | 12.3%                               | Х              | 58.0                                  | 55.1<br>55.4                         | 47.7                      | 57.5                 | 44.3                            | 35.4                   |  |  |  |
| 2-7           | VideoAuto-R1 (Liu et al., 2026)              | $-\frac{11.4\%}{100\%}$             | <mark>X</mark> | <mark>59.4</mark><br>63.2             | 55.4<br>58.9                         | - <b>49.2</b> - 55.0      | - <b>58.4</b> 60.0   | <b>45.7</b><br>53.6             | 35.9<br>41.5           |  |  |  |
| 12.           | + ResAdapt (Ours)                            | 23.8%                               | <b>✓</b>       | 60.4                                  | 58.9<br>57.1                         | 53.2                      | 61.1                 | 53.6                            | 38.7                   |  |  |  |
| Ş.            | + ResAdapt (Ours)                            | 23.8%<br>11.4%                      | <b>/</b>       | 59.3                                  | 56.3                                 | 51.8                      | 59.3                 | 49.1                            | 36.7                   |  |  |  |
| 5             | + ResAdapt (Odis)                            | 11.470                              | <b>,</b>       |                                       |                                      | 31.0                      | 39.3                 | 49.1                            | 30.7                   |  |  |  |
|               |                                              |                                     |                | 128 Fra                               |                                      |                           |                      |                                 |                        |  |  |  |
|               | Vanilla                                      | 100%                                | <mark>X</mark> | 65.3                                  | 60.3                                 | 53.1                      | 66.5                 | 47.9                            | 42.0                   |  |  |  |
|               | Random Drop                                  | 25.0%                               | x              | 64.9                                  | 61.2                                 | 50.8                      | 64.8                 | 48.1                            | 41.3                   |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 25.0%                               | X              | 65.1                                  | 61.6                                 | 51.9                      | 63.1                 | 46.6                            | 42.1                   |  |  |  |
|               | VisionZip (Yang et al., 2025c)               | 25.0%                               | Х              | 64.8                                  | 61.3                                 | 51.1                      | 64.5                 | 47.3                            | 41.5                   |  |  |  |
|               | ResAdapt (Ours) Random Drop                  | $-\frac{22.9\%}{10.0\%}$            | <mark>,</mark> | <mark>65.6</mark><br><del>6</del> 3.0 | <del>6</del> 0.2<br><del>5</del> 9.0 | 52.8                      | 65.9                 | <del>51.1</del><br>46.7         | $-\frac{42.1}{38.0}$   |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 10.0%                               | ×              | 60.6                                  | 56.3                                 | 45.8<br>44.2              | 63.4<br>63.5         | 40.7                            | 39.5                   |  |  |  |
|               | VisionZip (Yang et al., 2025c)               | 10.0%                               | x              | 61.8                                  | 56.1                                 |                           | 63.2                 | 42.1                            | 39.3                   |  |  |  |
|               | FixedScale                                   | 10.0%                               | x              | 64.1                                  | 60.9                                 | 44.8<br><b>49.6</b>       | 64.5                 | 46.9                            | 40.3                   |  |  |  |
|               | ResAdapt (Ours)                              | 11.1%                               | X              | 63.8                                  | 58.6                                 | 49.0                      | 64.3                 | 49.2                            | 39.9                   |  |  |  |
|               | VideoAuto-R1 (Liu et al., 2026)              | $-\frac{11.170}{100\%}$             | <del></del>    | <del>6</del> 4.7                      | <del>5</del> 9.1                     | 56.7                      | 65.1                 | <del>52.2</del>                 | 41.2                   |  |  |  |
|               | + ResAdapt (Ours)                            | 23.8%                               | /              | 66.2                                  | 60.2                                 | 53.5                      | 66.0                 | 52.6                            | 41.8                   |  |  |  |
|               | + ResAdapt (Ours)                            | 11.4%                               | ✓              | 64.7                                  | 57.8                                 | 52.4                      | 64.6                 | 51.3                            | 39.5                   |  |  |  |
|               |                                              |                                     |                | 32 Frai                               | mes                                  |                           |                      |                                 |                        |  |  |  |
|               | Vanilla                                      | 100%                                | Х              | 65.0                                  | 58.6                                 | 57.5                      | 64.0                 | 60.8                            | 40.2                   |  |  |  |
|               | Random Drop                                  | 25.0%                               | <del>x</del> : | 61.3                                  | 58.4                                 | 57.1                      | 60.2                 | 53.4                            | 37.8                   |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 25.0%                               | X              | 62.4                                  | 57.4                                 | 56.0                      | 60.8                 | 49.1                            | 36.4                   |  |  |  |
|               | VisionZip (Yang et al., 2025c)               | 25.0%                               | X              | 61.8                                  | 57.2                                 | 54.4                      | 60.6                 | 51.5                            | 37.3                   |  |  |  |
|               | FlashVid (Fan et al., 2026)                  | 30.0%                               | X              | 63.9                                  | 59.0                                 | 54.8                      | 61.9                 | 55.1                            | 38.5                   |  |  |  |
|               | ResAdapt (Ours)                              | 23.8%_                              | X              | 62.6                                  | 57.5                                 | 55.3                      | 61.0                 | 58.4                            | 38.5                   |  |  |  |
|               | Random Drop                                  | 10.0%                               | x -            | 58.8                                  | 54.7                                 | 53.2                      | 56.6                 | 47.1                            | 35.5                   |  |  |  |
| m             | ToMe (Bolya et al., 2022)                    | 10.0%                               | X              | 59.2                                  | 55.5                                 | 53.1                      | 58.5                 | 42.7                            | 35.8                   |  |  |  |
| <u>∞</u>      | VisionZip (Yang et al., 2025c)               | 10.0%                               | X              | 59.9                                  | 55.4                                 | 53.7                      | 58.8                 | 45.8                            | 35.4                   |  |  |  |
| <b>I</b> -    | FlashVid (Fan et al., 2026)                  | 12.2%                               | X              | 61.0                                  | 57.1                                 | 54.8                      | 59.1                 | 47.8                            | 37.1                   |  |  |  |
| n3            | FixedScale                                   | 12.3%                               | Х              | 60.8                                  | 54.9                                 | 53.8                      | 58.4                 | 52.6                            | 37.1                   |  |  |  |
| Qwen3-VL-8B   | ResAdapt (Ours)                              | 11.4%                               | Х              | 60.7                                  | 56.6                                 | 54.6                      | 59.6                 | 56.1                            | 37.3                   |  |  |  |
| J             | 128 Frames                                   |                                     |                |                                       |                                      |                           |                      |                                 |                        |  |  |  |
|               | Vanilla<br>Random Drop                       | $-\frac{100\%}{25.0\%}$             | <del>"</del>   | <del>6</del> 9.4<br><del>6</del> 7.2  | <del>64.3</del>                      | 58.5                      | $-\frac{72.7}{67.4}$ | <del>63.0</del> <del>55.3</del> | $\frac{45.7}{42.4}$    |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 25.0%<br>25.0%                      | X              |                                       | 61.3<br><b>62.0</b>                  | 56.8                      |                      |                                 |                        |  |  |  |
|               | VisionZip (Yang et al., 2025c)               | 25.0%<br>25.0%                      | X<br>X         | 67.2<br>67.1                          | 62.0<br>61.3                         | 55.9<br>55.7              | 70.4<br>69.2         | 53.5<br>56.8                    | 43.1<br>41.2           |  |  |  |
|               | ResAdapt (Ours)                              | 25.0%                               | ×              | 67.1                                  | 61.9                                 | 56.3                      | 70.8                 | 50.8<br><b>59.6</b>             | 43.3                   |  |  |  |
|               | Random Drop                                  | <del>22.9</del> % <del>10.0</del> % | <del>´x</del>  | <del>67.4</del>                       | <del>5</del> 8.3                     | - 55.4                    | 62.4                 | <del>55.</del> 5                | <del>43.3</del> - 38.8 |  |  |  |
|               | ToMe (Bolya et al., 2022)                    | 10.0%                               | X              | 64.7                                  | 58.6                                 | 55.1                      | 67.3                 | 46.3                            | 40.5                   |  |  |  |
|               |                                              | 10.0%                               | X              | 64.2                                  | 59.1                                 | 54.2                      | 66.8                 | 47.6                            | 39.4                   |  |  |  |
|               | VisionZin (Yang et al. 2025c)                |                                     |                |                                       |                                      |                           |                      |                                 |                        |  |  |  |
|               | VisionZip (Yang et al., 2025c)<br>FixedScale | 12.3%                               | x              | 66.7                                  | 59.5                                 | 54.4                      | 67.7                 | 56.3                            | 41.7                   |  |  |  |

### <span id="page-9-0"></span>Algorithm 1 ResAdapt Training (One Iteration)

```
Require: Prompt batch \{x_i\}, Allocator \pi_{\theta}, Backbone \pi_{\phi}, operator \mathcal{O}
  1: for each prompt x do
           Sample M allocations: s_m \sim \pi_{\theta}(\cdot \mid x) via Beta policy (Eq. 10)
 2:
           for each allocation m = 1, ..., M do
 3:
                Apply operator: \tilde{x}^{(m)} = (q, \{\mathcal{O}(f_t, s_t^{(m)})\}_{t=1}^T)
  4:
                Sample N rollouts: \boldsymbol{y}_{m,n} \sim \pi_{\phi}(\cdot \mid \tilde{\boldsymbol{x}}^{(m)})
 5:
                Compute task reward R_{mn}^{\text{task}}
  6:
           end for
 7:
           Compute CAPO advantages A_{m,n} (Eqs. 15–17)
Aggregate per-allocation: A_m^{\text{CAPO}} = \frac{1}{N} \sum_n A_{m,n}
 8:
 9:
11: Update Allocator: minimize \mathcal{L}_{alloc} (Eq. 23)
                                                                                                                                           ▷ Omit if frozen
12: Update Backbone: minimize \mathcal{L}_{\phi} (Eq. 24)
```

### 4. Experiments

### 4.1. Setup

**Implementation.** The Allocator  $\pi_{\theta}$  uses the SmolVLM architecture (Marafioti et al., 2025) for high-throughput front-end prediction. Throughout, we instantiate input-side allocation with *resize*, so the learned allocations are realized as per-frame resize factors. We train the Allocator on Qwen2.5-VL-7B-Instruct (Bai et al., 2025b) and additionally test transfer to Qwen3-VL-8B-Instruct (Bai et al., 2025a). We report two settings: **ResAdapt-RL**, obtained by jointly updating the Allocator and the backbone, and **ResAdapt**, which directly reuses the trained Allocator with a frozen backbone to evaluate plug-and-play generalization. Resize is used during training because it provides the continuous action space required by our optimizer; thresholded frame selection is treated only as the conceptual zero-budget limit of the same pre-encoding interface. Full hyperparameters, hardware, prompts, and reward definitions are deferred to Appendix B.

**Baselines.** We compare against three classes of methods: **heuristic baselines** (Random Drop, FixedScale), **model-side compression** (ToMe (Bolya et al., 2022), FlashVid (Fan et al., 2026), VisionZip (Yang et al., 2025c)), and **reasoning-time inference augmentation** (VideoAuto-R1 (Liu et al., 2026)). We use visual-token retention ratio  $R^1$  as the primary budget descriptor and report the exact retained budget for every method. For reasoning-time baselines, R measures only visual encoder tokens; unless latency is reported separately, these comparisons should therefore be read as visual-budget comparisons rather than total-inference-budget matches. Because several baselines admit only discrete operating points, some comparisons are only approximately budget matched and should be interpreted relative to the explicit trade-offs shown in each table.

Benchmarks. For video QA, we report results on VideoMME (Fu et al., 2025a), LongVideoBench (Wu et al., 2024), MMVU (Zhao et al., 2025b), MLVU (Zhou et al., 2025), VideoMMMU (Hu et al., 2025), and LVBench (Wang et al., 2025b). For temporal grounding, we report Recall@{0.3,0.5,0.7} and mIoU on Charades-STA (Gao et al., 2017) and ActivityNet (Fabian et al., 2015), plus grounding QA on NExT-GQA (Xiao et al., 2024). For image understanding, we evaluate on MathVista (Lu et al., 2023), MMMU (Yue et al., 2024), OCRBench (Liu et al., 2024), ChartQA (Masry et al., 2022), AI2D (Kembhavi et al., 2016), and

<span id="page-9-1"></span> $<sup>{}^{1}</sup>R$  corresponds to  $\rho(s)$  in Sec. 3.3; we use R in tables for compactness.

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

**Figure 3:** Efficiency–accuracy trade-offs and temporal reallocation. (a,b) VideoMMMU and VideoMME versus visual-token retention ratio *R*. ResAdapt is on or near the Pareto frontier, with the clearest advantage on reasoning-heavy settings at low retention. (c) Relative gain from trading spatial resolution for temporal coverage under a fixed 8-frame-equivalent budget.

TextVQA (Singh et al., 2019). Unless stated otherwise, figures and analyses use Qwen2.5-VL-7B with 32 input frames. All evaluations use lmms-eval (Zhang et al., 2024a); the exact token budgets and decoding limits are reported in Appendix B.

#### <span id="page-10-2"></span>4.2. Main Results

### 4.2.1. Video QA

We first test whether input-side allocation via continuous resizing improves low-budget operating points, especially on reasoning-heavy benchmarks (Table 1).

**Disproportionate gains on multi-step reasoning.** Under aggressive compression (~10% retention), contentagnostic methods inevitably discard sparse but decisive evidence. On Qwen2.5-VL with 32 frames, ResAdapt achieves **45.7** on VideoMMMU at 11.4% retention, substantially outperforming ToMe (**39.2**), VisionZip (**39.1**), FlashVid (**39.4**), and FixedScale (**44.3**), while maintaining competitiveness on perception-focused benchmarks. The gap is largest on VideoMMMU, the most reasoning-intensive benchmark in the suite, confirming that input-side allocation selectively preserves the sparse visual evidence that multi-step reasoning demands. The transferred Allocator remains robust on Qwen3-VL, securing **56.1** on VideoMMMU at the same 11.4% retention, confirming cross-architecture generalizability.

**Spatial savings reinvested as temporal coverage.** Extending the context from 32 to 128 frames drastically amplifies this advantage. At 22.9% retention on Qwen2.5-VL, ResAdapt reaches **51.1** on VideoMMMU, exceeding the **47.9** achieved by the 128-frame uncompressed model while recovering near-optimal perception performance at a fraction of the visual cost. Even at 11.1% retention, ResAdapt attains **49.2**, again surpassing the uncompressed 128-frame score. This validates the central claim of input-side adaptation: spatial budget savings translate directly into temporal headroom, enabling the model to process  $4 \times$  more frames without the native-resolution compute penalty (Figure 3).

#### <span id="page-10-1"></span>4.2.2. Temporal Grounding

Temporal grounding is far more sensitive to compression than standard QA, since localization depends on fine-grained temporal cues rather than holistic scene understanding. Table 2 compares methods across comparable operating points.

<span id="page-11-0"></span>**Table 2: Temporal grounding results** across two backbones and two temporal horizons. Notation follows Table 1. See Sec. 4.2.2 for analysis of compression sensitivity.

|               |                                 | Retention |              | Temporal Grounding Benchmark |         |      |      |             |      |                   |      |        |               |  |
|---------------|---------------------------------|-----------|--------------|------------------------------|---------|------|------|-------------|------|-------------------|------|--------|---------------|--|
| Backbone      | Method                          | Ratio R   | Reasoning    | Charades-STA                 |         |      |      | ActivityNet |      |                   |      | xT-GQA |               |  |
|               |                                 |           |              | 0.3                          | 0.5     | 0.7  | mIoU | 0.3         | 0.5  | 0.7               | mIoU | Acc    | mIoU          |  |
|               |                                 |           |              | 3                            | 32 Fram | es   |      |             |      |                   |      |        |               |  |
|               | Vanilla                         | 100%      | Х            | 71.0                         | 51.4    | 26.0 | 47.3 | 30.4        | 18.0 | 8.9               | 22.6 | 78.9   | 28.0          |  |
|               | Random Drop                     | 25.0%     | x            | 39.4                         | 23.2    | 11.0 | 25.7 | 15.2        | 8.1  | 3.7               | 11.7 | 77.5   | 16.6          |  |
|               | ToMe (Bolya et al., 2022)       | 25.0%     | ×            | 39.5                         | 23.9    | 11.4 | 26.0 | 16.0        | 8.4  | 4.0               | 12.1 | 77.8   | 16.3          |  |
|               | FlashVid (Fan et al., 2026)     | 31.3%     | ×            | 40.7                         | 24.2    | 11.3 | 26.6 | 15.8        | 8.4  | 3.8               | 12.0 | 78.1   | 16.5          |  |
|               | FixedScale                      | 25.0%     | ×            | 36.7                         | 24.7    | 12.3 | 24.9 | 18.6        | 9.4  | 4.3               | 14.1 | 77.7   | 12.3          |  |
|               | ResAdapt (Ours)                 | 16.2%     | X            | 53.8                         | 34.8    | 17.0 | 35.6 | 19.8        | 10.8 | 5.2               | 15.3 | 76.6   | 23.2          |  |
|               | Random Drop                     | 10.0%     | <del>x</del> | 36.9                         | 23.2    | 11.6 | 24.6 | 14.3        | 7.5  | 3.6               | 11.1 | 76.3   | 15.4          |  |
|               | ToMe (Bolya et al., 2022)       | 10.0%     | ×            | 41.3                         | 26.9    | 14.1 | 27.4 | 16.0        | 8.4  | 4.0               | 12.2 | 77.3   | 15.7          |  |
|               | FlashVid (Fan et al., 2026)     | 12.6%     | ×            | 38.2                         | 22.9    | 11.1 | 25.1 | 15.4        | 8.1  | 3.7               | 11.8 | 77.4   | 16.1          |  |
| Д             | FixedScale                      | 12.3%     | ×            | 48.0                         | 31.5    | 15.4 | 32.0 | 17.5        | 8.9  | 4.0               | 13.3 | 76.1   | 13.7          |  |
| 7             | FixedScale                      | 6.3%      | ×            | 39.9                         | 26.8    | 13.3 | 26.7 | 15.2        | 8.1  | 3.9               | 11.9 | 74.1   | 15.4          |  |
| 7.            | ResAdapt (Ours)                 | 6.8%      | X            | 41.0                         | 27.8    | 14.0 | 27.2 | 16.3        | 8.5  | 3.9               | 12.5 | 74.3   | 20.4          |  |
| 2.5           | VideoAuto-R1 (Liu et al., 2026) | 100%      | /            | 60.0                         | 48.3    | 27.2 | 41.5 | 50.8        | 34.1 | 17.4              | 34.4 | 73.6   | 33.8          |  |
| Qwen2.5-VL-7B | + ResAdapt (Ours)               | 6.8%      | ✓            | 43.5                         | 30.1    | 15.8 | 30.0 | 35.4        | 21.5 | 10.0              | 24.4 | 74.7   | 24.7          |  |
| Ş             | 128 Frames                      |           |              |                              |         |      |      |             |      |                   |      |        |               |  |
|               | Vanilla                         | 100%      | Х            | 77.5                         | 60.3    | 34.1 | 52.8 | 47.9        | 30.9 | 17.5              | 34.4 | 79.8   | 29.9          |  |
|               | Random Drop                     |           | <del>x</del> | 32.3                         | 19.6    | 7.9  | 20.7 | 26.7        | 13.9 | 6.3               | 18.8 | 80.3   | 10.7          |  |
|               | ToMe (Bolya et al., 2022)       | 25.0%     | ×            | 32.4                         | 19.8    | 7.9  | 20.7 | 27.2        | 14.4 | 6.4               | 19.1 | 80.3   | 10.9          |  |
|               | ResAdapt (Ours)                 | 16.1%     | ×            | 63.5                         | 43.6    | 21.3 | 42.0 | 33.1        | 19.3 | 10.2              | 24.3 | 78.1   | 27.2          |  |
|               | Random Drop                     | 10.0%     | ×            | 37.8                         | 23.8    | 11.2 | 24.7 | 23.8        | 12.0 | 5.3               | 17.0 | 79.4   | 12.8          |  |
|               | ToMe (Bolya et al., 2022)       | 10.0%     | ×            | 27.9                         | 16.2    | 7.3  | 17.9 | 22.9        | 11.8 | 5.5               | 16.4 | 79.1   | 11.1          |  |
|               | FixedScale                      | 12.3%     | ×            | 34.7                         | 22.3    | 10.5 | 22.7 | 25.0        | 13.8 | 5.9               | 18.3 | 77.9   | 11.3          |  |
|               | FixedScale                      | 6.3%      | ×            | 42.6                         | 28.4    | 14.3 | 28.3 | 22.8        | 12.8 | 5.7               | 17.1 | 75.7   | 12.9          |  |
|               | ResAdapt (Ours)                 | 6.8%      | ×            | 43.5                         | 29.8    | 15.0 | 28.9 | 23.5        | 12.9 | 6.1               | 17.2 | 76.2   | 23.9          |  |
|               | VideoAuto-R1 (Liu et al., 2026) |           |              | 40.3                         | 33.7    | 22.1 | 28.9 | 49.4        | 34.3 | 18.5              | 33.5 | 68.0   | 31.0          |  |
|               | + ResAdapt (Ours)               | 16.1%     | ✓            | 72.8                         | 53.0    | 27.5 | 49.1 | 65.8        | 44.9 | 23.8              | 44.7 | 79.3   | 35.3          |  |
|               | + ResAdapt (Ours)               | 6.8%      | ✓            | 50.1                         | 33.2    | 16.6 | 34.2 | 53.4        | 34.0 | 16.4              | 35.7 | 76.6   | 29.4          |  |
|               |                                 |           |              | 3                            | 32 Fram | es   |      |             |      |                   |      |        |               |  |
|               | Vanilla                         | 100%      | Х            | 73.0                         | 49.0    | 21.4 | 46.4 | 44.6        | 28.3 | 15.5              | 31.8 | 78.7   | 34.2          |  |
|               | Random Drop                     |           | <del>x</del> | $16.\bar{2}$                 | 8.6     | 3.8  | 12.1 | 12.4        | 6.7  | 3.2               | 10.0 | 77.2   | _ <u>15.6</u> |  |
|               | ToMe (Bolya et al., 2022)       | 25.0%     | ×            | 68.7                         | 42.1    | 17.6 | 43.1 | 45.9        | 28.8 | 15.6              | 32.6 | 77.1   | 31.7          |  |
|               | FlashVid (Fan et al., 2026)     | 31.3%     | ×            | 72.9                         | 52.3    | 25.1 | 47.7 | 51.9        | 33.4 | 19.0              | 36.8 | 77.8   | 33.9          |  |
|               | ResAdapt (Ours)                 | 16.2%     | X            | 64.4                         | 37.3    | 16.3 | 39.9 | 40.0        | 24.4 | 13.0              | 28.5 | 75.1   | 30.2          |  |
|               | Random Drop                     | 10.0%     | ×            | 4.1                          | 1.8     | 0.7  | 4.4  | 4.7         | 2.4  | $\bar{1}.\bar{0}$ | 5.0  | 74.3   | 11.3          |  |
|               | ToMe (Bolya et al., 2022)       | 10.0%     | ×            | 67.6                         | 39.3    | 16.6 | 41.8 | 46.3        | 31.0 | 19.2              | 34.1 | 79.2   | 34.0          |  |
| 89            | FlashVid (Fan et al., 2026)     | 12.6%     | ×            | 68.8                         | 46.9    | 22.9 | 44.6 | 49.9        | 31.5 | 17.4              | 35.2 | 75.6   | 31.8          |  |
| ž             | FixedScale                      | 12.3%     | ×            | 61.3                         | 34.3    | 14.6 | 37.9 | 39.6        | 24.2 | 13.1              | 28.4 | 74.2   | 29.9          |  |
| 3-7           | FixedScale                      | 6.3%      | X            | 52.7                         | 28.2    | 11.3 | 33.2 | 37.0        | 22.3 | 12.0              | 27.0 | 71.5   | 28.0          |  |
| Qwen3-VL-8B   | ResAdapt (Ours)                 | 6.8%      | Х            | 53.6                         | 29.0    | 11.8 | 33.6 | 37.5        | 22.5 | 12.3              | 27.2 | 71.8   | 28.2          |  |
| Ó             |                                 |           |              | 1                            | 28 Fran | ıes  |      |             |      |                   |      |        |               |  |
|               | Vanilla                         | 100%      | <b>X</b>     | 72.8                         | 46.0    | 20.1 | 45.6 | 45.8        | 31.1 | 19.2              | 33.9 | 81.1   | 36.6          |  |
|               | Random Drop                     | 25.0%     | ×            | 41.6                         | 25.2    | 10.6 | 27.4 | 36.1        | 21.1 | 12.7              | 26.3 | 79.3   | 22.4          |  |
|               | ResAdapt (Ours)                 | 16.1%     | <del>X</del> | 64.4                         | 37.0    | 15.9 | 39.8 | 40.6        | 26.7 | 15.7              | 30.0 | 76.8   | 33.3          |  |
|               | Random Drop                     | 10.0%     | x            | 32.6                         | 19.0    | 7.8  | 21.9 | 33.5        | 18.6 | 11.5              | 24.8 | 76.9   | 19.9          |  |
|               | ToMe (Bolya et al., 2022)       | 10.0%     | X            | 61.6                         | 33.8    | 13.3 | 38.1 | 42.4        | 27.6 | 16.6              | 31.4 | 77.4   | 31.5          |  |
|               | FixedScale                      | 12.3%     | X            | 61.7                         | 34.9    | 14.7 | 38.1 | 39.9        | 26.2 | 15.3              | 29.5 | 75.4   | 32.6          |  |
|               | FixedScale                      | 6.3%      | X            | 53.7                         | 28.2    | 11.8 | 33.6 | 37.9        | 24.3 | 14.3              | 28.1 | 73.0   | 39.1          |  |
|               | ResAdapt (Ours)                 | 6.8%      | ×            | 54.3                         | 28.0    | 11.7 | 33.7 | 38.3        | 24.5 | 14.4              | 28.4 | 73.2   | 43.9          |  |

<span id="page-12-0"></span>Table 3: Latency breakdown (ms,  $\downarrow$ ) on Qwen2.5-VL-7B with single-GPU Allocator and 4-GPU vLLM engine. Averaged over 200 runs after 5 warm-up; E2E latency = Scale Time + Gen. Time.

| Method   | #Frames | Retention |        |              | Sca            | le             |                |               |                             | Inference             |                                    | To                          | otal                  |
|----------|---------|-----------|--------|--------------|----------------|----------------|----------------|---------------|-----------------------------|-----------------------|------------------------------------|-----------------------------|-----------------------|
|          |         | Ratio R   | TFLOPs | Text<br>Enc. | Visual<br>Enc. | Scale<br>Pred. | Scale<br>Apply | Scale<br>Time | TFLOPs                      | TTFT                  | Gen.<br>Time                       | TFLOPs                      | E2E<br>Time           |
| Vanilla  | 16      | 100%      | _      | _            | -              | -              | -              | -             | 111.4                       | 378.9                 | 527.9                              | 111.4                       | 527.9                 |
| ResAdapt | 16      | 76.3%     | 1.5    | 19.8         | 94.1           | 85.6           | 6.3            | 205.8         | 77.2 (\10.7%)               | 272.5 (\128.1%)       | 370.7 (129.8%)                     | 80.1 (\128.1%)              | 576.5 (†9.2%)         |
| ResAdapt | 16      | 52.8%     | 1.5    | 19.9         | 102.9          | 94.5           | 8.4            | 225.7         | 51.5 (\$53.8%)              | 261.5 (\\31.0%)       | 313.1 (\.40.7%)                    | 54.4 (\$1.2%)               | 538.8 (†2.1%)         |
| ResAdapt | 16      | 28.9%     | 1.5    | 20.4         | 103.4          | 92.2           | 9.0            | 225.0         | 31.0 (\172.2%)              | 227.2 (140.0%)        | 237.9 (\$54.9%)                    | 33.9 (\169.6%)              | 462.9 (\12.3%)        |
| Vanilla  | 32      | 100%      | -      | -            | -              | -              | -              | -             | 222.5                       | 723.3                 | 881.9                              | 222.5                       | 881.9                 |
| ResAdapt | 32      | 74.4%     | 2.9    | 19.9         | 204.1          | 97.4           | 14.4           | 335.9         | 153.9 (\$\pmu30.8\%)        | 589.4 (\18.5%)        | 627.6 (\128.8%)                    | 159.7 (\(\pmu28.2\%)        | 963.5 (†9.2%)         |
| ResAdapt | 32      | 51.5%     | 2.9    | 20.0         | 193.2          | 92.0           | 16.2           | 321.4         | 102.4 (\$54.0%)             | 505.0 (\10.2%)        | 467.1 (147.0%)                     | 108.2 (\$\frac{1}{51.4}\%)  | 788.5 (\10.6%)        |
| ResAdapt | 32      | 28.2%     | 2.9    | 20.3         | 190.4          | 90.3           | 17.3           | 318.3         | 61.4 (\172.4%)              | 451.8 (\137.5%)       | 332.6 (\(\daggerightarrow{62.3\%)} | 67.2 (\169.8%)              | 650.9 (\126.2%)       |
| Vanilla  | 64      | 100%      | -      | -            | -              | -              | -              | -             | 444.6                       | 1457.5                | 2059.6                             | 444.6                       | 2059.6                |
| ResAdapt | 64      | 73.2%     | 5.8    | 19.8         | 389.5          | 95.8           | 26.4           | 531.5         | 307.3 (\$\pmu30.9\%)        | 1093.1 (\125.0%)      | 1327.0 (\135.6%)                   | 318.9 (\(\pm28.3\%)         | 1858.5 (49.8%)        |
| ResAdapt | 64      | 50.7%     | 5.8    | 20.1         | 382.1          | 94.9           | 29.9           | 527.0         | 204.3 (\$54.0%)             | 991.8 (\31.9%)        | 740.5 (164.0%)                     | 215.9 (\$\psi_51.4\%)       | 1267.5 (\$\pmu38.5\%) |
| ResAdapt | 64      | 27.8%     | 5.8    | 20.0         | 371.6          | 90.2           | 34.8           | 516.6         | 122.2 (\$\pm\$72.5%)        | 899.2 (\38.3%)        | 511.4 (175.2%)                     | 133.8 (\$69.9%)             | 1028.0 (\$50.1%)      |
| Vanilla  | 128     | 100%      | _      | _            | -              | -              | _              | _             | 888.9                       | 2936.3                | 4877.0                             | 888.9                       | 4877.0                |
| ResAdapt | 128     | 74.2%     | 11.6   | 20.1         | 766.3          | 95.0           | 53.1           | 934.5         | 614.1 (\130.9%)             | 2286.6 (\122.1%)      | 2323.6 (\$52.4%)                   | 637.3 (\128.3%)             | 3258.1 (\133.2%)      |
| ResAdapt | 128     | 51.4%     | 11.6   | 20.2         | 755.3          | 93.8           | 59.4           | 928.7         | 408.0 (\$54.1%)             | 2071.0 (\129.5%)      | 1496.0 (\.001469.3\%)              | 431.2 (\$\psi_51.5\%)       | 2424.7 (\$\pm\$50.3%) |
| ResAdapt | 128     | 28.2%     | 11.6   | 20.4         | 734.5          | 92.0           | 68.6           | 915.5         | $243.9~(\downarrow 72.6\%)$ | 1766.7 (\$\pm\$39.8%) | $1061.8 \; (\downarrow 78.2\%)$    | $267.1~(\downarrow 70.0\%)$ | 1977.3 (\$59.5%)      |

**Pre-encoding allocation dominates frame dropping.** On Qwen2.5-VL (32F), Random Drop, ToMe, FlashVid, and FixedScale severely degrade Charades-STA mIoU from **47.3** to **25.7**, **26.0**, **26.6**, and **24.9**, respectively, at  $\approx$ 25–31% retention. In contrast, operating at a strictly lower **16.2**% budget, ResAdapt preserves an mIoU of **35.6**. Allocating pixels *before* encoding—rather than dropping frames or pruning tokens post-hoc—confers robustness that these baselines cannot match, even at tighter budgets.

**Reasoning without temporal anchors regresses.** On VideoAuto-R1 (Qwen2.5-VL), naively extending from 32 to 128 frames *degrades* Charades-STA mIoU from **41.5** to **28.9**: longer reasoning chains cannot compensate for the diluted temporal signal that accompanies quadratically growing token sequences. Incorporating ResAdapt at 16.1% retention raises the 128-frame score to **49.1**, demonstrating that input-side allocation rescues long-context reasoning by concentrating visual budget on the temporally decisive frames.

**Emergent denoising.** On NExT-GQA (Qwen3-VL, 128F), ResAdapt improves mIoU from **36.6** to **43.9** at 6.8% retention. Aggressively suppressing question-irrelevant frames sharpens localization: removing noise is itself a form of signal enhancement.

### 4.2.3. Image-Task Transfer

We evaluate ResAdapt on static image benchmarks to characterize domain boundaries. The clearest positive result is ChartQA on Qwen2.5-VL, where the Allocator upscales chart-bearing images to 105% of the native budget. However, text-heavy tasks degrade once resolution drops below their evidence threshold. Full results appear in Appendix C.4; we treat image transfer as a boundary condition rather than a primary contribution.

### <span id="page-12-1"></span>4.3. Runtime Overhead

Table 3 measures pipeline latency using a dedicated single-GPU Allocator and a separate 4-GPU vLLM engine. The key trade-off is when downstream token savings amortize the front-end allocation cost.

At  $R\approx74\%$ , generation time drops **29–52**% but end-to-end (E2E) savings appear only at  $\geq$ 64 frames (**–9.8**%), growing to **–33.2**% at 128 frames. At  $R\approx51\%$ , break-even shifts to 32 frames (**–10.6**% E2E); at  $R\approx28\%$ , wall-clock savings emerge even at 16 frames (**–12.3**%), reaching **–59.5**% at 128 frames with

<span id="page-13-0"></span>![](_page_13_Figure_1.jpeg)

**Figure 4: Emergent active perception.** Per-frame scale  $s_t$  over frame index for six VideoMME videos, grouped by intra-video scale diversity  $\sigma$ . High-diversity videos show localized scale spikes on scene changes, text overlays, and rapid motion; low-diversity videos remain near-uniform.

**78**% generation-time reduction. Backbone savings compound faster than the fixed Allocator overhead as sequences grow—a direct consequence of the quadratic attention cost—making ResAdapt most impactful in the long-context regime.

### <span id="page-13-1"></span>4.4. Analysis

### 4.4.1. Emergent Active Perception

Figures 4 and 5 reveal the mechanism behind the main results. The Allocator does not learn a uniform compression policy; rather, it learns a strongly sparse temporal allocation, keeping most of the video at near-minimum resolution and concentrating budget on short bursts around text overlays, scene transitions, or other brief informative events.

This behavior is not a trivial positional prior. The median scale stays close to the low end of the range, while the mean is lifted by localized peaks—indicating that high-resolution allocation is the exception, not the default. The per-video heatmap further confirms that these peaks are content-driven segments, not a fixed bias toward the beginning or end of the clip. In short, the policy spends pixels where the answer is likely to be decided.

### 4.4.2. Ablation Studies

**CAPO reward design.** Two questions arise: *how* should cost enter the optimization, and *what* prevents the policy from collapsing to a uniform scaler?

Table 4 shows that the exact policy family is secondary:  $\beta$ -CAPO and  $\mathcal{N}$ -CAPO trade marginal advantages

<span id="page-14-0"></span>![](_page_14_Figure_1.jpeg)

**Figure 5: Global allocation statistics on VideoMME. (a)** Aggregate predicted scale by frame position. **(b)** Case×frame heatmap for the first 200 videos. High-scale allocation appears as localized bursts rather than a fixed positional pattern.

<span id="page-14-1"></span>**Table 4: Distribution family ablation for CAPO.** The two variants follow the same training protocol.

| Variant                   | Ī    | VideoMME | LongVideoBench | MMVU | V    | VideoMMMU |       | LVBench |
|---------------------------|------|----------|----------------|------|------|-----------|-------|---------|
|                           |      |          |                |      | Per. | Comp.     | Adap. |         |
| β-CAPO                    |      | 60.3     | 58.2           | 51.2 | 65.0 | 54.3      | 28.7  | 37.6    |
| $\mathcal{N}\text{-CAPO}$ | 0.60 | 61.0     | 57.4           | 51.8 | 66.0 | 50.0      | 30.3  | 37.2    |

across benchmarks with neither variant consistently dominating. The shared ingredient that matters is CAPO's asymmetric cost shaping, not the parametric form. Figure 6 makes this more explicit from a training-dynamics perspective. Direct cost penalties drive the policy rapidly toward the minimum-scale boundary, while removing cost altogether pushes toward the upper bound. CAPO stabilizes an intermediate operating point where the model is rewarded for being selective—not merely cheap and not merely accurate. Further analysis of per-sample adaptivity and convergence appears in Appendix C.2.

Operator generalization. Although ResAdapt is trained exclusively for adaptive resizing, its learned policy transfers zeroshot to frame selection. We repurpose the Allocator's predicted scales as importance scores to rank and filter 128 candidate frames. Table 5 shows that selecting and resizing the top-32 or top-64 frames consistently outperforms the vanilla 16-frame and 32-frame baselines, respectively, despite consuming fewer tokens. The policy thus captures an operator-agnostic notion of visual importance that generalizes beyond the training-time operator.

<span id="page-14-2"></span>**Table 5: Operator generalization.** Zero-shot transfer of ResAdapt scores to frame selection. Combining top-*K* selection with adaptive resizing from 128 candidate frames outperforms uniform sampling baselines at lower token budgets.

| Method                        | VideoMME      | LongVideoBench | LVBench       | MMVU          |
|-------------------------------|---------------|----------------|---------------|---------------|
| Budget: 8 frames              |               |                |               |               |
| Vanilla                       | 54.0          | 53.9           | 33.3          | 48.9          |
| Top-8 Select                  | 52.2          | 51.1           | 32.0          | 49.2          |
| Budget: 16 frames             |               |                |               |               |
| Vanilla                       | 58.9          | 56.0           | 36.1          | 50.9          |
| Threshold Select              | 58.0          | 57.4           | 36.4          | 51.0          |
| Avg. Budget (Retention Ratio) | 12.2f (9.5%)  | 23.2f (18.1%)  | 16.7f (13.0%) | 17.2f (13.4%) |
| Top-32 Select + Resize        | 60.6          | 57.2           | 38.9          | 50.2          |
| Avg. Budget (Retention Ratio) | 11.7f (9.1%)  | 16.9f (13.2%)  | 13.7f (10.7%) | 14.1f (11.0%) |
| Budget: 32 frames             |               |                |               |               |
| Vanilla                       | 62.3          | 58.7           | 39.5          | 52.0          |
| Top-32 Select                 | 59.7          | 55.7           | 37.0          | 51.2          |
| Top-64 Select + Resize        | 62.5          | 58.4           | 40.0          | 52.3          |
| Avg. Budget (Retention Ratio) | 23.8f (18.6%) | 36.2f (28.3%)  | 24.1f (18.8%) | 32.5f (25.4%) |

<span id="page-15-0"></span>![](_page_15_Figure_1.jpeg)

**Figure 6: Reward-design ablation.** Mean predicted scale *s*¯ during training and validation. Direct cost penalties collapse to the minimum scale, whereas CAPO variants converge to stable intermediate operating points.

**Temporal regularization complements CAPO.** CAPO determines how cost enters the learning signal, but it does not by itself force the Allocator to distinguish among visually redundant neighbors. Figure [7](#page-16-0) shows that removing Lsim collapses the scale trace toward a near-constant profile resembling FixedScale. Reintroducing Lsim restores sharp frame-level differentiation. The two mechanisms are complementary: CAPO stabilizes the accuracy–cost operating point, while Lsim breaks the symmetry that otherwise favors uniform allocation.

### *4.4.3. Robustness and Limitations*

Adaptive allocation is not a lossless compression layer. ResAdapt preserves a large majority of originally correct predictions, but can still miss decisive evidence when the relevant cue appears briefly against a simple background. Because the policy is open-loop—budget decisions are committed before any backbone processing begins—it cannot revise allocations once reasoning starts. We therefore interpret the performance gains as selective redistribution of visual budget rather than guaranteed preservation of all useful information. Detailed failure-case analysis is provided in Appendix [C.3.](#page-31-0)

# **5. Related Work**

**Input-side adaptation before visual encoding.** A growing body of work reduces visual cost *before* or *during* input construction. Early approaches primarily perform temporal downsampling through keyframe selection or clip condensation [\(Liang et al.,](#page-19-5) [2024,](#page-19-5) [Zhu et al.,](#page-23-7) [2025,](#page-23-7) [Sun et al.,](#page-21-5) [2025,](#page-21-5) [Tang et al.,](#page-21-6) [2025\)](#page-21-6). More recent methods incorporate query awareness and iterative search, tailoring frame selection to question types or intermediate evidence [\(Zou et al.,](#page-23-8) [2025,](#page-23-8) [Li et al.,](#page-19-6) [2025a,](#page-19-6) [Guo et al.,](#page-19-7) [2025b,](#page-19-7) [He et al.,](#page-19-8) [2025\)](#page-19-8). Beyond selecting *which* frames to process, several works allocate perceptual budgets via multi-resolution encoding. Slow–fast pipelines [\(Yang et al.,](#page-22-3) [2025a,](#page-22-3) [Zhang et al.,](#page-22-0) [2026\)](#page-22-0) use inter-frame similarity to route frames to highor low-resolution paths, but their binary, query-agnostic routing cannot adapt to the downstream question; we omit direct experimental comparisons because these systems target different backbone families and evaluation protocols. Query-aware multi-resolution strategies [\(Zhang et al.,](#page-23-9) [2025d\)](#page-23-9) and early truncation of less informative visual tokens [\(Chen et al.,](#page-18-7) [2026\)](#page-18-7) go further by conditioning on the query, yet still rely on handcrafted rules or fixed resolution bins. We note that QFrame [\(Zhang et al.,](#page-23-9) [2025d\)](#page-23-9) operates with

<span id="page-16-0"></span>![](_page_16_Figure_1.jpeg)

Figure 7:  $\mathcal{L}_{sim}$  ablation: per-frame scale profiles. Without temporal-similarity regularization, the Allocator approaches near-uniform scaling; with it, the policy concentrates resolution on selected frames and suppresses redundant neighbors.

predefined resolution tiers and rule-based selection, whereas ResAdapt learns continuous allocations end-toend from task reward; a controlled comparison would require re-implementing QFrame's proprietary routing logic on our backbone, which we leave for future work. In contrast, ResAdapt is an Input-side adaptation framework: it learns input-side allocations from task reward via RL and can realize them through different pre-encoding operators, including resizing and frame selection; the experiments in this paper study the continuous resize instantiation.

Model-side token economy after encoding. Post-encoding methods prune, merge, or redistribute visual tokens in embedding space. For images, representative approaches include token merging (Bolya et al., 2022), attention- or saliency-guided pruning (Chen et al., 2024, Yang et al., 2025c, Shang et al., 2025, Zhang et al., 2025c), progressive dropping (Xing et al., 2024, Zhang et al., 2024b), context compression (Liao et al., 2025a), KV cache sparsity (Liao et al., 2025b), and diversity-based budget allocation (Alvar et al., 2025, Yang et al., 2025b, Zhang et al., 2025a). Video-specific extensions exploit spatiotemporal redundancy via static/dynamic token separation (Huang et al., 2025, Shen et al., 2025a), hierarchical merging (Hyun et al., 2025), and segment-level fusion or budget allocation (Tao et al., 2025, Fu et al., 2024, Shao et al., 2025a). These methods are complementary to ResAdapt: they operate *after* visual encoding and cannot recover high-frequency details lost to undersampling before encoding. Our focus is earlier in the pipeline, deciding how many pixels to encode in the first place.

**Output-side agentic reasoning.** Another strategy leaves the input fixed and recovers efficiency through iterative reasoning: retrieve candidate frames, zoom into regions, then re-query the model. Approaches range from static toolsets with predefined cropping or clipping operators (Zheng et al., 2025b, Wang et al., 2025a, Song et al., 2026) to dynamic tooling via code-generation primitives (Zhang et al., 2025e, Zhao et al., 2025a, Hong et al., 2025), often exposed through executable interfaces (Wang et al., 2024). While these methods can target hard evidence precisely, they are multi-pass by construction and rely on an initial coarse view to trigger subsequent refinement. ResAdapt instead studies whether a *single-pass* pre-encoding allocation policy can recover much of this benefit without the latency and control overhead of iterative interaction.

**RL for multimodal reasoning and perception control.** Recent work has extended RL post-training from language models [\(Shao et al.,](#page-21-11) [2024,](#page-21-11) [Guo et al.,](#page-19-0) [2025a,](#page-19-0) [Tan et al.,](#page-21-12) [2025\)](#page-21-12) to multimodal reasoning and video understanding. Algorithmic refinements include improved advantage estimation and PPO-style stabilization [\(Liu et al.,](#page-20-11) [2025c,](#page-20-11) [Yu et al.,](#page-22-4) [2025,](#page-22-4) [Zheng et al.,](#page-23-3) [2025a\)](#page-23-3), while video-domain extensions strengthen reasoning through iterative frame selection and evidence refinement [\(Feng et al.,](#page-18-10) [2025,](#page-18-10) [Li et al.,](#page-19-13) [2025b,](#page-19-13) [Liu et al.,](#page-20-3) [2026,](#page-20-3) [Yang et al.,](#page-22-2) [2025d,](#page-22-2) [Chen et al.,](#page-18-11) [2025,](#page-18-11) [Wang et al.,](#page-22-12) [2025c,](#page-22-12) [Fu et al.,](#page-18-12) [2025b\)](#page-18-12). Our use of RL is orthogonal: we apply it to *input-side perception control*—learning frame-level visual allocations under an explicit accuracy–cost trade-off—rather than output-side reasoning policies. CAPO is designed for this setting, where naive cost penalties drive the policy to a degenerate low-budget solution.

# **6. Conclusion**

Mainstream video MLLM efficiency methods compress tokens *after* encoding, forcing models to absorb the full computational cost of dense pixel processing before realizing any savings. ResAdapt dismantles this bottleneck. By introducing a query-aware Allocator, our framework assigns per-frame visual budgets *prior* to feature extraction. The vision encoder therefore processes a strictly task-relevant, sparse input, leaving native kernel optimizations entirely untouched. To train this non-differentiable pipeline, we propose CAPO. This asymmetric reward formulation heavily penalizes wasteful allocations, while a temporal-similarity regularizer breaks uniform-resolution equilibria, forcing the policy to differentiate among visually redundant neighbors.

This input-side adaptation fundamentally alters the efficiency-accuracy tradeoff. Operating at merely ∼10% visual-token retention, ResAdapt matches or exceeds the performance of dense baselines on complex reasoning tasks. Crucially, it translates these spatial savings directly into expanded temporal coverage. Under equivalent compute, the model processes up to 16× more frames, yielding >15% relative gains on long-video benchmarks. Beyond macroscopic metrics, the learned policy demonstrates emergent active perception. Driven entirely by task-level rewards, it concentrates high resolution strictly on the sparse frames required to resolve the query.

Two boundaries remain. The current allocation mechanism operates open-loop: once the model commits a resolution budget, it cannot dynamically revise this decision during reasoning. Moreover, the specific instantiation of spatial resizing transfers unevenly to non-video modalities. Closing this loop defines the next critical frontier. By allowing intermediate backbone states to trigger selective re-encoding, future work can transform static budget prediction into a fully adaptive, reasoning-aware allocation paradigm.

# **References**

<span id="page-17-1"></span>Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 9392–9401, 2025.

<span id="page-17-0"></span>Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu,

- Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025a. URL <https://arxiv.org/abs/2511.21631>.
- <span id="page-18-3"></span>Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025b.
- <span id="page-18-0"></span>Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. *arXiv preprint arXiv:2210.09461*, 2022.
- <span id="page-18-8"></span>Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In *European Conference on Computer Vision*, pages 19–35. Springer, 2024.
- <span id="page-18-11"></span>Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, et al. Scaling rl to long videos. *arXiv preprint arXiv:2507.07966*, 2025.
- <span id="page-18-7"></span>Zeyuan Chen, Kai Zhang, Zhuowen Tu, and Yuanjun Xiong. Soft tail-dropping for adaptive visual tokenization. *arXiv preprint arXiv:2601.14246*, 2026.
- <span id="page-18-1"></span>Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In *International Conference on Learning Representations*, 2024.
- <span id="page-18-6"></span>Caba Heilbron Fabian, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In *Proceedings of the ieee conference on computer vision and pattern recognition*, pages 961–970, 2015.
- <span id="page-18-2"></span>Ziyang Fan, Keyu Chen, Ruilong Xing, Yulin Li, Li Jiang, and Zhuotao Tian. Flashvid: Efficient video large language models via training-free tree-based spatiotemporal token merging. *arXiv preprint arXiv:2602.08024*, 2026.
- <span id="page-18-10"></span>Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. *arXiv preprint arXiv:2503.21776*, 2025.
- <span id="page-18-4"></span>Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 24108–24118, 2025a.
- <span id="page-18-12"></span>Shenghao Fu, Qize Yang, Yuan-Ming Li, Xihan Wei, Xiaohua Xie, and Wei-Shi Zheng. Love-r1: Advancing long video understanding with an adaptive zoom-in mechanism via multi-step reasoning. *arXiv preprint arXiv:2509.24786*, 2025b.
- <span id="page-18-9"></span>Tianyu Fu, Tengxuan Liu, Qinghao Han, Guohao Dai, Shengen Yan, Huazhong Yang, Xuefei Ning, and Yu Wang. Framefusion: Combining similarity and importance for video token reduction on large vision language models. *arXiv preprint arXiv:2501.01986*, 2024.
- <span id="page-18-5"></span>Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In *Proceedings of the IEEE international conference on computer vision*, pages 5267–5275, 2017.

- <span id="page-19-0"></span>Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025a.
- <span id="page-19-7"></span>Weiyu Guo, Ziyang Chen, Shaoguang Wang, Jianxiang He, Yijie Xu, Jinhui Ye, Ying Sun, and Hui Xiong. Logicin-frames: Dynamic keyframe search via visual semantic-logical verification for long video understanding. *arXiv preprint arXiv:2503.13139*, 2025b.
- <span id="page-19-8"></span>Zefeng He, Xiaoye Qu, Yafu Li, Siyuan Huang, Daizong Liu, and Yu Cheng. Framethinker: Learning to think with long videos via multi-turn frame spotlighting. *arXiv preprint arXiv:2509.24304*, 2025.
- <span id="page-19-12"></span>Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, and Xing Yu. Deepeyesv2: Toward agentic multimodal model. *arXiv preprint arXiv:2511.05271*, 2025.
- <span id="page-19-3"></span>Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Videommmu: Evaluating knowledge acquisition from multi-discipline professional videos. *arXiv preprint arXiv:2501.13826*, 2025.
- <span id="page-19-10"></span>Xiaohu Huang, Hao Zhou, and Kai Han. Prunevid: Visual token pruning for efficient video large language models. In *Findings of the Association for Computational Linguistics: ACL 2025*, pages 19959–19973, 2025.
- <span id="page-19-11"></span>Jeongseok Hyun, Sukjun Hwang, Su Ho Han, Taeoh Kim, Inwoong Lee, Dongyoon Wee, Joon-Young Lee, Seon Joo Kim, and Minho Shim. Multi-granular spatio-temporal token merging for training-free acceleration of video llms. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 23990– 24000, 2025.
- <span id="page-19-4"></span>Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *European conference on computer vision*, pages 235–251. Springer, 2016.
- <span id="page-19-1"></span>Samir Khaki, Junxian Guo, Jiaming Tang, Shang Yang, Yukang Chen, Konstantinos N Plataniotis, Yao Lu, Song Han, and Zhijian Liu. Sparsevila: Decoupling visual sparsity for efficient vlm inference. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 23784–23794, 2025.
- <span id="page-19-2"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- <span id="page-19-6"></span>Jialuo Li, Bin Li, Jiahao Li, and Yan Lu. Divide, then ground: Adapting frame selection to query types for long-form video understanding. *arXiv preprint arXiv:2512.04000*, 2025a.
- <span id="page-19-13"></span>Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. *arXiv preprint arXiv:2504.06958*, 2025b.
- <span id="page-19-5"></span>Hao Liang, Jiapeng Li, Tianyi Bai, Xijie Huang, Linzhuang Sun, Zhengren Wang, Conghui He, Bin Cui, Chong Chen, and Wentao Zhang. Keyvideollm: Towards large-scale video keyframe selection. *arXiv preprint arXiv:2407.03104*, 2024.
- <span id="page-19-9"></span>Huanxuan Liao, Wen Hu, Yao Xu, Shizhu He, Jun Zhao, and Kang Liu. Beyond hard and soft: Hybrid context compression for balancing local and global information retention. *arXiv preprint arXiv:2505.15774*, 2025a.

- <span id="page-20-9"></span>Huanxuan Liao, Yixing Xu, Shizhu He, Guanchen Li, Xuanwu Yin, Dong Li, Emad Barsoum, Jun Zhao, and Kang Liu. Spark: Query-aware unstructured sparsity with recoverable kv cache channel pruning. *arXiv preprint arXiv:2508.15212*, 2025b.
- <span id="page-20-0"></span>Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context language modeling. *arXiv preprint arXiv:2503.17407*, 2025a.
- <span id="page-20-3"></span>Shuming Liu, Mingchen Zhuge, Changsheng Zhao, Jun Chen, Lemeng Wu, Zechun Liu, Chenchen Zhu, Zhipeng Cai, Chong Zhou, Haozhe Liu, et al. Videoauto-r1: Video auto reasoning via thinking once, answering twice. *arXiv preprint arXiv:2601.05175*, 2026.
- <span id="page-20-6"></span>Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. *Science China Information Sciences*, 67(12):220102, 2024.
- <span id="page-20-2"></span>Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 4122–4134, 2025b.
- <span id="page-20-11"></span>Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. *arXiv preprint arXiv:2503.20783*, 2025c.
- <span id="page-20-5"></span>Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. *arXiv preprint arXiv:2310.02255*, 2023.
- <span id="page-20-4"></span>Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. *arXiv preprint arXiv:2504.05299*, 2025.
- <span id="page-20-7"></span>Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In *Findings of the association for computational linguistics: ACL 2022*, pages 2263–2279, 2022.
- <span id="page-20-12"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining*, pages 3505–3506, 2020.
- <span id="page-20-8"></span>Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 22857–22867, 2025.
- <span id="page-20-10"></span>Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Holitom: Holistic token merging for fast video large language models. *arXiv preprint arXiv:2505.21334*, 2025a.
- <span id="page-20-1"></span>Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. When tokens talk too much: A survey of multimodal long-context token compression across images, videos, and audios. *arXiv preprint arXiv:2507.20198*, 2025b.

- <span id="page-21-11"></span>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.
- <span id="page-21-7"></span>Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, and Guiguang Ding. Fastvid: Dynamic density pruning for fast video large language models. *arXiv preprint arXiv:2503.11187*, 2025a.
- <span id="page-21-2"></span>Xiaoqian Shen, Min-Hung Chen, Yu-Chiang Frank Wang, Mohamed Elhoseiny, and Ryo Hachiuma. Zoom-zero: Reinforced coarse-to-fine video understanding via temporal zoom-in. *arXiv preprint arXiv:2512.14273*, 2025b.
- <span id="page-21-13"></span>Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In *Proceedings of the Twentieth European Conference on Computer Systems*, pages 1279–1297, 2025.
- <span id="page-21-0"></span>Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 26160–26169, 2025.
- <span id="page-21-4"></span>Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8317–8326, 2019.
- <span id="page-21-9"></span>Mingyang Song, Haoyu Sun, Jiawei Gu, Linjie Li, Luxin Xu, Ranjay Krishna, and Yu Cheng. Adareasoner: Dynamic tool orchestration for iterative visual reasoning. *arXiv preprint arXiv:2601.18631*, 2026.
- <span id="page-21-5"></span>Guangyu Sun, Archit Singhal, Burak Uzkent, Mubarak Shah, Chen Chen, and Garin Kessler. From frames to clips: Training-free adaptive key clip selection for long-form video understanding. *arXiv preprint arXiv:2510.02262*, 2025.
- <span id="page-21-12"></span>Yuqiao Tan, Minzheng Wang, Shizhu He, Huanxuan Liao, Chengfeng Zhao, Qiunan Lu, Tian Liang, Jun Zhao, and Kang Liu. Bottom-up policy optimization: Your language model policy secretly contains internal policies. *arXiv preprint arXiv:2512.19673*, 2025.
- <span id="page-21-6"></span>Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. *arXiv preprint arXiv:2502.21271*, 2025.
- <span id="page-21-1"></span>Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 18992–19001, 2025.
- <span id="page-21-8"></span>Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. *arXiv preprint arXiv:2505.15966*, 2025a.
- <span id="page-21-3"></span>Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 22958–22967, 2025b.
- <span id="page-21-10"></span>Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents. In *Forty-first International Conference on Machine Learning*, 2024.

- <span id="page-22-12"></span>Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. *arXiv preprint arXiv:2503.13377*, 2025c.
- <span id="page-22-6"></span>Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. *Advances in Neural Information Processing Systems*, 37:28828–28857, 2024.
- <span id="page-22-7"></span>Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13204–13214, 2024.
- <span id="page-22-9"></span>Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. *arXiv preprint arXiv:2410.17247*, 2024.
- <span id="page-22-1"></span>Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. *arXiv preprint arXiv:2510.09608*, 2025.
- <span id="page-22-3"></span>Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, Chengru Song, Chongling Rao, Chuan Yi, Da Li, Dunju Zang, et al. Kwai keye-vl 1.5 technical report. *arXiv preprint arXiv:2509.01563*, 2025a.
- <span id="page-22-10"></span>Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 19803–19813, 2025b.
- <span id="page-22-5"></span>Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19792–19802, 2025c.
- <span id="page-22-2"></span>Zuhao Yang, Sudong Wang, Kaichen Zhang, Keming Wu, Sicong Leng, Yifan Zhang, Bo Li, Chengwei Qin, Shijian Lu, Xingxuan Li, and Lidong Bing. Longvt: Incentivizing "thinking with long videos" via native tool calling. *arXiv preprint arXiv:2511.20785*, 2025d.
- <span id="page-22-4"></span>Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025.
- <span id="page-22-8"></span>Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9556–9567, 2024.
- <span id="page-22-0"></span>Boqiang Zhang, Lei Ke, Ruihan Yang, Qi Gao, Tianyuan Qu, Rossell Chen, Dong Yu, et al. Penguin-vl: Exploring the efficiency limits of vlm with llm-based vision encoders. *arXiv preprint arXiv:2603.06569*, 2026.
- <span id="page-22-11"></span>Ce Zhang, Kaixin Ma, Tianqing Fang, Wenhao Yu, Hongming Zhang, Zhisong Zhang, Yaqi Xie, Katia Sycara, Haitao Mi, and Dong Yu. Vscan: Rethinking visual token reduction for efficient large vision-language models. *arXiv preprint arXiv:2505.22654*, 2025a.

- <span id="page-23-1"></span>Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, and Bo Zheng. Rewatch-r1: Boosting complex video reasoning in large vision-language models through agentic data synthesis. *arXiv preprint arXiv:2509.23652*, 2025b.
- <span id="page-23-6"></span>Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024a. URL <https://arxiv.org/abs/2407.12772>.
- <span id="page-23-10"></span>Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 20857–20867, 2025c.
- <span id="page-23-9"></span>Shaojie Zhang, Jiahui Yang, Jianqin Yin, Zhenbo Luo, and Jian Luan. Q-frame: Query-aware frame selection and multi-resolution adaptation for video-llms. *arXiv preprint arXiv:2506.22139*, 2025d.
- <span id="page-23-12"></span>Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. *arXiv preprint arXiv:2508.11630*, 2025e.
- <span id="page-23-11"></span>Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. *arXiv preprint arXiv:2410.04417*, 2024b.
- <span id="page-23-13"></span>Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. *arXiv preprint arXiv:2507.07998*, 2025a.
- <span id="page-23-4"></span>Yilun Zhao, Haowei Zhang, Lujing Xie, Tongyan Hu, Guo Gan, Yitao Long, Zhiyuan Hu, Weiyuan Chen, Chuhan Li, Zhijian Xu, et al. Mmvu: Measuring expert-level multi-discipline video understanding. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 8475–8489, 2025b.
- <span id="page-23-3"></span>Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. *arXiv preprint arXiv:2507.18071*, 2025a.
- <span id="page-23-0"></span>Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. In *Advances in Neural Information Processing Systems*, 2024.
- <span id="page-23-2"></span>Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. *arXiv preprint arXiv:2505.14362*, 2025b.
- <span id="page-23-5"></span>Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. Mlvu: Benchmarking multi-task long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13691–13701, 2025.
- <span id="page-23-7"></span>Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, and Yang You. Focus: Efficient keyframe selection for long video understanding. *arXiv preprint arXiv:2510.27280*, 2025.
- <span id="page-23-8"></span>Yuanhao Zou, Shengji Jin, Andong Deng, Youpeng Zhao, Jun Wang, and Chen Chen. Air: Enabling adaptive, iterative, and reasoning-based frame selection for video question answering. *arXiv preprint arXiv:2510.04428*, 2025.

# **Limitations and Future Work**

ResAdapt substantially advances the efficiency–accuracy Pareto front for long-video MLLMs. Nonetheless, four specific design choices constrain our current findings.

- *(i) Front-end overhead is amortized only in the long-context regime.* The Allocator imposes a fixed pre-encoding cost—comprising coarse visual encoding, cross-frame fusion, and distribution prediction—before realizing any backbone savings. For short sequences (*T* ≤32), this constant overhead offsets a significant fraction of the downstream attention reduction. Consequently, definitive wall-clock speedups emerge primarily in the long-context regime (Sec. [4.3\)](#page-12-1). Future work must reduce this fixed cost via cached video features, lightweight front-ends, or distilled allocation rules.
- *(ii) Allocation relies on coarse visual evidence.* The Allocator operates on frozen coarse features *f <sup>t</sup>* ∈ **R***<sup>D</sup>* rather than full high-resolution frames. This representation suffices to detect broad redundancies and scene structures. However, it struggles with small text, subtle objects, and transient answer-critical cues embedded within otherwise simple frames (Figure [20\)](#page-40-0). Incorporating multi-scale conditioning, motion-aware features, or lightweight local refinement could bridge this gap without sacrificing the speed of the current front-end.
- *(iii) Single video-centric instantiation limits broad validation.* While we formulate ResAdapt as a general input-side adaptation framework, our experiments instantiate the operator strictly as resizing and train the policy primarily on video tasks. Transfer beyond this regime remains uneven. The learned policy occasionally identifies static images requiring higher fidelity, yet fails to deliver uniformly efficiency-preserving gains on image-centric benchmarks (Table [7\)](#page-34-0). Extending the training mixture to image–video data and exploring alternative operators, such as hard frame selection, remain open problems.
- *(iv) Open-loop allocation ignores reasoning state.* The framework commits all budget decisions before the backbone processes any visual tokens. The policy cannot revise a mistaken low-resolution choice once partial reasoning or uncertainty signals emerge. Closing this loop represents a natural extension. Early backbone states could trigger re-encoding, budget revision, or a secondary visual pass only when strictly necessary.

# **Software and Data**

The code for this paper is available at: <https://github.com/Xnhyacinth/ResAdapt>

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

# <span id="page-28-1"></span>**B. Implementation Details**

### **B.1. Training Data**

**Data Composition.** We construct the training corpus from the difficulty-filtered VideoAuto-R1 [\(Liu et al.,](#page-20-3) [2026\)](#page-20-3) dataset, strictly retaining image and video samples while discarding pure-text examples. To guarantee robust coverage of visually demanding subdomains, we inject 16,500 high-complexity video instances from Video-R1 [\(Feng et al.,](#page-18-10) [2025\)](#page-18-10), prioritizing OCR, free-form QA, and regression-style tasks. The finalized pool comprises approximately 93.4K training samples. We rigorously purge all evaluation examples from this corpus to preclude data leakage.

### **B.2. Training Configuration**

We train the models for one epoch using AdamW with a global batch size of 128. We apply a learning rate of 2 × 10−<sup>5</sup> to the Allocator and 1 × 10−<sup>6</sup> to the backbone. We enforce a weight decay of 0.01 and cap gradient clipping at 1.0. We constrain the maximum video token budget to 8,192, sample *T*=128 frames during training, and bound the scale factors within [*s*min,*s*max] = [0.2, 1.8]. This range explicitly enables both aggressive downscaling and selective upscaling. CAPO samples *M*=16 allocation trajectories per prompt and executes *N*=1 rollout per trajectory. We orchestrate training across 32 H100 GPUs utilizing VeRL [\(Sheng](#page-21-13) [et al.,](#page-21-13) [2025\)](#page-21-13), DeepSpeed [\(Rasley et al.,](#page-20-12) [2020\)](#page-20-12), and vLLM [\(Kwon et al.,](#page-19-2) [2023\)](#page-19-2). We execute evaluations via lmms-eval [\(Zhang et al.,](#page-23-6) [2024a\)](#page-23-6). We truncate standard response lengths at 256 tokens and extend the limit to 4,096 tokens for reasoning models.

### <span id="page-28-0"></span>**B.3. Reward Design**

We detail the reward structures supplementing Sec. [3.3.](#page-5-3) The base scalar reward *R* task *m*,*n* isolates task-specific performance. Efficiency constraints manifest through CAPO advantage shaping rather than primitive additive reward terms.

**Base Task Reward** ( $R_{m,n}^{\text{task}}$ ). We define objective metrics for four task types:

- *Question Answering*. For math problems, we extract the numeric answer and tolerate a  $10^{-2}$  deviation from the ground truth. For multiple-choice questions, we parse the exact option letter. For standard QA, we execute exact string matching post-normalization (case-folding and whitespace stripping). This yields a binary reward  $R_{\text{QA}}(\hat{o}, o) \in \{0, 1\}$ .
- *Free-form Generation*. We quantify open-ended generation quality via the ROUGE-L score between the prediction  $\hat{o}$  and the reference o:  $R_{\text{Gen}}(\hat{o}, o) = \text{ROUGE-L}(\hat{o}, o) \in [0, 1]$ .
- Temporal Grounding. Let  $\mathcal{G} = \{[s_j, e_j]\}_j$  denote ground-truth segments and  $\widehat{\mathcal{G}} = \{[\hat{s}_k, \hat{e}_k]\}_k$  denote predictions. We isolate the highest temporal IoU across all pairs:  $R_{TG}(\widehat{\mathcal{G}}, \mathcal{G}) = \max_{[\hat{s}, \hat{e}] \in \widehat{\mathcal{G}}, [s, e] \in \mathcal{G}} \text{tIoU}([\hat{s}, \hat{e}], [s, e]) \in [0, 1]$ . Invalid parsings default to 0.
- *Grounding QA*. We parse both the textual response and temporal segments, summing their independent scores:  $R_{\text{GQA}}(\hat{o}, \widehat{\mathcal{G}}; o, \mathcal{G}) = R_{\text{QA}}(\hat{o}, o) + R_{\text{TG}}(\widehat{\mathcal{G}}, \mathcal{G}) \in [0, 2]$ .

These metrics establish the scalar base reward  $R_{m,n}^{task}$ . CAPO subsequently defines a binary success indicator  $u_{m,n} \in \{0,1\}$ . Exact-match QA utilizes the binary outcome directly. Continuous metrics (ROUGE-L, temporal IoU, Grounding QA) apply a strict 0.35 threshold. Format validation, when active, injects a weighted penalty prior to GRPO normalization, while  $u_{m,n}$  strictly isolates the task metric.

**Format Reward.** We enforce a binary format reward  $R_{\text{fint}}(\hat{o}) \in \{0,1\}$  via rigid regex validation. The generation must output exactly one <think>...</think> block and one <answer>...</answer> block. The final answer must reside within \\boxed{...} inside the <answer> tags. Malformed outputs trigger a penalty, integrating into the scalar reward with a 0.2 weight.

### **B.4. Prompt Template**

We deploy the standard GRPO training prompt (Table 6). The model must encapsulate its reasoning trace within <think> </think> tags. While optional for ResAdapt (as the MLLM  $\pi_{\phi}$  internalizes reasoning), we mandate this structure to ensure parity with reasoning-based baselines. The final answer must emerge encased in \\boxed{}.

### C. Extended Analysis

This section extends the analysis of Sec. 4.4. We first examine the learned allocation policy at per-benchmark, per-duration, and per-category granularity (Sec. C.1), then present extended ablations on the temporal regularizer and reward design (Sec. C.2), qualitative case studies linking allocation to reasoning outcomes (Sec. C.3), and a boundary-case transfer test beyond video (Sec. C.4). Unless otherwise noted, all plots use Qwen2.5-VL-7B processing 32 uniformly sampled frames.

### <span id="page-29-0"></span>C.1. Extended Scale Policy Analysis

**Benchmark-level budget allocation.** Figure 8 reveals a clear benchmark-level ordering, despite the policy operating without any knowledge of benchmark identity during training. Reasoning-intensive tasks consistently command higher mean scales than perception-oriented tasks (0.435 vs. 0.417). MMMU-Adaptation anchors the high-fidelity extreme, while VideoMME anchors the low-fidelity extreme. The policy does not

<span id="page-30-1"></span>**Table 6: Prompt template used for CAPO training.** The template presents video frames and the task question, requires intermediate reasoning inside <think> tags, and places the final answer in \boxed{} within <answer> tags. This structure enables automatic reward extraction from MLLM outputs.

# **Prompt Template for Training with Thinking**

### **System Prompt:**

You are a helpful assistant.

You FIRST think about the reasoning process as an internal monologue and then provide the final answer.

The reasoning process MUST BE enclosed within <think> </think> tags and the answer MUST BE enclosed within <answer> </answer> tags.

The final answer MUST BE put in \boxed{} and the \boxed{} expression MUST BE contained entirely within the <answer> </answer> tags.

Do not include any reasoning or explanations outside these tags.

apply a fixed compression ratio; it calibrates its operating point to the anticipated visual complexity of each task.

**Long-context behavior.** Figure [9](#page-32-1) clarifies how the policy handles increasing clip length. As duration grows, the mean scale decreases (0.342→0.336→0.332) while within-video diversity simultaneously increases (0.085→∼0.095). The policy compresses longer videos more aggressively overall, but with far greater selectivity—precisely the regime where uniform resizing fails.

Figure [10](#page-32-2) provides a category-level view within VideoMME. The policy assigns maximum budget to *Sports Competition* (dense, high-motion) and minimum budget to *Artistic Performance* (visually sparse). Allocation tracks spatial complexity rather than task difficulty.

**Selectivity and success.** We measure frame-level selectivity via the Gini coefficient of predicted scales: a high Gini indicates concentrated budget on a sparse subset of frames. Figure [11](#page-33-0) shows that correct predictions consistently map to higher selectivity, peaking on MMMU-P. Success correlates not with larger average budgets, but with sharper concentration of resolution onto the decisive frames.

**Robustness and failure modes.** Figure [12](#page-33-1) examines whether adaptive compression preserves correct reasoning paths or merely reshuffles errors. Prediction stability is robust: 89% of originally correct samples survive compression. However, error correction and error induction rates remain comparable. The policy executes *selective redistribution*—rescuing certain failures by magnifying critical details, while occasionally destroying fine-grained evidence when the decisive cue is transient or visually inconspicuous.

### <span id="page-30-0"></span>**C.2. Extended Ablation Studies**

**Temporal similarity: cross-benchmark view.** Figure [13](#page-34-1) isolates the impact of Lsim. Without it, scale diversity collapses to near-zero across all benchmarks (*σ* < 0.003). Activating Lsim restores within-video variation by 4×–693×. CAPO controls the global budget level; Lsim breaks the uniform-scale equilibrium.

**Temporal similarity: structural diagnostics.** Figure [14](#page-35-0) provides four complementary views. With the

<span id="page-31-1"></span>![](_page_31_Figure_1.jpeg)

**Figure 8: Per-video mean scale across benchmarks.** Kernel density estimates of the per-video mean scale  $\bar{s}$ . Reasoning-heavy benchmarks shift toward larger  $\bar{s}$  than perception-heavy ones, indicating that the learned policy spends more fidelity where fine-grained evidence is more likely to matter.

regularizer active, the frame-scale histogram becomes bimodal, the per-video range expands, adjacent-frame variation increases, and the Gini coefficient rises. The policy transitions from a degenerate uniform allocator to a genuinely selective one.

**Reward design:** adaptivity. Figure 15 tracks the per-sample scale range  $s_{\text{max}} - s_{\text{min}}$  across training. CAPO maintains robust adaptivity on the validation split. Direct cost penalties collapse the range to zero, while cost-free training saturates at a uniform high-scale plateau.

**Reward design: convergence.** Figure 16 identifies the failure modes. Accuracy-only training saturates near  $s_{\text{max}}$ , abandoning compression. Direct cost optimization collapses to  $s_{\text{min}}$ , abandoning task quality. CAPO converges to a stable intermediate operating point that preserves content-adaptive allocation. The critical difference is not stability alone—both degenerate baselines are stable—but *where* the policy stabilizes.

### <span id="page-31-0"></span>C.3. Qualitative Case Studies

We provide four qualitative analyses mapping allocation behavior to reasoning outcomes (Figures 17–20). We render 32 uniformly sampled frames at their predicted scales; warmer borders denote aggressive upscaling.

Task-Dependent Operating Regimes. Figures 17 and 18 contrast two Video-MMMU tasks drawn from identical educational domains that trigger markedly different allocation strategies. In the comprehension task, evidence localizes strictly within diagram-heavy slides. The policy executes a sparse regime, aggressively compressing lecturer frames and suppressing an irrelevant quiz slide. Conversely, the adaptation task requires parsing a dense numeric table to compute a  $\chi^2$  statistic. The policy instantly shifts to a high-budget regime, broadly preserving fidelity and strongly upscaling the table frames. The policy reacts dynamically to downstream reasoning requirements, not merely superficial visual clutter.

Evidence Localization and Failure. Figure 19 (VideoMME) demonstrates precision localization. The policy

<span id="page-32-1"></span>![](_page_32_Figure_1.jpeg)

<span id="page-32-2"></span>**Figure 9: VideoMME broken down by video duration.** As clip duration grows, the policy lowers the average scale, increases within-video scale diversity, and faces lower task accuracy. Longer clips are therefore processed more aggressively and more selectively.

![](_page_32_Figure_3.jpeg)

**Figure 10: Scale allocation by VideoMME task category.** Mean  $\bar{s}$  varies substantially across categories, with larger budgets assigned to categories that contain crowded motion or finer local evidence. Accuracy annotations show that allocation is not a trivial proxy for which category is easiest.

isolates and magnifies brief frames containing critical date overlays, aggressively downscaling repetitive sky footage. Figure 20 exposes the prevailing failure mode. A decisive visual cue (a fork) appears briefly against a simple background. The policy mistakenly upscales an adjacent frame while compressing the critical frame, destroying the fine-grained evidence at the exact moment of relevance. This aligns with our robustness analysis: ResAdapt excels at broad concentration but remains brittle against highly transient, low-contrast cues.

### <span id="page-32-0"></span>C.4. Boundary-Case Transfer Beyond Video

While ResAdapt targets video QA and temporal grounding, we probe image transfer to identify operational boundaries. Table 7 shows that the video-trained policy occasionally identifies images requiring high fidelity (e.g., ChartQA), but does not deliver consistent efficiency-preserving gains on dense static-image benchmarks. The boundary is clear: input-side allocation generalizes across video tasks and operators, but a strictly video-trained policy requires explicit joint training to handle static image distributions reliably.

<span id="page-33-0"></span>![](_page_33_Figure_1.jpeg)

Scale Selectivity (Gini Coefficient): Correct vs. Incorrect Predictions

**Figure 11: Selectivity versus prediction correctness on three representative benchmarks.** Per-video Gini coefficients of the frame-level scales. Correct predictions tend to have higher Gini than incorrect ones, linking success to sharper concentration of resolution rather than merely larger average budgets.

<span id="page-33-1"></span>![](_page_33_Figure_4.jpeg)

**Figure 12: Sample-level robustness at 25% retention.** Most originally correct predictions remain correct, but corrected and newly introduced errors are of comparable magnitude. Adaptive allocation is therefore selective rather than lossless.

<span id="page-34-1"></span>![](_page_34_Figure_1.jpeg)

Figure 13: Cross-benchmark scale diversity with and without  $\mathcal{L}_{sim}$ . Per-video scale standard deviation  $\sigma$  across five benchmarks. Without the regularizer, diversity collapses toward zero; adding  $\mathcal{L}_{sim}$  restores broad within-video variation on every benchmark.

<span id="page-34-0"></span>**Table 7: Exploratory zero-shot transfer to image benchmarks.** Parenthetical values denote per-task retention ratio *R*, and ResAdapt-RL additionally fine-tunes the MLLM via RL.

| Model                          | <b>MathVista</b><br>testmini | <b>MMMU</b><br>val | OCRBench   | ChartQA    | AI2D       | <b>TextVQA</b><br>val |
|--------------------------------|------------------------------|--------------------|------------|------------|------------|-----------------------|
| Qwen2.5-VL-7B                  | 49.1(100%)                   | 50.9(100%)         | 84.2(100%) | 83.9(100%) | 82.5(100%) | 82.9(100%)            |
| Random Drop                    | 44.8(50%)                    | 49.0(50%)          | 74.8(50%)  | 71.6(50%)  | 80.3(50%)  | 78.1(50%)             |
| ToMe (Bolya et al., 2022)      | 46.2(50%)                    | 49.6(50%)          | 79.3(50%)  | 78.1(50%)  | 81.9(50%)  | 81.2(50%)             |
| VisionZip (Yang et al., 2025c) | 47.2(50%)                    | 48.6(50%)          | 79.6(50%)  | 77.9(50%)  | 81.9(50%)  | 81.3(50%)             |
| ResAdapt(Qwen2.5-VL-7B)        | 45.5(42%)                    | 51.0(29%)          | 80.0(64%)  | 85.9(105%) | 81.4(41%)  | 69.6(30%)             |
| ResAdapt-RL(Qwen2.5-VL-7B)     | 46.7(42%)                    | 50.9(29%)          | 80.8(64%)  | 86.6(105%) | 81.1(41%)  | 70.1(30%)             |
| Qwen3-VL-8B                    | 56.1(100%)                   | 53.4(100%)         | 85.0(100%) | 84.0(100%) | 83.5(100%) | 82.1(100%)            |
| Random Drop                    | 47.3(50%)                    | 48.7(50%)          | 62.9(50%)  | 70.2(50%)  | 79.7(50%)  | 76.6(50%)             |
| VisionZip (Yang et al., 2025c) | 47.8(50%)                    | 50.3(50%)          | 70.5(50%)  | 75.0(50%)  | 80.5(50%)  | 79.3(50%)             |
| ToMe (Bolya et al., 2022)      | 49.6(50%)                    | 50.6(50%)          | 70.3(50%)  | 75.2(50%)  | 80.5(50%)  | 79.4(50%)             |
| ResAdapt(Qwen3-VL-8B)          | 52.5(42%)                    | 50.9(29%)          | 82.7(64%)  | 83.2(105%) | 81.2(41%)  | 67.8(30%)             |

<span id="page-35-0"></span>![](_page_35_Figure_1.jpeg)

Figure 14: Four diagnostics of the  $\mathcal{L}_{sim}$  ablation on VideoMME. With the regularizer, the frame-scale histogram becomes bimodal, the per-video range expands, adjacent-frame variation increases, and the Gini coefficient rises. The policy moves from near-uniform allocation to a genuinely selective regime.

<span id="page-35-1"></span>![](_page_35_Figure_3.jpeg)

Figure 15: Per-sample scale adaptivity under different reward designs. Scale range  $s_{\text{max}} - s_{\text{min}}$  over training on (a) training and (b) validation splits. CAPO keeps a non-trivial adaptive range, whereas direct cost collapses and cost-free training saturates.

<span id="page-36-0"></span>![](_page_36_Figure_1.jpeg)

**Figure 16: Validation-time convergence under different reward designs.** CAPO variants converge to stable intermediate operating points, while cost-free training saturates at the upper boundary and direct cost collapses to the lower boundary. Stability alone is not sufficient; the key is where the policy stabilizes.

<span id="page-37-0"></span>**Q:** Evaluate five statements about Urban Geography City Models (concentric zone, Hoyt sector, multiple nuclei, galactic, Latin American); identify which are correct. *Please ignore the Quiz question in last frame of the video.*

![](_page_37_Figure_2.jpeg)

**Figure 17: Case 1: Video-MMMU Comprehension [\(Hu et al.,](#page-19-3) [2025\)](#page-19-3) (Vanilla** × → **ResAdapt** ✓**).** The policy concentrates resolution on diagram-bearing slide frames, compresses lecturer-only frames, and suppresses the final quiz frame that the prompt explicitly marks as irrelevant.

<span id="page-38-0"></span>**Q:** Watch and learn the video content. Then apply what you learned to answer: Table 11.47 provides a survey of the youngest online entrepreneurs (ages 17–30) whose net worth ≥ \$1M. We want to know whether ages and net worth are independent. *χ* 2 test statistic = \_\_\_\_\_\_

![](_page_38_Figure_2.jpeg)

**Figure 18: Case 2: Video-MMMU Adaptation [\(Hu et al.,](#page-19-3) [2025\)](#page-19-3) (Vanilla** × → **ResAdapt** ✓**).** When the answer depends on reading a numeric table and performing a *χ* 2 computation, the policy keeps a much higher global budget and strongly upscales the table-bearing frames.

<span id="page-39-0"></span>![](_page_39_Figure_1.jpeg)

![](_page_39_Figure_2.jpeg)

**Figure 19: Case 3: VideoMME [\(Fu et al.,](#page-18-4) [2025a\)](#page-18-4) (Vanilla** × → **ResAdapt** ✓**).** Frames containing the decisive date overlays are enlarged, while the largely homogeneous sky footage is compressed. The policy spends budget on answer-bearing evidence rather than on the surrounding context.

<span id="page-40-0"></span>**Q:** Which item does the man throw into the trash at the beginning of the video? (A) A fork, (B) A pair of chopsticks, (C) A box of noodles, (D) A spoon.

![](_page_40_Figure_2.jpeg)

**Figure 20: Case 4: VideoMME [\(Fu et al.,](#page-18-4) [2025a\)](#page-18-4) (Vanilla** ✓ → **ResAdapt** ×**; failure case).** A nearby frame is enlarged, but the actual fork-bearing frame is compressed. The decisive fine detail is therefore lost at exactly the wrong moment.