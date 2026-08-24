# 3 Methodology

### 3.1 Hybrid Reasoning with Gating

We first describe our notation and settings for hybrid latent reasoning. For input query x = [x1, x2, . . . , xt] and its corresponding token embeddings E = [e1, e2, . . . , et], we describe the raw hidden states from the LLM output at step t with hˆ <sup>t</sup>, namely:

$$\hat{H} = [\hat{h}_1, \hat{h}_2, \dots, \hat{h}_t] = \texttt{Transformer}(E), \tag{1}$$

<span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> <think> </think> We simply add both. 1. Rollout Hybrid Advantages Rollouts Query LM Head Policy Standardized Model Rewards 2. Policy Update Decoder Layers Policy Gating (1) Model Policy Loss **Embedding Layer** Model\* Ref. KL Model What is 2+2? Hybrid Reasoning **HRPO** Training
![](_page_3_Picture_0.jpeg)

Figure 2: Hybrid reasoning with gating (left) and hybrid reasoning policy optimization (right). During rollouts, the reasoning trajectory is generated hybridly with both discrete tokens and latent features, and for policy update, we compute the HRPO loss using the hybrid rollout buffer to update the model.

in which Transformer denotes the transformer model (i.e., decoder layers),  $\hat{H}$  represents the final-layer hidden states produced by the Transformer. With the LM head (Head), the next output token  $\hat{x}_{t+1}$  can be sampled from the output distribution over the vocabulary via:

$$\hat{x}_{t+1} \sim \text{softmax}(\text{Head}(\hat{h}_t)).$$
 (2)

However, hidden states often lie outside the model's token embedding manifold, which degrades generation quality when fed directly. To avoid this, we project  $\hat{h}_t$  back into the embedding space to ensure the inputs conform to the model's learned distribution. Specifically, we use the output probabilities  $p_{t+1}$  to compute a weighted interpolation over the vocabulary:

<span id="page-3-0"></span>
$$h_{t+1} = W_e^T \frac{p_{t+1}}{\|p_{t+1}\|}, \quad \text{with} \quad p_{t+1} = \text{softmax}(\frac{\text{Head}(\hat{h}_t)}{\tau}), \tag{3}$$

in which  $\tau$  is the temperature and  $W_e$  denotes the embedding matrix of the LLM. In other words, we compute the next input embedding as a weighted sum of all token embeddings, with weights given by  $p_{t+1}$ . In addition,  $p_{t+1}$  is normalized to preserve the scale and variance of the output vector. This sampling-free mapping ensures differentiability and aligns the projected embedding with the model's native input space, thus leading to improved training dynamics (see Section 4.3).

While interpolated embeddings preserve semantic continuity, directly feeding  $h_{t+1}$  as the next token input removes stochasticity and injects noise from irrelevant tokens, causing degraded generation within RL rollouts. As such, we design a hybrid approach for latent reasoning by gradually imposing hidden state representations into the sampled token embeddings with a gating mechanism. Drawing on gated recurrence models [5, 27], we formulate the gating mechanism as:

<span id="page-3-2"></span>
$$\begin{split} r_t &= \sigma(W_a \hat{e}_{t+1} + b_a), \\ i_t &= \sigma(W_x \hat{e}_{t+1} + b_x), \\ a_t &= \exp(-c \cdot \operatorname{softplus}(\Lambda) \odot r_t), \\ e_{t+1} &= \left\{ \begin{array}{ll} a_t \odot \hat{e}_{t+1} + \sqrt{1 - a_t^2} \odot (i_t \odot h_{t+1}) & t \in \operatorname{think}, \\ \hat{e}_{t+1} & t \not\in \operatorname{think}, \end{array} \right. \end{split}$$

 $e_{t+1}$  is the resulting hybrid input for the next step,  $\hat{e}_{t+1}$  denotes the embedding of the sampled discrete token  $\hat{x}_{t+1}$ , whereas  $h_{t+1}$  is the projected hidden states as in Equation (3). The gates  $r_t$  and  $i_t$  leverages sigmoid function  $\sigma$  to control the blending,  $a_t$  scales  $\hat{e}_{t+1}$ , c is a fixed scaling constant, and  $\Lambda$  is a learnable vector. Note that hybrid reasoning only applies during the reasoning phase (i.e.,  $t \in \mathtt{think}$ ), while the final answer is still generated via standard autoregressive decoding, as we show in Figure 2 (left). By initializing  $a_t \to 1$  (see Section A), the inputs first draw predominantly from the sampled token embeddings, thereby effectively preserving the LLM's generative capabilities. As the training progresses, the value range of  $a_t$  converges to an optimum range and thus incorporates informative features from both hidden representations and sampled tokens.

Overall, our hybrid reasoning approach projects hidden states into the embedding space via weighted interpolation. Moreover, the sampling steps preserve stochasticity for effective reinforcement learning. We employ a plug-and-play gating mechanism that initially prioritizes sampled token embeddings while gradually integrating latent signals, providing richer inputs for subsequent reasoning.

#### 3.2 Hybrid Reasoning Policy Optimization (HRPO)

Rather than relying on strong supervision, we optimize the policy model via hybrid rollouts using reinforcement learning (RL), fully harnessing LLMs' native reasoning capabilities. Inspired by recent RL advances such as group relative policy optimization (GRPO) [\[33\]](#page-11-11), we introduce hybrid reasoning policy optimization (HRPO), an efficient RL-driven framework that enable LLMs to fuse discrete tokens with continuous representations for hybrid reasoning.

As illustrated in Figure [2](#page-3-1) (right), the proposed HRPO optimizes the policy (parameterized by θ) to maximize the expected reward for input x drawn from dataset D and the sampled hybrid outputs y (discrete tokens) and H (hidden representations):

$$\max_{\theta} \mathbb{E}_{(x,y)\sim \mathcal{D},(\hat{y},H)\sim \pi_{\theta}(\cdot|x)}[r(a,y)],\tag{5}$$

where r is a simple outcome-based reward function and a denotes the ground truth answer (i.e., it outputs 1 for correct prediction in y and 0 otherwise). The rewards are computed solely on the discrete tokens within the answer span. To obtain an unbiased, low-variance advantage for hybrid latent reasoning, we generate g hybrid rollouts per input query and compute the advantages by standardizing the rewards within the group (i.e., for the i-th response, the advantage is calculated by Aˆ <sup>i</sup> = ri−mean([r1,r2,...,rg]) std([r1,r2,...,rg]) ). Consequently, the policy gradients can be estimated with:

<span id="page-4-0"></span>
$$\nabla_{\theta} \mathcal{J}_{HRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{(y_i, H_i)\}_{i=1}^g \sim \pi_{\theta}(\cdot | x)}$$

$$\left[ \frac{1}{g} \sum_{i=1}^g \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \nabla_{\theta} \log \pi_{\theta}(y_{i,t} | x, y_{i, < t}, H_{i, < t}) \hat{A}_{i,t} \right] - \beta \nabla_{\theta} \mathbb{D}_{KL}[\pi_{\theta} \| \pi_{ref}],$$
(6)

where πref denotes the reference model and KL-divergence acts as a regularizer, controlled by hyperparameter β. This objective follows a simple REINFORCE-style formulation, fusing discrete token inputs with continuous hidden representations across the reasoning span via the introduced gating mechanism. The hybrid trajectories that yield higher returns are assigned larger advantage estimates, encouraging policy updates to increase the log probabilities of their subsequent reasoning tokens. For the KL divergence term, we compute log probabilities using solely token IDs for πref, as we find it more effective in preserving training stability. Different from PPO / GRPO objectives, we omit the likelihood ratio and directly use raw log probabilities in Equation [\(6\)](#page-4-0) because ratio clipping is rarely encountered under our conservative learning schedule. Furthermore, since the hidden representations are directly tied to the parameters θ, each trajectory should only be used for a single gradient update; attempting to reuse it—even with importance sampling—violates the on-policy constraints. As such, our HRPO implementation remains lightweight, strictly on-policy and could be seamlessly combined with further RL optimizations.

In summary, the proposed HRPO framework unifies hybrid latent reasoning under a simple RL objective that fully leverages LLMs' intrinsic reasoning capabilities. During rollouts, the decoding process progressively fuses discrete and continuous representations through a learnable gate, preserving coherence while exploiting hidden states. For policy updates, HRPO derives advantages directly from outcome rewards and performs policy gradient steps with KL regularization. As a result, HRPO incentivizes LLMs to dynamically integrate sampled tokens with latent representations, delivering stable and efficient on-policy hybrid reasoning training without a separate value function.

