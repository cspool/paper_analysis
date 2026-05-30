# 1 Introduction

Diffusion language models (DLMs) [\[3,](#page-9-0) [25\]](#page-10-0) have recently emerged as an alternative paradigm for text generation, inspired by the success of diffusion models in continuous domains like images [\[15,](#page-9-1) [48\]](#page-11-0) and videos [\[28,](#page-10-1) [5\]](#page-9-2). Unlike autoregressive transformers (ARs) [\[14,](#page-9-3) [43,](#page-11-1) [12\]](#page-9-4) that generate text leftto-right at a time, a diffusion language model produces text by gradually refining a sequence of initially noisy tokens or masked tokens into a coherent output [\[24,](#page-10-2) [29\]](#page-10-3). Recent advancements in diffusion-based language models have underscored their versatility across both continuous [\[21\]](#page-10-4) and discrete formulations [\[40,](#page-11-2) [3\]](#page-9-0). In particular, discrete diffusion models have shown competitive performance in language modeling [\[37\]](#page-10-5), attracting growing interest for their potential to achieve faster decoding than traditional ARs while maintaining comparable generation quality.

One notable advantage of diffusion language models is their potential to decode an arbitrary number of tokens in parallel [\[49\]](#page-11-3), whereas ARs require one forward pass per generated token. This parallel decoding paradigm offers the potential for improved wall-clock inference time and higher throughput. However, despite their theoretical efficiency, current DLMs remain substantially slower than ARs in practice. This inefficiency is primarily due to two factors: the incompatibility of DLMs with the

<sup>∗</sup>Corresponding author

KV-Cache mechanism [33, 13], and the large number of network evaluations for denoising [17]. Specifically, generating a sequence of length L typically entails L denoising steps, each involving a full bidirectional attention pass, leading to a cubic time complexity of  $\mathcal{O}(L^3)$ . In contrast, AR models utilize KV-Cache to reduce per-step complexity to  $\mathcal{O}(L^2)$ , achieving much faster inference overall.

In this paper, we tackle the challenge of integrating the KV-Cache mechanism into diffusion language models and operate without autoregressive or semi-autoregressive structures [2]. We identify two core reasons that prevent the direct usage of KV-Cache in DLMs. (1) KV-Cache hinges on the assumption that the key and value states of previously generated tokens remain fixed during subsequent decoding steps. This property is preserved in autoregressive models through the use of a causal attention mask, which restricts each token to attend only to earlier positions [11]. However, DLMs adopt a bidirectional attention mechanism, similar to non-autoregressive models [20], allowing every token to attend to all others in the sequence and making it nontrivial to reuse previously cached keys and values. (2) KV-Cache presupposes a fixed and left-to-right sequential decoding, where the position of the next token is deterministic. This assumption enables ARs to compute QKV states selectively only at the current decoding position. However, DLMs break this paradigm by supporting flexible generation orders. At each denoising step, any token position may be selected for update, which is a key advantage of DLMs for tasks involving long-range dependencies and holistic planning [51].

To solve this problem, we propose the delayed KV-Cache, dKV-Cache, the KV-Cache for diffusion language models. The core design of our proposed dKV-Cache centers on how to enable caching of key and value states across denoising steps in diffusion language models. A key insight motivating this design is that, although DLMs employ bidirectional attention, intuitively incompatible with caching, the representations of key and value are not fundamentally unreusable, but rather require delayed and conditioned reuse. In particular, we observe that the evolution of key and value states is strongly influenced by whether a token has been decoded. This behavior motivates a delayed caching strategy, wherein only the key and value states of decoded tokens would be cached, delayed from the ARs that caching occurs immediately upon input. We further propose a one-step delayed caching mechanism, in which the caching of key/value states is postponed by another denoising step. This intentional delay substantially boosts the performance and also reduces memory overhead for KV storage. Besides the delayed caching, we further propose a more aggressive decoding strategy to reduce the computational complexity of diffusion language models from  $\mathcal{O}(L^3)$  to  $\mathcal{O}(L^2)$  by restricting the caching to a compact subset of tokens: the delayed tokens and the current to-be-decoded tokens, and the window tokens.

Our experiments demonstrate that the proposed method achieves  $2-10\times$  speedup on existing 7B-scale diffusion language models, including LLaDA [37] and Dream [52], across a broad range of benchmarks such as general language understanding, code generation, and mathematical problem solving. These efficiency gains come with only minor and often negligible performance degradation, highlighting the practical value of our approach for accelerating DLM inference without training. Furthermore, we demonstrate that dKV-Cache is robust across variations in prefill length, output length, and the number of sampling steps.

**Contributions.** (1) We propose the first KV-Cache mechanism for diffusion language models by leveraging the evolving dynamics of token representations. We introduce a delay caching strategy compatible with bidirectional attention. (2) We propose two practical variants of our method: dKV-Cache-Decode, which enables long-term cache reuse, and dKV-Cache-Greedy, which reduces the per-step time complexity for faster decoding. (3) Extensive experiments on DLMs demonstrate that our approach achieves  $2{\text -}10\times$  inference speedup with minimal or negligible performance loss.

#### 2 Related Work

#### 2.1 Diffusion Language Models

Diffusion models [24, 41] model the data generation as the inversion of a forward-noise process and demonstrated impressive generation quality in image [38], video [5], and audio generation [16]. For diffusion models on language generation, [25, 3] extends DDPM to categorical data and defines the transition matrices for the corruption and denoising. [29] introduces the continuous-time diffusion over the continuous word-embedding space, and [21] closes the quality of generation on par with GPT-2 via a simplex defined over the vocabulary. Besides the diffusion in the continuous space [47, 53]

for discrete distribution, another approach seeks the path in discrete language diffusion models. [22] training BERT to learn the reverse process of a discrete diffusion process with an absorbing state in D3PM. SEDD [33] introduces score entropy training, a novel loss extending score-matching to discrete data and MDLM [40] shows that simple masked discrete diffusion is competitive to all previous kinds of DLMs. Block diffusion [2] extends the current non-autogressive [20] discrete language diffusion models into a semi-autoregressive one [21, 55], making it feasible to generate sequences of arbitrary length. [37, 19] scaling the masked diffusion language models to billions of parameters, achieving performance comparable to leading autoregressive LLMs.

#### 2.2 Cache in Generative Models

Cache [45] is a small, fast memory that stores frequently accessed data, reducing the time that the CPU needs to fetch data from slower memory. Cache is first introduced in deep neural networks in transformers [44], where the KV-Cache caches previous tokens' key and value tensors. KV-Cache becomes a fundamental technique in transformers, and several improved techniques are proposed [18, 32, 26] to reduce the memory consumption of KV-Cache for long-context generation. Besides this, caches are also been explored in diffusion models for image [36, 46]. Those work leverage the temporal similarities between high-level features [35, 9], attention map [54, 31] to achieve faster inference of diffusion generation. This also has been explored in 3D generative modeling [50] and video generation [56, 34]. However, the cache for diffusion language models is less explored, especially the KV-Cache for diffusion language models. [2] explores the kv-cache in semi-autoregressive diffusion language models. This requires considering the KV-Cache in training, making twice the forward computation in the training and its form is still constrained in the autoregressive formula. [40] also considers cache, but under a strict condition that no new tokens have been calculated.

#### 3 Methods

#### 3.1 Preliminary

We primarily focus on continuous-time discrete language models, with particular attention to masked diffusion language models [40, 37], which have shown strong scalability to billion-parameter scales with high generation quality.

Consider a text sequence with L tokens  $\mathbf{x}_0^{1:L}$ , sampled from the target distribution  $p_{data}(\mathbf{x}_0)$ . Each token is represented by a one-hot vector with V categories, where V is the vocabulary size. The forward process adds noise in the original sequence  $\mathbf{x}_0$ , which, in the discrete diffusion models here, takes the form of masking some of the tokens randomly. The masking process can be controlled by a transition matrix  $U_t$ , where each element in this matrix  $[U_t]_{ij}$  represents the probability transition from token i to token j at step t [3]. The denoising process is discretized into T steps, and we define the continuous timestep c(t) = t/T, where  $t \in \{0, 1, \dots, T\}$ . We use *timestep* in the continuous temporal space of the denoising process and *step* in the discrete space. The forward diffusion can be modelled as:

$$q\left(\boldsymbol{x}_{c(t)}\mid\boldsymbol{x}_{0}\right)=\operatorname{Cat}\left(\boldsymbol{x}_{c(t)};\boldsymbol{p}=\boldsymbol{x}_{0}\overline{\boldsymbol{U}}_{t}\right), \quad \text{ where } \quad \overline{\boldsymbol{U}}_{t}=\prod_{i=1}^{t}\boldsymbol{U}_{i}$$

Then we can get the corrupted  $\mathbf{x}_0$ . The absorbing form for the transition matrix is used here, where each token either remains the same or transfers to the special [MASK] token at the probability of  $\beta_t$ . The cumulated transition matrix  $\overline{U}_t$ , as defined in the masked diffusion models, can be formulated as:

$$\begin{bmatrix} \overline{\boldsymbol{U}}_t \end{bmatrix}_{ij} = \left\{ \begin{array}{ll} 1 & \text{if} \quad i = j = [\text{MASK}] \\ \bar{\alpha}_t & \text{if} \quad i = j \neq [\text{MASK}] \\ 1 - \bar{\alpha}_t & \text{if} \quad j = m, i \neq [\text{MASK}] \end{array} \right. \quad \text{with } \bar{\alpha}_t = \prod_{i=1}^t \left(1 - \beta_i\right)$$

where  $\bar{\alpha}_t$  linearly decrease to 0 as t approaches T. The reverse process is learned by the models  $\theta$ , where  $p_{\theta}(\mathbf{x}_{c(t-1)}|\mathbf{x}_{c(t)})$  is learned to approximate  $q(\mathbf{x}_{c(t-1)}|\mathbf{x}_{c(t)},\mathbf{x}_0)$ . In  $p_{\theta}(\mathbf{x}_{c(t-1)}|\mathbf{x}_{c(t)})$ , the neural network with parameters  $\theta$  is optimized to predict the clean tokens  $\mathbf{x}_0$  given  $\mathbf{x}_{c(t)}$ .

![](_page_3_Figure_0.jpeg)

Figure 1: Illustration of dKV-Cache. At step t, no prior cache would be activated though the token  $D_{t-1}$  has been decoded.  $\mathbf{K}$  and  $\mathbf{V}$  are delayed to the next step to be reordered and reused.

Sampling Process of DLMs. Given the noisy sequence  $x_1^{1:L}$ , which is consisted only with the masked token. Each timestep, the denoising model  $p_{\theta}(\mathbf{x}_{c(t-1)}|\mathbf{x}_{c(t)})$  would be called, with  $x_0$  predict first and then remasking the rest by  $q(\mathbf{x}_{c(t-1)}|\mathbf{x}_0,\mathbf{x}_{c(t)})$ . The unmasked tokens would remain unchanged during the later denoising process. Several strategies are used in the remasking stage, e.g., random remasking [3], keeping the most confident ones [7] or by selecting the topk positions with the largest margin between the two most probable values [27].

**Formulation of KV-Cache.** Since diffusion language models still use the transformer architecture (with or without GQA [1] would not affect our method), we simply grab the formulation of KV-Cache in ARs first. In a transformer decoder, each layer would project the current hidden states  $\mathbf{h}_t$  into a query-key-value triplet  $(\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t)$  via learned projection  $\mathbf{W}_{\mathbf{Q}}, \mathbf{W}_{\mathbf{K}}$  and  $\mathbf{W}_{\mathbf{V}}$ . At step t, only the hidden states of the t-th tokens  $\mathbf{h}_t^{[t]}$  would be calculated. The recursive KV-Cache update is appending the new key-value pair to the running buffered KV-Cache:

<span id="page-3-0"></span>
$$\mathbf{z}_{t} = \operatorname{softmax} \left( \frac{\mathbf{Q}_{t}^{[t]} \left( \mathbf{K}_{t}^{[1:t]} \right)^{\top}}{\sqrt{d_{k}}} \right) \mathbf{V}^{[1:t]} \quad \text{with} \quad \begin{cases} \mathbf{K}_{t}^{[1:t]} = \operatorname{concat} \left( \mathbf{K}_{t-1}^{[1:t-1]}, \mathbf{K}_{t}^{[t]} \right) \\ \mathbf{V}_{t}^{[1:t]} = \operatorname{concat} \left( \mathbf{V}_{t-1}^{[1:t-1]}, \mathbf{V}_{t}^{[t]} \right) \end{cases}$$
(1)

where  $\mathbf{z}_t$  is the output of the attention head at step t and the dot products are scaled down by  $\sqrt{d_k}$ .

#### 3.2 Why KV-Cache Cannot be Used in DLMs?

The effectiveness of KV-Cache can be attributed to the reuse of previously computed K and V states, and the targeted computation only for the current decoding token. We conclude that standard KV-Cache is fundamentally incompatible with diffusion language models for two reasons:

- Timestep-variant key and value states. In the autoregressive setting, every time step shares a single, causally growing set of key and value tensors.  $\mathbf{K}_m^{[1:t-1]}$  and  $\mathbf{V}_m^{[1:t-1]}$  are the same at each step m from t-1 and later with the help of causal attention mask. By contrast, DLMs employ a bidirectional attention mask; consequently, the key and value representations that each token can attend to at timestep m,  $\mathbf{K}_m^{[1:t-1]}$  and  $\mathbf{V}_m^{[1:t-1]}$ , differ from those at timestep n. Put differently,  $\mathbf{K}_m^{[1:t-1]} \neq \mathbf{K}_n^{[1:t-1]}$  and  $\mathbf{V}_m^{[1:t-1]} \neq \mathbf{V}_n^{[1:t-1]}$  if  $n \neq m$ . The bidirectional attention introduced by diffusion language models therefore breaks the global reuse assumption that supports conventional KV-Cache.
- Non-sequential decoding order. Generation in DLMs does not follow a strictly left-to-right order. Instead, the model dynamically fills masked positions based on probabilities computed at each denoising step. As a result, the positions of decoded tokens are only revealed after the model forward pass, and the subsequent update may target any position in the sequence, rather than progressing sequentially. This uncertainty prevents us from pre-determining which token i will require the computation of its hidden states  $\mathbf{h}^{[i]}$  and its  $\mathbf{Q}^{[i]}$ ,  $\mathbf{K}^{[i]}$  and  $\mathbf{V}^{[i]}$ .

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 2: (a) We present a heatmap illustrating the pairwise similarities among the key states across different timesteps. Here we use LLaDA with  $L=128,\,T=128$  and the block size is set to 64. We then compute the Euclidean distance between consecutive steps t and t+1 to analyze the dynamics of intermediate representations. (b) We report the average distance measured before and after the decoding of each token. (c) We highlight the top-2 steps exhibiting the largest and smallest changes in the key and value states for each token in their decoded order.

Representation Dynamics for tokens in Diffusion Sampling We investigate in DLMs whether K and V can be reused. We focus on the dynamics of K and V for each token, and the results are shown in Figure 2. Interestingly, we observe several noteworthy patterns in the dynamics of QKV states: (1) Despite step-to-step differences, the key and value embeddings, K and V, exhibit consistently high similarity across timesteps, as shown in Figure 2(a). (2) Once a token is decoded, its representation becomes relatively stable in subsequent steps, whereas the representations of still-masked tokens continue to fluctuate significantly. This phenomenon is evident in Figure 2(b), where QKV fluctuations are more pronounced prior to decoding than thereafter. (3) The most substantial changes in K and V occur at the decoding step of each token and then in the early stages of the denoising process. This is reflected by prominent changes along the diagonal in Figure 2(c), corresponding to the i-th decoding step which decodes the i-th token.

These observations provide key insights into the temporal structure of discrete diffusion models and motivate the design of our KV-Cache mechanism for diffusion language modeling.

#### 3.3 Delayed KV-Cache for Masked Diffusion Language Models

We first present a more general non-sequential KV-Cache formulation that replaces the contiguous slice  $\mathbf{K}^{[1:t-1]}$  used in Eq.1 with an arbitrary-order index set  $\mathcal{S}_t \subseteq \mathcal{I} = \{1,\ldots,L\}$  and the next token position from the fixed t to  $D_t$ . The cached keys and values gathered from previous steps are now  $\mathbf{K}_t^{\mathcal{S}_t}$ , which retrieves cached states at the positions specified by indexes in  $\mathcal{S}_t$ :

<span id="page-4-1"></span>
$$\mathbf{z}_{t} = \operatorname{softmax} \left( \frac{\mathbf{Q}_{t}^{D_{t}} \left( \mathbf{K}_{t}^{\mathcal{S}_{t} \cup \{D_{t}\}} \right)^{\top}}{\sqrt{d_{k}}} \right) \mathbf{V}_{t}^{\mathcal{S}_{t} \cup \{D_{t}\}} \text{ with } \left\{ \begin{array}{l} \mathbf{K}_{t}^{\mathcal{S}_{t} \cup \{t\}} = \operatorname{concat\_reorder} \left( \mathbf{K}_{t-1}^{\mathcal{S}_{t}}, \mathbf{K}_{t}^{D_{t}} \right) \\ \mathbf{V}_{t}^{\mathcal{S}_{t} \cup \{t\}} = \operatorname{concat\_reorder} \left( \mathbf{V}_{t-1}^{\mathcal{S}_{t}}, \mathbf{V}_{t}^{D_{t}} \right) \end{array} \right.$$

where  $D_t$  is the denoised token at step t. If we use random sampling for diffusion, then before any inference, we can know the decoding order of the sequence. We here use the notation of  $D_t$ , and later would not depend on knowing the decoding order. The operator concat\_reorder is proposed to enhance the efficiency of the indexing and gathering of K and V here. We explain in the appendix about this operator and how it works with ROPE [42] and how it accelerates.

We first extend the formulation in Eq.2 by generalizing the query input  $\mathbf{Q}_t^{D_t}$  from the single-token decoding to the multiple and arbitrary token decoding. Specifically, at decoding step t, we construct a dynamic set  $\mathcal{M}_t$ , where  $\mathcal{M}_t$  denotes the subset of tokens that are not yet finalized during generation. And with the  $\mathbf{h}_t^{\mathcal{M}_t}$  calcuated, we also can get the corresponding  $\mathbf{Q}_t^{\mathcal{M}_t}$ ,  $\mathbf{K}_t^{\mathcal{M}_t}$  and  $\mathbf{V}_t^{\mathcal{M}_t}$ . We delay the caching of each token, not the time it is appended in the input, but rather on the step it is decoded:

$$\operatorname{concat\_reorder}\left(\begin{array}{c} \mathbf{K}_{t-1}^{\mathcal{S}_t} \\ \operatorname{Cache\ from} t-1 \end{array}, \begin{array}{c} \mathbf{K}_{t}^{D_t} \\ \operatorname{Calcualte\ at} t \end{array}\right) \Rightarrow \operatorname{concat\_reorder}\left(\begin{array}{c} \mathbf{K}_{t-1}^{\mathcal{I} \setminus \mathcal{M}_t} \\ \operatorname{Cache\ from} t-1 \end{array}, \begin{array}{c} \mathbf{K}_{t}^{\mathcal{M}_t} \\ \operatorname{Calcualte\ at} t \end{array}\right)$$
 (3)

where  $\mathcal{I}$  denotes the set of all tokens involved in the denoising process. This design reflects our core observation in the previous section: only the decoded tokens are eligible for caching, while

the remaining masked tokens must be re-encoded at each step. Besides, this method solves the problem that we need to predefine or predict the denoising order, as  $\mathcal{M}_t$  is explicitly known at each step. Cached keys and values corresponding to  $\mathcal{I} \setminus \mathcal{M}_t$  are reused across steps, while non-finalized positions are recomputed at each step to ensure correctness under bidirectional attention masking.

One-step Delayed Caching. As our analysis in Figure 2(c), the most significant change in K and V occurs exactly at the step where a token transitions from [MASK] to its decoded form. Currently,  $K_t^{D_t}$  would be used for caching, as it is no longer in the masked set  $\mathcal{M}_t$ . However,  $K_t^{D_t}$  can differ substantially from  $K_{t+1}^{D_t}$ , and prematurely reusing it lead to severe performance degradation. To address this, we introduce one-step delayed caching. At timestep t, we use the masking state from the previous step,  $\mathcal{M}_{t-1}$ , to determine which tokens are cacheable. The method, named dKV-Cache-Decode, is finally formalized as:

<span id="page-5-0"></span>
$$\mathbf{z}_{t} = \operatorname{softmax} \left( \frac{\mathbf{Q}_{t}^{\mathcal{M}_{t-1}} \left( \mathbf{K}_{t}^{\mathcal{I}} \right)^{\top}}{\sqrt{d_{k}}} \right) \mathbf{V}_{t}^{\mathcal{I}} \text{ with } \begin{cases} \mathbf{K}_{t}^{\mathcal{I}} = \operatorname{concat\_reorder} \left( \mathbf{K}_{t-1}^{\mathcal{I} \setminus \mathcal{M}_{t-1}}, \mathbf{K}_{t}^{\mathcal{M}_{t-1}} \right) \\ \mathbf{V}_{t}^{\mathcal{I}} = \operatorname{concat\_reorder} \left( \mathbf{V}_{t-1}^{\mathcal{I} \setminus \mathcal{M}_{t-1}}, \mathbf{V}_{t}^{\mathcal{M}_{t-1}} \right) \end{cases}$$

$$(4)$$

While this slightly reduces efficiency, we find it to be critical for maintaining accuracy and stability in the proposed dKV-Cache mechanism for diffusion language models.

**Cache Refreshing Mechanism.** While it is possible to apply caching after each token is decoded and reuse it throughout the denoising process, in practice, when the sequence is sufficiently long, occasionally recomputing the cache incurs small computational overhead. To maintain consistency and improve correctness during decoding, we add a cache refreshing mechanism. Every N steps, the stored cache would be discarded and refreshed. The calculation would revert back to the normal calculation, resulting in an empty set  $\emptyset$  to replace  $\mathcal{M}_{t-1}$  for this refresh step in Eq.4.

**dKV-Cache-Prefill and dKV-Cache-PD.** The set  $\mathcal{M}_{t-1}$  can be further divided into two subsets: decoded tokens and always-decoded tokens, i.e., prefill tokens. Our experiments show that prefill tokens primarily attend to each other, indicating limited influence from later tokens. Based on this, we adopt a special strategy, dKV-Cache-Prefill, that caches prefill tokens without refreshing. This design aligns with the disaggregation of prefill and decoding phases [57] for serving DLMs. Building on this, we have another variant, dKV-Cache-PD, which intermittently refreshes only the newly decoded tokens, while keeping the key and values of prefill tokens without any recomputation.

#### 3.4 dKV-Cache-Greedy: Greedy Formulation of dKV-Cache

However, the above method still incurs  $\mathcal{O}(L^3)$  complexity, which is less efficient than the  $\mathcal{O}(L^2)$  complexity of ARs. This is primarily because  $\mathcal{M}_t$  initially consists of the entire sequence with L tokens and only narrows to a single token at the end. To improve efficiency, it is essential to decouple  $|\mathcal{M}_t|$  for each step from the sequence length L.

Instead of the minimally refreshed caches proposed above, we adopt a more relaxed cache mechanism to refresh more so as to mitigate the performance degradation caused by stale  $\mathbf K$  and  $\mathbf V$ . Building on our earlier observation that token representations undergo significant changes at their decoding step, we define  $\mathcal M_t$  to include only three components: the token at the current step  $D_t$ , the token from the previous step D(t-1) (motivated by one-step delayed caching), and a local window  $\mathcal W(t)$ . For this local window, we extend it to include the token itself and its neighboring tokens within a fixed-size window  $\mathcal W_t = \left\{x_i \mid i \in \left[D_t - \left\lceil \frac{w}{2} \right\rceil, D_t + \left\lfloor \frac{w}{2} \right\rfloor\right]\right\}$ , where w is the window size. We evaluated local windows centered at both  $D_t$  and  $D_{t-1}$ , and found that the latter yields better performance. Since the window size  $|\mathcal W_t|$  is fixed (set to at most 6 in our experiments), this strategy introduces additional computation but retains an overall time complexity of  $\mathcal O(L^2)$ .

#### 4 Experiments

## 4.1 Experimental Setup

We tested our method under the original evaluation benchmark of LLaDA [37] and Dream [52]. **Datasets:** We conduct comprehensive evaluations across a diverse set of benchmarks that assess

<span id="page-6-2"></span>Table 1: Benchmark results on LLaDA-8B-Instruct. We use zero-shot evaluation here. Detailed configuration is listed in the Appendix. We set the cache refresh step for dKV-Cache-Decode to be 8 and dKV-Cache-Greedy to be 2. The window size of dKV-Cache-Greedy is listed in the bracket.

| Remasking | Base<br>(random) | Few-Steps<br>(random) | dKV-Cache-Greedy<br>(random) | dKV-Cache-Greedy<br>w/ Window (random) | Base (confidence) | Half-Steps<br>(confidence) | dKV-Cache-Decode<br>(confidence) |
|-----------|------------------|-----------------------|------------------------------|----------------------------------------|-------------------|----------------------------|----------------------------------|
| MMLU      | 51.79            | 43.19                 | 45.77                        | 47.70 (4)                              | 51.11             | 51.11                      | 51.00                            |
|           | 30.20            | 47.49 (1.67×)         | 50.56 (1.57×)                | 45.72 (1.51×)                          | 28.27             | 55.80 (1.97×)              | 66.52 (2.35×)                    |
| GSM8K     | 72.25            | 65.58                 | 67.93                        | 68.23 (4)                              | 77.56             | 77.91                      | 78.85                            |
|           | 15.16            | 24.08 (1.59×)         | 25.47 (1.68×)                | 24.76 (1.63×)                          | 14.31             | 28.71 (2.00×)              | 27.50 (1.92×)                    |
| Math500   | 27.4             | 21.8                  | 26.0                         | 27.0 (4)                               | 36.6              | 34.2                       | 36.8                             |
|           | 12.00            | 19.36 (1.61×)         | 20.34 (1.70×)                | 19.86 (1.66×)                          | 11.53             | 23.10 (2.00×)              | 24.46 (2.12×)                    |
| GPQA      | 27.46            | 24.78                 | 26.79                        | 28.35 (4)                              | 30.80             | 27.68                      | 28.13                            |
|           | 11.40            | 18.59 (1.63×)         | 19.27 (1.69×)                | 18.26 (1.60×)                          | 11.86             | 23.88 (2.01×)              | 28.73 (2.42×)                    |
| HumanEval | 19.88            | 15.61                 | 15.13                        | 15.37 (4)                              | 39.63             | 33.54                      | 46.34                            |
|           | 7.50             | 12.50 (1.67×)         | 12.31 (1.64×)                | 12.13 (1.62×)                          | 7.08              | 14.18 (2.00×)              | 13.76 (1.83×)                    |
| MBPP      | 21.4             | 15.6                  | 17.8                         | 20.4 (2)                               | 40.4              | 33.8                       | 40.4                             |
|           | 7.51             | 12.97 (1.73×)         | 12.55 (1.67×)                | 12.44 (1.66×)                          | 7.50              | 15.01 (2.00×)              | 13.93 (1.86×)                    |

<span id="page-6-3"></span>Table 2: Benchmark results on Dream-Base-7B. We use the few-shot ICL here and the configuration is in Appendix. We set the cache refresh interval for dKV-Cache-Decode and dKV-Cache-PD to 4.

|                                  |         | Dream-7B              | Half-Steps            | dKV-Cache-Decode       | dKV-Cache-Prefill      | dKV-Cache-PD          |
|----------------------------------|---------|-----------------------|-----------------------|------------------------|------------------------|-----------------------|
| GSM8K<br>(8-shot)                | T = 256 | 76.88<br>15.1 (1.00×) | 68.08<br>30.3 (2.00×) | 76.57<br>31.6 (2.09×)  | 75.66<br>53.6 (3.55×)  | 74.07<br>50.2 (3.32×) |
| L = 256                          | T = 128 | 68.81<br>30.3 (2.01×) | 46.63<br>60.5 (4.01×) | 65.35<br>62.31 (4.13×) | 65.96<br>107.4 (7.11×) | 63.31<br>99.5 (6.6×)  |
| MBPP<br>(3-shot)                 | T = 512 | 55.8<br>5.4 (1.00×)   | 45.2<br>10.8 (2.00×)  | 53.4<br>10.4 (1.93×)   | 55.2<br>13.6 (2.52×)   | 51.0<br>14.5 (2.69×)  |
| L = 512                          | T = 256 | 45.2<br>10.8 (2.00×)  | 26.2<br>21.5 (3.98×)  | 43.4<br>20.6 (3.81×)   | 41.8<br>27.1 (5.02×)   | 42.6<br>28.9 (5.35 ×) |
| HumanEval<br>(0-shot)<br>L = 512 | T = 512 | 57.93<br>10.3 (1.00×) | 37.20<br>20.5 (1.99×) | 57.32<br>15.5 (1.50×)  | 56.10<br>14.4 (1.40×)  | 59.76<br>17.4 (1.69×) |
|                                  | T = 256 | 37.20<br>20.5 (1.99×) | 18.29<br>40.9 (3.97×) | 31.70<br>31.1 (3.02×)  | 33.54<br>28.7 (2.79×)  | 31.70<br>34.8 (3.38×) |

general language understanding [23], mathematical reasoning [10, 30, 39], and code generation [8, 4]. Since the multi-choice evaluation based on the token likelihood doesn't require more than one step for inference, we request the models to generate the answer letter and match the generated answer with the ground-truth answer. **Evaluation:** We follow the prompt in simple-evals<sup>2</sup> for LLaDA, making the model reason step by step. On Dream, we follow the evaluation setting of Dream to conduct few-shot in-context learning [6]<sup>3</sup>. Other implementation details are listed in the Appendix. **Baseline:** We choose the few-step sampling method (50% steps for Half-Steps and 62.5% steps for Few-Steps) as our baseline and select the number of steps such that their sampling speed is comparable to or slower than ours, and showing that our method can have better performance.

**Metric.** We report the accuracy for performance and token/s for speed. We tested the speed on A6000 (for LLaDA) and H20 (for Dream). Besides this, we use one more metric to show the cache ratio, calculated as:  $\frac{1}{T} \sum_{i=1}^{T} \left| \mathcal{T}_i^{\text{cache}} \right| / |\mathcal{T}_i|$ , where T is the total number of decoding steps.  $\mathcal{T}_i$  denotes the number of tokens processed at step i (normally the whole sequence) and  $\mathcal{T}_i^{\text{cache}}$  is the subset of tokens whose KV pairs are reused from cache at timestep i, which is  $|\mathcal{M}_i|$ .

#### 4.2 Performance and Speed with dKV-Cache

We begin by addressing the central question: What are the performance trade-offs and speedups introduced by applying dKV-Cache? For LLaDA, we evaluate two variants, dKV-Cache-Greedy and dKV-Cache-Decode, against baselines that accelerate generation by halving or reducing the number of denoising steps. As shown in Table 1, dKV-Cache-Greedy consistently outperforms few-step

<span id="page-6-0"></span><sup>&</sup>lt;sup>2</sup>https://github.com/openai/simple-evals

<span id="page-6-1"></span><sup>&</sup>lt;sup>3</sup>https://github.com/EleutherAI/lm-evaluation-harness

<span id="page-7-0"></span>Table 3: Results on long prefill settings. dKV-Cache-Decode uses a refresh step of 4; dKV-Cache-Prefill never refreshes.

|                           |         | Dream-7B              | Half-Steps            | dKV-Cache-Decode      | dKV-Cache-Prefill                  |
|---------------------------|---------|-----------------------|-----------------------|-----------------------|------------------------------------|
| MMLU<br>(5-shot)<br>L = 8 | T = 8   | 72.19<br>9.1 (1.00×)  | 72.21<br>18.1 (1.99×) | 71.74<br>25.2 (2.77×) | 71.76<br>57.6 (6.33×)              |
|                           | T = 4   | 72.21<br>18.1 (1.99×) | 71.63<br>36.1 (3.97×) | 71.69<br>49.2 (5.41×) | 71.71<br>67.3 <mark>(7.40×)</mark> |
| GPQA<br>(5-shot)          | T = 128 | 36.83<br>7.4 (1.00×)  | 35.49<br>14.7 (1.99×) | 35.71<br>18.2 (2.46×) | 35.27<br>75.40 (10.19 ×)           |
| L = 128                   | T = 64  | 35.49<br>14.7 (1.99×) | 35.27<br>29.4 (3.97×) | 34.15<br>36.8 (4.97×) | 35.27<br>139.9 (18.91×)            |

![](_page_7_Figure_2.jpeg)

Figure 3: Effect of onestep delayed caching

baselines across most benchmarks, except for HumanEval. Notably, integrating a lightweight cache window yields substantial gains with negligible computational overhead. For dKV-Cache-Decode, dKV-Cache-Decode achieves near-lossless performance with a high cache ratio and only a few refresh steps. Among all strategies, dKV-Cache-Decode delivers the best trade-off, outperforming both dKV-Cache-Greedy and the baselines. Crucially, it maintains accuracy nearly indistinguishable from the full model, demonstrating that KV-Cache can also be applied in diffusion language models without sacrificing performance. Since dKV-Cache-Greedy relies on a predefined (e.g., random) decoding order, which results in a marked accuracy drop relative to low-confidence remasking, we concentrate our experiments more on dKV-Cache-Decode. We provide several case studies in the Appendix that compare the generated text before and after applying dKV-Cache.

The results on Dream are shown in Table 2 and Table 3. There is a small difference in the position of the decoded token since Dream is adapted from auto-regressive models and shifts the token position. We provide a detailed illustration in the appendix for this. Due to the use of few-shot in-context learning, the model requires a long input context, leading to significant overhead from encoding those tokens repeatedly. In this setting, dKV-Cache-Prefill provides substantial speed improvements; for instance, on MMLU and GPQA, it achieves up to a  $10\times$  acceleration. Across all tested datasets, we observe that dKV-Cache largely outperforms the baseline under different prefilling and decoding lengths. We further evaluate the impact of applying dKV-Cache to few-step diffusion models and observe consistent trends: as the number of diffusion steps increases, our method yields even larger gains over the baseline. For example, on GSM8K with a decoding length of 256, the baseline model with 64 steps achieves 46.63 Pass@1 with a  $4\times$  speedup, whereas dKV-Cache attains a  $6.6\times$  speedup while significantly improving performance to 63.31 (+16.68).

## 4.3 Analysis

**The one-step delay in dKV-Cache.** Figure 3 illustrates the impact of applying a one-step delay to the cache mechanism. Without the delay, performance remains acceptable at low cache ratios but degrades rapidly as the cache ratio increases, ultimately collapsing to near-zero accuracy. In contrast, introducing a one-step delay stabilizes the generation quality, enabling the model to maintain nearly lossless performance even under high cache ratios.

**Performance on different decoding length, denoising steps and refreshing steps.** Figure 4 presents the results of applying dKV-Cache-Decode and dKV-Cache-Greedy under various configurations, including different decoding lengths, refresh intervals, sampling steps, and window sizes. Overall, our method consistently achieves performance comparable to the original model without dKV-Cache, effectively pushing the Pareto front forward. We observe the following findings: (1) Decoding robustness: The performance impact of our method is largely insensitive to the number of decoding steps, indicating strong robustness across varying generation lengths and sampling steps. (2) Enhanced long-form generation: In tasks involving longer outputs (e.g., L = 512), our method outperforms the baseline, even improving generation quality from 80.97% to 83.13% on GSM8K and from 39.63% to 46.34% for HumanEval. The results imply a potential inefficiency in how bidirectional attention aggregates contextual signals, pointing to redundancy or underutilization in long-context modeling. (3) Effectiveness with only a few refreshing: Even with infrequent refreshes (e.g., every 16 steps), the performance degradation remains small. (4) Effectiveness of local windows:

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 4: dKV-Cache-Decode(left) and dKV-Cache-Greedy(right) on GSM8K with different settings: decoding length *L*, sampling steps *S*, refresh intervals and the window size.

![](_page_8_Figure_2.jpeg)

Figure 5: Speed and memory for dKV-Cache-Decode and dKV-Cache-Greedy. The number (2 and 8) means that in every n steps, the cache would be refreshed.

Incorporating a local window notably enhances the performance of dKV-Cache-Greedy with minimal additional computational cost.

Memory and speed analysis. We analyze the speed and memory footprint of dKV-Cache-Decode and dKV-Cache-Greedy across varying decoding and prefill lengths. Our method achieves substantial inference acceleration, ranging from  $1.75 \times$  to  $3.3 \times$ , while introducing only a modest increase in memory usage. Notably, dKV-Cache-Greedy demonstrates greater potential for accelerating inference while dKV-Cache-Decode would be capped. In our main experiments, we observed that setting the refresh interval larger than 2 for dKV-Cache-Greedy may degrade performance. However, under the same refresh interval, dKV-Cache-Greedy consistently achieves higher speedups than dKV-Cache-Decode, highlighting its potential advantage when refresh frequency is relaxed.

## 5 Conclusions and Limitations

In this work, we explore the feasibility of incorporating the caching mechanism into diffusion language models. Specifically, we propose a delayed KV-Cache for DLMs, motivated by our empirical observations on the dynamics of the token representations throughout the diffusion process. Building on this insight, we introduce two cache variants, dKV-Cache-Decode and dKV-Cache-Greedy, each designed to leverage delayed caching for improved compatibility with diffusion-based generation. Our analysis reveals that introducing a delay is crucial for the cache to function effectively in this setting. Extensive experiments demonstrate that our approach largely accelerates inference while maintaining model performance.

One primary limitation of this work lies in its focus on algorithmic design in isolation. While our proposed method introduces an effective caching mechanism from a purely algorithmic perspective, diffusion language models also exhibit substantial room for improvement at the system level. We believe that future research integrating algorithmic innovations with system-level optimization, such as memory management, parallelism, and hardware-aware execution, could unlock further efficiency gains and performance improvements for DLMs.

