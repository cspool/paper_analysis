# **1 Introduction**

Variations on linear attention [\(Katharopoulos et al.,](#page-11-0) [2020\)](#page-11-0) have proliferated in recent research [\(Peng et al.,](#page-12-0) [2024;](#page-12-0) [Qin et al.,](#page-12-1) [2024;](#page-12-1) [Katsch,](#page-11-1) [2024;](#page-11-1) [Yang et al.,](#page-15-0) [2024\)](#page-15-0), approaching the performance of traditional Multi-Headed Scaled Dot Product Attention (MHA) [\(Vaswani et al.,](#page-15-1) [2023\)](#page-15-1) while achieving lower inference costs. In MHA, the model's effective memory is bounded by its context length, with the attention calculation resulting in quadratic time complexity with regard to that length. Conversely, most forms of linear attention can be computed recurrently in O(1) time per time-step. Instead of inspecting the entire context length to generate each new token, recurrent linear attention uses a fixed-size hidden state that is updated at each time-step, functioning as its memory of the past. The limited size of this state constrains the capacity of this memory.

The success of Large Language Models (LLMs) has motivated interest in ultra-long context length language models. For example, Gemini Pro [\(Team et al.,](#page-13-0) [2024\)](#page-13-0) offers a 1 million+ token length window. However, if based on attention, these extra large context lengths come with large associated costs due to the need for MHA to examine every prior token within the context when generating a next token [\(Liu & Abbeel,](#page-12-2) [2023;](#page-12-2) [Liu et al.,](#page-12-3) [2023\)](#page-12-3). Although a naive inference implementation would recalculate every key and value at every layer in a traditional transformer, it is common practice to store these in a key-value cache ("KV-Cache")[\(Pope et al.,](#page-12-4) [2022\)](#page-12-4) and retrieve rather than recompute them. KV-Cache memory costs can be very high. For example, a 1 million token cache for an 80 layer traditional transformer model of hidden dimension 8192 would take up over 2.5 terabytes at bfloat16 precision. We turn our focus to reducing the memory costs of this cache while also reducing computational complexity and memory usage for processing the initial context of a request.

Our contribution is the combination of several innovations to create the GoldFinch architecture, which improves pre-fill and decoding efficiency, as well as downstream modeling performance, and introduces the following innovations:

<span id="page-0-0"></span><sup>1</sup>Code at: <https://github.com/recursal/GoldFinch-paper> Model weights at: <https://huggingface.co/recursal/GoldFinch-paper>

- 1. employs a novel parameter-efficient modification of Finch (RWKV-6), which we call "Finch-C2", for the first 2/3 of its layers
- 2. uses these the output of these Finch-C2 layers to produce an extremely small *compressed* global key cache using a novel mechanism we call "TokenCat". Our cache thus requires only  $\frac{1}{16}d_{model}$  per token plus the original input token indices, instead of  $2d_{model}n_{layer}$  for traditional KV-caches.
- 3. employs a novel modification of the traditional transformer architecture, which we call "GOLD", for the last 1/3 of its layers to consume this key cache and produce outputs without even requiring a traditional value cache.

![](_page_1_Figure_3.jpeg)

Figure 1: GoldFinch Architecture Block Diagram

| Architecture    | Pre-fill time<br>complexity<br>per token | KV-Cache<br>entries<br>per token | KV-Cache Bytes<br>256k context, 32 layers<br>4096 hidden dim |
|-----------------|------------------------------------------|----------------------------------|--------------------------------------------------------------|
| Llama2          | O(N)                                     | $2d_{model}n_{layer}$            | 128GB                                                        |
| Llama3 (w/ GQA) | O(N)                                     | $8d_{head}n_{layer}$             | 32GB                                                         |
| DeepSeek-V2     | O(N)                                     | $\frac{9}{2}d_{head}n_{layer}$   | 18GB                                                         |
| Zamba           | O(N)                                     | $\frac{2}{7}d_{model}n_{layer}$  | 18.3GB                                                       |
| Jamba           | O(N)                                     | $\frac{8}{7}d_{head}n_{layer}$   | 4GB                                                          |
| YOCO            | <b>O</b> (1)                             | $2d_{model}$                     | 4GB                                                          |
| GoldFinch       | <b>O</b> (1)                             | $1 + \frac{d_{model}}{16}$       | 0.068GB                                                      |

Table 1: Time and space complexity comparisons of models with full softmax attention. No KV-Cache quantization is shown.

The GOLD layers are an adaptation of a novel improved transformer we call "GPTAlpha" that can also be used as a standalone transformer model for improved non-hybrid performance.

This new architecture brings a series of significant benefits:

1. We are able to reuse the same KV-Cache on every transformer layer while maintaining greater than Llama (Touvron et al., 2023) performance. This reduces the KV-Cache size by a factor of the total number of layers of the model.

- 2. We eliminate the values from the KV-Cache, leaving only a key cache. Instead of caching values, we store the input indices and generate the values from these, reducing the KV-Cache size by another factor of nearly 2.
- 3. We are able to compress our key cache by applying a form of Low Rank Adaptation (LoRA) [\(Hu et al.,](#page-11-2) [2021\)](#page-11-2) to the output of a single layer, and re-expanding the compressed version by concatenating the compressed version with the original token embeddings, further reducing the size by 128 times. ("TokenCat")
- 4. We use the input embedding table and RWKV-style token shift to generate values for attention without sacrificing performance.
- 5. By using Finch-C2 blocks at the start of the model, the key cache automatically encodes the underlying implicit positional representation, thereby removing the need for positional encoding within our transformer layers for trained context lengths. We do still require an additional positional encoding method for extrapolation to new context lengths unseen during training.
- 6. There are many use cases of LLMs that involve relatively short responses to questions about long documents. Because our compressed key cache is generated by an RNN with an operating time and space complexity of O(1) per token with regard to sequence length, we are able to generate the cache in these cases extremely inexpensively and apply the O(N) per token cost GOLD transformer portion of our calculations only to new token generation, for which relatively few iterations are often required.

To obtain our Finch-C2 architecture we improve the Finch time-mixer by removing the gate, swapping out GroupNorm for a LayerNorm across all heads, doing a new multiplication (of the key by one minus the decay) to keep the kv-state rows normalized, and replacing Finch's *u* ("bonus") term with a new data-dependent separately token-shifted second Value. These changes result in improved performance with little to no speed penalty and significantly fewer total parameters.

To obtain our GPTAlpha architecture we improve the Llama architecture by replacing the transformer feed-forward network (FFN) with the RWKV channel mixer, and adding RWKV style token shifts and extra LayerNorms to attention layers.

Both Finch-C2 and GPTAlpha can be used either as standalone model architectures with improved performance over their counterparts, or as part of the GoldFinch hybrid model architecture.

The GOLD transformer architecture (GPTAlpha Over Linear transformer Decoder) removes the key and value weights from GPTAlpha in favor of producing keys and values from a combination of the original token indices passed through the embeddings table, a highly compressed version of the outputs of the Finch-C2 layers, and a data-driven LoRA.

GoldFinch stacks a set of GOLD transformer layers on top of a Finch-C2 linear transformer, passing the outputs for the Finch layers both into a key compressor to be stored for every sequence position, and also through the current timestep as part of the normal residual stream.

We train GoldFinch models up to 1.45 billion parameters on 1.5 trillion tokens of *minipile* [\(Kaddour,](#page-11-3) [2023\)](#page-11-3) and compare them to slightly larger equivalently trained Finch [\(Peng et al.,](#page-12-0) [2024\)](#page-12-0) and Llama [\(Touvron et al.,](#page-15-2) [2023\)](#page-15-2) models. We find that GoldFinch significantly outperforms both Llama and Finch in downstream performance and perplexity across nearly every benchmark we tested, while maintaining fewer parameters, a much smaller cache than Llama, and perfect MQAR recall due to its use of full attention.

## **2 Background**

Transformers have become the de-facto choice for most sequence modeling tasks, and have been shown to be especially effective in the context of language modeling. However, they present computational challenges when processing long context lengths, which has hindered their adoption for long sequence tasks. Specifically, the formulation of multi-head scaled dot-product attention (MHA) has a computational complexity of O(*N* 2 ) with respect to context length. Additionally, inference engines typically rely on the use of a KV-Cache to enable autoregressive token generation in O(N) time per token. This cache grows linearly with context length, and becomes challenging to fit into limited Video Random-Access Memory (VRAM) for longer sequences.

Recent transformer models such as the Llama series rely on Grouped-Query Attention (GQA) (Ainslie et al., 2023) to help ameliorate this cache size problem. At a typical number of groups  $n_g = 8$ , GQA reduces the KV-Cache size by  $\frac{n_g}{n_h}$  times, where  $n_h$  is the number of heads. This is helpful, especially on consumer grade hardware, but leads to a reduction in downstream performance, and longer sequences still cause a significant problem in terms of VRAM usage.

The recently proposed YOCO (Sun et al., 2024) improves the computational complexity for pre-fill of the initial request context and also reduces the KV-cache size by introducing a new global KV-Cache instead of the usual per-layer cache. The computational improvement is achieved by replacing the first half of the layers in the model with Linear Attention based RetNet-G layers (Sun et al., 2023), which is a recurrent neural network (RNN) architecture that requires only linear time with respect to sequence length. YOCO stores the output of these first layers as a global KV-Cache, which is then used by the second half of the layers, featuring MHA. Overall, this reduces the KV-Cache size by a factor of the number of layers, without a reported performance reduction. Goldfinch takes a related approachetNet-G, and processes the output differently, creating an effective but much smaller cache via our TokenCat mechanism, which is then consumed by our enhanced transformer GOLD layers.

Hungry Hungry Hippos (H3) (Fu et al., 2023) train a hybrid recurrent SSM/transformer model containing just two layers of attention, which they find outperforms transformers. This served as a warning shot that SSM(or linear attention)-transformer hybrids have the potential to step in as higher performance replacements for transformers alone.

Recognizing the challenges posed at inference time by the KV-Cache, DeepSeek-V2 (DeepSeek-AI et al., 2024) proposes a replacement for MHA called Multi-head Latent Attention (MLA). This uses low-rank joint key-value compression to reduce the size of the KV-Cache from  $2n_hd_hl$  to  $\frac{9}{2}d_hl$ , equivalent to the KV-Cache size required for GQA with only 2.25 groups. Because the low-rank key-value compression requires fewer parameters than full rank key and value matrices, MLA achieves greater per-parameter performance than MHA. GoldFinch also improves performance via this kind of compression-based relative parameter reduction.

HGRN2 (Qin et al., 2024) replaces the per-head GroupNorm (Wu & He, 2018) with a full-width LayerNorm, and we do the same in our Finch-C2 architecture. HGRN2 sets their key to be equal to one minus the decay, and we do something related but slightly different, multiplying our key by one minus the decay.

Inspired by these works, we propose a new method that further reduces the KV-Cache by orders of magnitude and reduces the cost of the initial context load to become linear with respect to sequence length, all while achieving greater than Llama performance.

#### 2.1 Other Concurrent Related Work

Other concurrent work on hybrid models bear some similarities to portions of our architecture:

Zamba (Glorioso et al., 2024) interleaves Global Shared Attention (GSA) every N Mamba blocks (Gu & Dao, 2024). Instead of using the residual output of the prior Mamba block as its input, Zamba concatenates the original embeddings generated before layer zero onto this residual output, and use the double-width combination as the input to attention. Although their GSA blocks share parameters, they are not able to share the same KV-Cache. The concatenation of embeddings bears similarity to our new "TokenCat" technique.

Jamba (Lieber et al., 2024) is a mixture-of-experts (MoE) (Shazeer et al., 2017) Mamba-based (Gu & Dao, 2024) model that inserts attention layers periodically within its architecture, for a total of 1:7 ratio of attention-to-Mamba layers. Similarly to Goldfinch's ability to rely upon RWKV's implicit positional encoding within the pre-trained context length, they find that explicit positional encoding may not be required for their hybrid Mamba-based architecture.

Samba (Ren et al., 2024) is a hybrid model that repeats blocks containing a Mamba layer, an MLP layer, a sliding-window attention (SWA) layer featuring RoPE (Su et al., 2023), and another MLP layer. The use of SWA allows a fixed cost of execution per token, regardless of context length.

#### 3 Method

GoldFinch follows the general structure of the Finch architecture, which is also the common pre-norm decoder transformer structure used in Llama and RWKV. It consists of a series of layers, each containing a time mixing sub-layer followed by a channel mixing sub-layer. All channel mixing sub-layers are Finch channel mixers.

The following formulae describe the three varieties of GoldFinch sub-layers. All matrices W are learned per layer, unless described otherwise. We show all time mixing formulae per-head for conciseness, except the formulae for those layer outputs where heads are combined via concat. Model dimension is denoted as D, head size as H, and number of heads as N. All values are  $\in \mathbb{R}^H$  unless otherwise noted.

#### 3.1 Finch-C2 Time Mixing

The first two-thirds of time mixing sub-layers use a variation on the Finch time mixer we call Finch-C2.

We customize the Finch time-mixing sub-layers by removing the gate, swapping out GroupNorm for a LayerNorm across all heads and doing a new multiplication of the key by one minus the decay. Finally, we replace Finch's u ("bonus") term with a new data-dependent separately token-shifted second Value, computed using the same weights as the base Value, with an additional LoRA added to the result. We find that this allows us to remove all of the Gate parameters while retaining performance.

Along the lines of (Peng et al., 2024), we introduce the following notation for common operators in the model, using the square subscript to denote a variable:

$$\operatorname{lerp}(a, b, t) = a + (b - a) \circ t, \tag{1}$$

$$lora_{\square}(x) = \lambda_{\square} + tanh(xA_{\square})B_{\square}, \tag{2}$$

$$ddlerp_{\square}(a,b) = a + (b-a) \odot lora_{\square}(a + (b-a) \odot \mu_x), \tag{3}$$

Then, the Finch-C2 block can be formalized as:

$$d_t = \text{lora}_!(\text{ddlerp}_d(x_t, x_{t-1})), \tag{4}$$

$$w_t = \exp(-\exp(d_t)),\tag{5}$$

$$r_t = \mathrm{ddlerp}_r(x_t, x_{t-1}) \mathbf{W}^R, \tag{6}$$

$$k_t = \operatorname{ddlerp}_k(x_t, x_{t-1}) \mathbf{W}^K \cdot (1 - w_t), \tag{7}$$

$$v_t = \text{ddlerp}_{i,i}(x_t, x_{t-1}) \mathbf{W}^V, \tag{8}$$

$$u_t = ddlerp_u(x_t, x_{t-1}), \tag{9}$$

$$u_t' = u_t W^V + \tanh(u_t W^{UD}) W^{UU}. \tag{10}$$

(11)

And after splitting the hidden dimension into *N* heads:

$$wkv_t = \sum_{i=1}^{t-1} \operatorname{diag}\left(\bigcap_{j=i+1}^{t-1} w_j\right) \cdot k_i^{\mathrm{T}} \cdot v_i \in \mathbb{R}^{H \times H}, \tag{12}$$

$$o_t = \text{LayerNorm}(\text{concat}(r_t \cdot wkv_t + u_t'))W^O \in \mathbb{R}^D.$$
 (13)

Please note that the calculation for  $u'_t$  reuses the same weights  $\mathbf{W}^V$  - this is an intentional parameter count savings and not a typo.

#### 3.2 GOLD Key Compression

The output from the first two-thirds of the model is used in two ways: it is passed on to the next layer in the usual manner, and also compressed down via multiplication with the global (not per-layer) learned matrix  $W^{KD} \in \mathbb{R}^{Dx(D/16)}$  to one sixteenth its original size and stored into a unified single-layer compressed key cache:

$$c_t = x_t \mathbf{W}^{KD} \in \mathbb{R}^{(D/16)}. \tag{14}$$

#### 3.3 GOLD Key Decompression (TokenCat)

The compressed key cache is decompressed via a two-step method. The first step is "TokenCat", short for "Token conCatenation", in which the compressed key is concatenated with the original input token embedding from the very beginning of the model. The concatenated result is then multiplied with the global (not per-layer) learned matrix  $W^{KU} \in \mathbb{R}^{(D+D/16)xD}$  and RMSNormed to obtain the decompressed attention proto-keys, which are common to all GOLD attention sub-layers.

$$k_t^D = \text{RMSNorm} \left( \text{concat} \left( x_t^0, c_t \right) W^{KU} \right). \tag{15}$$

### 3.4 GOLD Attention Time Mixing

The remaining time mixing sub-layers are a variation on GPTAlpha attention sub-layers employing MHA that we call GOLD attention.

Each GOLD attention sub-layer calculates its own unique attention keys and values from the decompressed proto-keys and the original input token embeddings, respectively. Each is passed through a data-dependent token shift, with the result passed through an additive LoRA. We call this process "DDLoRAdapt", introducing the relevant notation below, using the square subscript to denote a variable:

$$\operatorname{loradapt}_{\square}(x) = x + \tanh(xC_{\square})D_{\square}. \tag{16}$$

The following are the formulae for GOLD attention time mixing:

$$q_t = \text{LayerNorm}(\text{ddlerp}_q(x_t, x_{t-1})W^Q), \tag{17}$$

$$a_t = \text{lerp}(x_t^0, x_{t-1}^0, \mu_x),$$
 (18)

$$k_t = \text{LayerNorm} \left( \text{loradapt}_k \left( \text{lerp} \left( k_t^D, k_{t-1}^D, \text{lora}_k \left( a_t \right) \right) \right) \right),$$
 (19)

$$v_t = \text{LayerNorm} \left( \text{loradapt}_v \left( \text{lerp} \left( x_t^0, x_{t-1}^0, \text{lora}_v \left( a_t \right) \right) \right) \right), \tag{20}$$

$$o_t = \text{LayerNorm}(\text{concat}(\text{attention}(q_t, k, v)))W^O \in \mathbb{R}^D.$$
 (21)

Please note the receptance-like Finch style token-shift on queries, and additional data-driven token-shift on keys and values, with keys being reconstituted from compressed key cache entries  $c_t$  and values coming from the original token embeddings  $x^0$ .  $x^0$  is the embedding input to the first sub-layer in the model, and can be reconstituted during inference from the token indices by storing those indices, usually only an additional two bytes per context length.

Data dependent token shift (ddlerp) is a specialized low-parameter cost variety of two-step 1D convolution that originated in the RWKV architecture. It allows the model to dynamically linearly interpolate between the current and previous time-step on a per channel basis. We use our DDLoRAdapt version of the technique to inexpensively apply contextual information to the keys and values, increasing the amount of information from which they are generated without significantly increasing parameter count.

Note that the token shift cannot be dependent on the hidden-state, as that would make recurrent calculation impossible for older keys and values, and would require a full KV-Cache to be stored. Instead, we use the original input token embeddings as the data upon which the key and value token-shifts depend.

Pre-fill of the compressed key cache to prepare for autoregressive generation can be computed in linear time with respect to the number of tokens. This is accomplished by running only the Finch-C2 section of the model on those tokens. One important implementation caveat is that token shift requires the prior layer hidden-state output from the previous time-step. At first glance this appears problematic, as the GOLD layers require full quadratic attention, which is what we were trying to avoid during pre-fill. But the solution is simple: given G GOLD layers in the model, there must be 2G-1 sub-layers that require such a previous time-step hidden state but are directly or indirectly reliant on the outputs of quadratic attention. Therefore, the last 2G-1 tokens of pre-fill must be run through the full model (not just the Finch-C2 layers) to generate these hidden-states. These 2G-1 computations can be done in a single call to the full model to leverage the same kinds of parallelism used during training.

Only the compressed key cache entries and original input token indices must be permanently kept in VRAM during inference, as the key cache can be reconstituted via decompression on-demand.

Because decompression and token shift can be done on contiguous regions of key value pairs instead of all of them at once, extremely low VRAM usage can be achieved during inference by calculating attention incrementally across the sequence for each layer and decompressing as you go.

### 3.5 GoldFinch Channel Mixing (same as Finch Channel Mixing)

Goldfinch channel mixing is identical to Finch channel mixing. It is used as the feed forward network component on all layers of the model, both Finch-C2 and GOLD. We reproduce it here for reference. Please note that variables have their own independent definitions in this subsection.

$$r_t = \operatorname{lerp}_r(x_t, x_{t-1}, \mu_r) \mathbf{W}^R \in \mathbb{R}^D, \tag{22}$$

$$k_t = \text{lerp}_k(x_t, x_{t-1}, \mu_k) \mathbf{W}^K \in \mathbb{R}^{3.5D},$$
 (23)

$$\nu_t = \text{ReLU}(k_t)^2 \mathbf{W}^V \in \mathbb{R}^D, \tag{24}$$

$$o_t = \sigma(r_t) \odot \nu_t \in \mathbb{R}^D. \tag{25}$$

## 3.6 GPTAlpha Time Mixing

For completeness and to show how it can be used in a pure transformer architecture, we list the formulae for GPTAlpha time mixing when not used in conjunction with TokenCat below:

$$q_t = \text{LayerNorm}(\text{ddlerp}_q(x_t, x_{t-1})W^Q), \tag{26}$$

$$k_t = \text{LayerNorm}(\text{ddlerp}_k(x_t, x_{t-1})W^K),$$
 (27)

$$v_t = \text{LayerNorm}(\text{ddlerp}_v(x_t, x_{t-1})W^V),$$
 (28)

$$o_t = \text{LayerNorm}(\text{concat}(\text{attention}(q_t, k, v))) W^O \in \mathbb{R}^D.$$
 (29)

## 4 Experiments

### 4.1 Architecture Comparisons

We trained 1.5B parameter-class models with 24 layers, 2048 hidden-dimension, 2048 context length of Finch, Llama, and GoldFinch for comparison on *minipile* (Kaddour, 2023), all using the same RWKV World tokenizer. GoldFinch ends with dramatically lower final loss than the others (by over 0.1 out of 2.39), and uses over 100 million fewer parameters than its Finch counterpart.

We additionally trained a GoldFinch with no compression, to show that there is very little lost with our choice of a 16:1 hidden-dimension compression ratio.

In the interest of fairly comparing performance for Llama by giving it the most favorable conditions, we add the RWKV small init embeddings optimization (LayerNorm after embeddings with small initialized values) [\(Peng et al.,](#page-12-9) [2023\)](#page-12-9) and do not employ Grouped Query Attention. All architectures used the same hyperparameters and were trained on 4 GPUs, with per-GPU per-step batch size of 8, two steps of gradient accumulation, and a 10 step learning rate warm-up followed by cosine decay annealed from 3e-5 to 1e-5. We train with Adam betas of 0.9 and 0.99, epsilon 1e-8 and weight decay 0.001. Weight decay was applied only to matrix parameters that are not part of LoRAs or the GoldFinch key compression/expansion steps.

![](_page_7_Figure_2.jpeg)

Figure 2: Loss curves of 1.5B class models.

| Architecture (L24 D2048 ctx2048)                  | Parameters | Loss ↓ |
|---------------------------------------------------|------------|--------|
| Llama                                             | 1.47B      | 2.3905 |
| Finch                                             | 1.60B      | 2.3856 |
| GoldFinch, last 1/3 layers GOLD, 16:1 compression | 1.45B      | 2.2762 |
| GoldFinch, last 1/3 layers GOLD, 1:1 compression  | 1.45B      | 2.2762 |

Table 2: Final loss values for various models of size L24 D2048 ctx2048 trained on *minipile*

In addition to comparing training and validation losses, we ran a series of common benchmark evaluations on the three 1.5B parameter class models trained on *minipile*. Finch and Llama scored similarly to one another, and GoldFinch significantly outperformed both.

| Model           | lmbd<br>ppl ↓ | avg<br>acc ↑ | lmbd<br>acc ↑ | piqa<br>acc ↑ | hella<br>acc ↑ | winog<br>acc ↑ | arc_c<br>acc ↑ | arc_e<br>acc ↑ | sciq<br>acc ↑ |
|-----------------|---------------|--------------|---------------|---------------|----------------|----------------|----------------|----------------|---------------|
| Finch 1.60B     | 81.9          | 42.8%        | 24.3%         | 62.4%         | 28.7%          | 49.0%          | 19.6%          | 44.9%          | 70.8%         |
| Llama 1.47B     | 71.7          | 43.0%        | 26.3%         | 61.6%         | 28.1%          | 50.5%          | 19.3%          | 43.9%          | 71.0%         |
| GoldFinch 1.45B | 48.2          | 44.2%        | 29.1%         | 63.4%         | 29.1%          | 50.2%          | 18.3%          | 45.9%          | 73.7%         |

Table 3: Common benchmark evaluations for various models of size L24 D2048 ctx2048 trained on *minipile*

## **4.2 Ablation Studies**

We ran various smaller scale ablation studies to determine the contributions of different parts of the GoldFinch architecture relative to both Finch, Llama, GPTAlpha, and a hybrid of our improved Finch and GPTAlpha with no KV-Cache compression or key/value sharing. The new second value added in Finch-C2 had the smallest positive impact of anything measured. Surprisingly, GoldFinch performed very slightly better than even the Finch-C2/GPTAlpha hybrid with no KV compression at all. Each test trained a 12 layer 768 hidden-dimension model at 1024 context length with the same RWKV World tokenizer on the full *minipile* dataset. All architectures used the same hyperparameters and were trained on single GPUs, with per-step batch size of 32, two steps of gradient accumulation, and a 10 step learning rate warm-up followed by cosine decay annealed from 6e-5 to 2e-5. We train with Adam betas of 0.9 and 0.99, epsilon 1e-8 and weight decay 0.001. Weight decay was applied only to matrix parameters that are not part of LoRAs or the GoldFinch key compression/expansion steps.

| Architecture (L12 D768 ctx1024)           | Loss ↓ |
|-------------------------------------------|--------|
| Finch-C2 without k∗ =<br>1−w              | 2.7293 |
| Finch                                     | 2.7191 |
| Llama                                     | 2.7125 |
| Finch-C2 without second value             | 2.7105 |
| Finch-C2                                  | 2.7082 |
| GPTAlpha with RoPE                        | 2.6684 |
| GoldFinch, last 1/2 layers GOLD           | 2.6637 |
| GoldFinch, last 1/3 layers GOLD with RoPE | 2.6590 |
| Finch-C2, last 1/3 layers GPTAlpha        | 2.6586 |
| GoldFinch, last 1/3 layers GOLD           | 2.6582 |
| GoldFinch, last 1/6 layers GOLD           | 2.6578 |

Table 4: Final loss values for various ablations of model size L12 D768 ctx1024 trained on *minipile*

#### <span id="page-8-1"></span>**4.3 Associative Recall**

Associative recall (AR) is a synthetic task designed to emulate the human ability to associate and retrieve information. It evaluates a model's skill in recalling previously mentioned information within a given context. Previous studies suggest that a model's performance in AR is a good indicator of its efficacy in in-context learning [\(Elhage et al.,](#page-11-10) [2021;](#page-11-10) [Olsson et al.,](#page-12-10) [2022\)](#page-12-10). Consequently, AR has been employed as a benchmark for developing new language model architectures [\(Fu](#page-11-5) [et al.,](#page-11-5) [2023;](#page-11-5) [Poli et al.,](#page-12-11) [2023;](#page-12-11) [Lutati et al.,](#page-12-12) [2023\)](#page-12-12). [Arora et al.](#page-11-11) [\(2023\)](#page-11-11) evaluated a variety of models for multi-query associative recall (MQAR) and discovered a performance gap between different linear transformer architectures and the traditional transformer with attention.

<span id="page-8-0"></span>![](_page_8_Figure_5.jpeg)

Figure 3: MQAR tasks. An increase in sequence length correlates with increased task difficulty.

In [Figure 3,](#page-8-0) we used the same experimental settings as [Arora et al.](#page-11-11) [\(2023\)](#page-11-11) and show that GoldFinch achieves perfect MQAR scores, outperforming traditional attention-free language models. As a hybrid architecture that leverages attention, GoldFinch can solve MQAR as well as transformer models with attention. Additionally, we trained GoldFinch on a context length of 1024 to demonstrate that this trend continues, as depicted in [Figure 4.](#page-9-0)

#### **4.4 Long Context Experiments**

We tested the loss of our small Finch and GoldFinch models pre-trained on *minipile* at all context lengths up to 65536 on the *PG19* [\(Rae et al.,](#page-12-13) [2019\)](#page-12-13) dataset of older books. These pre-trained models were all trained at only 1024 context length. The Finch model is able to maintain a fairly low loss throughout the 65536 context length. The base GoldFinch model trained with no positional

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 4: Finch and GoldFinch on the same MQAR task with increased sequence length

encoding goes up in loss significantly starting at around double the trained context length, then plateauing at a high loss. The GoldFinch model trained with RoPE on its GOLD attention sublayers performs better, but loss still increases somewhat as the sequence progresses. However, by applying interpolated RoPE values we are able to obtain low loss throughout the extended context length. We conclude that for GoldFinch models in which extrapolation beyond the maximum trained context length is desired, the GOLD attention sub-layers should be trained with RoPE, with interpolation employed upon inference.

We then fine-tuned the RoPE and non-RoPE models mentioned above on 165 million tokens of *minipile* at longer context lengths. During this fine-tuning, we froze the entire RWKV portion of the model up to the first GOLD layer, allowing the optimizer to update the parameters of only the GOLD layers and output head. This saves a significant amount of time and VRAM during fine-tuning, allowing an even longer context length to fit into memory and using roughly 3x fewer FLOPS per token. We theorize that because the GOLD attention portion of the model can use keys generated from the RWKV output, this is enough to support sophisticated attention matching across the entire context length.

Our experiments showed that indeed the RoPE model with GOLD layers fine-tuned at longer context lengths exhibited significantly lower losses against *PG19* up through those lengths and even beyond. On the non-RoPE model this process was somewhat successful within the fine-tuned context length, while still failing at extrapolation. This was unexpected, since the RWKV layers were not updated and the GOLD layers included no positional encoding mechanism. We postulate that token-shift may supply some minimal positional information to the model.

## **4.5 Checkpoint Upgrade Training**

We have attempted upgrading existing pre-trained Finch models to a more limited version of GoldFinch that uses the Finch architecture for its RWKV layers instead of the Finch-C2 component. We tried many variations on two methods, one that adds new GOLD layers on top for a total of around 11% more parameters, and another which keeps the layer count the same as the pre-trained model. Thus far with only small amounts of upgrade training neither method has performed to our satisfaction.

Both methods were attempted on a 1.6B Finch checkpoint that had been pre-trained on 2.5 trillion tokens.

For the first method we appended 4 GOLD layers on top of the pre-trained 1.6B Finch checkpoint before the language modeling head, and continued training it for 100 million tokens using two different learning rates. The original 24 pre-trained layers were kept at the same 1e-5 LR at which their pre-training had ended upon, while the LR for the 4 new GOLD layers was annealed along a cosine schedule from 3e-4 to 1e-5. While the performance of this model was in line with the original model, it was unclear if the resultant model from this method really learned anything of value in its GOLD layers.

The second method involved freezing the embedding and RWKV layers and importing but not freezing the final 1/3 of the channel mixer sub-layers that were paired with freshly initialized GOLD attention sub-layers. We then trained this model on a relatively small amount of data (in our case around 7.5 billion tokens of a new internal dataset) while annealing the learning rate to the final learning rate seen in the pre-trained base model. The resultant model obtained a similar validation loss on *minipile* to the base model, despite being trained on a completely different dataset and the base model having been already trained for over 2.25 trillion tokens. However, the new model's LAMBADA scores were worse. We attribute this loss of performance to the 'brain surgery' required to keep the layer count the same, in which we effectively erased the Finch time-mix parameters in the upper 1/3rd of the model.

We are still doing further experimentation on these upgrade methods to see just how well they can be made to perform. We hope to be able to inexpensively upgrade even the largest 14B Finch model to this reduced GoldFinch format and see significant performance improvements at larger context lengths due to the GOLD attention being able to look back across the entire context with no state-size based memory limitations.

