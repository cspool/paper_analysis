# 3 Redundancy-aware KV Cache Compression (R-KV)

To address the redundant thinking issue, we propose a *redundancy-aware decoding-time KV cache compression method* (**R-KV**) that explicitly targets the compression of redundant tokens in reasoning models. Our approach balances *importance* and *non-redundancy* in token selection, ensuring that KV cache storage is allocated to both highly informative and diverse content. By incorporating

redundancy estimation into the selection process, our method effectively mitigates unnecessary KV cache growth while preserving the model's reasoning capabilities.

Specifically, R-KV consists of three key components: (1) an *importance scoring mechanism* (§3.2) leveraging attention weights, (2) a *redundancy estimation mechanism* (§3.3) based on semantic similarity of key vectors, and (3) a *joint selection strategy* (§3.4) that optimizes cache efficiency by balancing redundancy and importance.

### <span id="page-3-0"></span>3.1 Decoding-time Compression

Different from existing KV cache compression methods[3, 5, 4] that focus on the *prefilling stage* to manage long-context inputs, our R-KV focuses on the *decoding stage* for reasoning models—a distinctive setting where the generated output is significantly longer than the prompt.

Specifically, R-KV allocates memory for two components: a cache of budget size  $B_{\text{budget}}$  to store retained KV tokens, and a buffer of size  $B_{\text{buffer}}$  for newly generated text tokens. The total memory requirement is thus  $B_{\text{total}} = B_{\text{budget}} + B_{\text{buffer}}$ . After the model generates each fixed-length text segment in the buffer, R-KV performs KV cache compression. At the end of each text segment, the last  $\alpha$  tokens are always retained in the cache as **observation tokens**, following prior work [3]. Next, we concatenate the existing  $B_{\text{budget}}$  tokens in the cache with the first  $B_{\text{buffer}} - \alpha$  tokens in the buffer, resulting in  $n = B_{\text{budget}} + B_{\text{buffer}} - \alpha$  candidate KV tokens. Each candidate is assigned a selection score (§3.4), and we select the top  $k = B_{\text{budget}} - \alpha$  tokens to fit in the rest of the cache budget, in addition to the  $\alpha$  observation tokens. This process compresses the KV cache while preserving critical context, enabling efficient memory utilization during autoregressive decoding.

## <span id="page-3-1"></span>3.2 Importance Scoring via Attention Weights

Following attention-based methods (e.g., SnapKV [3], PyramidKV [5]), R-KV estimates token importance using attention weights, leveraging the intuition that tokens receiving higher attention contribute more to decoding and are thus more critical for preserving model performance. Specifically, we compute each key token's attention scores received from the last  $\alpha$  **observation tokens** during decoding. In addition to the standard multi-head attention mainly adopted by the prior works [3], we also propose the importance score estimation using the grouped-query attention. Below, we detail the estimation on top of these two popular attention mechanisms used by current LLMs.

**Multi-Head Attention (MHA).** Given the last  $\alpha$  observation tokens as query  $\mathbf{Q}^h \in \mathbb{R}^{\alpha \times d}$  and n key states  $\mathbf{K}^h \in \mathbb{R}^{n \times d}$  for each attention head h, the attention scores  $\mathbf{A}^h \in \mathbb{R}^{\alpha \times n}$  are computed as:

<span id="page-3-3"></span><span id="page-3-2"></span>
$$\mathbf{A}^{h} = \operatorname{softmax}(\mathbf{Q}^{h} \cdot (\mathbf{K}^{h})^{\top} / \sqrt{d}). \tag{1}$$

**Grouped-Query Attention (GQA).** In GQA, each key/value head h is shared among a group of G distinct query heads indexed by  $g \in [0, G)$ . Correspondingly, we denote the shared key/value states as  $K^h, V^h \in \mathbb{R}^{n \times d}$ , and the G query states as  $Q^{h,0}, \ldots, Q^{h,G-1} \in \mathbb{R}^{\alpha \times d}$  within the head group indexed by h, where n is the number of key/value states, d is the head hidden dimension. The attention score for each of the G query heads within the group is computed as:

$$\mathbf{A}_{\text{group}}^{h,g} = \mathbf{Q}^{h,g} \cdot (\mathbf{K}^h)^{\top} / \sqrt{d} \in \mathbb{R}^{\alpha \times n}, \quad \text{for } g = 0, \dots, G - 1.$$
 (2)

These G individual matrices are then aggregated into a single consolidated matrix  $\boldsymbol{A}_{\text{group}}^h$  for the head group h using a max-pooling operation across the group dimension. The final attention weight  $\boldsymbol{A}^h$  for the head group h is then obtained by renormalizing  $\boldsymbol{A}_{\text{group}}^h$  along the key token dimension.

$$\boldsymbol{A}_{\text{group}}^{h} = \text{maxpool}\left(\left[\boldsymbol{A}_{\text{group}}^{h,0}, \ldots, \boldsymbol{A}_{\text{group}}^{h,G-1}\right]\right) \in \mathbb{R}^{\alpha \times n}, \quad \boldsymbol{A}^{h} = \text{softmax}\left(\boldsymbol{A}_{\text{group}}^{h}\right) \in \mathbb{R}^{\alpha \times n} \quad (3)$$

**Stabilization and Importance Estimation.** We use  $A^h$  hereafter to denote the attention weights calculated using either MHA or GQA. Note that the per-token importance scores derived from  $A^h$  may contain outliers with excessively high values, resulting in unstable estimation of importance scores. To mitigate this influence, we follow the prior work [3] and apply a max-pooling operation to these per-token importance scores over a sliding window of size 2W across recent tokens. Specifically, we denote  $A^h_{j,i}$  as the attention score from the j-th query to the i-th key in  $A^h$ . We obtain the stabilized

attention score  $\tilde{A}^h$  by computing its (i, j) entry, and finally obtain the importance score of retaining the *i*-th token in the KV cache as  $I_i^h$  for each attention head h, as shown below:

<span id="page-4-3"></span>
$$\tilde{A}_{j,i}^{h} = \max\left(A_{j,i-W}^{h}, \dots, A_{j,i}^{h}, \dots, A_{j,i+W-1}^{h}\right), \quad I_{i}^{h} = \frac{1}{\alpha} \sum_{j=0}^{\alpha-1} \tilde{A}_{j,i}^{h} \in \mathbb{R}.$$
 (4)

#### <span id="page-4-0"></span>3.3 Redundancy Estimation via Semantic Similarity

To identify redundant tokens, we measure the semantic similarity between key states using cosine similarity. Tokens with high similarity to others are considered potentially redundant and can be selectively removed to optimize KV cache memory.

Cosine Similarity between Key Tokens: Given the key tokens  $\boldsymbol{K}^h \in \mathbb{R}^{n \times d}$  for a specific head h, We first normalize each key vector  $\boldsymbol{K}_i^h, \forall i \in [0,1)$  into  $\overline{\boldsymbol{K}}_i^h$ , and then compute the cosine similarity matrix  $\boldsymbol{S}^h$  using the normalized key vectors.

$$\overline{\boldsymbol{K}}_{i}^{h} = \frac{\boldsymbol{K}_{i}^{h}}{\|\boldsymbol{K}_{i}^{h}\|_{2} + \epsilon} \in \mathbb{R}^{d}, \quad \boldsymbol{S}^{h} = \overline{\boldsymbol{K}}^{h}(\overline{\boldsymbol{K}}^{h})^{\top} \in \mathbb{R}^{n \times n}, \quad S_{i,i}^{h} \leftarrow 0, \forall i \in [0, n),$$
 (5)

where  $\|\cdot\|_2$  is the L2 norm and  $\epsilon$  is a small constant (e.g.,  $10^{-8}$ ) for numerical stability. To prevent tokens from being marked as redundant with themselves, we zero out the diagonal elements  $S_{i,i}^h$ .

Enforce Retention of Recent Tokens. While redundant, such tokens may still carry meaningful information. Thus, naively removing all redundant tokens can impair model performance. To address this, we retain only the  $\beta$  most recently generated tokens among those exhibiting high similarity, as these later tokens tend to better support the model's decoding than earlier ones. To enforce this, we further zero out the similarity scores in  $S^h$  corresponding to these  $\beta$  most recent similar tokens. Formally, for each token  $i \in [0, n)$ , we identify the set of indices of highly similar tokens:  $\mathcal{I}_i^h = \{j \mid S_{j,i}^h > T, j \in [0, n)\}$ , where T is a fixed hyperparameter for similarity threshold. For this set, we extract the subject  $\mathcal{I}_{i,\beta}^h \subseteq \mathcal{I}_i^h$ , containing up to the  $\beta$  largest indices—i.e., the  $\beta$  most recent similar tokens to token i, or fewer if not enough such tokens exist. We then suppress their influence by zeroing out their similarity scores with token i in  $S^h$ , i.e.,  $S_{j,i}^h \leftarrow 0$ ,  $\forall j \in \mathcal{I}_{i,\beta}^h$ . This modification effectively nullifies the direct similarity links from token i to its  $\beta$  most recent highly similar tokens.

**Redundancy Score Estimation:** Finally, we compute normalized redundancy scores for all key tokens in Eq. (6). First, for each key token  $i \in [0,n)$  in each head h, we compute its average similarity score  $\bar{S}_i^h$ . Intuitively,  $\bar{S}_i^h$  measures how similar token i is, on average, to all other key tokens in the sequence. A high  $\bar{S}_i^h$  indicates that the semantic content of token i is largely shared with other tokens, suggesting potential redundancy. Next, to obtain per-token redundancy scores  $R_i^h$  within a fixed numerical range for each head h, we normalize  $\bar{S}_i^h$  using a softmax operation. The resulting score  $R_i^h$  reflects the redundancy of token i for head h, with higher values indicating greater redundancy.

$$\bar{S}_{i}^{h} = \frac{1}{n} \sum_{i=0}^{n-1} S_{j,i}^{h}, \quad R_{i}^{h} = \left( \text{softmax} \left( [\bar{S}_{0}^{h}, \dots, \bar{S}_{n-1}^{h}] \right) \right)_{i}$$
 (6)

#### <span id="page-4-1"></span>3.4 Joint Selection Strategy for KV Cache Retention

To efficiently manage KV cache storage while retaining essential context, we employ a joint selection strategy that integrates both importance and redundancy scores. Given a predefined token budget  $B_{budget}$  per attention head, our goal is to retain tokens that maximize information diversity while minimizing redundancy. The final selection score  $Z_i^h$  for each token i in head h is computed as:

<span id="page-4-2"></span>
$$Z_i^h = \lambda I_i^h - (1 - \lambda) R_i^h, \tag{7}$$

where the importance score  $I_i^h$  and the redundancy score  $R_i^h$  are computed in Eq. (4) and Eq. (6) respectively. A higher  $I_i^h$  indicates that a token is more important and should ideally be retained, while a higher  $R_i^h$  suggests higher token redundancy. The hyperparameter  $\lambda$  controls the trade-off

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Results of R-KV compared with SnapKV and FullKV on the MATH-500 and AIME24 datasets for R1-Llama-8B (**top**) and R1-Qwen-14B (**bottom**). Results are reported as pass@1 based on 64 generated responses per question.

between prioritizing important tokens and reducing redundant tokens. We discuss the rationale for choosing  $\lambda$  through a sensitivity analysis in §5.1. This strategy ensures that the KV cache prioritizes storing tokens that are both important and semantically diverse, thereby improving memory efficiency without compromising model performance.

