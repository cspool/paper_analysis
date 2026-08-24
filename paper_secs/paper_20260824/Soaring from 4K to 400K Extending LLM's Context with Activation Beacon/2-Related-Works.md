# 2 Related Works

Recently, processing long context has become a fundamental capability of modern LLMs [\[33;](#page-13-0) [18;](#page-10-0) [39;](#page-14-1) [16\]](#page-10-2). The recipe of context window extension is roughly the same: modifying the rotary position embedding [\[35\]](#page-13-3) by extrapolation and interpolation [\[9;](#page-10-3) [1;](#page-9-1) [34;](#page-13-4) [17\]](#page-10-4), and leveraging long-dependency data in both the pre-training and post-training stage. Despite the impressive progress in effectiveness, LLMs face significant challenges in efficiency. There is significant computational cost due to the quadratic complexity of transformer, and huge memory cost because LLMs need to hold the KV activations of the entire sequence on GPU for faster decoding. Multiple threads of research endeavour to reduce these costs, which are discussed as follows.

Sparse Attention. Conventional sparse attention methods require re-training a model from scratch using the designated sparse patterns [\[40;](#page-14-3) [5\]](#page-10-5). However, extensive recent studies have identified that the attention pattern of LLMs are naturally sparse despite they are densely trained [\[26;](#page-12-4) [38;](#page-14-4) [22;](#page-12-5) [43\]](#page-14-5).

<span id="page-2-0"></span>> **[图片提取文字 (无描述)]:**
> there be there light light and was ; **=**; 0  $\langle b \rangle_1^1$  $(b)_{1}^{2}$ light  $\langle b \rangle_2^1$  $\langle b \rangle_1^1$ (b)2 there light there be Let and was Chunk 1 Chunk 2
![](_page_2_Figure_0.jpeg)

Figure 1: Overview of Activation Beacon. The context is partitioned into chunks. Each chunk is further split into fine-grained units and interleaved with beacon tokens according to a compression ratio (2 in the figure). The LLM encodes one chunk at a time, compressing the context into beacon tokens' activations, which are *accumulated* and *reused* for encoding following chunks.

They also propose to dynamically set appropriate sparse patterns for each head so that the attention mass can be largely preserved, leading to competitive performance against the full-attention method with reduced computation. However, these methods require holding all KV activations on chip to dynamically determine the optimal sparse patterns, making them unsuitable for KV cache reduction.

**KV** Compression. This line of research focuses on compressing the KV activations to reduce the attention computation as well as the cache size. Since the KV activations are per-layer, per-head, per-token, and per-channel float numbers, they can be reduced from all the five dimensions (including the numerical dimension). For example, CLA [7] shares the KV cache across multiple layers; GQA [2] compresses multiple key/value heads into a single one; MLA [16] compresses the channels into fewer and more compact ones; and KIVI [44] quantizes the numerical value in the activations. The token-wise compression, as introduced in the following, is also known as context compression, which is orthogonal to the compression along other dimensions and hence can be jointly used.

Context Compression. This type of methods aim to compress the raw context into shorter yet more compact representations. Existing studies are usually tailored for compressing short context (less than 1K), which tend to be sub-optimal for long-context compression. Specifically, Gisting [32] compresses the user instruction into gist activations all at once. As a result, it cannot process context longer than the backbone LLM's window. ICAE [20] and AutoCompressor [11] alleviate this problem by segmenting the long context into chunks and compressing each chunk. However, both of them compress the context into soft tokens, which are the major bottleneck to encapsulate the complex information in long contexts. Their compression workflow also lacks fine-grained handling of the chunked inputs, resulting in inferior compression quality. Moreover, these soft tokens require re-encoding before generation, which introduces extra overhead. Lastly, since the number of soft tokens are pre-defined, it is hard to flexibly assign the compression ratio for downstream tasks. CCM [28] is specifically designed for compressing conversations in online chatting, which cannot be used in general long context tasks such as long document understanding. Another branch of methods [25, 31] propose to delete unimportant tokens to realize compression. However, they depend on the input question to accurately estimate the token importance, leading to low efficiency in real-world multi-turn scenarios. Compared with existing approaches, Activation Beacon is able to achieve more effective, efficient, and flexible compression.

### 3 Methodology

LLMs accomplish arbitrary tasks in the form of next-token prediction. Formally, given the context  $X = [x_1, \ldots, x_n]$ , the LLM generates the next token based on all preceding tokens and its well-trained parameters:  $\Pr(x_{n+1} \mid x_1, \ldots, x_n; \Theta)$ . Transformer-based LLMs incur heavy computation cost due to the quadratic complexity of self attention; besides, they require tremendous GPU memory to store the KV cache of  $x_{\leq n+1}$  for faster decoding [42]. Both the costs in computation and memory significantly expand when the context length increases.

<span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> Forward 1  $x_1^1$  $x_2^1$  $\langle b \rangle_1^1$ **FFN**  $x_3^1$  $\chi_4^1$ Layer Norm  $\langle b \rangle_2^1$  $x_1^2$ Forward 2  $x_{2}^{2}$ Self-Attn  $\langle b \rangle_1^2$ Self-Attnb  $x_3^2$  $x_4^2$ Layer Norm  $\langle b \rangle_2^2$  $(x_1^1 \quad x_2^1 \quad \langle b \rangle_1^1 \quad x_3^1 \quad x_4^1 \quad \langle b \rangle_2^1 \quad x_1^2 \quad x_2^2 \quad \langle b \rangle_1^2 \quad x_3^2 \quad x_4^2 \quad \langle b \rangle_2^2$
![](_page_3_Figure_0.jpeg)

Figure 2: Activation Beacon performs compression during self attention while reusing all other modules of the LLM. *Forward* ①: encode and compress the first chunk. *Forward* ②: encode and compress the second chunk conditioned on activations of preceding beacon tokens.

Activation Beacon employs a new special token, namely beacon token  $\langle b \rangle$ , and condenses the raw context X into beacon tokens' activations  $\Psi$  (i.e. their keys and values at every layer). The next-token prediction is converted to condition on the compressed context instead of the plain one. Given  $|\Psi| < |X|$ , both the computation cost and the KV cache size are reduced. Additionally, the LLM is enabled to handle context longer than its window size based on the compressed representations. We tailor the compression mechanism and the learning method of Activation Beacon towards achieving effective, efficient, and flexible compression, which will be elaborated in the following.

#### 3.1 Compression Mechanism

**Overview.** We propose to progressively compress each fine-grained units of long contexs. Specifically, given the input context X whose length may exceed the LLM's context window N, it is first partitioned into chunks of the same size w (e.g., 1024):

$$[x_1, \dots, x_n] \xrightarrow{\text{Partition}} [X_1, \dots X_{\lceil n/w \rceil}], \ X_i = [x_{(i-1)w+1}, \dots, x_{iw}]^3 = [x_1^i, \dots, x_w^i].$$
 (1)

Next, for each chunk  $X_i$ , we determine a compression ratio  $\alpha_i$  (w is evenly divisible by  $\alpha_i$ ). The chunk is further split into fine-grained units of size  $\alpha$ . Then a group of  $k_i = w/\alpha_i$  beacon tokens,  $B_i = [\langle b \rangle_{1}^{i}, \ldots, \langle b \rangle_{k_i}^{i}]$ , are *interleaved* with these units. In other words, one beacon token is dispatched to the end of every unit:

$$X_i \xrightarrow{\text{Interleave } B_i} X_i' = [x_1^i, \dots, x_{\alpha_i}^i, \langle \mathbf{b} \rangle_1^i, \dots, x_{w-\alpha_i+1}^i, \dots, x_w^i, \langle \mathbf{b} \rangle_{k_i}^i]. \tag{2}$$

The LLM encodes these chunks *one by one*, compressing the contextual information of each chunk into the corresponding beacon tokens' activations during self attention. After encoding  $X'_i$ , we *discard* activations of all the raw tokens  $X_i$ , while we *accumulate* the activations of the beacon tokens  $B_i$ . When encoding the next chunk  $X'_{i+1}$ , the LLM directly conditions on the accumulated beacon activations as a proxy to the raw context  $X_{\leq i}$ .

This progressive workflow benefits both compression quality and running efficiency. On one hand, it enables thorough distillation of complex information within long contexts and allows for the compression of inputs that exceed the LLM's context window. On the other hand, by caching and reusing beacon tokens' activations, it avoids redudant computation and allows for incrementally update of the compression results in multi-turn interactions.

**Encoding and Compression.** As shown in Figure 2, Activation Beacon reuses all modules of the LLM except imposing a slight modification on self attention. Without loss of generality, for the *i*-th

<span id="page-3-0"></span><sup>&</sup>lt;sup>3</sup>The last chunk  $X_{\lceil t/w \rceil}$  may be shorter than w, which is omitted for simplicity.

<span id="page-4-0"></span>> **[图片提取文字 (无描述)]:**
> (A) Llama-2-7B (B) Qwen-2-7B (C) Qwen-2-72B 1e4 1e4 1e5 4.0 Method Method Method 3.0 2.0 3.5 Full Attention Full Attention Full Attention 3.0 Activation Beacon (x2) Activation Beacon (x2) Activation Beacon (x2) Activation Beacon (x4) Activation Beacon (x4) Activation Beacon (x4) 2.5 2.0 2.0 Activation Beacon (x8) Activation Beacon (x8) Activation Beacon (x8) TFLOPs 12 TFLOPS 1.0 1.0 1.0 0.5 0.5 0.5 0.0 0.0 0.0 256K 128K 128K 32K 64K 128K 64K 256K 32K 64K 256K Context Length Context Length Context Length
![](_page_4_Figure_0.jpeg)

Figure 3: Comparison of the forward FLOPs of different models using full attention and Activation Beacon (the compression ratio is annotated in the brackets).

chunk  $X'_i$ , the encoding process can be written as:

LLM( 
$$(b)_{1}^{i}, \dots, (b)_{k_{i-1}}^{i-1}, \dots, (b)_{k_{i-1}}^{i-1}, \dots, x_{\alpha_{i}}^{i}, (b)_{1}^{i}, \dots, x_{w-\alpha_{i}+1}^{i}, \dots, x_{w}^{i}, (b)_{k_{i}}^{i}),$$
 (3)

beacon activations accumulated from  $X'_{< i}$  the current chunk  $X'_{i}$ 

where the input to the LLM is a mix of the activations accumulated from previous chunks and the tokens to be encoded within the current chunk. Let D denote the LLM's hidden size,  $\mathbf{H} \in \mathbb{R}^{(w+k_i)\times D}$  denote input hidden states to self attention in an arbitrary layer of the LLM. We first slice out the hidden states of raw tokens and beacon tokens:

$$\mathbb{I}^r = \{ j \mid x_j^i \neq \langle \mathbf{b} \rangle \}, \quad \mathbb{I}^b = \{ j \mid x_j^i = \langle \mathbf{b} \rangle \}; \quad \boldsymbol{H}^r = \boldsymbol{H}[\mathbb{I}^r], \quad \boldsymbol{H}^b = \boldsymbol{H}[\mathbb{I}^b]. \tag{4}$$

Then the hidden states are projected into queries, keys, and values:

$$Q^r = W_Q^r H^r,$$
  $K^r = W_K^r H^r,$   $V^r = W_V^r H^r,$   $Q^b = W_Q^b H^b,$   $K^b = W_K^b H^b,$   $V^b = W_V^b H^b,$  (5)

where  $\boldsymbol{W}_*^r$  are the LLM's original projection matrices and  $\boldsymbol{W}_*^b$  are the newly introduced matrices to handle beacon tokens only. Afterwards, the query/key/value states of raw tokens and beacon tokens are scattered back to acquire  $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V} \in \mathbb{R}^{(w+k_i) \times D}$ :

$$Q[\mathbb{I}^r] = Q^r, \ Q[\mathbb{I}^b] = Q^b; \quad K[\mathbb{I}^r] = K^r, \ K[\mathbb{I}^b] = K^b; \quad V[\mathbb{I}^r] = V^r, \ V[\mathbb{I}^b] = V^b.$$
 (6)

Finally, the standard self-attention is computed over the entire input:

$$A = \operatorname{softmax} \left( \operatorname{mask} \left( \frac{Q \{K^{ac}; K\}^T}{\sqrt{D}} \right) \right), \quad V = A \{V^{ac}; V\}.$$
 (7)

In the above equations,  $\{\cdot\,;\,\cdot\}$  denotes matrix concatenation.  $\boldsymbol{K}^{ac}, \boldsymbol{V}^{ac} \in \mathbb{R}^{m_{i-1} \times D}$  are the beacon tokens' activations accumulated from previous chunks where  $m_{i-1} = \sum_{j=1}^{i-1} k_j$ , and mask denotes the causal attention mask. During self attention, all tokens are encoded by their relative positions  $([m_{i-1},\ldots,m_i+w-1]$  for queries and  $[0,\ldots,m_i+w-1]$  for keys). The value states  $\boldsymbol{V}$ , are further processed by other modules (e.g., output projection, MLP, and LayerNorm) before passing to the next layer. After self attention, the keys and values of beacon tokens, i.e.  $\boldsymbol{K}^b$  and  $\boldsymbol{V}^b$ , have distilled the contextual information of  $X_i$ . They are incrementally accumulated:

$$\boldsymbol{K}^{ac} = \{\boldsymbol{K}^{ac}; \boldsymbol{K}^{b}\}, \quad \boldsymbol{V}^{ac} = \{\boldsymbol{V}^{ac}; \boldsymbol{V}^{b}\}. \tag{8}$$

In our default setting, the beacon tokens are interleaved with raw tokens. This leads to a differentiated attention scope for each beacon token  $(\langle b \rangle_j^i)$  attends to one more interval than  $\langle b \rangle_{j-1}^i$ , contributing to the *fine-grained* compression of the context. We also explore the setting to dispatch all beacon tokens at the end of the chunk, which results in inferior compression quality (§4.6).

Note that unlike ICAE [20] and LLMLingua [25], Activation Beacon unifies generation and compression operations within a single forward pass of the LLM. That is to say, the hidden states of the last input token  $\boldsymbol{H}[\mathbb{R}^r[-1]]$  is directly used to decode the next token without resorting to another decoder model.

Efficiency Analysis. Activation Beacon reduces the KV cache by  $\alpha$  times where  $\alpha$  is the *average compression ratio* and hence the memory cost. This is because it only needs to store the compressed activations of the preceding chunks instead of the raw activations. In terms of computation, the situation is a bit more complex. Specifically, Activation Beacon significantly reduces the computation in self attention, because each token only needs to interact with local tokens within the chunk and preceding beacon tokens, which are approximately  $\alpha$  times shorter than the raw context. However, it also triggers more computation to encode the inserted beacon tokens in other modules (e.g., MLP). Formally, given an LLM with a fixed number of layers, attention heads, and hidden size, let s denote the input context length, s denote the cached context length, the forward FLOPs is:

$$FLOPs = F^{Att}(s, s^{pst}) + F^{Oth}(s),$$
(9)

where  $F^{Att}$  is the computation during self attention, and  $F^{Oth}$  is the computation of other modules. For full-attention models,  $s=n, s^{pst}=0$ . For "beaconed" models, the FLOPs is:

$$FLOPs^{bcn} = \sum_{i=1}^{\lceil \frac{n}{w} \rceil} F^{Att} \left( \frac{(\alpha+1)w}{\alpha}, \frac{(i-1)w}{\alpha} \right) + F^{Oth} (n + \lceil \frac{n}{\alpha} \rceil).$$
 (10)

Since the implementation of  $F^{Att}$  and  $F^{Oth}$  depends on the actual setting of the LLM (see Appendix B), we visualize the FLOPs curve of three different LLMs in Figure 3. It can be observed that Activation Beacon consistently saves computational costs across different model settings and scales. The extent of saving amplifies as the context length grows, finally achieving more than x4 reduction at 256K context. The specific implication on latency is studied in  $\S4.3$ .

#### 3.2 Learning Method

**Compression-Based Auto-Regression.** Activation Beacon is learned to optimize the generation quality conditioned on the mixture of the compressed context and the local context. Formally, the compression-based next-token prediction loss is minimized:

$$\min_{\boldsymbol{\Theta}^b} \cdot \sum_{i=2}^{\lceil N/w \rceil} \sum_{j=1}^w \Pr(x_j^i \mid \langle \mathbf{b} \rangle_1^1, \dots, \langle \mathbf{b} \rangle_{k_{i-1}}^{i-1}, x_1^i, \dots x_{j-1}^i; \boldsymbol{\Theta}, \boldsymbol{\Theta}^b).$$
(11)

 $\Theta$  denotes the parameters of the LLM itself, which are *fixed* throughout the training process.  $\Theta^b$  includes the projection matrices for beacon tokens at each layer  $W_Q^b$ ,  $W_K^b$ ,  $W_V^b$ , and the token embedding of beacon token  $e_{\langle b \rangle}$  (we use one *shared* embedding for all beacon tokens). The training loss can be obtained from all tokens except the ones in the first chunk. Such a property leads to high sample efficiency that maximizes the use of training data. Note that we exclude the beacon tokens from the above loss (setting their labels to -100) because they are solely intended for compression.

No Stop Gradients. Recurrent memory methods [11; 8] stop the gradients back-propagation at a given chunk number to improve the training efficiency. This is because these methods depend on the *final-layer* outputs of preceding chunks to encode the current chunk, which results in deepened computation graph as more chunks are involved. In contrast, Activation Beacon only depends on the *previous-layer* outputs of preceding chunks (the encoding of  $X_i'$  at layer l only conditions on the results of  $X_{l-1}'$  at layer l-1), which is the same as any auto-regressive LLMs. Thus, the gradients can naturally flow through all chunks to optimize the compression effect over long contexts.

**Chunk-Wise Random Compression Ratio.** To teach the model to flexibly support diverse compression granularities, the compression ratio  $\alpha_i$  for the *i*-th chunk is *randomly sampled* from  $\{2,4,8,16,32\}$  during training. At inference, one can choose one compression ratio according to the specific efficiency requirement in downstream tasks and stick to it for all chunks.

### 4 Experiments

Our experiment mainly study Activation Beacon's effectiveness ( $\S4.2$ ), efficiency ( $\S4.3$ ), and flexibility ( $\S4.4$ ) in long context compression. Besides, we explore Activation Beacon's impact on short-context capabilities of the backbone LLM ( $\S4.5$ ) and the effect of each technical design ( $\S4.6$ ).

<span id="page-6-2"></span>

| Table 1: Evaluation on LongBench [3]. Activation Beacon maintains comparable performance to the |
|-------------------------------------------------------------------------------------------------|
| uncompressed baseline (Full-FT), outperforming other compression methods.                       |

| Model      | Method                                | Length                         | Single-Doc                                  | <b>Multi-Doc</b>                            | Summ.                                       | Few-Shot                             | Code                                        |
|------------|---------------------------------------|--------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|--------------------------------------|---------------------------------------------|
| Llama-2-7B | Full<br>Full-FT                       | 4K<br>32K                      | 24.7<br>34.8                                | 22.4<br><b>27.5</b>                         | 24.6<br>23.2                                | <b>63.2</b> 61.8                     | 57.7<br><b>57.8</b>                         |
|            | AutoCompr. ICAE LongLLML. SnapKV Ours | 32K<br>32K<br>32K<br>4K<br>32K | 12.9<br>19.5<br>21.5<br>24.2<br><b>34.9</b> | 16.4<br>19.2<br>18.8<br>22.6<br><b>27.5</b> | 16.3<br>19.5<br>21.7<br>16.3<br><b>25.0</b> | 23.8<br>24.8<br>49.5<br>60.1<br>61.4 | 39.4<br>27.8<br>53.2<br>57.7<br><b>57.8</b> |
| Qwen-2-7B  | Full<br>Full-FT                       | 32K<br>32K                     | 38.8<br><b>41.0</b>                         | 37.5<br><b>40.6</b>                         | 26.7<br><b>26.8</b>                         | <b>70.1</b> 68.5                     | 60.3<br>66.1                                |
|            | LongLLML.<br>SnapKV<br>Ours           | 32K<br>32K<br>32K              | 24.7<br>38.7<br>40.5                        | 20.3<br>37.6<br>40.3                        | 26.3<br>26.2<br><b>26.8</b>                 | 55.9<br>67.1<br>68.4                 | 50.1<br>60.3<br><b>66.4</b>                 |

<span id="page-6-3"></span>> **[图片提取文字 (无描述)]:**
> (A) Llama-2-7B with Activation Beacon (B) Qwen-2-7B with Activation Beacon 11 11 22 5.0 22 Depth Percent Percent 33 33 5.0 Depth 55 66 9.0 7.0 1.0 77 88 88 100 100 Context Length Context Length
![](_page_6_Figure_2.jpeg)

Figure 4: Evaluation on Needle-in-a-Haystack. Activation Beacon can accurately retrieves the needle most of the time, despite the context is far longer than its training data.

### 4.1 Settings

**Implementation.** Activation Beacon is applied to Llama-2-7B (chat)<sup>4</sup> and Qwen-2-7B (instruct). The chunk size w is 1024 for Llama-2 and 2048 for Qwen-2. FlashAttention-2 [15] is used to speed up attention computation. For all our experiments, we use Huggingface framework [37] and one 8xA800 (80G) machine.

**Training.** The training consists of two phases. In pre-training, we use 1B tokens sampled from RedPajama [14]. The eos token is appended to the end of every document. In fine-tuning, we leverage LongAlpaca [10], BookSum [29], and synthetic data from GPT-3.5 (details in Appendix A). All the training samples are shorter than 20K. The batch size is 8. The learning rate is 5e-5 for pre-training and 1e-5 for fine-tuning, with linear decay and no warmup. As introduced, the LLM's original parameters are frozen throughout the training process.

**Baselines.** We compare Activation Beacon with the uncompressed baseline (denoted as Full) and the uncompressed baseline fine-tuned with the same training data (denoted as Full-FT). Besides, we include the following context compression methods that can tackle long context for comparison, including AutoCompressors [11], ICAE [20], LongLLMLingua [25], and SnapKV [31]. The first two methods only support Llama-2. To guarantee fair comparison, we fine-tune their official checkpoints using the same training data.

#### <span id="page-6-0"></span>4.2 Compression Effectiveness

To verify the compression effectiveness of Activation Beacon, we evaluate it on LongBench [3], which consists of a variety of long-context tasks with 32K maximum length, including question

<span id="page-6-1"></span><sup>&</sup>lt;sup>4</sup>We use Llama-2 because AutoCompressor and ICAE are based on it, both of which are important baselines.

<span id="page-7-0"></span>Table 2: Evaluation on Multi-Needle-in-a-Haystack where the questions are issued one-by-one in a multi-turn conversation setting. All compression methods use a x8 compression ratio. Activation Beacon consistently outperforms other compression baselines while enjoying lower latency, especially when the context lengthens and the turn number increases.

| Model      | Length | Method                                            | 1-Turn                               |                                           | 2-Turn                               |                                           | 3-Turn                               |                                           |
|------------|--------|---------------------------------------------------|--------------------------------------|-------------------------------------------|--------------------------------------|-------------------------------------------|--------------------------------------|-------------------------------------------|
|            |        |                                                   | Acc                                  | Latency                                   | Acc                                  | Latency                                   | Acc                                  | Latency                                   |
| Llama-2-7B |        | Full-FT                                           | 9.75                                 | 1.336                                     | 9.45                                 | 1.532                                     | 9.10                                 | 1.726                                     |
|            | 32K    | AutoCompr.<br>ICAE<br>LongLLML.<br>SnapKV<br>Ours | 1.60<br>2.15<br>2.05<br>1.00<br>9.75 | 2.135<br>1.182<br>2.813<br>0.859<br>1.153 | 1.50<br>2.15<br>2.00<br>1.00<br>9.40 | 2.561<br>1.476<br>5.062<br>1.656<br>1.356 | 1.50<br>2.00<br>2.00<br>1.00<br>9.05 | 2.994<br>1.805<br>7.034<br>2.199<br>1.638 |
|            |        | Full-FT                                           | 9.75                                 | 4.399                                     | 9.50                                 | 5.254                                     | 9.20                                 | 6.153                                     |
| Qwen-2-7B  | 128K   | LongLLML.<br>SnapKV<br>Ours                       | 2.00<br>9.45<br>9.70                 | 10.455<br>3.955<br>2.445                  | 1.55<br>8.95<br>9.35                 | 19.768<br>7.803<br>2.773                  | 1.50<br>8.85<br>9.10                 | 27.751<br>10.659<br>2.981                 |

answering, summarization, few-shot learning, and code completion. Since Llama-2 has a context window of 4K, we truncate the context longer than 4K from middle before inputting to it. For compression methods implemented on Llama-2, we set *adaptive* compression ratio, translating to x2 compression for 4K-8K contexts, x4 compression for 8K-16K contexts, and x8 compression for 16K-32K contexts. For methods implemented on Qwen-2, we apply a uniform compression ratio of x4. The results are reported in Table [1.](#page-6-2) We highligh two observations in the following.

Firstly, Activation Beacon achieves superior compression quality over other compression baselines across all tasks. Concretely, it siginificantly outperforms ICAE and AutoCompressor, which verifies that several soft tokens are not enough to encapsulate the rich information within long contexts. LongLLMLingua also lags far behind Activation Beacon because it need to delete too many tokens given a high compression ratio (e.g., x4, x8), which may destroy the coherence of the context and lose important information. Despite SnapKV's top performance among baselines, it cannot compress context longer than the backbone LLM's window. This is because it estimates the token importance based on self attention, which becomes inaccurate once the context exceeds the window size, limiting its practical usage when compressing long contexts.

Secondly, Activation Beacon achieves comparable performance to the fine-tuned uncompressed baseline (Full-FT) even though Full-FT takes in the entire context without compression. This indicates that Activation Beacon is able to compress long contexts without evident information loss, which validates its high compression quality yielded from the progressive compression workflow. Furthermore, Activation Beacon improves upon Llama-2 by a large margin despite their context window is the same, i.e. 4K. The gain is because Llama-2 (Full) directly uses the truncated 4K context, while Activation Beacon compresses the 32K context into 4K compact activations. This implies that Activation Beacon can effectively introduce useful information from Llama-2's unseen context. Therefore, it can be viewed as an efficient approach for context extension.

We further evaluate Activation Beacon on Needle-in-a-Haystack (NIAH) following the official settings [\[21\]](#page-12-8) to investigate whether it will lose fine-grained information. The accuracy is estimated by ChatGPT (ranges from 1 to 10). For both Llama-2 and Qwen-2, we set adaptive compression ratio as introduced above. The results are shown in Figure [4.](#page-6-3) It can be observed that Activation Beacon precisely retrieves the needle most of the time. Note that Activation Beacon conducts *query-independent* compression, which means it has no prior knowledge of what to compress and what not. Hence, this remarkable performance again validates our tailored compression mechanism and learning method can preserve the fine-grained contextual information. Moreover, Activation Beacon is only trained on context shorter than 20K, while its compression capability can generalize to far longer contexts (e.g., 128K).

<span id="page-8-3"></span>> **[图片提取文字 (无描述)]:**
> -- AutoCompressor -x-- ICAE ---- LongLLMLingua SnapKV Activation Beacon (A) Context Length=1K (B) Context Length=4K (C) Context Length=32K --------------------------------------Accuracy · · · · · · · · · · · · · · · · · · · x2 x16 Compression Ratio Compression Ratio Compression Ratio
![](_page_8_Figure_0.jpeg)

Figure 5: Evaluation on Needle-in-a-Haystack with various compression ratios based on Llama-2. Activation Beacon achieves top compression quality across all compression configurations.

#### <span id="page-8-0"></span>4.3 Compression Efficiency

We evaluate the efficiency of Activation Beacon based on the Multi-Needle-in-a-Haystack task following NeedleBench [30]. Specifically, we fix the context length to 32K for Llama-2 and 128K for Qwen-2, and insert 3 different needles at different positions. The task is organized in a multi-turn conversation setting, where the model is asked to retrieve one specific needle in each turn. The experiment is repeated 20 times for each model with distinct needle positions. In Table 2, we report the accuracy and the end-to-end latency of compression & generation (measured in seconds).

It can be observed that **Activation Beacon enjoys lower latency than other compression baselines**. Notably, it is 1.8x faster than AutoCompressor because it does not have to re-encode the soft tokens from previous chunks. It also leads to 9.3x and 3.6x acceleration upon LongLLMLingua and SnapKV given three turns, respectively. This is because both baselines are query-dependent while Activation Beacon is not, which eliminates the need to re-compute the compression results for different input questions. Moreover, Activation Beacon demonstrates consistent speed-up over the Full-FT baseline, achieving **2x acceleration at 128K context length**. This matches our estimation in Figure 3(b) as Activation Beacon (x8) saves half of the computation. In the meanwhile, since the compression ratio is x8, it leads to **8x reduction of the KV cache**. Lastly, Activation Beacon always attains nearly-lossless generation quality against the uncompressed baseline, which is in line with previous observations.

### <span id="page-8-1"></span>4.4 Compression Flexibility

Activation Beacon is learned to support various compression ratios during training. In Figure 5, we evaluate its compression quality under different compression ratios and context lengths. According to the figure, Activation Beacon maintains top accuracy across all compression ratios, outperforming most compression baselines by a large margin. Though SnapKV performs on par with our method at 1K and 4K context length, it fails to compress inputs longer than the LLM's window size, which may limit its practical usage. To summarize, Activation Beacon is a flexible solution to long context compression with the support of diverse compression ratios and various context lengths. Generally, we recommend to use x8 compression ratio as it preserves most information with high efficiency.

#### <span id="page-8-2"></span>4.5 Short-Context Capabilities

Since Activation Beacon interleaves beacon tokens with raw tokens and is primarily trained with long-context tasks, it is intriguing to examine whether the current recipe will impair the short-context capabilities of the backbone LLM. In Table 3, we compare Activation Beacon with the original LLM (Full) on popular benchmarks, including MMLU [23], ARC-Challenge [6],

<span id="page-8-4"></span>Table 3: Activation Beacon preserves the short-context capabilities of the backbone LLM.

| Model      | Method | MMLU | ARC-C | BoolQ | GSM8K |
|------------|--------|------|-------|-------|-------|
| Llama-2-7B | Full   | 47.5 | 48.5  | 86.2  | 9.2   |
|            | Ours   | 46.6 | 48.4  | 86.5  | 9.3   |
| Qwen-2-7B  | Full   | 70.1 | 62.7  | 87.1  | 76.0  |
|            | Ours   | 69.1 | 62.7  | 87.2  | 76.2  |

#### BoolQ [\[12\]](#page-10-12), and GSM8K [\[13\]](#page-10-13). We

can observe that Activation Beacon leads to very little performance degradation on short-context tasks. In other words, the short-context capabilities are well preserved. We conjecture that the primary reason is the LLM's original parameters are frozen throughout the training process.

#### <span id="page-9-3"></span>4.6 Ablation Studies

We study the impact of each technical factor, including the compression of fine-grained context units, the sampling strategy of compression ratio, and training stages. The experiments are based on Qwen-2-7B and Single-Doc QA task from LongBench (32K context with x4 compression ratio). The results are shown in Table [4.](#page-9-5) Firstly, instead of splitting the chunk into fine-grained units and interleaving beacon tokens, we append all beacon tokens at the end of the chunk so that their attention scopes are the same. It can be

<span id="page-9-5"></span>Table 4: The impact of different technical factors.

| Method                                                                                             | Single-Doc                   |
|----------------------------------------------------------------------------------------------------|------------------------------|
| Default                                                                                            | 40.5                         |
| w/o Fine-Grained Compression<br>w/o Chunk-Wise Random Ratio<br>w/o Pre-training<br>w/o Fine-tuning | 35.2<br>37.7<br>34.9<br>35.5 |

observed that such operation results in significant information loss after compression, which justifies the effectiveness of our fine-grained compression mechanism. Secondly, we replace the chunk-wise random compression ratio with the instance-wise one, which randomly selects one compression ratio for each training instance rather than each chunk. We can observe that the chunk-wise setting facilitates better learning of the compression functionality. Lastly, we remove either pre-training or fine-tuning. It can be observe that both stages are useful, and the combination of both leads to the optimal performance. This also implies that the compression quality of Activation Beacon can be further enhanced given more abundant and targeted training.

