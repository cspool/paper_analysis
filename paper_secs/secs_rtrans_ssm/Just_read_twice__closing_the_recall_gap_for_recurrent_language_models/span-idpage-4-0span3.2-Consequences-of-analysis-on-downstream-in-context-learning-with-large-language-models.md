# <span id="page-4-0"></span>3.2 Consequences of analysis on downstream in-context learning with large language models

We next show that our analysis holds consequences for in-context learning on real-world tasks.

**JRT-PROMPT approach.** In-context learning tasks take as input  $(\mathcal{C}, \mathcal{Q}, \mathcal{Y})$  where  $\mathcal{C}$  is some context (e.g., document or code repository),  $\mathcal{Q}$  is some question or request to the model given the context, and  $\mathcal{Y}$  is the answer. For standard in-context learning with autoregressive LM  $\mathcal{A}$ , we input  $\mathcal{C}$  and  $\mathcal{Q}$  and evaluate the generated output  $\hat{\mathcal{Y}} = \mathcal{A}(\mathcal{C}, \mathcal{Q})$  against the true completion  $\mathcal{Y}$ .

We propose JRT-PROMPT, an exceedingly simple method in which information from the prompt (e.g. questions and documents) is *repeated* in-context before the model is prompted to output the answer, *e.g.*,  $\hat{\mathcal{Y}} = \mathcal{A}(\mathcal{C}, \mathcal{Q}, \mathcal{C}, \mathcal{Q})$ , as depicted in Figure 1 (Right). As a result, during the second occurrence of the context, the model can condition on a full view of the context when deciding what to store. We provide the prompts that we use in Appendix E, and release our code to reproduce the table.

**Evaluation.** JRT-PROMPT can be used with off-the-shelf LLMs. We evaluate the following LMs on a suite of recall-intensive in-context learning tasks, with zero-shot prompting:

- Based [7] pretrained LMs at the 1.3B parameter scale trained on 10 50B tokens of the Pile [41]. Transformer++ and Mamba models trained on the exact same tokens and data order are provided for quality references: https://huggingface.co/collections/hazyresearch/
- Mamba [1] pretrained LMs at the 130M, 370M, 1.4B, 2.8B parameter scales, trained on 300B tokens of the Pile [41]: https://huggingface.co/state-spaces
- Gated Linear Attention [9] pretrained LMs at the 1.3B and 2.7B parameter scales, trained on 100B tokens of SlimPajama data [42]: https://huggingface.co/fla-hub

<span id="page-4-1"></span><sup>&</sup>lt;sup>2</sup>A JR-p prompt is simply repeating the input p times (see Definition G.28).

<span id="page-4-3"></span><span id="page-4-2"></span><sup>&</sup>lt;sup>3</sup>This matches the architecture in our experiments.

<sup>&</sup>lt;sup>4</sup>This bound is for the case where the IP kernel is dependent on A and B; if we use an *input-independent* IP kernel, then we get an upper bound of  $O\left((\min\{|A|,|B|\})^2 \cdot n\right)$  (see Remark G.23). Further, this result needs one layer of BaseConv where the convolution kernel is input dependent as well.

<span id="page-5-0"></span>

| Architecture  | Params | Tokens | FDA               | SWDE              | NQ                | SQUAD             | TriviaQA          | Drop              | Average            |
|---------------|--------|--------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|
| Transformer++ | 1.3B   | 10B    | 74.4/86.1         | 41.4/ <b>52.5</b> | 28.2/ <b>31.9</b> | 39.0/ <b>53.1</b> | <b>49.5</b> /49.3 | 22.3/ <b>33.6</b> | 42.5 / <b>51.1</b> |
| Mamba         | 1.3B   | 10B    | 23.3/40.3         | 15.5/31.8         | 19.4/ <b>25.8</b> | 26.6/48.5         | 46.4/ <b>51.1</b> | 21.3/32.1         | 25.1 / <b>38.2</b> |
| Based         | 1.3B   | 10B    | 48.6/ <b>58.9</b> | 27.6/44.7         | 19.7/ <b>28.4</b> | 31.0/46.7         | 44.1/ <b>51.9</b> | 19.5/ <b>34.6</b> | 31.8 / <b>44.2</b> |
| Transformer++ | 1.3B   | 50B    | 83.7/ <b>89.2</b> | 50.8/ <b>65.0</b> | 32.8/ <b>37.5</b> | 41.1/58.1         | 56.6/ <b>58.8</b> | 21.5/ <b>37.9</b> | 47.8 / <b>57.8</b> |
| Mamba         | 1.3B   | 50B    | 41.9/55.7         | 32.6/45.4         | 26.9/ <b>33.9</b> | 31.5/ <b>53.5</b> | 54.9/ <b>56.7</b> | 20.4/ <b>33.8</b> | 34.7 / <b>46.5</b> |
| Based         | 1.3B   | 50B    | 60.2/ <b>68.3</b> | 37.1/ <b>54.0</b> | 29.4/35.2         | 38.9/ <b>56.3</b> | 54.5/ <b>57.6</b> | 21.7/39.1         | 40.3 / <b>51.8</b> |
| GLA           | 1.3B   | 100B   | 48.3/ <b>68.6</b> | 37.7/ <b>53.6</b> | 26.6/31.3         | 34.7/54.8         | <b>55.5</b> /54.6 | 19.6/ <b>33.3</b> | 36.7 / <b>48.9</b> |
| GLA           | 2.7B   | 100B   | 47.1/ <b>65.8</b> | 43.6/ <b>54.5</b> | 27.1/32.9         | 37.2/55.7         | 57.9/57.0         | 22.2/34.0         | 39.2/ <b>50.0</b>  |
| Mamba         | 130M   | 300B   | 25.7/ <b>32.8</b> | 17.5/31.5         | 16.8/ <b>21.7</b> | 27.1/ <b>51.9</b> | 43.5/ <b>50.1</b> | 17.4/30.7         | 24.7 / <b>36.5</b> |
| Mamba         | 370M   | 300B   | 41.9/ <b>58.3</b> | 27.6/42.2         | 23.8/ <b>31.1</b> | 34.9/ <b>51.0</b> | 53.6/51.7         | 19.3/ <b>33.2</b> | 33.5 / <b>44.6</b> |
| Mamba         | 1.4B   | 300B   | 45.8/ <b>60.9</b> | 37.6/46.0         | 31.0/ <b>36.6</b> | 39.9/ <b>59.6</b> | 60.5/ <b>61.3</b> | 20.9/ <b>36.4</b> | 39.3 / <b>50.1</b> |
| Mamba         | 2.8B   | 300B   | 54.3/ <b>66.6</b> | 38.9/48.9         | 33.5/40.1         | 43.9/ <b>59.4</b> | 66.2/63.9         | 19.8/ <b>36.9</b> | 42.8 / <b>52.6</b> |
| Mamba-2       | 130M   | 300B   | 32.2/ <b>50.9</b> | 29.5/43.3         | 20.6/ <b>28.9</b> | 30.4/47.0         | 43.7/ <b>47.2</b> | 18.0/ <b>34.0</b> | 29.1 / <b>42.0</b> |
| Mamba-2       | 370M   | 300B   | 60.8/ <b>76.7</b> | 38.3/52.1         | 26.6/ <b>33.6</b> | 35.3/ <b>51.8</b> | 54.6/ <b>54.7</b> | 22.4/ <b>36.3</b> | 39.7 / <b>50.9</b> |
| Mamba-2       | 1.3B   | 300B   | 66.8/ <b>74.7</b> | 50.0/ <b>59.6</b> | 33.6/40.5         | 42.9/ <b>59.6</b> | 63.8/62.4         | 23.2/ <b>36.6</b> | 46.7 / <b>55.6</b> |
| Mamba-2       | 2.7B   | 300B   | 68.7/ <b>81.6</b> | 55.2/60.8         | 34.4/41.7         | 45.4/ <b>59.4</b> | 66.4/66.5         | 23.0/42.5         | 48.9 / <b>58.8</b> |

Table 1: Evaluation of pre-trained language models. In each cell, we report in-context learning accuracy for the default zero-shot / JRT-PROMPT methods (using prompts provided in Appendix F). We evaluate across a suite of popular recall-intensive benchmarks. The zero-shot prompt includes up to 1k tokens in the input and JRT-PROMPT includes up to 2k tokens in the input for all tasks (due to repeating twice).

• Mamba-2 [36] pretrained LMs at the 130M, 370M, 1.3B, 2.7B parameter scales, trained on 300B tokens of the Pile [41]: https://huggingface.co/state-spaces

The results are summarized in Table 1. Arora et al. [7] finds that linear recurrent models like Mamba drastically underperform Transformers on these recall-intensive tasks. Architectures like Based increase the recurrent state size, improving both quality and efficiency, and recently Mamba-2 adopts this approach as well. Complementing the approach of increasing state size, we find the JRT-PROMPT modification provides  $11.0 \pm 1.3$  points of improvement, averaged across models and tasks: Based models with JRT-PROMPT outperform the Transformer models with standard prompting on average. We also find that JRT-PROMPT can benefit the Transformer models and that the method appears more effective than few-shot learning for these tasks (Appendix E). Notably, Springer et al. [43] recently proposes repeating the context for the goal of generating embeddings using autoregressive Transformer-based models, and our findings are in similar spirit. We focus on sub-quadratic architectures and in-context learning tasks.

JRT-PROMPT increases the context length due to repetition, however using using sub-quadratic recurrent architectures, this is still asymptotically more efficient than using quadratic Transformer models. We find at sequence length N=32768, batch size 16, Based with JRT-PROMPT (2N the sequence length) can provide  $11.9 \times$  higher throughput than FlashAttention-2 (N sequence length) on an NVidia H100 (see Section 5).

## <span id="page-5-1"></span>4 JRT-RNN: an encoder-decoder recurrent architecture

We have shown that the recall quality of causal fixed-memory recurrent models varies depending on the order in which the information appears in context, making them brittle for in-context learning. To improve reliability, we next propose a simple linear attention architecture that goes beyond causal modeling.

A long line of work has demonstrated the strength of non-causal bidirectional neural networks in language modeling [13, 44, 45, 46, 47, 48]. However, it is challenging to use them for fast text generation because the context must be re-processed for each generated token [14, 48, 49]. Encoder-decoder architectures with a bidirectional encoder and causal decoder offer a way to achieve fast causal generation while reaping the benefits of bidirectional LMs. Nonetheless, decoder-only causal LMs remain the norm and encoder-decoder architectures have received little attention in the context of sub-quadratic efficient LLMs.

#### 4.1 Preliminaries

Baseline linear recurrent architecture. We start from a recurrent architecture, linear attention, introduced in [50, 51, 52]. Current strong recurrent LMs (e.g., Based [7], GLA [9], Mamba-2 [36]) adopt linear attention with large recurrent state sizes. Prior work also theoretically shows that linear attention and state space models like Mamba [1] are closely related [7, 23, 36].

state space models like Mamba [1] are closely related [7, 23, 36]. Let q, k, v be linear projections of the input  $u \in \mathbb{R}^{N \times d}$ . The exponential in softmax attention is replaced by a feature map  $\phi : \mathbb{R}^d \to \mathbb{R}^{\tilde{d}}$ , from model dimension d to feature dimension  $\tilde{d}$ , such that  $\phi(\mathbf{q}_i)^{\top}\phi(\mathbf{k}_i) \approx \exp(\mathbf{q}_i^{\top}\mathbf{k}_i/\sqrt{d})$ . The linear attention computation can then be written as:

<span id="page-6-0"></span>
$$\mathbf{y}_i = \frac{\phi(\mathbf{q}_i) \sum_{j=1}^i \left( \phi(\mathbf{k}_j)^\top \mathbf{v}_j \right)}{\phi(\mathbf{q}_i) \sum_{j=1}^i \phi(\mathbf{k}_j)}$$
(1)

Multiplying keys and values first, the time and space complexity is  $\mathcal{O}(Nd\tilde{d})$  vs.  $O(N^2d)$  for softmax attention. Recurrent inference is split into two phases: prefill to process the input prompt and decoding to generate one token of the output at a time. During prefill, a length-l prompt is processed in parallel according to Equation (1) resulting in a "KV-state"  $s_l = \sum_{j=1}^{l} \phi(\mathbf{k}_j)^{\top} \mathbf{v}_j$  and "K-state"  $\mathbf{z}_l = \sum_{j=1}^{l} \phi(\mathbf{k}_j)^{\top}$ . During decoding, we can compute Equation (1) as:

<span id="page-6-2"></span>
$$s_i = s_{i-1} + \phi(\mathbf{k}_i)^{\top} \mathbf{v}_i, \qquad \mathbf{z}_i = \mathbf{z}_{i-1} + \phi(\mathbf{k}_i)^{\top}, \qquad \mathbf{y}_i = \frac{\phi(\mathbf{q}_i) \mathbf{s}_i}{\phi(\mathbf{q}_i) \mathbf{z}_i}$$
 (2)

where  $s_i \in \mathbb{R}^{d \times \tilde{d}}$  and  $z_i \in \mathbb{R}^{\tilde{d}}$ . Each decode step has O(1) time and space complexity as the sequence length grows, improving upon O(N) for softmax attention with KV-caching.

**Prefix-LM** architecture. Prefix-LM is a category of encoder-decoder models where inputs of length N are split into two regions: the first of length M is processed non-causally and the latter of length (N-M) is processed causally [13]. During loss computation, the former tokens are ignored and next-token-prediction loss is computed on the latter region. Excitingly, the design is quite simple, however prior instantiations of Prefix-LMs use inefficient softmax attention backbones and have not provided compelling benefits over decoder-only Transformers [15]. Prior prefix LM architectures have seen limited adoption.

#### 4.2 JRT-RNN architecture

JRT-RNN draws inspiration from Prefix-LMs, but focuses on expanding the Pareto frontier of the quality-efficiency tradeoff space. To improve quality, JRT-RNN uses separate  $k_e$ ,  $v_e$  projections on the encoder side and  $k_d$ ,  $v_d$  projections on the decoder side. While Prefix LM models use shared projection weights for the encoder and decoder regions, we find that using two sets of projections improves quality. This observation appears in early work on recurrent encoder-decoder architectures (Sutskever et al. [37]).

For efficiency, JRT-RNN uses non-causal linear attention for the encoder plus standard causal linear attention for the decoder. We term this Prefix Linear Attention (PLA) (Figure 1 (Right)):

<span id="page-6-1"></span>
$$\mathbf{y}_{i} = \frac{\phi(\mathbf{q}_{i})(\sum_{j=1}^{i} \phi(\mathbf{k}_{d_{j}})^{\top} \mathbf{v}_{d_{j}} + \sum_{j=1}^{M} \phi(\mathbf{k}_{e_{j}})^{\top} \mathbf{v}_{e_{j}})}{\phi(\mathbf{v}q_{i})(\sum_{j=1}^{i} \phi(\mathbf{k}_{d_{j}})^{\top} + \sum_{j=1}^{M} \phi(\mathbf{k}_{e_{j}})^{\top})}$$
(3)

Prior work has proposed many different instantiations of linear attention by varying the feature map  $\phi$  – PLA is a general approach, agnostic to the choice of feature map.

PLA retains the linear recurrent view,  $\mathcal{O}(1)$  time and space complexity for the inference decode step and the sub-quadratic in sequence length training complexity of standard causal linear attention [53]. During prefill, we process a length-l prompt in parallel according to Equation (3). If l < M, we left-pad the prefill to length M and mask the padded region during the linear attention computation. The recurrent state is initialized as:

$$\boldsymbol{s}_{M} = \sum_{j=1}^{M} (\phi(\boldsymbol{k}_{e_{j}})^{\top} \boldsymbol{v}_{e_{j}} + \phi(\boldsymbol{k}_{d_{j}})^{\top} \boldsymbol{v}_{d_{j}}), \qquad \boldsymbol{z}_{M} = \sum_{j=1}^{M} (\phi(\boldsymbol{k}_{e_{j}})^{\top} + \phi(\boldsymbol{k}_{d_{j}})^{\top})$$
(4)

Decoding for outputs  $y_i$ , i > M proceeds according to Equation (2), without modification.

Efficiency. Although linear attention is theoretically more efficient than softmax attention, existing implementations are generally *slower* than well-optimized standard attention implementations (e.g., FlashAttention [12]). Excitingly, [7] recently provides an IO-aware kernel that realizes the efficiency benefits of the Based linear attention architecture by carefully paritioning and storing the large matrix-valued recurrent state

across warp-registers during prefill (Algorithm 1 in [\[7\]](#page-12-6)). We extend their algorithm to support PLA, using the Based feature map (defined in Appendix [D\)](#page-28-0) in Algorithm [2](#page-29-0) and provide the efficiency results in Section [5.](#page-7-0) Additional details of our implementation are provided in Appendix [D.](#page-28-0)

The baseline causal linear attention takes 2BNHD FLOPS to compute the feature map on qd, kd, and 4BNHdD FLOPS for the kd, v<sup>d</sup> dot product, cumulative sum, q<sup>d</sup> dot product, and sum along the feature dimension D respectively. PLA increases the FLOPS by BMHD to compute the feature map on k<sup>e</sup> and 3BMHdD to compute the ke, v<sup>e</sup> dot product, sum along D, and sum the state with the decoder KV-state. PLA uses the same amount of memory (recurrent state size) during the inference decoding step as the original causal linear attention architecture.

