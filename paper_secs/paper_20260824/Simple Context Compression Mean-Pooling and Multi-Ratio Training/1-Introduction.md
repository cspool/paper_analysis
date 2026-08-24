# **1 Introduction**

Repeated reasoning over long documents is core for retrieval-augmented generation (RAG), where the same evidence is likely repeatedly processed across queries. This is computationally costly, both in time (processing and attending over the document) and memory (the key-value cache). *Context compression* addresses these costs through strategies such as soft compression, key-value (KV) cache compression, and hard prompt compression (e.g., [Ge](#page-10-0) [et al.,](#page-10-0) [2024;](#page-10-0) [Cheng et al.,](#page-9-0) [2024;](#page-9-0) [Dai et al.,](#page-9-1) [2025\)](#page-9-1).

Despite growing interest, progress remains difficult to measure. Evaluations differ in datasets, metrics, context lengths, and model scales, and even within soft compression the most common baseline, causal compression tokens, sets a low bar. Without a shared evaluation framework, it is unclear whether reported improvements represent real advances or favorable experimental choices. We address this gap by making two contributions.

First, we design a standardized, easy-to-reproduce evaluation suite for context compression, *BenchPress* [\(Section 4\)](#page-2-0). Although applicable to any compression paradigm, we instantiate it here for soft context compression, where we find that even simple methods can meaningfully advance the state of the art. The suite covers short (<1K tokens) and mid-range (<8K tokens) contexts with an explicit in-domain/out-of-domain split, teacher-normalized scoring for fair cross-model comparison, and a standardized training mixture to disentangle methodological contributions from data effects.

Second, we establish two simple but strong baseline methods for soft context compression. The first, *mean pooling*, is a compression operator that averages adjacent hidden states produced by a bidirectionally-encoding fine-tuned LLM, introducing no parameters beyond the encoder backbone. The second, *bidirectional compression tokens*, is a simple modification to the widely-used causal compression-token approach in which compression tokens attend bidirectionally among themselves, making the model aware of its compression budget, while retaining causal attention over the context. This modification has not been explored in prior work, despite its simplicity. Both baselines substantially outperform the standard causal compression-token approach.

Using our evaluation suite, we find that mean pooling achieves the strongest results overall, particularly at ratios up to  $16 \times$ ; at  $128 \times$ , bidirectional tokens are competitive or superior in several settings. A shared insight is that *bidirectional attention during encoding is critical for compression quality*. We also find that multi-ratio training is feasible with minor degradation for mean pooling, while bidirectional tokens even benefit from it. Compression quality scales favorably with model size, and mean pooling's advantages are even larger at longer contexts (8K tokens). We release our code and data at https://github.com/lil-lab/benchpress.

#### 2 Task Definition

we formally define soft context compression, the paradigm targeted by our baselines, although our evaluation suite (Section 4) is applicable to any compression paradigm. In the soft compression setting, a document of length L is mapped to a sequence of dense continuous vectors of length C, with  $L \gg C$ . This process allows an LLM that uses the compressed version of the document to invest significantly less computation, both in time and KV cache space, both reduced from dependence on L to dependence on C. This benefit increases with repeated use of the document, as is likely in RAG scenarios.

We define soft context compression to support flexible compression ratios. Let  $\mathcal{M}$  be a language model and  $\mathcal{R} \subseteq \mathbb{N}_+$  the admissible set of compression ratios. Let  $\mathcal{V}$  denote the vocabulary and d the embedding dimension of  $\mathcal{M}$ . The goal of learning is to construct a compression function

$$f_c: \mathcal{V}^L \times \mathcal{R} \to \mathbb{R}^{C \times d}$$
, (1)

which maps a token sequence  $T = (t_1, ..., t_L)$ ,  $t_i \in \mathcal{V}$  of length L and a ratio  $r \in \mathcal{R}$  to a compressed representation of C vectors of dimension d. The length C is determined by the specified ratio  $C = \lceil L/r \rceil$ .

An ideal compressor  $f_c$  preserves the conditional distribution of the model using the compressed version for any prompt P:

$$p_{\mathcal{M}}(\cdot \mid T, P) \approx p_{\tilde{\mathcal{M}}}(\cdot \mid f_c(T; r), P) ,$$
 (2)

where  $\tilde{\mathcal{M}}$  is a potentially adapted version of  $\mathcal{M}$ , for example augmented with lightweight parameter fine-tuning via LoRA (Hu et al., 2022) modules that can be fused into the base model without altering its capacity.

