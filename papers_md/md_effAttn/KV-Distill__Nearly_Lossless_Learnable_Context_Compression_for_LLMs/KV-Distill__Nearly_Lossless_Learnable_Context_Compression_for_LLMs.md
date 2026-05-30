# KV-DISTILL: Nearly Lossless Learnable Context Compression for LLMs

Vivek Chari<sup>1</sup> , Guanghui Qin<sup>2</sup> , Benjamin Van Durme<sup>1</sup>,<sup>2</sup> 1 Johns Hopkins University <sup>2</sup>Microsoft {vchari2,vandurme}@jhu.edu

## Abstract

Sequence-to-sequence tasks often benefit from long contexts, but the quadratic complexity of self-attention in standard Transformers renders this non-trivial. During generation, temporary representations – stored in the so-called KV cache – account for a large portion of GPU memory usage and scale linearly with context length. We introduce KV-DISTILL , a Transformer compression framework that distills long context KV caches into significantly shorter representations in a *questionindependent* fashion. KV-DISTILL can be trained as a parameter-efficient adaptor for pretrained models, and enables the compression of arbitrary spans of a context while preserving pre-trained model capabilities. We treat a compressed-uncompressed cache as a studentteacher pairing and apply a KL-type divergence to match the generated outputs. KV-DISTILL outperforms other compression techniques in worst-case extractive tasks and approaches uncompressed performance in long context question answering and summarization, and it can be fine-tuned on domain-specific contexts to reduce lengths by up to 99% while preserving downstream performance. We demonstrate the generalizability of KV-DISTILL across various model sizes and architectures.[1](#page-0-0)

### 1 Introduction

Harnessing the full potential of attention-based large language models (LLMs) often requires them to condition on long contexts. However, use of expansive contexts is complicated by the quadratic complexity of self-attention. In particular, during generation, one must maintain a store of all past key and value representations of past tokens (called the KV cache) that grows linearly with sequence length. The memory burden imposed by the KV cache is significant, and often limits the length of the sequences that a model can handle.

Much work has been devoted to architectural improvements to attention in order to reduce memory during generation. Strategies include augmenting sequences with memory tokens [\(Rae et al.,](#page-8-0) [2020;](#page-8-0) [Wu et al.,](#page-9-0) [2022\)](#page-9-0), sparsifying attention patterns [\(Beltagy et al.,](#page-8-1) [2020\)](#page-8-1), and using conditional computation to only process essential tokens [\(Ainslie](#page-8-2) [et al.,](#page-8-2) [2023\)](#page-8-2). However, such techniques have seen little widespread adoption due to performance drops on downstream tasks, or inefficient training/inference procedures. Even when given long contexts without compression, LLMs fail to fully utilize them [\(Qin et al.,](#page-8-3) [2022;](#page-8-3) [Liu et al.,](#page-8-4) [2024;](#page-8-4) [Lu](#page-8-5) [et al.,](#page-8-5) [2024\)](#page-8-5). Together this suggests long contexts may allow for significant compression while yielding large memory savings.

In what follows, we suppose that a prompt to a LLM is composed of contextual text(s) followed by a question whose answer is dependent on the provided context. KV compression can be divided into two paradigms: *question-aware*, and *questionindependent*. In question-aware compression, we have access to the question that we need answered, and can compress the context with this in mind. In question-independent compression, we do not know what questions will be asked in the future. For instance, consider a scenario in which a fixed textual context will be used to respond to many questions; the goal of question-independent compression is to compress this context once for reuse across many question.

Prior work in training-free context compression has primarily focused on which representations in the KV cache to select for eviction, with excellent results [\(Zhang et al.,](#page-9-1) [2023;](#page-9-1) [Li et al.,](#page-8-6) [2024\)](#page-8-6). In practice we observe that the performance of this selection procedure suffers greatly in the questionindependent paradigm. Furthermore, we anticipate that there is room for performance improvements in general-purpose context compression when the model is trained to handle for compression.

<span id="page-0-0"></span><sup>1</sup>Our code and checkpoints will be made available soon at this [link](https://github.com/vnchari/kv-distill)

![](_page_1_Figure_0.jpeg)

Figure 1: We subselect tokens from the KV cache and distill into the smaller subset

Prior work in trainable context compression have typically utilized a combination of cross-entropy and autoencoding objectives to pre-train general context compressors [\(Qin et al.,](#page-8-7) [2024;](#page-8-7) [Ge et al.,](#page-8-8) [2024;](#page-8-8) [Rae et al.,](#page-8-0) [2020\)](#page-8-0), which are suitable for question-independent compression. These loss functions have led to significant performance loss at high compression rates.

In this work we design a general-purpose trainable context compression method for LLMs that outperforms prior methods in both the *questionindependent* and *question-aware* paradigms. Our method, KV-DISTILL , accomplishes this, while also maintaining pretrained model capabilities, being suitable for long contexts, and having minimal performance penalty on downstream tasks. KV-DISTILL can support coherent, useful generation at compression ratios as high as 1000x.

To achieve this we train a scorer which retains the most important context tokens, while applying a parameter efficient adapter to conditionally modify important tokens' activations in-place. We further apply a token-level KL-type divergence to match the next-token prediction distributions, treating the compressed cache as a student, and the uncompressed cache as a teacher. KV-DISTILL only need be applied once to a fixed context, has zero overhead during auto-regressive decoding, and can compress arbitrary (sub)spans of a given context. We show improvements on several model families, considering extractive and abstractive tasks, with both short and long contexts, and at multiple model scales. KV-DISTILL is general purpose and has broad applicability to the LLM community.

## 2 Background

### <span id="page-1-0"></span>2.1 Key-Value Cache

Transformer-based language models (LMs) [\(Vaswani et al.,](#page-9-2) [2017\)](#page-9-2) use self-attention to aggregate context information and make predictions. A decoder-only transformer LM *autoregressively* predicts new tokens, and each step requires the LM to obtain the key and value states of all past tokens. To avoid re-computing the KV state of past tokens, most LM implementations (e.g. [Wolf](#page-9-3) [et al.](#page-9-3) [\(2020\)](#page-9-3)) cache the key and values states, in a structure called the KV cache. When making new predictions, self-attention is performed on query states of the new token and the KV -cache, and the new token's key and value representations are appended to the KV cache. Because the KV cache grows proportional to the number of tokens generated, maintaining the full KV cache in memory is a primary bottleneck when conditioning on large contexts. The goal of this work is to alleviate this by *compressing KV cache in the dimension of sequence length*, especially in the question-independent regime

### <span id="page-1-1"></span>2.2 Related Work

Much prior work has tackled the problem of reducing the complexity of the self-attention mechanism itself. Previous work tries to sparsify the attention patterns[\(Beltagy et al.,](#page-8-1) [2020;](#page-8-1) [Zaheer et al.,](#page-9-4) [2020\)](#page-9-4), use recurrence attention[\(Yang et al.,](#page-9-5) [2019\)](#page-9-5), or kernelize the attention matrix[\(Choromanski et al.,](#page-8-9) [2021\)](#page-8-9), but they require a considerable amount of further training.

Similar to our work, one line of work involves compressing the hidden states (KV cache) of past tokens into a shorter sequence of representations. For example, some methods learns "soft representations" of context (Qin and Eisner, 2021). Mu et al. (2023) compress particular prompts into much shorter "gist tokens", but do not attempt more general context compression. Furthermore, their method demonstrates poor generalizability, as performance does not scale with the number of gist tokens used. Zeng et al. (2023) propose to recognize and prioritize some important tokens (VIP tokens) during inference. Most relevant here, the following methods employ a similar idea of dynamically compressing the context prior to inference.

#### 2.3 Trainable Compression

Ge et al. (2024) design In-Context Autoencoder (ICAE) to compress long contexts for use in large language models (LLMs). ICAE consists of two main components: a learnable encoder and a fixed decoder. The encoder compresses the input context into a small number of memory slots. These memory slots are then used by the frozen LLMs (decoder) to reconstruct the context or respond to prompts. ICAE is pretrained using autoencoding and language modeling objectives on a large pretraining corpus and further fine-tuned using instruction data to maintain instruction-tuning. However, there is still a gap in downstream task performance when using an ICAE-compressed context, compared to an uncompressed context, and the method falters under high compression ratios.

Qin et al. (2024) propose DODO to compress sub-select KV activations to a set of "nugget" to-kens, which grow proportionally with the length of context sequence. Their method is trained with auto-encoding or language modeling objectives. However, DODO models operate at a fixed compression ratio, require training both an encoder and decoder, and still show a large gap in downstream task performance when compared to an uncompressed context.

#### 2.4 Training-Free Compression

Zhang et al. (2023) propose  $H_2$  to reduce memory usage during generation.  $H_2$  identifies "heavy-hitter" tokens, which significantly influence attention scores during inference. Specifically,  $H_2$  calculates the accumulated attention for each key and retains the top-k key-value pairs with the highest scores. In the question-aware setting, the accumu-

lated attention scores include scores from tokens in the question attending to the context. This effectively uses the question to scan for important details in the context. This allows H<sub>2</sub> to maintain nearly uncompressed performance at moderate compression ratios, by focusing on tokens most relevant to the current question. However, performance still degrades when compression ratios exceed 20×.

In the question-independent paradigm, the H<sub>2</sub> selection mechanism is applied solely to the context (as opposed to the context and question in the question-aware setting). We then allow the question to attend to only to this compressed context. We empirically observe that in the question-independent paradigm, H<sub>2</sub> performance plummets drastically, highlighting the need for improved question-independent compression methods. Lastly, H<sub>2</sub> offers no way to further improve compressive performance given prior domain knowledge.

Similarly SnapKV (Li et al., 2024) uses the attentions of a window of recent tokens to determine which context tokens are "heavy-hitters"; in the question-independent setting, this is undesirable, as the last tokens of a context may not necessarily provide additional information regarding attention patterns. In the question-independent paradigm we find that SnapKV performs similarly to H<sub>2</sub>, so do not compare against it in the remainder here.

# 3 Key-Value Distillation

We consider a transformer-based language model (Vaswani et al., 2017), denoted by LM, that is defined on the vocabulary  $\mathcal{V}$ . The KV-DISTILL process is then: (1) a set of important tokens in the input context is determined; (2) an adapted language model LM $_{\theta}$  is used to encode the context into a KV cache, and sub-select the aforementioned important tokens from the generated KV cache; and (3) the unmodified LM conditions on the compressed KV cache to auto-regressively generate it's output.

#### 3.1 State Selection for Cache Compression

Let  $\mathbf{c} = \{w_i\}_{i=1}^N$  represent a context consisting of N tokens, where  $w_i \in \mathcal{V}$  and  $\vec{c} \in \mathcal{V}^N$ . In a typical scenario, LM predicts a sequence of new tokens, denoted by  $\vec{y}$ , conditioning on  $\mathbf{c}$ . For example,  $\mathbf{c}$  may be a prompt and LM generates  $\vec{y}$  as a response. Future token prediction draws on information from past tokens via *attention* by having LM encode the context tokens into key and value hidden states

 $\vec{X}_l^{(K)}, \vec{X}_l^{(V)} \in \mathbb{R}^{N \times d}$ , which taken together form the KV cache (Section 2.1), where d is the dimension of the transformer and l is the layer of LM. We may drop the subscript l and superscripts  $^{(K)}$  and  $^{(V)}$  and use  $\vec{X}$  to generally denote the key/value states of transformers at any layer.

Transformers assume that  $\vec{X}$  fully describes and represents the context  $\vec{c}$ . However, attending to  $\vec{X}$  can be inefficient when  $\vec{c}$  is long. Therefore, we further assume that retaining a subset of key/value states is sufficient for approximating the next-token distribution conditioned on all key/value states. That is, we could retain rows from  $\vec{X}$  to form  $\tilde{\vec{X}} \in \mathbb{R}^{k \times d}$ , where  $k \leq N$  is the number of selected rows. We use a subset of the tokens' hidden states to represent the complete context, which is plausible because representations in  $\tilde{\vec{X}}$  are conditioned on the prior context. Suppose we determine the  $(i_1,\ldots,i_k)$ -th tokens are to be retained in layer l. We use a hard selection matrix  $\vec{S}_l \in \{0,1\}^{k \times N}$  to derive (layer-specific)  $\tilde{\vec{X}}_l$  from  $\vec{X}_l$  by

$$\tilde{\vec{X}}_l = \vec{S}_l \vec{X}_l, \quad \vec{S}_l = [\vec{e}_{(i_1)}, \dots, \vec{e}_{(i_k)}], \quad (1)$$

where  $\vec{e}_{(i)} \in \{0,1\}^N$  is the *i*-th standard basis vector. Note that this formulation does not require that the same tokens be selected across each layer.

The problem of determining which indices  $(i_1, \ldots, i_k)$  to retain still remains. We would like the subselection  $\vec{S}$  to retain most of the context information given a fixed k. One possibility is to use a feedforward neural network to measure the importance of each token position:

$$\vec{s} = \text{FFN}_{\theta} \left( \vec{X'}_{\eta} \right) \tag{2}$$

where  $\theta$  is the parameters of the FFN,  $\vec{s} \in \mathbb{R}^N$  and  $\vec{s}_i$  indicate the "importance score" of the i-th token and  $\vec{X}'_{\eta}$  indicates the hidden states at the  $\eta$ -th layer. The indices  $i_{1:k}$  can then be derived by taking the tokens with the top-k scores. We can control the percent of the KV cache retained by scaling k with the length of the context. In our we retain the same  $i_{1:k}$  across all layers and take  $\eta=6$ .

The above selection procedure is rendered non-differentiable by the top-k operator. We may propagate gradients to the scorer by decaying the attention weights of tokens attending to  $\tilde{\vec{X}}$  inversely proportionally to their computed importance scores. More precisely, let  $\vec{z} \in \mathbb{R}^d$  represent the hidden

<span id="page-3-0"></span>![](_page_3_Picture_8.jpeg)

Figure 2: Selected tokens are routed to trainable, LoRA-adapted  $\vec{W}^Q$  and  $\vec{W}^O$  matrices ( $\vec{W}^O$  is omitted in this figure); all other tokens pass through the original (frozen) model parameters.

state of a single token attending to  $\vec{\tilde{X}}$  with unnormalized attention weights  $\alpha$ :

$$\alpha = \left(\vec{z}\vec{W}^{Q}\right)\left(\tilde{\vec{X}}^{(K)}\right)^{\top} \tag{3}$$

we decay  $\alpha$  to produce scorer-informed attention weights  $\alpha'$ :

$$\alpha' = \sigma(\vec{s}) \odot \alpha, \tag{4}$$

where  $\odot$  denotes the element-wise (Hadamard) product and  $\sigma$  the sigmoid function. We note that the above formulation is one of many possible scoring functions that can be used with KV-DISTILL, that could be learnable or parameter-free, and could potentially have layer-wise specificity. We leave exploring different scoring functions as future work.

#### <span id="page-3-1"></span>3.2 Architecture

After sub-selecting important token indices, we pass the context  $\vec{c}$  through a modified  $LM_{\theta}$  that uses conditional computation to condense the context into  $\tilde{X}$ . This allows for the representations of important tokens to be "packed" with information from unselected tokens, and is strictly more expressive than only subselection. We instantiate  $LM_{\theta}$  with LoRA adaptors (Hu et al., 2022) to minimize the number of trainable parameters.

More importantly, within  $LM_{\theta}$ , the subselected tokens are routed to trainable  $\vec{W}^Q, \vec{W}^O$  matrices, where  $\vec{W}^Q, \vec{W}^O$  are the query and output matrices of transformers, while discarded tokens are routed to the original (frozen) matrices, as shown in Figure 2. This has the effect of informing  $LM_{\theta}$  as to which tokens are selected, allowing for specialized aggregation of the value representations for selected tokens. This method of informing  $LM_{\theta}$  has minimal overhead (the LoRA matrices account for

under 500MB of GPU memory for a 27B parameter model), and only a single set parameters must be maintained in memory.

We anticipate that other architectural forms could make KV-DISTILL effective. However, we find that applying conditional computation to inform LM<sup>θ</sup> of selected tokens is important to the performance of KV-DISTILL . We find that some methods of informing the model of selected tokens, such as by adding a trainable embedding to these tokens, do not work well (see Appendix [B\)](#page-10-0). The particular architecture chosen has the advantage of lower memory usage during training, and provides excellent performance. We leave the task of finding even more efficient architectures to future work.

### 3.3 Objective Function

After generating compressed cache ˜X⃗ , we aim to match the output of LM when conditioned on ˜X⃗ to the output of LM when conditioned on X. Previous compression methods [\(Ge et al.,](#page-8-8) [2024;](#page-8-8) [Qin](#page-8-13) [and Van Durme,](#page-8-13) [2023\)](#page-8-13) rely on the autoencoding objective to pretrain LMθ. However, given that LM predicts future tokens during inference, there is a discrepancy in pretraining and downstream usage, which could result in performance loss. Instead we propose matching the next-token probability distribution of tokens conditioned on X⃗ and ˜X⃗ . Consider a generative language model that predicts the next token ⃗y<sup>t</sup> conditioned on the past tokens ⃗y<t and a fixed context ⃗c that is represented by either X⃗ or ˜X⃗ . We would like to minimize the difference between their next-token distributions, i.e. p ⃗yt | ⃗y<t, X⃗ and q<sup>θ</sup> ⃗yt | ⃗y<t, ˜X⃗ . Let q<sup>θ</sup> indicate the distribution that conditions on the distilled KV cache ˜X⃗ . Also note that the only learnable parameters in this formulation arise from *encoding* ˜X⃗ ; during auto-regressive generation we use the original frozen parameters of LM.

Given probability distributions p, qθ, we use the forward and reverse KL divergences to measure their similarity. With simplified notations we have:

$$\mathcal{D}_{\mathrm{KL}}(p||q_{\theta}) = \mathbb{E}_{y \sim p(\cdot)} \left[ \log \left( \frac{p(y)}{q_{\theta}(y)} \right) \right]$$

$$\mathcal{D}_{\mathrm{KL}}(q_{\theta}||p) = \mathbb{E}_{y \sim q_{\theta}(\cdot)} \left[ \log \left( \frac{q_{\theta}(y)}{p(y)} \right) \right] \quad (5)$$

The mode-seeking and mean-seeking behavior of the reverse- and forward- KL divergences respectively is well known. To incorporate both behaviors

into the objective, we sum the forward and reverse divergences:

<span id="page-4-1"></span>
$$\mathcal{L}(\theta) = \lambda \cdot \mathcal{D}_{KL}(p||q_{\theta}) + (1 - \lambda) \cdot \mathcal{D}_{KL}(q_{\theta}||p), (6)$$

where a hyperparameter λ controls the balance between forward and reverse KL divergence.

Given both p and q<sup>θ</sup> are categorical distribution, both KL divergences in eq. [\(5\)](#page-4-0) can be analytically solved. Tthe L1-norm of the gradient of the reverse divergence dominates nearly everywhere. As such we propose scaling the forward and reverse terms by having λ > 0.5 in eq. [\(6\)](#page-4-1). The benefit of λ is confirmed with the ablations in Appendix [B.](#page-10-0)

## 4 Experiments

To assess the efficacy of KV-DISTILL , we conduct experiments on LLAMA-2 7B, LLAMA-3 8B, MIS-TRAL 7B,GEMMA-2 9B and GEMMA-2 27B. In all cases we use the instruction-tuned model. A KV-DISTILL model is obtained by distilling on a large corpus to obtain strong general-purpose context compressors.

Data We curate a large instruction dataset from Self-Instruct, P3, LongAlpaca, and Super-Natural Instructions [\(Soboleva et al.,](#page-9-7) [2023;](#page-9-7) [Wang et al.,](#page-9-8) [2022a;](#page-9-8) [Sanh et al.,](#page-9-9) [2021;](#page-9-9) [Chen et al.,](#page-8-14) [2023;](#page-8-14) [Wang](#page-9-10) [et al.,](#page-9-10) [2022b\)](#page-9-10). Training instances are split into (Context, Instruction, Answer) triples. In cases where the context is sufficiently long (more than 1536 tokens), we pad to a multiple of 1536 and fold the context to a batch of N × 1536 instances, compress the resulting KV cache, and then unfold the cache. Empirically, we observe little performance degradation when applying folding during pretraining, while allowing the model to see longer examples. We also always leave the first few (< 10) tokens of the context uncompressed, as we find that retaining them improves performance; this is not a new observation, see [Han et al.](#page-8-15) [\(2024\)](#page-8-15) and [Xiao](#page-9-11) [et al.](#page-9-11) [\(2023\)](#page-9-11).

Training The general training procedure is as follows: (1) pass the (Context, Instruction, Answer) triple through the original model to obtain target logits, (2) apply the KV-DISTILL architecture to obtain logits conditioned on the compressed cache (we compress the context, and leave the instruction uncompressed), (3) apply Equation [6](#page-4-1) between the obtained and target logits.

<span id="page-4-0"></span>We use rank-stabilized LoRA on the Q, K, V, O matrices with r = 128 to train LM<sup>θ</sup> [\(Hu et al.,](#page-8-12) [2022;](#page-8-12) [Kalajdzievski,](#page-8-16) [2023\)](#page-8-16). Note that the K, V

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 3: Needle-in-a-Haystack results; The x-axis shows the length of the document, the y-axis indicates the compression ratio applied, and the color the accuracy of retrieval under those settings averaged across different locations in the document. Left: H2I. Right: KV-DISTILL .

are trainable for all tokens, not just selected tokens. The behavior of the Q, O adapters is discussed in Section [3.2.](#page-3-1) Optimization is done using Deepspeed Stage 2, and the AdamW optimizer [\(Rasley et al.,](#page-9-12) [2020\)](#page-9-12). During pretraining, we sample KV retention fractions between 0.1-80%. As such, all KV-DISTILL models support arbitrary retention rates. All models are distilled on a cluster of 8 NVIDIA A100 80GB GPUs. All models except GEMMA 27B converged within 3 days, while GEMMA 27B took 4 days. See Appendix [A](#page-9-13) for further details.

<span id="page-5-0"></span>

| Dataset   | Average | Max |
|-----------|---------|-----|
| SQuAD     | 225     | 1k  |
| QuALITY   | 6k      | 9k  |
| SQuALITY  | 7k      | 11k |
| GovReport | 10k     | 71k |

Table 1: Evaluation dataset example length statistics in tokens under the LLAMA-3 tokenizer

Evaluation In all datasets, we have a natural (Context, Question) pairing. We always compress the context and leave the question uncompressed. Evaluations are performed with greedy decoding. Summary statistics regarding the context length of evaluation dataset are provided in Table [1.](#page-5-0)

Methods Tested We evaluate against DODO [\(Qin](#page-8-7) [et al.,](#page-8-7) [2024\)](#page-8-7), ICAE [\(Ge et al.,](#page-8-8) [2024\)](#page-8-8), and H<sup>2</sup> in both the question-aware (H2A) and questionindependent (H2I) forms. Please refer to Sec [2.2](#page-1-1) for further description about the selected methods. To improve the performance of H<sup>2</sup> , we also retain a set of sink tokens as described in [Xiao et al.](#page-9-11) [\(2023\)](#page-9-11).

### 5 Results

## 5.1 Needle-In-a-Haystack

Motivation The Needle-in-a-Haystack test [\(Kamradt,](#page-8-17) [2023\)](#page-8-17) evaluates a model's ability to accurately retrieve information from a sentence (a "needle") embedded within a large document (a "haystack"), in which a sentence is randomly positioned. In our experiments, the needle is a semantic one. In Figure [3,](#page-5-1) we show the results of H2I (left) and KV-DISTILL (right) at various compression ratios at different document lengths. The accuracy is computed across the placement of the needle within a document. Crucially, at compression time, the model does not know that a is needle being sought, nor that a needle is placed within the context.

Results We see that KV-DISTILL significantly outperforms H2I at almost all compression ratios and document lengths. In particular, KV-DISTILL demonstrates near-perfect accuracy even after removing 90% of the KV .

### 5.2 Extractive Question Answering

Motivation SQuAD is an extractive questionanswering task. We hypothesize that extractive tasks will suffer the largest performance loss under context compression. As such, we choose to use performance on SQuAD as a proxy for generalpurpose compressive ability of a model. In all the following experiments, we choose the pretraining checkpoint with the best SQuAD performance for further experimentation. To assess accuracy we generate an answer conditioned on the compressed context, checking if the generated response is contained in the ground-truth answer.

<span id="page-6-0"></span>

| Model      |                          | % KV                     | 0-Shot Acc.                                         |
|------------|--------------------------|--------------------------|-----------------------------------------------------|
| LLAMA3     | BASE                     | 100%                     | 87.6 ± .6%                                          |
|            | KVD<br>KVD               | 25%<br>20%               | 86.6 ± .7%<br>86.0 ± .7%                            |
|            | H2A<br>H2A<br>H2I<br>H2I | 25%<br>20%<br>25%<br>20% | 84.0 ± .7%<br>83.0 ± .7%<br>56.6 ± .9%<br>51.7 ± 1% |
|            | DODO                     | 20%                      | 73.3 ± .8%                                          |
| LLAMA2 7B  | BASE                     | 100%                     | 82.5 ± .7%                                          |
|            | KVD<br>KVD               | 25%<br>20%               | 79.1 ± .8%<br>77.6 ± .8%                            |
|            | H2A<br>H2A<br>H2I<br>H2I | 25%<br>20%<br>25%<br>20% | 77.9 ± .7%<br>76.7 ± .7%<br>55.2 ± .9%<br>50.3 ± 1% |
|            | ICAE                     | 57%                      | 75.0 ± .8%                                          |
| GEMMA 9B   | BASE                     | 100%                     | 85.15 ± .7%                                         |
|            | KVD<br>KVD               | 25%<br>20%               | 84.55 ± .7%<br>83.1 ± .7%                           |
| GEMMA 27B  | BASE                     | 100%                     | 85.3 ± .8%                                          |
|            | KVD<br>KVD               | 25%<br>20%               | 83.1 ± 1%<br>82.2 ± 1%                              |
| MISTRAL 7B | BASE                     | 100%                     | 87.1 ± .6%                                          |
|            | KVD<br>KVD               | 25%<br>20%               | 84.1 ± .7%<br>82.5 ± .7%                            |

Table 2: Zero-shot accuracy on SQuAD at selected KV retention ratios.

Results Table [2](#page-6-0) contains SQuAD accuracy results. We see that in all cases, KV-DISTILL models perform within a few percentage points of base models, even under a "worst-case" task. Furthermore, KV-DISTILL models significantly outperform prior trainable methods (ICAE, DODO), even when retaining less of the KV cache. KV-DISTILL models significantly outperform H2I, demonstrating the ability of the pretraining objective and architecture to create reusable compressed KV representations. KV-DISTILL models also enjoy a slight improvement over H2A at similar compression ratios, demonstrating the effectiveness of its compression at capturing almost all salient information in the context, even without question-awareness.

When retaining under 20% of KV , we observe rapid declines in performance across all methods, indicating the difficulty of the task under high context compression. Lastly, we note that initial pretraining hyperparameters for all models were set based on initial experimentation with LLAMA-3 and SQuAD; as such, we anticipate that perfor-

<span id="page-6-1"></span>![](_page_6_Figure_4.jpeg)

Figure 4: QuALITY accuracy against compression.

mance of most models can be improved with hyperparameter tuning during the pre-training process.

### 5.3 Long Context Question Answering

Motivation QuALITY is a long document multiple-choice question answering dataset that assesses reading comprehension. We use QuAL-ITY to assess the decision making capabilities of models equipped with distilled contexts. To assess QuALITY accuracy, we use the same evaluation procedure used by LLAMA-3 [\(AI@Meta,](#page-8-18) [2024\)](#page-8-18). Results Figure [4](#page-6-1) shows the experiment results on QuALITY, with data points at the following retention rates highlighted: {100%, 25%, 20%, 10%, 5%, 1%, 0.1%}. We observe that KV-DISTILL performs similarly to the uncompressed cache, with only minor losses in performance at 10x compression. Although not included in Figure [4,](#page-6-1) 0% cache retention results in accuracy of 32.4%, 25.8%, and 24.4% for the LLAMA-3, MISTRAL,

# 5.4 Long Context Abstractive Summarization

despite eliminating 99.9% of the context.

and GEMMA-2 models respectively, demonstrating the neccessity of the context for the task. Impressively, we see significant improvements over the random accuracy even when distilling to as few as 7 tokens from a 7k input passage; for example, on LLAMA-3 we observe only a 20% drop in accuracy

Motivation SQuALITY is a question-focused summarization dataset based on the same collection of long documents as the QuALITY benchmark. We use it to evaluate the abstractive summarization capabilities of models trained with distilled contexts. We compute the Rouge-L scores [\(Lin,](#page-8-19) [2004\)](#page-8-19) between the generated summaries and ground-truth

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 5: Rouge-L on SQuALITY

answers, following the same evaluation protocol used by LLAMA-3 [\(AI@Meta,](#page-8-18) [2024\)](#page-8-18).

Result Figure [5](#page-7-0) show Rouge-L performance on SQuALITY. We observe that KV-DISTILL models perform as well or better than uncompressed models when retaining more that 20% of the KV cache. When retaining under 20%, we observe different performance falloff behaviors for different models; in particular, we observe that textscLlama-3 and GEMMA-2 have stable performance until 100x compression, at which point performance dips drastically. This difference in the behavior of the compression-performance trade-off could be attributed to the larger vocabulary sizes of LLAMA-3 and GEMMA-2, which allows the KL-loss to capture more fine-grained features of the output distribution during pretraining. These results demonstrate that KV-DISTILL can support very high compression ratios with minimal performance penalty on abstractive tasks.

### 5.5 Finetuned Long Context Summarization

Motivation GovReport is a long document summarization dataset that consists (Report, Summary) pairs written by government research agencies. In contrast to the evaluations on QuALITY and SQuALITY (which are performed in a zero-shot fashion using the best pretraining checkpoint), we perform additional finetuning distillation with Equation [6](#page-4-1) on the GovReport training set before evaluation. As with SQuALITY, we use GovReport to assess the abstractive summarization ability of models equipped with KV-DISTILL .

Results Table [3](#page-7-1) shows results for GovReport for both query-aware H2A and query-independent H2I paradigms, as well as KV-DISTILL prior to fine-

<span id="page-7-1"></span>

|              |      |      | KVD       |          |
|--------------|------|------|-----------|----------|
| KV retention | H2A  | H2I  | Zero Shot | Finetune |
| 100%         | 23.7 | 23.7 | 23.7      | 23.7     |
| 20%          | 22.8 | 20.6 | 22.3      | 23.5     |
| 10%          | 22.4 | 18.6 | 21.8      | 23.3     |
| 5%           | 21.9 | 18.5 | 21.1      | 23.2     |
| 1%           | 21.1 | 18.3 | 20.1      | 22.8     |

Table 3: ROUGE-L on GovReport summarization.

tuning (zero-shot) and KV-DISTILL after finetuning on LLAMA-3. We observe that KV-DISTILL and query-aware H<sup>2</sup> perform close to each other on this evaluation in the zero-shot setting, while KV-DISTILL outperforms H2I at all compression ratios. However, upon finetuning, we observe a practical improvement in performance with KV-DISTILL , with little degradation from uncompressed performance across all compression rates. In particular, we note the improvement in performance is greater at more severe compression ratios, confirming the utility of KV-DISTILL in supporting ultra-high compression ratios.

### 6 Discussion and Conclusion

We develop a method to reduce the memory requirements of long-context conditioned LM generation. Our method sub-selects tokens from the KV cache, and applies a token-level KL-type loss between the output of the LM when conditioned on sub-selected tokens and when conditioned on the uncompressed cache. We evaluate our method on long-context extractive and abstractive tasks, and demonstrate improved performance over competing compression methods. We further demonstrate that continued training on domain-specific data can allow for use of compression ratios as high as 100x with negligible losses in performance.

As part of this work we release distilled checkpoints across various model language families. These artifacts allow efficient text generation conditioned on significantly larger inputs than before, with much lower memory burden, and support compression ratios as high as 1000x. We anticipate these artifacts will be of great practical benefit, enabling exciting new applications and research directions in language processing.

### 7 Limitations

The time-consuming and stochastic nature of distilling a model means that it cannot be guaranteed that the process will work well across all model

families. Furthermore, we are unsure as to the root cause of performance discrepancies between model architectures after distillation; this issue merits further research. Lastly, the 8k token context-capacity of LLAMA-3 limited many of our experiments, and is small by the standards of currently available language models; to address this, we will be releasing a KV-DISTILL LLAMA-3.1 model with a 128k token context capacity.

Used Artifacts In this work, we used the publicly released codes and checkpoints of LLAMA. Per the license attached to LLAMA, we agree not to re-distribute their parameters and limit the usage of the models for research purposes only

## References

<span id="page-8-18"></span>AI@Meta. 2024. [Llama 3 model card.](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)

- <span id="page-8-2"></span>Joshua Ainslie, Tao Lei, Michiel de Jong, Santiago Ontañón, Siddhartha Brahma, Yury Zemlyanskiy, David Uthus, Mandy Guo, James Lee-Thorp, Yi Tay, Yun-Hsuan Sung, and Sumit Sanghai. 2023. [CoLT5:](http://arxiv.org/abs/2303.09752) [Faster Long-Range Transformers with Conditional](http://arxiv.org/abs/2303.09752) [Computation.](http://arxiv.org/abs/2303.09752) In *Proceedings of Conference on Empirical Methods in Natural Language Processing (EMNLP)*.
- <span id="page-8-1"></span>Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. [Longformer: The Long-Document Transformer.](http://arxiv.org/abs/2004.05150)
- <span id="page-8-14"></span>Yukang Chen, Shaozuo Yu, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023. Long alpaca: Long-context instruction-following models. [https://github.com/dvlab-research/](https://github.com/dvlab-research/LongLoRA) [LongLoRA](https://github.com/dvlab-research/LongLoRA).
- <span id="page-8-9"></span>Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. 2021. [Rethinking Attention with Per](http://arxiv.org/abs/2009.14794)[formers.](http://arxiv.org/abs/2009.14794) In *Proceedings of International Conference on Learning Representations (ICLR)*.
- <span id="page-8-8"></span>Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. [In-context Autoencoder for Context Com](http://arxiv.org/abs/2307.06945)[pression in a Large Language Model.](http://arxiv.org/abs/2307.06945) In *Proceedings of International Conference on Learning Representations (ICLR)*.
- <span id="page-8-15"></span>Chi Han, Qifan Wang, Hao Peng, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. 2024. [LM](https://doi.org/10.18653/v1/2024.naacl-long.222)[infinite: Zero-shot extreme length generalization for](https://doi.org/10.18653/v1/2024.naacl-long.222) [large language models.](https://doi.org/10.18653/v1/2024.naacl-long.222) In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 3991–4008, Mexico City, Mexico. Association for Computational Linguistics.

- <span id="page-8-12"></span>Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. [LoRA: Low-Rank Adaptation of Large](http://arxiv.org/abs/2106.09685) [Language Models.](http://arxiv.org/abs/2106.09685) In *Proceedings of International Conference on Learning Representations (ICLR)*.
- <span id="page-8-16"></span>Damjan Kalajdzievski. 2023. [A rank stabilization scal](https://arxiv.org/abs/2312.03732)[ing factor for fine-tuning with LoRA.](https://arxiv.org/abs/2312.03732) *Preprint*, arXiv:2312.03732.
- <span id="page-8-17"></span>G Kamradt. 2023. Needle in a haystack–pressure testing llms.
- <span id="page-8-6"></span>Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: Llm knows what you are looking for before generation. *arXiv preprint arXiv:2404.14469*.
- <span id="page-8-19"></span>Chin-Yew Lin. 2004. [ROUGE: A package for auto](https://aclanthology.org/W04-1013)[matic evaluation of summaries.](https://aclanthology.org/W04-1013) In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.
- <span id="page-8-4"></span>Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. [Lost in the Middle: How Language](https://doi.org/10.1162/tacl_a_00638) [Models Use Long Contexts.](https://doi.org/10.1162/tacl_a_00638) *Transactions of the Association for Computational Linguistics (TACL)*, 12:157–173.
- <span id="page-8-5"></span>Taiming Lu, Muhan Gao, Kuai Yu, Adam Byerly, and Daniel Khashabi. 2024. [Insights into LLM long](https://arxiv.org/abs/2406.14673)[context failures: When transformers know but don't](https://arxiv.org/abs/2406.14673) [tell.](https://arxiv.org/abs/2406.14673) *Preprint*, arXiv:2406.14673.
- <span id="page-8-11"></span>Jesse Mu, Xiang Lisa Li, and Noah Goodman. 2023. [Learning to Compress Prompts with Gist Tokens.](http://arxiv.org/abs/2304.08467) In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*.
- <span id="page-8-10"></span>Guanghui Qin and Jason Eisner. 2021. [Learning How to](https://doi.org/10.18653/v1/2021.naacl-main.410) [Ask: Querying LMs with Mixtures of Soft Prompts.](https://doi.org/10.18653/v1/2021.naacl-main.410) In *Proceedings of Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*.
- <span id="page-8-3"></span>Guanghui Qin, Yukun Feng, and Benjamin Van Durme. 2022. [The nlp task effectiveness of long-range trans](https://www.semanticscholar.org/paper/8db711adf1beb3e0c2ec492f3936841d827404e9)[formers.](https://www.semanticscholar.org/paper/8db711adf1beb3e0c2ec492f3936841d827404e9) In *Conference of the European Chapter of the Association for Computational Linguistics*.
- <span id="page-8-7"></span>Guanghui Qin, Corby Rosset, Ethan C. Chau, Nikhil Rao, and Benjamin Van Durme. 2024. [Dodo: Dy](https://doi.org/10.48550/arXiv.2310.02409)[namic Contextual Compression for Decoder-only](https://doi.org/10.48550/arXiv.2310.02409) [LMs.](https://doi.org/10.48550/arXiv.2310.02409) In *Proceedings of Annual Meeting of the Association for Computational Linguistics (ACL)*.
- <span id="page-8-13"></span>Guanghui Qin and Benjamin Van Durme. 2023. [Nugget:](https://proceedings.mlr.press/v202/qin23a.html) [Neural Agglomerative Embeddings of Text.](https://proceedings.mlr.press/v202/qin23a.html) In *Proceedings of International Conference on Machine Learning (ICML)*.
- <span id="page-8-0"></span>Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, and Timothy P. Lillicrap. 2020. [Compressive Trans](http://arxiv.org/abs/1911.05507)[formers for Long-Range Sequence Modelling.](http://arxiv.org/abs/1911.05507) In

*Proceedings of International Conference on Learning Representations (ICLR)*.

<span id="page-9-12"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. [DeepSpeed: System Opti](https://doi.org/10.1145/3394486.3406703)[mizations Enable Training Deep Learning Models](https://doi.org/10.1145/3394486.3406703) [with Over 100 Billion Parameters.](https://doi.org/10.1145/3394486.3406703) In *Proceedings of International Conference on Knowledge Discovery and Data Mining (KDD)*.

<span id="page-9-9"></span>Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush. 2021. [Multi](https://arxiv.org/abs/2110.08207)[task prompted training enables zero-shot task gener](https://arxiv.org/abs/2110.08207)[alization.](https://arxiv.org/abs/2110.08207) *Preprint*, arXiv:2110.08207.

<span id="page-9-7"></span>Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. 2023. [SlimPajama: A 627B token cleaned and deduplicated](https://huggingface.co/datasets/cerebras/SlimPajama-627B) [version of RedPajama.](https://huggingface.co/datasets/cerebras/SlimPajama-627B)

<span id="page-9-2"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention Is All](https://arxiv.org/pdf/1706.03762.pdf) [You Need.](https://arxiv.org/pdf/1706.03762.pdf) In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*.

<span id="page-9-8"></span>Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022a. Self-instruct: Aligning language model with self generated instructions.

<span id="page-9-10"></span>Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022b. [Super-NaturalInstructions: Generaliza](https://aclanthology.org/2022.emnlp-main.340)[tion via declarative instructions on 1600+ NLP tasks.](https://aclanthology.org/2022.emnlp-main.340) In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*.

<span id="page-9-3"></span>Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick Von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin

Lhoest, and Alexander Rush. 2020. [Transformers:](https://doi.org/10.18653/v1/2020.emnlp-demos.6) [State-of-the-Art Natural Language Processing.](https://doi.org/10.18653/v1/2020.emnlp-demos.6) In *Proceedings of Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

<span id="page-9-0"></span>Y. Wu, M. N. Rabe, D. Hutchins, and C. Szegedy. 2022. Memorizing Transformers. In *Proceedings of International Conference on Learning Representations (ICLR)*.

<span id="page-9-11"></span>Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. *arXiv*.

<span id="page-9-5"></span>Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. 2019. [XLNet: Generalized Autoregressive Pretraining for](https://arxiv.org/abs/1906.08237) [Language Understanding.](https://arxiv.org/abs/1906.08237) In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*.

<span id="page-9-4"></span>Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. 2020. [Big Bird: Transformers for](http://arxiv.org/abs/2007.14062) [Longer Sequences.](http://arxiv.org/abs/2007.14062) In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*.

<span id="page-9-6"></span>Zhanpeng Zeng, Cole Hawkins, Mingyi Hong, Aston Zhang, Nikolaos Pappas, Vikas Singh, and Shuai Zheng. 2023. [VCC: Scaling Transformers to 128K](http://arxiv.org/abs/2305.04241) [Tokens or More by Prioritizing Important Tokens.](http://arxiv.org/abs/2305.04241) In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*.

<span id="page-9-1"></span>Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. [H2o: Heavy-Hitter](http://arxiv.org/abs/2306.14048) [Oracle for Efficient Generative Inference of Large](http://arxiv.org/abs/2306.14048) [Language Models.](http://arxiv.org/abs/2306.14048) In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*.

### <span id="page-9-13"></span>A Training & Evaluation Details

We train all KV-DISTILL models with the following parameters at bf16 precision on 8 NVIDIA A100s. Please see Table [4](#page-9-14) for further details. All models had 150M trainable parameters. The QuALITY,

<span id="page-9-14"></span>

| Hyperparameter | Value |  |
|----------------|-------|--|
| Optimizer      | AdamW |  |
| Learning Rate  | 5e-5  |  |
| Batch Size     | 32    |  |
| LoRA Rank      | 128   |  |
| λ              | 0.6   |  |
| η              | 6     |  |

Table 4: Hyperparameters for training

SQuALITY, GovReport, and SQuAD evaluations

are performed on the test set, if public, else results are reported on the development set. To measure SQuAD accuracy, we generate up to 128 tokens, normalize the output by stripping punctuation, and check if the correct answer is contained in the generated answer. For SQuALITY and QuALITY, we follow the evaluation procedure of [AI@Meta](#page-8-18) [\(2024\)](#page-8-18). For GovReport, we prompt the model to summarize the report, and then greedily generate 630 tokens.

# <span id="page-10-0"></span>B Pretraining Objective Ablations

We assess the necessity of both the forward and reverse terms in the loss by evaluating SQuAD performance on multiple different pre-training losses with varying λ values in Equation [6.](#page-4-1) In Table [5](#page-10-1) we observe that using either the pure forward or reverse divergences performs markedly worse than using a mixture of both. Furthermore, using solely the auto-encode + cross-entropy (used in ICAE and DODO), performs significantly worse than Equation [6,](#page-4-1) demonstrating the significant benefits that the weighted distillation objective provide. We also note that replacing the routing mechanism with a learnable embedding (added to important tokens) does not perform well.

<span id="page-10-1"></span>

| Loss       | SQuAD Acc. (%) |  |
|------------|----------------|--|
| λ = 1      | 83.4%          |  |
| λ = 0.6    | 86.0%          |  |
| λ = 0.4    | 85.3%          |  |
| λ = 0      | 82.7%          |  |
| AE + LM    | 79.1%          |  |
| NO ROUTING | 67.4 %         |  |

Table 5: Effect of Pretraining Loss wrt LLAMA-3.

### C Qualitative Analysis of Generated Text

In Figure [6](#page-10-2) we present representative summaries of the short story "Galactic Ghost". LLAMA-3 was prompted to summarize the short story in a few sentences. As the KV retention ratio decreases, we observe an increase in statements that are unfaithful to the original context. Particularly, at extremely high compression ratios, the model begins to hallucinate details, but impressively still maintains understanding of the general plot of the story. Note the varying degrees of *semantic compression* in summaries generated under high compression ratios; qualitatively, we observe that models with severely compressed contexts generally understand the text, but resort to making vague statements.

#### (a) 100% KV retention

<span id="page-10-2"></span>The story follows Willard, a space explorer who is left alone after his friend and space mate, Dobbin, dies. Willard is haunted by the memory of Dobbin's final words, "I saw the Ghost Ship," and becomes obsessed with finding the ship. He spends years searching for it, and eventually, he sees a strange, ghostly ship that seems to be heading towards him. As he gets closer, he realizes that it's not just a hallucination, but a real ship from Earth that's been traveling through space for centuries. In the end, Willard is taken aboard the ship, but learns that the ship and its crew are mere shells of their former selves, and that they are doomed to remain in space forever. Ultimately, Willard is forced to confront the reality that he'll never return to Earth.

#### (b) 2.5% KV retention

The story follows the journey of Willard, a space traveler who has been alone in space for years after the death of his friend and partner, Dobbin. Willard is struggling to come to terms with his isolation and the loss of his friend, and is haunted by the legend of the Ghost Ship, a mythical vessel that is said to roam the solar system, never to be seen again. One day, Willard spots a strange, rocket-shaped object approaching him, and as it draws closer, he realizes it is the Mary Lou. The ship is now a mere shadow of its former self, and Willard is invited to board it, where he is greeted by a rescue team who have been sent to retrieve him. As he adjusts to life on the rescue ship, Willard comes to terms with his past and the reality of his situation, and ultimately accepts that he will never return to Earth. The story concludes with Willard's poignant realization that he and the crew of the rescue ship are the Ghosts, doomed to roam the solar system forever.

### (c) .25% KV retention

The story follows the journey of Willard, a space explorer who has been traveling through the galaxy for years, searching for a new home for humanity. He has lost his friend and companion, Dobbin, and is struggling to come to terms with his own mortality. One day, he discovers a faint light in the distance, which he believes to be a sign of life. As he approaches the light, he realizes it is a ghost ship, a ship that has been drifting through space for millions of years, its crew and passengers long dead. The ship is a relic of a bygone era, and Willard is forced to confront the reality that he will never be able to return to Earth, as the ship would pass through it, making it impossible to survive. In the end, Willard comes to accept his fate, realizing that he and the crew of the ghost ship are doomed to drift through space forever, a reminder of the transience of human existence.

Figure 6: LLAMA-3 was tasked with summarizing a 6k token short story at low KV retention rates. Inaccuracies in the summary are highlighted yellow, and were determined by hand.