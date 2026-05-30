# <span id="page-1-0"></span>2 Related Work

Reducing the memory and computational overhead of the KV cache in Transformer decoders has been a focal point of recent research. MQA [\[38\]](#page-12-5) reduces KV cache size by sharing a single key and value head across all query heads, while GQA [\[1\]](#page-10-3) divides query heads into groups and each shares a single key and value head. MLA [\[26\]](#page-11-2) compresses KV representations into a lower-dimensional latent space, offering better expressiveness than GQA and comparable or improved accuracy over MHA. Additionally, techniques like MiniCache [\[28\]](#page-11-3) and MLKV [\[50\]](#page-12-7) reduce memory by sharing KV caches across layers, though this may harm performance due to layer-specific attention patterns.

Another line of work explores linear attention models such as Linear Transformers [\[23,](#page-11-4) [45\]](#page-12-8), RWKV [\[35\]](#page-11-5), and Mamba [\[19\]](#page-11-6), which reduce memory via linear time complexity. However, they often struggle with long-range dependencies, impacting tasks that rely on complex context. Recent theoretical analysis [\[2\]](#page-10-4) also proves that truly subquadratic inference time can not solve challenging tasks such as document similarity. Despite the cost, quadratic attention remains crucial for fine-grained token interactions, motivating our focus on Transformer attention.

Beyond architectural modifications, various engineering techniques have been proposed to optimise Transformers. Dynamic token pruning methods, such as LazyLLM [\[17\]](#page-10-5) and SnapKV [\[24\]](#page-11-7), reduce memory usage by selectively removing less important tokens from the KV cache. [\[49\]](#page-12-9) divides the context into chunks and inserts beacon tokens that store and accumulate information, effectively representing previous chunks to achieve context compression. Pruning can also be applied to attention heads or dimensions, though it may compromise contextual understanding and complicate the pipeline [\[30\]](#page-11-1). In addition, KV quantisation [\[27\]](#page-11-8) can further reduce memory by lowering KV cache precision. Furthermore, FlashAttention [\[12,](#page-10-6) [13\]](#page-10-7) restructures the attention computation to minimise memory access overhead, enhancing both speed and efficiency. While these tricks enhance Transformer efficiency, this paper focuses on directly compressing the KV cache along the temporal dimension, an under-explored direction that can greatly reduce memory and computation for long-sequence tasks. [32] retrofits pre-trained LLMs by temporally compressing the KV cache, but cannot train from scratch and requires extra losses for each attention layer and head. In contrast, this work proposes a new attention mechanism requiring no changes beyond the attention module itself.

### 3 Preliminaries and Background

This section reviews some important background on the use of a KV-cache in auto-regressive inference and the operation of standard multi-head attention. The approaches taken by the MQA, GQA and MLA methods for reducing the size of the KV-cache are then outlined.

**Key-Value Cache in Auto-regressive Inference** At inference, the model generates one next token  $x_i$  at a time, using past tokens  $x_1, \dots, x_{i-1}$ . To reduce computation, Transformers cache previously computed key and value vectors instead of re-computing the attention context for each step.

Given a query vector  $q_i \in \mathbb{R}^{1 \times d}$  at step i, where d is the model dimension, and the cached key and value matrices  $\mathbf{K}_{< i} \in \mathbb{R}^{(i-1) \times d}$  and  $\mathbf{V}_{< i} \in \mathbb{R}^{(i-1) \times d}$ , the attention output is computed as:

$$\operatorname{Attention}(\mathbf{q}_{i}, \mathbf{K}_{< i}, \mathbf{V}_{< i}) = \operatorname{softmax}\left(\frac{\mathbf{q}_{i} \mathbf{K}_{< i}^{\top}}{\sqrt{d}}\right) \mathbf{V}_{< i} \tag{1}$$

Here,  $q_i$  is computed from  $x_i$ , and  $\mathbf{K}_{< i}$ ,  $\mathbf{V}_{< i}$  are cached from previous steps. Without caching,  $\mathbf{K}_{< i}$  and  $\mathbf{V}_{< i}$  must be re-computed at every step, leading to redundant computation and quadratic time.

**Multi-Head Attention (MHA)** Given an input sequence  $\mathbf{X} \in \mathbb{R}^{T \times d}$ , where T denotes the sequence length, MHA [44] projects it into query  $\mathbf{Q}$ , key  $\mathbf{K}$ , and value tensors  $\mathbf{V}$  using learned weight matrices:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_O \in \mathbb{R}^{T \times (n_h \cdot d_h)}, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{T \times (n_h \cdot d_h)}, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{T \times (n_h \cdot d_h)}$$
(2)

where  $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times (n_h \cdot d_h)}$  are learned matrices, and  $n_h$  is the number of attention heads.

**Multi-Query Attention (MQA)** MQA [38] shares key and value matrices across heads to reduce memory. Each head h has its own query  $\mathbf{Q}^{(h)} = \mathbf{X}\mathbf{W}_O^{(h)} \in \mathbb{R}^{T \times d_h}$ , but all heads share:

$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{T \times d_h}, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{T \times d_h}$$
 (3)

**Group-Query Attention (GQA)** GQA [1] groups heads into g sets, each sharing a key and value.

$$\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{T \times (g \cdot d_h)}, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{T \times (g \cdot d_h)}$$
 (4)

Heads in group i share  $\mathbf{K}^{(i)}, \mathbf{V}^{(i)} \in \mathbb{R}^{T \times d_h}$ . Each head has independent queries as in MHA.

**Multi-Head Latent Attention (MLA)** MLA [26] compresses the key-value memory into a latent sequence  $C \in \mathbb{R}^{T \times r}$  with a smaller hidden dimension r < d. The attention computation becomes:

$$\mathbf{C} = \mathbf{X}\mathbf{W}_r \in \mathbb{R}^{T \times r} \tag{5}$$

$$\mathbf{K} = \mathbf{C}\mathbf{W}_K \in \mathbb{R}^{T \times (n_h \cdot d_h)}, \quad \mathbf{V} = \mathbf{C}\mathbf{W}_V \in \mathbb{R}^{T \times (n_h \cdot d_h)}$$
(6)

where C is saved as KV cache and directly used for attention computation, avoiding explicit K and V computation by absorbing  $W_K$  into  $W_Q$  and  $W_V$  into the output projection.

### 4 Multi-head Temporal Latent Attention (MTLA)

This paper proposes MTLA, which, building upon compressing the Key-Value (KV) cache into a low-rank latent space as in MLA, further compresses the KV cache along the temporal dimension. Hence, MTLA can greatly reduce GPU memory usage and accelerate inference. Meanwhile, MTLA addresses the challenge of mismatched KV cache length and generated sequence length by introducing a stride-aware causal mask, enabling efficient parallel training.

As illustrated in Fig. 1, unlike conventional Multi-Head Attention (MHA) that maintains separate key and value cache vectors for each attention head, MTLA employs a shared low-rank latent vector

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Figure 1: Illustration of MTLA. Blue arrows denote transformations by linear layers, and the red dashed lines indicate content attended to during attention. The example corresponds to 4 attention heads. (a) Standard MHA; (b) MTLA with a temporal compression ratio of 2.  $\oplus$  denotes addition. The transformation from compressed temporal-latent KV cache to multi-head KVs can be absorbed into the query/output linear layers via matrix multiplication associativity, avoiding redundant computation.

to compress key and value information across heads, following [26]. Furthermore, MTLA merges adjacent latent vectors along the temporal dimension to store them as the KV cache.

Specifically, given an input sequence  $\mathbf{X} \in \mathbb{R}^{T \times d}$ , where T is the sequence length and d is the model dimension, the multi-head queries  $\mathbf{Q} = (\mathbf{q}_1, \mathbf{q}_2, \cdots, \mathbf{q}_T)$  are computed following standard MHA:

<span id="page-3-4"></span>
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_O \in \mathbb{R}^{T \times (n_h \cdot d_h)} \tag{7}$$

where  $\mathbf{W}_Q \in \mathbb{R}^{d \times (n_h \cdot d_h)}$  are learned linear weight matrices. Following [26], low-rank compression (dimension is r) is performed to obtain the low-rank latent vectors  $\mathbf{C} = (\mathbf{c}_1, \mathbf{c}_2, \cdots, \mathbf{c}_T)$ :

<span id="page-3-2"></span>
$$\mathbf{C} = \mathbf{X}\mathbf{W}_r \in \mathbb{R}^{T \times r} \tag{8}$$

where  $\mathbf{W}_r \in \mathbb{R}^{d \times r}$  is a trainable weight matrix. Layer normalisation [4] is then applied to  $\mathbf{C}$  to stabilise training, following the implementation in [26]. MTLA further applies learnable weights  $(w_1, w_2, \ldots, w_T)$  to compress the latent sequence  $\mathbf{C}$  along the temporal dimension, yielding a shorter compressed temporal-latent KV sequence  $\hat{\mathbf{C}} = (\hat{c}_1, \hat{c}_2, \cdots, \hat{c}_t) \in \mathbb{R}^{t \times r}$ , where  $t = \lceil T/s \rceil$  and s denotes the temporal compression ratio.

As illustrated in Fig. 1, assuming s=2, every 2 temporally adjacent latent vectors in  ${\bf C}$  are merged using the corresponding weights  $(w_1,w_2,\ldots,w_T)$ ; for example,  $\hat{{\bf c}}_1=w_1\cdot{\bf c}_1+w_2\cdot{\bf c}_2$ , and  $\hat{{\bf c}}_2=w_3\cdot{\bf c}_3+w_4\cdot{\bf c}_4$ . Since the length of  $(w_1,w_2,\ldots,w_T)$  varies dynamically with the input and cannot be handled using static parameters, MTLA utilises a hyper-network that takes  ${\bf C}$  as input to generate  $(w_1,w_2,\ldots,w_T)$ . Further details of this hyper-network are given in refer to Sections 4.1 and 4.2. The choice of s effectively controls the extent of KV cache compression in MTLA. However, choosing too large a value can caused marked performance degradation.

With the cached  $\hat{\mathbf{C}} \in \mathbb{R}^{t \times r}$ , the keys  $\mathbf{K}$  and values  $\mathbf{V}$  can be obtained through up-projection matrices and used for attention computation:

<span id="page-3-5"></span>
$$\mathbf{K} = \hat{\mathbf{C}}\mathbf{W}_K \in \mathbb{R}^{t \times (n_h \cdot d_h)},\tag{9}$$

$$\mathbf{V} = \hat{\mathbf{C}}\mathbf{W}_V \in \mathbb{R}^{t \times (n_h \cdot d_h)},\tag{10}$$

<span id="page-3-1"></span>
$$\mathbf{Y} = \operatorname{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{d_h}} \right) \mathbf{V} \mathbf{W}_O \in \mathbb{R}^{T \times d}$$
 (11)

where  $\mathbf{W}_K$ ,  $\mathbf{W}_V \in \mathbb{R}^{r \times (n_h \cdot d_h)}$ , and  $\mathbf{W}_O \in \mathbb{R}^{(n_h \cdot d_h) \times d}$  are are learned linear weight matrices. Note that due to the associative property of matrix multiplication, Eq. 11 can be rewritten as:

<span id="page-3-3"></span>
$$\operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}}\right)\mathbf{V}\mathbf{W}_O = \operatorname{softmax}\left(\frac{\mathbf{X}(\mathbf{W}_Q\mathbf{W}_K^{\top})\hat{\mathbf{C}}^{\top}}{\sqrt{d_h}}\right)\hat{\mathbf{C}}(\mathbf{W}_V\mathbf{W}_O)$$
(12)

![](_page_4_Figure_0.jpeg)

<span id="page-4-2"></span>Figure 2: Illustration of MTLA inference and training with temporal compression ratio 2.  $q_i$ : query,  $x_i$ : attention input,  $\hat{c}_j$ : compressed KV cache,  $\hat{c}_j'$ : temporary version updated later. (a) Incremental inference in MTLA, where at certain steps (e.g., 1, 3, 5), the model attends to the temporary  $\hat{c}_j'$  (b) KV cache generated by simple pre-downsampling, which mismatches MTLA inference. (c) MTLA training, where a stride-aware causal mask is used to match the inference condition.

Therefore, the cached  $\hat{\mathbf{C}}$  can be directly used for attention computation without explicitly computing the keys and values, as  $\mathbf{W}_K$  and  $\mathbf{W}_V$  can be absorbed into  $\mathbf{W}_Q$  and  $\mathbf{W}_O$ , respectively.

#### <span id="page-4-0"></span>4.1 Inference using MTLA

Fig. 2(a) illustrates inference using MTLA. Specifically, given a new input vector  $x_i$ , the corresponding low-rank latent vector  $c_i$  is first computed following Eq. 8. Then,  $c_i$  is fed into the hyper-network to generate the corresponding weight  $w_i$ . Specifically, the weight is computed as follows:

<span id="page-4-3"></span>
$$w_i = \text{Sigmoid}\left(\text{Linear}(\boldsymbol{c}_i) \cdot \text{Linear}(\boldsymbol{p}\boldsymbol{e}_i)\right)$$
 (13)

where  $j = \lceil i/s \rceil$ , Linear denotes a linear layer transformation,  $pe_j$  is the positional embedding at step j [44], and  $\cdot$  denotes element-wise multiplication.

Once  $w_i$  is obtained, the compressed temporal-latent KV cache can be updated. If the remainder of i/s equals 1 (assuming i starts from 1), the cache is updated as  $\hat{\mathbf{C}} = \operatorname{Concat}(\hat{\mathbf{C}}, w_i \mathbf{c}_i)$  where Concat denote concatenation; otherwise, the j-th cache vector is updated as  $\hat{\mathbf{c}}_j = \hat{\mathbf{c}}_j + w_i \mathbf{c}_i$ . Note that until the remainder of i/s equals 0, each  $\hat{\mathbf{c}}_j$  here actually corresponds to  $\hat{\mathbf{c}}_j'$  in Fig. 2, which will be updated in later steps. Then, the attention output is computed following Eq. 12.

#### <span id="page-4-1"></span>4.2 MTLA Training with Stride-aware Causal Mask

As shown in Fig. 2(a), during inference, queries at certain steps attend to the temporary  $\hat{c}_j'$ . As shown in Fig. 2(b), simply using pre-downsampling to obtain compressed KV vectors for attention computation during training fails to match inference behaviour. Therefore, enabling efficient parallel training poses a challenge. This paper proposes a stride-aware causal mask to address this issue.

During training, as shown in Fig. 2(c), MTLA computes the compressed temporal-latent KV sequence as:

$$\hat{\mathbf{C}}' = (\hat{\underline{c}}'_1, \dots, \hat{\underline{c}}_1, \dots, \hat{\underline{c}}'_t, \dots, \hat{\underline{c}}_t)$$
(14)

where s is the temporal compression ratio and  $t = \lceil T/s \rceil$ . Therefore, this sequence length remains T (only in training). To compute the sequence  $\hat{\mathbf{C}}'$ , the compressed low-rank latent vectors  $\mathbf{C}$  are first passed through a hyper-network. To ensure parallel training efficiency, MTLA computes  $\hat{\mathbf{C}}'$  using matrix multiplication. Specifically, the hyper-network generates a weight matrix based on the input

 $\mathbf{C}$ :

$$\mathbf{PE} = (\underbrace{pe_1, \dots, pe_1}_{s}, \dots, \underbrace{pe_t, \dots, pe_t}_{s})$$
(15)

<span id="page-5-2"></span>
$$\mathbf{W} = \operatorname{Sigmoid}(\operatorname{Linear}(\mathbf{PE}) \times \operatorname{Linear}(\mathbf{C})) \in \mathbb{R}^{T \times T}$$
(16)

where **PE** consists of the replicated positional embedding vectors  $pe_j$  and × denotes matrix multiplication. As shown in the upper part of Fig. 2(c), after applying chunk masking (commonly used in streaming Transformer encoders [10]) to the resulting **W**, it is multiplied with **C** to obtain  $\hat{\mathbf{C}}'$ .

The resulting  $\hat{\mathbf{C}}'$  is then used for attention computation as in Eq. 12 (serving as  $\hat{\mathbf{C}}$  in Eq. 12). However, instead of using a standard causal mask to prevent access to future information before the softmax, a stride-aware causal mask is proposed, as shown in the lower part of Fig. 2(c), to match the attention pattern of MTLA during incremental inference. Specifically, let m denote the row index and n the column index; the stride-aware causal mask is zero only when n=m or n < m and  $n \mod s = 0$ , and  $-\infty$  elsewhere. With this stride-aware causal mask, MTLA training retains the parallel efficiency of standard attention.

### <span id="page-5-1"></span>4.3 Decoupled Rotary Position Embedding in MTLA

If Rotary Position Embedding (RoPE) [39] is to be used, similar to MLA [26], MTLA also requires the use of decoupled RoPE [26]. A simple method is proposed in this paper to compress the cached keys of decoupled RoPE along the temporal dimension. Specifically, the queries obtained from Eq. 7 are rotated with a position-dependent matrix to produce RoPE queries  $\mathbf{Q}^R = (\mathbf{q}_1^R, \mathbf{q}_2^R, \cdots, \mathbf{q}_T^R) \in \mathbb{R}^{T \times (n_h \cdot d_h^R)}$ , where  $d_h^R$  denotes per-head dimension for the decoupled RoPE. Similarly, the keys can also be obtained as in Eq. 9 and rotated with a position-dependent matrix to obtain RoPE keys  $\mathbf{K}^R = (\mathbf{k}_1^R, \mathbf{k}_2^R, \cdots, \mathbf{k}_T^R) \in \mathbb{R}^{T \times d_h^R}$ .

Next,  $\mathbf{K}^R$  is compressed along the temporal dimension to obtain  $\hat{\mathbf{K}}^R = (\hat{\boldsymbol{k}}_1^R, \hat{\boldsymbol{k}}_2^R, \cdots, \hat{\boldsymbol{k}}_t^R) \in \mathbb{R}^{t \times d_h^R}$ . At inference, the most recent element in the RoPE key cache  $\hat{\mathbf{K}}^R$  can also be updated. If the remainder of i/s equals 1, this cache is updated as  $\hat{\mathbf{K}}^R = \operatorname{Concat}(\hat{\mathbf{K}}^R, \boldsymbol{k}_i^R)$ ; otherwise, the j-th cache vector is updated as  $\hat{\boldsymbol{k}}_j^R = \boldsymbol{k}_i^R$ . Then, the RoPE query-key pairs are used to augment the attention computation and Eq. 11 and Eq. 12 can be rewritten as:

<span id="page-5-0"></span>
$$\mathbf{Y} = \operatorname{softmax} \left( \frac{\mathbf{X}(\mathbf{W}_Q \mathbf{W}_K^{\top}) \hat{\mathbf{C}}^{\top} + \mathbf{Q}^R (\hat{\mathbf{K}}^R)^{\top}}{\sqrt{d_h}} \right) \hat{\mathbf{C}}(\mathbf{W}_V \mathbf{W}_O)$$
(17)

where  $\mathbf{X} \in \mathbb{R}^{1 \times d}$  in incremental inference, and when multiplying  $\mathbf{Q}^R \in \mathbb{R}^{T \times (n_h \cdot d_h^R)}$  with  $(\hat{\mathbf{K}}^R)^{\top} \in \mathbb{R}^{d_h^R \times T}$ , the head number of keys must first be repeated, following MQA [38].

This design of compressing decoupled RoPE keys along the temporal dimension simplifies the training process: based on Eq. 17, the original  $\mathbf{K}^R \in \mathbb{R}^{T \times d_h^R}$  can be directly used in place of  $\hat{\mathbf{K}}^R$  (also using  $\hat{\mathbf{C}}'$  instead of  $\hat{\mathbf{C}}$  as mentioned in Section 4.2), and the attention output can be computed with the proposed stride-aware causal mask.

Assuming the number of self-attention layers is l, then for standard MHA, each token corresponds to  $2d_hn_hl$  elements in the KV cache. For MTLA, for simplicity, this paper follows the hyperparameter settings of [26], setting  $r=4d_h$  and  $d_h^R=d_h/2$ . Therefore, the average number of KV cache elements per token in MTLA is  $9d_hl/(2s)$ . The default value of s is set to 2, making  $9d_hl/(2s)=2.25d_hl$  close to the KV cache elements per token in MQA (i.e.  $2d_hl$ ).

### 5 Experimental Setup

In this section, the proposed MTLA approach is evaluated on a range of tasks, including speech translation (ST), text summarisation, automatic speech recognition (ASR), and spoken language understanding (SLU), and is compared with standard MHA and advanced MLA. Since this work focuses on self-attention, the experiments are conducted using a Transformer-based decoder-only architecture, implemented within the Fairseq [33] toolkit.

#### 5.1 Datasets

The ST task uses the MuST-C [16] v1.0 English-German (En-De) dataset, with data preprocessing following the Fairseq example. The text summarisation task is conducted on the XSum [31] dataset. For the ASR task, the AMI [8] dataset is employed. For the SLU task, the SLURP [5] dataset is used to evaluate intent classification. More details of the datasets used are given in Appendix C.

### 5.2 Model Specifications

Since this paper focuses on self-attention, the model is built based on a Transformer decoder, where the encoder output is prepended to the input of the self-attention module as a prompt, and the cross-attention module is removed. This is sometimes referred to as a decoder-only structure. As a result, the cached keys and values will contain information from the encoder output. The proposed MTLA, along with the standard MHA and the MLA technique, are each used as the self-attention module to build the model, while all other components are kept strictly identical. In the following sections, the overall models built with MTLA, MHA, and MLA self-attention modules are referred to as MTLA, MHA, and MLA for simplicity.

The decoder used for all tasks shares the same configuration with 512 attention dimensions and 8 heads. For MTLA and MLA, r in Eq. 8 is set to 256 and  $d_h^R$  is set to 32. In MTLA, the temporal compression rate s is set to 2 by default unless otherwise specified. For the ST task, following the Fairseq example, a Transformer encoder is used and initialised with ASR task weights. For the text summarisation task, a standard Transformer encoder is used. For the ASR task, a Transformer encoder is employed. For the SLU task, a Conformer [20] encoder is used. More details can be found in Appendix D.

#### <span id="page-6-0"></span>5.3 Metrics

All inference speed tests are conducted on the same NVidia RTX 6000 Ada GPU. To ensure a fair comparison, all models used the same batch size and beam size during inference. Inference time and the average GPU memory usage during inference are reported to evaluate efficiency. For the ST task, case-sensitive detokenized BLEU [34] is reported. For the text summarisation task, ROUGE [25] is used to evaluate summarisation quality, and ROUGE-1, ROUGE-2 (unigram and bigram overlap), and ROUGE-L (longest common subsequence) scores are reported. For speech recognition, word error rate (WER) results are reported. For the SLU task, accuracy is used to measure intent classification (IC).

### 6 Experimental results

This paper evaluates the proposed MLTA across tasks, including ST, text summarisation, ASR, and SLU, as both speech sequences and document texts are long sequences. Due to our computational resource constraints that make large-scale pre-training infeasible, all experiments are conducted using decoder-only architectures trained from scratch, allowing the effectiveness of MTLA to be assessed. To ensure reproducibility, this paper builds upon standard open-source implementations, such as the Transformer-based ST example in Fairseq. The goal is not to pursue task-specific state-of-the-art results, but to systematically compare MTLA with MHA and MLA under consistent and general model configurations. For each task, representative published results are reported to provide context. Appendix E presents MTLA's performance on the LRA benchmark [41], while Appendix F provides machine translation results to further evaluate MTLA on tasks involving relatively shorter sequences.

#### 6.1 ST Task Results

The ST results are shown in Table 1. Overall, the models built in this paper achieve competitive performance on the MuST-C En-De benchmark dataset. The published results listed in Table 1 also use Transformer models, but based on an encoder-decoder architecture with cross-attention. Table 1 results show that our built decoder-only architecture can achieve similar performance with the same data and model scale. Comparing MHA and MLA, it is clear that MLA performs well: MLA results in only a limited reduction in translation quality drop (by 0.19 BLEU points) and offers improved inference speed and memory efficiency compared to MHA. Building upon MLA,

<span id="page-7-0"></span>Table 1: BLEU (↑) results on the MuST-C En-De tst-COMMON set for multi-head attention (MHA), multi-head latent attention (MLA), and multi-head temporal latent attention (MTLA). ESPnet-ST [\[21\]](#page-11-15) published results are broadly comparable (same data/scale; minor implementation differences).

|                        | Quality                       | Inference |            | Inference GPU Memory (MiB) |      |
|------------------------|-------------------------------|-----------|------------|----------------------------|------|
| ST Model               | Speedup<br>(BLEU)<br>Time (s) |           | Avg. Usage | Reduction Factor           |      |
| ESPnet-ST [21]         | 22.9                          | —         | —          | —                          | —    |
| MHA                    | 23.18                         | 281.3     | 1.00×      | 18646                      | 1.00 |
| MLA                    | 22.97                         | 97.0      | 2.90×      | 5065                       | 3.68 |
| Proposed MTLA          | 23.28                         | 65.6      | 4.29×      | 2835                       | 6.58 |
| Proposed MTLA w/ s = 3 | 23.25                         | 52.7      | 5.34×      | 2251                       | 8.28 |
| Proposed MTLA w/ s = 4 | 23.05                         | 48.7      | 5.78×      | 1921                       | 9.71 |

our proposed MTLA further improves the efficiency of the attention mechanism. With the default temporal compression ratio (i.e., 2), MTLA even slightly outperforms MHA in translation quality, suggesting that compressing redundant historical KV information may sometimes benefit model performance. Compared to MHA, MTLA achieves 4.29× speedup in inference and reduces average GPU memory consumption by a factor of 6.58.

Assuming the sequence length is T, MTLA reduces the per-token computational complexity during decoding from O(T) to O(T /s). Since self-attention is not the only component in the model (e.g., feed-forward networks also contribute), setting s = 2 does not directly halve the inference time. Moreover, the reported GPU memory usage includes both activation memory and the storage of KV Cache, so memory consumption is not halved either. Nevertheless, setting s = 2 already yields substantial efficiency gains: MTLA achieves a 1.48× speedup in overall inference and reduces overall GPU memory consumption by 1.79× compared to MLA. These gains become even more substantial with larger s. For instance, with s = 4, GPU memory usage is reduced by 2.64×.

### 6.2 Results on Other Tasks

<span id="page-7-1"></span>Table 2: ROUGE (↑) results on the XSum test set. ROUGE-1 (R1) (↑), ROUGE-2 (R2) (↑), and ROUGE-L (RL) F1 (↑) scores are reported. The published result of TransformerABS [\[29\]](#page-11-16) is broadly comparable to our results.

| Model               | R1    | R2   | RL       | Inference | Speedup |            | Inference GPU Memory (MiB) |
|---------------------|-------|------|----------|-----------|---------|------------|----------------------------|
|                     |       |      | Time (s) |           |         | Avg. Usage | Reduction Factor           |
| TransformerABS [29] | 29.41 | 9.77 | 23.01    | —         | —       | —          | —                          |
| MHA                 | 28.83 | 9.67 | 23.33    | 352.3     | 1.00×   | 16141      | 1.00                       |
| MLA                 | 29.39 | 9.87 | 23.78    | 141.1     | 2.50×   | 3746       | 4.30                       |
| Proposed MTLA       | 29.14 | 9.79 | 23.60    | 105.2     | 3.35×   | 2198       | 7.34                       |

Table 3: WER (↓) results on the AMI IHM test set for MHA, MLA, and the proposed MTLA. ESPnet published [\[46\]](#page-12-12) results are listed but not directly comparable to our built models.

<span id="page-7-2"></span>

|               |       | Inference |         | Inference GPU Memory (MiB) |                  |  |
|---------------|-------|-----------|---------|----------------------------|------------------|--|
| Model         | WER   | Time (s)  | Speedup | Avg. Usage                 | Reduction Factor |  |
| ESPnet [46]   | 16.49 | —         | —       | —                          | —                |  |
| MHA           | 12.98 | 269.4     | 1.00×   | 17509                      | 1.00             |  |
| MLA           | 12.67 | 105.3     | 2.56×   | 4415                       | 3.97             |  |
| Proposed MTLA | 12.66 | 71.8      | 3.75×   | 2364                       | 7.41             |  |

Experiment conclusions across text summarisation, ASR, and SLU tasks (Tables [2,](#page-7-1) [3,](#page-7-2) and [4\)](#page-8-0) are generally consistent with those from the ST experiments. First, our built models achieve competitive performance across different tasks. Second, compared to MHA, MLA achieves competitive accuracy (ROUGE scores, WER, and IC accuracy) and better inference efficiency. Our proposed MTLA further improves inference efficiency. Compared to MHA, MTLA achieves up to 3.75× speedup and

<span id="page-8-0"></span>Table 4: Accuracy (↑) results of intent classification (IC) on the SLURP test set for MHA, MLA, and the proposed MTLA. ESPnet-SLU [\[3\]](#page-10-13) published result is generally comparable to our built models.

|                |          | Inference |         | Inference GPU Memory (MiB) |                  |  |
|----------------|----------|-----------|---------|----------------------------|------------------|--|
| Model          | Accuracy | Time (s)  | Speedup | Avg. Usage                 | Reduction Factor |  |
| ESPnet-SLU [3] | 86.3     | —         | —       | —                          | —                |  |
| MHA            | 86.83    | 133.1     | 1.00×   | 14370                      | 1.00             |  |
| MLA            | 86.93    | 61.2      | 2.17×   | 3343                       | 4.30             |  |
| Proposed MTLA  | 86.80    | 52.7      | 2.53×   | 2051                       | 7.01             |  |

reductions in GPU memory use by more than a factor of 7, while maintaining or even improving task performance. These results highlight the broad applicability and practical benefits of our decoder-only architecture and MTLA KV cache compression method across various sequence tasks.

### <span id="page-8-2"></span>6.3 Comparisons with Related Work

<span id="page-8-1"></span>Table 5: BLEU (↑) results on the MuST-C En-De tst-COMMON set for related methods, including Multi-Query Attention (MQA) and Group-Query Attention (GQA) with a group size of 2.

|                        | Quality | Inference |         | Inference GPU Memory (MiB) |                  |  |
|------------------------|---------|-----------|---------|----------------------------|------------------|--|
| ST Model               | (BLEU)  | Time (s)  | Speedup | Avg. Usage                 | Reduction Factor |  |
| MHA                    | 23.18   | 281.3     | 1.00×   | 18646                      | 1.00             |  |
| MQA                    | 22.70   | 168.1     | 1.67×   | 3074                       | 6.07             |  |
| GQA                    | 22.75   | 190.6     | 1.48×   | 5313                       | 3.51             |  |
| MLA                    | 22.97   | 97.0      | 2.90×   | 5065                       | 3.68             |  |
| MLA w/ SnapKV [24]     | 21.76   | 80.8      | 3.48×   | 4222                       | 4.42             |  |
| Mamba-2 [14]           | 18.62   | 157.5     | 1.78×   | 5676                       | 3.29             |  |
| Proposed MTLA          | 23.28   | 65.6      | 4.29×   | 2835                       | 6.58             |  |
| Proposed MTLA w/ s = 3 | 23.25   | 52.7      | 5.34×   | 2251                       | 8.28             |  |
| Proposed MTLA w/ s = 4 | 23.05   | 48.7      | 5.78×   | 1921                       | 9.71             |  |

This subsection further compares our work with other approaches, including MQA and GQA. First, MLA and our MTLA follow the hyper-parameter settings of [\[26\]](#page-11-2), as discussed in Section [4.3.](#page-5-1) Under this configuration, each token in MLA results in a KV cache size equivalent to that of GQA with 2.25 groups. Therefore, the GPU memory usage for inference is similar between MLA and GQA. Note that the GPU memory usage reported here includes both intermediate activations and the KV cache.

Importantly, MLA achieves faster inference than GQA and also outperforms MQA in speed, demonstrating that storing KV information in low-rank latent vectors and directly using them in attention reduces computation accelerates inference. Moreover, MLA also outperforms GQA in translation quality, which is why this paper focuses comparisons to it.

For our proposed MTLA, with the default temporal compression rate s = 2, its pre-token KV cache elements are equivalent to GQA with 2.25/2 = 1.125 groups. Since MQA corresponds to GQA with 1 group, the KV cache size of MTLA becomes roughly equivalent to that of MQA. This motivates our choice of s = 2 as the default setting. As shown in Table. [5,](#page-8-1) MTLA yields similar memory usage as MQA while delivering 2.56× inference speedup. This is because MTLA inherits the low-rank compression benefits of MLA and further reduces per-token complexity from O(T) to O(T /s), with T as the sequence length. In contrast, MQA and GQA offer limited speedups over MHA and mainly reduce GPU memory usage.

As noted in Sec. [5.3,](#page-6-0) all inference speed tests use the same batch and beam size across models. MTLA is a more advanced KV compression method than MQA (i.e., GQA with 1 group), which cannot reduce group count further, while MTLA allows further compression by increasing s. For example, with s = 4, MTLA significantly outperforms MQA in translation quality (p < 0.05, statistically tested via SacreBLEU [\[37\]](#page-12-13)), while also yielding greater inference speed and GPU memory reduction.

This section further applies SnapKV [\[24\]](#page-11-7), a representative token compression method, to MLA for comparison with MTLA. The results in Table. [5](#page-8-1) show that while MLA with SnapKV improves

inference efficiency compared to MLA, it also leads to some reduction in translation quality. MTLA outperforms MLA with SnapKV in terms of quality, inference time, and GPU memory usage. The speech translation task is a strong test of whether sufficient information can be preserved when compressing tokens or context, and the results demonstrate that MTLA excels in this regard.

As a further point of comparison, this section implements Mamba-2 [\[14\]](#page-10-14) and compares it to MTLA. The results in Table. [5](#page-8-1) show that MTLA outperforms Mamba-2 in inference efficiency on this task and also on translation quality. While linear-complexity models like Mamba-2 will certainly yield more efficient inference than quadratic attention mechanisms when dealing with extremely long sequences, the model performance can also suffer. In summary, MTLA follows the mainstream approach of using quadratic attention mechanisms and therefore benefits from stronger model performance while greatly improving inference time and GPU memory usage.

### 6.4 Extended Results with FlashAttention-2

This section further employs FlashAttention-2 [\[12\]](#page-10-6) to evaluate MTLA under a stronger inference implementation. Since the official FlashAttention-2 does not directly support MTLA, this paper extends it by implementing custom CUDA kernels for MTLA inference[1](#page-9-0) . As shown in Table [6,](#page-9-1) while using FlashAttention-2 certainly accelerates inference, it does not change the conclusion, as MTLA still achieves a 3.99× speedup in inference and reduces average GPU memory consumption by a factor of 7.34 compared to MHA.

<span id="page-9-1"></span>Table 6: BLEU (↑) results on the MuST-C En-De tst-COMMON set, with or without FlashAttention-2.

|                              | Quality | Inference |         | Inference GPU Memory (MiB) |                  |  |
|------------------------------|---------|-----------|---------|----------------------------|------------------|--|
| ST Model                     | (BLEU)  | Time (s)  | Speedup | Avg. Usage                 | Reduction Factor |  |
| MHA                          | 23.18   | 281.3     | 1.00×   | 18646                      | 1.00             |  |
| w/ FlashAttention-2          | 23.16   | 145.7     | 1.93×   | 9244                       | 2.02             |  |
| Proposed MTLA                | 23.28   | 65.6      | 4.29×   | 2835                       | 6.58             |  |
| w/ extended FlashAttention-2 | 23.29   | 36.5      | 7.71×   | 1259                       | 14.81            |  |

## 7 Conclusions

This paper proposes MTLA, the first self-attention mechanism capable of compressing the temporal dimension of the KV cache. Building upon the low-rank KV compression of MLA, MTLA employs a hyper-network to dynamically merge adjacent KV caches, enabling effective temporal compression. A stride-aware causal mask is proposed to ensure that MTLA maintains efficient parallel training while matching the attention behaviour during incremental inference, addressing the mismatch between the compressed KV cache length and the processed sequence length. Experiments across ST, text summarisation, ASR, and SLU show that MTLA greatly accelerates inference and reduces GPU memory usage at inference without sacrificing accuracy. With a temporal compression rate of 2, MTLA already matches the KV cache compression level of MQA while delivering better accuracy and speed, and it supports further compression, establishing itself as a more advanced KV cache compression method. Further comparisons show that MTLA consistently outperforms MLA with SnapKV and the linear Mamba-2 model in both quality and efficiency. Even with FlashAttention-2 acceleration, MTLA maintains up to 3.99× faster inference and 7.34× lower memory usage than MHA, confirming its effectiveness as a general and advanced KV cache compression approach. Future work will explore applying MTLA to LLMs, where the KV cache size is a key bottleneck during inference. MTLA temporal compression of the KV cache offers a promising way to scale LLMs to longer contexts while balancing memory use, latency, and accuracy.

