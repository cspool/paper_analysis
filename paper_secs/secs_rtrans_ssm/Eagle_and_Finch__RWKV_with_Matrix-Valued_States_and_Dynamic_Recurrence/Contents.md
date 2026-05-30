# **Contents**

| 1 | Introduction                                                                                                                                                                                        | 3                               |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| 2 | Background                                                                                                                                                                                          | 4                               |
| 3 | Eagle/Finch Architecture                                                                                                                                                                            | 5                               |
| 4 | Method                                                                                                                                                                                              | 6                               |
|   | 4.1<br>Eagle<br><br>4.1.1<br>Eagle Token Shift<br><br>4.1.2<br>Eagle Time Mixing<br><br>4.1.3<br>Channel Mixing<br>4.2<br>Finch<br><br>4.2.1<br>Finch Token Shift<br>4.2.2<br>Finch Time Mixing<br> | 6<br>6<br>7<br>7<br>7<br>7<br>8 |
| 5 | RWKV World Tokenizer                                                                                                                                                                                | 8                               |
| 6 | RWKV World v2 Dataset                                                                                                                                                                               | 9                               |
| 7 | Pre-Trained Models                                                                                                                                                                                  | 9                               |
| 8 | Language Modeling Experiments                                                                                                                                                                       | 9                               |
|   | 8.1<br>LM Evaluation Harness Benchmarks<br>8.2<br>Associative Recall<br>8.3<br>Long Context Experiments<br><br>8.4<br>Bamboo Benchmark<br>                                                          | 9<br>11<br>12<br>12             |
| 9 | Speed and Memory Benchmarks                                                                                                                                                                         | 14                              |
|   | 10 Multimodal Experiments                                                                                                                                                                           | 15                              |
|   | 10.1 RWKV Music Modelling<br><br>10.2 VisualRWKV<br>                                                                                                                                                | 15<br>15                        |
|   | 11 RWKV on Audio                                                                                                                                                                                    | 16                              |
|   | 12 Conclusions                                                                                                                                                                                      | 17                              |
| A | Author Contributions                                                                                                                                                                                | 28                              |
| B | Additional Architecture Details                                                                                                                                                                     | 29                              |
| C | Additional Related Work                                                                                                                                                                             | 32                              |
|   | D Training Dataset Details                                                                                                                                                                          | 33                              |
| E | Computing Costs                                                                                                                                                                                     | 33                              |
| F | New Tokenizer Details                                                                                                                                                                               | 35                              |
|   | F.1<br>Designation<br><br>F.2<br>Efficiency Experiments<br>F.3<br>Speed                                                                                                                             | 35<br>35<br>36                  |
| G | Additional Evaluations                                                                                                                                                                              | 36                              |
|   | G.1 Alignment Benchmark<br>G.2<br>MTBench<br>G.3<br>Self-Learning<br><br>G.4<br>Zero-shot evaluation on additional NLP tasks<br>                                                                    | 36<br>37<br>37<br>38            |

| Н | Hyperparameters                        | 38 |
|---|----------------------------------------|----|
| I | Parameter Initializations              | 38 |
| J | <b>Architectural Ablations</b>         | 40 |
| K | DDLerp Ablations                       | 41 |
| L | Non-English Chat Examples              | 41 |
| M | Chat Examples - Comparison with RWKV-4 | 43 |

#### <span id="page-2-0"></span>1 Introduction

Advancements in Large Language Models (LLMs) have significantly impacted Natural Language Processing (NLP) tasks. The field has traditionally been dominated by the transformer architecture (Vaswani et al., 2023). However, the expressive attention mechanism of transformers leads them to suffer from quadratic time complexity with respect to input sequence length. Various methods have been proposed to achieve sub-quadratic time complexity without significantly changing the core attention mechanism, typically relying on some form of sparsity techniques (Child et al., 2019a; Beltagy et al., 2020; Zaheer et al., 2020).

Recent works have achieved sub-quadratic time complexity without significantly sacrificing performance by introducing new mechanisms to replace attention at the core of the Transformer architecture. These models include gated recurrences (Fu et al., 2023; Gu & Dao, 2023; Gu et al., 2021; Sun et al., 2023; Katsch, 2023; Qin et al., 2023; Smith et al., 2023), gated convolutions (Poli et al., 2023; Peng et al., 2023), data-dependent linear attention (Yang et al., 2023; Katharopoulos et al., 2020b), sparse attentions (Tay et al., 2020; Child et al., 2019b; Zaheer et al., 2020; Qiu et al., 2019) and their combinations (De et al., 2024; Qin et al., 2024; 2022). We build off RWKV-4 introduced in Peng et al. (2023), which provides efficient inference and training along with a parallelizable implementation compared to competing architectures as shown in Table 1.

<span id="page-2-1"></span>

| Architecture       | Inference |          |              | Training      |          |  |
|--------------------|-----------|----------|--------------|---------------|----------|--|
|                    | Time      | Memory   | Parallel     | Time          | Memory   |  |
| LSTM/LMU           | O(1)      | O(1)     | Х            | O(N)          | O(N)     |  |
| Transformer        | O(N)      | $O(N)^a$ | $\checkmark$ | $O(N^2)$      | $O(N)^b$ |  |
| Linear Transformer | O(1)      | O(1)     | $\checkmark$ | O(N)          | O(N)     |  |
| H3/S4              | O(1)      | O(1)     | $\checkmark$ | $O(N \log N)$ | O(N)     |  |
| Hyena              | O(N)      | O(N)     | $\checkmark$ | $O(N \log N)$ | O(N)     |  |
| RWKV/Mamba/RetNet  | O(1)      | O(1)     | $\checkmark$ | O(N)          | O(N)     |  |

Table 1: Comparative analysis of RWKV-4/5/6 and other LLM architectures regarding time and memory complexity for both inference per token and training per sequence, and training parallelizability across the sequence dimension. The context/sequence length is denoted by N.

<sup>a</sup>O(1) without KV cache <sup>b</sup> With Flash Attention

In this paper, we introduce two new architectures: **Eagle** (RWKV-5) and **Finch** (RWKV-6). First, Eagle improves upon the architecture and learned decay schedule from RWKV-4 (Peng et al., 2023) through the use of expressive multi-headed matrix-valued states (as opposed to vector-valued states), a reformulated receptance, and an additional gating mechanism. Finch further improves the expressivity and flexibility of the architecture by introducing new data-dependent functions for both the time-mixing and token-shift modules, consisting of parameterized linear interpolations. Additionally, Finch proposes a novel use of the Low Rank Adaptation (Hu et al., 2022) function to allow for trainable weight matrices to efficiently augment the learned data decay vectors in a context-dependent manner. Finally, we introduce a new tokenizer, the RWKV World Tokenizer, and a new dataset, RWKV World v2 (1.12 trillion tokens), specially designed to improve performance on multilingual and code data.

Through extensive experimentation, we show that the Eagle and Finch models perform competitively, or improve upon existing models under a wide variety of sequence modeling domains and

tasks. Specifically, we evaluate our trained models on commonly used English-only and multilingual text benchmarks, associative recall, music modeling, and vision-language benchmarks. Our experiments demonstrate that the advancements in Eagle and Finch provide significant progress towards developing more efficient AI models

In summary, our main contributions are:

- The Eagle (RWKV-5) and Finch (RWKV-6) RWKV architectures, which significantly improve over RWKV-4 on benchmarks for LLMs.
- The RWKV World Tokenizer which contains underrepresented languages' vocabulary and which performs fast tokenization with Trie-based greedy matching.
- The RWKV World v2 public dataset, comprised of 1.12 trillion tokens of publicly available multilingual data.
- Public release of four pre-trained Eagle models, scaling from 0.46 to 7.5 billion parameters, and two Finch models, with 1.6 and 3.1 billion parameters. Demonstrating that these novel architectures are competitive to transformers when trained using enough FLOPs to make meaningful scaling conclusions.
- A completely open training pipeline to enable interpretability and reproducibility of alternative-architecture LLMs (See Table 2).

<span id="page-3-1"></span>

| Model           | Context           | Training             | Open    | Open      |          | Open    |
|-----------------|-------------------|----------------------|---------|-----------|----------|---------|
|                 | Length            | Tokens               | Weights | Inference | Training | Dataset |
| GPT-4           | 128k <sup>a</sup> | Undisclosed          | 0       | 0         | 0        | 0       |
| LLaMA2 7B       | 4k                | $2.0 \times 10^{12}$ | •       | •         | $\circ$  | 0       |
| Mistral 7B v0.1 | $32k^b$           | Undisclosed          | •       | •         | $\circ$  | 0       |
| Gemma 7B        | 8k                | $6.0 \times 10^{12}$ | •       | •         | •        | 0       |
| StableLM 7B v2  | 4k                | $1.1 \times 10^{12}$ | •       | •         | •        | •       |
| Pythia 6.9B     | 2k                | $3.3 \times 10^{11}$ | •       | •         | •        | •       |
| Eagle 7B        | Indefinite $^c$   | $1.1\times10^{12}$   | •       | •         | •        | •       |

Table 2: Comparison of the openness and accessibility of public foundational LLMs with 7B+ parameters regarding model weights, official inference/training code, and dataset. Widely available but not under an open source license is indicated by  $\P$ .

#### <span id="page-3-0"></span>2 Background

Eagle and Finch are RNNs based on a multi-headed hybridization of the RWKV-4 architecture and linear attention. We discuss related work and the evolution of these two architectures below, with a more detailed review given in Appendix C.

Recurrent Neural Networks (RNNs) are well suited to provide inexpensive inference on sequence modelling tasks, typically operating in O(1) time complexity per step with respect to sequence length. They model sequences with time dependencies by generating a hidden state  $h_t$  at each time step, which is fed back in at the next time step as a secondary input. Classic RNNs (e.g. LSTM (Hochreiter & Schmidhuber, 1997) and GRU (Cho et al., 2014)) became widely used for sequence modelling, but are difficult to parallelize across the time dimension for training.

The Transformer architecture has enjoyed remarkable success in generative sequence modelling, and language modelling in particular (Vaswani et al., 2023; Radford et al., 2018), providing SOTA performance across many tasks. However, the use of multi-headed dot-product self-attention (MHA) leads to a quadratic time complexity with respect to sequence length. The deficiencies of classic RNNs and Transformers led to many attempts to develop architectures incorporating the best features of both in a single model, namely O(1) per token time complexity and fast highly parallelizable training.

Linear Attention (Schmidhuber, 1992; Katharopoulos et al., 2020a) replaces the numerator of MHA's softmax( $QK^T$ )V with  $\phi(Q)\phi(K)^TV$ , allowing a reordering of operations via associativity to

<sup>&</sup>lt;sup>a</sup>OpenAI's gpt-4-0125-preview model <sup>b</sup>With sliding window attention <sup>c</sup>Pretrained with context length 4096, but no fundamental context length limitation or relationship to speed, see 8.3 for extrapolation details

 $\phi(Q)(\phi(K)^TV)$ , where  $\phi$  represents a non-negative feature-map function. It can be computed as an RNN in O(1) time per step by adding  $\phi(K_i^T)V_i$  to a recurrent state at each time step i, or trained in parallel much like MHA. This accomplishes the main goals outlined above, but naive linear attention suffers from significantly reduced performance compared to MHA-based transformers.

A modified form of linear attention, the Attention Free Transformer (AFT) (Zhai et al., 2021), paved the way for the RWKV architecture, by using a number of attention heads equal to the size of the feature dimension and incorporating a set of learned pairwise positional biases, denoted as w.

$$AFTAttn_t = \sigma_q(q_t) \odot \frac{\sum_{i=1}^t \exp(k_i + w_{i,t}) \odot v_i}{\sum_{i=1}^t \exp(k_i + w_{i,t})}$$
(1)

RWKV-4 reformulates the AFT equation by replacing the pair-wise positional biases with a channel-wise vector of additive weight decay rates w. It also adds a bonus term u to offset the weight of only the current input specially.

$$wkv_{t} = \frac{\sum_{i=1}^{t-1} \exp(-(t-1-i)w + k_{i}) \odot v_{i} + \exp(u + k_{t}) \odot v_{t}}{\sum_{i=1}^{t-1} \exp(-(t-1-i)w + k_{i}) + \exp(u + k_{t})}.$$
 (2)

RWKV-4 also adds token-shift and gating to both attention and feed-forward sub-blocks of transformer, and small embedding initialization and normalization to quickly arrive at well-distributed token embeddings. Combining all of these architectural changes led RWKV-4 to become the first RNN to rival the performance of Transformers, while maintaining fast parallelizable training and O(1) time complexity per token.

There has been a recent revival of RNNs in NLP research (Tiezzi et al., 2024). HGRN(Qin et al., 2023) is a recent time-parallelizable data-dependent RNN that employs input and forget gates. TransNormer(Qin et al., 2022) applies RMSNorm to linear attention to bound its output. Other new time-parallelizable data-dependent RNNs have also been invented concurrently with our work including GLA (Yang et al., 2023) and Griffin (De et al., 2024).

State Space Models (SSMs) employ a hidden state of basis function weights to model an approximation of the input function (Gu et al., 2020), updating that hidden state via a differential equation. Earlier SSMs (Gu et al., 2022) were historically computed using long convolutions in  $O(N\log N)$  time per sequence, but could also be formulated as a recurrent network. Recently, it has been shown that SSMs can be parallelized across the time dimension via techniques including associative scan (Smith et al., 2023). A new class of SSMs has also emerged concurrently with our work (Katsch, 2023; Gu & Dao, 2023) that feature data-dependent A and B terms, which function similarly to the data-dependent dynamic recurrence used in Finch.

