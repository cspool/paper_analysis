# Flexibly Scaling Large Language Models' Contexts Through Extensible Tokenization

Ninglu Shao1,2<sup>∗</sup> Shitao Xiao<sup>1</sup> Zheng Liu1† Peitian Zhang1,<sup>2</sup> 1: Beijing Academy of Artificial Intelligence, 2: Gaoling School of Artificial Intelligence, Renmin University

## Abstract

Large language models (LLMs) are in need of sufficient contexts to handle many critical applications, such as retrieval-augmented generation and few-shot learning. However, due to the constrained window size, the LLMs can only access to the information within a limited context. Although the size of context window can be extended by fine-tuning, it will result in a substantial cost in both training and inference stage. In this paper, we present Extensible Tokenization as an alternative method which realizes the flexible scaling of LLMs' context. Extensible Tokenization stands as a midware in between of the tokenized context and the LLM, which transforms the raw token embeddings into the extensible embeddings. Such embeddings provide a more compact representation for the long context, on top of which the LLM is able to perceive more information with the same context window. Extensible Tokenization is also featured by its flexibility: the scaling factor can be flexibly determined within a feasible scope, leading to the extension of an arbitrary context length at the inference time. Besides, Extensible Tokenization is introduced as a drop-in component, which can be seamlessly plugged into not only the LLM itself and but also its fine-tuned derivatives, bringing in the extended contextual information while fully preserving the LLM's existing capabilities. We perform comprehensive experiments on long-context language modeling and understanding tasks, which verify Extensible Tokenization as an effective, efficient, flexible, and compatible method to extend LLM's context. Our model and source code will be made publicly available at https://github.com/FlagOpen/FlagEmbedding.

# 1 Introduction

Large language models (LLMs) need to process long-sequence data in order to accomplish many critical tasks, like retrieval augmented generation and in-context learning. Unfortunately, the existing LLMs are constrained by the sizes of their context windows, which are far from enough to fully cover the input data in the corresponding scenarios. Although the LLM's context window can be extended by fine-tuning [\[6;](#page-9-0) [11;](#page-9-1) [25\]](#page-10-0), it will lead to a considerable cost at both training and inference time. Besides, the continual fine-tuning over long-sequence data is likely to impair the LLM's original capability on shorter contexts, which is unfavorable to its practical usage. Alternatively, the LLMs can be also modified by many other techniques to establish longer context windows, such as sparse attention [\[9;](#page-9-2) [3;](#page-9-3) [36;](#page-11-0) [15\]](#page-9-4), stream processing [\[33;](#page-10-1) [18\]](#page-10-2), compressed memory [\[4;](#page-9-5) [8;](#page-9-6) [24;](#page-10-3) [27;](#page-10-4) [20\]](#page-10-5), etc. However, the existing methods are still limited by manifold problems. For example, the sparse attention calls for customized GPU kernels which are not supported by the standard infrastructures; the stream processing ignores the information beyond the context window instead of making effective

<sup>∗</sup>Ninglu Shao (ninglu\_shao@ruc.edu.cn), Shitao Xiao, and Zheng Liu are the co-first authors.

<sup>†</sup>Zheng Liu (zhengliu1026@gmail.com) is the correponding author.

> **[图片提取文字 (无描述)]:**
> --- Position Interpolation ...... LongLLaMA -+ Extensible Tokenization -x- NTK-Aware OOM OOM OOM OOM 60 4.0 100 3.5 50 80 (GB) (s) 3.0 Perplexity Memory 30 1.5 1.0 GPU +----20 20 0.5 0.0 10 4K 8K 100K 8K 16K 32K 100K 4K 8K 16K 32K 100K 4K 16K 32K Context Length Context Length Context Length
![](_page_1_Figure_0.jpeg)

Figure 1: Comparison between Extensible Tokenization and other context extension methods, including 1) Position Interpolation [5], 2) NTK-Aware Scaled RoPE [1], 3) LongLLaMA [31]. Extensible Tokenization presents a superior long-context language modeling capability, along with better efficiency in terms of memory and time. PPL is measured on PG19 [26] following the method in [7]

use of it; the compressed memory are still limited by the substantial loss of contextual information and the incompatibility with the existing LLMs.

In fact, the LLM's context capacity, i.e. the contextual information that the LLM is able to perceive, is jointly determined by two critical factors. One is the *size of context window*, which regulates the amount of input units the LLM can intake. The other one is the *information density*, which reflects the information encoded by each input unit. Currently, the typical input unit to LLM is token embedding, which is relatively information sparse because it only encodes the information about each individual token. Once with more compact and informative input units, the LLM will be able to perceive more information with the same context window.

Based on this fundamental principle, we propose **Extensible Tokenization** as a new method to extend the LLM's context. Extensible Tokenization stands as a midware in between of the input context and the LLM, which transforms raw token embeddings into *extensible embeddings* for the compact representation of the input context. Extensible Tokenization shares a common high-level spirit as the previous methods on context compression [7; 24; 8; 12]. However, it enjoys a substantially improved performance on context extension, a high flexibility of usage, and a high compatibility with the downstream LLM thanks to a series of characters about its architecture and training.

Instead of modifying the original model architecture, we employ another pre-trained language model as the stand-alone module for Extensible Tokenization. It transforms the raw input into its output embeddings, and then performs down-scaling by the scaling factor k (e.g., k=32) for the extensible embeddings. Given the dramatic scale of compression, the learning process must be properly designed such that the contextual information can be fully preserved. In our work, we propose to learn the extensible embeddings through auto-regression (AR), where the compressed context can well assist the downstream LLM for the prediction of the next tokens. Besides, we further optimize the sample efficiency of training through the two-stream processing. With merely two passes of feed-forward, it will let the training loss to be derived from every single token of one training sample. By minimizing the comprehensive prediction losses from the data, the extensible embeddings can be effectively learned as compact but equally informative representations of the context.

Extensible Tokenization presents a highly flexible solution to extend the LLM's context. In particular, the scaling factor k can be freely specified at the inference time, leading to the flexible extension for diverse context lengths. Besides, the extensible embeddings and the raw token embeddings can be seamlessly blended with each other, with the extensible embeddings applied for the verbose part of the input (e.g., the retrieved documents) and the raw token embeddings used for the concise part of the input (e.g., questions or instructions). The collaborative play of both types of embeddings contributes to a superior quality for the LLM's utilization of the extended context.

The extensible tokenizer is learned with the downstream LLM's parameters fixed all the time. Therefore, it can work as a compatible plug-and-play component, introducing the extended context as extensible embeddings without compromising the LLM's original performance with the raw token embeddings. Notably, beyond the LLM where the training is conducted, it is empirically observed that the extensible tokenizer also exhibits a strong compatibility with various fine-tuned derivatives

> **[图片提取文字 (无描述)]:**
> unalienable Rights that among these are life, liberty and the pursuit of happiness chunk 1 Downstream LLM 🦀 chunk 2 chunk 3 [ET1] [ET2] [ET3] [ET5] [ET6] unalienable Rights that among these are life, liberty and the pursuit of Down Scaling (\*k) Down Scaling (\*k) Down Scaling (\*k) Extensible Tokenizer Extensible Tokenizer Extensible Tokenizer endowed by their Creator with certain We hold these truths to be self-evident, that all men are created equal, that they are chunk 1 chunk 2 chunk 3
![](_page_2_Figure_0.jpeg)

<span id="page-2-0"></span>Figure 2: Extensible Tokenization. The input data is chunked into equal-sized sub-sequences. Each sub-sequence is transformed and compressed as the extensible embeddings. The new tokens are predicted based on the extensible embeddings from the preceding chunks and the token embeddings in the same chunk. The extensible tokenizer is learned with a fixed downstream LLM.

of the LLM. Such a property indicates the potential of Extensible Tokenization as a versatile module for the extension of the context across a family of closely related LLMs.

With Extensible Tokenization, the inference can be streamingly conducted by sessions. In each session, a new token is predicted based on the extensible embeddings from previous sessions and the preceding raw tokens in the same session. The stream processing leads to a linear time complexity and a substantial reduction of memory usage, which benefits the efficiency of processing long-sequence data. For many real-world scenarios, e.g., retrieval-augmented generation, the extensible embeddings can be pre-computed and cached, which will further save the computation cost of online inference.

We apply the first 8 layers of LLaMA-2-7B (chat) model [\[30\]](#page-10-8) as the extensible tokenizer's backbone, where it is used for the context extension for another downstream LLaMA-2-7B (chat) model. The training is performed on top of the sampled data from RedPajama [\[10\]](#page-9-11) and LongAlpaca [\[6\]](#page-9-0). So far, Extensible Tokenization is trained with a maximum scaling factor of 32, enabling the context length of LLaMA-2-7B to be extended over 100K. In our experiment, Extensible Tokenization achieves superior performances on both long-context language modeling and understanding tasks. Besides, the context extension capability can be well transferred to the fine-tuned derivatives of LLaMA-2-7B, like LongChat [\[11\]](#page-9-1) and LongAlpaca [\[6\]](#page-9-0). Therefore, the Extensible Tokenization is verified as an effective, efficient, flexible, and compatible method for the extension of LLM's context.

# 2 Extensible Tokenization

#### 2.1 Framework

The workflow of Extensible Tokenization is shown as Figure [2.](#page-2-0) For each long-sequence input X, we perform the following three steps which enables the long context to be utilized by the LLM. Firstly, the input X is chunked into sub-sequences: {X1, ...X<sup>N</sup> }. The sequence length of each chunk L<sup>i</sup> is set to be the maximum window size of the extensible tokenizer, e.g., L<sup>i</sup> = 4096 with LLaMA-2, which will best preserve the coherence of the chunking result. Secondly, the sub-sequence of each chunk is transformed by the extensible tokenizer into the output embeddings. The output embeddings are down-scaled by the scaling factor k (e.g., k = 16 or 32), where L/k extensible embeddings (denoted as ET) are generated as the condensed representation of the raw input. Finally, the new tokens are predicted conditioned on the extensible embeddings from the preceding chunks and the raw token embeddings within the recent context.

> **[图片提取文字 (无描述)]:**
> Inference pass for super tokens Decoding pass for x<sub>4-6</sub> and x<sub>7-9</sub> Decoding pass for x<sub>10-12</sub> and x<sub>13-15</sub>  $\{et_1: x_{1-3}, et_2: x_{4-6}, et_3: x_{7-9}, et_4: x_{10-12}\}$ et<sub>1</sub> et, Extensible Token Embeddings {et<sub>1</sub> ··· et<sub>4</sub>} } et<sub>2</sub> et2 et<sub>3</sub> et<sub>3</sub> et<sub>4</sub> eta X4 X10 Extensible Tokenizer X11 X5 x6 X12 X7 X13 X8 X14 Raw Token Embeddings {x1 ··· x12} Xq X<sub>15</sub>
![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Figure 3: Two-Stream AR. In the first pass, the raw token embeddings are transformed into extensible embeddings (with the scaling factor k = 3). In the second pass (given a window size of 10), the auto-regression is accomplished in two steps, with the x1−<sup>3</sup> and x4−<sup>6</sup> predicted in the first step, and x7−<sup>9</sup> and x10−<sup>12</sup> predicted in the second step.

#### 2.2 Extensible Embedding

As introduced, the raw token embedding, which is merely corresponding to one individual token, is information sparse. In contrast, the extensible embedding is presented to serve as a highly compact but equally informative representation of the context. For this purpose, we employ another language model as the extensible tokenizer (denoted as LMet), which transforms the raw input of each subsequence X<sup>i</sup> : {xi,1, ...xi,L} into the sequence of output embeddings O<sup>i</sup> :

$$O_i: \{o_{i,1}, ..., o_{i,L}\} \leftarrow LM_{et}(x_{i,1}, ..., x_{i,L}; \theta_{et}).$$
 (1)

On top of an expressive language model, the rich contextual information within Xi,:<sup>l</sup> can be encoded by the corresponding output embedding oi,l. The output embeddings are further down-scaled by the scaling factor k, where m (m = L/k) extensible embeddings (eti,<sup>∗</sup>) are generated for X<sup>i</sup> :

$$\{et_{i,1}, \dots, et_{i,m}\} \leftarrow \text{DownScale}(\{o_{i,1}, \dots o_{i,L}\}). \tag{2}$$

There can be many alternative ways to realize the functionality of down-scaling, e.g., with any pooling functions along the sequence dimension. In our work, we simply down-scale the output embeddings by through strided sampling, where the last embedding in every k steps is chosen, i.e., eti,j ← oi,k×<sup>j</sup> . Despite simplicity, such a realization is empirically effective and leads to a high flexibility of usage.

#### 2.3 Two-Stream AR

Extensible Tokenization can be learned by auto-regression (AR), where the loss is minimized for the prediction of next tokens conditioned on the extensible embeddings from the preceding context. Although the auto-regression tasks can be simply performed by having the long context transformed into extensible embeddings and predicting the last few tokens within a training instance (e.g., predicting the answer to a question based on the extensible embeddings of a long document), it will severely restrict the training effect because the long context accounts for the majority of computation cost whereas no prediction loss can be derived from it.

In our work, we propose the two-stream AR to optimize the sample efficiency of training (Figure [3\)](#page-3-0). In the first pass of inference, the extensible embeddings are generated for the entire context. For example, with a chunk size of 3 and an scaling factor of 3, the input data X = {x1, ...x15} is transformed into the extensible embeddings {et1,1, et2,1, et3,1, et4,1} (the last chunk is exempted). In the second past, each single token within the long context is streamingly predicted by chunks. Particularly, the prediction is made conditioned on the extensible embeddings from the previous chunks and the preceding raw token embeddings within the same chunk. Formally,

$$\min_{\theta_{et}} \sum_{X} \sum_{i>1} \log P(x_{i,j}|et_{1,1}, ...et_{i-1,k}, x_{i,1}, ...x_{i,j-1}|\theta, \theta_{et}).$$
 (3)

For example, x<sup>6</sup> is predicted based on st<sup>1</sup> (representing x1−3) and x4. Note that the chunk size of training is made much smaller than the LLM's window size (e.g., 512), where the prediction of new tokens can mostly rely on the contextual information offered by the extensible embeddings. Thanks to the above processing, the prediction loss can be comprehensively derived from the each training

instance, enabling the extensible tokenizer to be effectively learned from general long-context data, such as RedPajama [\[10\]](#page-9-11) and LongAlpaca [\[6\]](#page-9-0). We also randomly sample the extension ratio k from a candidate scope (e.g., [2, 4, 8, 16, 32]) for each training instance, which helps the model to generalize for the extension of diverse context lengths.

When the extensible tokenizer is learned, the downstream LLM's parameters (θ) are always fixed. Consequently, the extensible tokenizer can work as a compatible drop-in component to the downstream LLM, bringing in new information from the extended context without affecting the LLM's performance with the raw token embeddings. Besides, we also empirically find that the extensible tokenizer can maintain a high compatibility with many full-parameter fine-tuned derivatives of the downstream LLM. Such a property makes it a general module to extend the context length for a family of closely related LLMs.

# 3 Inference

The inference process with the Extensible Tokenization can be divided into the online and offline scenario. Particularly, the online usage deals with the scenario where the long-sequence data is streamingly presented (e.g., conversation). In this scenario, the generation process is conducted in consecutive sessions. In each session, the downstream LLM predicts the new tokens based on the extensible embeddings from the previous sessions and the raw token embeddings within the current session. The current session comes to its end when the total sum of both types of embeddings (denoted as Net and Nraw, respectively) reaches the maximum window size of the downstream LLM (L ∗ ): Net + Nraw = L ∗ . The newly generated tokens will be condensed as the extensible embeddings at the end of the current session (Net ← Net + Nraw/k), where the new session can be conditioned on the augmented extensible embeddings. The session-based workflow is free from processing the long context directly, which will preserve a small memory footprint. Besides, the inference time will become linear to the context length, which will benefit the processing of long-sequence data.

As for the offline scenario, the long-sequence data is fully presented in advance (e.g., the documents for RAG and reading-compression). As a result, the extensible embeddings can be pre-computed for the entire data, which will lead to a more competitive efficiency at the inference time. In fact, it is OK to simply save the entire output embeddings in the offline stage, and flexibly sample for the extensible embeddings at the inference time based on the concrete scaling factor.

## 4 Experiments

In this section, we conduct experimental studies to investigate the following key issues about Extensible Tokenization. 1) The effectiveness of context extension. 2) The flexibility and compatibility. 3) The running efficiency. 4) The influential factors about the empirical performance.

#### 4.1 Experimental Settings

In our experiment, we make use of the LLaMA-2-7B (chat) model [\[29\]](#page-10-9) as our downstream LLM. By default, we take the first 8 layers of LLaMA-2-7B (chat) as the initialized backbone for the extensible tokenizer. The training process takes place on a single Nvidia 8×A800 GPU machine, with a batch size of 8 and a learning rate of 5e <sup>−</sup><sup>5</sup> using the linear scheduler. The training is consecutively performed with 90K sampled instances from Redpajama [\[10\]](#page-9-11) and 10K training instances collected by LongAlpaca [\[6\]](#page-9-0). As introduced, the extensible tokenizer is trained while the downstream LLM's parameters are always fixed.

We consider the following baselines in our experiment. 1) The context extension method without fine-tuning, including Positional Interpolation (PI) [\[5\]](#page-9-7), the NTK-Aware Scaled RoPE (NTK) [\[1\]](#page-9-8), and StreamingLLM (Stream) [\[34\]](#page-10-10). 2) The finetuned full-attention method, including LongAlpaca-7B-16K [\[6\]](#page-9-0) and LongChat-7B-32K [\[11\]](#page-9-1). 3) The finetuned method with extra architectures to handle long contexts, including AutoCompressor-7B-6K [\[26\]](#page-10-7) and LongLlama [\[31\]](#page-10-6). All of the baselines are based on LLaMA-2-7B, except LongLLaMA which leverages CodeLLaMA [\[28\]](#page-10-11).

| Model             | PG19 |      |       |       | Books3 |      |      |      |      |      |
|-------------------|------|------|-------|-------|--------|------|------|------|------|------|
|                   | 4K   | 8K   | 16K   | 32K   | 100K   | 4K   | 8K   | 16K  | 32K  | 100K |
| LLaMA-2-7B        | 7.77 | >103 | >103  | >103  | OOM    | 4.21 | >103 | >103 | >103 | OOM  |
| PI                | 7.77 | 8.68 | 18.65 | >102  | OOM    | 4.21 | 5.99 | 11.4 | 69.8 | OOM  |
| NTK               | 7.77 | 8.13 | 10.71 | 55.22 | OOM    | 4.21 | 5.10 | 7.71 | 52.3 | OOM  |
| Streaming         | 7.98 | 8.01 | 8.00  | 8.00  | 8.00   | 4.32 | 4.34 | 4.33 | 4.33 | 4.34 |
| LongAlpaca-16K    | 8.45 | 8.15 | 8.12  | >103  | OOM    | 4.93 | 4.67 | 4.64 | >103 | OOM  |
| LongChat-32K      | 7.59 | 7.25 | 7.00  | 6.85  | OOM    | 4.12 | 3.95 | 3.87 | 3.85 | OOM  |
| AutoCompressor-6K | 26.9 | >103 | 103   | >104  | OOM    | 17.1 | >103 | >103 | >104 | OOM  |
| LongLLaMA-32K     | 7.12 | 6.95 | 6.78  | OOM   | OOM    | 3.99 | 3.90 | 3.84 | OOM  | OOM  |
| ExtenToken (×16)  | 7.75 | 7.48 | 7.38  | 7.31  | >102   | 4.32 | 4.20 | 4.15 | 4.13 | >103 |
| ExtenToken (×32)  | 8.61 | 8.15 | 7.87  | 7.69  | 7.54   | 4.67 | 4.48 | 4.36 | 4.28 | 4.25 |

<span id="page-5-0"></span>Table 1: Language modeling performance (measured by perplexity) on PG19 and Books3.

> **[图片提取文字 (无描述)]:**
> LongAlpaca LongAlpaca w/ Extensible Tokenization (16x) ··· LongAlpaca w/ Extensible Tokenization (32x) LongChat w/ Extensible Tokenization (16x) -- LongChat w/ Extensible Tokenization (32x) LongChat 100 80 Perplexity 60 40 20 4K 8K 16K 32K 100K 200K 500K 1M Context Length
![](_page_5_Figure_2.jpeg)

<span id="page-5-1"></span>Figure 4: The extensible tokenizer trained on LLaMA-2-7B can be directly utilized by LongAlpaca-16K and LongChat-32K, leading to further scaling of their context lengths by ×16 and ×32 (with PPL measured on PG19). Remarkable, the context length of LongChat can be extended to 1 million.

#### 4.2 Main Results

#### 4.2.1 Long-Context Language Modeling

The long-context language modeling is evaluated with PG19 [\[26\]](#page-10-7) and Books3 [\[16\]](#page-9-12) dataset. Following the method used by Alexis et al. [\[7\]](#page-9-9), the PPL is measured by predicting the last 512 tokens based on the preceding context. There are two alternative settings about the Extensible Tokenization, denoted as ExtenToken (×16) and ExtenToken (×32), where the scaling factor is set as 16 and 32, respectively. The evaluation results are reported in Table [1,](#page-5-0) where the following observations can be derived.

Firstly, with the extended context length, Extensible Tokenization leads to a notable advantage over LLaMA-2-7B. Besides, its relative improvement is more significant than the fine-tuning free methods, while being comparable to the finetuned baselines. Such an observation indicates that the extended contextual information introduced by Extensible Tokenization can be effectively utilized by the LLM. Secondly, by switching to larger scaling factors, Extensible Tokenization is able to flexibly support longer expansions of the context length than other baselines. Particularly, by switching the scaling factor from 16 to 32, LLaMA-2-7B's context length can be continually extended up to 128K (32 × 4K). The extended contextual information further improves the performance of language modeling. Thirdly, the well-trained extensible tokenizer for LLaMA-2-7B preserves a high compatibility to its fine-tuned derivatives, as it can be directly applied for the context extension of LongAlpaca and LongChat (Figure [4\)](#page-5-1). In particular, the context lengths of the two models can also be effectively extended on top of Extensible Tokenization, with LongChat-32K's context length going beyond one million tokens remarkably (32 × 32K).

| Model                         | Length | Single-Doc QA | Multi-Doc QA | Summarization |
|-------------------------------|--------|---------------|--------------|---------------|
| Llama-2-7B                    | 4k     | 24.90         | 22.60        | 24.70         |
| Llama-2-7B w. PI              | 16k    | 18.98         | 17.16        | 25.03         |
| Llama-2-7B w. NTK             | 16k    | 23.21         | 23.34        | 24.40         |
| Llama-2-7B w. Stream          | 16k    | 21.47         | 22.22        | 22.20         |
| Llama-2-7B w. ExtenToken*     | 4k     | 25.56         | 26.92        | 24.63         |
| LongAlpaca-16K (4k)           | 4k     | 26.81         | 24.44        | 26.93         |
| LongAlpaca-16K (8k)           | 8k     | 28.61         | 24.83        | 27.91         |
| LongAlpaca-16K (16k)          | 16k    | 28.36         | 28.16        | 27.77         |
| LongAlpaca-16K w. ExtenToken* | 4k     | 28.61         | 28.32        | 26.88         |
| LongChat-32K (4k)             | 4k     | 28.14         | 21.88        | 26.59         |
| LongChat-32K (8k)             | 8k     | 29.39         | 21.69        | 27.03         |
| LongChat-32K (16k)            | 16k    | 30.85         | 23.33        | 26.79         |
| LongChat-32K (32k)            | 32k    | 30.98         | 23.96        | 26.82         |
| LongChat-32K w. ExtenToken*   | 4k     | 30.12         | 23.51        | 25.91         |

<span id="page-6-0"></span>Table 2: The evaluation of long-context understanding tasks from LongBench.

| Model            | GPU Memory (GB) |       |       |       |       | Inference Time (s) |      |      |      |      |
|------------------|-----------------|-------|-------|-------|-------|--------------------|------|------|------|------|
|                  | 4K              | 8K    | 16K   | 32K   | 100K  | 4K                 | 8K   | 16K  | 32K  | 100K |
| LongChat-32K     | 18.12           | 23.68 | 34.79 | 57.03 | OOM   | 0.32               | 0.65 | 1.43 | 3.32 | OOM  |
| StreamingLLM     | 15.11           | 15.11 | 15.11 | 15.11 | 15.11 | -                  | -    | -    | -    | -    |
| LongLLaMA        | 17.73           | 21.40 | 33.41 | OOM   | OOM   | 0.60               | 1.44 | 3.30 | OOM  | OOM  |
| ExtenToken (on)  | 20.33           | 21.59 | 21.59 | 21.59 | 21.59 | 0.28               | 0.49 | 0.86 | 1.57 | 3.43 |
| ExtenToken (off) | 13.96           | 14.21 | 14.75 | 15.79 | 17.54 | 0.08               | 0.08 | 0.10 | 0.12 | 0.23 |

<span id="page-6-1"></span>Table 3: Efficiency analysis (FlashAttention-2 is enabled for LongChat).

#### 4.2.2 Long-Context Understanding

We perform additional evaluations using three long-context understanding tasks from LongBench [\[2\]](#page-9-13), including single-doc QA, multi-doc QA, and summarization. For each evaluation sample, the scaling factor is adjusted case-by-case, which ensures the input data can just fit into the context window of the corresponding LLM (e.g., 4K with LLaMA-2-7B). The evaluation results are presented in Table [2,](#page-6-0) showing that Extensible Tokenization (denoted as Llama-2-7B w. ExtenToken) substantially improves upon the LLaMA-2-7B baseline for both single-doc QA and multi-doc QA tasks. Interestingly, the observation differs for summarization, where the extended context contributes little to the empirical performance (so is the case with LongAlpaca and LongChat where the improvements are mainly resulted from fine-tuning rather than the extended context). The above improvements are pronounced, considering that Extensible Tokenization works with a shorter context length (resulting in much less GPU memory usage and inference time) and does not impact the LLM's original parameters.

Similar with our previous exploration, we directly apply the well-trained extensible tokenizer from LLaMA-2-7B for LongAlpaca and LongChat (denoted as LongAlphaca w. ExtenToken and LongChat w. ExtenToken, respectively). Because the majority of the evaluation samples' sequence lengths can be fully covered by the two models (whose context lengths are 16K and 32K, respectively), the extensible tokenizer is introduced mainly for the compression of the data instead of introducing extra information. In other words, LongAlpaca and LongChat can make use of much shorter inputs (within 4K) for the completion of their tasks. Notably, the extensible tokenizer exhibits a strong compatibility with the two fine-tuned derivatives of LLaMA-2-7B: for both single-doc QA and multi-doc QA, it effectively preserves the two models' full-scale performances with the highly compressed inputs.

#### 4.2.3 Efficiency Analysis

We make analysis for the running efficiency in terms of GPU memory usage and inference time. The performance is measured by taking the average value of 100 forward passes where the last

| Factor                    | Setting                     | 4K   | 8K   | 16K  | 32k  | Single-doc QA |
|---------------------------|-----------------------------|------|------|------|------|---------------|
| Down scaling method       | Random down-sampling        | 8.22 | 7.86 | 7.86 | 7.64 | 23.39         |
|                           | Terminal down-sampling      | 8.23 | 7.88 | 7.66 | 7.58 | 24.04         |
|                           | Strided down-sampling*      | 7.75 | 7.48 | 7.38 | 7.31 | 25.56         |
| Scaling factor (k)        | Monotonous (k = 16)         | 7.55 | 7.40 | 7.32 | 7.29 | 21.37         |
|                           | Dynamic Sampling*           | 7.75 | 7.48 | 7.38 | 7.31 | 25.56         |
| Extensible tokenizer size | First 4-layer (Llama-2-7B)  | 7.89 | 7.64 | 7.52 | 7.46 | 23.32         |
|                           | First 8-layer (Llama-2-7B)* | 7.75 | 7.48 | 7.38 | 7.31 | 25.56         |

<span id="page-7-0"></span>Table 4: Ablation studies. The default settings are marked with "\*".

512 tokens are predicted based on the preceding context. All the experiments are based on one single Nvidia A800-80G GPU. LongChat is the full-attention method, where the FlashAttention-2 is enabled [\[13\]](#page-9-14). StreamingLLM relies on stream processing, whose window size is set to 2048. It is exempted from time evaluation because its current stepwise implementation is too slow. We consider the two alternative ways of inference with Extensible Tokenization (k = 32): the online processing (on) where the input context is streaming presented, and the offline processing (off) where the input context is presented in advance (where the extensible embeddings can be pre-computed). The following observations can be derived from the evaluation results in Table [3.](#page-6-1)

First of all, Extensible Tokenization leads to a very economic usage of GPU memory, which is much smaller than the full-attention methods when the input sequence is long. As introduced, the memory usage of ExtenToken comes from two sources. One is the generation of extensible embeddings, which is performed by sessions with sequence length no more than 4K (ExtenToken (off) is exempted from this step, thus taking even less GPU memory). The other one is the final inference stage based on the extensible embeddings, where the sequence length is much shorter than the raw input. As a consequence, Extensible Tokenization is free from processing the entire long sequence simultaneously, which substantially reduces the memory cost and ensures the extension for a super long context.

Secondly, Extensible Tokenization exhibits a linear growth of the inference time, as the majority of computation is spent on the session-based generation of extensible embeddings. Besides, with the extensible embeddings pre-computed during the offline stage, the inference time of ExtenToken (off) substantially outperforms other methods. This property suggests its potential value to scenarios, like retrieval augmented generation, where the long-context data can be presented in advance.

#### 4.3 Ablation Studies

We perform ablation studies investigate a series of factors which are influential to the performance of Extensible Tokenization, including the down-scaling method, the sampling of scaling factor, and the initialized architecture of the extensible tokenizer. The performances are evaluated with the language modeling task on PG19 and the long-context understanding task on Sing-doc QA (Table [4\)](#page-7-0).

First of all, we explore the impact of down-scaling with two alternative methods: 1) random downsampling, which randomly choose L/k embeddings from the extensible tokenizer's output embeddings (L is the chunk size or session length), 2) terminal down-sampling, which select the last L/k results from the extensible tokenizer's output embeddings. Both alternatives are inferior to our default setting, i.e. the strided down-sampling, where the last embedding in every k steps is chosen. In fact, the strided method is not only simple to implement, but also favorable to the representation quality due to the comprehensive coverage of the context window.

Secondly, we investigate the necessity of dynamically sampling the scaling factor in the training process (denoted as Dynamic Sampling). For comparison, employ a consistent scaling factor throughout the training (denoted as Monotonous). When evaluating the performance of language modeling, both methods utilize the same scaling factor, k = 16, to extend the context, where the Monotonous setting results in a slightly improved performance. However, when it comes to to single-Doc QA, the extensive tokenizer must work with different scaling factors to accommodate inputs within the context window. In this scenario, dynamic sampling demonstrates a notable advantage over the monotonous method, indicating its versatility to make the extension for diverse context lengths.

Thirdly, we analyze the impact of the extensible tokenizer's architecture. All alternatives are initialized with LLaMA-2-7B (chat). However, the model sizes differ where the first 4 and 8 transformer layers are taken from the foundation model. It can be observed that the expansion of model size leads to the improved performances on both language modeling and single-Doc QA. This observation is intuitive, as larger models are of higher expressiveness, which is able to make better compression of the context. However, the larger models also lead to extra costs on training and inference. Indeed, it remains to explore the optimal cost-effectiveness of Extensible Tokenization for each specific scenario.

## 5 Related Works

Long Context Extension. Numerous methods have been proposed to extend the context length of LLMs. One important direction involves modifying position encoding, e.g., Position Interpolation [\[5\]](#page-9-7) and NTK-Aware [\[1\]](#page-9-8), which allows the LLMs to extend their context lengths during the inference time. These methods can also be applied to the pretrained models and get fine-tuned for better long context generation performance [\[25\]](#page-10-0). However, training on extended context data is computationally expensive. Although the training efficiency can be improved by techniques, like LoRA [\[6;](#page-9-0) [19\]](#page-10-12), sparse attention [\[6;](#page-9-0) [9\]](#page-9-2), and FlashAttention [\[14\]](#page-9-15), the cost of training and inference on long context remains substantial. Another line of research focuses on external memory to enhance the LLMs' long context capability. Typically, these methods divide the context by chunks and store them in an additional memory module, which can be retrieved to assist the generation process. For instance, Memorizing Transformers [\[32\]](#page-10-13) directly caches key-value pairs of the context, and utilizes the Top-K retrieval to find the most relevant neighbors for the presented query. Similar strategies are adopted by Landmark Attention [\[22\]](#page-10-14), which employs landmark tokens to represent the chunks for a better retrieval efficiency, and Focused Transformers [\[31\]](#page-10-6) , which leverages contrastive learning to enhance retrieval accuracy. Finally, the context can also be extended by using sliding windows. For example, StreamingLLM [\[33\]](#page-10-1) and LM-Infinite [\[18\]](#page-10-2) only maintain the LLM's activations for the latest tokens, which enables the processing of infinite context. Compared with the above methods, Extensible Tokenization has its unique advantages in terms of flexibility, compatibility, and efficiency. Besides, it is able to collaborate with the existing methods for more effective extension of the context.

Context Compression. There have been continuous effort made for the compression of context. One research direction relies on the explicit compression, where the input text is simplified through summarization or extraction. For instance, LLMLingua [\[21\]](#page-10-15) introduces a budget controller to maintain semantic integrity under high compression ratios and a token-level iterative compression algorithm to condense the context. RECOMP [\[35\]](#page-11-1) proposes both extractive and abstractive compressors to compress the context and improve the performance of RAG. Apart from the explicit methods, another research direction focuses on the implicit compression, which are more similar with our work. One early work was made by Funnel-Transformer [\[12\]](#page-9-10), which gradually shrinks the sequence length of hidden states in different layers of transformers. However, it calls for the presence of the entire input sequence, which is not appropriate for context extension but mainly for the reduction of computation cost. Besides, several recent works make use of special summarizing tokens to compress the context [\[8;](#page-9-6) [23;](#page-10-16) [17\]](#page-10-17). In contrast to the previous works, our method is able to bring in superior performances in scaling the LLM's context thanks to its flexible architecture (i.e. the extensive tokenizer with down-scaling) and sample efficient training method (i.e. the two-stream auto-regression).

## 6 Conclusion

In this paper, we present Extensible Tokenization as a new method to extend the LLM's context. It compresses the raw token embeddings as extensible embeddings, whereby the LLM can perceive more information with the same context window. On top of the auto-regression tasks with the optimized sample efficiency based on two-stream processing, the extensible embeddings can be learned as highly more compact but equally informative representations of the context. Extensible Tokenization is featured by its high flexibility, where the extension for diverse context lengths can be realized by simply making the switch to different scaling factors at the inference time. Besides, the Extensible Tokenization can be introduced as a plug-and-play module, which exhibits a high compatibility with not only the downstream LLM where the extensible tokenizer is trained but also many of its fine-tuned derivatives. Comprehensive experimental studies verify Extensible Tokenization as an effective, efficient, flexible, and compatible method for the extension of LLM's context.

# References

- <span id="page-9-8"></span>[1] Localllama. ntk-aware scaled rope allows llama models to have extended (8k+) con- text size without any fine-tuning and minimal perplexity degradation. [https:](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) [//www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware\\_scaled\\_rope\\_](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) [allows\\_llama\\_models\\_to\\_have/](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), 2023.
- <span id="page-9-13"></span>[2] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint arXiv:2308.14508*, 2023.
- <span id="page-9-3"></span>[3] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. *CoRR*, abs/2004.05150, 2020.
- <span id="page-9-5"></span>[4] Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Scaling transformer to 1m tokens and beyond with RMT. *CoRR*, abs/2304.11062, 2023.
- <span id="page-9-7"></span>[5] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. *arXiv preprint arXiv:2306.15595*, 2023.
- <span id="page-9-0"></span>[6] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. *arXiv preprint arXiv:2309.12307*, 2023.
- <span id="page-9-9"></span>[7] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. *arXiv preprint 2305.14788*, 2023.
- <span id="page-9-6"></span>[8] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 3829–3846. Association for Computational Linguistics, 2023.
- <span id="page-9-2"></span>[9] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*, 2019.
- <span id="page-9-11"></span>[10] Together Computer. Redpajama: an open dataset for training large language models, 2023.
- <span id="page-9-1"></span>[11] Li Dacheng, Shao Rulin, Xie Anze, Sheng Ying, Zheng Lianmin, E. Gonzalez Joseph, Stoica Ion, Ma Xuezhe, and Zhang Hao. How long can open-source llms truly promise on context length?, June 2023.
- <span id="page-9-10"></span>[12] Zihang Dai, Guokun Lai, Yiming Yang, and Quoc Le. Funnel-transformer: Filtering out sequential redundancy for efficient language processing. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020.
- <span id="page-9-14"></span>[13] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. *CoRR*, abs/2307.08691, 2023.
- <span id="page-9-15"></span>[14] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. *Advances in Neural Information Processing Systems*, 35:16344–16359, 2022.
- <span id="page-9-4"></span>[15] Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. *arXiv preprint arXiv:2307.02486*, 2023.
- <span id="page-9-12"></span>[16] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

- <span id="page-10-17"></span>[17] Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. *arXiv preprint arXiv:2307.06945*, 2023.
- <span id="page-10-2"></span>[18] Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Simple on-the-fly length generalization for large language models. *CoRR*, abs/2308.16137, 2023.
- <span id="page-10-12"></span>[19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021.
- <span id="page-10-5"></span>[20] Xinting Huang and Nora Hollenstein. Long-range language modeling with selective cache. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pages 4838–4858. Association for Computational Linguistics, 2023.
- <span id="page-10-15"></span>[21] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. *arXiv preprint arXiv:2310.05736*, 2023.
- <span id="page-10-14"></span>[22] Amirkeivan Mohtashami and Martin Jaggi. Landmark attention: Random-access infinite context length for transformers. *arXiv preprint arXiv:2305.16300*, 2023.
- <span id="page-10-16"></span>[23] Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. *arXiv preprint arXiv:2304.08467*, 2023.
- <span id="page-10-3"></span>[24] Jesse Mu, Xiang Lisa Li, and Noah D. Goodman. Learning to compress prompts with gist tokens. *CoRR*, abs/2304.08467, 2023.
- <span id="page-10-0"></span>[25] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. *arXiv preprint arXiv:2309.00071*, 2023.
- <span id="page-10-7"></span>[26] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, Chloe Hillier, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling. *arXiv preprint*, 2019.
- <span id="page-10-4"></span>[27] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020.
- <span id="page-10-11"></span>[28] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. *arXiv preprint arXiv:2308.12950*, 2023.
- <span id="page-10-9"></span>[29] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.
- <span id="page-10-8"></span>[30] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-10-6"></span>[31] Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłos. Focused transformer: Contrastive training for context scaling. ´ *arXiv preprint arXiv:2307.03170*, 2023.
- <span id="page-10-13"></span>[32] Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. In *The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022.
- <span id="page-10-1"></span>[33] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*, 2023.
- <span id="page-10-10"></span>[34] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*, 2023.

- <span id="page-11-1"></span>[35] Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and selective augmentation. *arXiv preprint arXiv:2310.04408*, 2023.
- <span id="page-11-0"></span>[36] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. *Advances in neural information processing systems*, 33:17283–17297, 2020.