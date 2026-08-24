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

