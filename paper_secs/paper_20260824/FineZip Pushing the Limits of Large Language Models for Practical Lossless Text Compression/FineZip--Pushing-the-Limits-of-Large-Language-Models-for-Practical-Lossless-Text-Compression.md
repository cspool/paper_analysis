# **FineZip** : Pushing the Limits of Large Language Models for Practical Lossless Text Compression

Fazal Mittu<sup>1</sup> , Yihuan Bu<sup>1</sup> , Akshat Gupta<sup>1</sup> , Ashok Devireddy<sup>1</sup> , Alp Eren Ozdarendeli<sup>1</sup> , Anant Singh<sup>2</sup> , Gopala Anumanchipalli<sup>1</sup>

> <sup>1</sup>UC Berkeley, <sup>2</sup>NYU <akshat.gupta@berkeley.edu>

### Abstract

While the language modeling objective has been shown to be deeply connected with compression, it is surprising that modern LLMs are not employed in practical text compression systems. In this paper, we provide an in-depth analysis of neural network and transformer-based compression techniques to answer this question. We compare traditional text compression systems with neural network and LLM-based text compression methods. Although LLM-based systems significantly outperform conventional compression methods, they are highly impractical. Specifically, LLMZip, a recent text compression system using Llama3-8B requires 9.5 days to compress just 10 MB of text, although with huge improvements in compression ratios. To overcome this, we present FineZip - a novel LLM-based text compression system that combines ideas of online memorization and dynamic context to reduce the compression time immensely. FineZip can compress the above corpus in approximately 4 hours compared to 9.5 days, a 54 times improvement over LLMZip and comparable performance. FineZip outperforms traditional algorithmic compression methods with a large margin, improving compression ratios by approximately 50%. With this work, we take the first step towards making lossless text compression with LLMs a reality. While FineZip presents a significant step in that direction, LLMs are still not a viable solution for large-scale text compression. We hope our work paves the way for future research and innovation to solve this problem.

### 1 Introduction

While the relationship between language modeling and compression has long been known [\(Schmidhu](#page-4-0)[ber and Heil,](#page-4-0) [1996;](#page-4-0) [Mahoney,](#page-4-1) [2000;](#page-4-1) [Goyal et al.,](#page-4-2) [2018;](#page-4-2) [Bellard,](#page-4-3) [2019\)](#page-4-3), recent works [\(Delétang et al.,](#page-4-4) [2024;](#page-4-4) [Huang et al.,](#page-4-5) [2024\)](#page-4-5) have reinforced this connection. [Delétang et al.](#page-4-4) [\(2024\)](#page-4-4) recently showed

large language models (LLMs) can be used to compress data from various modalities. [Huang et al.](#page-4-5) [\(2024\)](#page-4-5) followed up this work by showing that increasing compression abilities of LLMs is linearly correlated to downstream task performance.

Previous works have exploited this connection for lossless text compression. Neural network based models have been implemented for text compression [\(Schmidhuber and Heil,](#page-4-0) [1996;](#page-4-0) [Mahoney,](#page-4-1) [2000;](#page-4-1) [Goyal et al.,](#page-4-2) [2018\)](#page-4-2) and have reached better compression performance than traditional algorithmic compressors such as gzip. More recent methods have explored using LSTM and transformer models [\(Bellard,](#page-4-3) [2019,](#page-4-3) [2021\)](#page-4-6). These methods fall under the "online" compressors category, where a randomly initialized model is directly trained on the data being compressed. In this case, the model parameters also become part of the compression. A recent effort, LLMZip [\(Valmeekam et al.,](#page-4-7) [2023\)](#page-4-7), tested the use of LLMs for lossless compression. Given an LLM's ability to predict the next token provided a fixed-length context window, a tokenized text can be stored as probabilistic ranks produced by an LLM predicting the next token. This is a type of "offline" compression, with a fixed system used for both compression and decompression of all incoming text.

In this paper, we build on prior work and introduce FineZip, which uses LLMs for lossless text compression with both online and offline components. FineZip combines an "online" component, which memorizes the data being compressed, with an "offline" component in the form of pre-trained LLMs for compression. The "online" memorization is done by fine-tuning the model on the data being compressed in a parameter-efficient way [\(Hu](#page-4-8) [et al.,](#page-4-8) [2021;](#page-4-8) [Dettmers et al.,](#page-4-9) [2023\)](#page-4-9) with an additional constant overhead of the learned embeddings during fine-tuning. The "offline" component of the system is the pre-trained LLM which remains fixed across different corpora. Figure [1](#page-1-0) depicts the sys-

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> Memorization (PEFT) happy happy am am 0.8 0.5 0.2 0.1 0.3 Pretrained Weights Input text "Today is a sunny" **Layer Norm Tokenizer Ranks PEFT Adapter Traditional Feed Forward** Compression day 0.7 ZIP **Dynamic Layer Norm** life 0.2 Context Window 0.1 sky Multi-Head ... ... **PEFT** Attention **Embeddings**
![](_page_1_Figure_0.jpeg)

Figure 1: System diagram for FineZip.

tem diagram for FineZip. With this approach, we can leverage the benefits of online compression for improved performance without the drawback of requiring additional storage for model parameters.

Additionally, with FineZip we allow for a dynamic context where each token being compressed has a context size of equal to its position in a sentence. This allows us to batch compression and decompression steps using LLMs, allowing for significant speed-up. "Online memorization" using PEFT methods also allows the model to compensate for loss of performance due to a dynamic context, while a dynamic context allows for batching which allows compression and decompression of many batches of text in parallel within a fixed compute budget. With FineZip, we can achieve 54 times faster compression times with minor loss of performance when compared to LLMZip, still outperforming traditional text compression methods by a huge margin. Our work also shows that compression rates of LLM-based methods are still not low enough for practical use cases, and although FineZip pushes the limits of using LLMs lossless text compression in practice, much work still needs to be done. The code for our work can be found here - [https://github.com/fazalmittu/](https://github.com/fazalmittu/FineZip) [FineZip](https://github.com/fazalmittu/FineZip).

### 2 Introducing **FineZip**

The most basic form of compression using LLMs would be to tokenize the input text. Since each char-

acter in a word occupies 8 bits (1 byte in UTF-8 encoding), representing the word as a token, essentially converting it into a number, will almost always reduce the number of bytes needed to represent it. This connection was also observed in [Delé](#page-4-4)[tang et al.](#page-4-4) [\(2024\)](#page-4-4). As a next step, we can use the predictive capabilities of LLMs for compression. This idea is used in LLMZip [\(Valmeekam et al.,](#page-4-7) [2023\)](#page-4-7) where they use a pre-trained LLM for text compression. The connection between language modeling and compression becomes intuitive when we take a deeper look at the language modeling objective, implemented using a cross-entropy loss. It aims to make each token in the training data the most probable token given the context preceding it, thus minimizing the number of bits required to represent the rank of the token in the vocabulary list, when ranked in descending order according to their probability. Following this line of thought, we propose an intuitive yet effective way of enhancing this - fine-tuning the model on the data being compressed.

A challenge towards fine-tuning modern LLMs is that they are memory-intensive. Additionally, if we fine-tune the entire model on the text being compressed, then the entire LLM becomes part of the compression, requiring an additional space equal to the space required to store the model for decompression. Thus, we propose FineZip, a compression framework that involves parameter-efficient fine-tuning (PEFT) [\(Mangrulkar et al.,](#page-4-10) [2022\)](#page-4-10) on the input text as an "online" step prior to compression. We call this the "online memorization" step which makes the data being compressed more probable for the LLM. This fine-tuning is implemented using LoRA [\(Hu et al.,](#page-4-8) [2021\)](#page-4-8) and is much faster than full fine-tuning, requires much less GPU memory, and requires a very small amount of additional storage for the trained embeddings. The additional embedding storage does not scale with the dataset being compressed and becomes negligible at large sizes of corpora.

Another key difference between LLMZip and FineZip is that FineZip adopts a dynamic context size approach rather than maintaining a fixed sliding window. LLMZip uses a permanent sliding window approach, where the rank of each token produced has a fixed context window of a preset context size (512 as chosen by original authors). This by design makes the compression process extremely autoregressive and non-parallelizable, as to produce the rank of a token, you need the previous 512 tokens.

FineZip overcomes this limitation by employing a two-step dynamic context window technique:

- 1. Divide the corpus into chunks of a pre-decided window length.
- 2. Produce the ranks of each token within the window such that the rank for the i th token is produced based on the tokens preceding it

The dynamic context window gives a variable context size to each token in a chunk. For a uniform comparison, we use a chunking size of 512 in FineZip, which is the same as the context window size chosen by LLMZip. In FineZip, the i th token in a chunk has a context size of i − 1, thus only the final token in a chunk has access to full context length of 512. In contrast, every token in LLMZip has access to the full context length of 512. The dynamic context leads to some loss of performance, which is made up for by online memorization.

### 3 Experiments

We begin by comparing FineZip with (i) traditional text compression methods - bzip2 [\(Julian](#page-4-11) [Seward,](#page-4-11) [2024\)](#page-4-11), zlib [\(Jean-loup Gailly,](#page-4-12) [2024\)](#page-4-12), and gzip [\(Jean-loup Gailly,](#page-4-13) [1992\)](#page-4-13), (ii) neural network based text compression methods - NNCP [\(Bellard,](#page-4-6) [2021\)](#page-4-6), and the (iii) recent LLM-based text compression method called LLMZip. For both FineZip and LLMZip, we use Llama-3 8B [\(Dubey et al.,](#page-4-14) [2024\)](#page-4-14).

<span id="page-2-0"></span>

| Method       | Compression Ratio | Time (min) |
|--------------|-------------------|------------|
| zlib         | 0.3251            | 0.0083     |
| gzip         | 0.3238            | 0.0141     |
| bzip2        | 0.2374            | 0.0437     |
| NNCP         | 0.15021           | 251        |
| LLMZip (AC)  | 0.0795            | 13571      |
| LLMZip       | 0.1163            | 13651      |
| Finezip (AC) | 0.0797            | 13118      |
| Finezip      | 0.12799           | 250        |
| Finezip-4bit | 0.1445            | 67         |

Table 1: Comparison of Compression Methods on 10mb

Modifications to LLMZip: LLMZip originally used Llama-1-7B [\(Touvron et al.,](#page-4-15) [2023a\)](#page-4-15) while we leverage Llama-3-8B for both LLMZip and FineZip for uniform comparison. Additionally, LLMZip used two methods for compression - one using arithmetic coding (AC) and the other using a secondary compression methods on generated ranks. LLMZip uses zlib [\(Jean-loup Gailly,](#page-4-12) [2024\)](#page-4-12) as a secondary compression method over ranks whereas our experiments show that bzip2 provides a much better compression ratio (Appendix: [A.1\)](#page-5-0). Thus, we use bzip2 as our secondary compression method for LLM ranks in both LLMZip and FineZip. We also refer to bzip2 as the baseline for text compression using traditional compression methods (Table [1\)](#page-2-0). To offer a better comparison, we also create a version of FineZip that incorporates arithmetic coding. The process uses the logits that the LLM outputs for each new token as the probability distribution update for the arithmetic coding scheme.

We used the first 10mb of the enwik8 [\(Marcus](#page-4-16) [Hutter,](#page-4-16) [2006\)](#page-4-16) dataset which is a standard benchmark for compression tasks. Though compression ratio (ratio of compressed file size and original file size) is the key metric, we are also interested in measuring time taken by these compression systems to evaluate practicality. The results are shown in Table [1.](#page-2-0) The first key observation is that neural network and LLM based compression methods have significantly better compression ratios than traditional text compression methods (zlib, gzip, bzip2), thus highlighting the potential impact of these methods for text compression. The second key observation is that neural network and LLM based methods takes a long time to compress even small amounts of text, thus preventing their use in practice. This is especially true when using AC for compression in LLM-based methods, which produces exceptional compression ratios but also requires unprecedentedly large amounts of time.

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Compression Ratio by Fine-tuning Epochs 0.20 Compression Ratio 91.0 91.0 81.0 Model GPT2-XL Llama 2 7b Llama 38b 0.12 64 256 Fine-tuning Epochs
![](_page_3_Figure_0.jpeg)

Figure 2: FineZip ablations for different fine-tune epochs

For LLMZip with AC, the time taken to compress 10MB of data is approximately 9.5 days. Thus, we do not explore AC-based LLM compression further and strictly compare only rank-based LLM baselines.

Table [1](#page-2-0) shows that FineZip is able to achieve comparable or better compression ratios than both NNCP and LLMZip with a much faster compression time. Specifically, we see that FineZip has a much better compression ratio than NNCP with comparable amount of compression time, while the 4-bit quantized FineZip is approximately 4 times faster than NNCP and still exhibits a better compression ratio. FineZip compresses enwik8 within 4 hours, compared to approximately 227 hours taken by LLMZip. This is a 54x improvement on compression time with a minor drop of 1 percentage point in compression ratio.

#### 3.1 **FineZip** Ablations

FineZip uses an "online memorization step" as shown in Figure [1](#page-1-0) before performing compression. This is done using Low-Rank Adaptation (LoRA) [\(Hu et al.,](#page-4-8) [2021\)](#page-4-8). We compare the effect of finetuning on compression using 3 different language models: GPT2-XL 1.3B [\(Radford et al.,](#page-4-17) [2019\)](#page-4-17), LLama-2 7B [\(Touvron et al.,](#page-4-18) [2023b\)](#page-4-18), and LLama-3 8B [\(Dubey et al.,](#page-4-14) [2024\)](#page-4-14). We see that for each model, memorization improves the absolute compression ratio by at least 1 percentage point or a relative improvement of about 8% over its nonfine-tuned baseline as shown in Figure [2.](#page-3-0) This is significant especially when dealing with such low compression rates. It should be noted that the time taken for memorization is negligible compared to compression time and can be ignored.

Quantization: We saw in Table [1](#page-2-0) that dynamic context helps speed up the compression process by significant amounts, while online memorization

<span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> Quantization vs Compression Ratio & Time<sub>16000</sub> 14000 0.18 Compression Ratio -12000 -10000 0.16 Batch Size: 70 Taken 8000 Batch Size: 64 Batch Size: 56 0.14 6000 Batch Size: 16 4000 0.12 2000 0.10 32 bit 4 bit 8 bit 16 bit Bit Configuration
![](_page_3_Figure_7.jpeg)

Figure 3: Compressing 10mb dataset with LLama-3 8B loaded with 4, 8, 16, and 32-bit precision. Purple bar shows compression ratio, red line shows time taken to compress. Each batch size was chosen to max out memory on a 48GB GPU.

is able to mitigate the loss in performance. We further push the limits of compression time using quantization. We perform the memorization step using QLoRA [\(Dettmers et al.,](#page-4-9) [2023\)](#page-4-9) and perform compression using the quantized model. We do this using a fixed compute budget of 48GB GPU memory on a single NVIDIA A6000 GPU. Lower precision models will allow us to increase batch size and in turn, decrease time needed to compress a file by a sizable amount. Figure [3](#page-3-1) shows that finetuning/compressing a 4 bit model allows us to fit a batch size of 70 on one A6000 GPU and achieve a compression time of 67 minutes. This 4x speed up makes FineZip not only a competitive compressor out-performning traditional text compression systems by a huge margin, but also the fastest neural network/transformer based compression currently available.

