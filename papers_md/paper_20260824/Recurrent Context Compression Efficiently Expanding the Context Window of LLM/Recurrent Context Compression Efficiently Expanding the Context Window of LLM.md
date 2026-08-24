# Recurrent Context Compression: Efficiently Expanding the Context Window of LLM

Chensen Huang<sup>1</sup> , Guibo Zhu2,3, Xuepeng Wang2,3 , Yifei Luo<sup>1</sup> , Guojing Ge2,3 , Haoran Chen1,2 , Dong Yi2,3 , Jinqiao Wang2,3

> <sup>1</sup>University of Chinese Academy of Sciences, 2 Institute of Automation, Chinese of Academy, <sup>3</sup>Wuhan AI Research

> > Correspondence: [huangchensen2022@ia.ac.cn](mailto:ken@yuniversity.edu)

### Abstract

To extend the context length of Transformerbased large language models (LLMs) and improve comprehension capabilities, we often face limitations due to computational resources and bounded memory storage capacity. This work introduces a method called Recurrent Context Compression (RCC), designed to efficiently expand the context window length of LLMs within constrained storage space. We also investigate the issue of poor model responses when both instructions and context are compressed in downstream tasks, and propose an instruction reconstruction method to mitigate this problem. We validated the effectiveness of our approach on multiple tasks, achieving a compression rate of up to 32x on text reconstruction tasks with a BLEU4 score close to 0.95, and nearly 100% accuracy on a passkey retrieval task with a sequence length of 1M. Finally, our method demonstrated competitive performance in long-text question-answering tasks compared to non-compressed methods, while significantly saving storage resources in long-text inference tasks. Our code, models, and demo are available at [https://github.](https://github.com/WUHU-G/RCC_Transformer) [com/WUHU-G/RCC\\_Transformer](https://github.com/WUHU-G/RCC_Transformer)

### 1 Introduction

With the rapid advancement of natural language processing technologies, Transformer-based large language models (LLMs) have become a key driving force in this field. However, when handling long text inputs, LLMs often encounter limitations in context window length. These limitations stem from several inherent factors in the model architecture and training methods. Firstly, during the inference phase, models are constrained by the pretraining text length, leading to a significant decline in quality when the generated sequence exceeds the pretrained context window. Secondly, the design of the Transformer architecture requires storing information from the entire input sequence, which

results in a substantial memory footprint due to the KV-Cache during inference.

To address these issues, related research works [\(Hochreiter and Schmidhuber,](#page-8-0) [1997;](#page-8-0) [Child et al.,](#page-8-1) [2019;](#page-8-1) [Wu et al.,](#page-9-0) [2022;](#page-9-0) [Rae et al.,](#page-9-1) [2019;](#page-9-1) [Bulatov](#page-8-2) [et al.,](#page-8-2) [2022;](#page-8-2) [Liu et al.,](#page-8-3) [2023;](#page-8-3) [Mohtashami and Jaggi,](#page-9-2) [2023;](#page-9-2) [Beltagy et al.,](#page-8-4) [2020\)](#page-8-4) have optimized training methods, model structures, and KV-Cache optimization, thereby extending the context window of LLMs. Among these, context compression techniques [\(Rae et al.,](#page-9-1) [2019;](#page-9-1) [Snell et al.,](#page-9-3) [2022;](#page-9-3) [Cheva](#page-8-5)[lier et al.,](#page-8-5) [2023;](#page-8-5) [Wingate et al.,](#page-9-4) [2022;](#page-9-4) [Mu et al.,](#page-9-5) [2023;](#page-9-5) [Ge et al.,](#page-8-6) [2023;](#page-8-6) [Munkhdalai et al.,](#page-9-6) [2024;](#page-9-6) [Ren et al.,](#page-9-7) [2023;](#page-9-7) [Li et al.,](#page-8-7) [2023\)](#page-8-7) are considered promising because they can compress context or prompts into shorter forms while maintaining good performance, thus enabling the inference of longer context windows within limited resources. Figure [1](#page-1-0) compares the memory resource consumption of our method with non-compression methods. Additionally, most text compression-based works can be integrated and combined with other context window extension techniques to enhance performance.

However, existing context compression methods face three major challenges in long-text language modeling. Firstly, the efficiency of compression has certain limitations. For example, ICAE with 14B parameters [\(Ge et al.,](#page-8-6) [2023\)](#page-8-6) experiences a significant performance drop beyond an 8x compression rate. Secondly, most context compression research focuses on shorter sequences rather than long texts. We found that language models trained for context compression on short sequences perform poorly when directly extended to long sequences, necessitating new methods to improve long-text compression performance. Lastly, in practical applications, we observed that contextcompressed language models face the issue of context-instruction confusion in downstream tasks. When both context and instructions are compressed simultaneously, the model often struggles to fol-

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> Pythia-1.4b - 0-16k RCC-1.4x2b - 0-16k 19914 20000 -Pythia-1.4b 0-16k 20000 RCC-1.4x2b - 0-16k 17500 17500 (a) 15000 ∫ | (a) 15000 | (b) (c) (c) (c) (c) (c) (c) (c) (c) (c) (c ) 12500 -0 10000 e 12500 -11752 10000 -10000 Memory 8182 7910 7788 7664 7500 -7500 -6322 5000 -5000 4078 2500 2500 16 16 Context Length (k) Context Length (k)
![](_page_1_Figure_0.jpeg)

Figure 1: GPU memory Consumption of Different Models with Increasing Length. Left: Pythia-1.4b, Right: RCC model using Pythia-1.4b for both encoder and decoder. Both models utilize FlashAttention-2 (Dao, 2023). A more detailed analysis of GPU memory consumption can be found in Appendix B.

low instructions correctly, resulting in poor controllability. This issue has not been emphasized or addressed in previous studies, which typically compress only context or instruction texts individually.

To address the aforementioned issues, this paper makes the following contributions:

Firstly, we propose a context compression model structure based on an autoencoder, which we call the Recurrent Context Compression (RCC) architecture. RCC significantly reduces information loss during the compression process, greatly enhancing compression efficiency. In experiments, we achieved nearly 95% BLEU-4 scores with a 32x compression rate on text reconstruction tasks. Although the encoder occupies some memory resources, the memory required by traditional language models exceeds that of the context compression model when the sequence length surpasses a certain threshold, as shown in Figure 1.

Secondly, we propose a new training method to adapt long-text context compression language models. We introduce a recurrent compression mechanism to overcome the context window limitations of the encoder, allowing the model to compress texts beyond the encoder window length. When training long-text context compression language models, the length of the context sequence input to the encoder is proportional to the required computational resources, limiting the extension of context length during training. Therefore, we propose a simple yet effective solution: initially, conducting full-parameter training on shorter sequences. Subsequently, we freeze the encoder on the saved model weights and continue training on longer sequences, enabling the extension of training context length under constrained computational resources. The detailed information can be found in Section 3.3.1.

Lastly, in downstream text generation tasks, we found that when both context and instruction texts are compressed, the model struggles to follow instructions properly, leading to a decline in response quality. To mitigate this issue, we leverage the text reconstruction capability of the context compression language model, allowing the decoder to reconstruct the instruction content from the compressed vectors and continue generating responses based on the instructions. This significantly improves the output quality when both context and instructions are compressed, achieving results close to those obtained by inputting instructions directly into the encoder.

#### 2 Related Work

**Context Compression:** Early approaches to context compression aimed to derive sentence representation vectors for tasks such as document retrieval. Transformer-based autoencoder architectures like TSDAE (Wang et al., 2021) and Nugget (Qin and Van Durme, 2023) are relevant to our work. In TSDAE, noise such as word deletion or swapping is added to input sentences to train sentence embedding vectors. The encoder compresses the corrupted sentence into a fixed-size vector, which the decoder then reconstructs into the original input text. However, such approaches cannot be directly applied to text generation tasks. Gist(Mu et al., 2023) leverages Transformer-based large language models (LLMs) as autoencoders. During training, a clever masking matrix compresses prompts into a few Gist tokens, which can still prompt the language model for responses. Similar prompt compression work was proposed by [\(Wingate et al.,](#page-9-4) [2022\)](#page-9-4). However, these tasks only compress prompts.

Several works [\(Snell et al.,](#page-9-3) [2022;](#page-9-3) [Chevalier et al.,](#page-8-5) [2023;](#page-8-5) [Ge et al.,](#page-8-6) [2023;](#page-8-6) [Ren et al.,](#page-9-7) [2023;](#page-9-7) [Li et al.,](#page-8-7) [2023;](#page-8-7) [Jiang et al.,](#page-8-9) [2023b;](#page-8-9) [Munkhdalai et al.,](#page-9-6) [2024\)](#page-9-6) have focused on context compression. ICAE[\(Ge](#page-8-6) [et al.,](#page-8-6) [2023\)](#page-8-6) is similar to our work but suffers from lower compression efficiency and lacks extensive research on longer sequences. Additionally, [\(Jiang](#page-8-9) [et al.,](#page-8-9) [2023b\)](#page-8-9) employed non-vector-based context compression by using smaller language models to input long texts and generate more compact short texts. Selective context, proposed by [\(Li](#page-8-7) [et al.,](#page-8-7) [2023\)](#page-8-7), identifies and prunes redundancy in the input context to enhance LLM inference efficiency, making inputs more compact. Recently, [\(Munkhdalai et al.,](#page-9-6) [2024\)](#page-9-6) proposed a similar work combining two attention mechanisms with context compression functionality, showing promising results. However, this new attention mechanism cannot be directly applied to pre-trained open-source LLMs and faces attention optimization challenges in practical applications. None of the studies have thoroughly investigated the problem of instructional confusion that arises when both instructions and contextual information are subjected to compression. Our work introduces a new solution to mitigate this issue.

Additionally, our work is inspired by language models with recurrent structures [\(Hochreiter and](#page-8-0) [Schmidhuber,](#page-8-0) [1997;](#page-8-0) [Gu and Dao,](#page-8-10) [2023;](#page-8-10) [Sun et al.,](#page-9-10) [2023;](#page-9-10) [Peng et al.,](#page-9-11) [2023\)](#page-9-11). These models compress historical context within a certain range into the hidden state of a single time step, enabling the current token to access information from the previous step for inference. They demonstrate strong competitiveness with Transformer models, indicating that compressing token information over a certain length can achieve lossless inference.

Long Context LLM: LLMs typically fix the context window length during training, such as the Pythia [\(Biderman et al.,](#page-8-11) [2023\)](#page-8-11), LLaMA [\(Touvron](#page-9-12) [et al.,](#page-9-12) [2023a](#page-9-12)[,b\)](#page-9-13), and Mistral [\(Jiang et al.,](#page-8-12) [2023a\)](#page-8-12) series. Consequently, researchers have explored various methods to extend the context window length of pre-trained language models. These methods[\(Chen et al.,](#page-8-13) [2024;](#page-8-13) [Tworkowski et al.,](#page-9-14) [2023;](#page-9-14) [Chen et al.,](#page-8-14) [2023;](#page-8-14) [Liu et al.,](#page-8-3) [2023,](#page-8-3) [2024\)](#page-8-15), which have achieved notable results based on existing pre-trained models. Our approach combines these methods, allowing us to apply them to the encoder

or decoder to achieve more extended compression effects.

### 3 Design of RCC

### 3.1 Method Overview

Our encoder design is inspired by the Mambabased LLM [\(Gu and Dao,](#page-8-10) [2023\)](#page-8-10). Mamba is essentially a state space model, similar to an RNN. In Mamba, the current token only needs to access the state vector from the previous timestep to complete the current inference step. However, as the context length increases, the performance of Mamba deteriorates. This indicates that the state vector at each timestep in Mamba can store only a limited length of historical context information. Therefore, we propose a compromise: for long sequences, we can divide them into fixed-length short sequences and iteratively compress each short sequence into a state vector. We concatenate the state vectors of each short sequence as the historical state information during inference. This approach maximizes the retention of complete historical information while leveraging the model's compression capabilities to save memory. In this paper, we use compression rate to reflect the maximum context length that a state vector at each timestep can store. Our experiments show that Transformers also have this capability because a Transformer can be viewed as a special state space model or RNN. Recent studies have shown that attention can be viewed as an RNN [\(Feng et al.,](#page-8-16) [2024\)](#page-8-16).

### 3.2 RCC Model Architecture

As shown in Figure [2,](#page-3-0) the RCC model architecture is similar to ICAE[\(Ge et al.,](#page-8-6) [2023\)](#page-8-6), consisting of an encoder and a decoder. Unlike the connection method in ICAE, where the final layer vector of the encoder is used as the input to the decoder, we take a different approach. We use the output information from each layer of the encoder. This information is then linearly mapped and input into the decoder. This method obtains more feature information, and the relevant ablation experiments can be found in Figure [3a.](#page-5-0) The encoder can be a Transformer-based LLM or an RNN-based LLM, while the decoder is a Transformer-based LLM. The encoder is responsible for compressing the information, and the decoder reads the compressed information and performs inference. The decoder can fully learn the compressed information vector at any position using the attention mechanism. Af-

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> h<sub>80</sub> ... ... ... ... Dncoder LLM Layer;+1 Context compressed vector Residual connections MLP h<sub>2049</sub> h<sub>4096</sub> h<sub>2048</sub> h<sub>2560</sub> ... ... ... h<sub>80</sub> ... ••• ••• h<sub>128</sub> ••• Short Context hidden state of instruction or prompt Encoder LLM layer; Long Context h<sub>8192</sub> h<sub>2049</sub> h<sub>5120</sub> ... ... ... h<sub>2048</sub> ••• h<sub>2560</sub> ... h<sub>4096</sub> ••• Waiting for compression Compression complete Compressing
![](_page_3_Figure_0.jpeg)

Figure 2: The structure of the encoder and decoder in RCC layer i. The maximum context window of the encoder is 2048. The encoder has a compression rate of 32, and we use the vectors at positions that are multiples of 32 in the output as the compressed vectors. Each segment will generate a compressed vector of length 64. When the sequence length exceeds 2048, the encoder performs cyclic segmentation and compression. The compressed vectors produced between segments in the encoder are independent, while those generated within a segment are correlated. The decoder's input is the residual connection between the input vector from the previous layer and the compressed vector after linear mapping. All compressed vectors will interact within the decoder.

ter proper training, the maximum context length that the RCC model can support is the encoder compression rate multiplied by the decoder context window length.

### 3.2.1 RCC Encoder

The primary task of the RCC encoder is to compress long sequences. The initialized encoder is a pretrained language model, which can be based on either Mamba or Transformer architectures. By setting a compression rate, we divide long sequences into fixed-length short sequences and iteratively feed these short sequences into the encoder. As illustrated in Figure [2,](#page-3-0) we locate the token at each position multiple of the compression rate within each short sequence. The output vectors of these tokens from each layer serve as compressed vectors, which are concatenated to form the final compressed vector for the entire sequence. Through this method, the RCC encoder accomplishes the compression modeling of the entire long context.

## 3.2.2 RCC Decoder

The decoder of the RCC is a Transformer-based language model responsible for the final text inference. Its inputs include the compressed vectors from the encoder and token embedding vectors related to the prompts. Each layer's compressed vector from the encoder passes through a linear layer

before being input into the decoder. For the first layer's mapped vector, we concatenate it with the decoder's token embedding vector and then feed it into the first block of the decoder. Subsequently, the output part of each block is connected with the corresponding layer's compressed vector through residual connections. It is crucial to note that only the output vectors corresponding to the compressed information will have residual connections, while the other output parts remain unchanged. If the number of compressed vector layers does not match the number of decoder layers, we apply simple rulebased mappings, either by duplicating to increase the number of layers or averaging to reduce the number of layers.

#### 3.3 Model Training Tasks

For the model training tasks, we require the model to possess both contextual memory and contextual reasoning abilities. Therefore, we selected text reconstruction and text continuation tasks. Traditional autoencoding text reconstruction tasks [\(Ge et al.,](#page-8-6) [2023\)](#page-8-6) are not suitable for text generation paradigms, so we replaced them with a random prompt text reconstruction task. Specifically, we randomly extract a short text segment from the encoder's input text as a prompt to the decoder, requiring the decoder to reconstruct the content following the prompt by leveraging the

compressed information and the prompt. This task enhances the model's memory ability. Additionally, to strengthen the model's reasoning ability, we employed text continuation tasks. Relevant formulas can be found in Appendix [C.](#page-10-1)

### <span id="page-4-0"></span>3.3.1 Long Text Training Methods

We adopt a cyclic segmentation approach for computing long sequences. During model inference, we only need to store the compressed vectors. However, during training, we also need to store the gradient information for the entire long sequence, which significantly exceeds memory limits. We mitigate this issue using a simple yet effective twostage training method. In the first stage, we perform full-parameter fine-tuning with a large number of shorter sequences, allowing the encoder to sufficiently learn how to compress the context into a single vector. Correspondingly, the decoder learns to infer or reconstruct from the compressed vectors. After the first stage, the encoder is capable of producing standardized compressed vectors. At this point, we input longer sequences and freeze the encoder, enabling the decoder to learn how to infer from more compressed vectors. This method does not require complex gradient optimization algorithms or substantial GPU memory resources[\(Liu](#page-8-3) [et al.,](#page-8-3) [2023;](#page-8-3) [Chevalier et al.,](#page-8-5) [2023;](#page-8-5) [Liu et al.,](#page-8-3) [2023\)](#page-8-3) and can efficiently scale to longer sequences. We validated the effectiveness of this method on a 1Mlength key retrieval task.

### 3.3.2 Instruction Reconstruction Method

To address the poor performance when both instructions and text are compressed simultaneously, we propose an instruction reconstruction method. During the fine-tuning phase, we input the instruction as part of the context to the model's encoder, with the instruction randomly placed at the beginning or end of the context. The decoder is then required to first reconstruct the instruction and subsequently answer the question based on the instruction.

### 4 Experiments

First, we conducted text compression rate experiments using the random prompt text reconstruction task, selecting ICAE [\(Ge et al.,](#page-8-6) [2023\)](#page-8-6) as the baseline model. Subsequently, we evaluated our method's performance on long text tasks, including the passkey context block retrieval task with 1M characters and the long document questionanswering benchmark in Longbench [\(Bai et al.,](#page-8-17)

[2023\)](#page-8-17).For model architecture, we used pythia-1.4b [\(Biderman et al.,](#page-8-11) [2023\)](#page-8-11) as the encoder and tested mamba-1.4b [\(Gu and Dao,](#page-8-10) [2023\)](#page-8-10), both supporting a 2048 context window. The decoder was pythia-1.4b. We randomly sampled about 5 billion tokens from the pile [\(Gao et al.,](#page-8-18) [2020\)](#page-8-18) dataset as the training set, concatenating these tokens into a continuous ultra-long one-dimensional array. The learning rate was set to 1e-4. Training was stopped if the model failed to converge after one epoch or converged prematurely within an epoch, typically completing within 30 hours on a server with 8 A800 GPUs.

#### 4.1 Text Reconstruction

In the initial training phase, we used a combination of random prompt text reconstruction tasks and Text Continuation tasks, with a ratio of 9:1. This was followed by fine-tuning with the random prompt text reconstruction task. The input sequence length for the encoder was set to 2048 tokens, while the uncompressed part of the decoder had an input length of 512 tokens. For the random prompt text reconstruction task, the uncompressed part of the decoder's input was a subset of the encoder's input sequence, including prompts and the text to be reconstructed. The prompts formed the initial part of this subsequence and were excluded from the loss calculation; only the loss of the reconstructed text following the prompts was calculated. In the Text Continuation task, the decoder's uncompressed input sequence was the continuation of the encoder's input sequence. We assessed the model's compression performance using the BLEU-4 score, comparing the reconstructed text to the actual text. To achieve this, we created 100 encoder input samples, each with a token length of 2048. To ensure fairness in the scores, we selected 5 text segments as prompts for the decoder at every 300-token interval from the sample. Each decoder then calculated the reconstruction score for prompts at 5 different positions, and we averaged these scores. The prompt text length was about 10 tokens, and the reconstruction text length was about 500 tokens. Reconstruction examples can be found in Appendix [A.](#page-10-2)

As shown in Figur[e3a,](#page-5-0) we compared different models under a 64× compression rate. Both RCC-64-mamba and RCC-64-transformer achieved a BLEU-4 score close to 0.82, but mamba's training time was nearly 1.5 times that of transformer. RCC-64-transformer-last-hidden-layer, which uses

<span id="page-5-0"></span>> **[图片提取文字 (无描述)]:**
> 1.0 1.0 0.9 0.9 0.8 0.8 0.7 0.7 0.6 0.6 BLUE-4 BLUE-4 0.5 0.5 RCC-32-transformer RCC-64-transformer RCC-64-mamba RCC-64-transformer 0.4 RCC-64-transformer-last-hidden-layer RCC-128-transformer 0.3 ICAE-64 0.3 0.2 0.2 0.1 0.1 X 0.0 -X X 0.0 200 400 100 300 500 100 200 300 400 500 length length (b) Scores at Different Compression Rates (a) Text Reconstruction Scores of Different Models.
![](_page_5_Figure_0.jpeg)

Figure 3: Text reconstruction score

only the encoder's last layer compression vectors, achieved a BLEU-4 score of approximately 0.6. This approach, common in traditional autoencoder models (Wang et al., 2021; Qin and Van Durme, 2023; Ge et al., 2023), retains less textual information compared to using compression vectors from all layers. Additionally, ICAE performed poorly under a 64× compression rate, with a BLEU-4 score of about 0.1, confirming our method's effectiveness in preserving text information.

As shown in Figure 3b, we tested reconstruction performance under different compression rates. At a 32× compression rate, the BLEU-4 score reached 0.95. At a 64× compression rate, the score dropped to between 0.8 and 0.85. At a 128× compression rate, we encountered convergence issues, preventing BLEU-4 score computation. This indicates that higher compression rates increase the difficulty of text reconstruction.

#### 4.2 Passkey Retrieval Task

We utilize the passkey retrieval task (Mohtashami and Jaggi, 2023) to validate the effectiveness of the two-stage training method mentioned in section 3.3.1. Additionally, we observe that our method exhibits certain length extrapolation capabilities, enabling it to handle compressed vector lengths during inference that far exceed those seen during training. This indicates that the compressed vectors generated by our method can be reliably recognized by the encoder, with minimal influence from positional encoding. Passkey retrieval task involves embedding a random number into a long sequence composed of repeated fixed short phrases, with the overall text length controlled by adjusting

the number of repetitions of these short phrases. The task requires the model to accurately retrieve the hidden number from these long sequences. Detailed construction methods for passkey retrieval task samples are provided in Appendix D. In this task, we employed a compression rate of 512x. Although this compression rate might not be effective for reconstruction tasks, experiments show that the fine-tuned RCC model performs well in the passkey retrieval task. The model was first pre-trained on a dataset containing only the random prompt text reconstruction task, with an encoder input length set to 8k and a non-compressed decoder input length of 512. After pre-training, we constructed nearly 30,000 passkey retrieval task samples with context lengths of 8k and 32k, respectively. These samples formed the fine-tuning dataset. We conducted a two-stage fine-tuning process. In the first stage, we fine-tuned the entire parameter set using the 8k context length samples. After completing the first stage, we proceeded to the second stage with the 32k context length samples. During this stage, we froze the encoder parameters to accommodate the constraints of limited available memory.

From Table 1, we can observe that even with the encoder using only an 8k context window, the model achieves almost 90% accuracy in passkey retrieval tasks up to 1000k, demonstrating the strong length extrapolation capabilities of our model. After the second stage of fine-tuning with sequences up to 32k, the model achieves nearly 100% performance on passkey retrieval tasks up to 1000k, proving the effectiveness of our two-stage training method, even with the encoder parameters frozen at this stage. Table 1 also shows that our method is

<span id="page-6-0"></span>

| Model                  | 32K       | 128K     | 256K                                       | 512K     | 1M                                                      |
|------------------------|-----------|----------|--------------------------------------------|----------|---------------------------------------------------------|
| RCC-Mamba-FT-8k        | 98/100/97 | 96/98/94 | 95/93/89                                   | 94/95/96 | 94/96/96A                                               |
| RCC-Transformer-FT-8k  | 97/95/96  | 96/97/96 | 92/96/96                                   | 92/89/95 | 97/96/96                                                |
| RCC-Mamba-FT-32k       |           |          |                                            |          | 100/100/100100/100/100100/100/100100/100/100100/99/100  |
| RCC-Transformer-FT-32k |           |          |                                            |          | 99/100/100 100/100/100100/100/10098/100/100 100/100/100 |
| Infini-Transformer-FT  |           |          | 100/100/100100/100/100100/100/10097/99/100 |          | 96/94/100                                               |

Table 1: The performance of the different models on the passkey retrieval tasks ranging from 32k to 1M sequence lengths, RCC-512-FT-8k denotes that the RCC model is trained with full parameters on a fine-tuning dataset with a length of 8k. RCC-512-FT-64K is trained on a fine-tuning dataset with a length of 64K based on RCC-512-FT-8k, while in this case, we freeze the encoder.

highly competitive compared to recent similar work like Infini-attention [\(Munkhdalai et al.,](#page-9-6) [2024\)](#page-9-6). Unlike Infini-attention, our method can be fine-tuned on existing open-source LLMs with a small amount of data, without requiring the reconstruction of the LLM model and pre-training with hundreds of billions of tokens.

### 4.3 Long-Text Benchmark Evaluation

#### 4.3.1 Evaluation Dataset

LongBench [\(Bai et al.,](#page-8-17) [2023\)](#page-8-17) is a benchmark designed to evaluate the capabilities of large language models in understanding long contexts. To assess our model's performance on texts of different lengths, we selected the LongBench-E set for evaluation because it evenly covers test samples of various length ranges, allowing us to analyze the impact of length variation on performance. Due to limitations in the fine-tuning dataset, our work focuses on using the single-document QA and multidocument QA tasks for evaluation. These two document QA tasks consist of four subtasks, with each task containing between 150 to 300 samples. Detailed information on the evaluation dataset can be found in Appendix [E.](#page-10-4)

### 4.3.2 Instruction Fine-Tuning

First, we conducted pretraining on the random prompt text reconstruction task and the text continuation task with a ratio of 1:9. Similar to the two-stage training method, the first stage involves training with full parameters on texts with a length of 2k. In the second stage, the encoder is frozen, and training is conducted on texts with a length of 16k. For question answering instruction fine-tuning, we used the Prompt-with-Context (PwC) [\(Ge et al.,](#page-8-6) [2023\)](#page-8-6) and hotpotQA [\(Yang et al.,](#page-10-5) [2018\)](#page-10-5) datasets. These datasets include context with instructions and outputs, teaching the model to use context for

answering questions rather than relying solely on internal knowledge. We concatenated the context and instructions as the encoder's input, while the instructions and output results formed the decoder's uncompressed input. We repeated the instructions twice to train the encoder to reconstruct instructions, enhancing mixed instruction and context text effectiveness during inference. The PwC dataset has 240k samples, and hotpotQA has 90k samples. Additionally, to improve instruction reconstruction and maintain the decoder's instruction-following capability, we randomly selected 50k instruction samples from the orcal dataset [\(Mukherjee et al.,](#page-9-15) [2023\)](#page-9-15). These samples lack explicit context fields and typically mix instructions with context. We input the instructions as context to the encoder and as instructions and outputs to the decoder. During fine-tuning, the encoder's input context length was set to 2048 tokens, the uncompressed part of the decoder's input was set to 512 tokens, and the compression rate remained 32.

### 4.3.3 Evaluation

We fine-tuned Pythia-1.4b with instruction pairs constructed from PwC, hotpotQA, and some ORCA data to ensure it follows instructions. Due to limitations in the fine-tuning dataset, we only selected document QA tasks. We evaluated the fine-tuned Pythia-1.4b and our model using Long-Bench's[\(Bai et al.,](#page-8-17) [2023\)](#page-8-17) automated evaluation tools, covering two document QA tasks as shown in Appendix [E.](#page-10-4) As shown in Table [2,](#page-7-0) the fine-tuned Pythia-1.4b significantly improved in following instructions. Notably, Pythia-1.4b supports a maximum sequence length of 2048 tokens, so we only used samples under 2k tokens for its evaluation. Our method supports LongBench's maximum input length of 15k tokens within the effective window length of the decoder. We further evaluated

<span id="page-7-0"></span>

| Method                 | 0-2k  | 2-4k  | 4-8k  | 8k+   | average |  |
|------------------------|-------|-------|-------|-------|---------|--|
| Pythia-SFT             | 30.54 | -     | -     | -     | -       |  |
| Pythia-No-SFT          | 4.41  | -     | -     | -     | -       |  |
| RCC-Ins-Reconstruction | 28.12 | 23.37 | 21.24 | 17.72 | 22.61   |  |
| RCC-Ins-Human          | 25.36 | 25.15 | 23.63 | 20.48 | 23.15   |  |
| RCC-Ins-Compress       | 18.77 | 21.36 | 20.02 | 18.14 | 19.61   |  |

Table 2: Scores of different models on the task of Document QA.

the following types for our method:

RCC-Ins-Reconstruction, which reconstructs instructions from compressed vectors and responds using instruction reconstruction techniques (Table 2), scored 28.12 at a length of 2k. This score is competitive with Pythia-sft, demonstrating that RCC can maintain high-quality inference even with a compression ratio of up to 32x. This method's average score surpasses that of RCC-Ins-Compress, which compresses both instructions and context, verifying the effectiveness of instruction reconstruction. Due to the fine-tuning dataset being limited to 2k tokens, RCC-Ins-Reconstruction performs poorly in instruction reconstruction when handling longer samples.

RCC-Ins-Human directly inputs real instruction texts into the decoder (Table 2). Compared to the performance fluctuations of RCC-Ins-Reconstruction with increasing sample length, RCC-Ins-Human exhibits more stable performance, especially maintaining efficient inference at lengths beyond 8k. We attribute this to the decline in instruction reconstruction quality in RCC-Ins-Reconstruction for long texts, whereas RCC-Ins-Human employs fixed instructions, unaffected by length.

RCC-Ins-Compress compresses both context and instructions simultaneously (Table 2). The encoder receives concatenated texts, and the decoder is only prompted with brief information, such as " n Answer:". This strategy's limited capability to recognize instructions and context results in an average score as low as 19.61, particularly underperforming compared to RCC-Ins-Human and RCC-Ins-Reconstruction in samples under 8k. However, for ultra-long samples (8k+), its performance converges with RCC-Ins-Reconstruction, likely due to the latter's deficiencies in instruction reconstruction at extreme lengths. Specific model generation results can be found in Appendix [E.](#page-10-4)

In Figure [1,](#page-1-0) we observe that when RCC pro-

cesses text up to 16k tokens, the GPU memory usage only increases by approximately 0.5 GB. In contrast, the original Pythia-1.4b experiences a 2 GB increase in memory usage for 2k token text. When processing 16k token text, Pythia-1.4b's total memory usage will be twice that of our method. Since we use a compression rate of 32x, as text length increases, RCC can save up to nearly 32x in storage space. Although the parameter count of RCC's encoder and decoder is twice that of Pythia, the impact of the model's parameter count on storage space significantly diminishes with increased text length.

### 5 Conclusion

Utilizing compression techniques to mitigate challenges in long-text training and inference has proven to be a highly promising strategy. Our work quantitatively analyzes the impact of different context lengths on compression performance, while also achieving higher compression rates than previous methods, thereby significantly enhancing the ability of large language models (LLMs) to handle long texts. The RCC method demonstrated outstanding performance across multiple test tasks, particularly in context compression reconstruction, long-document question answering, and keycontext block retrieval tasks with sequences up to 1 million tokens. Additionally, we analyzed the issues arising from simultaneous compression of context and instructions and introduced an instruction reconstruction method that effectively alleviated these problems. Furthermore, to address the substantial resource consumption of long-text training, we proposed a staged training strategy that further improved the efficiency of the model in handling long-text training.

### 6 Limitations

The RCC method has significantly advanced text compression efficiency and long-document question answering, but it has limitations. For example, RCC risks errors during instruction reconstruction, and if instructions are too long, the decoder may struggle to reconstruct them within the limited window. In future research, we plan to adopt a hybrid training approach: using instruction compression for long instructions and instruction reconstruction for short ones to achieve results comparable to manually input instructions.Additionally, the lack of long-text instruction fine-tuning data has caused performance bottlenecks for RCC. Our experiments show the critical impact of training data on the model's performance. The effectiveness of language models fine-tuned with instructions depends largely on the quality and coverage of those instructions. These issues provide clear directions for our future research.

## References

- <span id="page-8-17"></span>Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint arXiv:2308.14508*.
- <span id="page-8-4"></span>Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. [Longformer: The long-document transformer.](https://arxiv.org/abs/2004.05150) *Preprint*, arXiv:2004.05150.
- <span id="page-8-11"></span>Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. [Pythia:](https://arxiv.org/abs/2304.01373) [A suite for analyzing large language models across](https://arxiv.org/abs/2304.01373) [training and scaling.](https://arxiv.org/abs/2304.01373) *Preprint*, arXiv:2304.01373.
- <span id="page-8-2"></span>Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. 2022. Recurrent memory transformer. In *Advances in Neural Information Processing Systems*, volume 35, pages 11079–11091. Curran Associates, Inc.
- <span id="page-8-14"></span>Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. [Extending context window of](https://arxiv.org/abs/2306.15595) [large language models via positional interpolation.](https://arxiv.org/abs/2306.15595) *Preprint*, arXiv:2306.15595.
- <span id="page-8-13"></span>Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2024. [Longlora:](https://arxiv.org/abs/2309.12307) [Efficient fine-tuning of long-context large language](https://arxiv.org/abs/2309.12307) [models.](https://arxiv.org/abs/2309.12307) *Preprint*, arXiv:2309.12307.
- <span id="page-8-5"></span>Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 3829–3846, Singapore. Association for Computational Linguistics.

- <span id="page-8-1"></span>Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019. [Generating long sequences with](https://arxiv.org/abs/1904.10509) [sparse transformers.](https://arxiv.org/abs/1904.10509) *Preprint*, arXiv:1904.10509.
- <span id="page-8-8"></span>Tri Dao. 2023. [Flashattention-2: Faster attention with](https://arxiv.org/abs/2307.08691) [better parallelism and work partitioning.](https://arxiv.org/abs/2307.08691) *Preprint*, arXiv:2307.08691.
- <span id="page-8-16"></span>Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Mohamed Osama Ahmed, Yoshua Bengio, and Greg Mori. 2024. [Attention as an rnn.](https://arxiv.org/abs/2405.13956) *Preprint*, arXiv:2405.13956.
- <span id="page-8-18"></span>Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. [The pile: An](https://arxiv.org/abs/2101.00027) [800gb dataset of diverse text for language modeling.](https://arxiv.org/abs/2101.00027) *Preprint*, arXiv:2101.00027.
- <span id="page-8-6"></span>Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and Furu Wei. 2023. In-context autoencoder for context compression in a large language model. In *International Conference on Learning Representations*.
- <span id="page-8-10"></span>Albert Gu and Tri Dao. 2023. [Mamba: Linear](https://arxiv.org/abs/2312.00752)[time sequence modeling with selective state spaces.](https://arxiv.org/abs/2312.00752) *Preprint*, arXiv:2312.00752.
- <span id="page-8-0"></span>Sepp Hochreiter and Jürgen Schmidhuber. 1997. [Long](https://arxiv.org/abs/https://direct.mit.edu/neco/article-pdf/9/8/1735/813796/neco.1997.9.8.1735.pdf) [short-term memory.](https://arxiv.org/abs/https://direct.mit.edu/neco/article-pdf/9/8/1735/813796/neco.1997.9.8.1735.pdf) *Neural Computation*, 9(8):1735– 1780.
- <span id="page-8-12"></span>Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023a. [Mistral 7b.](https://arxiv.org/abs/2310.06825) *Preprint*, arXiv:2310.06825.
- <span id="page-8-9"></span>Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023b. [Longllmlingua: Accelerating and enhancing llms](https://arxiv.org/abs/2310.06839) [in long context scenarios via prompt compression.](https://arxiv.org/abs/2310.06839) *Preprint*, arXiv:2310.06839.
- <span id="page-8-7"></span>Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023. [Compressing context to enhance inference ef](https://doi.org/10.18653/v1/2023.emnlp-main.391)[ficiency of large language models.](https://doi.org/10.18653/v1/2023.emnlp-main.391) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 6342–6353, Singapore. Association for Computational Linguistics.
- <span id="page-8-15"></span>Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. 2024. [World model on million-length video and](https://arxiv.org/abs/2402.08268) [language with blockwise ringattention.](https://arxiv.org/abs/2402.08268) *Preprint*, arXiv:2402.08268.
- <span id="page-8-3"></span>Hao Liu, Matei Zaharia, and Pieter Abbeel. 2023. [Ring attention with blockwise transformers for near](https://arxiv.org/abs/2310.01889)[infinite context.](https://arxiv.org/abs/2310.01889) *Preprint*, arXiv:2310.01889.

- <span id="page-9-2"></span>Amirkeivan Mohtashami and Martin Jaggi. 2023. Random-access infinite context length for transformers. In *Advances in Neural Information Processing Systems*, volume 36, pages 54567–54585. Curran Associates, Inc.
- <span id="page-9-5"></span>Jesse Mu, Xiang Li, and Noah Goodman. 2023. Learning to compress prompts with gist tokens. In *Advances in Neural Information Processing Systems*, volume 36, pages 19327–19352. Curran Associates, Inc.
- <span id="page-9-15"></span>Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. [Orca: Progressive learning from](https://arxiv.org/abs/2306.02707) [complex explanation traces of gpt-4.](https://arxiv.org/abs/2306.02707) *Preprint*, arXiv:2306.02707.
- <span id="page-9-6"></span>Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. 2024. [Leave no context behind:](https://arxiv.org/abs/2404.07143) [Efficient infinite context transformers with infini](https://arxiv.org/abs/2404.07143)[attention.](https://arxiv.org/abs/2404.07143) *Preprint*, arXiv:2404.07143.
- <span id="page-9-11"></span>Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. 2023. [Rwkv: Reinventing rnns](https://arxiv.org/abs/2305.13048) [for the transformer era.](https://arxiv.org/abs/2305.13048) *Preprint*, arXiv:2305.13048.
- <span id="page-9-9"></span>Guanghui Qin and Benjamin Van Durme. 2023. Nugget: neural agglomerative embeddings of text. In *Proceedings of the 40th International Conference on Machine Learning*, ICML'23. JMLR.org.
- <span id="page-9-1"></span>Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, and Timothy P. Lillicrap. 2019. [Compressive trans](https://arxiv.org/abs/1911.05507)[formers for long-range sequence modelling.](https://arxiv.org/abs/1911.05507) *Preprint*, arXiv:1911.05507.
- <span id="page-9-7"></span>Siyu Ren, Qi Jia, and Kenny Zhu. 2023. [Context com](https://doi.org/10.18653/v1/2023.emnlp-main.794)[pression for auto-regressive transformers with sen](https://doi.org/10.18653/v1/2023.emnlp-main.794)[tinel tokens.](https://doi.org/10.18653/v1/2023.emnlp-main.794) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 12860–12867, Singapore. Association for Computational Linguistics.
- <span id="page-9-3"></span>Charlie Snell, Dan Klein, and Ruiqi Zhong. 2022. Learning by distilling context. In *International Conference on Learning Representations*.
- <span id="page-9-10"></span>Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. 2023. [Retentive network: A successor to](https://arxiv.org/abs/2307.08621) [transformer for large language models.](https://arxiv.org/abs/2307.08621) *Preprint*, arXiv:2307.08621.
- <span id="page-9-12"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal

- Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. [Llama: Open](https://arxiv.org/abs/2302.13971) [and efficient foundation language models.](https://arxiv.org/abs/2302.13971) *Preprint*, arXiv:2302.13971.
- <span id="page-9-13"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. [Llama 2: Open foundation and](https://arxiv.org/abs/2307.09288) [fine-tuned chat models.](https://arxiv.org/abs/2307.09288) *Preprint*, arXiv:2307.09288.
- <span id="page-9-14"></span>Szymon Tworkowski, Konrad Staniszewski, Mikoł aj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Mił os. 2023. Focused transformer: Contrastive train- ´ ing for context scaling. In *Advances in Neural Information Processing Systems*, volume 36, pages 42661– 42688. Curran Associates, Inc.
- <span id="page-9-8"></span>Kexin Wang, Nils Reimers, and Iryna Gurevych. 2021. [TSDAE: Using transformer-based sequential denois](https://doi.org/10.18653/v1/2021.findings-emnlp.59)[ing auto-encoderfor unsupervised sentence embed](https://doi.org/10.18653/v1/2021.findings-emnlp.59)[ding learning.](https://doi.org/10.18653/v1/2021.findings-emnlp.59) In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 671–688, Punta Cana, Dominican Republic. Association for Computational Linguistics.
- <span id="page-9-4"></span>David Wingate, Mohammad Shoeybi, and Taylor Sorensen. 2022. Prompt compression and contrastive conditioning for controllability and toxicity reduction in language models. In *Conference on Empirical Methods in Natural Language Processing*.
- <span id="page-9-16"></span>Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. [Transform](https://www.aclweb.org/anthology/2020.emnlp-demos.6)[ers: State-of-the-art natural language processing.](https://www.aclweb.org/anthology/2020.emnlp-demos.6) In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online. Association for Computational Linguistics.
- <span id="page-9-0"></span>Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, and Christian Szegedy. 2022. Memorizing transformers.

In *International Conference on Learning Representations*.

<span id="page-10-5"></span>Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. [HotpotQA: A dataset for](https://doi.org/10.18653/v1/D18-1259) [diverse, explainable multi-hop question answering.](https://doi.org/10.18653/v1/D18-1259) In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

### <span id="page-10-2"></span>A Effects of Text Reconstruction

The example of our method's reconstruction effect at 32x compression rate is shown below. As the table [3](#page-11-0) indicates, our method has almost completely reconstructed the context.

## <span id="page-10-0"></span>B GPU Memory Consumption Analysis

We ran the model on an A800 GPU using Hugging-Face's Transformers library [\(Wolf et al.,](#page-9-16) [2020\)](#page-9-16) and tested the GPU memory consumption of different models. As shown in Figure [4,](#page-12-0) the GPU memory usage of Pythia-1.4b increases rapidly with the length of the input context. When the context window reaches 64k, the model's GPU memory usage exceeds 60GB. RCC-1.4 x 2b, where both the encoder and decoder are Pythia-1.4b models with a compression rate of 32, shows that when its GPU memory usage exceeds 60GB, it processes a context length close to 2048k tokens. This is 30 times the length Pythia-1.4b can handle, nearly matching the compression rate.

## <span id="page-10-1"></span>C Random Prompt Text Reconstruction Tasks

The random prompt text reconstruction tasks involves an original text sequence (w1, . . . , wn), where the encoder compresses the entire original text and produces a compressed hidden vector (H). The decoder then needs to reconstruct the text that follows the random prompt word in the original text based on the encoder's input vector and the random prompt word from a segment of the original text. The prompt word is a substring of the original text (w<sup>i</sup> , . . . , w<sup>j</sup> ), denoted as p, and the target sentence to be reconstructed, which follows the prompt word, is (w<sup>j</sup> , . . . , wx), denoted as c.

$$\mathcal{L}_{\text{RAE}} = \max_{h,\dots,p} P\left(\boldsymbol{c} \mid h,\dots,p;\Theta_{LLM}\right)$$

In the text continuation task, the prompt is no longer a substring of the encoder's input text but is the immediately following segment of text, and the target sentence is still the text that comes right after the prompt. The formula for the text continuation task is the same as that for the random prompt text reconstruction task.

### <span id="page-10-3"></span>D Format of Passkey Retrieval

We follow the text format for passkey retrieval from existing works [\(Chen et al.,](#page-8-13) [2024;](#page-8-13) [Mohtashami and](#page-9-2) [Jaggi,](#page-9-2) [2023\)](#page-9-2). The format of the document is as follows:

There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.

The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat M times)

The pass key is **56994**. Remember it. **56994** is the pass key. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat N times)

What is the pass key? The pass key is

## <span id="page-10-4"></span>E Fine-Tuning Datasets and Model-Generated Cases

Table [4](#page-11-1) displays information such as the sources, average lengths, and computational metrics for various tasks. Below is a sample data entry for document question answering, primarily consisting of three parts: '*input*', '*context*', and '*answers*'. The '*input*' represents the prompt or instruction, the '*context*' is the surrounding text the model needs to search through, which is often lengthy, and the '*answers*' represent the possible answers derived from the context. Example:

input: "Which park is further south within Spain, Picos de Europa National Park or Timanfaya National Park?"

context: 'Passage 1:Lake Ercina Lake Ercina is a small highland lake . . . The population is 47 (INE 2016).'

answers: ['Timanfaya National Park']

When using RCC-Ins-Reconstruction for instruction reconstruction inference, we concatenate the '*context*' and '*input*' parts of the sample with a newline character and input them into the model's

<span id="page-11-0"></span>The Access nodes and storage daemons make up a data plane, while the core provides its control plane. Also: How IBM Watson is revolutionizing 10 industries TechRepublic So, what does all mean for customers? Its multi-cloud ´ storage management, which enables allows you to manage, deploy, and migrate data storage across private and major public clouds. This includes Alibaba, AWS, Azure, and Google Cloud. Its easy to see why Red Hat values this. ´ It gives their customers a way to manage storage without sweating the details across multiple platforms. As Ranga Rangachari, Red Hats vice ´ president of Storage and Hyperconverged Infrastructure, said in a statement: "Data portability is a key imperative for organizations building and deploying cloud-native applications across private and multiple clouds. NooBaas technolo- ´ gies will augment our portfolio and strengthen our ability to meet the needs of developers in todays hybrid and multicloud world. We are ´ thrilled to welcome a technical team of nine to the Red Hat family as we work together to further solidify Red Hat as a leading provider of open hybrid-cloud technologies." Related stories: Kidderminster-based Renault UK Clio Cup ace Dan Rowbottom will join Ciceley Motorsport for the 2019 British Touring Car Championship. Backed by Cataclean, the lead valuable additive to clean and fuel engine restore and exhaust systems, Rowbottom will graduate from the Renault UK Clio Cup into one of Ciceley's Mercedes-Benz A-Class cars for the forthcoming campaign. He was a triple race winner last season his way to fourth place

The Access nodes and storage daemons make up a data plane, while the core provides its control plane. Also: How IBM Watson is revolutionizing 10 industries TechRepublic So, what does all mean for customers? Its multi-cloud ´ storage management, which enables allows you to manage, deploy, and migrate data storage across private and major public clouds. This includes Alibaba, AWS, Azure, and Google Cloud. Its easy to see why Red Hat values this. ´ It gives their customers a way to manage storage without sweating the details across multiple platforms. As Ranga Rangachari, Red Hats vice ´ president of Storage and Hyperconverged Infrastructure, said in a statement: "Data portability is a key imperative for organizations building and deploying cloud-native applications across private and multiple clouds. NooBaas technolo- ´ gies will augment our portfolio and strengthen our ability to meet the needs of developers in todays hybrid and multicloud world. We are ´ thrilled to welcome a technical team of nine to the Red Hat family as we work together to further solidify Red Hat as a leading provider of open hybrid-cloud technologies." Related stories: Kidderminster-based Renault UK Clio Cup ace Dan Rowbottom will join Ciceley Motorsport for the 2019 British Touring Car Championship. Backed by Cataclean, the leading fuel additive to clean and restore engine fuel and exhaust systems, Rowbottom will graduate from the Renault UK Clio Cup into one of Ciceley's Mercedes-Benz A-Class cars for the forthcoming campaign. He was a triple race winner last season his way to fourth place

Table 3: Effect of random prompt text reconstruction

<span id="page-11-1"></span>

| Dataset         | Task               | Source      | Avg len | Metric | #data |
|-----------------|--------------------|-------------|---------|--------|-------|
| Qasper          | Single-Document QA | Science     | 4,620   | F1     | 224   |
| MultiFieldQA    | Single-Document QA | Multi-field | 4,558   | F1     | 150   |
| HotpotQA        | Multi-Doc QA       | Wikipedia   | 6,657   | F1     | 300   |
| 2WikiMultihopQA | Multi-Doc QA       | Wikipedia   | 6,146   | F1     | 300   |

Table 4: LongBench-E Information

<span id="page-12-0"></span>> **[图片提取文字 (无描述)]:**
> RCC-1.4x2b - 0-16k Pythia-1.4b - 0-16k Pythia-1.4b 0-16k RCC-1.4x2b - 0-16k 15000 (MB) 12500 (MB) 7500 (MB) 7500 15000 - (MB) 12500 - (MB) 10000 - 7500 - (MB) Context Length (k) Context Length (k) Pythia-1.4b - 8-64k RCC-1.4x2b - 64-2048k Pythia-1.4b - 8-64k RCC-1.4x2b - 64-2048k Memory Usage (MB) 40000 200000 Memory Usage (MB) 40000 200000 200000 10000 -Context Length (k) Context Length (k)
![](_page_12_Figure_0.jpeg)

Figure 4: When the GPU memory approaches 60GB, the memory occupation of different models. Left: Pythia-1.4b, Right: RCC model using Pythia-1.4b for both encoder and decoder. Both models utilize FlashAttention-2 (Dao, 2023).

encoder for compression. Simultaneously, the decoder's input is a fixed prompt:

prompt: "system: You are a helpful
assistant. user: "

The decoder, starting with this prompt, first reconstructs the instruction and then answers the question based on it. The content generated by the model is shown in blue font:

"system: You are a helpful assistant. user: Which park is further south within Spain, Picos de Europa National Park or Timanfaya National Park? assistant: Timanfaya National Park"

The model accurately reconstructed the instruction and provided the correct answer, '*Timanfaya National Park*'.

Additionally, we tested the RCC-Ins-compress model. The input to the RCC-Ins-compress encoder is identical to that of the RCC-Ins-Reconstruction, but the decoder's prompt is:

Prompt: "Response of system:"

Since RCC-Ins-compress has not been trained on instruction reconstruction tasks, it does not reconstruct the instruction in its output. Instead, it directly answers the question based on the mixed compressed context and instruction, which may result in the model failing to follow the instruction. The content generated by the model is shown in blue font:

"Response of system: Panic of 1797"

It can be seen that the model made an error in following the instructions.