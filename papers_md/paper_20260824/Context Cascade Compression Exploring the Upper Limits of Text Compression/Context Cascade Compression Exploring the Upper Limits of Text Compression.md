# Context Cascade Compression: Exploring the Upper Limits of Text Compression

#### Fanfan Liu, Haibo Qiu

# Abstract

Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression (C3) to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at [https://github.com/liufanfanlff/C3-](https://github.com/liufanfanlff/C3-Context-Cascade-Compression) [Context-Cascade-Compression](https://github.com/liufanfanlff/C3-Context-Cascade-Compression)

> **[图片提取文字 (无描述)]:**
> deepseek-ocr 64 toks deepseek-ocr 100 toks c3 32 toks c3 64 toks c3 100 toks -O- 32 toks -O- 64 toks -O- 100 toks 45x 99.7% 99.7% 99.7% 99.5% 98.8% 98.8% 98.7% 98.6% 100% 98.5% 98.4% 97.5% 39.4 96.8% 96.8% 40x 94.0% 93.3% 91.5% 89.8% 35x 90% 87.1% 83.8% <sup>1</sup> 30x € Precision (%) 25x 25x 20x dwo 20x dwo 15x 76.3% 19.7 60% 59.1% 10x 50% -5x 40% Text Tokens in Per Page (Ground-truth)
![](_page_0_Figure_5.jpeg)

Figure 1: Reconstruction precision and compression ratio of C3 versus Deepseek-OCR on the Fox benchmark. Bars represent precision (left axis), while lines indicate the compression ratio (right axis). C3 (solid bars) demonstrates significantly higher precision than Deepseek-OCR (striped bars) across all tested latent token counts (32, 64, and 100).

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> .....We present DeepSeek-OCR as an initial investigation into the feasibility of compressing longcontexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens. Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio < 10x), the model can achieve decoding (OCR) precision of 97%. Even at a compression ratio of 20x, the OCR accuracy still remains at about 60%. This shows considerable promise for research areas such as historical long-context compression and memory forgetting mechanisms in LLMs. Beyond this, DeepSeek-OCR also demonstrates high practical value.... long text small Ilm vit image plain text 40x compression 10x compression text tokens image latent tokens text latent tokens ~1.2k ~128 ~32 Original Contexts Optical Compression C3 Compression
![](_page_1_Figure_0.jpeg)

Figure 2: Conceptual comparison of context compression pipelines. The figure contrasts three methods: (1) Original: The baseline tokenization of plain text, leading to a large token count (~1.2k). (2) Contexts Optical Compression: An indirect method where text is converted to an image and encoded by a ViT, achieving ~10x compression into ~128 latent tokens. (3) C3 Compression: Our proposed direct method, where a small LLM compresses the text into a minimal set of ~32 latent tokens, achieving ~40x compression.

# 1 Introduction

Recently, Large Language Models (LLMs) [\[OpenAI,](#page-9-0) [2023\]](#page-9-0) have achieved significant advancements in various fields. As the capabilities of LLMs continue to grow and their application scenarios expand, the computational and memory challenges they face in long-context scenarios have become increasingly evident. Recent research has explored several effective methods to alleviate this issue. An approach is to modify the attention mechanism, for example, using sparse [\[DeepSeek-AI,](#page-9-1) [2025\]](#page-9-1) or linear attention [\[QwenTeam,](#page-9-2) [2025\]](#page-9-2), to reduce the quadratic complexity of self-attention and improve computational efficiency. However, as the context length grows to hundreds of thousands of tokens, the overall overhead remains substantial because the number of tokens is unchanged. Alternatively, retrieval-augmented methods [\[Lewis et al.,](#page-9-3) [2021\]](#page-9-3) shorten the input length through external retrieval, but this is essentially a form of lossy information compression and introduces additional latency. Another novel approach uses visual modality as an efficient medium for compressing textual information [\[Wei et al.,](#page-9-4) [2025\]](#page-9-4); an image containing a document's text can represent the information with far fewer tokens than the text itself. Based on this insight, DeepSeek-OCR [\[Wei et al.,](#page-9-4) [2025\]](#page-9-4) employs Optical Character Recognition (OCR) as an intermediate modality to bridge vision and language. It deeply explores this compression paradigm, achieving a 10x text token compression rate while maintaining a 97% decoding accuracy.

We revisit the entire pipeline of DeepSeek-OCR and find that when using the visual modality as a compression medium for text, as shown in figure [2,](#page-1-0) the LLM process shifts from T ext → T extT okens → LLM to T ext → Image → V isualLatentT okens → LLM. In this new pipeline, we hypothesize that information compression may not stem primarily from T ext → Image conversion, but rather from the fact that latent tokens are a more efficient information representation compared to discrete text tokens. Based on this hypothesis, we propose a novel context compression paradigm: Context Cascade Compression (C3). In C3, we directly cascade two LLMs of different sizes. The first, smaller LLM compresses discrete text tokens into textual latent tokens, functioning similarly to a visual encoder in a VLM. The second, larger LLM then performs the downstream task. (For this paper, we focus on information preservation, tasking the model to reconstruct the original text from the compressed latent tokens, a process equivalent to the OCR task of recognizing text from visual latent tokens). Compared to DeepSeek-OCR, C3 employs a more streamlined pipeline for pure text compression, as it disregards other factors in optical character compression, such as layout, color, and information loss from the visual encoder. Consequently, C3 achieves a compression rate

far exceeding that of optical methods while maintaining the same decoding accuracy. It also provides a potential theoretical upper bound for the ratio of text tokens to visual tokens in future work on optical character compression and OCR.

In summary, our main contributions are as follows:

- We propose Context Cascade Compression (C3), a novel architecture to achieve highly efficient compression of long contexts.
- We achieve a performance that far surpasses optical character compression. At a 40x compression ratio (where the number of text tokens is 40 times the number of latent tokens), our model maintains a decoding accuracy of 93%, compared to approximately 10x for DeepSeek-OCR. Our architecture offers a more efficient and feasible approach for future ultra-long context inputs.
- We analyze the forgetting pattern of C3 and find that it exhibits sequential information loss, with errors tending to concentrate at the end of the text. This process is highly analogous to the decay of human memory.

In conclusion, our work provides a preliminary exploration into the upper limits of text compression rates and demonstrates that direct text compression is a more efficient method than optical character compression. Moreover, our method presents a viable solution for handling future ultra-long context inputs. From another perspective, our architecture can also serve as an autoencoder structure for diffusion language models and latent auto-regressive models, enabling the conversion of variablelength contexts into fixed-length latent tokens.

# 2 Related Work

In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities. However, their core component, the self-attention mechanism, exhibits quadratic computational and memory complexity with respect to the input sequence length. This has severely limited the context length that models can effectively handle. To overcome this bottleneck, both academia and industry have explored various techniques for long-context processing and compression. Our work is primarily related to context compression, with a specific focus on the emerging field of contexts optical compression.

Contexts Compression. Handling long contexts is a frontier in current LLM research. Existing methods can be broadly categorized as follows:

Efficient Attention Mechanisms: A significant body of research focuses on optimizing the attention mechanism at an architectural level to reduce its quadratic complexity. For instance, sparse attention methods, such as Longformer [\[Beltagy et al.,](#page-9-5) [2020\]](#page-9-5) and BigBird [\[Zaheer et al.,](#page-9-6) [2020\]](#page-9-6), reduce computational load by limiting the attention scope of each token. Other approaches, like Linear Attention [\[Katharopoulos et al.,](#page-9-7) [2020\]](#page-9-7) and State-Space Models (e.g., Mamba [\[Dao and Gu,](#page-9-8) [2023\]](#page-9-8)), attempt to approximate the performance of standard attention with linear or near-linear complexity. Although effective, these methods often require fundamental architectural modifications and may trade off some performance on certain tasks.

Retrieval-Augmented Generation: RAG [\[Lewis et al.,](#page-9-3) [2021\]](#page-9-3) adopts a different strategy by storing long contexts in an external knowledge base (e.g., a vector database) and retrieving only the most relevant snippets to inject into the model's input at inference time. This approach excels in informationretrieval tasks but is inherently a "loss" compression method, as it relies on the accuracy of the retrieval step and may lose global context or inter-snippet relationships.

Pluggable Module: Drawing parallels to modules like Q-Former in VLMs, some prior work has also explored prepending a carefully designed external module to reduce the length of input tokens for the main LLM. Notable examples include [\[Ge et al.,](#page-9-9) [2023,](#page-9-9) [Wang et al.,](#page-9-10) [2024\]](#page-9-10). However, these modules have generally demonstrated severely limited compression performance.

Memory and Summarization: Other works have attempted to compress context by generating intermediate summaries or memory tokens. For example, some models periodically generate summaries of long texts to represent historical information [\[Bulatov et al.,](#page-9-11) [2022\]](#page-9-11). Our work, C3, shares a philosophy similar with these methods in that it generates a compact representation of the context.

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Output Long Context LLM Latent Context prompt: "Repeat the text: " Contexts Compression LLM Input Long Context Context Query
![](_page_3_Figure_0.jpeg)

Figure 3: An overview of the C3, which utilizes a cascaded two-LLM design. A smaller encoder LLM compresses a variable-length Input Long Context into a fixed-length Latent Context guided by learnable Context Query tokens. Subsequently, a larger decoder LLM uses this compact Latent Context and a prompt to perform the downstream task, such as reconstructing the original text.

However, C3 is distinct in its approach: rather than generating a human-readable text summary, we train a dedicated encoder LLM to directly distill textual information into a fixed-length set of noninterpretable latent tokens. This end-to-end, latent-space compression paradigm potentially preserves richer semantic information and provides a more direct and efficient input for the downstream decoder compared to generative summarization.

Contexts Optical Compression. Using the visual modality as a compression medium for text is a novel and promising research direction, which we term "Contexts Optical Compression." [\[Wei et al.,](#page-9-4) [2025\]](#page-9-4)

The core idea is to render long text into one or more images and then leverage a powerful visual encoder to compress the high-dimensional pixel information into a series of visual tokens. The theoretical basis for this approach is that visual encoders have demonstrated exceptional ability in extracting dense features from complex images, and a text-rendered image can be viewed as a highly structured visual signal.

The most recent representative work in this area are DeepSeek-OCR [\[Wei et al.,](#page-9-4) [2025\]](#page-9-4), Glyph [\[Cheng](#page-9-12) [et al.,](#page-9-12) [2025\]](#page-9-12). This model employs the OCR task as a bridge between vision and language, compressing long text by rendering it into images. It achieves a token compression rate of up to 10x while maintaining high decoding accuracy, proving the feasibility of the optical compression pathway. Our work, C3, offers new insights and improvements upon this paradigm.

# 3 Method

## 3.1 Architecture

As shown in Figure [3,](#page-3-0) C3 employs a cascaded two-LLM architecture, comprising a context compression encoder LLM and a decoder LLM. The context compression encoder is responsible for information compression, transforming text tokens into latent tokens. The decoder then utilizes these latent tokens and a given prompt to generate the desired output. Specifically, we use Qwen2.5 1.5B [\[Qwen et al.,](#page-9-13) [2025\]](#page-9-13) as the context compression encoder and Qwen2.5 3B [\[Qwen et al.,](#page-9-13) [2025\]](#page-9-13) as the decoder. In the following sections, we will delve into the details of the model components, data engineering, and training methodology.

### 3.2 Context Compression Encoder LLM

In this section, we detail the architecture of our text compression encoder. The primary objective is to formulate a component that is both architecturally concise and computationally efficient for compressing long textual sequences. We posit that modern, large-scale, pre-trained language models, by virtue of their extensive training, have already developed a sophisticated capacity for information extraction, semantic understanding, and summarization. Capitalizing on this inherent capability, we eschew designing a compression module from scratch and instead directly adapt a pre-trained LLM for this purpose.

Specifically, the architectural backbone of our context compression encoder is initialized using the weights of a pre-trained Qwen2.5 1.5B model. To facilitate the compression, we introduce a set of learnable embeddings designated as the context query. This query is materialized as a trainable tensor Q of dimensions N × D, where N represents the fixed number of tokens desired for the output latent context, and D corresponds to the hidden dimension of the Qwen2.5 1.5B model's embedding layer.

The input sequence fed to the encoder is a concatenation of the original long context (as text tokens) and context query embeddings. Crucially, the model processes this hybrid sequence uniformly. The context query embeddings are treated identically to standard text tokens within the model's self-attention mechanism. No architectural modifications, such as introducing cross-attention layers, are required. The entire forward pass relies solely on the model's native causal attention mechanism. This design choice ensures simplicity and leverages the full expressive power of the pre-trained Transformer architecture.Upon completion of the forward pass, the final layer's output hidden states corresponding to the positions of the context query tokens are extracted. This resulting tensor, of shape N × D, serves as an efficient and dense representation of the original text. This output constitutes the latent context, which is then passed to the downstream decoder.

#### 3.3 Decoder LLM

Following the established paradigm in advanced generative models, such as Vision-Language Models (VLMs) and recent OCR systems [\[Wei et al.,](#page-9-14) [2024\]](#page-9-14), we employ a Large Language Model (LLM) as the decoder component. The primary function of this decoder is to interpret the dense, compressed latent context provided by the encoder and to generate a coherent textual output that fulfills a specified downstream task.

For the scope of this paper, our investigation is focused exclusively on the fundamental task of text reconstruction. This task serves as a direct and rigorous benchmark for evaluating the information fidelity of our C3 compression architecture. By tasking the model with perfectly recreating the original input text from its compressed representation, we can quantitatively measure the degree of information preserved throughout the compression-decompression cycle. This setup provides a clear, objective measure of the compression's "lossiness" and establishes a baseline for the model's capabilities.

Operationally, the input to the decoder is a concatenated sequence comprising the latent context and a task-specific prompt. When performing the reconstruction task, we use the explicit instruction *"repeat the text: "*. The decoder is then trained to auto-regressively generate a token sequence that is identical to the original, ground-truth text, thereby demonstrating that the semantic integrity of the input has been successfully maintained within the latent context.

# 4 Experiment

# 4.1 Data

We trained our model using ground truth from OCR data. We collected 1 million pages of diverse OCR data from the internet, primarily in English and Chinese. The dataset is predominantly composed of document-related data. Notably, our approach demonstrates considerable robustness with respect to data preparation. We found that extensive data engineering or complex curation was not necessary. A straightforward aggregation of text samples of diverse lengths proved sufficient to train the model effectively, indicating that our proposed architecture is inherently resilient to variations in input length and does not depend on meticulous data preprocessing to achieve strong performance.

#### 4.2 Training Setup

The model was trained on a high-performance computing cluster equipped with 8 NVIDIA H800 GPUs. We configured the training with a per-device batch size of 2 and employed 16 gradient accumulation steps. This configuration resulted in an effective global batch size of 256 (2 per device × 8 GPUs × 16 accumulation steps). The model was optimized using AdamW [\[Loshchilov and Hutter,](#page-9-15) [2019\]](#page-9-15) optimizer with a peak learning rate of 1e-5. A cosine [\[Loshchilov and Hutter,](#page-9-16) [2017\]](#page-9-16) learning rate scheduler was employed to anneal the learning rate over the course of training, preceded by a linear warmup [\[Vaswani et al.,](#page-10-0) [2017\]](#page-10-0) phase over the initial 100 steps. The entire training process was conducted for a total of 40,000 steps.

#### 4.3 Context Compression Study

#### 4.3.1 Quantitative

To rigorously evaluate the compression and reconstruction fidelity of our C3 architecture, we follow the experimental protocol established by DeepSeek-OCR , thereby enabling a direct and fair comparison of performance. We adopt the Fox [\[Liu et al.,](#page-10-1) [2024\]](#page-10-1) benchmark, to assess our model's capabilities. Specifically, our evaluation on the English document portion of the Fox benchmark. The ground-truth texts are tokenized using the official Qwen tokenizer, and for our test set, we select documents with token counts ranging from 600 to 1300 tokens. This selection ensures a consistent basis for comparing the performance of our text-to-latent paradigm against the optical compression paradigm. While minor discrepancies in token counts may arise due to differences between our tokenizer and the one used in the original work, we posit that this effect on the final results is negligible. In our experiments, we benchmark performance at multiple compression levels. To align with the baseline, we evaluate settings where the contexts are compressed into 64 and 100 latent tokens. Furthermore, to explore the upper limits of our compression method, we introduce a more aggressive setting, evaluating the performance with a highly compressed representation of only 32 latent tokens. Across all evaluations, the text reconstruction task is initiated using the prompt *"repeat the text: "*.

The experimental results, presented in Table [1,](#page-6-0) provide a comprehensive comparison between our C3 architecture and the Deepseek-OCR baseline on the Fox benchmark. The data unequivocally demonstrates that C3's direct text-to-latent compression paradigm significantly outperforms the optical compression approach across all tested conditions, establishing a new state-of-the-art in highfidelity context compression. A detailed analysis of the results reveals several key insights. In the Vision Tokens = 100 setting, where Deepseek-OCR achieves its peak performance of approximately 97% precision at a 7-10x compression ratio, our C3 model consistently maintains a precision of over 99.5% under similar compression rates. This indicates that even at moderate compression levels, our direct approach offers superior information preservation. The performance gap widens dramatically at higher compression ratios, particularly in the more aggressive Vision Tokens = 64 setting. While Deepseek-OCR's performance begins to degrade notably as the compression ratio exceeds 10x (dropping to 83.8% at 13.2x), C3 sustains near-perfect reconstruction. Most strikingly, when compressing the context by nearly 20x (1200-1300 text tokens into 64 latent tokens), C3 maintains an exceptional precision of 98.4%. In stark contrast, the performance of Deepseek-OCR under similar conditions plummets to 59.1%. This result provides compelling evidence that our method is far more robust and effective for extreme context compression.

The superior performance of C3 can be attributed to its fundamental architectural design. The Deepseek-OCR analysis hypothesizes that its performance decline is due to factors like "complex layout" and "image blurring at lower resolutions"—inherent limitations of the optical pathway. Our C3 paradigm, by operating directly in the textual domain, is entirely immune to these visual-domain artifacts. It avoids the information loss associated with rendering text to pixels and then encoding those pixels. Instead, it leverages a pre-trained LLM's powerful semantic understanding to distill textual information directly into an efficient latent representation.

To further probe the upper limits of our C3 architecture, we conducted a more aggressive set of experiments, compressing long contexts into just 32 latent tokens. The results, detailed in Table [2,](#page-6-1) are highly compelling and underscore the remarkable efficiency and robustness of our direct text-to-latent paradigm.The data reveals that even under these extreme compression settings, C3 maintains an exceptionally high level of reconstruction fidelity. Even when compressing documents by over 30x (for texts of 900-1000 tokens), C3 sustains a precision of 98.8%. At the far end of

<span id="page-6-0"></span>Table 1: Performance comparison between Deepseek-OCR (DS-OCR) and C3 at different compression levels. The table reports reconstruction precision and compression ratios for documents of varying lengths, using 64 and 100 latent tokens.

|             |       | Latent Tokens = 64 |             |       | Latent Tokens = 100 |             |  |
|-------------|-------|--------------------|-------------|-------|---------------------|-------------|--|
| Text Tokens | C3    | DS-OCR             | Compression | C3    | DS-OCR              | Compression |  |
| 600-700     | 99.7% | 96.5%              | 10.5×       | 99.9% | 98.5%               | 6.7×        |  |
| 700-800     | 99.9% | 93.8%              | 11.8×       | 99.5% | 97.3%               | 7.5×        |  |
| 800-900     | 99.7% | 83.8%              | 13.2×       | 99.7% | 96.8%               | 8.5×        |  |
| 900-1000    | 99.9% | 85.9%              | 15.1×       | 99.7% | 96.8%               | 9.7×        |  |
| 1000-1100   | 99.7% | 79.3%              | 16.5×       | 99.5% | 91.5%               | 10.6×       |  |
| 1100-1200   | 98.6% | 76.4%              | 17.7×       | 98.5% | 89.8%               | 11.3×       |  |
| 1200-1300   | 98.4% | 59.1%              | 19.7×       | 97.5% | 87.1%               | 12.6×       |  |

our test, compressing 1200-1300 tokens by nearly 40x, the model still retains a remarkable 93.3% precision. Deepseek-OCR, which, as shown previously (in Table [1\)](#page-6-0), degrade to approximately 60% precision at a mere 20x compression ratio. Our C3 model not only surpasses that benchmark but does so while operating at nearly double the compression rate.

<span id="page-6-1"></span>These findings conclusively demonstrate the advantage of the C3 architecture. By avoiding the information bottlenecks and potential artifacts inherent in the visual modality (e.g., image resolution limits, layout complexity), our method gracefully handles extreme compression with minimal information loss. The results from this aggressive test case solidify our claim that direct text-to-latent compression is a fundamentally more efficient and powerful paradigm than its optical counterparts.

Table 2: Precision and Compression Ratio at 32 Latent Tokens

| Latent Tokens = 32 |           |             |  |  |  |
|--------------------|-----------|-------------|--|--|--|
| Text Tokens        | precision | Compression |  |  |  |
| 600-700            | 99.7%     | 21.0×       |  |  |  |
| 700-800            | 99.7%     | 23.6×       |  |  |  |
| 800-900            | 98.8%     | 26.4×       |  |  |  |
| 900-1000           | 98.7%     | 30.2×       |  |  |  |
| 1000-1100          | 98.8%     | 31.0×       |  |  |  |
| 1100-1200          | 94.0%     | 35.4×       |  |  |  |
| 1200-1300          | 93.3%     | 39.4×       |  |  |  |

### 4.3.2 Qualitative

To provide a more intuitive understanding of our model's capabilities, we present several qualitative case studies in Figure [4.](#page-7-0) These examples showcase C3's reconstruction performance at an extreme compression level, where long contexts are compressed into just 32 latent tokens.

The first two examples (top row) demonstrate the model's high-fidelity reconstruction on wellstructured English prose and classical Chinese. In both instances, the model achieves a near-perfect, verbatim copy of the original input, confirming its efficacy on standard, coherent text.

More notably, the bottom two examples test the model's robustness against unconventional inputs: an English paragraph interspersed with non-semantic random characters, and a structurally disordered Chinese paragraph. In both challenging scenarios, C3 successfully reconstructs the input verbatim, including the random characters and the disordered sentence structure. It effectively learns to "zip" and "unzip" the textual data, regardless of its semantic coherence.

In addition to these successful cases, it is instructive to examine the model's failure modes under extreme pressure. When errors do occur, they tend to be concentrated in the latter of the text. In contrast to optical compression methods, which exhibit a uniform degradation of information across the entire text as the number of latent tokens decreases, our approach demonstrates a sequential information loss. Fidelity is often perfectly preserved for the initial parts of the text, while errors

gradually manifest towards the end. This behavior is more analogous to the human process of forgetting, where information loss occurs progressively over time rather than as a diffuse, simultaneous blur. As show in Figure [5,](#page-8-0) this suggests that while the fixed-length latent context captures the vast majority of information, the immense compression pressure may lead to a gradual decay of fidelity for the final portions of the source text.

# <span id="page-7-0"></span>b. Chinese long text text token len: 743, latent token len:32, compression: 23x a. English long text text token len: 902, latent token len: 32, compression: 28x

*"Repeat the text: "*

#### c. Disordered english text text token len: 209, latent token len:32, compression: 6x

#### d. Disordered chinese text text token len: 185, latent token len:32, compression: 5x

#### e. Large compression factor text token len: 1701, latent token len:32, compression: 53x

Figure 4: Qualitative results of text reconstruction using C3 at an extreme compression level (32 latent tokens). Each panel displays the original long text on the left and the model's reconstructed output on the right. The examples showcase the model's high-fidelity performance across diverse scenarios, including: (a) standard English prose, (b) classical Chinese, (c) English text containing non-semantic random characters, and (d) structurally disordered Chinese text. The near-perfect reconstruction in all cases highlights C3's capability for near-lossless compression. Furthermore, we present an analysis of failure cases that occur under extreme compression ratios. (e) A key observation is that in these "bad cases", reconstruction errors tend to be concentrated in the latter half of the original text.

<span id="page-8-0"></span>> **[图片提取文字 (无描述)]:**
> The changes in human memory over tim abcdefghigkl abcdefghigkl Memory The precision of optical compression changes with text length. abcdefghigkl abcdefghigkl dpsk-ocr The precision of C3 changes with text length. abcdefqhiqkl abcdefghigkl C3
![](_page_8_Figure_0.jpeg)

Figure 5: An analogy of information loss patterns. This figure contrasts two failure modes. Optical compression (middle) leads to a uniform 'blurring' of the entire context. In contrast, C3's information loss is sequential (bottom), fading from the end, which is analogous to the process of human memory decay (top).

# 5 Conclusion and Future Work

#### 5.1 Conclusion

In this work, we introduced Context Cascade Compression (C3), a novel paradigm for highly efficient long-context compression in Large Language Models. C3 proposes a more direct, pure text-to-latent pathway, implemented via a cascaded architecture of two LLMs.

Our extensive experiments on the Fox benchmark unequivocally demonstrate the superiority of this approach. C3 consistently outperforms state-of-the-art optical compression methods like Deepseek-OCR across all metrics. At compression ratios approaching 20x, where Deepseek-OCR's precision drops to approximately 60%, C3 maintains an exceptional reconstruction precision of over 98%. Even at extreme compression rates of nearly 40x, C3 retains a remarkable level of information fidelity. Furthermore, we identified a fundamental difference in the failure modes between the two paradigms. Unlike the uniform, diffuse blurring of information seen in optical compression, C3 exhibits a sequential information loss, where errors tend to concentrate at the end of the text. This behavior is compellingly analogous to the human process of memory decay, suggesting a more natural and potentially more predictable mechanism for handling information overload.

# 5.2 Future Work

The success of C3 opens up several promising avenues for future research and practical application within the LLM ecosystem.

Enabling Ultra-long Context for LLMs: The most immediate application of C3 is to serve as a powerful "front-end" compressor for existing LLMs. By compressing context that exceeds a model's native window size (e.g., millions of tokens) into a manageable number of latent tokens, C3 can unlock new capabilities in processing entire books, extensive legal documents, or large codebases for tasks like question-answering, summarization, and analysis.This will significantly reduce costs.

Multimodal Cascaded Architecture: Reducing costs in long-context scenarios by integrating a lightweight VLM with a LLM for processing visually-rich Documents, multi-image and video inputs.

A Foundational Component for Next-Generation Generative Models: C3 can function as an autoencoder architecture for encoding and decoding variable-length text. The resulting latent tokens are highly applicable to next-generation generative models, such as Diffusion Language Models and Latent Auto-Regressive Models.

# References

- <span id="page-9-0"></span>OpenAI. Gpt-4 technical report, 2023.
- <span id="page-9-1"></span>DeepSeek-AI. Deepseek-v3.2-exp. <https://api-docs.deepseek.com/>, 2025. Accessed: 2025- 11-14.
- <span id="page-9-2"></span>QwenTeam. Qwen3-next. <https://qwen.ai/blog?id=qwen3-next>, 2025. Accessed: 2025-11-15.
- <span id="page-9-3"></span>Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL [https:](https://arxiv.org/abs/2005.11401) [//arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401).
- <span id="page-9-4"></span>Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. *arXiv preprint arXiv:2510.18234*, 2025.
- <span id="page-9-5"></span>Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv:2004.05150*, 2020.
- <span id="page-9-6"></span>Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. *Advances in Neural Information Processing Systems*, 33, 2020.
- <span id="page-9-7"></span>Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Francois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In *ICML*, 2020.
- <span id="page-9-8"></span>Tri Dao and Albert Gu. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.
- <span id="page-9-9"></span>Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. *arXiv preprint arXiv:2307.06945*, 2023.
- <span id="page-9-10"></span>Xiangfeng Wang, Zaiyi Chen, Tong Xu, Zheyong Xie, Yongyi He, and Enhong Chen. In-context former: Lightning-fast compressing context for large language model. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 2445–2460, 2024.
- <span id="page-9-11"></span>Aydar Bulatov, Yuri Kuratov, and Mikhail Burtsev. Recurrent memory transformer. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=Uynr3iPhksa>.
- <span id="page-9-12"></span>Jiale Cheng, Yusen Liu, Xinyu Zhang, Yulin Fei, Wenyi Hong, Ruiliang Lyu, Weihan Wang, Zhe Su, Xiaotao Gu, Xiao Liu, Yushi Bai, Jie Tang, Hongning Wang, and Minlie Huang. Glyph: Scaling context windows via visual-text compression. *arXiv preprint arXiv:2510.17800*, 2025.
- <span id="page-9-13"></span>Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL <https://arxiv.org/abs/2412.15115>.
- <span id="page-9-14"></span>Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. *arXiv preprint arXiv:2409.01704*, 2024.
- <span id="page-9-15"></span>Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL [https:](https://arxiv.org/abs/1711.05101) [//arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101).
- <span id="page-9-16"></span>Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts, 2017. URL <https://arxiv.org/abs/1608.03983>.

<span id="page-10-0"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, 2017.

<span id="page-10-1"></span>Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang. Focus anywhere for fine-grained multi-page document understanding. *arXiv preprint arXiv:2405.14295*, 2024.