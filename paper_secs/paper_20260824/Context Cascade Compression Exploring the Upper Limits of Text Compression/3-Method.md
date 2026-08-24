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

